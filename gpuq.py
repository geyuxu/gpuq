#!/usr/bin/env python3
"""gpuq — General-purpose single-GPU job queue.

A lightweight single-GPU job scheduler that:
  - Queues experiments and runs them sequentially (one GPU job at a time)
  - Auto-discovers .venv in the working directory
  - Logs stdout/stderr per job with timestamps
  - Tracks status (pending/running/done/failed) in SQLite
  - Supports dependencies between jobs
  - Retries failed jobs (configurable)
  - Adopts already-running GPU processes into management
  - Recovers state after restart — selectively re-run interrupted jobs
  - Sends desktop notification on completion (notify-send)
"""

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────
STATE_DIR = Path.home() / ".gpuq"
DB_FILE = STATE_DIR / "gpuq.db"
LOG_DIR = STATE_DIR / "logs"

DEFAULT_ENV_VARS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_VISIBLE_DEVICES": "0",
}


# ── Database ────────────────────────────────────────────────
def _ensure_dirs():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _get_db() -> sqlite3.Connection:
    _ensure_dirs()
    db = sqlite3.connect(str(DB_FILE), timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    _init_db(db)
    return db


def _init_db(db: sqlite3.Connection):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            script      TEXT NOT NULL,
            args        TEXT NOT NULL DEFAULT '[]',
            python      TEXT NOT NULL DEFAULT 'python3',
            cwd         TEXT NOT NULL,
            env         TEXT NOT NULL DEFAULT '{}',
            status      TEXT NOT NULL DEFAULT 'pending',
            after_job   INTEGER,
            retries     INTEGER NOT NULL DEFAULT 0,
            attempt     INTEGER NOT NULL DEFAULT 0,
            pid         INTEGER,
            added_at    TEXT,
            started_at  TEXT,
            finished_at TEXT,
            exit_code   INTEGER,
            log_file    TEXT
        );
    """)
    # Migration: add pid column if missing (upgrade from older version)
    cols = {r[1] for r in db.execute("PRAGMA table_info(jobs)")}
    if "pid" not in cols:
        db.execute("ALTER TABLE jobs ADD COLUMN pid INTEGER")
        db.commit()


def _job_dict(row) -> dict:
    """Convert sqlite3.Row to plain dict with parsed JSON fields."""
    d = dict(row)
    d["args"] = json.loads(d["args"])
    d["env"] = json.loads(d["env"])
    return d


def _migrate_json():
    """One-time migration from old JSON state to SQLite."""
    json_file = STATE_DIR / "queue.json"
    if not json_file.exists():
        return
    try:
        data = json.loads(json_file.read_text())
    except (json.JSONDecodeError, KeyError):
        return
    if not data.get("jobs"):
        json_file.rename(json_file.with_suffix(".json.bak"))
        return

    db = _get_db()
    for j in data["jobs"]:
        db.execute("""
            INSERT INTO jobs (name, script, args, python, cwd, env, status,
                              after_job, retries, attempt, pid, added_at,
                              started_at, finished_at, exit_code, log_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            j.get("name", "?"), j["script"],
            json.dumps(j.get("args", [])), j.get("python", "python3"),
            j.get("cwd", "."), json.dumps(j.get("env", {})),
            j["status"], j.get("after"), j.get("retries", 0),
            j.get("attempt", 0), j.get("pid"),
            j.get("added_at"), j.get("started_at"),
            j.get("finished_at"), j.get("exit_code"), j.get("log_file"),
        ))
    db.commit()
    db.close()
    json_file.rename(json_file.with_suffix(".json.bak"))
    print(f"Migrated {len(data['jobs'])} jobs from JSON to SQLite.")


def _find_python(directory: str) -> str:
    d = Path(directory)
    for check_dir in [d, d.parent]:
        venv_python = check_dir / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
    return "python3"


def _pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_proc_info(pid: int) -> dict:
    """Read process info from /proc (Linux)."""
    info = {"pid": pid, "cmdline": None, "cwd": None, "name": None}
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return info
    try:
        cmdline = (proc / "cmdline").read_bytes().decode(errors="replace")
        info["cmdline"] = cmdline.replace("\x00", " ").strip()
    except OSError:
        pass
    try:
        info["cwd"] = str((proc / "cwd").resolve())
    except OSError:
        pass
    try:
        info["name"] = (proc / "comm").read_text().strip()
    except OSError:
        pass
    return info


# ── Commands ────────────────────────────────────────────────
def cmd_add(args):
    db = _get_db()
    script = args.script
    extra_args = args.extra_args or []
    cwd = os.path.abspath(args.dir if args.dir else os.getcwd())

    python_path = args.python if args.python else _find_python(cwd)

    script_path = Path(cwd) / script
    if not script_path.exists():
        if Path(script).exists():
            script = str(Path(script).resolve())
        else:
            print(f"WARNING: {script_path} does not exist (will fail at runtime)")

    if args.name:
        name = args.name
    else:
        name = f"{Path(cwd).name}/{Path(script).stem}"

    job_env = {}
    if args.env:
        for item in args.env:
            if "=" in item:
                k, v = item.split("=", 1)
                job_env[k] = v

    now = datetime.now().isoformat()
    cur = db.execute("""
        INSERT INTO jobs (name, script, args, python, cwd, env, status,
                          after_job, retries, attempt, added_at, log_file)
        VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, 0, ?, ?)
    """, (
        name, script, json.dumps(extra_args), python_path, cwd,
        json.dumps(job_env), args.after, args.retries, now, "",
    ))
    job_id = cur.lastrowid
    log_file = str(LOG_DIR / f"job_{job_id:03d}_{Path(script).stem}.log")
    db.execute("UPDATE jobs SET log_file = ? WHERE id = ?", (log_file, job_id))
    db.commit()
    db.close()

    dep_str = f" (after job #{args.after})" if args.after else ""
    print(f"[+] Job #{job_id}: {name} {' '.join(extra_args)}{dep_str}")


def cmd_adopt(args):
    """Adopt an already-running process into the queue."""
    pid = args.pid
    if not _pid_alive(pid):
        print(f"PID {pid} is not running.")
        return

    info = _read_proc_info(pid)
    cwd = info["cwd"] or os.getcwd()
    cmdline = info["cmdline"] or f"PID {pid}"

    if args.name:
        name = args.name
    else:
        name = info["name"] or f"adopted-{pid}"

    # Try to extract script from cmdline
    parts = cmdline.split()
    script = "unknown"
    script_args = []
    for i, p in enumerate(parts):
        if p.endswith(".py"):
            script = p
            script_args = parts[i+1:]
            break

    python_path = parts[0] if parts else "python3"
    now = datetime.now().isoformat()
    log_file = str(LOG_DIR / f"job_adopted_{pid}_{Path(script).stem}.log")

    db = _get_db()
    cur = db.execute("""
        INSERT INTO jobs (name, script, args, python, cwd, env, status,
                          retries, attempt, pid, added_at, started_at, log_file)
        VALUES (?, ?, ?, ?, ?, '{}', 'running', 0, 1, ?, ?, ?, ?)
    """, (name, script, json.dumps(script_args), python_path, cwd,
          pid, now, now, log_file))
    job_id = cur.lastrowid
    # Update log_file with actual ID
    log_file = str(LOG_DIR / f"job_{job_id:03d}_{Path(script).stem}.log")
    db.execute("UPDATE jobs SET log_file = ? WHERE id = ?", (log_file, job_id))
    db.commit()
    db.close()

    print(f"[+] Adopted PID {pid} as Job #{job_id}: {name}")
    print(f"    cmd: {cmdline}")
    print(f"    cwd: {cwd}")


def cmd_status(args):
    db = _get_db()
    # Auto-check running jobs — mark stale ones
    running = db.execute("SELECT id, pid FROM jobs WHERE status = 'running' AND pid IS NOT NULL").fetchall()
    for r in running:
        if not _pid_alive(r["pid"]):
            db.execute("""
                UPDATE jobs SET status = 'interrupted', finished_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), r["id"]))
    db.commit()

    jobs = db.execute("SELECT * FROM jobs ORDER BY id").fetchall()
    db.close()

    if not jobs:
        print("Queue is empty.")
        return

    symbols = {
        "pending": "⏳", "running": "🔄", "done": "✅",
        "failed": "❌", "cancelled": "🚫", "interrupted": "⚡",
    }

    print(f"{'ID':>3}  {'St':>2}  {'Name':<30} {'Script':<25} {'Time':>8}  {'Info'}")
    print("─" * 95)

    for j in jobs:
        sym = symbols.get(j["status"], "?")
        elapsed = ""
        end_time = j["finished_at"] or (datetime.now().isoformat() if j["started_at"] else None)
        if j["started_at"] and end_time:
            t0 = datetime.fromisoformat(j["started_at"])
            t1 = datetime.fromisoformat(end_time)
            secs = (t1 - t0).total_seconds()
            elapsed = f"{int(secs//60)}m{int(secs%60):02d}s"

        info_parts = []
        if j["after_job"]:
            info_parts.append(f"after #{j['after_job']}")
        if j["pid"]:
            info_parts.append(f"pid={j['pid']}")
        if j["exit_code"] is not None:
            info_parts.append(f"exit={j['exit_code']}")
        if j["attempt"] > 1:
            info_parts.append(f"attempt={j['attempt']}")
        info = " ".join(info_parts)

        name = j["name"] or "?"
        if len(name) > 30:
            name = "..." + name[-27:]

        script_short = j["script"]
        if len(script_short) > 25:
            script_short = "..." + script_short[-22:]

        print(f"{j['id']:>3}  {sym}  {name:<30} {script_short:<25} {elapsed:>8}  {info}")


def cmd_log(args):
    db = _get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (args.job_id,)).fetchone()
    db.close()
    if not job:
        print(f"Job #{args.job_id} not found.")
        return
    log_path = Path(job["log_file"])
    if not log_path.exists():
        print(f"No log file yet for job #{args.job_id}.")
        return
    n = args.lines or 30
    lines = log_path.read_text().splitlines()
    for line in lines[-n:]:
        print(line)


def cmd_cancel(args):
    db = _get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (args.job_id,)).fetchone()
    if not job:
        print(f"Job #{args.job_id} not found.")
        db.close()
        return
    if job["status"] != "pending":
        print(f"Job #{args.job_id} is {job['status']}, can only cancel pending jobs.")
        db.close()
        return
    db.execute("UPDATE jobs SET status = 'cancelled' WHERE id = ?", (args.job_id,))
    db.commit()
    db.close()
    print(f"[x] Job #{args.job_id} cancelled.")


def cmd_clear(args):
    db = _get_db()
    cur = db.execute("DELETE FROM jobs WHERE status IN ('done', 'failed', 'cancelled')")
    db.commit()
    cleared = cur.rowcount
    remaining = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    db.close()
    print(f"Cleared {cleared} jobs, {remaining} remaining.")


def cmd_recover(args):
    """Recover after restart — find interrupted jobs and offer to re-run."""
    db = _get_db()

    # Mark any "running" jobs whose PID is dead as "interrupted"
    running = db.execute("SELECT id, pid, name FROM jobs WHERE status = 'running'").fetchall()
    interrupted = []
    still_running = []
    for r in running:
        if r["pid"] and _pid_alive(r["pid"]):
            still_running.append(r)
        else:
            db.execute("""
                UPDATE jobs SET status = 'interrupted', finished_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), r["id"]))
            interrupted.append(r)
    db.commit()

    if still_running:
        print(f"Still running ({len(still_running)}):")
        for r in still_running:
            print(f"  #{r['id']} {r['name']} (pid={r['pid']})")

    # Find all recoverable jobs
    recoverable = db.execute(
        "SELECT * FROM jobs WHERE status = 'interrupted' ORDER BY id"
    ).fetchall()

    if not recoverable:
        print("No interrupted jobs to recover.")
        db.close()
        return

    print(f"\nInterrupted jobs ({len(recoverable)}):")
    for j in recoverable:
        print(f"  #{j['id']} {j['name']} — {j['script']}")

    if args.all:
        # Re-queue all interrupted
        ids = [j["id"] for j in recoverable]
    elif args.jobs:
        ids = args.jobs
    else:
        # Interactive: ask user
        print()
        ans = input("Re-queue which jobs? (all / comma-separated IDs / none): ").strip()
        if ans.lower() in ("all", "a"):
            ids = [j["id"] for j in recoverable]
        elif ans.lower() in ("none", "n", ""):
            ids = []
        else:
            ids = [int(x.strip()) for x in ans.split(",") if x.strip().isdigit()]

    count = 0
    for jid in ids:
        res = db.execute("UPDATE jobs SET status = 'pending', pid = NULL, started_at = NULL, finished_at = NULL WHERE id = ? AND status = 'interrupted'", (jid,))
        count += res.rowcount
    db.commit()
    db.close()

    if count:
        print(f"\nRe-queued {count} job(s). Run `gpuq run` to execute.")
    else:
        print("No jobs re-queued.")


def cmd_run(args):
    if args.daemon:
        daemon_log = STATE_DIR / "daemon.log"
        print(f"Daemonizing... logs at {daemon_log}")
        pid = os.fork()
        if pid > 0:
            print(f"Daemon PID: {pid}")
            return
        os.setsid()
        sys.stdout = open(daemon_log, "a")
        sys.stderr = sys.stdout

    print(f"\n{'='*60}")
    print(f"gpuq runner started at {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    while True:
        db = _get_db()

        # Wait for adopted running jobs to finish before picking next
        adopted_running = db.execute(
            "SELECT id, pid, name FROM jobs WHERE status = 'running' AND pid IS NOT NULL"
        ).fetchall()
        for ar in adopted_running:
            if ar["pid"] and _pid_alive(ar["pid"]):
                db.close()
                time.sleep(5)
                continue

        pending = db.execute(
            "SELECT * FROM jobs WHERE status = 'pending' ORDER BY id"
        ).fetchall()

        if not pending:
            db.close()
            break

        runnable = None
        for j in pending:
            j = _job_dict(j)
            if j["after_job"]:
                dep = db.execute("SELECT status FROM jobs WHERE id = ?", (j["after_job"],)).fetchone()
                if dep and dep["status"] not in ("done", "failed", "cancelled"):
                    continue
            runnable = j
            break

        if not runnable:
            running = db.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running'").fetchone()[0]
            db.close()
            if running:
                time.sleep(5)
                continue
            else:
                print("WARNING: Deadlock — pending jobs have unresolvable dependencies.")
                break

        db.close()
        _run_job(runnable)

    _notify_done()
    db = _get_db()
    n_done = db.execute("SELECT COUNT(*) FROM jobs WHERE status = 'done'").fetchone()[0]
    n_fail = db.execute("SELECT COUNT(*) FROM jobs WHERE status = 'failed'").fetchone()[0]
    db.close()
    print(f"\n{'='*60}")
    print(f"gpuq finished at {datetime.now().isoformat()}")
    print(f"Results: {n_done} done, {n_fail} failed")
    print(f"{'='*60}")


def _run_job(job: dict):
    db = _get_db()
    attempt = job.get("attempt", 0) + 1
    started = datetime.now().isoformat()

    cmd = [job["python"], "-u", job["script"]] + job.get("args", [])
    env = {**os.environ, **DEFAULT_ENV_VARS, **job.get("env", {})}

    name = job.get("name", "?")
    print(f">>> [{job['id']}] {name} {' '.join(job.get('args', []))}")
    print(f"    cwd: {job['cwd']}")
    print(f"    log: {job['log_file']}")
    print(f"    started: {started}")

    log_path = Path(job["log_file"])
    with open(log_path, "a") as log_f:
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Job #{job['id']} attempt #{attempt}\n")
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"Started: {started}\n")
        log_f.write(f"{'='*60}\n\n")
        log_f.flush()

        try:
            proc = subprocess.Popen(
                cmd, cwd=job["cwd"], env=env,
                stdout=log_f, stderr=subprocess.STDOUT,
            )
            # Record PID immediately
            db.execute("""
                UPDATE jobs SET status = 'running', attempt = ?, pid = ?,
                                started_at = ?
                WHERE id = ?
            """, (attempt, proc.pid, started, job["id"]))
            db.commit()
            db.close()

            proc.wait()
            exit_code = proc.returncode
        except Exception as e:
            log_f.write(f"\nEXCEPTION: {e}\n")
            exit_code = -1

        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Exit code: {exit_code}\n")
        log_f.write(f"Finished: {datetime.now().isoformat()}\n")

    finished = datetime.now().isoformat()
    db = _get_db()

    if exit_code == 0:
        status = "done"
        print(f"    ✅ done (exit 0)")
    else:
        retries = job.get("retries", 0)
        if attempt < retries + 1:
            status = "pending"
            print(f"    ⚠️  failed (exit {exit_code}), will retry ({attempt}/{retries+1})")
        else:
            status = "failed"
            print(f"    ❌ failed (exit {exit_code})")

    db.execute("""
        UPDATE jobs SET status = ?, finished_at = ?, exit_code = ?, pid = NULL
        WHERE id = ?
    """, (status, finished, exit_code, job["id"]))
    db.commit()
    db.close()


def _notify_done():
    try:
        db = _get_db()
        n_done = db.execute("SELECT COUNT(*) FROM jobs WHERE status = 'done'").fetchone()[0]
        n_fail = db.execute("SELECT COUNT(*) FROM jobs WHERE status = 'failed'").fetchone()[0]
        db.close()
        msg = f"gpuq: {n_done} done, {n_fail} failed"
        subprocess.run(
            ["notify-send", "-u", "normal", "GPU Queue Complete", msg],
            timeout=5, capture_output=True,
        )
    except Exception:
        pass


# ── CLI ─────────────────────────────────────────────────────
def main():
    # Auto-migrate from JSON on first run
    if not DB_FILE.exists():
        _migrate_json()

    parser = argparse.ArgumentParser(
        prog="gpuq",
        description="General-purpose single-GPU job queue",
    )
    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Add a job")
    p_add.add_argument("script", help="Python script to run")
    p_add.add_argument("--dir", "-d", default=None, help="Working directory (default: cwd)")
    p_add.add_argument("--python", "-p", default=None, help="Python interpreter path")
    p_add.add_argument("--name", default=None, help="Job name for display")
    p_add.add_argument("--after", type=int, default=None, help="Run after job #N completes")
    p_add.add_argument("--retries", type=int, default=0, help="Number of retries on failure")
    p_add.add_argument("--env", "-e", action="append", help="Extra env var (KEY=VALUE), repeatable")

    # adopt
    p_adopt = sub.add_parser("adopt", help="Adopt a running process")
    p_adopt.add_argument("pid", type=int, help="PID of the running process")
    p_adopt.add_argument("--name", default=None, help="Job name for display")

    # status
    sub.add_parser("status", help="Show queue")

    # log
    p_log = sub.add_parser("log", help="Show job log")
    p_log.add_argument("job_id", type=int)
    p_log.add_argument("-n", "--lines", type=int, default=30)

    # cancel
    p_cancel = sub.add_parser("cancel", help="Cancel pending job")
    p_cancel.add_argument("job_id", type=int)

    # clear
    sub.add_parser("clear", help="Remove finished jobs")

    # recover
    p_recover = sub.add_parser("recover", help="Recover interrupted jobs after restart")
    p_recover.add_argument("--all", action="store_true", help="Re-queue all interrupted jobs")
    p_recover.add_argument("--jobs", type=int, nargs="+", help="Re-queue specific job IDs")

    # run
    p_run = sub.add_parser("run", help="Execute the queue")
    p_run.add_argument("--daemon", action="store_true", help="Run in background")

    args, unknown = parser.parse_known_args()

    if args.command == "add":
        args.extra_args = unknown
        cmd_add(args)
    elif args.command == "adopt":
        cmd_adopt(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "cancel":
        cmd_cancel(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "recover":
        cmd_recover(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
