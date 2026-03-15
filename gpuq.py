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
import re
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
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
    info = {"pid": pid, "cmdline": None, "cwd": None, "name": None, "started_at": None}
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
    # Read real start time from /proc/pid/stat
    try:
        stat_data = (proc / "stat").read_text()
        # Field 22 (0-indexed after comm) is starttime in clock ticks
        fields = stat_data.split(")")[-1].split()
        starttime_ticks = int(fields[19])
        clk_tck = os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        boot_time = time.time() - uptime
        start_epoch = boot_time + starttime_ticks / clk_tck
        info["started_at"] = datetime.fromtimestamp(start_epoch).isoformat()
    except (OSError, IndexError, ValueError):
        pass
    return info


# ── ETA estimation ──────────────────────────────────────────
# Patterns to extract progress from log output (current, total)
_PROGRESS_PATTERNS = [
    # trl/transformers: " Step 50/300 " or "step 50/300"
    re.compile(r'[Ss]tep\s+(\d+)\s*/\s*(\d+)'),
    # "[50/300]" — common in tqdm and training loops
    re.compile(r'\[(\d+)/(\d+)\]'),
    # "Epoch 2/10" or "epoch 2/10"
    re.compile(r'[Ee]poch\s+(\d+)\s*/\s*(\d+)'),
    # "Progress: 50%" or "50.0%"
    re.compile(r'(\d+(?:\.\d+)?)\s*%'),
    # "iteration 50 of 300"
    re.compile(r'[Ii]teration\s+(\d+)\s+of\s+(\d+)'),
]


def _estimate_eta(job) -> dict:
    """Estimate ETA for a running job.

    Returns dict with keys: current, total, pct, eta_str, method
    Returns empty dict if no estimate possible.
    """
    if job["status"] != "running" or not job["started_at"]:
        return {}

    started = datetime.fromisoformat(job["started_at"])
    elapsed = (datetime.now() - started).total_seconds()
    if elapsed < 1:
        return {}

    # Method 1: Parse log file for progress patterns
    result = _eta_from_log(job, elapsed)
    if result:
        return result

    # Method 2: Parse cmdline args for --max_steps and scan checkpoints
    result = _eta_from_checkpoints(job, elapsed)
    if result:
        return result

    return {}


def _eta_from_log(job, elapsed: float) -> dict:
    """Extract progress from job log file."""
    log_path = Path(job["log_file"]) if job["log_file"] else None
    if not log_path or not log_path.exists():
        return {}

    # Read last 200 lines for efficiency
    try:
        lines = log_path.read_text().splitlines()[-200:]
    except OSError:
        return {}

    # Search backwards for the latest progress match
    for line in reversed(lines):
        for pat in _PROGRESS_PATTERNS:
            m = pat.search(line)
            if not m:
                continue
            groups = m.groups()
            if len(groups) == 1:
                # Percentage pattern
                pct = float(groups[0])
                if 0 < pct <= 100:
                    remaining = elapsed * (100 - pct) / pct if pct > 0 else 0
                    return {
                        "pct": pct,
                        "eta_str": _fmt_duration(remaining),
                        "method": "log-%",
                    }
            elif len(groups) == 2:
                current, total = int(groups[0]), int(groups[1])
                if 0 < current <= total and total > 1:
                    pct = current / total * 100
                    remaining = elapsed * (total - current) / current
                    return {
                        "current": current, "total": total,
                        "pct": pct,
                        "eta_str": _fmt_duration(remaining),
                        "method": "log",
                    }

    return {}


def _eta_from_checkpoints(job, elapsed: float) -> dict:
    """Estimate progress from checkpoint files and --max_steps in args."""
    args = json.loads(job["args"]) if isinstance(job["args"], str) else job["args"]
    cwd = job["cwd"]

    # Find max_steps and output_dir from args
    max_steps = None
    output_dir = None
    for i, a in enumerate(args):
        if a in ("--max_steps", "--max-steps", "--num_train_steps") and i + 1 < len(args):
            try:
                max_steps = int(args[i + 1])
            except ValueError:
                pass
        if a in ("--output_dir", "--output-dir") and i + 1 < len(args):
            output_dir = args[i + 1]

    if not max_steps:
        return {}

    # Determine scan directory: output_dir (relative to cwd) > cwd
    if output_dir:
        scan_dir = Path(cwd) / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    else:
        scan_dir = Path(cwd)

    if not scan_dir.exists():
        return {}

    current_step = 0

    # HuggingFace style: checkpoint-{step} (only direct children to avoid stale nested ones)
    for d in scan_dir.glob("checkpoint-*"):
        if d.is_dir():
            try:
                step = int(d.name.split("-")[-1])
                current_step = max(current_step, step)
            except ValueError:
                pass

    # Completion files: completions_{step:05d}.parquet (in completions/ subdir or directly)
    for pattern in ["completions/completions_*.parquet", "completions_*.parquet"]:
        for f in scan_dir.glob(pattern):
            try:
                step = int(f.stem.split("_")[-1])
                current_step = max(current_step, step)
            except ValueError:
                pass

    if current_step > 0 and current_step <= max_steps:
        pct = current_step / max_steps * 100
        remaining = elapsed * (max_steps - current_step) / current_step
        return {
            "current": current_step, "total": max_steps,
            "pct": pct,
            "eta_str": _fmt_duration(remaining),
            "method": "checkpoint",
        }

    return {}


def _fmt_duration(seconds: float) -> str:
    """Format seconds into human readable duration."""
    if seconds < 0:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h{m:02d}m"


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
    started_at = info.get("started_at") or now
    log_file = str(LOG_DIR / f"job_adopted_{pid}_{Path(script).stem}.log")

    db = _get_db()
    cur = db.execute("""
        INSERT INTO jobs (name, script, args, python, cwd, env, status,
                          retries, attempt, pid, added_at, started_at, log_file)
        VALUES (?, ?, ?, ?, ?, '{}', 'running', 0, 1, ?, ?, ?, ?)
    """, (name, script, json.dumps(script_args), python_path, cwd,
          pid, now, started_at, log_file))
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

    # Filter by status
    done_statuses = ("done", "failed", "cancelled", "preempted")
    if args.done:
        jobs = db.execute(
            f"SELECT * FROM jobs WHERE status IN ({','.join('?' * len(done_statuses))}) ORDER BY id",
            done_statuses
        ).fetchall()
    elif args.all:
        jobs = db.execute("SELECT * FROM jobs ORDER BY id").fetchall()
    else:
        jobs = db.execute(
            f"SELECT * FROM jobs WHERE status NOT IN ({','.join('?' * len(done_statuses))}) ORDER BY id",
            done_statuses
        ).fetchall()
    db.close()

    if not jobs:
        if args.done:
            print("No completed jobs.")
        elif args.all:
            print("Queue is empty.")
        else:
            print("No active jobs. Use --all to see completed jobs.")
        return

    symbols = {
        "pending": "⏳", "running": "🔄", "done": "✅",
        "failed": "❌", "cancelled": "🚫", "interrupted": "⚡", "preempted": "⏸️",
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
            elapsed = _fmt_duration(secs)

        info_parts = []
        if j["after_job"]:
            info_parts.append(f"after #{j['after_job']}")
        if j["pid"]:
            info_parts.append(f"pid={j['pid']}")
        if j["exit_code"] is not None:
            info_parts.append(f"exit={j['exit_code']}")
        if j["attempt"] > 1:
            info_parts.append(f"attempt={j['attempt']}")

        # ETA for running jobs
        if j["status"] == "running":
            eta = _estimate_eta(j)
            if eta:
                info_parts.append(f"{eta['pct']:.0f}% ETA {eta['eta_str']}")

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


def cmd_eta(args):
    """Show detailed ETA for a running job."""
    db = _get_db()
    if args.job_id:
        jobs = db.execute("SELECT * FROM jobs WHERE id = ?", (args.job_id,)).fetchall()
    else:
        jobs = db.execute("SELECT * FROM jobs WHERE status = 'running' ORDER BY id").fetchall()
    db.close()

    if not jobs:
        print("No running jobs." if not args.job_id else f"Job #{args.job_id} not found.")
        return

    for j in jobs:
        if j["status"] != "running":
            print(f"Job #{j['id']} is {j['status']}, not running.")
            continue

        started = datetime.fromisoformat(j["started_at"])
        elapsed = (datetime.now() - started).total_seconds()
        eta = _estimate_eta(j)

        print(f"Job #{j['id']}: {j['name']}")
        print(f"  Elapsed: {_fmt_duration(elapsed)}")

        if eta:
            if "current" in eta:
                print(f"  Progress: {eta['current']}/{eta['total']} ({eta['pct']:.1f}%)")
            else:
                print(f"  Progress: {eta['pct']:.1f}%")
            print(f"  ETA: {eta['eta_str']}")
            finish_time = datetime.now() + timedelta(seconds=_parse_eta_seconds(eta['eta_str']))
            print(f"  Est. finish: {finish_time.strftime('%H:%M:%S')}")
            print(f"  Method: {eta['method']}")
        else:
            print("  Progress: unknown (no progress pattern detected in logs or checkpoints)")


def _parse_eta_seconds(eta_str: str) -> float:
    """Parse ETA string back to seconds."""
    total = 0
    m = re.match(r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?', eta_str)
    if m:
        if m.group(1): total += int(m.group(1)) * 3600
        if m.group(2): total += int(m.group(2)) * 60
        if m.group(3): total += int(m.group(3))
    return total


def cmd_preempt(args):
    """Gracefully stop a running job so GPU can be used for something else.

    Sends SIGTERM → waits for checkpoint save → marks as 'preempted'.
    The job can later be resumed with `gpuq resume`.
    """
    db = _get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (args.job_id,)).fetchone()
    if not job:
        print(f"Job #{args.job_id} not found.")
        db.close()
        return
    if job["status"] != "running":
        print(f"Job #{args.job_id} is {job['status']}, can only preempt running jobs.")
        db.close()
        return
    if not job["pid"] or not _pid_alive(job["pid"]):
        print(f"Job #{args.job_id} has no live process.")
        db.close()
        return

    pid = job["pid"]
    timeout = args.timeout

    print(f"Sending SIGTERM to PID {pid} (Job #{args.job_id}: {job['name']})...")
    print(f"Waiting up to {timeout}s for graceful shutdown (checkpoint save)...")

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        print(f"Failed to send SIGTERM: {e}")
        db.close()
        return

    # Wait for process to exit
    for i in range(timeout):
        if not _pid_alive(pid):
            break
        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"  Still waiting... {i+1}s")
    else:
        if _pid_alive(pid):
            if args.force:
                print(f"Timeout. Sending SIGKILL...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
            else:
                print(f"Timeout after {timeout}s. Process still running. Use --force to SIGKILL.")
                db.close()
                return

    # Find latest checkpoint
    j = _job_dict(job)
    checkpoint = _find_latest_checkpoint(j)

    db.execute("""
        UPDATE jobs SET status = 'preempted', finished_at = ?, pid = NULL
        WHERE id = ?
    """, (datetime.now().isoformat(), args.job_id))
    db.commit()
    db.close()

    print(f"Job #{args.job_id} preempted.")
    if checkpoint:
        print(f"  Latest checkpoint: {checkpoint}")
    print(f"  Use `gpuq resume {args.job_id}` to continue later.")


def cmd_resume(args):
    """Resume a preempted/interrupted job from its latest checkpoint."""
    db = _get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (args.job_id,)).fetchone()
    if not job:
        print(f"Job #{args.job_id} not found.")
        db.close()
        return
    if job["status"] not in ("preempted", "interrupted"):
        print(f"Job #{args.job_id} is {job['status']}, can only resume preempted/interrupted jobs.")
        db.close()
        return

    j = _job_dict(job)
    checkpoint = _find_latest_checkpoint(j)

    if not checkpoint:
        print(f"No checkpoint found for Job #{args.job_id}.")
        print("Re-queuing from scratch.")
        db.execute("""
            UPDATE jobs SET status = 'pending', pid = NULL,
                            started_at = NULL, finished_at = NULL
            WHERE id = ?
        """, (args.job_id,))
        db.commit()
        db.close()
        print(f"Job #{args.job_id} re-queued. Run `gpuq run` to execute.")
        return

    # Build resumed args: inject --resume_from_checkpoint
    new_args = list(j["args"])
    # Remove existing resume flags if present
    resume_flags = ("--resume_from_checkpoint", "--resume-from-checkpoint")
    filtered = []
    skip_next = False
    for a in new_args:
        if skip_next:
            skip_next = False
            continue
        if a in resume_flags:
            skip_next = True
            continue
        filtered.append(a)
    filtered.extend(["--resume_from_checkpoint", checkpoint])

    now = datetime.now().isoformat()
    name = j["name"]
    if not name.endswith("(resumed)"):
        name = f"{name} (resumed)"

    # Create a new job that continues from checkpoint
    cur = db.execute("""
        INSERT INTO jobs (name, script, args, python, cwd, env, status,
                          after_job, retries, attempt, added_at, log_file)
        VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, 0, 0, ?, ?)
    """, (
        name, j["script"], json.dumps(filtered), j["python"], j["cwd"],
        json.dumps(j["env"]), now, "",
    ))
    new_id = cur.lastrowid
    log_file = str(LOG_DIR / f"job_{new_id:03d}_{Path(j['script']).stem}.log")
    db.execute("UPDATE jobs SET log_file = ? WHERE id = ?", (log_file, new_id))
    db.commit()
    db.close()

    print(f"[+] Job #{new_id}: {name}")
    print(f"    Resuming from: {checkpoint}")
    print(f"    Run `gpuq run` to execute.")


def _find_latest_checkpoint(job: dict) -> str:
    """Find the latest checkpoint directory for a job."""
    args = job.get("args", [])
    if isinstance(args, str):
        args = json.loads(args)
    cwd = job["cwd"]

    # Find output_dir from args
    output_dir = None
    for i, a in enumerate(args):
        if a in ("--output_dir", "--output-dir") and i + 1 < len(args):
            output_dir = args[i + 1]
            break

    if output_dir:
        scan_dir = Path(cwd) / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    else:
        scan_dir = Path(cwd)

    if not scan_dir.exists():
        return ""

    # Find checkpoint-{step} dirs, return the one with highest step
    best_step = -1
    best_path = ""
    for d in scan_dir.glob("checkpoint-*"):
        if d.is_dir():
            try:
                step = int(d.name.split("-")[-1])
                if step > best_step:
                    best_step = step
                    best_path = str(d)
            except ValueError:
                pass

    return best_path


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
    cur = db.execute("DELETE FROM jobs WHERE status IN ('done', 'failed', 'cancelled', 'preempted')")
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
        has_running = False
        for ar in adopted_running:
            if ar["pid"] and _pid_alive(ar["pid"]):
                has_running = True
                break
            else:
                # Adopted process died — mark as done (exit code unknown)
                db.execute("""
                    UPDATE jobs SET status = 'done', finished_at = ?, pid = NULL
                    WHERE id = ? AND status = 'running'
                """, (datetime.now().isoformat(), ar["id"]))
                db.commit()
                print(f"    Job #{ar['id']} ({ar['name']}): adopted process finished.")
        if has_running:
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
    p_status = sub.add_parser("status", help="Show queue")
    p_status.add_argument("-a", "--all", action="store_true", help="Show all jobs including completed")
    p_status.add_argument("--done", action="store_true", help="Show only completed/failed jobs")

    # log
    p_log = sub.add_parser("log", help="Show job log")
    p_log.add_argument("job_id", type=int)
    p_log.add_argument("-n", "--lines", type=int, default=30)

    # eta
    p_eta = sub.add_parser("eta", help="Show ETA for running jobs")
    p_eta.add_argument("job_id", type=int, nargs="?", default=None, help="Job ID (default: all running)")

    # preempt
    p_preempt = sub.add_parser("preempt", help="Gracefully stop a running job for later resume")
    p_preempt.add_argument("job_id", type=int, help="Job ID to preempt")
    p_preempt.add_argument("--timeout", type=int, default=60, help="Seconds to wait for graceful exit (default: 60)")
    p_preempt.add_argument("--force", action="store_true", help="SIGKILL if timeout")

    # resume
    p_resume = sub.add_parser("resume", help="Resume a preempted job from checkpoint")
    p_resume.add_argument("job_id", type=int, help="Job ID to resume")

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
    elif args.command == "eta":
        cmd_eta(args)
    elif args.command == "preempt":
        cmd_preempt(args)
    elif args.command == "resume":
        cmd_resume(args)
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
