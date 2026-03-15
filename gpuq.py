#!/usr/bin/env python3
"""gpuq — General-purpose single-GPU job queue.

A lightweight single-GPU job scheduler that:
  - Queues experiments and runs them sequentially (one GPU job at a time)
  - Auto-discovers .venv in the working directory
  - Logs stdout/stderr per job with timestamps
  - Tracks status (pending/running/done/failed) in a JSON state file
  - Supports dependencies between jobs
  - Retries failed jobs (configurable)
  - Sends desktop notification on completion (notify-send)

Usage:
  # Add jobs (uses current directory as working dir)
  cd ~/repo/my-project && gpuq add train.py --lr 1e-4
  gpuq add eval.py --checkpoint ./outputs/best

  # Specify working directory explicitly
  gpuq add --dir ~/repo/my-project train.py --lr 1e-4

  # Specify python interpreter explicitly
  gpuq add --python ~/repo/my-project/.venv/bin/python train.py

  # Add with dependency (waits for job #1 to finish)
  gpuq add eval.py --after 1

  # Add with a name for easier identification
  gpuq add --name "finetune-bert" train.py --model bert-base

  # Set env vars for this job
  gpuq add --env CUDA_VISIBLE_DEVICES=1 train.py

  # Run the queue (blocks until all done, or daemonize)
  gpuq run
  gpuq run --daemon

  # Monitor
  gpuq status         # show queue state
  gpuq log 1          # tail log of job #1
  gpuq cancel 3       # cancel pending job #3
  gpuq clear          # remove completed/failed jobs
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────
STATE_DIR = Path.home() / ".gpuq"
STATE_FILE = STATE_DIR / "queue.json"
LOG_DIR = STATE_DIR / "logs"

DEFAULT_ENV_VARS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_VISIBLE_DEVICES": "0",
}


# ── State management ────────────────────────────────────────
def _ensure_dirs():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _load_state() -> dict:
    _ensure_dirs()
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"next_id": 1, "jobs": []}


def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _find_python(directory: str) -> str:
    """Find a .venv python in the given directory, walking up to 1 parent."""
    d = Path(directory)
    for check_dir in [d, d.parent]:
        venv_python = check_dir / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
    return "python3"


# ── Commands ────────────────────────────────────────────────
def cmd_add(args):
    """Add a job to the queue."""
    state = _load_state()
    job_id = state["next_id"]

    script = args.script
    extra_args = args.extra_args or []

    # Resolve working directory
    cwd = os.path.abspath(args.dir if args.dir else os.getcwd())

    # Resolve python
    if args.python:
        python_path = args.python
    else:
        python_path = _find_python(cwd)

    # Resolve script path (relative to cwd)
    script_path = Path(cwd) / script
    if not script_path.exists():
        # Try as absolute path
        if Path(script).exists():
            script_path = Path(script).resolve()
            script = str(script_path)
        else:
            print(f"WARNING: {script_path} does not exist (will fail at runtime)")

    # Job name: user-provided or derived from directory + script
    if args.name:
        name = args.name
    else:
        dir_name = Path(cwd).name
        script_base = Path(script).stem
        name = f"{dir_name}/{script_base}"

    # Parse extra env vars
    job_env = {}
    if args.env:
        for item in args.env:
            if "=" in item:
                k, v = item.split("=", 1)
                job_env[k] = v

    job = {
        "id": job_id,
        "name": name,
        "script": script,
        "args": extra_args,
        "python": python_path,
        "cwd": cwd,
        "env": job_env,
        "status": "pending",
        "after": args.after,
        "retries": args.retries,
        "attempt": 0,
        "added_at": datetime.now().isoformat(),
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "log_file": str(LOG_DIR / f"job_{job_id:03d}_{Path(script).stem}.log"),
    }

    state["jobs"].append(job)
    state["next_id"] = job_id + 1
    _save_state(state)

    dep_str = f" (after job #{args.after})" if args.after else ""
    print(f"[+] Job #{job_id}: {name} {' '.join(extra_args)}{dep_str}")


def cmd_status(args):
    """Show queue status."""
    state = _load_state()
    jobs = state["jobs"]

    if not jobs:
        print("Queue is empty.")
        return

    symbols = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌", "cancelled": "🚫"}

    print(f"{'ID':>3}  {'St':>2}  {'Name':<30} {'Script':<25} {'Time':>8}  {'Info'}")
    print("─" * 90)

    for j in jobs:
        sym = symbols.get(j["status"], "?")
        elapsed = ""
        if j["started_at"] and j["finished_at"]:
            t0 = datetime.fromisoformat(j["started_at"])
            t1 = datetime.fromisoformat(j["finished_at"])
            secs = (t1 - t0).total_seconds()
            elapsed = f"{int(secs//60)}m{int(secs%60):02d}s"
        elif j["started_at"]:
            t0 = datetime.fromisoformat(j["started_at"])
            secs = (datetime.now() - t0).total_seconds()
            elapsed = f"{int(secs//60)}m{int(secs%60):02d}s"

        info = ""
        if j.get("after"):
            info += f"after #{j['after']} "
        if j["exit_code"] is not None:
            info += f"exit={j['exit_code']}"
        if j.get("attempt", 0) > 1:
            info += f" attempt={j['attempt']}"

        name = j.get("name", j.get("exp", "?"))
        if len(name) > 30:
            name = "..." + name[-27:]

        script_short = j["script"]
        if len(script_short) > 25:
            script_short = "..." + script_short[-22:]

        print(f"{j['id']:>3}  {sym}  {name:<30} {script_short:<25} {elapsed:>8}  {info}")


def cmd_log(args):
    """Show log for a job."""
    state = _load_state()
    job = next((j for j in state["jobs"] if j["id"] == args.job_id), None)
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
    """Cancel a pending job."""
    state = _load_state()
    job = next((j for j in state["jobs"] if j["id"] == args.job_id), None)
    if not job:
        print(f"Job #{args.job_id} not found.")
        return
    if job["status"] != "pending":
        print(f"Job #{args.job_id} is {job['status']}, can only cancel pending jobs.")
        return
    job["status"] = "cancelled"
    _save_state(state)
    print(f"[x] Job #{args.job_id} cancelled.")


def cmd_clear(args):
    """Remove completed/failed/cancelled jobs."""
    state = _load_state()
    before = len(state["jobs"])
    state["jobs"] = [j for j in state["jobs"] if j["status"] in ("pending", "running")]
    after = len(state["jobs"])
    _save_state(state)
    print(f"Cleared {before - after} jobs, {after} remaining.")


def cmd_run(args):
    """Run the queue — execute pending jobs sequentially."""
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
        state = _load_state()
        pending = [j for j in state["jobs"] if j["status"] == "pending"]

        if not pending:
            break

        runnable = None
        for j in pending:
            if j.get("after"):
                dep = next((d for d in state["jobs"] if d["id"] == j["after"]), None)
                if dep and dep["status"] not in ("done", "failed", "cancelled"):
                    continue
            runnable = j
            break

        if not runnable:
            running = [j for j in state["jobs"] if j["status"] == "running"]
            if running:
                time.sleep(5)
                continue
            else:
                print("WARNING: Deadlock — pending jobs have unresolvable dependencies.")
                break

        _run_job(runnable)

    _notify_done()
    state = _load_state()
    n_done = sum(1 for j in state["jobs"] if j["status"] == "done")
    n_fail = sum(1 for j in state["jobs"] if j["status"] == "failed")
    print(f"\n{'='*60}")
    print(f"gpuq finished at {datetime.now().isoformat()}")
    print(f"Results: {n_done} done, {n_fail} failed")
    print(f"{'='*60}")


def _run_job(job: dict):
    """Execute a single job."""
    state = _load_state()

    job_ref = next(j for j in state["jobs"] if j["id"] == job["id"])
    job_ref["status"] = "running"
    job_ref["attempt"] = job_ref.get("attempt", 0) + 1
    job_ref["started_at"] = datetime.now().isoformat()
    _save_state(state)

    cmd = [job["python"], "-u", job["script"]] + job.get("args", [])
    env = {**os.environ, **DEFAULT_ENV_VARS, **job.get("env", {})}

    name = job.get("name", job.get("exp", "?"))
    print(f">>> [{job['id']}] {name} {' '.join(job.get('args', []))}")
    print(f"    cwd: {job['cwd']}")
    print(f"    log: {job['log_file']}")
    print(f"    started: {job_ref['started_at']}")

    log_path = Path(job["log_file"])
    with open(log_path, "a") as log_f:
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Job #{job['id']} attempt #{job_ref['attempt']}\n")
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"Started: {job_ref['started_at']}\n")
        log_f.write(f"{'='*60}\n\n")
        log_f.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=job["cwd"],
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            proc.wait()
            exit_code = proc.returncode
        except Exception as e:
            log_f.write(f"\nEXCEPTION: {e}\n")
            exit_code = -1

        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Exit code: {exit_code}\n")
        log_f.write(f"Finished: {datetime.now().isoformat()}\n")

    state = _load_state()
    job_ref = next(j for j in state["jobs"] if j["id"] == job["id"])
    job_ref["finished_at"] = datetime.now().isoformat()
    job_ref["exit_code"] = exit_code

    if exit_code == 0:
        job_ref["status"] = "done"
        print(f"    ✅ done (exit 0)")
    else:
        retries = job_ref.get("retries", 0)
        if job_ref["attempt"] < retries + 1:
            job_ref["status"] = "pending"
            print(f"    ⚠️  failed (exit {exit_code}), will retry ({job_ref['attempt']}/{retries+1})")
        else:
            job_ref["status"] = "failed"
            print(f"    ❌ failed (exit {exit_code})")

    _save_state(state)


def _notify_done():
    """Send desktop notification."""
    try:
        state = _load_state()
        n_done = sum(1 for j in state["jobs"] if j["status"] == "done")
        n_fail = sum(1 for j in state["jobs"] if j["status"] == "failed")
        msg = f"gpuq: {n_done} done, {n_fail} failed"
        subprocess.run(
            ["notify-send", "-u", "normal", "GPU Queue Complete", msg],
            timeout=5, capture_output=True
        )
    except Exception:
        pass


# ── CLI ─────────────────────────────────────────────────────
def main():
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

    # run
    p_run = sub.add_parser("run", help="Execute the queue")
    p_run.add_argument("--daemon", action="store_true", help="Run in background")

    args, unknown = parser.parse_known_args()

    if args.command == "add":
        args.extra_args = unknown
        cmd_add(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "cancel":
        cmd_cancel(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
