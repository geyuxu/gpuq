"""Integration tests for gpuq (SQLite backend)."""
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

GPUQ = str(Path(__file__).resolve().parent.parent / "gpuq.py")
DUMMY_JOB = str(Path(__file__).resolve().parent / "dummy_job.py")
DUMMY_DIR = str(Path(__file__).resolve().parent)
STATE_DIR = Path.home() / ".gpuq"
DB_FILE = STATE_DIR / "gpuq.db"


def gpuq(*args, check=True, input_text=None):
    """Run gpuq CLI and return stdout."""
    result = subprocess.run(
        [sys.executable, GPUQ, *args],
        capture_output=True, text=True, timeout=60,
        input=input_text,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"gpuq {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def reset():
    """Reset queue state."""
    for f in STATE_DIR.glob("gpuq.db*"):
        f.unlink()
    # Also remove old JSON files from previous test runs
    for f in STATE_DIR.glob("queue.json*"):
        f.unlink()


def get_db():
    db = sqlite3.connect(str(DB_FILE))
    db.row_factory = sqlite3.Row
    return db


def get_db_with_tables():
    """Get DB and ensure tables exist (for tests that insert directly)."""
    # Run any gpuq command to init the DB
    gpuq("status")
    return get_db()


def get_jobs():
    db = get_db()
    jobs = [dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY id").fetchall()]
    db.close()
    return jobs


# ── Tests ───────────────────────────────────────────────────

def test_add_and_status():
    reset()
    out = gpuq("add", "--dir", DUMMY_DIR, "--name", "t1", "dummy_job.py", "--job-name", "hello", "--duration", "1")
    assert "Job #1" in out

    out = gpuq("add", "--dir", DUMMY_DIR, "--name", "t2", "dummy_job.py", "--job-name", "world", "--duration", "1")
    assert "Job #2" in out

    jobs = get_jobs()
    assert len(jobs) == 2
    assert jobs[0]["status"] == "pending"
    assert jobs[1]["status"] == "pending"

    out = gpuq("status")
    assert "t1" in out
    assert "t2" in out
    print("PASS: test_add_and_status")


def test_run_basic():
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "basic-1", "dummy_job.py", "--job-name", "a", "--duration", "1")
    gpuq("add", "--dir", DUMMY_DIR, "--name", "basic-2", "dummy_job.py", "--job-name", "b", "--duration", "1")
    gpuq("run")

    jobs = get_jobs()
    assert jobs[0]["status"] == "done"
    assert jobs[1]["status"] == "done"
    assert jobs[0]["exit_code"] == 0
    assert jobs[1]["exit_code"] == 0
    # PID should be recorded then cleared
    assert jobs[0]["pid"] is None  # cleared after done
    print("PASS: test_run_basic")


def test_failure():
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "fail-job", "dummy_job.py", "--job-name", "oops", "--duration", "1", "--fail")
    gpuq("run")

    jobs = get_jobs()
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["exit_code"] == 1
    print("PASS: test_failure")


def test_dependency():
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "dep-first", "dummy_job.py", "--job-name", "first", "--duration", "1")
    gpuq("add", "--dir", DUMMY_DIR, "--name", "dep-second", "dummy_job.py", "--job-name", "second", "--duration", "1", "--after", "1")
    gpuq("run")

    jobs = get_jobs()
    assert jobs[0]["status"] == "done"
    assert jobs[1]["status"] == "done"
    assert jobs[1]["started_at"] >= jobs[0]["finished_at"]
    print("PASS: test_dependency")


def test_retry():
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "retry-job", "dummy_job.py",
         "--job-name", "retry", "--duration", "1", "--fail", "--retries", "1")
    gpuq("run")

    jobs = get_jobs()
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["attempt"] == 2
    print("PASS: test_retry")


def test_cancel():
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "will-cancel", "dummy_job.py", "--duration", "1")
    out = gpuq("cancel", "1")
    assert "cancelled" in out

    jobs = get_jobs()
    assert jobs[0]["status"] == "cancelled"
    print("PASS: test_cancel")


def test_clear():
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "clear-1", "dummy_job.py", "--duration", "1")
    gpuq("add", "--dir", DUMMY_DIR, "--name", "clear-2", "dummy_job.py", "--duration", "1")
    gpuq("run")

    out = gpuq("clear")
    assert "Cleared 2" in out
    assert get_jobs() == []
    print("PASS: test_clear")


def test_log():
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "log-job", "dummy_job.py", "--job-name", "logged", "--duration", "1")
    gpuq("run")

    out = gpuq("log", "1")
    assert "logged" in out
    print("PASS: test_log")


def test_env_override():
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "env-job", "--env", "MY_TEST_VAR=hello123",
         "dummy_job.py", "--job-name", "env", "--duration", "1")

    jobs = get_jobs()
    env = json.loads(jobs[0]["env"])
    assert env["MY_TEST_VAR"] == "hello123"
    print("PASS: test_env_override")


def test_mixed_success_and_failure():
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "mix-ok-1", "dummy_job.py", "--job-name", "ok1", "--duration", "1")
    gpuq("add", "--dir", DUMMY_DIR, "--name", "mix-fail", "dummy_job.py", "--job-name", "bad", "--duration", "1", "--fail")
    gpuq("add", "--dir", DUMMY_DIR, "--name", "mix-ok-2", "dummy_job.py", "--job-name", "ok2", "--duration", "1")
    gpuq("run")

    jobs = get_jobs()
    assert jobs[0]["status"] == "done"
    assert jobs[1]["status"] == "failed"
    assert jobs[2]["status"] == "done"
    print("PASS: test_mixed_success_and_failure")


def test_adopt():
    """Test adopting a running process."""
    reset()
    # Start a long-running dummy process
    proc = subprocess.Popen(
        [sys.executable, DUMMY_JOB, "--job-name", "adopted", "--duration", "30"],
        cwd=DUMMY_DIR,
    )
    try:
        time.sleep(0.5)
        out = gpuq("adopt", str(proc.pid), "--name", "my-adopted")
        assert "Adopted" in out
        assert str(proc.pid) in out

        jobs = get_jobs()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "running"
        assert jobs[0]["pid"] == proc.pid
        assert jobs[0]["name"] == "my-adopted"

        # Status should show it as running
        out = gpuq("status")
        assert "🔄" in out
        assert "my-adopted" in out
    finally:
        proc.terminate()
        proc.wait()

    # After kill, status should detect it as interrupted
    out = gpuq("status")
    assert "⚡" in out
    jobs = get_jobs()
    assert jobs[0]["status"] == "interrupted"
    print("PASS: test_adopt")


def test_recover():
    """Test recovering interrupted jobs."""
    reset()
    # Simulate an interrupted job by inserting directly
    db = get_db_with_tables()
    db.execute("""
        INSERT INTO jobs (name, script, args, python, cwd, env, status,
                          retries, attempt, pid, added_at, started_at, log_file)
        VALUES ('interrupted-job', 'dummy_job.py', '["--duration", "1"]', ?, ?, '{}',
                'interrupted', 0, 1, NULL, ?, ?, ?)
    """, (sys.executable, DUMMY_DIR, "2025-01-01T00:00:00", "2025-01-01T00:00:00",
          str(Path(DUMMY_DIR) / "dummy.log")))
    db.commit()
    db.close()

    out = gpuq("recover", "--all")
    assert "Re-queued 1" in out

    jobs = get_jobs()
    assert jobs[0]["status"] == "pending"
    print("PASS: test_recover")


def test_recover_selective():
    """Test selectively recovering specific jobs."""
    reset()
    db = get_db_with_tables()
    for i in range(3):
        db.execute("""
            INSERT INTO jobs (name, script, args, python, cwd, env, status,
                              retries, attempt, added_at, started_at, log_file)
            VALUES (?, 'dummy_job.py', '[]', ?, ?, '{}', 'interrupted',
                    0, 1, ?, ?, ?)
        """, (f"int-{i}", sys.executable, DUMMY_DIR,
              "2025-01-01T00:00:00", "2025-01-01T00:00:00", "dummy.log"))
    db.commit()
    db.close()

    # Recover only job #1 and #3
    out = gpuq("recover", "--jobs", "1", "3")
    assert "Re-queued 2" in out

    jobs = get_jobs()
    assert jobs[0]["status"] == "pending"   # #1 recovered
    assert jobs[1]["status"] == "interrupted"  # #2 not recovered
    assert jobs[2]["status"] == "pending"   # #3 recovered
    print("PASS: test_recover_selective")


def test_pid_tracking():
    """Test that PID is recorded during execution."""
    reset()
    # Start a slow job in background via daemon-like approach
    gpuq("add", "--dir", DUMMY_DIR, "--name", "pid-track", "dummy_job.py", "--job-name", "slow", "--duration", "5")

    # Run in background
    proc = subprocess.Popen(
        [sys.executable, GPUQ, "run"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    # Wait a moment for the job to start
    time.sleep(2)

    jobs = get_jobs()
    assert len(jobs) == 1
    assert jobs[0]["status"] == "running"
    assert jobs[0]["pid"] is not None
    assert jobs[0]["pid"] > 0

    # Wait for completion
    proc.wait(timeout=30)
    jobs = get_jobs()
    assert jobs[0]["status"] == "done"
    assert jobs[0]["pid"] is None  # cleared after done
    print("PASS: test_pid_tracking")


def test_json_migration():
    """Test migration from old JSON format to SQLite."""
    reset()
    # Create old-style JSON state
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    json_state = {
        "next_id": 3,
        "jobs": [
            {
                "id": 1, "name": "old-job-1", "script": "train.py",
                "args": ["--lr", "1e-4"], "python": "python3",
                "cwd": "/tmp", "env": {}, "status": "done",
                "after": None, "retries": 0, "attempt": 1,
                "added_at": "2025-01-01T00:00:00",
                "started_at": "2025-01-01T00:00:01",
                "finished_at": "2025-01-01T00:01:00",
                "exit_code": 0, "log_file": "/tmp/job.log",
            },
            {
                "id": 2, "name": "old-job-2", "script": "eval.py",
                "args": [], "python": "python3",
                "cwd": "/tmp", "env": {}, "status": "pending",
                "after": 1, "retries": 0, "attempt": 0,
                "added_at": "2025-01-01T00:00:00",
                "started_at": None, "finished_at": None,
                "exit_code": None, "log_file": "/tmp/job2.log",
            },
        ],
    }
    json_file = STATE_DIR / "queue.json"
    json_file.write_text(json.dumps(json_state))

    # Running any command should trigger migration
    out = gpuq("status")
    assert "old-job-1" in out
    assert "old-job-2" in out

    # JSON file should be renamed to .bak
    assert not json_file.exists()
    assert json_file.with_suffix(".json.bak").exists()

    # Verify data in SQLite
    jobs = get_jobs()
    assert len(jobs) == 2
    assert jobs[0]["name"] == "old-job-1"
    assert jobs[0]["status"] == "done"
    assert jobs[1]["after_job"] == 1
    print("PASS: test_json_migration")


if __name__ == "__main__":
    tests = [
        test_add_and_status,
        test_run_basic,
        test_failure,
        test_dependency,
        test_retry,
        test_cancel,
        test_clear,
        test_log,
        test_env_override,
        test_mixed_success_and_failure,
        test_adopt,
        test_recover,
        test_recover_selective,
        test_pid_tracking,
        test_json_migration,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
