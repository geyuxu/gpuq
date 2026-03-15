"""Integration tests for gpuq."""
import json
import os
import subprocess
import sys
from pathlib import Path

GPUQ = str(Path(__file__).resolve().parent.parent / "gpuq.py")
DUMMY_JOB = str(Path(__file__).resolve().parent / "dummy_job.py")
DUMMY_DIR = str(Path(__file__).resolve().parent)
STATE_DIR = Path.home() / ".gpuq"
STATE_FILE = STATE_DIR / "queue.json"


def gpuq(*args, check=True):
    """Run gpuq CLI and return stdout."""
    result = subprocess.run(
        [sys.executable, GPUQ, *args],
        capture_output=True, text=True, timeout=60,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"gpuq {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout.strip()


def reset():
    """Reset queue state."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def load_state():
    return json.loads(STATE_FILE.read_text())


# ── Tests ───────────────────────────────────────────────────

def test_add_and_status():
    """Test adding jobs and checking status."""
    reset()
    out = gpuq("add", "--dir", DUMMY_DIR, "--name", "t1", "dummy_job.py", "--job-name", "hello", "--duration", "1")
    assert "Job #1" in out

    out = gpuq("add", "--dir", DUMMY_DIR, "--name", "t2", "dummy_job.py", "--job-name", "world", "--duration", "1")
    assert "Job #2" in out

    state = load_state()
    assert len(state["jobs"]) == 2
    assert state["jobs"][0]["status"] == "pending"
    assert state["jobs"][1]["status"] == "pending"

    out = gpuq("status")
    assert "t1" in out
    assert "t2" in out
    print("PASS: test_add_and_status")


def test_run_basic():
    """Test basic sequential execution."""
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "basic-1", "dummy_job.py", "--job-name", "a", "--duration", "1")
    gpuq("add", "--dir", DUMMY_DIR, "--name", "basic-2", "dummy_job.py", "--job-name", "b", "--duration", "1")
    gpuq("run")

    state = load_state()
    assert state["jobs"][0]["status"] == "done"
    assert state["jobs"][1]["status"] == "done"
    assert state["jobs"][0]["exit_code"] == 0
    assert state["jobs"][1]["exit_code"] == 0
    print("PASS: test_run_basic")


def test_failure():
    """Test that failing jobs are marked as failed."""
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "fail-job", "dummy_job.py", "--job-name", "oops", "--duration", "1", "--fail")
    gpuq("run")

    state = load_state()
    assert state["jobs"][0]["status"] == "failed"
    assert state["jobs"][0]["exit_code"] == 1
    print("PASS: test_failure")


def test_dependency():
    """Test that --after dependency is respected."""
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "dep-first", "dummy_job.py", "--job-name", "first", "--duration", "1")
    gpuq("add", "--dir", DUMMY_DIR, "--name", "dep-second", "dummy_job.py", "--job-name", "second", "--duration", "1", "--after", "1")
    gpuq("run")

    state = load_state()
    assert state["jobs"][0]["status"] == "done"
    assert state["jobs"][1]["status"] == "done"
    # Second job must start after first finishes
    assert state["jobs"][1]["started_at"] >= state["jobs"][0]["finished_at"]
    print("PASS: test_dependency")


def test_retry():
    """Test retry on failure."""
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "retry-job", "dummy_job.py",
         "--job-name", "retry", "--duration", "1", "--fail", "--retries", "1")
    gpuq("run")

    state = load_state()
    job = state["jobs"][0]
    assert job["status"] == "failed"
    assert job["attempt"] == 2  # original + 1 retry
    print("PASS: test_retry")


def test_cancel():
    """Test cancelling a pending job."""
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "will-cancel", "dummy_job.py", "--duration", "1")

    out = gpuq("cancel", "1")
    assert "cancelled" in out

    state = load_state()
    assert state["jobs"][0]["status"] == "cancelled"
    print("PASS: test_cancel")


def test_clear():
    """Test clearing finished jobs."""
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "clear-1", "dummy_job.py", "--duration", "1")
    gpuq("add", "--dir", DUMMY_DIR, "--name", "clear-2", "dummy_job.py", "--duration", "1")
    gpuq("run")

    out = gpuq("clear")
    assert "Cleared 2" in out

    state = load_state()
    assert len(state["jobs"]) == 0
    print("PASS: test_clear")


def test_log():
    """Test viewing job logs."""
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "log-job", "dummy_job.py", "--job-name", "logged", "--duration", "1")
    gpuq("run")

    out = gpuq("log", "1")
    assert "logged" in out
    print("PASS: test_log")


def test_env_override():
    """Test --env passes environment variables."""
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "env-job", "--env", "MY_TEST_VAR=hello123",
         "dummy_job.py", "--job-name", "env", "--duration", "1")

    state = load_state()
    assert state["jobs"][0]["env"]["MY_TEST_VAR"] == "hello123"
    print("PASS: test_env_override")


def test_mixed_success_and_failure():
    """Test a queue with both passing and failing jobs."""
    reset()
    gpuq("add", "--dir", DUMMY_DIR, "--name", "mix-ok-1", "dummy_job.py", "--job-name", "ok1", "--duration", "1")
    gpuq("add", "--dir", DUMMY_DIR, "--name", "mix-fail", "dummy_job.py", "--job-name", "bad", "--duration", "1", "--fail")
    gpuq("add", "--dir", DUMMY_DIR, "--name", "mix-ok-2", "dummy_job.py", "--job-name", "ok2", "--duration", "1")
    gpuq("run")

    state = load_state()
    assert state["jobs"][0]["status"] == "done"
    assert state["jobs"][1]["status"] == "failed"
    assert state["jobs"][2]["status"] == "done"  # should still run after #2 fails
    print("PASS: test_mixed_success_and_failure")


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
