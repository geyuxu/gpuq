# gpuq

A lightweight single-GPU job queue for running experiments sequentially.

[中文版](README_zh.md)

## Features

- Queue experiments and run them one at a time (avoids GPU OOM)
- Auto-discovers `.venv` in the working directory
- **Adopt running processes** — bring existing GPU jobs under management
- **SQLite backend** — crash-safe state persistence
- **Recovery after restart** — detect interrupted jobs and selectively re-run
- PID tracking with automatic stale process detection
- Job dependencies (`--after N`)
- Configurable retries on failure
- Daemon mode for background execution
- Desktop notification on completion

## Install

```bash
git clone git@github.com:geyuxu/gpuq.git
cd gpuq
chmod +x gpuq.py
ln -sf $(pwd)/gpuq.py ~/.local/bin/gpuq
```

## Usage

```bash
# Add a job (uses current directory, auto-finds .venv)
gpuq add train.py --lr 1e-4 --epochs 10

# Specify working directory
gpuq add --dir ~/repo/my-project train.py --lr 1e-4

# Specify python interpreter
gpuq add --python ~/.venv/bin/python train.py

# Name a job for display
gpuq add --name "bert-finetune" train.py --model bert-base

# Job dependency (runs after job #1 completes)
gpuq add eval.py --after 1

# Retry on failure
gpuq add train.py --retries 2

# Set env vars
gpuq add --env CUDA_VISIBLE_DEVICES=1 train.py

# Run the queue
gpuq run            # foreground
gpuq run --daemon   # background

# Monitor
gpuq status         # show all jobs (auto-detects dead processes)
gpuq log 1          # tail log of job #1
gpuq log 1 -n 100   # last 100 lines

# Manage
gpuq cancel 3       # cancel a pending job
gpuq clear          # remove finished jobs
```

### Adopt Running Processes

Bring an already-running GPU process under gpuq management:

```bash
# Find GPU processes
nvidia-smi

# Adopt by PID (auto-reads cmdline and cwd from /proc)
gpuq adopt 12345 --name "my-training"

# Status will track it
gpuq status
```

### Recovery After Restart

If the machine restarts or gpuq crashes, `status` auto-detects dead processes and marks them as interrupted:

```bash
# See what happened
gpuq status

# Re-queue all interrupted jobs
gpuq recover --all

# Or selectively
gpuq recover --jobs 1 3

# Then run
gpuq run --daemon
```

## AI Assistant Integration

gpuq ships with a skill file (`skill/gpuq.md`) that lets AI coding assistants manage the queue via natural language — no need to remember CLI syntax.

### Claude Code

```bash
ln -sf $(pwd)/skill/gpuq.md ~/.claude/commands/gpuq.md
```

Then use `/gpuq check the queue`, `/gpuq run train.py in my-project`, etc.

### Other AI tools

Copy or symlink `skill/gpuq.md` into your tool's custom command/skill directory.

## How It Works

- State: `~/.gpuq/gpuq.db` (SQLite with WAL mode)
- Logs: `~/.gpuq/logs/`
- Jobs run sequentially — one GPU job at a time
- PID recorded for every running job; `status` checks if process is alive
- Auto-discovers `.venv/bin/python` in the job's working directory
- Default env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_VISIBLE_DEVICES=0`
- Override with `--env KEY=VALUE`
- Automatic migration from old JSON state (`queue.json`) to SQLite on first run
