# gpuq

A lightweight single-GPU job queue for running experiments sequentially.

[中文版](README_zh.md)

## Features

- Queue experiments and run them one at a time (avoids GPU OOM)
- Auto-discovers `.venv` in the working directory
- Logs stdout/stderr per job with timestamps
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
gpuq status         # show all jobs
gpuq log 1          # tail log of job #1
gpuq log 1 -n 100   # last 100 lines

# Manage
gpuq cancel 3       # cancel a pending job
gpuq clear          # remove finished jobs
```

## Claude Code Integration

gpuq ships with a [Claude Code](https://claude.ai/claude-code) slash command skill. Once registered, you can manage the queue via `/gpuq` using natural language — no need to remember CLI syntax.

```bash
# Register the skill
ln -sf $(pwd)/claude-skill/gpuq.md ~/.claude/commands/gpuq.md
```

Then in Claude Code:
- `/gpuq check the queue`
- `/gpuq run train.py in my-project`
- `/gpuq run train first, then eval`

## How It Works

- State: `~/.gpuq/queue.json`
- Logs: `~/.gpuq/logs/`
- Jobs run sequentially — one GPU job at a time
- Auto-discovers `.venv/bin/python` in the job's working directory
- Default env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_VISIBLE_DEVICES=0`
- Override with `--env KEY=VALUE`
