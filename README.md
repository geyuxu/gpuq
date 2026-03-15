# gpuq

A lightweight single-GPU job queue for running experiments sequentially.

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
# Clone
git clone git@github.com:geyuxu/gpuq.git
cd gpuq

# Symlink to PATH
ln -sf $(pwd)/gpuq.py ~/.local/bin/gpuq
chmod +x gpuq.py
```

## Claude Code Skill

gpuq 附带一个 [Claude Code](https://claude.ai/claude-code) slash command skill，让你可以在 Claude Code 中通过 `/gpuq` 来管理队列。

注册方法：

```bash
# 将 skill 文件软链接到 Claude Code 的 commands 目录
ln -sf $(pwd)/claude-skill/gpuq.md ~/.claude/commands/gpuq.md
```

注册后在 Claude Code 中使用 `/gpuq status`、`/gpuq add train.py --lr 1e-4` 等即可。

## Usage

```bash
# Add a job (uses current directory)
cd ~/repo/my-project
gpuq add train.py --lr 1e-4 --epochs 10

# Add with explicit working directory
gpuq add --dir ~/repo/my-project train.py --lr 1e-4

# Add with explicit python interpreter
gpuq add --python ~/repo/my-project/.venv/bin/python train.py

# Add with a name
gpuq add --name "bert-finetune" train.py --model bert-base

# Add with dependency (runs after job #1)
gpuq add eval.py --after 1

# Add with retries
gpuq add train.py --retries 2

# Set env vars for a job
gpuq add --env CUDA_VISIBLE_DEVICES=1 train.py

# Run the queue
gpuq run

# Run in background
gpuq run --daemon

# Check status
gpuq status

# View job log
gpuq log 1
gpuq log 1 -n 100

# Cancel a pending job
gpuq cancel 3

# Clear finished jobs
gpuq clear
```

## How it works

- State is stored in `~/.gpuq/queue.json`
- Logs go to `~/.gpuq/logs/`
- Jobs run sequentially — one GPU job at a time
- Auto-discovers `.venv/bin/python` in the job's working directory
- Default env vars: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_VISIBLE_DEVICES=0`
- Override with `--env KEY=VALUE`
