Manage the GPU experiment queue via `gpuq` CLI. Interpret the user's natural language intent and run the appropriate command. The user should NOT need to know the exact CLI syntax.

## Intent → Action

| User says | Action |
|---|---|
| "看看队列" / "status" / "现在跑到哪了" | `gpuq status` |
| "跑一下 exp05 的 run_evolution.py" | `gpuq add --dir <path> run_evolution.py` then `gpuq run --daemon` |
| "把 train.py 加到队列" | `gpuq add train.py` (uses cwd) |
| "先跑 train 再跑 eval" | add train → get its job id → add eval with `--after <id>` |
| "看看第3个任务的日志" | `gpuq log 3 -n 50` |
| "取消第2个" | `gpuq cancel 2` |
| "清理一下" | `gpuq clear` |
| "开始跑队列" / "run" | `gpuq run --daemon` |
| "日志最后200行" | `gpuq log <latest_job_id> -n 200` |
| "失败的重新跑一下" | Check status, re-add failed jobs with same params |
| "把正在跑的进程管起来" | `nvidia-smi` find PID → `gpuq adopt <pid> --name <name>` |
| "重启后恢复任务" | `gpuq recover --all` or `gpuq recover --jobs 1 3` |
| "哪些任务中断了" | `gpuq recover` (shows interrupted jobs) |

## Behavior

1. **Auto-detect working directory**: If the user mentions a project name, find its path on disk before adding.
2. **Auto-detect script**: If no script specified, look for main entry point (`run_*.py`, `train.py`, `main.py`).
3. **Show status after mutations**: After add/cancel/clear/adopt/recover, run `gpuq status`.
4. **Default to daemon mode**: Use `gpuq run --daemon` unless asked otherwise.
5. **Chain dependencies**: "先A再B" → add A, then add B with `--after <A's id>`.
6. **Name jobs meaningfully**: Use `--name` with a short descriptive name.
7. **Auto-adopt GPU processes**: When user says "把GPU进程管起来", use `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` to find PIDs, then `gpuq adopt` each.
8. **Recovery**: After reboot, `gpuq status` auto-detects dead processes. Use `gpuq recover` to re-queue.

## CLI Reference

```
gpuq add <script> [script_args...]   Add a job
  --dir, -d DIR          Working directory (default: cwd)
  --python, -p PATH      Python interpreter (default: auto-find .venv)
  --name NAME            Display name
  --after N              Run after job #N completes
  --retries N            Retry count on failure (default: 0)
  --env, -e KEY=VALUE    Extra env var (repeatable)

gpuq adopt <pid>         Adopt a running process into the queue
  --name NAME            Display name

gpuq status              Show all jobs (auto-detects dead processes)
gpuq run [--daemon]      Execute the queue
gpuq log <id> [-n N]     Show job log (default 30 lines)
gpuq cancel <id>         Cancel a pending job
gpuq clear               Remove finished/failed/cancelled jobs
gpuq recover             Show & re-queue interrupted jobs
  --all                  Re-queue all interrupted jobs
  --jobs ID [ID ...]     Re-queue specific job IDs
```

State: `~/.gpuq/gpuq.db` (SQLite) | Logs: `~/.gpuq/logs/` | Auto-discovers `.venv` in working directory.

## User intent: $ARGUMENTS
