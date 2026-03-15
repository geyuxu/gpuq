# gpuq

轻量级单 GPU 任务队列，按顺序执行实验任务。

[English](README.md)

## 特性

- 排队执行实验，一次只跑一个（避免 GPU OOM）
- 自动发现工作目录中的 `.venv`
- 每个任务独立记录 stdout/stderr 日志（带时间戳）
- 任务依赖（`--after N`）
- 失败自动重试
- 后台守护进程模式
- 完成后桌面通知

## 安装

```bash
git clone git@github.com:geyuxu/gpuq.git
cd gpuq
chmod +x gpuq.py
ln -sf $(pwd)/gpuq.py ~/.local/bin/gpuq
```

## 使用

```bash
# 添加任务（使用当前目录，自动查找 .venv）
gpuq add train.py --lr 1e-4 --epochs 10

# 指定工作目录
gpuq add --dir ~/repo/my-project train.py --lr 1e-4

# 指定 Python 解释器
gpuq add --python ~/.venv/bin/python train.py

# 给任务命名
gpuq add --name "bert-finetune" train.py --model bert-base

# 任务依赖（等 #1 完成后再跑）
gpuq add eval.py --after 1

# 失败重试
gpuq add train.py --retries 2

# 设置环境变量
gpuq add --env CUDA_VISIBLE_DEVICES=1 train.py

# 执行队列
gpuq run            # 前台
gpuq run --daemon   # 后台

# 监控
gpuq status         # 查看所有任务
gpuq log 1          # 查看任务 #1 的日志
gpuq log 1 -n 100   # 最后 100 行

# 管理
gpuq cancel 3       # 取消等待中的任务
gpuq clear          # 清理已完成的任务
```

## Claude Code 集成

gpuq 自带 [Claude Code](https://claude.ai/claude-code) slash command skill。注册后可以用自然语言通过 `/gpuq` 管理队列，不需要记任何参数。

```bash
# 注册 skill
ln -sf $(pwd)/claude-skill/gpuq.md ~/.claude/commands/gpuq.md
```

注册后在 Claude Code 中直接说：
- `/gpuq 看看队列`
- `/gpuq 跑一下 train.py`
- `/gpuq 先跑 train 再跑 eval`

## 工作原理

- 状态文件：`~/.gpuq/queue.json`
- 日志目录：`~/.gpuq/logs/`
- 任务按顺序执行，同时只跑一个 GPU 任务
- 自动查找工作目录中的 `.venv/bin/python`
- 默认环境变量：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`CUDA_VISIBLE_DEVICES=0`
- 用 `--env KEY=VALUE` 覆盖
