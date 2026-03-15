# gpuq

轻量级单 GPU 任务队列，按顺序执行实验任务。

[English](README.md)

## 特性

- 排队执行实验，一次只跑一个（避免 GPU OOM）
- 自动发现工作目录中的 `.venv`
- **接管运行中的进程** — 把已有的 GPU 任务纳入管理
- **SQLite 存储** — 崩溃安全的状态持久化
- **重启后恢复** — 自动检测中断的任务，选择性重新执行
- PID 追踪，自动检测进程存活状态
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
gpuq status         # 查看所有任务（自动检测死进程）
gpuq log 1          # 查看任务 #1 的日志
gpuq log 1 -n 100   # 最后 100 行

# 管理
gpuq cancel 3       # 取消等待中的任务
gpuq clear          # 清理已完成的任务
```

### 接管运行中的进程

把已经在跑的 GPU 进程纳入 gpuq 管理：

```bash
# 查看 GPU 进程
nvidia-smi

# 通过 PID 接管（自动从 /proc 读取命令行和工作目录）
gpuq adopt 12345 --name "my-training"

# 查看状态
gpuq status
```

### 重启后恢复

机器重启或 gpuq 崩溃后，`status` 会自动检测死进程并标记为中断：

```bash
# 查看状态
gpuq status

# 重新排队所有中断的任务
gpuq recover --all

# 或者选择性恢复
gpuq recover --jobs 1 3

# 然后执行
gpuq run --daemon
```

## AI 助手集成

gpuq 自带 AI skill 文件（`ai-skill/gpuq.md`），让 AI 编程助手可以用自然语言管理队列，不需要记任何参数。

### Claude Code

```bash
ln -sf $(pwd)/ai-skill/gpuq.md ~/.claude/commands/gpuq.md
```

然后直接说：`/gpuq 看看队列`、`/gpuq 跑一下 train.py`、`/gpuq 先跑 train 再跑 eval`

### 其他 AI 工具

将 `ai-skill/gpuq.md` 复制或软链接到对应工具的自定义命令目录即可。

## 工作原理

- 状态数据库：`~/.gpuq/gpuq.db`（SQLite WAL 模式）
- 日志目录：`~/.gpuq/logs/`
- 任务按顺序执行，同时只跑一个 GPU 任务
- 每个运行中的任务记录 PID，`status` 自动检测进程是否存活
- 自动查找工作目录中的 `.venv/bin/python`
- 默认环境变量：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`CUDA_VISIBLE_DEVICES=0`
- 用 `--env KEY=VALUE` 覆盖
- 首次运行自动从旧版 JSON 状态（`queue.json`）迁移到 SQLite
