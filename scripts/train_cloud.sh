#!/bin/bash
# 云端训练启动脚本（8×H100）
# 在 tmux 会话内运行，支持传递额外参数如 --resume --config
# 用法：
#   bash scripts/train_cloud.sh                    # 默认 cloud_8xh100.yaml
#   bash scripts/train_cloud.sh --resume           # 续训
#   bash scripts/train_cloud.sh --config configs/custom.yaml
#   bash scripts/train_cloud.sh --config configs/custom.yaml --resume

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# 激活虚拟环境
source .venv/bin/activate

# 构建训练命令
TRAIN_CMD="python -m rl.train --config configs/cloud_8xh100.yaml"

# 透传额外参数
if [ $# -gt 0 ]; then
    TRAIN_CMD="python -m rl.train $@"
fi

echo "=========================================="
echo "中国象棋 AlphaZero RL 训练启动"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "命令: $TRAIN_CMD"
echo "日志: logs/train.log"
echo "时间戳: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 执行训练（所有参数透传）
exec $TRAIN_CMD
