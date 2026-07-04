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

# 构建训练命令数组
cmd_array=("python" "-m" "rl.train")

# 检查用户是否显式传递了 --config
has_config=false
for arg in "$@"; do
    if [[ "$arg" == --config* ]]; then
        has_config=true
        break
    fi
done

# 如果用户没有指定 config，使用默认的
if [ "$has_config" = false ]; then
    cmd_array+=("--config" "configs/cloud_8xh100.yaml")
fi

# 追加用户传递的所有参数
if [ $# -gt 0 ]; then
    cmd_array+=("$@")
fi

# 构建显示用的命令字符串（用于日志）
TRAIN_CMD=$(printf '%s ' "${cmd_array[@]}")

echo "=========================================="
echo "中国象棋 AlphaZero RL 训练启动"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "命令: $TRAIN_CMD"
echo "日志: logs/train.log"
echo "时间戳: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 执行训练
exec "${cmd_array[@]}"
