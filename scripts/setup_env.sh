#!/bin/bash
# 云端 Linux 环境配置脚本（CUDA 12.8，多 GPU）
# 幂等性：可重复运行，已存在的组件跳过
# 用法：bash scripts/setup_env.sh

set -e

PYTHON_VERSION="3.11"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== 中国象棋 AlphaZero RL 系统 · 云端环境配置 ==="
echo "项目目录: $PROJECT_DIR"

# 检查 uv 安装，若无则安装
if ! command -v uv &> /dev/null; then
    echo "[INFO] 未检测到 uv，开始安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 更新 PATH，使后续 uv 命令可用
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "[OK] uv 安装完成"
else
    echo "[OK] uv 已安装: $(uv --version)"
fi

# 创建虚拟环境（若已存在，uv venv 会跳过）
cd "$PROJECT_DIR"
if [ ! -d ".venv" ]; then
    echo "[INFO] 创建虚拟环境 (.venv)..."
    uv venv --python "$PYTHON_VERSION" .venv
    echo "[OK] 虚拟环境创建完成"
else
    echo "[OK] 虚拟环境已存在"
fi

# 激活虚拟环境
source .venv/bin/activate
echo "[OK] 虚拟环境已激活"

# 安装 PyTorch（CUDA 12.8）
echo "[INFO] 安装 PyTorch CUDA 12.8..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
echo "[OK] PyTorch 安装完成"

# 安装其他依赖
echo "[INFO] 安装项目依赖..."
uv pip install \
    numpy \
    pyyaml \
    rich \
    fastapi \
    "uvicorn[standard]" \
    tensorboard \
    pytest
echo "[OK] 所有依赖安装完成"

# 验证环境
echo ""
echo "=== 环境验证 ==="
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('警告: CUDA 不可用（检查驱动与 NVIDIA 工具包）')
"

python -c "
import numpy
import yaml
import rich
import fastapi
import tensorboard
print('依赖检查: numpy, pyyaml, rich, fastapi, tensorboard ✓')
"

echo ""
echo "=== 配置完成 ==="
echo "后续使用（云端 tmux 中）:"
echo "  cd $PROJECT_DIR"
echo "  source .venv/bin/activate"
echo "  python -m rl.train --config configs/cloud.yaml"
echo ""
