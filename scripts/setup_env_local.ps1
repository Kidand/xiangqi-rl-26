# Windows 本地 CPU 环境配置脚本
# 幂等性：可重复运行
# 用法：Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process; .\scripts\setup_env_local.ps1

param(
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Write-Host "=== 中国象棋 AlphaZero RL 系统 · 本地 Windows 环境配置 ===" -ForegroundColor Green
Write-Host "项目目录: $ProjectDir"

# 检查并安装 uv
Write-Host "[INFO] 检查 uv..."
try {
    $uvVersion = & uv --version 2>$null
    Write-Host "[OK] uv 已安装: $uvVersion" -ForegroundColor Green
} catch {
    Write-Host "[INFO] 安装 uv..."
    pip install uv
    Write-Host "[OK] uv 安装完成" -ForegroundColor Green
}

# 创建虚拟环境
Set-Location $ProjectDir
if (-not (Test-Path ".venv")) {
    Write-Host "[INFO] 创建虚拟环境 (.venv)..."
    uv venv --python $PythonVersion .venv
    Write-Host "[OK] 虚拟环境创建完成" -ForegroundColor Green
} else {
    Write-Host "[OK] 虚拟环境已存在" -ForegroundColor Green
}

# 激活虚拟环境
Write-Host "[INFO] 激活虚拟环境..."
& .\.venv\Scripts\Activate.ps1
Write-Host "[OK] 虚拟环境已激活" -ForegroundColor Green

# 安装 PyTorch（CPU 版本）
Write-Host "[INFO] 安装 PyTorch CPU 版本..."
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
Write-Host "[OK] PyTorch 安装完成" -ForegroundColor Green

# 安装其他依赖
Write-Host "[INFO] 安装项目依赖..."
uv pip install `
    numpy `
    pyyaml `
    rich `
    fastapi `
    "uvicorn[standard]" `
    tensorboard `
    pytest
Write-Host "[OK] 所有依赖安装完成" -ForegroundColor Green

# 验证环境
Write-Host ""
Write-Host "=== 环境验证 ===" -ForegroundColor Cyan

python -c @"
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('✓ 正常 (本地 CPU 版本)')
"

python -c @"
import numpy
import yaml
import rich
import fastapi
import tensorboard
print('依赖检查: numpy, pyyaml, rich, fastapi, tensorboard ✓')
"

Write-Host ""
Write-Host "=== 配置完成 ===" -ForegroundColor Green
Write-Host "后续使用（在虚拟环境中）:"
Write-Host "  python -m gui.server              # 启动 GUI 服务（http://127.0.0.1:8000）"
Write-Host "  python -m rl.train --config configs/laptop.yaml   # CPU 小规模试训"
Write-Host "  python -m pytest tests/           # 运行测试"
Write-Host ""
