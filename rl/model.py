"""策略价值网络（AlphaZero 风格 ResNet）——DESIGN.md §7。

网络结构：
  Stem  : Conv3×3(C_in→F) + BN + ReLU
  Body  : N × ResidualBlock（可选 SE）
  Policy head : Conv1×1(F→F) + BN + ReLU → Conv1×1(F→90)，通道=终点格、空间=起点格，
                转置展平为 8100 维 logits（action = from_sq*90 + to_sq）
  Value  head : Conv1×1(F→8) + BN + ReLU → Flatten → FC(256) + ReLU → FC(1) → tanh

预设尺寸（参见 DESIGN §7）：
  small  : 5块 × 64通道  ≈ 0.6M 参数
  medium : 10块 × 128通道 ≈ 3.2M 参数（默认）
  large  : 15块 × 192通道 ≈ 10M 参数

forward(x) -> (policy_logits[B, 8100], value[B])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.config import ModelConfig
from xiangqi.constants import ACTION_SIZE, ROWS, COLS

# 棋盘格数
_BOARD_SQUARES = ROWS * COLS  # 90


# ---------------------------------------------------------------------------
# Squeeze-Excitation 块
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """通道 Squeeze-Excitation（DESIGN §7 可选）。

    压缩比固定为 4（filters // 4），如 filters<4 则退化为恒等。
    """

    def __init__(self, filters: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(filters // reduction, 1)
        self.fc1 = nn.Linear(filters, mid)
        self.fc2 = nn.Linear(mid, filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: 全局均值池化 (B, F)
        se = x.mean(dim=(2, 3))
        # Excitation: 两层全连接
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        # Scale: 逐通道缩放
        return x * se.unsqueeze(2).unsqueeze(3)


# ---------------------------------------------------------------------------
# 残差块
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """单个残差块：Conv-BN-ReLU-Conv-BN-(SE)-Skip-ReLU。"""

    # TorchScript 要求显式声明可选子模块类型
    se_block: Optional[SEBlock]

    def __init__(self, filters: int, se: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.se_block = SEBlock(filters) if se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se_block is not None:
            out = self.se_block(out)
        return F.relu(out + residual)


# ---------------------------------------------------------------------------
# 主网络
# ---------------------------------------------------------------------------

class XiangqiNet(nn.Module):
    """中国象棋策略价值网络。

    forward 输入：x (B, C_in, 10, 9)  float32
    forward 输出：(policy_logits (B, 8100), value (B,))
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        F = cfg.filters
        C_in = cfg.in_channels  # 14*T + 3

        # Stem
        self.stem_conv = nn.Conv2d(C_in, F, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(F)

        # 残差体
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(F, se=cfg.se) for _ in range(cfg.blocks)]
        )

        # Policy head：全卷积结构化头（输出通道 = 终点格 to_sq，空间位置 = 起点格 from_sq）
        # 相比 Flatten→FC(2880→8100) 省约 23M 参数，且保留终点格的空间结构
        self.policy_conv = nn.Conv2d(F, F, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(F)
        self.policy_out = nn.Conv2d(F, _BOARD_SQUARES, 1)

        # Value head
        self.value_conv = nn.Conv2d(F, 8, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * _BOARD_SQUARES, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stem
        out = F.relu(self.stem_bn(self.stem_conv(x)))

        # 残差块
        out = self.res_blocks(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = self.policy_out(p)                    # (B, 90, 10, 9) 通道=to_sq
        p = p.flatten(2)                          # (B, 90_to, 90_from)
        # action = from_sq*90 + to_sq，故先转置再展平
        policy_logits = p.transpose(1, 2).reshape(-1, ACTION_SIZE)  # (B, 8100)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)                          # (B, 8*90)
        v = F.relu(self.value_fc1(v))             # (B, 256)
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (B,)

        return policy_logits, value


# ---------------------------------------------------------------------------
# 工厂与检查点工具
# ---------------------------------------------------------------------------

def create_model(cfg: ModelConfig) -> XiangqiNet:
    """根据 ModelConfig 创建网络实例（未初始化权重）。"""
    return XiangqiNet(cfg)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> Tuple[XiangqiNet, Dict[str, Any]]:
    """从磁盘加载 checkpoint，重建网络并 load_state_dict。

    参数
    ----
    path   : checkpoint 文件路径（.pt）
    device : 加载到的设备字符串，如 "cpu" 或 "cuda:0"

    返回
    ----
    (model, ckpt_dict)
      model     : 已加载权重的 XiangqiNet（eval 模式）
      ckpt_dict : 原始 checkpoint 字典
    """
    ckpt: Dict[str, Any] = torch.load(
        path, map_location=torch.device(device), weights_only=False
    )

    # 从 model_config 字典重建 ModelConfig
    mc_dict = ckpt["model_config"]
    cfg = ModelConfig(
        blocks=int(mc_dict["blocks"]),
        filters=int(mc_dict["filters"]),
        se=bool(mc_dict.get("se", False)),
        history_steps=int(mc_dict.get("history_steps", 2)),
    )

    model = XiangqiNet(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(torch.device(device))
    model.eval()
    return model, ckpt


def save_checkpoint(
    model: XiangqiNet,
    cfg: ModelConfig,
    path: str | Path,
    iteration: int = 0,
    optimizer_state: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """将模型权重与配置保存为标准 checkpoint 格式（DESIGN §7）。

    格式：
    {
      "model_state"  : state_dict,
      "model_config" : {"blocks","filters","se","history_steps"},
      "iteration"    : int,
      "optimizer_state" : dict | None,
      "meta"         : dict,
    }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": {
            "blocks": cfg.blocks,
            "filters": cfg.filters,
            "se": cfg.se,
            "history_steps": cfg.history_steps,
        },
        "iteration": iteration,
        "optimizer_state": optimizer_state,
        "meta": meta if meta is not None else {},
    }
    torch.save(ckpt, path)
