# -*- coding: utf-8 -*-
"""大模型迁移蒸馏工具（scripts/distill_bigmodel.py）CPU 微型端到端测试。

覆盖：
- onecycle_lr：warmup 线性升、峰值、余弦退到 lr_min 的边界。
- train_distill：小 buffer + 小模型跑几十步，断言损失有限且下降。
- 产出 checkpoint 能被 rl.model.load_checkpoint 重建并前向、model_config 四字段齐全。

全程 CPU、几秒内，不加 slow 标记。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ 不是已安装包（pyproject packages.find 只含 xiangqi/rl/gui），把仓库根
# 加入 sys.path 才能 import scripts.distill_bigmodel。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.config import Config, ModelConfig
from rl.model import create_model, load_checkpoint, save_checkpoint
from rl.replay_buffer import ReplayBuffer
from rl.train import _build_optimizer
from xiangqi.constants import ACTION_SIZE

from scripts.distill_bigmodel import _arena_config, onecycle_lr, train_distill


_HISTORY = 2
_CHANNELS = 14 * _HISTORY + 3  # 31
_STATE_SHAPE = (_CHANNELS, 10, 9)


def _make_buffer(n: int = 400, k: int = 6) -> ReplayBuffer:
    """构造 n 条随机样本的小 buffer：states uint8 (31,10,9)、稀疏 π（k 个合法索引、
    概率归一）、value∈[-1,1]。为让蒸馏损失可下降，π 与 value 由 states 的一个确定性
    函数生成（有可学习的信号，而非纯噪声）。"""
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=n + 10, state_shape=_STATE_SHAPE)
    states = rng.integers(0, 2, (n, *_STATE_SHAPE), dtype=np.uint8)
    sparse = []
    values = np.zeros(n, dtype=np.float32)
    for i in range(n):
        # 稀疏 π：k 个不重复合法索引，概率集中在由样本决定的一个主索引上。
        idx = rng.choice(ACTION_SIZE, k, replace=False).astype(np.int16)
        probs = rng.random(k).astype(np.float32)
        probs[0] += 2.0  # 制造一个主峰，π 非均匀才有可拟合信号
        probs /= probs.sum()
        sparse.append((idx, probs))
        # value：由 states 均值映射到 [-1,1] 的确定性信号
        values[i] = np.float32(np.tanh(states[i].mean() * 4.0 - 2.0))
    buf.add_game(states, sparse, values)
    return buf


def test_arena_config_matches_stage2_gate_strength():
    """seed-best 用的 arena 段须与 cloud_stage2_15x192se.yaml 的训练门控同强度，
    防止未来误改回 ArenaConfig 默认值（40 盘/200 sims/temp 0.5/8 步——对局高度
    相关，score>=0.5 判据下覆盖 best.pt 约等于抛硬币）。"""
    cfg = _arena_config("cpu", 1)
    assert cfg.arena.arena_games == 160
    assert cfg.arena.arena_sims == 400
    assert cfg.arena.arena_temp_moves == 16
    assert cfg.arena.arena_temperature == pytest.approx(0.8)


def test_onecycle_lr_shape():
    total = 1000
    peak, lo, warm = 1e-3, 1e-4, 100
    # warmup：线性升，末步接近峰值，且严格递增
    lr0 = onecycle_lr(0, total, peak, lo, warm)
    lr_mid_warm = onecycle_lr(50, total, peak, lo, warm)
    lr_end_warm = onecycle_lr(warm - 1, total, peak, lo, warm)
    assert 0 < lr0 < lr_mid_warm < lr_end_warm <= peak + 1e-12
    assert lr_end_warm == pytest.approx(peak, rel=1e-6)
    # warmup 后余弦退火：单调不增，末步落到 lr_min
    lr_after = onecycle_lr(warm, total, peak, lo, warm)
    lr_last = onecycle_lr(total, total, peak, lo, warm)
    assert lr_after <= peak + 1e-9
    assert lr_last == pytest.approx(lo, abs=1e-9)


def test_train_distill_decreases_and_finite():
    import torch

    # 固定全局 RNG：模型初始化（torch）与 buffer.sample（np.random 全局）都依赖它，
    # 显式播种使本用例的「损失下降」断言不受同进程其他测试遗留 RNG 状态影响。
    torch.manual_seed(1234)
    np.random.seed(1234)
    buf = _make_buffer(n=400)
    cfg = ModelConfig(blocks=3, filters=32, se=False, history_steps=_HISTORY)
    model = create_model(cfg)
    opt_cfg = Config()
    opt_cfg.train.optimizer = "adamw"
    opt_cfg.train.lr = 1e-3
    optimizer = _build_optimizer(model, opt_cfg)

    metrics = train_distill(
        model,
        optimizer,
        buf,
        epochs=8,
        batch_size=64,
        device="cpu",
        lr_peak=2e-3,
        lr_min=1e-4,
        warmup=5,   # 小规模，缩短 warmup
        log_fn=lambda *_: None,
    )

    ce = metrics["epoch_policy_ce"]
    vmse = metrics["epoch_value_mse"]
    assert len(ce) == 8 and len(vmse) == 8
    # 全程有限
    assert all(math.isfinite(x) for x in ce)
    assert all(math.isfinite(x) for x in vmse)
    assert math.isfinite(metrics["final_policy_ce"])
    assert math.isfinite(metrics["final_value_mse"])
    # 下降：末轮显著低于首轮（有可学习信号）
    assert ce[-1] < ce[0]
    assert vmse[-1] < vmse[0]
    assert metrics["steps"] == 8 * max(1, math.ceil(len(buf) / 64))


def test_distill_checkpoint_roundtrip(tmp_path: Path):
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    buf = _make_buffer(n=200)
    cfg = ModelConfig(blocks=3, filters=32, se=True, history_steps=_HISTORY)
    model = create_model(cfg)
    opt_cfg = Config()
    opt_cfg.train.optimizer = "adamw"
    opt_cfg.train.lr = 1e-3
    optimizer = _build_optimizer(model, opt_cfg)

    train_distill(
        model, optimizer, buf, epochs=2, batch_size=64, device="cpu",
        lr_peak=1e-3, lr_min=1e-4, warmup=5, log_fn=lambda *_: None,
    )

    out = tmp_path / "iter_0620.pt"
    save_checkpoint(model, cfg, out, iteration=620, optimizer_state=optimizer.state_dict())
    assert out.exists()

    # 能被 load_checkpoint 按 model_config 重建并前向
    import torch

    loaded, ckpt = load_checkpoint(out, device="cpu")
    mc = ckpt["model_config"]
    for key in ("blocks", "filters", "se", "history_steps"):
        assert key in mc, f"model_config 缺字段 {key}"
    assert mc["blocks"] == 3 and mc["filters"] == 32
    assert mc["se"] is True and mc["history_steps"] == _HISTORY
    assert ckpt["iteration"] == 620

    x = torch.zeros(2, _CHANNELS, 10, 9)
    with torch.no_grad():
        logits, value = loaded(x)
    assert logits.shape == (2, ACTION_SIZE)
    assert value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    assert float(value.abs().max()) <= 1.0 + 1e-6
