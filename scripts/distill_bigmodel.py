# -*- coding: utf-8 -*-
"""大模型迁移：从落盘 replay buffer 的 (state, π, z) 直接监督训练大网。

背景（迁移路线，见 configs/cloud_stage2_15x192se.yaml 头部 runbook）：
阶段二末 20 迭代用高 sims（num_sims_late）把整窗 buffer 刷成高质量 π 目标后，
用本脚本把这批 (state, π, z) 监督训练进更大的网络（15×192+SE），产出的
iter_XXXX.pt 再用大模型 yaml --resume 接续自博弈训练。

**刻意不蒸 best 的 logits**：best 先验近均匀（阶段二诊断的根因），蒸它的 logits
得到的还是均匀分布；直接拟合 buffer 里搜索得到的 π 目标才有信息量。

损失与主训练（rl/train.py `_train_phase`）同构：policy CE = -Σ π·log_softmax(logits)、
value MSE。数据用 ReplayBuffer.load 后多 epoch 随机 sample()（监督蒸馏无需顺序遍历）。
LR 用独立 one-cycle（warmup→lr_peak→余弦退到 lr_min），**不复用** `_train_phase` 的
`_cosine_lr`（后者绑定主训练的 iteration/total_iterations 全局进度，语义不符）。

用法::

    python scripts/distill_bigmodel.py \
        --buffer-dir ckpts/buffer --blocks 15 --filters 192 --se \
        --history-steps 2 --iteration 620 --epochs 8 --batch-size 4096 \
        --device cuda --lr-peak 1e-3 \
        --arena-check ckpts/best.pt --seed-best

产出 ``ckpts/iter_{iteration:04d}.pt``（文件名符合 iter_XXXX.pt 规则，供大模型 yaml
``--resume`` 的 glob 按名找到）。checkpoint 随 model_config 写入——train.py resume 用
yaml 建骨架再 strict load，四字段（blocks/filters/se/history_steps）与 yaml 不一致
会直接崩，故 **本脚本 --blocks/--filters/--se/--history-steps 必须与迁移 yaml 完全一致**。
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# 允许以脚本方式直接运行（把仓库根加入 sys.path；scripts/ 非已安装包）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.config import Config, ModelConfig  # noqa: E402
from rl.model import create_model, save_checkpoint  # noqa: E402
from rl.replay_buffer import ReplayBuffer  # noqa: E402
from rl.train import _build_optimizer  # noqa: E402


# ---------------------------------------------------------------------------
# 独立 one-cycle 学习率（不复用 _train_phase 的 _cosine_lr）
# ---------------------------------------------------------------------------
def onecycle_lr(
    step: int, total_steps: int, lr_peak: float, lr_min: float = 1e-4, warmup: int = 100
) -> float:
    """one-cycle：前 ``warmup`` 步线性升到 ``lr_peak``，其后余弦退火到 ``lr_min``。

    与主训练 ``_cosine_lr`` 无关——那条曲线按主循环全局 iteration/total_iterations
    进度退火；蒸馏是独立一段监督训练，自成 warmup+cosine 周期。
    """
    lr_peak = float(lr_peak)
    lr_min = float(lr_min)
    if step < warmup:
        # 线性 warmup：第 0 步给 lr_peak/warmup，第 warmup-1 步接近 lr_peak。
        return lr_peak * float(step + 1) / float(max(1, warmup))
    denom = max(1, total_steps - warmup)
    progress = min(max((step - warmup) / denom, 0.0), 1.0)
    return lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# bf16 预检（SE 的全局池化 + sigmoid 在 bf16 下从未实测，迁移前置门）
# ---------------------------------------------------------------------------
def bf16_precheck(
    model, in_channels: int, device: str, log_fn: Callable[[str], None] = print, batch: int = 8
) -> float:
    """同一随机 batch 分别 fp32/bf16 前向，断言无 NaN/inf、value∈[-1,1]，返回最大偏差。

    SE 块的 squeeze（全局均值池化）+ excitation（sigmoid）路径在 bf16 autocast 下
    数值行为此前未实测；迁移到 15×192+SE 前先在此确认不炸。
    """
    import torch

    dev = torch.device(device)
    model.eval()
    x = torch.rand(batch, in_channels, 10, 9, device=dev)

    with torch.no_grad():
        logits32, v32 = model(x)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_bf, v_bf = model(x)

    logits_bf_f = logits_bf.float()
    v_bf_f = v_bf.float()

    for name, t in (("logits_bf16", logits_bf_f), ("value_bf16", v_bf_f),
                    ("logits_fp32", logits32), ("value_fp32", v32)):
        if not torch.isfinite(t).all():
            raise RuntimeError(f"bf16 预检失败：{name} 含 NaN/inf")
    vmin, vmax = float(v_bf_f.min()), float(v_bf_f.max())
    if vmin < -1.0001 or vmax > 1.0001:
        raise RuntimeError(f"bf16 预检失败：value 越界 [{vmin:.4f}, {vmax:.4f}]")

    logits_dev = float((logits_bf_f - logits32).abs().max())
    value_dev = float((v_bf_f - v32).abs().max())
    max_dev = max(logits_dev, value_dev)
    log_fn(
        f"[bf16 预检] 通过：value∈[{vmin:.4f},{vmax:.4f}] "
        f"logits 最大偏差={logits_dev:.4f} value 最大偏差={value_dev:.4f}"
    )
    return max_dev


# ---------------------------------------------------------------------------
# 蒸馏训练循环（可 import，供测试用）
# ---------------------------------------------------------------------------
def train_distill(
    model,
    optimizer,
    buffer: ReplayBuffer,
    *,
    epochs: int,
    batch_size: int,
    device: str,
    lr_peak: float,
    lr_min: float = 1e-4,
    warmup: int = 100,
    value_loss_weight: float = 1.0,
    grad_clip: float = 5.0,
    use_amp: bool = False,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, object]:
    """在 ``buffer`` 上做 ``epochs`` 轮监督蒸馏，返回指标 dict。

    steps_per_epoch = ceil(len(buffer) / batch_size)（一轮的等效步数；随机 sample 不
    保证覆盖每个样本，仅用于把总步数与 buffer 规模挂钩）。总步数 = epochs × 该值。
    损失与主训练 `_train_phase` 同构：policy CE + value_loss_weight·value MSE。

    Returns
    -------
    {"samples", "steps", "epoch_policy_ce":[...], "epoch_value_mse":[...],
     "final_policy_ce", "final_value_mse"}
    """
    import torch
    import torch.nn.functional as F

    n_samples = len(buffer)
    if n_samples == 0:
        raise RuntimeError("buffer 为空，无法蒸馏")
    batch = int(batch_size)
    steps_per_epoch = max(1, math.ceil(n_samples / batch))
    total_steps = int(epochs) * steps_per_epoch
    dev = torch.device(device)
    use_amp = bool(use_amp) and dev.type == "cuda"

    log_fn(
        f"[蒸馏] 样本量={n_samples} batch={batch} "
        f"steps/epoch={steps_per_epoch} epochs={epochs} 总步数={total_steps} "
        f"lr_peak={lr_peak:g} lr_min={lr_min:g} warmup={warmup} amp={use_amp}"
    )

    model.train()
    epoch_ce: List[float] = []
    epoch_vmse: List[float] = []
    global_step = 0
    last_pce = last_vmse = float("nan")

    for epoch in range(int(epochs)):
        ce_sum = vmse_sum = 0.0
        cnt = 0
        for _ in range(steps_per_epoch):
            states_np, policy_np, values_np = buffer.sample(batch)
            states = torch.from_numpy(states_np).to(dev)
            policy_target = torch.from_numpy(policy_np).to(dev)
            value_target = torch.from_numpy(values_np).to(dev)

            lr = onecycle_lr(global_step, total_steps, lr_peak, lr_min, warmup)
            for grp in optimizer.param_groups:
                grp["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, value_pred = model(states)
                    log_probs = F.log_softmax(logits, dim=1)
                    policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
                    value_loss = F.mse_loss(value_pred, value_target)
                    loss = policy_loss + value_loss_weight * value_loss
            else:
                logits, value_pred = model(states)
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value_pred, value_target)
                loss = policy_loss + value_loss_weight * value_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            pce = float(policy_loss.item())
            vmse = float(value_loss.item())
            ce_sum += pce
            vmse_sum += vmse
            cnt += 1
            global_step += 1
            last_pce, last_vmse = pce, vmse

        ce_avg = ce_sum / max(1, cnt)
        vmse_avg = vmse_sum / max(1, cnt)
        epoch_ce.append(ce_avg)
        epoch_vmse.append(vmse_avg)
        log_fn(
            f"[蒸馏] epoch {epoch + 1}/{epochs} lr={lr:.3e} "
            f"policy_ce={ce_avg:.5f} value_mse={vmse_avg:.5f}"
        )

    model.eval()
    return {
        "samples": n_samples,
        "steps": total_steps,
        "epoch_policy_ce": epoch_ce,
        "epoch_value_mse": epoch_vmse,
        "final_policy_ce": last_pce,
        "final_value_mse": last_vmse,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _arena_config(device: str, num_gpus: int) -> Config:
    """给 rl.evaluate.arena 用的 Config：arena/train 段用默认值，仅覆盖设备。

    arena() 走 load_checkpoint 按各自 model_config 重建，新旧异构（大网 vs 旧 best）
    因此安全——两边各自读自己的四字段建骨架。
    """
    cfg = Config()
    cfg.train.device = device
    cfg.train.num_gpus = int(num_gpus)
    return cfg


def run(args: argparse.Namespace, log_fn: Callable[[str], None] = print) -> Dict[str, object]:
    """CLI 主逻辑（拆出以便测试/复用）。"""
    import torch

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        log_fn(f"[警告] --device={device} 但 CUDA 不可用，回退 cpu")
        device = "cpu"

    model_cfg = ModelConfig(
        blocks=int(args.blocks),
        filters=int(args.filters),
        se=bool(args.se),
        history_steps=int(args.history_steps),
    )
    in_channels = model_cfg.in_channels
    state_shape = (in_channels, 10, 9)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / f"iter_{int(args.iteration):04d}.pt"

    # ── seed-best 前置校验：无 arena 结果 ≥0.5 时拒绝执行（避免用未验证的大网覆盖 best）──
    if args.seed_best and not args.arena_check:
        raise SystemExit(
            "拒绝 --seed-best：未提供 --arena-check，无法确认新大网 ≥0.5，不覆盖 best.pt。"
        )

    # ── 加载 buffer（channels 必须与模型一致，否则前向才炸）──────────────────
    buf_dir = Path(args.buffer_dir)
    buffer = ReplayBuffer.load(buf_dir, capacity=None, state_shape=state_shape, warn=log_fn)
    if tuple(buffer.state_shape) != state_shape:
        raise SystemExit(
            f"buffer channels 不匹配：buffer.state_shape={buffer.state_shape} "
            f"≠ 模型期望 {state_shape}（history_steps={model_cfg.history_steps}）。"
            f"history_steps 变了会让旧 buffer 静默加载、前向才炸——核对 --history-steps。"
        )
    log_fn(f"[蒸馏] buffer 载入：dir={buf_dir} size={len(buffer)} state_shape={buffer.state_shape}")

    model = create_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(
        f"[蒸馏] 模型 {model_cfg.blocks}b×{model_cfg.filters}f "
        f"se={model_cfg.se} history_steps={model_cfg.history_steps} "
        f"in_channels={in_channels} 参数量={n_params:,}"
    )

    # ── bf16 预检（cuda 且 --se）───────────────────────────────────────────
    if device.startswith("cuda") and model_cfg.se:
        bf16_precheck(model, in_channels, device, log_fn=log_fn)

    # ── 优化器（复用主训练 _build_optimizer：BN/bias 免 weight decay）────────
    opt_cfg = Config()
    opt_cfg.train.optimizer = "adamw"
    opt_cfg.train.lr = float(args.lr_peak)  # 初值；one-cycle 每步覆盖
    optimizer = _build_optimizer(model, opt_cfg)

    use_amp = device.startswith("cuda")
    t0 = time.monotonic()
    metrics = train_distill(
        model,
        optimizer,
        buffer,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device=device,
        lr_peak=float(args.lr_peak),
        lr_min=float(args.lr_min),
        use_amp=use_amp,
        log_fn=log_fn,
    )
    log_fn(
        f"[蒸馏] 完成：steps={metrics['steps']} "
        f"final policy_ce={metrics['final_policy_ce']:.5f} "
        f"value_mse={metrics['final_value_mse']:.5f} "
        f"耗时={time.monotonic() - t0:.1f}s"
    )

    # ── 写 checkpoint（model_config 随 checkpoint 写入，供 yaml resume strict load）──
    save_checkpoint(
        model,
        model_cfg,
        out_path,
        iteration=int(args.iteration),
        optimizer_state=optimizer.state_dict(),
        meta={
            "distill": True,
            "buffer_dir": str(buf_dir),
            "buffer_size": len(buffer),
            "epochs": int(args.epochs),
            "lr_peak": float(args.lr_peak),
        },
    )
    log_fn(f"[蒸馏] 已写出 {out_path}")

    # ── 可选 arena 校验（新大网 vs 旧 best）─────────────────────────────────
    arena_score: Optional[float] = None
    if args.arena_check:
        from rl.evaluate import arena

        acfg = _arena_config(device, args.num_gpus)
        log_fn(f"[蒸馏] arena 校验：{out_path.name} vs {args.arena_check}（{acfg.arena.arena_games} 盘）")
        astats = arena(str(out_path), str(args.arena_check), acfg, device)
        arena_score = float(astats["score"])
        log_fn(
            f"[蒸馏] arena score={arena_score:.4f} "
            f"(W{astats['wins']}/D{astats['draws']}/L{astats['losses']})"
        )

    # ── 可选 seed-best：备份旧 best 后覆盖（要求 arena_score ≥ 0.5）────────────
    if args.seed_best:
        if arena_score is None or arena_score < 0.5:
            raise SystemExit(
                f"拒绝 --seed-best：arena score={arena_score}（需 ≥0.5），不覆盖 best.pt。"
            )
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            backup = ckpt_dir / f"best_pre_distill_iter{int(args.iteration)}.pt"
            shutil.copy2(best_path, backup)
            log_fn(f"[蒸馏] 已备份旧 best → {backup}")
        save_checkpoint(
            model,
            model_cfg,
            best_path,
            iteration=int(args.iteration),
            meta={"seeded_from_distill": True, "arena_score": arena_score},
        )
        log_fn(f"[蒸馏] 已用大网 seed best.pt（arena score={arena_score:.4f}）")

    metrics["out_path"] = str(out_path)
    metrics["n_params"] = n_params
    metrics["arena_score"] = arena_score
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="大模型迁移：从 replay buffer 监督蒸馏大网")
    p.add_argument("--buffer-dir", dest="buffer_dir", default="ckpts/buffer",
                   help="落盘 replay buffer 目录（默认 ckpts/buffer）")
    p.add_argument("--ckpt-dir", dest="ckpt_dir", default="ckpts",
                   help="checkpoint 输出目录（默认 ckpts）")
    p.add_argument("--blocks", type=int, default=15, help="残差块数")
    p.add_argument("--filters", type=int, default=192, help="通道数")
    p.add_argument("--se", action="store_true", help="启用 Squeeze-Excitation")
    p.add_argument("--history-steps", dest="history_steps", type=int, default=2,
                   help="历史步数 T（C_in=14T+3，必须与迁移 yaml 一致）")
    p.add_argument("--iteration", type=int, required=True,
                   help="产出写 ckpts/iter_{iteration:04d}.pt；应为续训起点迭代号")
    p.add_argument("--epochs", type=int, default=8, help="蒸馏轮数")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=4096)
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    p.add_argument("--lr-peak", dest="lr_peak", type=float, default=1e-3,
                   help="one-cycle 峰值 LR")
    p.add_argument("--lr-min", dest="lr_min", type=float, default=1e-4,
                   help="one-cycle 余弦退火下限")
    p.add_argument("--num-gpus", dest="num_gpus", type=int, default=8,
                   help="arena 校验用的 GPU 数")
    p.add_argument("--arena-check", dest="arena_check", default=None,
                   help="旧 best 路径；给出则跑 arena 校验新大网 vs 旧 best")
    p.add_argument("--seed-best", dest="seed_best", action="store_true",
                   help="备份旧 best→best_pre_distill_iter{N}.pt 后用大网覆盖；"
                        "无 arena 结果 ≥0.5 时拒绝执行")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
