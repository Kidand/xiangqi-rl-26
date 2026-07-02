# -*- coding: utf-8 -*-
"""主训练循环（同步迭代制）——DESIGN.md §9-§11。

入口::

    python -m rl.train --config configs/cloud_8xh100.yaml [--resume]

每迭代：
  1. Self-play：spawn workers（best 模型执双方），主进程边收 queue 边按 log_interval_sec
     实时打印红黑胜率进度；收完 join，样本入 buffer，逐局 record append 到
     ``records/selfplay/iter_XXXX.jsonl``。
  2. Train：buffer ≥ min_buffer_to_train 后训练 train_steps（AdamW/SGD、余弦 lr、
     bf16 amp（cpu 自动关）、grad_clip），每 train_log_interval 步打印。
  3. Arena：新模型 vs best 门控，晋升则更新 ``checkpoints/best.pt``。
  4. 始终存 ``iter_XXXX.pt``、buffer、metrics.csv 汇总行。

--resume：找最新 iter_XXXX.pt，载模型/optimizer/buffer 续训。
训练设备 cuda:0（单卡训练 + 多卡自博弈，不做 DDP）；device=cpu 全程可跑。
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from queue import Empty
from typing import Dict, Optional, Tuple

import numpy as np

from xiangqi.records import append_jsonl
from rl.config import Config
from rl.evaluate import arena
from rl.logger import TrainLogger
from rl.model import create_model, save_checkpoint
from rl.replay_buffer import ReplayBuffer
from rl.selfplay import num_selfplay_workers, selfplay_worker

__all__ = ["run_training", "main"]


# ---------------------------------------------------------------------------
# 设备 / 随机种子
# ---------------------------------------------------------------------------
def _train_device(cfg: Config) -> str:
    """训练设备：cuda 时固定 cuda:0，否则 cpu。"""
    import torch

    if str(cfg.train.device).startswith("cuda") and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _set_global_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed & 0x7FFFFFFF)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed & 0x7FFFFFFF)


# ---------------------------------------------------------------------------
# 优化器 / 学习率
# ---------------------------------------------------------------------------
def _build_optimizer(model, cfg: Config):
    import torch

    params = model.parameters()
    if str(cfg.train.optimizer).lower() == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
            nesterov=True,
        )
    return torch.optim.AdamW(
        params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )


def _cosine_lr(cfg: Config, global_step: int, total_steps: int) -> float:
    """余弦退火：从 lr 退火到 lr_min（按全程训练步进度）。"""
    lr_max = float(cfg.train.lr)
    lr_min = float(cfg.train.lr_min)
    if total_steps <= 0:
        return lr_max
    progress = min(max(global_step / total_steps, 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# checkpoint / buffer 路径
# ---------------------------------------------------------------------------
def _iter_ckpt_path(ckpt_dir: Path, iteration: int) -> Path:
    return ckpt_dir / f"iter_{iteration:04d}.pt"


def _best_ckpt_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "best.pt"


def _buffer_dir(ckpt_dir: Path) -> Path:
    return ckpt_dir / "buffer"


def _find_latest_iter(ckpt_dir: Path) -> Optional[int]:
    """返回最新 iter_XXXX.pt 的迭代号，无则 None。"""
    best = None
    for p in ckpt_dir.glob("iter_*.pt"):
        stem = p.stem  # iter_0007
        try:
            n = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if best is None or n > best:
            best = n
    return best


# ---------------------------------------------------------------------------
# Self-play 阶段（spawn workers + queue 聚合 + 实时进度）
# ---------------------------------------------------------------------------
def _selfplay_phase(
    cfg: Config,
    iteration: int,
    best_path: Path,
    logger: TrainLogger,
    buffer: ReplayBuffer,
    records_path: Path,
) -> Dict[str, float]:
    """并行自博弈，收集样本入 buffer、record 写盘，返回本迭代统计。"""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    stop_event = ctx.Event()
    n_workers = num_selfplay_workers(cfg)
    total_games = int(cfg.selfplay.games_per_iteration)

    procs = []
    for pid in range(n_workers):
        wseed = (int(cfg.seed) * 1_000_003 + iteration * 7_919 + pid) & (2**63 - 1)
        p = ctx.Process(
            target=selfplay_worker,
            args=(pid, cfg.train.device, str(best_path), cfg, iteration, queue, stop_event, wseed),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # 聚合统计。
    agg = {
        "games": 0,
        "red_win": 0,
        "black_win": 0,
        "draw": 0,
        "plies_sum": 0,
        "moves": 0,
        "nn_evals": 0,
        "resigned": 0,
        "calib_games": 0,
        "calib_would": 0,
        "fp": 0,
    }

    t0 = time.monotonic()
    last_log = t0
    done_workers = 0

    def _emit_progress() -> None:
        elapsed = max(1e-6, time.monotonic() - t0)
        g = agg["games"]
        logger.selfplay_progress(
            {
                "iter": iteration,
                "games": g,
                "total_games": total_games,
                "red_win": agg["red_win"],
                "black_win": agg["black_win"],
                "draw": agg["draw"],
                "avg_plies": (agg["plies_sum"] / g) if g > 0 else 0.0,
                "moves_per_sec": agg["moves"] / elapsed,
                "nn_evals_per_sec": agg["nn_evals"] / elapsed,
                "resign_rate": (agg["resigned"] / g) if g > 0 else 0.0,
                "buffer_size": len(buffer),
                "elapsed_sec": elapsed,
            }
        )

    while done_workers < n_workers:
        try:
            msg = queue.get(timeout=1.0)
        except Empty:
            # 所有进程已退出且队列空 → 收尾（防御死锁）。
            if all(not p.is_alive() for p in procs) and queue.empty():
                break
            continue

        if msg.get("type") == "done":
            done_workers += 1
            continue

        # 一盘对局：样本入 buffer、record 写盘、更新统计。
        states = msg["states"]
        sparse = list(zip(msg["pol_idx"], msg["pol_prob"]))
        values = msg["values"]
        if len(states) > 0:
            buffer.add_game(states, sparse, values)
        append_jsonl(records_path, msg["record"])

        st = msg["stats"]
        agg["games"] += 1
        winner = st["winner"]
        if winner == "red":
            agg["red_win"] += 1
        elif winner == "black":
            agg["black_win"] += 1
        else:
            agg["draw"] += 1
        agg["plies_sum"] += int(st["plies"])
        agg["moves"] += int(st["plies"])
        agg["nn_evals"] += int(st["nn_evals"])
        if st.get("resigned"):
            agg["resigned"] += 1
        if st.get("is_calibration"):
            agg["calib_games"] += 1
            if st.get("would_resign"):
                agg["calib_would"] += 1
                if st.get("fp"):
                    agg["fp"] += 1

        now = time.monotonic()
        if now - last_log >= float(cfg.log.log_interval_sec):
            _emit_progress()
            last_log = now

    _emit_progress()  # 收尾行
    for p in procs:
        p.join(timeout=30.0)
        if p.is_alive():
            p.terminate()

    g = max(1, agg["games"])
    fp_rate = (agg["fp"] / agg["calib_would"]) if agg["calib_would"] > 0 else 0.0
    return {
        "games": agg["games"],
        "red_win": agg["red_win"],
        "black_win": agg["black_win"],
        "draw": agg["draw"],
        "red_winrate": agg["red_win"] / g,
        "black_winrate": agg["black_win"] / g,
        "draw_rate": agg["draw"] / g,
        "avg_plies": agg["plies_sum"] / g,
        "resign_rate": agg["resigned"] / g,
        "resign_fp_rate": fp_rate,
        "elapsed": time.monotonic() - t0,
    }


# ---------------------------------------------------------------------------
# 训练阶段
# ---------------------------------------------------------------------------
def _train_phase(
    model,
    optimizer,
    buffer: ReplayBuffer,
    cfg: Config,
    iteration: int,
    device: str,
    logger: TrainLogger,
) -> Dict[str, float]:
    """在 buffer 上训练 train_steps_per_iteration 步，返回平均指标。"""
    import torch
    import torch.nn.functional as F

    steps = int(cfg.train.train_steps_per_iteration)
    batch = int(cfg.train.batch_size)
    total_steps = int(cfg.train.total_iterations) * steps
    dev = torch.device(device)
    use_amp = bool(cfg.train.amp) and dev.type == "cuda"

    model.train()

    sums = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "value_mae": 0.0,
        "grad_norm": 0.0,
    }
    last_lr = float(cfg.train.lr)
    n = 0

    for step in range(steps):
        states_np, policy_np, values_np = buffer.sample(batch)
        states = torch.from_numpy(states_np).to(dev)
        policy_target = torch.from_numpy(policy_np).to(dev)
        value_target = torch.from_numpy(values_np).to(dev)

        global_step = iteration * steps + step
        lr = _cosine_lr(cfg, global_step, total_steps)
        for grp in optimizer.param_groups:
            grp["lr"] = lr
        last_lr = lr

        optimizer.zero_grad(set_to_none=True)

        t_start = time.monotonic()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, value_pred = model(states)
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value_pred, value_target)
                loss = policy_loss + cfg.train.value_loss_weight * value_loss
        else:
            logits, value_pred = model(states)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred, value_target)
            loss = policy_loss + cfg.train.value_loss_weight * value_loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg.train.grad_clip)
        )
        optimizer.step()

        # 指标（不参与反传）。
        with torch.no_grad():
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=1).mean()
            value_mae = (value_pred - value_target).abs().mean()

        elapsed = max(1e-6, time.monotonic() - t_start)
        samples_per_sec = batch / elapsed

        l = float(loss.item())
        pl = float(policy_loss.item())
        vl = float(value_loss.item())
        ent = float(entropy.item())
        vmae = float(value_mae.item())
        gn = float(grad_norm)

        sums["loss"] += l
        sums["policy_loss"] += pl
        sums["value_loss"] += vl
        sums["entropy"] += ent
        sums["value_mae"] += vmae
        sums["grad_norm"] += gn
        n += 1

        if (step + 1) % int(cfg.log.train_log_interval) == 0 or step == steps - 1:
            logger.train_step(
                step=step + 1,
                loss=l,
                policy_loss=pl,
                value_loss=vl,
                entropy=ent,
                value_mae=vmae,
                lr=lr,
                samples_per_sec=samples_per_sec,
                grad_norm=gn,
                iteration=iteration,
            )

    model.eval()
    n = max(1, n)
    return {
        "loss": sums["loss"] / n,
        "policy_loss": sums["policy_loss"] / n,
        "value_loss": sums["value_loss"] / n,
        "entropy": sums["entropy"] / n,
        "value_mae": sums["value_mae"] / n,
        "grad_norm": sums["grad_norm"] / n,
        "lr": last_lr,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_training(cfg: Config, resume: bool = False) -> Dict[str, object]:
    """执行完整训练循环。返回摘要 dict（供 smoke_test 校验）。"""
    import torch

    _set_global_seed(int(cfg.seed))
    device = _train_device(cfg)

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    records_dir = Path(cfg.log.records_dir) / "selfplay"
    records_dir.mkdir(parents=True, exist_ok=True)
    buf_dir = _buffer_dir(ckpt_dir)

    logger = TrainLogger(log_dir=cfg.log.log_dir, tensorboard=bool(cfg.log.tensorboard))

    C = cfg.model.in_channels
    state_shape = (C, 10, 9)

    model = create_model(cfg.model).to(device)
    optimizer = _build_optimizer(model, cfg)

    best_path = _best_ckpt_path(ckpt_dir)
    start_iter = 0

    def _save_iter(iteration: int) -> Path:
        path = _iter_ckpt_path(ckpt_dir, iteration)
        save_checkpoint(
            model,
            cfg.model,
            path,
            iteration=iteration,
            optimizer_state=optimizer.state_dict(),
            meta={"seed": cfg.seed, "device": device},
        )
        return path

    if resume:
        latest = _find_latest_iter(ckpt_dir)
        if latest is not None:
            ckpt = torch.load(
                str(_iter_ckpt_path(ckpt_dir, latest)),
                map_location=torch.device(device),
                weights_only=False,
            )
            model.load_state_dict(ckpt["model_state"])
            if ckpt.get("optimizer_state") is not None:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:  # 优化器结构变更时容错
                    logger.warn(f"optimizer 状态加载失败，重置: {e}")
            start_iter = latest + 1
            logger.info(f"--resume：从 iter_{latest:04d}.pt 续训，start_iter={start_iter}")
        else:
            logger.warn("--resume 但未找到 iter_XXXX.pt，从头开始")
        if buf_dir.exists() and (buf_dir / "meta.json").exists():
            try:
                buffer = ReplayBuffer.load(buf_dir)
                logger.info(f"buffer 续载，size={len(buffer)}")
            except Exception as e:
                logger.warn(f"buffer 加载失败，新建: {e}")
                buffer = ReplayBuffer(int(cfg.train.buffer_window), state_shape)
        else:
            buffer = ReplayBuffer(int(cfg.train.buffer_window), state_shape)
        if not best_path.exists():
            save_checkpoint(model, cfg.model, best_path, iteration=start_iter, meta={"init": True})
    else:
        buffer = ReplayBuffer(int(cfg.train.buffer_window), state_shape)
        # 初始 best：随机模型（自博弈冷启动用）。
        save_checkpoint(model, cfg.model, best_path, iteration=-1, meta={"init": True})
        logger.info(f"初始化完成，device={device}，模型 {cfg.model.blocks}b×{cfg.model.filters}f")

    total_iterations = int(cfg.train.total_iterations)
    last_summary: Dict[str, object] = {}

    for iteration in range(start_iter, total_iterations):
        iter_t0 = time.monotonic()
        records_path = records_dir / f"iter_{iteration:04d}.jsonl"

        # 1) Self-play（best 模型执双方）。
        sp_t0 = time.monotonic()
        sp_stats = _selfplay_phase(cfg, iteration, best_path, logger, buffer, records_path)
        selfplay_sec = time.monotonic() - sp_t0

        # 2) Train（buffer 达阈值才训练）。
        tr_t0 = time.monotonic()
        trained = len(buffer) >= int(cfg.train.min_buffer_to_train)
        if trained:
            tr_stats = _train_phase(model, optimizer, buffer, cfg, iteration, device, logger)
        else:
            logger.info(
                f"[iter {iteration}] buffer {len(buffer)} < min_buffer_to_train "
                f"{cfg.train.min_buffer_to_train}，跳过训练"
            )
            tr_stats = {}
        train_sec = time.monotonic() - tr_t0

        # 3) 存本迭代 checkpoint（含 optimizer），供 arena 与续训使用。
        iter_path = _save_iter(iteration)

        # 4) Arena 门控（新 iter vs best）。
        arena_stats = arena(str(iter_path), str(best_path), cfg, device)
        promoted = bool(trained and arena_stats["score"] >= float(cfg.arena.gate_threshold))
        if promoted:
            save_checkpoint(
                model, cfg.model, best_path, iteration=iteration, meta={"promoted_from": iteration}
            )

        logger.arena_result(
            wins=arena_stats["wins"],
            draws=arena_stats["draws"],
            losses=arena_stats["losses"],
            red_wins=arena_stats["red_wins"],
            red_draws=arena_stats["red_draws"],
            red_losses=arena_stats["red_losses"],
            black_wins=arena_stats["black_wins"],
            black_draws=arena_stats["black_draws"],
            black_losses=arena_stats["black_losses"],
            score=arena_stats["score"],
            threshold=float(cfg.arena.gate_threshold),
            promoted=promoted,
            iteration=iteration,
        )

        # 5) buffer 持久化 + CSV 汇总行。
        try:
            buffer.save(buf_dir)
        except Exception as e:
            logger.warn(f"buffer 保存失败: {e}")

        total_sec = time.monotonic() - iter_t0
        row = {
            "iter": iteration,
            "games": sp_stats["games"],
            "red_win": sp_stats["red_win"],
            "black_win": sp_stats["black_win"],
            "draw": sp_stats["draw"],
            "red_winrate": round(sp_stats["red_winrate"], 4),
            "black_winrate": round(sp_stats["black_winrate"], 4),
            "draw_rate": round(sp_stats["draw_rate"], 4),
            "avg_plies": round(sp_stats["avg_plies"], 2),
            "resign_rate": round(sp_stats["resign_rate"], 4),
            "resign_fp_rate": round(sp_stats["resign_fp_rate"], 4),
            "buffer_size": len(buffer),
            "loss": round(tr_stats["loss"], 5) if trained else "",
            "policy_loss": round(tr_stats["policy_loss"], 5) if trained else "",
            "value_loss": round(tr_stats["value_loss"], 5) if trained else "",
            "entropy": round(tr_stats["entropy"], 5) if trained else "",
            "value_mae": round(tr_stats["value_mae"], 5) if trained else "",
            "lr": (f"{tr_stats['lr']:.3e}" if trained else ""),
            "arena_score": round(arena_stats["score"], 4),
            "arena_red_score": round(arena_stats["red_score"], 4),
            "arena_black_score": round(arena_stats["black_score"], 4),
            "promoted": promoted,
            "selfplay_sec": round(selfplay_sec, 2),
            "train_sec": round(train_sec, 2),
            "total_sec": round(total_sec, 2),
        }
        logger.iteration_summary(row)
        last_summary = row

    logger.info("训练循环结束")
    logger.close()
    return {"iterations": total_iterations, "last_row": last_summary, "checkpoint_dir": str(ckpt_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Xiangqi AlphaZero 训练主循环")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--resume", action="store_true", help="从最新 checkpoint 续训")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    run_training(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
