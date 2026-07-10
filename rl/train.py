# -*- coding: utf-8 -*-
"""主训练循环（同步迭代制）——DESIGN.md §9-§11。

入口::

    python -m rl.train --config configs/cloud.yaml [--resume]

每迭代：
  1. Self-play（推理服务架构，DESIGN §9）：主进程创建共享内存与队列 → spawn
     ``num_servers`` 个 EvalServer（cuda 每 GPU 一个 / cpu 一个，持有 best 模型）
     + ``num_workers`` 个纯 CPU selfplay worker，主进程边收 game_queue 边按
     log_interval_sec 实时打印红黑胜率进度（含 evals/s 与 avg_batch）；全部 done 后
     置 stop_event、join 全部进程，样本入 buffer，逐局 record append 到
     ``records/selfplay/iter_XXXX.jsonl``。
  2. Train：buffer ≥ min_buffer_to_train 后训练 train_steps（AdamW/SGD、余弦 lr、
     bf16 amp（cpu 自动关）、grad_clip），每 train_log_interval 步打印。
  3. Arena：新模型 vs best 门控，晋升则更新 best.pt（``cfg.train.checkpoint_dir``）。
  4. 始终存 ``iter_XXXX.pt``、buffer、metrics.csv 汇总行。

--resume：找最新 iter_XXXX.pt，载模型/optimizer/buffer 续训。
训练设备 cuda:0（单卡训练 + 多卡自博弈，不做 DDP）；device=cpu 全程可跑。
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from xiangqi.records import append_jsonl
from rl.config import Config
from rl.evaluate import arena, passes_gate
from rl.logger import TrainLogger
from rl.model import create_model, save_checkpoint
from rl.replay_buffer import ReplayBuffer
from rl.inference_server import create_shared_buffers, eval_server_proc
from rl.selfplay import (
    cpu_budget_summary,
    num_eval_servers,
    num_selfplay_workers,
    selfplay_worker,
)

__all__ = [
    "run_training",
    "main",
    "resolve_train_steps",
    "load_latest_checkpoint",
]

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 训练步数解析（自动 / 固定）
# ---------------------------------------------------------------------------
def resolve_train_steps(new_samples: int, cfg: Config) -> int:
    """本迭代训练步数（纯函数，DESIGN §9.3）。

    - ``train.train_steps_per_iteration > 0``：固定步数，原样返回。
    - ``== 0``：自动 ``clamp(ceil(新样本数 × sample_reuse / batch_size), 10, 5000)``，
      防止小 buffer 期每个样本被重复学习过多遍导致过拟合。
    """
    fixed = int(cfg.train.train_steps_per_iteration)
    if fixed > 0:
        return fixed
    batch = max(1, int(cfg.train.batch_size))
    steps = math.ceil(int(new_samples) * float(cfg.train.sample_reuse) / batch)
    return max(10, min(5000, int(steps)))


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
def _param_groups(model, weight_decay: float):
    """把参数拆成 decay / no-decay 两组：BN 与所有 bias（1D 张量）不做 weight decay。

    对 L2 正则而言，对 BatchNorm 的 scale/shift 与卷积/全连接的 bias 施加 weight decay
    既无必要也有害（会把归一化统计往 0 拉）。仅对 2D+ 权重张量做 weight decay。
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1:  # BN weight/bias 与各层 bias
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build_optimizer(model, cfg: Config):
    import torch

    groups = _param_groups(model, cfg.train.weight_decay)
    if str(cfg.train.optimizer).lower() == "sgd":
        return torch.optim.SGD(
            groups,
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            nesterov=True,
        )
    return torch.optim.AdamW(groups, lr=cfg.train.lr)


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


def load_latest_checkpoint(
    ckpt_dir: "str | Path",
    device: str = "cpu",
    warn: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """按迭代号从新到旧逐个尝试加载 iter_XXXX.pt，返回首个可用的 (ckpt_dict, path)。

    容错：某个检查点 torch.load 失败（如迭代末保存时被 Ctrl-C 截断），打 WARN
    「iter_XXXX.pt 损坏，回退上一个」后继续尝试更旧的一个。

    返回
    ----
    - 无任何 iter_*.pt：``(None, None)``（交由调用方决定从头训练）。
    - 至少一个可加载：``(ckpt_dict, path)``，其中 path 为该检查点文件路径。
    - 有文件但全部损坏：抛 RuntimeError。
    """
    import torch

    if warn is None:
        warn = _LOG.warning
    ckpt_dir = Path(ckpt_dir)

    # 收集 (iter_num, path)，按迭代号降序。
    cands: list[Tuple[int, Path]] = []
    for p in ckpt_dir.glob("iter_*.pt"):
        try:
            n = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        cands.append((n, p))
    cands.sort(key=lambda x: x[0], reverse=True)

    if not cands:
        return None, None

    for n, p in cands:
        try:
            ckpt = torch.load(
                str(p), map_location=torch.device(device), weights_only=False
            )
            return ckpt, p
        except Exception as e:  # 截断/损坏 → 回退更旧的一个
            warn(f"iter_{n:04d}.pt 损坏，回退上一个: {e}")

    raise RuntimeError(
        f"所有 iter_*.pt 检查点均损坏（共 {len(cands)} 个），无法续训：{ckpt_dir}"
    )


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
    """并行自博弈（推理服务架构），收集样本入 buffer、record 写盘，返回本迭代统计。

    拓扑：num_servers 个 EvalServer（持有 best 模型）+ num_workers 个纯 CPU worker，
    共享内存传叶子/结果，队列传元数据。任一 server 死亡 → 置 stop_event、报错退出迭代。
    """
    import torch
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")

    n_workers = num_selfplay_workers(cfg)
    n_servers = num_eval_servers(cfg)
    total_games = int(cfg.selfplay.games_per_iteration)
    use_cuda = str(cfg.train.device).startswith("cuda") and torch.cuda.is_available()
    # 核数预算一览：容器限额（cgroup 配额/亲和性）会压缩有效核数，一眼看穿环境。
    logger.info(
        f"[iter {iteration}] selfplay: workers={n_workers} servers={n_servers} "
        f"({cpu_budget_summary()})"
    )

    # 共享内存：每 worker 一套（MAX_LEAVES = games_per_worker × mcts.batch_size）。
    max_leaves = int(cfg.selfplay.games_per_worker) * max(1, int(cfg.mcts.batch_size))
    shm = create_shared_buffers(n_workers, max_leaves, int(cfg.model.in_channels))

    # 统计共享张量（每 server 一行）：[n_evals, n_batches, sum_batch_leaves]。
    stats_shm = torch.zeros((n_servers, 3), dtype=torch.float64)
    stats_shm.share_memory_()

    # 队列：每 server 一个请求队列；每 worker 一个响应队列；一个全局 game_queue。
    req_queues = [ctx.Queue() for _ in range(n_servers)]
    resp_queues = [ctx.Queue() for _ in range(n_workers)]
    game_queue = ctx.Queue()
    stop_event = ctx.Event()

    # worker → server round-robin 绑定。
    worker_server = [w % n_servers for w in range(n_workers)]

    # spawn EvalServer 进程。
    servers = []
    for sid in range(n_servers):
        my_workers = [w for w in range(n_workers) if worker_server[w] == sid]
        shm_dict = {w: shm[w] for w in my_workers}
        dev = f"cuda:{sid}" if use_cuda else "cpu"
        p = ctx.Process(
            target=eval_server_proc,
            args=(sid, dev, str(best_path), cfg, shm_dict, req_queues[sid],
                  resp_queues, stop_event, stats_shm),
            daemon=False,
        )
        p.start()
        servers.append(p)

    # spawn 纯 CPU selfplay worker 进程。
    workers = []
    for w in range(n_workers):
        sid = worker_server[w]
        wseed = (int(cfg.seed) * 1_000_003 + iteration * 7_919 + w) & (2**63 - 1)
        p = ctx.Process(
            target=selfplay_worker,
            args=(w, cfg, iteration, shm[w], req_queues[sid], resp_queues[w],
                  game_queue, stop_event, wseed, n_workers),
            daemon=False,
        )
        p.start()
        workers.append(p)

    # 聚合统计。
    agg = {
        "games": 0,
        "new_samples": 0,
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
    server_died = False

    def _server_stats() -> Tuple[float, float]:
        """从 stats_shm 汇总 evals/s 与 avg_batch。"""
        s = stats_shm.sum(dim=0)
        n_evals = float(s[0].item())
        n_batches = float(s[1].item())
        sum_leaves = float(s[2].item())
        elapsed = max(1e-6, time.monotonic() - t0)
        evals_ps = n_evals / elapsed
        avg_batch = (sum_leaves / n_batches) if n_batches > 0 else 0.0
        return evals_ps, avg_batch

    def _emit_progress() -> None:
        elapsed = max(1e-6, time.monotonic() - t0)
        g = agg["games"]
        evals_ps, avg_batch = _server_stats()
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
                "evals_per_sec": evals_ps,
                "avg_batch": avg_batch,
                "resign_rate": (agg["resigned"] / g) if g > 0 else 0.0,
                "buffer_size": len(buffer),
                "elapsed_sec": elapsed,
            }
        )

    while done_workers < n_workers:
        # server 存活监控：任一 server 死亡 → 置 stop_event、报错退出迭代。
        dead = [i for i, p in enumerate(servers) if not p.is_alive()]
        if dead and not server_died:
            server_died = True
            stop_event.set()
            logger.warn(f"[iter {iteration}] EvalServer {dead} 进程异常退出，终止本迭代自博弈")
            break

        try:
            msg = game_queue.get(timeout=1.0)
        except Empty:
            # 所有 worker 已退出且队列空 → 收尾（防御死锁）。
            if all(not p.is_alive() for p in workers) and game_queue.empty():
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
            agg["new_samples"] += len(states)
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

    # 全部 done（或 server 死亡）后置 stop_event、join 全部进程。
    stop_event.set()
    for p in workers:
        p.join(timeout=30.0)
        if p.is_alive():
            p.terminate()
            # terminate 只发 SIGTERM，必须再 join 才会被内核 reap——否则该进程在训练
            # 主进程存活期间永久滞留为 zombie（曾实测累积上百个，逼近 pod pids.max
            # 会让后续 fork 全部失败）。仍不退（极端卡死）则 SIGKILL 兜底。
            p.join(timeout=5.0)
            if p.is_alive():
                p.kill()
                p.join()
    for p in servers:
        p.join(timeout=30.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
            if p.is_alive():
                p.kill()
                p.join()

    if server_died:
        raise RuntimeError(f"iter {iteration}: EvalServer 进程死亡，自博弈中断")

    g = max(1, agg["games"])
    fp_rate = (agg["fp"] / agg["calib_would"]) if agg["calib_would"] > 0 else 0.0
    return {
        "games": agg["games"],
        "new_samples": agg["new_samples"],
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
    steps: int,
) -> Dict[str, float]:
    """在 buffer 上训练 ``steps`` 步（由 resolve_train_steps 解析），返回平均指标。"""
    import torch
    import torch.nn.functional as F

    steps = int(steps)
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

    # GPU 传输优化：batch 形状固定，cuda 设备下预分配 pinned CPU 张量，每步 copy_ 后
    # 以 non_blocking 异步拷到 GPU（overlap H2D 与计算）；cpu 设备走原路径，数值不变。
    use_cuda = dev.type == "cuda"
    pin_states = pin_policy = pin_values = None

    for step in range(steps):
        states_np, policy_np, values_np = buffer.sample(batch)
        if use_cuda:
            if pin_states is None:
                pin_states = torch.empty(states_np.shape, dtype=torch.float32, pin_memory=True)
                pin_policy = torch.empty(policy_np.shape, dtype=torch.float32, pin_memory=True)
                pin_values = torch.empty(values_np.shape, dtype=torch.float32, pin_memory=True)
            pin_states.copy_(torch.from_numpy(states_np))
            pin_policy.copy_(torch.from_numpy(policy_np))
            pin_values.copy_(torch.from_numpy(values_np))
            states = pin_states.to(dev, non_blocking=True)
            policy_target = pin_policy.to(dev, non_blocking=True)
            value_target = pin_values.to(dev, non_blocking=True)
        else:
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
# buffer 后台保存编排器（快照 + 线程，消除每迭代 buffer.save 的整机空转）
# ---------------------------------------------------------------------------
class _BufferSaver:
    """把 ``buffer.save`` 从主循环挪到后台线程，与下一迭代 selfplay/train/arena 重叠。

    背景：configs/cloud 下 ``buffer.save`` 压缩 ~1.5M 样本 npz 约 100-150s，同步
    执行期间整机空转（selfplay/train/arena 都已结束、下一迭代未开始）。本类把压缩落盘
    放进后台线程与下一迭代真正并行——``np.savez_compressed`` 的 zlib 在 C 层释放 GIL、
    训练主循环是 GPU-bound（torch 算子亦释放 GIL），故是真并行而非 GIL 假并行。

    模式（``cfg.train.async_buffer_save``，缺省 "snapshot"）：
      - "snapshot"（默认，推荐）：主线程取一致性快照（``ReplayBuffer.snapshot``：
        states/values 实体拷贝、稀疏 π 两 list 浅拷贝、冻结 size/pos），后台线程压缩快照。
        save 可跨入下一迭代 selfplay 完全隐藏，仅暴露 ~2-5s 的实体拷贝；RAM 峰值约
        +4.2GB（满 buffer states 一份）。多 GPU 服务器 RAM 通常 TB 级，可忽略。
      - "freeze_window"：不拷贝，直接对「只读窗口」内的活 buffer 起线程（该窗口从
        selfplay 结束到下一迭代 selfplay 首个 add_game 之间 buffer 只读），在下一迭代
        selfplay 改写 buffer 前无条件 join。零额外 RAM，但只把 save 藏进 train+arena，
        残余仍阻塞下一 selfplay。RAM 紧张时的降级选择。
      - "sync"：同步保存（原行为，零重叠、零额外 RAM）。

    **单写者不变式**（ed69f24 的 tmp/.old 原子换名要求）：任一时刻至多一个线程触碰
    ``buf_dir``——两个并发 save 的 rename/.old 换名会互相破坏。由「发起下一次保存前先
    join 上一次」保证；主线程自身不再直接调 ``buffer.save``（sync 模式本就串行）。

    **退出**：所有退出路径（正常跑完 / 异常 / Ctrl-C）都必须 ``close()`` 以 join 未完成
    的后台保存——否则进程退出可能停在 rename 中途（load 虽能回退 .old，但不必冒险），或
    丢最后一迭代 buffer。故绝不用 daemon=True。save() 内部原子 tmp+rename+.old
    （ed69f24）保证即便 join 前被 kill，磁盘上 <buffer> 或 <buffer>.old 至少一份完整。

    **崩溃版本差窗口（有意的权衡）**：iter N 的 buffer 落盘与 iter N+1 全程并行，若在
    N+1 期间被 kill 且后台写尚未换名完成，磁盘 buffer 可能仍是 iter N-1 的——buffer 与
    checkpoint 的版本差从同步时代的 ~秒级扩大到最多一整迭代。--resume 对 buffer 落后/
    超前一代均容忍（load 回退链完好，样本是自再生软状态），非正确性问题，勿"修复"。
    """

    _MODES = ("snapshot", "freeze_window", "sync")

    def __init__(
        self,
        buf_dir: "str | Path",
        mode: str = "snapshot",
        warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._buf_dir = buf_dir
        self._mode = str(mode).lower()
        if self._mode not in self._MODES:
            raise ValueError(
                f"async_buffer_save={mode!r} 不合法，可选 {'/'.join(self._MODES)}"
            )
        self._warn = warn if warn is not None else _LOG.warning
        self._thread: Optional[threading.Thread] = None

    def _bg_save(self, payload: ReplayBuffer) -> None:
        """线程体：压缩落盘；失败必须在线程内 catch 并走 warn（子线程未捕获异常不会
        传播到主线程，静默丢失或崩溃线程都不可接受），保持 ed69f24 的
        「保存失败 logger.warn 不中断训练」语义。"""
        try:
            payload.save(self._buf_dir)
        except Exception as e:
            self._warn(f"buffer 保存失败: {e}")

    def before_selfplay(self) -> None:
        """下一迭代 selfplay 前调用。"freeze_window" 模式必须在此 join——后台线程持有的
        是活 buffer 的引用，而 selfplay 即将 add_game 原地改写它。snapshot/sync 模式无活
        buffer 依赖，此处不 join（否则会白白丢掉与 selfplay 的重叠）。"""
        if self._mode == "freeze_window":
            self.join()

    def save(self, buffer: ReplayBuffer) -> None:
        """发起本迭代保存。先 join 上一次（保证 buf_dir 单写者，并释放上一份快照 RAM）。"""
        # 单写者：join 上一次未完成的保存（snapshot 模式此处顺带回收上一份快照）。
        self.join()
        if self._mode == "sync":
            self._bg_save(buffer)
            return
        if self._mode == "freeze_window":
            # 直接对只读活 buffer 起线程；before_selfplay 会在下次改写前 join。
            payload: ReplayBuffer = buffer
        else:  # "snapshot"（默认）
            # 主线程取一致性快照（~2-5s，~4.2GB 实体拷贝）；此后仅后台线程引用它，
            # 内联传参不留主线程本地引用，join 时随 _thread 置 None 一并回收，RAM 峰值
            # 仅 +1 份快照 states。
            payload = buffer.snapshot()
        self._thread = threading.Thread(
            target=self._bg_save, args=(payload,), daemon=False
        )
        self._thread.start()

    def join(self) -> None:
        """join 未完成的后台保存并丢弃其引用（Thread 持有 payload，置 None 才能回收快照）。"""
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def close(self) -> None:
        """退出时调用（正常/异常/Ctrl-C），join 最后一份未落盘的保存。"""
        self.join()


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
        # 从新到旧逐个尝试，跳过被 Ctrl-C 截断/损坏的检查点（全坏才报错）。
        ckpt, ckpt_path = load_latest_checkpoint(ckpt_dir, device, warn=logger.warn)
        if ckpt is not None and ckpt_path is not None:
            latest = int(ckpt_path.stem.split("_")[1])
            model.load_state_dict(ckpt["model_state"])
            if ckpt.get("optimizer_state") is not None:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:  # 优化器结构变更时容错
                    logger.warn(f"optimizer 状态加载失败，重置: {e}")
            start_iter = latest + 1
            logger.info(f"--resume：从 {ckpt_path.name} 续训，start_iter={start_iter}")
        else:
            logger.warn("--resume 但未找到 iter_XXXX.pt，从头开始")
        # buffer 容错加载：主目录损坏时回退 <buffer>.old，都不可用则新建空 buffer。
        if buf_dir.exists() or (buf_dir.with_name(buf_dir.name + ".old")).exists():
            buffer = ReplayBuffer.load(
                buf_dir,
                capacity=int(cfg.train.buffer_window),
                state_shape=state_shape,
                warn=logger.warn,
            )
            logger.info(f"buffer 续载，size={len(buffer)}")
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

    # buffer 后台保存编排器：把 ~100-150s 的 npz 压缩落盘挪出主循环、与下一迭代重叠。
    # 缺省 "snapshot"（配置未声明该字段时的默认）；getattr 兜底以免依赖 configs/ 改动。
    saver = _BufferSaver(
        buf_dir,
        mode=str(getattr(cfg.train, "async_buffer_save", "snapshot")),
        warn=logger.warn,
    )

    # try/finally：所有退出路径（正常跑完 / 异常 / Ctrl-C）都 join 后台保存，避免留下
    # 半写状态或丢最后一迭代 buffer（save 内部 tmp+rename+.old 原子，最坏也可回退 .old）。
    try:
        for iteration in range(start_iter, total_iterations):
            iter_t0 = time.monotonic()
            # freeze_window 模式：下一迭代 selfplay 会 add_game 原地改写活 buffer，
            # 须在此 join 上一份「活 buffer」后台保存（snapshot/sync 模式此处 no-op，
            # 快照与活 buffer 解耦、可继续跨入本迭代 selfplay）。
            saver.before_selfplay()
            records_path = records_dir / f"iter_{iteration:04d}.jsonl"
            # --resume 崩溃重跑同一迭代时，该文件可能是上次中途写入的部分残留；
            # append_jsonl 是纯追加，不清理会导致行数混入两次运行、超过 games_per_iteration。
            records_path.unlink(missing_ok=True)

            # 1) Self-play（best 模型执双方）。
            sp_t0 = time.monotonic()
            sp_stats = _selfplay_phase(cfg, iteration, best_path, logger, buffer, records_path)
            selfplay_sec = time.monotonic() - sp_t0

            # 2) buffer 持久化发起（后台线程）。此刻起到下一迭代 selfplay 的首个
            #    add_game 之前 buffer 只读：snapshot 模式在此取快照可与 train+arena+
            #    下一迭代 selfplay 全程重叠；freeze_window 模式恰好把压缩藏进
            #    train+arena（before_selfplay 会在下次改写前 join）。save 内部先 join
            #    上一次保存（buf_dir 单写者、释放上份快照 RAM）；失败在线程内
            #    logger.warn，不中断训练（保持 ed69f24 语义）。
            saver.save(buffer)

            # 3) Train（buffer 达阈值才训练）。
            tr_t0 = time.monotonic()
            trained = len(buffer) >= int(cfg.train.min_buffer_to_train)
            if trained:
                new_samples = int(sp_stats.get("new_samples", 0))
                steps = resolve_train_steps(new_samples, cfg)
                mode = "自动" if int(cfg.train.train_steps_per_iteration) == 0 else "固定"
                logger.info(
                    f"[iter {iteration}] 本迭代新样本 {new_samples}，训练步数 {steps}（{mode}）"
                )
                tr_stats = _train_phase(model, optimizer, buffer, cfg, iteration, device, logger, steps)
            else:
                logger.info(
                    f"[iter {iteration}] buffer {len(buffer)} < min_buffer_to_train "
                    f"{cfg.train.min_buffer_to_train}，跳过训练"
                )
                tr_stats = {}
            train_sec = time.monotonic() - tr_t0

            # 4) 存本迭代 checkpoint（含 optimizer），供 arena 与续训使用。
            iter_path = _save_iter(iteration)

            # 5) Arena 门控（新 iter vs best）。多进程并行铺满全部 GPU（rl/evaluate.py）。
            ar_t0 = time.monotonic()
            arena_stats = arena(str(iter_path), str(best_path), cfg, device)
            arena_sec = time.monotonic() - ar_t0
            promoted = bool(trained and passes_gate(arena_stats, cfg.arena))
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
                elapsed_sec=arena_sec,
            )
            logger.info(
                f"[iter {iteration} | arena] se {arena_stats['score_se']:.3f}"
                f" | openings {arena_stats['distinct_openings']}/{arena_stats['games']}"
                f"（gate_lcb_z={float(cfg.arena.gate_lcb_z):.2f}）"
            )

            # 6) CSV 汇总行（buffer 落盘已在步骤 2 后台进行）。
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
    finally:
        # 所有退出路径（正常跑完 / 异常 / Ctrl-C）都 join 最后一份未落盘的后台保存——
        # 否则进程退出可能停在 rename 中途或丢最后一迭代 buffer（虽 save 原子换名 + load
        # 回退 .old 能兜底，但不必冒险）。必须在 logger.close() 之前完成。
        saver.close()

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
