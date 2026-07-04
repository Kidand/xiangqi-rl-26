# -*- coding: utf-8 -*-
"""Arena：新旧模型对抗门控（DESIGN.md §9.4 §11）。

``arena(new_ckpt, best_ckpt, cfg, device) -> stats``：
- ``arena_games`` 盘，红黑对调各半（新模型执红一半、执黑一半）。
- 每步 ``arena_sims`` 次 MCTS、**无 Dirichlet 噪声**；前 ``arena_temp_moves`` 步以
  ``arena_temperature`` 采样（轻微随机避免完全重复对局），其余 argmax。
- 返回总体 W/D/L、执红/执黑分拆、综合 score（胜=1，和=0.5，负=0）。
- **并行执行**（``arena_workers``，0=自动）：对局切块分给 spawn 进程池、每 worker
  绑定一张 GPU（arena 阶段 selfplay worker/EvalServer 已全部退出，整机空闲——串行
  单进程会让 8 卡空转数分钟）。统计语义与串行完全一致；1 worker 时走原串行路径
  （小规模/测试/monkeypatch 场景）。

坐标/编码/搜索一律复用现有模块，本模块不重复定义规则。
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict

import numpy as np

from xiangqi.board import Board
from xiangqi.constants import RED
from rl.mcts import search
from rl.model import load_checkpoint
from rl.selfplay import build_eval_fn, sample_action_from_counts

__all__ = ["arena"]


def _resolve_device(device: str, num_gpus: int) -> str:
    """把 "cuda"/"cpu" 解析为具体设备（arena 固定用 cuda:0 或 cpu）。"""
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _play_one(
    new_eval: Callable,
    best_eval: Callable,
    new_is_red: bool,
    cfg,
    rng: np.random.Generator,
) -> str:
    """对弈一盘，返回结果串 "1-0"/"0-1"/"1/2-1/2"（棋盘绝对视角）。"""
    board = Board(
        max_plies=int(cfg.selfplay.max_game_plies),
        no_capture_plies=int(cfg.selfplay.no_capture_draw_plies),
    )
    # 注入树内和棋惩罚（与 selfplay 同一来源 selfplay.draw_penalty，避免两处配置漂移）。
    # arena 不引入 temp_final：仍是 arena_temp_moves 之后 argmax。
    mcfg = dataclasses.replace(cfg.mcts, draw_value=float(cfg.selfplay.draw_penalty))
    sims = int(cfg.arena.arena_sims)
    hist = int(cfg.model.history_steps)
    temp_moves = int(cfg.arena.arena_temp_moves)
    temperature = float(cfg.arena.arena_temperature)
    ply = 0

    while board.result() is None:
        side = board.side_to_move
        # 新模型执红 → 红方走子时用新模型，否则用 best；执黑时反之。
        use_new = (side == RED) == new_is_red
        eval_fn = new_eval if use_new else best_eval

        counts, _ = search(board, eval_fn, mcfg, sims, add_noise=False, history_steps=hist)

        tau = temperature if (ply < temp_moves and temperature > 0.0) else 0.0
        move = sample_action_from_counts(counts, tau, rng)
        if move is None:
            legal = board.legal_moves()
            if not legal:
                break
            move = legal[0]
        board.push(move)
        ply += 1

    res = board.result()
    return res if res is not None else "1/2-1/2"


def _split_games(red_games: int, black_games: int, n_chunks: int) -> list[tuple[int, int]]:
    """把 (执红盘数, 执黑盘数) 均匀切成 n_chunks 块，各色总和精确不变。"""
    out = []
    for c in range(n_chunks):
        n_red = red_games * (c + 1) // n_chunks - red_games * c // n_chunks
        n_black = black_games * (c + 1) // n_chunks - black_games * c // n_chunks
        if n_red + n_black > 0:
            out.append((n_red, n_black))
    return out


_WORKER_DEVICE = None


def _arena_worker_init(dev_queue) -> None:
    """进程池 initializer：每个 worker 领取一个固定设备（每进程仅一个 CUDA 上下文），
    并限制单线程 BLAS（瓶颈在 Python MCTS，防多 worker CPU 超订）。"""
    global _WORKER_DEVICE
    _WORKER_DEVICE = dev_queue.get()
    import torch

    torch.set_num_threads(1)


def _arena_chunk(task: dict) -> Dict[str, int]:
    """子进程：加载两个 checkpoint，打一块对局，返回计数（与 arena 聚合语义一致）。"""
    dev = _WORKER_DEVICE or "cpu"
    new_model, _ = load_checkpoint(task["new_ckpt"], device=dev)
    best_model, _ = load_checkpoint(task["best_ckpt"], device=dev)
    new_model.eval()
    best_model.eval()
    new_eval = build_eval_fn(new_model, dev)
    best_eval = build_eval_fn(best_model, dev)

    cfg = task["cfg"]
    rng = np.random.default_rng(task["seed"])
    counts = {k: 0 for k in (
        "wins", "draws", "losses",
        "red_wins", "red_draws", "red_losses",
        "black_wins", "black_draws", "black_losses",
    )}
    for new_is_red in [True] * task["n_red"] + [False] * task["n_black"]:
        result = _play_one(new_eval, best_eval, new_is_red, cfg, rng)
        color = "red" if new_is_red else "black"
        if result == "1/2-1/2":
            counts["draws"] += 1
            counts[f"{color}_draws"] += 1
        elif (result == "1-0") == new_is_red:
            counts["wins"] += 1
            counts[f"{color}_wins"] += 1
        else:
            counts["losses"] += 1
            counts[f"{color}_losses"] += 1
    return counts


def _resolve_workers(cfg, games: int, dev: str) -> int:
    """解析 arena 并行度：显式 arena_workers 优先；0=自动
    （cuda 且盘数足够时 = GPU 数×4，受 CPU 核数与盘数约束；否则串行）。"""
    w = int(getattr(cfg.arena, "arena_workers", 0))
    if w == 1:
        return 1
    if w <= 0:
        if not dev.startswith("cuda") or games < 16:
            return 1
        import multiprocessing as mp
        import torch

        gpus = max(1, torch.cuda.device_count())
        w = min(max(1, mp.cpu_count() - 4), gpus * 4)
    return max(1, min(w, games))


def _arena_parallel(new_ckpt: str, best_ckpt: str, cfg, workers: int, dev: str) -> Dict[str, float]:
    """多进程 arena：全部对局切块经 spawn 进程池执行，worker 均匀铺满各 GPU。"""
    import multiprocessing as mp
    import torch

    if dev.startswith("cuda") and torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]

    games = int(cfg.arena.arena_games)
    red_games = games - games // 2
    chunks = _split_games(red_games, games - red_games, min(games, workers * 2))
    base_seed = int(cfg.seed) + 777_777
    tasks = [
        {"new_ckpt": new_ckpt, "best_ckpt": best_ckpt, "cfg": cfg,
         "n_red": nr, "n_black": nb, "seed": base_seed + i}
        for i, (nr, nb) in enumerate(chunks)
    ]

    ctx = mp.get_context("spawn")
    dev_queue = ctx.Queue()
    for i in range(workers):
        dev_queue.put(devices[i % len(devices)])
    agg = {k: 0 for k in (
        "wins", "draws", "losses",
        "red_wins", "red_draws", "red_losses",
        "black_wins", "black_draws", "black_losses",
    )}
    with ctx.Pool(processes=workers, initializer=_arena_worker_init, initargs=(dev_queue,)) as pool:
        for counts in pool.imap_unordered(_arena_chunk, tasks):
            for k in agg:
                agg[k] += counts[k]

    def _score(w: int, d: int, n: int) -> float:
        return (w + 0.5 * d) / n if n > 0 else 0.0

    return {
        "games": games,
        **agg,
        "score": _score(agg["wins"], agg["draws"], games),
        "red_score": _score(agg["red_wins"], agg["red_draws"], red_games),
        "black_score": _score(agg["black_wins"], agg["black_draws"], games - red_games),
    }


def arena(new_ckpt: str, best_ckpt: str, cfg, device: str) -> Dict[str, float]:
    """新模型 vs best 模型对抗，返回门控统计。

    Returns
    -------
    dict，含键：
      games, wins, draws, losses,
      red_wins, red_draws, red_losses, black_wins, black_draws, black_losses,
      score, red_score, black_score
    （wins/draws/losses 均以"新模型"为视角。）
    """
    dev = _resolve_device(device, int(cfg.train.num_gpus))

    workers = _resolve_workers(cfg, int(cfg.arena.arena_games), dev)
    if workers > 1:
        return _arena_parallel(new_ckpt, best_ckpt, cfg, workers, dev)

    new_model, _ = load_checkpoint(new_ckpt, device=dev)
    best_model, _ = load_checkpoint(best_ckpt, device=dev)
    new_model.eval()
    best_model.eval()
    new_eval = build_eval_fn(new_model, dev)
    best_eval = build_eval_fn(best_model, dev)

    games = int(cfg.arena.arena_games)
    rng = np.random.default_rng(int(cfg.seed) + 777_777)

    # 新模型执红的盘数（奇数时红多一盘）。
    red_games = games - games // 2

    wins = draws = losses = 0
    red_w = red_d = red_l = 0
    black_w = black_d = black_l = 0

    for g in range(games):
        new_is_red = g < red_games
        result = _play_one(new_eval, best_eval, new_is_red, cfg, rng)

        if result == "1/2-1/2":
            draws += 1
            if new_is_red:
                red_d += 1
            else:
                black_d += 1
        else:
            new_won = (result == "1-0" and new_is_red) or (result == "0-1" and not new_is_red)
            if new_won:
                wins += 1
                if new_is_red:
                    red_w += 1
                else:
                    black_w += 1
            else:
                losses += 1
                if new_is_red:
                    red_l += 1
                else:
                    black_l += 1

    def _score(w: int, d: int, n: int) -> float:
        return (w + 0.5 * d) / n if n > 0 else 0.0

    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "red_wins": red_w,
        "red_draws": red_d,
        "red_losses": red_l,
        "black_wins": black_w,
        "black_draws": black_d,
        "black_losses": black_l,
        "score": _score(wins, draws, games),
        "red_score": _score(red_w, red_d, red_games),
        "black_score": _score(black_w, black_d, games - red_games),
    }
