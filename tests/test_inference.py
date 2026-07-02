# -*- coding: utf-8 -*-
"""推理服务架构测试（DESIGN.md §8 稀疏评估 API + §9 推理服务）。

覆盖：
  (a) apply_results 与 apply_results_sparse 数值等价（同一随机局面/伪 policy）。
  (b) encode_board 向量化实现 vs 内嵌参考实现逐元素对拍。
  (c) run_eval_batch 单测（tiny 模型，验证与直接 forward+softmax+gather 一致、
      idx_len 截断正确）。
  (d) 端到端 mini：真实 spawn 1 CPU server + 2 workers 跑完 2 盘并收到 2 局 record。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from xiangqi.board import Board
from xiangqi.constants import (
    ACTION_SIZE,
    BLACK,
    COLS,
    NUM_SQUARES,
    RED,
    ROWS,
    flip_sq,
)
from rl.config import Config, MCTSConfig, ModelConfig
from rl.encoding import encode_board
from rl.inference_server import MAX_MOVES, run_eval_batch
from rl.mcts import SearchTree

C_DEFAULT = 14 * 2 + 3  # history_steps=2 -> 31


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def _fake_eval(states: np.ndarray):
    """确定性伪 net_eval：states -> (probs[B,8100], values[B])。

    仅依赖 states 内容（与随机数无关），保证两棵树在相同 states 序列上得到相同结果。
    """
    b = states.shape[0]
    s = states.reshape(b, -1).astype(np.float64).sum(axis=1)  # (b,)
    a = np.arange(ACTION_SIZE, dtype=np.float64)
    logits = np.sin(0.0007 * a[None, :] + 0.013 * s[:, None])  # (b, 8100)
    probs = _softmax(logits).astype(np.float32)
    values = np.tanh(0.01 * s).astype(np.float32)
    return probs, values


# ============================================================
# (a) apply_results 与 apply_results_sparse 等价
# ============================================================
def test_apply_results_sparse_equivalent_to_dense():
    board = Board()
    cfg = MCTSConfig(num_sims=96, batch_size=8)

    tree_dense = SearchTree(board, cfg, add_noise=False)
    tree_sparse = SearchTree(board, cfg, add_noise=False)

    def _run(tree, sparse: bool):
        while tree.total_visits < cfg.num_sims:
            before = tree.total_visits
            states, tokens = tree.select_leaves(min(cfg.batch_size, cfg.num_sims - before))
            if tokens:
                probs, values = _fake_eval(states)
                if sparse:
                    idx_list = tree.policy_indices(tokens)
                    sparse_probs = [probs[k][idx_list[k]] for k in range(len(tokens))]
                    tree.apply_results_sparse(tokens, sparse_probs, values)
                else:
                    tree.apply_results(tokens, probs, values)
            if tree.total_visits == before and not tokens:
                break

    _run(tree_dense, sparse=False)
    _run(tree_sparse, sparse=True)

    assert tree_dense.total_visits == tree_sparse.total_visits

    def _assert_node_equal(nd, ns):
        np.testing.assert_array_equal(nd.child_N, ns.child_N)
        np.testing.assert_allclose(nd.child_W, ns.child_W, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(nd.child_P, ns.child_P, rtol=1e-6, atol=1e-7)
        assert nd.child_moves == ns.child_moves

    root_d = tree_dense._root
    root_s = tree_sparse._root
    _assert_node_equal(root_d, root_s)
    # 递归比较已展开的子节点。
    for cd, cs in zip(root_d.child_nodes, root_s.child_nodes):
        if cd is not None and cs is not None and cd.expanded and cs.expanded:
            _assert_node_equal(cd, cs)


# ============================================================
# (b) encode_board 向量化 vs 参考实现对拍
# ============================================================
def _encode_board_reference(board, history_steps: int = 2) -> np.ndarray:
    """encode_board 的旧标量参考实现（对拍基准，逐元素应与向量化实现一致）。"""
    C = 14 * history_steps + 3
    planes = np.zeros((C, ROWS, COLS), dtype=np.float32)

    side = board.side_to_move
    flip = side == BLACK

    snapshots = board.snapshots(history_steps)
    for t, snap in enumerate(snapshots):
        if snap is None:
            continue
        base = t * 14
        for raw_sq in range(NUM_SQUARES):
            raw_val = snap[raw_sq]
            piece = raw_val - 7
            if piece == 0:
                continue
            enc_sq = flip_sq(raw_sq) if flip else raw_sq
            enc_row = enc_sq // COLS
            enc_col = enc_sq % COLS
            if flip:
                if piece < 0:
                    pt_idx = (-piece) - 1
                    planes[base + pt_idx, enc_row, enc_col] = 1.0
                else:
                    pt_idx = piece - 1
                    planes[base + 7 + pt_idx, enc_row, enc_col] = 1.0
            else:
                if piece > 0:
                    pt_idx = piece - 1
                    planes[base + pt_idx, enc_row, enc_col] = 1.0
                else:
                    pt_idx = (-piece) - 1
                    planes[base + 7 + pt_idx, enc_row, enc_col] = 1.0

    if side == RED:
        planes[C - 3, :, :] = 1.0

    last_move = getattr(board, "last_move", None)
    if last_move is not None:
        from_sq, to_sq = last_move
        if flip:
            from_sq = flip_sq(from_sq)
            to_sq = flip_sq(to_sq)
        planes[C - 2, from_sq // COLS, from_sq % COLS] = 1.0
        planes[C - 1, to_sq // COLS, to_sq % COLS] = 1.0

    return planes


def _random_playout_boards(seed: int, n_positions: int, max_plies: int = 30):
    """从初始局面随机走子，收集若干中途局面（覆盖红/黑双方视角、含 last_move）。"""
    rng = np.random.default_rng(seed)
    boards = []
    board = Board()
    boards.append(board.copy())  # 初始（红方，空 last_move）
    for _ in range(max_plies):
        if board.result() is not None:
            break
        legal = board.legal_moves()
        if not legal:
            break
        board.push(legal[rng.integers(0, len(legal))])
        boards.append(board.copy())
        if len(boards) >= n_positions:
            break
    return boards


@pytest.mark.parametrize("history_steps", [1, 2, 4])
def test_encode_board_matches_reference(history_steps):
    boards = _random_playout_boards(seed=2024, n_positions=25, max_plies=40)
    # 至少应包含红黑双方视角的局面。
    sides = {b.side_to_move for b in boards}
    assert RED in sides and BLACK in sides
    for b in boards:
        got = encode_board(b, history_steps=history_steps)
        ref = _encode_board_reference(b, history_steps=history_steps)
        assert got.shape == ref.shape
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, ref)


# ============================================================
# (c) run_eval_batch 单测
# ============================================================
def test_run_eval_batch_matches_manual_forward():
    import torch

    from rl.model import create_model

    torch.manual_seed(0)
    mc = ModelConfig(blocks=1, filters=8, history_steps=2)
    model = create_model(mc)
    model.eval()
    C = mc.in_channels

    rng = np.random.default_rng(7)
    n = 5
    states = rng.integers(0, 2, size=(n, C, ROWS, COLS)).astype(np.uint8)

    # 每叶子随机合法着法数 Li（1..MAX_MOVES），索引在 [0,8100) 内，pad 位置填 0。
    idx = np.zeros((n, MAX_MOVES), dtype=np.int16)
    idx_len = np.zeros((n,), dtype=np.int16)
    lens = [1, 3, MAX_MOVES, 10, 44]
    for i in range(n):
        L = lens[i]
        idx_len[i] = L
        idx[i, :L] = rng.integers(0, ACTION_SIZE, size=L).astype(np.int16)

    probs, values = run_eval_batch(model, "cpu", states, idx, idx_len, fp16=False)

    assert probs.shape == (n, MAX_MOVES)
    assert values.shape == (n,)
    assert probs.dtype == np.float32

    # 手动前向 + softmax + gather 对拍。
    with torch.inference_mode():
        x = torch.from_numpy(states.astype(np.float32))
        logits, val = model(x)
        full = torch.softmax(logits.float(), dim=1).numpy()
        val = val.float().numpy().reshape(-1)

    for i in range(n):
        L = int(idx_len[i])
        expected = full[i, idx[i, :L].astype(np.int64)]
        np.testing.assert_allclose(probs[i, :L], expected, rtol=1e-5, atol=1e-6)
        # idx_len 截断：pad 位置必须为 0。
        assert np.all(probs[i, L:] == 0.0)
    np.testing.assert_allclose(values, val, rtol=1e-5, atol=1e-6)


# ============================================================
# (d) 端到端 mini：真实 spawn 1 CPU server + 2 workers
# ============================================================
def test_end_to_end_mini_selfplay(tmp_path: Path):
    from rl.logger import TrainLogger
    from rl.model import create_model, save_checkpoint
    from rl.replay_buffer import ReplayBuffer
    from rl.train import _selfplay_phase

    cfg = Config()
    cfg.model = ModelConfig(blocks=1, filters=8, history_steps=2)
    cfg.mcts = MCTSConfig(num_sims=8, batch_size=4, temp_moves=4)
    cfg.selfplay.games_per_iteration = 2
    cfg.selfplay.num_workers = 2
    cfg.selfplay.games_per_worker = 1
    cfg.selfplay.eval_max_batch = 32
    cfg.selfplay.eval_wait_ms = 1.0
    cfg.selfplay.eval_fp16 = False
    cfg.selfplay.max_game_plies = 20
    cfg.selfplay.resign_enabled = False
    cfg.train.device = "cpu"
    cfg.train.num_gpus = 0
    cfg.log.tensorboard = False
    cfg.log.log_interval_sec = 1.0

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    model = create_model(cfg.model)
    save_checkpoint(model, cfg.model, best_path, iteration=-1, meta={"init": True})

    logger = TrainLogger(log_dir=str(tmp_path / "logs"), tensorboard=False)
    buffer = ReplayBuffer(50_000, (cfg.model.in_channels, 10, 9))
    records_path = tmp_path / "records" / "iter_0000.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    stats = _selfplay_phase(cfg, 0, best_path, logger, buffer, records_path)
    elapsed = time.monotonic() - t0
    logger.close()

    assert stats["games"] == 2, f"应产出 2 盘，实际 {stats['games']}"
    n_lines = sum(1 for ln in records_path.read_text(encoding="utf-8").splitlines() if ln.strip())
    assert n_lines == 2, f"record 应有 2 行，实际 {n_lines}"
    assert len(buffer) > 0
    # 若超 60s 应改标 slow；此处断言在预算内完成（tiny 模型 CPU）。
    assert elapsed < 120.0
