# -*- coding: utf-8 -*-
"""批量化 PUCT MCTS 测试（见 DESIGN.md §8 §15）。

覆盖：
- 一步杀局面：随机 eval_fn 下 100 sims 内绝大多数访问落到杀着（终局捷径）。
- visit 总数 == num_sims（多种 num_sims）。
- select_leaves / apply_results 两阶段流程 sanity + virtual loss 收支平衡。
- search() 便捷封装（mock eval_fn）返回棋盘真实坐标访问计数与根价值。
- 终局根局面、黑方视角真实坐标、树复用 advance。
"""

from __future__ import annotations

import numpy as np
import pytest

from xiangqi.board import Board
from xiangqi.constants import (
    ACTION_SIZE,
    BLACK,
    RED,
    move_to_action,
    uci_to_move,
)
from rl.config import MCTSConfig
from rl.encoding import move_to_policy_index
from rl.mcts import SearchTree, search

# 自构一步杀：红先手，唯一杀着 d7f7（车 d7->f7 将死黑将 f9）。
# 黑方另有 车+双炮 的反击子力，故红方除杀着外的其它着法并不决定性 —— MCTS 必须
# 靠终局捷径把绝大多数访问集中到这唯一杀着上。红方共 11 个合法着法。
MATE_FEN = "5k3/9/3R2c2/3r5/9/9/3c5/9/9/4K4 w - - 0 1"
MATE_MOVE = "d7f7"

C_DEFAULT = 14 * 2 + 3  # history_steps=2 -> 31 通道


def _uniform_eval(seed: int = 0):
    """未训练的随机 eval_fn：均匀策略 + 随机价值（非终局）。

    均匀策略模拟"对局面一无所知"，价值随机；MCTS 只能靠终局捷径把访问推向杀着。
    """
    rng = np.random.default_rng(seed)

    def fn(states: np.ndarray):
        b = states.shape[0]
        policy = np.full((b, ACTION_SIZE), 1.0 / ACTION_SIZE, dtype=np.float32)
        values = rng.uniform(-0.3, 0.3, size=b).astype(np.float32)
        return policy, values

    return fn


def _zero_eval(states: np.ndarray):
    """确定性 mock：均匀策略 + 价值 0。"""
    b = states.shape[0]
    policy = np.full((b, ACTION_SIZE), 1.0 / ACTION_SIZE, dtype=np.float32)
    values = np.zeros(b, dtype=np.float32)
    return policy, values


# --------------------------------------------------------------------------- 一步杀
def test_mate_shortcut_concentrates_visits():
    board = Board.from_fen(MATE_FEN)
    mate_action = move_to_action(uci_to_move(MATE_MOVE))
    cfg = MCTSConfig(num_sims=100, batch_size=8)

    counts, value = search(board, _uniform_eval(seed=1234), cfg, num_sims=100)

    assert counts.shape == (ACTION_SIZE,)
    total = int(counts.sum())
    assert total == 100                              # visit 总数 == num_sims
    # 杀着获得绝大多数访问（argmax + 明显多数），即便策略先验均匀无信息。
    assert int(counts.argmax()) == mate_action
    assert counts[mate_action] >= 0.6 * total        # 实测 >= 86%，阈值取保守值
    # 其余任何单一着法都远少于杀着。
    others = counts.copy()
    others[mate_action] = 0
    assert counts[mate_action] > others.max() * 3
    # 红方必胜局面，根价值应明显为正。
    assert value > 0.5


def test_mate_shortcut_robust_across_seeds():
    board_fen = MATE_FEN
    mate_action = move_to_action(uci_to_move(MATE_MOVE))
    for seed in range(6):
        board = Board.from_fen(board_fen)
        cfg = MCTSConfig(num_sims=100, batch_size=16)
        counts, _ = search(board, _uniform_eval(seed=seed), cfg, num_sims=100)
        assert int(counts.sum()) == 100
        assert int(counts.argmax()) == mate_action


# --------------------------------------------------------------------------- visit 总数
@pytest.mark.parametrize("num_sims", [1, 3, 16, 50, 100, 137])
def test_total_visits_equals_num_sims(num_sims):
    board = Board()  # 初始局面
    cfg = MCTSConfig(num_sims=num_sims, batch_size=8)
    counts, value = search(board, _zero_eval, cfg, num_sims=num_sims)
    assert int(counts.sum()) == num_sims
    assert -1.0 <= value <= 1.0
    # 所有非零访问必须对应合法着法（棋盘真实坐标）。
    legal_actions = {move_to_action(m) for m in board.legal_moves()}
    for idx in np.nonzero(counts)[0]:
        assert int(idx) in legal_actions


# --------------------------------------------------------------------------- 两阶段流程
def test_select_apply_flow_and_virtual_loss():
    board = Board()
    cfg = MCTSConfig(num_sims=64, batch_size=8)
    tree = SearchTree(board, cfg, add_noise=False)

    assert tree.total_visits == 0

    # 第一次 select：根尚未展开 -> 恰好返回根这一个叶子。
    states, tokens = tree.select_leaves(8)
    assert states.shape == (1, C_DEFAULT, 10, 9)
    assert states.dtype == np.float32
    assert len(tokens) == 1

    policy, values = _zero_eval(states)
    tree.apply_results(tokens, policy, values)
    assert tree.total_visits == 0                    # 根展开不计入模拟数
    assert tree._root.expanded

    # 第二次 select：应能凑出多个不同叶子（virtual loss 分流兄弟节点）。
    states, tokens = tree.select_leaves(8)
    assert states.shape[0] == len(tokens)
    assert len(tokens) >= 2
    assert states.shape[1:] == (C_DEFAULT, 10, 9)

    tree.apply_results(tokens, *_zero_eval(states))
    assert tree.total_visits == len(tokens)

    # virtual loss 收支平衡：apply 后根及其已展开子节点的 vloss 全部归零。
    assert (tree._root.child_vloss == 0).all()
    for child in tree._root.child_nodes:
        if child is not None and child.expanded:
            assert (child.child_vloss == 0).all()


def test_select_leaves_limited_by_available_nodes():
    # 全新树请求大量叶子：受可展开节点数限制，返回数量有限且不崩溃。
    board = Board()
    cfg = MCTSConfig(batch_size=64)
    tree = SearchTree(board, cfg, add_noise=False)
    states, tokens = tree.select_leaves(64)
    # 根未展开时只能产出根一个叶子。
    assert len(tokens) == 1
    assert states.shape[0] == 1


def test_dirichlet_noise_applied_at_root():
    board = Board()
    cfg = MCTSConfig(batch_size=8)

    tree_noise = SearchTree(board, cfg, add_noise=True, rng=np.random.default_rng(0))
    states, tokens = tree_noise.select_leaves(1)
    tree_noise.apply_results(tokens, *_zero_eval(states))
    priors_noisy = tree_noise._root.child_P.copy()

    tree_plain = SearchTree(board, cfg, add_noise=False)
    states, tokens = tree_plain.select_leaves(1)
    tree_plain.apply_results(tokens, *_zero_eval(states))
    priors_plain = tree_plain._root.child_P.copy()

    # 无噪声：均匀 mock 策略 -> 先验均匀；有噪声：明显偏离均匀。
    assert np.allclose(priors_plain, priors_plain[0])
    assert not np.allclose(priors_noisy, priors_noisy[0])
    assert abs(float(priors_noisy.sum()) - 1.0) < 1e-4


# --------------------------------------------------------------------------- 视角 / 真实坐标
def test_black_to_move_visit_counts_are_real_coords():
    # 黑方走子：visit_counts 必须是棋盘真实坐标（非翻转视角）。
    board = Board.from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1")
    assert board.side_to_move == BLACK
    cfg = MCTSConfig(num_sims=60, batch_size=8)
    counts, _ = search(board, _zero_eval, cfg, num_sims=60)
    assert int(counts.sum()) == 60

    legal = board.legal_moves()
    real_actions = {move_to_action(m) for m in legal}
    policy_indices = {move_to_policy_index(m, BLACK) for m in legal}
    nz = set(int(i) for i in np.nonzero(counts)[0])
    # 非零访问全部落在真实坐标合法集合内。
    assert nz.issubset(real_actions)
    # 真实坐标与视角索引确实不同（否则测不出翻转），且访问用的是真实坐标。
    assert real_actions != policy_indices


# --------------------------------------------------------------------------- 终局根
def test_terminal_root_returns_result_value():
    # 黑被将死（红胜）。黑走子且无着法。
    board = Board.from_fen("3k5/9/9/9/9/3R5/9/9/9/4K4 b - - 0 1")
    assert board.result() == "1-0"
    cfg = MCTSConfig(num_sims=50, batch_size=8)
    counts, value = search(board, _zero_eval, cfg, num_sims=50)
    assert int(counts.sum()) == 0                    # 无可搜索
    assert value == -1.0                             # 当前走子方（黑）视角：被将死 = -1


# --------------------------------------------------------------------------- 树复用
def test_advance_reuses_subtree():
    board = Board()
    cfg = MCTSConfig(num_sims=80, batch_size=8)
    tree = SearchTree(board, cfg, add_noise=False)

    # 先搜索建树。
    while tree.total_visits < 80:
        before = tree.total_visits
        states, tokens = tree.select_leaves(8)
        if tokens:
            tree.apply_results(tokens, *_zero_eval(states))
        if tree.total_visits == before and not tokens:
            break
    assert tree.total_visits == 80

    # 选一个被访问过的根着法推进。
    counts = tree.visit_counts()
    best_action = int(counts.argmax())
    from xiangqi.constants import action_to_move

    move = action_to_move(best_action)
    tree.advance(move)

    # 复用子树后可继续搜索，总访问数以新根为准（<= 之前子边访问数 + 后续）。
    assert tree.total_visits >= 0
    states, tokens = tree.select_leaves(8)
    if tokens:
        tree.apply_results(tokens, *_zero_eval(states))
    # 新根 visit_counts 合法且落在推进后局面的真实坐标合法集合内。
    b2 = board.copy()
    b2.push(move)
    legal_actions = {move_to_action(m) for m in b2.legal_moves()}
    for idx in np.nonzero(tree.visit_counts())[0]:
        assert int(idx) in legal_actions


def test_dirichlet_noise_reapplied_after_advance():
    """树复用：advance 到已展开子树后，新根必须重新混入 Dirichlet 噪声（DESIGN §8）。

    该子节点在非根位置展开时先验里没有噪声；若 advance 后不补噪声，自博弈开局将
    塌缩（每步只有真正的根第一次搜索有噪声）。此测试固定 mock 为均匀策略：无噪声时
    新根先验会是均匀的，补了噪声则明显偏离均匀。
    """
    board = Board()
    cfg = MCTSConfig(num_sims=200, batch_size=8)
    tree = SearchTree(board, cfg, add_noise=True, rng=np.random.default_rng(7))

    while tree.total_visits < 200:
        before = tree.total_visits
        states, tokens = tree.select_leaves(8)
        if tokens:
            tree.apply_results(tokens, *_zero_eval(states))
        if tree.total_visits == before and not tokens:
            break

    root = tree._root
    best_i = int(np.argmax(root.child_N))
    child = root.child_nodes[best_i]
    # 前提：被推进的子节点确实已展开（否则测不出补噪声）。
    assert child is not None and child.expanded
    move = root.child_moves[best_i]

    tree.advance(move)
    new_root = tree._root
    assert new_root.expanded
    priors = new_root.child_P
    # 均匀 mock 策略下，若未补噪声先验会保持均匀；补了噪声则偏离均匀。
    assert not np.allclose(priors, priors[0]), "advance 后新根缺少 Dirichlet 噪声"
    assert abs(float(priors.sum()) - 1.0) < 1e-4


def test_no_noise_advance_keeps_uniform_prior():
    """add_noise=False（arena/GUI）时 advance 不得引入噪声，先验保持确定性。"""
    board = Board()
    cfg = MCTSConfig(num_sims=200, batch_size=8)
    tree = SearchTree(board, cfg, add_noise=False)
    while tree.total_visits < 200:
        before = tree.total_visits
        states, tokens = tree.select_leaves(8)
        if tokens:
            tree.apply_results(tokens, *_zero_eval(states))
        if tree.total_visits == before and not tokens:
            break
    root = tree._root
    best_i = int(np.argmax(root.child_N))
    child = root.child_nodes[best_i]
    assert child is not None and child.expanded
    tree.advance(root.child_moves[best_i])
    priors = tree._root.child_P
    assert np.allclose(priors, priors[0]), "无噪声模式不应在 advance 后混入噪声"


def test_search_matches_manual_loop_shape():
    board = Board()
    cfg = MCTSConfig(num_sims=40, batch_size=8)
    counts, value = search(board, _zero_eval, cfg, num_sims=40)
    assert counts.shape == (ACTION_SIZE,)
    assert counts.dtype == np.float32
    assert int(counts.sum()) == 40
    assert isinstance(value, float)
