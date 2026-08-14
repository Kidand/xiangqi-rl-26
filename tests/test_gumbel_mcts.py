# -*- coding: utf-8 -*-
"""Gumbel 训练搜索模式测试（DESIGN.md §8 的 Gumbel 块）。

覆盖：
- Sequential Halving 配额数学（m=8/N=24、m=16/N=600 + 补齐）；
- halving 保留 top、候选恒 ⊆ legal、驱动循环必然终止且用满预算；
- m=1（唯一合法着法）→ 目标 one-hot、best = 该着；gumbel_m=1 单候选调度；
- 合成 bandit 的策略改进 E_π'[q] ≥ E_prior[q]（多随机种子）；
- v_mix 公式手算对照、σ 单调性、q̂ 视角（与 q_values/root_value 同向）；
- algorithm="puct" 时 Gumbel 代码路径完全不进入 + 仍混 Dirichlet（零回归红线）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xiangqi.board import Board
from xiangqi.constants import ACTION_SIZE, move_to_action, uci_to_move
from rl.config import MCTSConfig
from rl.mcts import SearchTree, _sh_schedule, search

# 红方仅一个合法着法：红帅 e0，黑车 d5/e5 分别封死 d0 与 e1（e5 同时将军），只剩 e0f0。
ONE_MOVE_FEN = "3k5/9/9/9/3rr4/9/9/9/9/4K4 w - - 0 1"
ONE_MOVE_UCI = "e0f0"
# 一步杀：红车 d7->f7 将死（与 test_mcts.py 同一局面）。
MATE_FEN = "5k3/9/3R2c2/3r5/9/9/3c5/9/9/4K4 w - - 0 1"
MATE_UCI = "d7f7"


def _cfg(**kw) -> MCTSConfig:
    base = dict(num_sims=64, batch_size=8, algorithm="gumbel", gumbel_m=8)
    base.update(kw)
    return MCTSConfig(**base)


def _tree(board: Board, cfg: MCTSConfig, seed: int = 0, num_sims=None) -> SearchTree:
    return SearchTree(
        board,
        cfg,
        add_noise=True,
        rng=np.random.default_rng(seed),
        num_sims=cfg.num_sims if num_sims is None else num_sims,
    )


def _drive(
    tree: SearchTree,
    target: int,
    batch: int,
    q_true: np.ndarray | None = None,
    root_prior: np.ndarray | None = None,
    v_root: float = 0.0,
    max_rounds: int = 5000,
):
    """按 selfplay worker 的循环驱动搜索（while total_visits < target: select_leaves）。

    q_true 给定时构造"合成 bandit"：根子边 i 的整棵子树价值恒为 q_true[i]（**根走子方
    视角**）。叶子在深度 d 的走子方视角值 = q_true[i]·(-1)^d，经 backup 逐层取负后，
    根子边累计的 W/N 恰好收敛到 q_true[i]（见 _backup 的符号推导）。
    返回实际驱动轮数。
    """
    rounds = 0
    while tree.total_visits < target and rounds < max_rounds:
        rounds += 1
        before = tree.total_visits
        states, tokens = tree.select_leaves(min(batch, target - before))
        if tokens:
            sparse = []
            values = []
            for tok in tokens:
                _node, path, legal, _side = tok
                d = len(path)
                if d == 0:
                    p = root_prior if root_prior is not None else np.full(len(legal), 1.0 / len(legal))
                    sparse.append(np.asarray(p, dtype=np.float64).copy())
                    values.append(float(v_root))
                else:
                    sparse.append(np.full(len(legal), 1.0 / len(legal)))
                    if q_true is None:
                        values.append(0.0)
                    else:
                        values.append(float(q_true[path[0][1]]) * ((-1.0) ** d))
            tree.apply_results_sparse(tokens, sparse, np.asarray(values))
        elif tree.total_visits == before:
            break
    return rounds


# --------------------------------------------------------------- SH 配额数学
def test_sh_schedule_math():
    # m=8, N=24 -> 8×1 + 4×2 + 2×4 = 24（恰好用满）
    assert _sh_schedule(8, 24) == [(8, 1), (4, 2), (2, 4)]
    assert sum(m * q for m, q in _sh_schedule(8, 24)) == 24
    # m=16, N=600 -> 16×9 + 8×18 + 4×37 + 2×75 = 586，余 14 由补齐阶段消化
    sched = _sh_schedule(16, 600)
    assert sched == [(16, 9), (8, 18), (4, 37), (2, 75)]
    assert sum(m * q for m, q in sched) == 586
    assert 600 - 586 == 14
    # phases = ceil(log2 m)；m=1 时 1 个 phase，配额 = 全部预算
    assert len(_sh_schedule(1, 30)) == 1 and _sh_schedule(1, 30) == [(1, 30)]
    for m in (2, 3, 5, 8, 16, 17):
        assert len(_sh_schedule(m, 100)) == max(1, math.ceil(math.log2(m)))
    # 配额下限 1（预算远小于 phases·m 时不能出 0，否则 phase 永远推不动）
    assert all(q >= 1 for _m, q in _sh_schedule(16, 3))


def test_sh_visit_distribution_matches_schedule():
    """m=8 / N=24：实际访问分布必须是 [7,7,3,3,1,1,1,1]（1 + 2 + 4 的累加）。"""
    cfg = _cfg(num_sims=24, batch_size=8, gumbel_m=8)
    tree = _tree(Board(), cfg, seed=3)
    _drive(tree, 24, 8)
    assert tree.total_visits == 24
    counts = tree.visit_counts()
    nz = np.sort(counts[counts > 0])[::-1]
    assert list(nz.astype(int)) == [7, 7, 3, 3, 1, 1, 1, 1]


def test_overflow_fills_budget_and_loop_terminates():
    """最终 phase 后仍有预算 -> 对存活候选 round-robin 补齐；驱动循环必须终止且用满。"""
    cfg = _cfg(num_sims=20, batch_size=4, gumbel_m=4)
    tree = _tree(Board(), cfg, seed=5)
    rounds = _drive(tree, 20, 4)
    # 调度只覆盖 4×2 + 2×5 = 18，剩余 2 次必须补齐到 20
    assert sum(m * q for m, q in _sh_schedule(4, 20)) == 18
    assert tree.total_visits == 20
    assert rounds < 100
    # 4×2 + 2×5 = 18，余 2 次必须 round-robin 均摊在末 phase 的 2 个存活候选上
    # -> [2+5+1, 2+5+1, 2, 2]；若补齐时只盯着单个候选会得到 [9,7,2,2]
    counts = tree.visit_counts()
    assert list(np.sort(counts[counts > 0])[::-1].astype(int)) == [8, 8, 2, 2]
    assert len(tree._g_cand) == 2                    # 末 phase 不再折半
    for c in tree._g_cand:
        assert int(tree._root.child_N[c]) == 8
    # 各种 (m, N, batch) 组合下都不能死循环、不能欠预算
    for m, n, b in [(1, 7, 4), (2, 9, 1), (3, 13, 5), (8, 24, 3), (16, 40, 8), (16, 40, 64)]:
        cfg = _cfg(num_sims=n, batch_size=b, gumbel_m=m)
        t = _tree(Board(), cfg, seed=m + n)
        _drive(t, n, b)
        assert t.total_visits == n, (m, n, b, t.total_visits)


def test_select_leaves_makes_progress_every_call():
    """预算未满时 select_leaves 不得"零叶子且零推进"（死循环红线）。"""
    cfg = _cfg(num_sims=32, batch_size=4, gumbel_m=8)
    tree = _tree(Board(), cfg, seed=7)
    while tree.total_visits < 32:
        before = tree.total_visits
        states, tokens = tree.select_leaves(min(4, 32 - before))
        assert tokens or tree.total_visits > before, "预算未满却零产出"
        if tokens:
            n = states.shape[0]
            tree.apply_results(
                tokens,
                np.full((n, ACTION_SIZE), 1.0 / ACTION_SIZE, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
            )
    assert tree.total_visits == 32


def test_mate_position_budget_and_terminal_quota():
    """终局叶子内部直接回传，也必须计入对应候选的配额（否则预算永远填不满）。"""
    cfg = _cfg(num_sims=48, batch_size=8, gumbel_m=8)
    tree = _tree(Board.from_fen(MATE_FEN), cfg, seed=11)
    _drive(tree, 48, 8)
    assert tree.total_visits == 48


# ------------------------------------------------------------- 候选与 halving
def test_candidates_are_topm_of_g_plus_logits_and_subset_of_legal():
    board = Board()
    legal = board.legal_moves()
    cfg = _cfg(num_sims=24, batch_size=8, gumbel_m=8)
    tree = _tree(board, cfg, seed=13)
    states, tokens = tree.select_leaves(1)   # 根尚未展开：返回根叶子
    n = len(tokens[0][2])
    # 用非均匀先验展开根，保证 logits 有区分度
    rng = np.random.default_rng(99)
    prior = rng.dirichlet(np.full(n, 0.7))
    tree.apply_results_sparse(tokens, [prior], np.array([0.0]))
    tree.select_leaves(1)              # 触发惰性初始化
    cand = tree._g_cand
    assert len(cand) == min(8, n)
    assert set(cand.tolist()) <= set(range(n))          # 候选 ⊆ legal
    assert len(set(cand.tolist())) == len(cand)
    # 与 g+logits 的 top-m 完全一致
    score0 = tree._g_gum + tree._g_logits
    expect = np.argsort(-score0, kind="stable")[: len(cand)]
    assert list(cand) == list(expect)
    # logits = log(掩码归一先验 + 1e-12)，且未混 Dirichlet
    assert np.allclose(tree._root.child_P.astype(np.float64), prior, atol=1e-6)
    assert np.allclose(tree._g_logits, np.log(tree._root.child_P.astype(np.float64) + 1e-12))
    assert len(tree._root.child_moves) == len(legal)


def test_halving_keeps_top_scores():
    """phase 末折半：按 g+logits+σ(q̂) 保留前 ceil(m_phase/2)（至少 1）。"""
    cfg = _cfg(num_sims=24, batch_size=8, gumbel_m=8)
    tree = _tree(Board(), cfg, seed=17)
    _drive(tree, 8, 8)                                  # 跑完 phase 0（8×1）
    assert tree._g_ready
    old_cand = tree._g_cand.copy()
    assert len(old_cand) == 8
    scores = tree._g_scores()[old_cand]                 # 折半发生前的分数快照
    tree._g_advance_phase()
    new_cand = tree._g_cand
    assert len(new_cand) == 4                           # ceil(8/2)
    expect = old_cand[np.argsort(-scores, kind="stable")[:4]]
    assert list(new_cand) == list(expect)
    assert set(new_cand.tolist()) <= set(old_cand.tolist())
    # 保留的确实是分数更高的一半
    kept = {int(i) for i in new_cand}
    smin = min(float(s) for c, s in zip(old_cand, scores) if int(c) in kept)
    smax = max(float(s) for c, s in zip(old_cand, scores) if int(c) not in kept)
    assert smin >= smax


def test_final_survivors_shrink_monotonically():
    cfg = _cfg(num_sims=24, batch_size=1, gumbel_m=8)
    tree = _tree(Board(), cfg, seed=19)
    sizes = []
    prev = None
    while tree.total_visits < 24:
        before = tree.total_visits
        states, tokens = tree.select_leaves(1)
        if tokens:
            tree.apply_results(
                tokens,
                np.full((1, ACTION_SIZE), 1.0 / ACTION_SIZE, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            )
        elif tree.total_visits == before:
            break
        if tree._g_ready:
            cur = tree._g_cand.copy()
            if prev is None or len(cur) != len(prev):
                sizes.append(len(cur))
                if prev is not None:
                    assert set(cur.tolist()) <= set(prev.tolist())
            prev = cur
    assert sizes == [8, 4, 2]


# ----------------------------------------------------------------- m = 1
def test_single_legal_move_one_hot_target():
    board = Board.from_fen(ONE_MOVE_FEN)
    assert len(board.legal_moves()) == 1
    action = move_to_action(uci_to_move(ONE_MOVE_UCI))
    cfg = _cfg(num_sims=16, batch_size=4, gumbel_m=16)
    tree = _tree(board, cfg, seed=23)
    _drive(tree, 16, 4)
    assert tree.total_visits == 16
    target = tree.gumbel_policy_target()
    assert target.shape == (ACTION_SIZE,)
    assert pytest.approx(1.0, abs=1e-12) == float(target.sum())
    assert float(target[action]) == pytest.approx(1.0, abs=1e-12)   # one-hot
    assert int((target > 0).sum()) == 1
    assert tree.gumbel_best_action() == action
    assert len(tree._g_cand) == 1                                    # m = min(16, 1)


def test_gumbel_m_one_single_candidate():
    """gumbel_m=1：phases=1、配额=全部预算，所有访问落在唯一候选上（目标仍覆盖全部合法着法）。"""
    cfg = _cfg(num_sims=12, batch_size=4, gumbel_m=1)
    board = Board()
    tree = _tree(board, cfg, seed=29)
    _drive(tree, 12, 4)
    assert tree.total_visits == 12
    counts = tree.visit_counts()
    assert int((counts > 0).sum()) == 1
    assert int(counts.max()) == 12
    cand = tree._g_cand
    assert len(cand) == 1
    assert tree.gumbel_best_action() == move_to_action(tree._root.child_moves[int(cand[0])])
    target = tree.gumbel_policy_target()
    assert int((target > 0).sum()) == len(board.legal_moves())       # completed-Q 覆盖全部合法着法
    assert pytest.approx(1.0, abs=1e-12) == float(target.sum())


# --------------------------------------------------------- v_mix / σ / 视角
def test_v_mix_formula_hand_computed():
    """v_mix = (v_net + ΣN·q̄_π)/(1+ΣN)，q̄_π = Σ_{N>0}π Q / Σ_{N>0}π（手算对照）。"""
    cfg = _cfg(num_sims=8, batch_size=4, gumbel_m=4)
    tree = _tree(Board(), cfg, seed=31)
    states, tokens = tree.select_leaves(1)               # 根叶子
    n = len(tokens[0][2])
    prior = np.full(n, 1.0 / n)
    tree.apply_results_sparse(tokens, [prior], np.array([0.25]))     # v_net(root) = 0.25
    root = tree._root
    assert root.net_value == pytest.approx(0.25)
    # 手工摆一个根统计：子边 0 访问 3 次累计 +1.5（Q=0.5），子边 1 访问 1 次累计 -0.5（Q=-0.5）
    root.child_N[:] = 0
    root.child_W[:] = 0.0
    root.child_N[0], root.child_W[0] = 3, 1.5
    root.child_N[1], root.child_W[1] = 1, -0.5
    P = np.zeros(n)
    P[0], P[1] = 0.4, 0.2
    P[2:] = 0.4 / (n - 2)
    root.child_P = P.astype(np.float32)
    # 手算：q̄_π = (0.4*0.5 + 0.2*(-0.5))/0.6 = 1/6；ΣN=4
    #       v_mix = (0.25 + 4/6)/5 = 0.18333333...
    expect = (0.25 + 4.0 * (1.0 / 6.0)) / 5.0
    assert expect == pytest.approx(0.1833333333333333, abs=1e-12)
    assert tree._g_v_mix() == pytest.approx(expect, abs=1e-9)
    # q̂：已访问取 W/N，未访问取 v_mix
    q = tree._g_qhat()
    assert q[0] == pytest.approx(0.5) and q[1] == pytest.approx(-0.5)
    assert np.allclose(q[2:], expect)
    # 无任何已访问子边时退化为 v_net(root)
    root.child_N[:] = 0
    root.child_W[:] = 0.0
    assert tree._g_v_mix() == pytest.approx(0.25)


def test_sigma_monotonic_and_scaling():
    cfg = _cfg(num_sims=8, batch_size=4, gumbel_m=4, gumbel_c_visit=50.0, gumbel_c_scale=1.0)
    tree = _tree(Board(), cfg, seed=37)
    states, tokens = tree.select_leaves(1)
    n = len(tokens[0][2])
    tree.apply_results_sparse(tokens, [np.full(n, 1.0 / n)], np.array([0.0]))
    root = tree._root
    q = np.array([-1.0, -0.5, 0.0, 0.25, 1.0])
    s0 = tree._g_sigma(q)
    assert np.all(np.diff(s0) > 0)                       # 关于 q̂ 严格单调递增
    assert np.allclose(s0, (50.0 + 0.0) * 1.0 * q)       # max N = 0
    root.child_N[0] = 7                                  # max_b N(b) 变大 -> 幅度变大
    s1 = tree._g_sigma(q)
    assert np.allclose(s1, (50.0 + 7.0) * q)
    assert abs(s1[-1]) > abs(s0[-1]) and abs(s1[0]) > abs(s0[0])
    # c_scale 线性缩放
    cfg2 = _cfg(num_sims=8, gumbel_c_visit=50.0, gumbel_c_scale=2.0)
    t2 = _tree(Board(), cfg2, seed=37)
    st2, tk2 = t2.select_leaves(1)
    t2.apply_results_sparse(tk2, [np.full(len(tk2[0][2]), 1.0 / len(tk2[0][2]))], np.array([0.0]))
    assert np.allclose(t2._g_sigma(q), 2.0 * 50.0 * q)


def test_qhat_perspective_matches_q_values_and_root_value():
    """q̂ 必须与 q_values()/root_value() 同向（根走子方视角），不得多取一次负。"""
    cfg = _cfg(num_sims=32, batch_size=4, gumbel_m=8)
    board = Board()
    tree = _tree(board, cfg, seed=41)
    rng = np.random.default_rng(41)
    n = len(board.legal_moves())
    q_true = rng.uniform(-0.9, 0.9, size=n)
    _drive(tree, 32, 4, q_true=q_true, v_root=0.3)
    qhat = tree._g_qhat()
    qv = tree.q_values()
    root = tree._root
    seen = 0
    for i, move in enumerate(root.child_moves):
        if root.child_N[i] > 0:
            seen += 1
            assert qhat[i] == pytest.approx(float(qv[move_to_action(move)]), abs=1e-6)
            # 合成 bandit 下 W/N 应精确收敛到 q_true[i]（视角自洽的直接证据）
            assert qhat[i] == pytest.approx(float(q_true[i]), abs=1e-9)
    assert seen >= 4
    # root_value = ΣW/ΣN，与子边 Q 同一尺度、同一视角
    N = root.child_N
    assert tree.root_value() == pytest.approx(
        float((qhat[N > 0] * N[N > 0]).sum() / N.sum()), abs=1e-6
    )


# ------------------------------------------------------------ 策略改进（bandit）
def test_policy_improvement_on_synthetic_bandit():
    """固定先验 + 固定真值 q 的合成 bandit：E_π'[q] ≥ E_prior[q]（多种子）。"""
    board = Board()
    legal = board.legal_moves()
    n = len(legal)
    actions = np.array([move_to_action(m) for m in legal])
    gains = []
    for seed in range(8):
        rng = np.random.default_rng(1000 + seed)
        q_true = rng.uniform(-1.0, 1.0, size=n)
        prior = rng.dirichlet(np.full(n, 0.8))
        cfg = _cfg(num_sims=48, batch_size=4, gumbel_m=8)
        tree = _tree(board, cfg, seed=seed)
        _drive(tree, 48, 4, q_true=q_true, root_prior=prior, v_root=0.0)
        assert tree.total_visits == 48
        target = tree.gumbel_policy_target()
        assert pytest.approx(1.0, abs=1e-9) == float(target.sum())
        e_new = float((target[actions] * q_true).sum())
        e_old = float((prior * q_true).sum())
        gains.append(e_new - e_old)
        assert e_new >= e_old - 1e-9, (seed, e_new, e_old)
        # 落子也必须是改进策略下的高价值着法（≥ 先验期望）
        best = tree.gumbel_best_action()
        i_best = int(np.nonzero(actions == best)[0][0])
        assert q_true[i_best] >= e_old
    assert float(np.mean(gains)) > 0.1, gains


def test_policy_target_is_normalized_over_all_legal():
    cfg = _cfg(num_sims=32, batch_size=8, gumbel_m=8)
    board = Board()
    tree = _tree(board, cfg, seed=43)
    _drive(tree, 32, 8)
    target = tree.gumbel_policy_target()
    assert target.dtype == np.float64
    assert target.shape == (ACTION_SIZE,)
    assert pytest.approx(1.0, abs=1e-12) == float(target.sum())
    legal_actions = {move_to_action(m) for m in board.legal_moves()}
    assert set(np.nonzero(target)[0].tolist()) <= legal_actions
    assert int((target > 0).sum()) == len(legal_actions)     # 未访问着法也被 completed-Q 补全
    # 搜索未完成（甚至根未展开）时也可调
    fresh = _tree(board, cfg, seed=44)
    assert float(fresh.gumbel_policy_target().sum()) == 0.0
    assert fresh.gumbel_best_action() == -1
    fresh.select_leaves(1)
    assert float(fresh.gumbel_policy_target().sum()) == 0.0   # 根仍未展开


def test_best_action_is_argmax_over_survivors():
    cfg = _cfg(num_sims=24, batch_size=4, gumbel_m=8)
    tree = _tree(Board(), cfg, seed=47)
    _drive(tree, 24, 4)
    cand = tree._g_cand
    scores = tree._g_scores()
    i = int(cand[int(np.argmax(scores[cand]))])
    assert tree.gumbel_best_action() == move_to_action(tree._root.child_moves[i])
    # 存活候选一定是被访问过的（预算全部投在候选上）
    assert all(tree._root.child_N[c] > 0 for c in cand)


# ------------------------------------------------------------------ 树复用
def test_advance_resets_gumbel_state():
    cfg = _cfg(num_sims=16, batch_size=4, gumbel_m=8)
    board = Board()
    tree = _tree(board, cfg, seed=53)
    _drive(tree, 16, 4)
    assert tree._g_ready
    old_g = tree._g_gum.copy()
    best = tree.gumbel_best_action()
    move = next(m for m in board.legal_moves() if move_to_action(m) == best)
    tree.advance(move)
    assert not tree._g_ready and tree._g_cand is None
    _drive(tree, 16, 4)                      # 新根重新初始化（重新采 Gumbel 噪声）
    assert tree._g_ready
    assert tree._g_gum.shape != old_g.shape or not np.allclose(tree._g_gum, old_g)
    assert tree.total_visits >= 16
    assert set(tree._g_cand.tolist()) <= set(range(len(tree._root.child_moves)))


def test_reused_root_prior_has_no_dirichlet():
    """gumbel 模式：换根后也不得混 Dirichlet（先验保持网络原样）。"""
    cfg = _cfg(num_sims=24, batch_size=8, gumbel_m=8)
    board = Board()
    tree = _tree(board, cfg, seed=59)
    _drive(tree, 24, 8)
    best = tree.gumbel_best_action()
    move = next(m for m in board.legal_moves() if move_to_action(m) == best)
    tree.advance(move)
    root = tree._root
    if root.expanded:
        # 子树里的先验是均匀的（_drive 对非根叶子喂均匀先验），Dirichlet 会打破均匀性
        p = root.child_P.astype(np.float64)
        assert np.allclose(p, p[0], atol=1e-6)


# ---------------------------------------------------- PUCT 零回归（红线）
def test_puct_mode_never_enters_gumbel_path(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("PUCT 模式进入了 Gumbel 代码路径")

    monkeypatch.setattr(SearchTree, "_g_init", boom)
    monkeypatch.setattr(SearchTree, "_g_select_root", boom)
    monkeypatch.setattr(SearchTree, "_g_qhat", boom)
    monkeypatch.setattr(SearchTree, "_g_v_mix", boom)
    monkeypatch.setattr(SearchTree, "_g_advance_phase", boom)

    cfg = MCTSConfig(num_sims=32, batch_size=8)          # algorithm 默认 "puct"
    board = Board()
    tree = SearchTree(board, cfg, add_noise=True, rng=np.random.default_rng(2))
    assert tree._gumbel is False
    _drive(tree, 32, 8)
    assert tree.total_visits == 32
    tree.advance(board.legal_moves()[0])

    def fn(states):
        b = states.shape[0]
        return (
            np.full((b, ACTION_SIZE), 1.0 / ACTION_SIZE, dtype=np.float32),
            np.zeros(b, dtype=np.float32),
        )

    counts, value = search(board, fn, cfg, num_sims=32, add_noise=True)
    assert int(counts.sum()) == 32


def test_gumbel_disabled_without_noise():
    """arena / GUI（add_noise=False）即便 algorithm="gumbel" 也恒走 PUCT。"""
    cfg = _cfg(num_sims=16)
    t_arena = SearchTree(Board(), cfg, add_noise=False, rng=np.random.default_rng(0))
    assert t_arena._gumbel is False
    t_train = SearchTree(Board(), cfg, add_noise=True, rng=np.random.default_rng(0))
    assert t_train._gumbel is True


def test_puct_still_mixes_dirichlet_gumbel_does_not():
    board = Board()
    n = len(board.legal_moves())
    prior = np.full(n, 1.0 / n)

    puct = SearchTree(
        Board(), MCTSConfig(num_sims=16, batch_size=4), add_noise=True,
        rng=np.random.default_rng(3),
    )
    states, tokens = puct.select_leaves(1)
    puct.apply_results_sparse(tokens, [prior.copy()], np.array([0.0]))
    p_puct = puct._root.child_P.astype(np.float64)
    assert not np.allclose(p_puct, prior, atol=1e-4)     # Dirichlet 打破均匀

    gum = _tree(Board(), _cfg(num_sims=16, batch_size=4), seed=3)
    states, tokens = gum.select_leaves(1)
    gum.apply_results_sparse(tokens, [prior.copy()], np.array([0.0]))
    assert np.allclose(gum._root.child_P.astype(np.float64), prior, atol=1e-7)


def test_puct_rng_stream_untouched_by_gumbel_code():
    """gumbel 分支不得从 self._rng 取数：同种子下 PUCT 的先验序列必须完全可复现。"""
    board = Board()
    n = len(board.legal_moves())
    prior = np.full(n, 1.0 / n)
    out = []
    for _ in range(2):
        t = SearchTree(
            Board(), MCTSConfig(num_sims=16, batch_size=4), add_noise=True,
            rng=np.random.default_rng(12345),
        )
        states, tokens = t.select_leaves(1)
        t.apply_results_sparse(tokens, [prior.copy()], np.array([0.0]))
        out.append(t._root.child_P.astype(np.float64).copy())
    assert np.array_equal(out[0], out[1])
    # 与"不经过 SearchTree"的直接抽样一致：Dirichlet 是该 rng 的第一次抽样
    ref = np.random.default_rng(12345).dirichlet([0.2] * n)
    expect = (0.75 * prior + 0.25 * ref).astype(np.float32).astype(np.float64)
    assert np.allclose(out[0], expect, atol=0)
