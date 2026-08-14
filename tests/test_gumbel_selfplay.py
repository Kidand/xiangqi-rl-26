# -*- coding: utf-8 -*-
"""Gumbel 训练搜索模式在 selfplay 侧的接入（DESIGN §8）。

钉死 ``_WorkerEngine._decide`` 的两条分支：
- ``algorithm="gumbel"``：π 目标取 ``tree.gumbel_policy_target()`` 且**恒等**入库
  （不过 policy_target_temp/prune_frac 锐化），落子取 ``tree.gumbel_best_action()``
  （不走 sample_action_from_counts / 温度期 / 开局采样）；
- ``algorithm="puct"``（默认）：行为与现状完全一致（锐化生效、按访问计数选着）。

用假 SearchTree（monkeypatch ``rl.selfplay.SearchTree``）驱动真实 ``_GameSlot`` /
``_WorkerEngine``，board 为真实 ``Board``——_advance_slot 的一步杀捷径与 result 判定
都跑真代码。三处返回值（target / counts / best_action）刻意指向**三个不同**的合法
着法，任何"落子其实来自 π-argmax 或访问计数"的实现都会被区分出来。
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
    move_to_uci,
    uci_to_move,
)
from rl.config import Config
from rl.encoding import move_to_policy_index
import rl.selfplay as sp
from rl.selfplay import _WorkerEngine, sharpen_counts


# --------------------------------------------------------------------------- 假树
class _FakeTree:
    """最小 SearchTree 替身：只实现 _decide / _push_move 用到的接口。

    构造签名与真实 SearchTree 一致（_GameSlot.__init__ 直接调用）。
    """

    def __init__(self, board, cfg, add_noise=False, history_steps=2, rng=None, num_sims=None):
        self.board = board
        self.add_noise = add_noise
        self.counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.target = np.zeros(ACTION_SIZE, dtype=np.float64)
        self.best_action = -1
        self.value = 0.0
        self.advanced: list = []
        self.calls: dict = {"target": 0, "best": 0, "counts": 0, "root_value": 0}

    # --- 真实 SearchTree 的公共接口子集
    def visit_counts(self) -> np.ndarray:
        self.calls["counts"] += 1
        return self.counts

    def root_value(self) -> float:
        self.calls["root_value"] += 1
        return float(self.value)

    def gumbel_policy_target(self) -> np.ndarray:
        self.calls["target"] += 1
        return self.target

    def gumbel_best_action(self) -> int:
        self.calls["best"] += 1
        return int(self.best_action)

    def advance(self, move) -> None:
        self.advanced.append(move)

    @property
    def total_visits(self) -> int:
        return int(self.counts.sum())


@pytest.fixture(autouse=True)
def _fake_tree(monkeypatch):
    """全测试文件替换 SearchTree（_GameSlot.__init__ 里按模块全局名解析）。"""
    monkeypatch.setattr(sp, "SearchTree", _FakeTree)


def _cfg(algorithm: str) -> Config:
    """共同基线：锐化旋钮开着（gumbel 下必须无效）、采样旋钮关死（puct 下可预期）。"""
    cfg = Config()
    cfg.mcts.algorithm = algorithm
    cfg.mcts.policy_target_temp = 0.5      # T<1 锐化
    cfg.mcts.policy_target_prune_frac = 0.2
    cfg.mcts.temp_moves = 0
    cfg.mcts.temperature = 1.0
    cfg.mcts.temp_final = 0.0              # puct 分支 → argmax，可确定性断言
    cfg.selfplay.opening_random_plies = 0
    cfg.selfplay.resign_enabled = False
    return cfg


def _engine(cfg: Config, iteration: int = 0) -> _WorkerEngine:
    return _WorkerEngine(cfg, iteration, None, np.random.default_rng(0), proc_id=0)


def _slot(engine: _WorkerEngine, red_prefix: list[str] | None = None):
    """新建 slot；red_prefix 里的着法直接推 board（换手用，不进 moves/tree）。"""
    slot = engine.new_slot(0)
    for uci in red_prefix or []:
        slot.board.push(uci_to_move(uci))
        slot.ply += 1
    return slot


def _three_moves(board: Board):
    """取三个互不相同的合法着法（作为 target-argmax / counts-argmax / A*）。"""
    legal = board.legal_moves()
    assert len(legal) >= 3
    return legal[0], legal[1], legal[2]


# --------------------------------------------------------------------------- gumbel
class TestGumbelDecide:
    def _setup(self, engine, slot):
        """target-argmax=m_t、counts-argmax=m_c、A*=m_a 三者互不相同。"""
        m_t, m_c, m_a = _three_moves(slot.board)
        t = np.zeros(ACTION_SIZE, dtype=np.float64)
        t[move_to_action(m_t)] = 0.48
        t[move_to_action(m_c)] = 0.30
        t[move_to_action(m_a)] = 0.20
        # 0.02 < prune_frac(0.2)*max(0.48)=0.096 → 若误过 sharpen_counts 会被剪掉。
        m_tiny = slot.board.legal_moves()[3]
        t[move_to_action(m_tiny)] = 0.02
        slot.tree.target = t
        slot.tree.counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        slot.tree.counts[move_to_action(m_c)] = 100.0   # 访问计数 argmax（应被忽略）
        slot.tree.counts[move_to_action(m_t)] = 1.0
        slot.tree.best_action = move_to_action(m_a)
        slot.tree.value = 0.375
        return m_t, m_c, m_a, m_tiny

    def test_target_is_not_sharpened(self, monkeypatch):
        """π 入库 = gumbel_policy_target 逐项恒等（锐化旋钮开着也不生效）。"""
        eng = _engine(_cfg("gumbel"))
        assert eng.gumbel is True
        slot = _slot(eng)
        m_t, m_c, m_a, m_tiny = self._setup(eng, slot)
        # 锐化/温度采样两条 PUCT 专用路径在 gumbel 下一次都不许被调用。
        monkeypatch.setattr(sp, "sharpen_counts", lambda *a, **k: pytest.fail("走了锐化路径"))
        monkeypatch.setattr(
            sp, "sample_action_from_counts", lambda *a, **k: pytest.fail("走了温度采样")
        )

        eng._decide(slot)

        nz = np.nonzero(slot.tree.target)[0]
        assert len(slot.sample_pol_prob) == 1
        np.testing.assert_array_equal(
            slot.sample_pol_prob[0], slot.tree.target[nz].astype(np.float32)
        )
        # 小权重边（0.02）仍在：prune_frac 未生效。
        assert len(slot.sample_pol_prob[0]) == 4
        assert float(slot.sample_pol_prob[0].sum()) == pytest.approx(1.0)
        # 对照：若误过锐化，这条边会被剪掉、其余值也会变。
        sharp = sharpen_counts(slot.tree.target, 0.5, 0.2)
        assert np.count_nonzero(sharp) == 3
        assert sharp[move_to_action(m_tiny)] == 0.0
        assert slot.sample_pol_prob[0][list(nz).index(move_to_action(m_tiny))] == np.float32(0.02)

    def test_index_is_current_side_view_red(self):
        """idx = move_to_policy_index(move, 当前走子方)，红方走子。"""
        eng = _engine(_cfg("gumbel"))
        slot = _slot(eng)
        self._setup(eng, slot)
        assert slot.board.side_to_move == RED
        target = slot.tree.target
        eng._decide(slot)

        nz = np.nonzero(target)[0]
        expected = np.array(
            [move_to_policy_index(sp.action_to_move(int(a)), RED) for a in nz], dtype=np.int16
        )
        np.testing.assert_array_equal(slot.sample_pol_idx[0], expected)

    def test_index_is_current_side_view_black(self):
        """黑方走子时 idx 必须是 180° 翻转视角（本项目最大符号陷阱）。"""
        eng = _engine(_cfg("gumbel"))
        slot = _slot(eng, red_prefix=["h2e2"])  # 红先走一步 → 轮到黑
        assert slot.board.side_to_move == BLACK
        m_t, m_c, m_a, m_tiny = self._setup(eng, slot)
        target = slot.tree.target
        eng._decide(slot)

        nz = np.nonzero(target)[0]
        expected_black = np.array(
            [move_to_policy_index(sp.action_to_move(int(a)), BLACK) for a in nz], dtype=np.int16
        )
        expected_red = np.array(
            [move_to_policy_index(sp.action_to_move(int(a)), RED) for a in nz], dtype=np.int16
        )
        np.testing.assert_array_equal(slot.sample_pol_idx[0], expected_black)
        assert not np.array_equal(expected_black, expected_red)  # 确实翻了

    def test_move_comes_from_gumbel_best_action(self):
        """落子 = action_to_move(gumbel_best_action())，既非 π-argmax 也非访问计数 argmax。"""
        eng = _engine(_cfg("gumbel"))
        slot = _slot(eng)
        m_t, m_c, m_a = self._setup(eng, slot)[:3]
        eng._decide(slot)

        assert len({m_t, m_c, m_a}) == 3         # 三个来源确实指向不同着法
        assert slot.moves == [move_to_uci(m_a)]
        assert slot.tree.advanced == [m_a]       # 树复用同步
        assert slot.ply == 1

    def test_root_value_and_list_alignment(self):
        """root_value 入样本；逐步列表等长（build_payload 断言依赖）。"""
        eng = _engine(_cfg("gumbel"))
        slot = _slot(eng)
        self._setup(eng, slot)
        eng._decide(slot)
        self._setup(eng, slot)  # 按新局面（黑方）重配假树
        eng._decide(slot)       # 再走一步，验证持续对齐

        assert slot.sample_root_values == [0.375, 0.375]
        n = len(slot.sample_states)
        assert n == 2
        assert (
            len(slot.sample_pol_idx)
            == len(slot.sample_pol_prob)
            == len(slot.sample_sides)
            == len(slot.sample_mat_red)
            == len(slot.sample_root_values)
            == n
        )
        assert slot.sample_sides == [RED, BLACK]  # 逐步换手

    def test_resign_path_unchanged(self):
        """认输判定照旧：样本已入库、不落子、判负。"""
        cfg = _cfg("gumbel")
        cfg.selfplay.resign_enabled = True
        cfg.selfplay.resign_start_iter = 0
        cfg.selfplay.resign_consecutive = 1
        cfg.selfplay.resign_disable_frac = 0.0   # 不做校准局
        eng = _engine(cfg, iteration=0)
        slot = _slot(eng)
        self._setup(eng, slot)
        slot.tree.value = -0.99                  # < resign_threshold(-0.92)
        eng._decide(slot)

        assert slot.state == "finished"
        assert slot.termination == "resign"
        assert slot.result == "0-1"              # 红方认输 → 黑胜
        assert len(slot.sample_states) == 1      # 当前局面样本仍入库
        assert slot.moves == []                  # 未落子

    def test_degenerate_falls_back_without_sample(self):
        """gumbel_best_action 无效 / 目标全零（不应发生）→ 兜底走合法着法且不记样本。"""
        for best, target_ok in [(-1, True), (0, False)]:
            eng = _engine(_cfg("gumbel"))
            slot = _slot(eng)
            m_t, m_c, m_a, _ = self._setup(eng, slot)
            slot.tree.best_action = best if best < 0 else move_to_action(m_a)
            if not target_ok:
                slot.tree.target = np.zeros(ACTION_SIZE, dtype=np.float64)
            eng._decide(slot)

            assert slot.sample_states == []
            assert len(slot.moves) == 1
            assert slot.moves[0] == move_to_uci(slot.tree.advanced[0])

    def test_mate_shortcut_priority_unchanged(self):
        """一步杀捷径仍优先于 Gumbel：one-hot π + root_value=1.0，且不问树要目标。"""
        eng = _engine(_cfg("gumbel"))
        slot = _slot(eng)
        slot.board = Board("4k4/R8/9/9/9/9/9/9/9/1R1K5 w - - 0 1")
        slot.tree.calls = {"target": 0, "best": 0, "counts": 0, "root_value": 0}
        eng._advance_slot(slot)

        assert slot.moves[0] == "b0b9"                      # 车 b1-b10 将死
        assert slot.sample_root_values[0] == 1.0
        assert len(slot.sample_pol_idx[0]) == 1
        assert slot.sample_pol_prob[0][0] == pytest.approx(1.0)
        assert slot.tree.calls["target"] == 0 and slot.tree.calls["best"] == 0
        assert slot.state == "finished" and slot.result == "1-0"


# --------------------------------------------------------------------------- puct 零回归
class TestPuctBranchUnchanged:
    def test_default_algorithm_is_puct(self):
        eng = _engine(Config())
        assert eng.gumbel is False

    def test_puct_uses_counts_and_sharpening(self, monkeypatch):
        """algorithm="puct"：π 走锐化后的访问计数、落子取计数 argmax；不碰 Gumbel API。"""
        eng = _engine(_cfg("puct"))
        assert eng.gumbel is False
        slot = _slot(eng)
        m_t, m_c, m_a = _three_moves(slot.board)
        slot.tree.counts[move_to_action(m_c)] = 4.0     # argmax
        slot.tree.counts[move_to_action(m_t)] = 2.0
        slot.tree.counts[move_to_action(m_a)] = 1.0     # 1 < 0.2*4=0.8? 否 → 保留
        # gumbel 目标刻意指向别处：puct 下必须完全不看。
        slot.tree.target[move_to_action(m_a)] = 1.0
        slot.tree.value = -0.25
        monkeypatch.setattr(
            _FakeTree, "gumbel_best_action", lambda self: pytest.fail("puct 下调了 Gumbel API")
        )
        monkeypatch.setattr(
            _FakeTree, "gumbel_policy_target", lambda self: pytest.fail("puct 下调了 Gumbel API")
        )

        eng._decide(slot)

        assert slot.moves == [move_to_uci(m_c)]         # 计数 argmax（tau=0）
        counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        counts[move_to_action(m_c)] = 4.0
        counts[move_to_action(m_t)] = 2.0
        counts[move_to_action(m_a)] = 1.0
        expect_full = sharpen_counts(counts, 0.5, 0.2)   # 锐化确实生效
        nz = np.nonzero(expect_full)[0]
        np.testing.assert_array_equal(
            slot.sample_pol_prob[0], expect_full[nz].astype(np.float32)
        )
        # 与"未锐化"对照：证明这一分支确实过了锐化（gate 方向不可反）。
        raw = counts[nz] / counts[nz].sum()
        assert not np.allclose(slot.sample_pol_prob[0], raw)
        assert slot.sample_root_values == [-0.25]
