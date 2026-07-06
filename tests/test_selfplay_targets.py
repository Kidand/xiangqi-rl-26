# -*- coding: utf-8 -*-
"""钉死 selfplay 训练目标 z 的符号约定（rl/selfplay.py:_WorkerEngine.build_payload，
约 460-467 行），弥补现有测试只覆盖 blend_value/apply_draw_penalty/material_score
等纯函数本身、未覆盖 build_payload 内部符号组装的缺口。

``build_payload`` 是 ``_WorkerEngine`` 的方法（依赖 self.iteration / self.cfg），
不是可独立调用的纯函数；这里直接构造一个最小 ``_WorkerEngine`` 和一个满足其
读取需要的最小 slot 替身（不走 ``_GameSlot.__init__``，因为那需要真实
Board/SearchTree——build_payload 只读取几个字段，无需真实对局状态），
按契约要求“不碰产品代码”，纯从测试侧构造可控输入。

覆盖点（对应 selfplay.py 460-467 行）：
    z = 1.0 if winner == side else -1.0          # 464 行
    mat_side = mat_red * side                     # 465 行
    values[i] = blend_value(z, mat_tanh, lam)      # 467 行

两处可疑 mutation 及本文件中捕获它们的断言：
    1) 464 行 ``winner == side`` 误写成 ``winner != side``
       -> test_z_sign_matches_winner_and_loser 会失败（胜方 z 变负、负方 z 变正）。
    2) 465 行 ``mat_red * side`` 误写成 ``mat_red``（丢掉 ``* side``）
       -> test_material_blend_sign_flips_with_side 会失败（红方步与黑方步的材料
          混合项不再符号相反，退化为相等）。
这两个测试各自把 λ（value_blend）钉在 0 或 1 的极端值以隔离 z 与材料项，
使得每个 mutation 都精确落在恰好一个断言组里失败，互不干扰。
"""

from __future__ import annotations

import numpy as np
import pytest

from xiangqi.constants import BLACK, MATERIAL_SCALE, RED
from rl.config import Config, MCTSConfig
from rl.heuristics import blend_value, lambda_schedule, sims_schedule
from rl.selfplay import _WorkerEngine, sharpen_counts


class _FakeResign:
    """最小 stub：build_payload 只读取这两个属性，无需真实 ResignTracker 状态机。"""

    def __init__(self) -> None:
        self.resigned = False
        self.is_calibration_game = False


class _FakeSlot:
    """绕过 ``_GameSlot.__init__``（需要真实 Board/SearchTree）的最小替身。

    只填充 ``build_payload`` 实际读取的字段：result/termination、逐步
    sample_sides + sample_mat_red（z 目标计算的核心输入）、样本占位数组、
    moves、resign、_calib_resign_side。
    """

    def __init__(self, sides, mat_reds, result: str, channels: int) -> None:
        n = len(sides)
        assert len(mat_reds) == n
        self.result = result
        self.termination = "test"
        self.sample_sides = list(sides)
        self.sample_mat_red = list(mat_reds)
        # 逐步根价值（当前走子方视角）；β=0 默认下不参与数值，但 build_payload 断言其等长。
        self.sample_root_values = [0.0 for _ in range(n)]
        self.sample_states = [np.zeros((channels, 10, 9), dtype=np.uint8) for _ in range(n)]
        self.sample_pol_idx = [np.array([0], dtype=np.int16) for _ in range(n)]
        self.sample_pol_prob = [np.array([1.0], dtype=np.float32) for _ in range(n)]
        self.moves = [f"a{i}a{i}" for i in range(n)]  # 内容不参与断言，占位即可
        self.resign = _FakeResign()
        self.nn_evals = 0
        self._calib_resign_side = None


def _make_engine(
    value_blend_init: float,
    value_blend_iters: int,
    draw_penalty: float = -0.1,
    iteration: int = 0,
) -> tuple[_WorkerEngine, Config]:
    """构造一个不依赖 EvalClient/GPU 的最小 ``_WorkerEngine``（client=None 足够，
    build_payload 不会用到它）。"""
    cfg = Config()
    cfg.selfplay.value_blend_init = value_blend_init
    cfg.selfplay.value_blend_iters = value_blend_iters
    cfg.selfplay.draw_penalty = draw_penalty
    engine = _WorkerEngine(cfg, iteration, None, np.random.default_rng(0), proc_id=0)
    return engine, cfg


def _channels(cfg: Config) -> int:
    return 14 * int(cfg.model.history_steps) + 3


# --------------------------------------------------------------------------- (a)
def test_z_sign_matches_winner_and_loser():
    """胜方走子步 z 应为正、负方走子步应为负（selfplay.py 464 行）。

    lam 恒为 0（value_blend_iters=0 -> lambda_schedule 恒返回 0）以隔离材料混合
    项，使 blend_value(z, mat_tanh, 0) == z 精确成立——材料分仍取非零值，验证
    z 的符号不受材料项数值大小干扰。
    """
    engine, cfg = _make_engine(value_blend_init=0.5, value_blend_iters=0)
    C = _channels(cfg)

    # 红胜局：红方步（胜方）z 应为 +1，黑方步（负方）z 应为 -1。
    slot = _FakeSlot(sides=[RED, BLACK], mat_reds=[3.0, 3.0], result="1-0", channels=C)
    values = engine.build_payload(slot)["values"]
    assert values[0] > 0
    assert values[1] < 0
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(-1.0)

    # 黑胜局对称验证：黑方步（胜方）z 应为 +1，红方步（负方）z 应为 -1。
    slot2 = _FakeSlot(sides=[RED, BLACK], mat_reds=[-2.0, -2.0], result="0-1", channels=C)
    values2 = engine.build_payload(slot2)["values"]
    assert values2[0] < 0
    assert values2[1] > 0
    assert values2[0] == pytest.approx(-1.0)
    assert values2[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- (b)
def test_draw_penalty_same_for_both_sides():
    """和棋时双方样本应同为 draw_penalty（同号同值，不随 side 翻转）。

    同样把 lam 钉在 0 以隔离材料项；材料分故意给出双方不同的非零值，验证
    draw_penalty 的赋值确实与材料分/走子方无关（只有 winner==0 分支）。
    """
    engine, cfg = _make_engine(value_blend_init=0.5, value_blend_iters=0, draw_penalty=-0.15)
    C = _channels(cfg)

    slot = _FakeSlot(sides=[RED, BLACK], mat_reds=[5.0, -5.0], result="1/2-1/2", channels=C)
    values = engine.build_payload(slot)["values"]
    assert values[0] == pytest.approx(-0.15)
    assert values[1] == pytest.approx(-0.15)


# --------------------------------------------------------------------------- (c)
def test_material_blend_sign_flips_with_side():
    """材料混合项的符号应随走子方翻转：红优局面下红方步 value 高于黑方步
    （selfplay.py 465 行 ``mat_red * side``）。

    lam 恒为 1（value_blend_init=1.0 且 iteration=0 < value_blend_iters -> lambda
    恒为 init 值）使 blend_value 退化为纯材料项，隔离 z 的干扰。用同一份红优
    material_score（mat_red=3.0 > 0）分别标成红方步与黑方步：正确实现下二者应
    互为相反数（tanh 是奇函数）；若 465 行丢掉 ``* side``，两步的 mat_side 会
    相等，本测试的精确期望值与符号翻转断言均会失败。
    """
    engine, cfg = _make_engine(value_blend_init=1.0, value_blend_iters=10, iteration=0)
    C = _channels(cfg)

    mat_red = 3.0  # 红方优势，非零
    slot = _FakeSlot(sides=[RED, BLACK], mat_reds=[mat_red, mat_red], result="1-0", channels=C)
    values = engine.build_payload(slot)["values"]

    expected_red = float(np.tanh((mat_red * RED) / MATERIAL_SCALE))
    expected_black = float(np.tanh((mat_red * BLACK) / MATERIAL_SCALE))
    assert values[0] == pytest.approx(expected_red)
    assert values[1] == pytest.approx(expected_black)
    assert values[0] > 0 > values[1]
    assert values[0] == pytest.approx(-values[1])  # 奇函数：符号翻转、幅值相同


# ======================================================================
# π 训练目标锐化：sharpen_counts
# ======================================================================

def _old_sparse_probs(counts: np.ndarray) -> np.ndarray:
    """锐化前的原始归一（counts[nz] / sum），作为默认路径 bit-exact 对照。"""
    nz = np.nonzero(counts)[0]
    p = counts[nz].astype(np.float64)
    p /= p.sum()
    return p


class TestSharpenCounts:
    def test_default_is_identity_bitexact(self):
        """temp=1.0, prune_frac=0.0 时与原始归一逐 bit 相等（老 buffer 零回归）。"""
        rng = np.random.default_rng(0)
        for _ in range(50):
            counts = np.zeros(8100, dtype=np.float32)
            k = int(rng.integers(1, 20))
            idx = rng.choice(8100, k, replace=False)
            counts[idx] = rng.integers(1, 400, k).astype(np.float32)

            out = sharpen_counts(counts, 1.0, 0.0)
            nz = np.nonzero(counts)[0]
            np.testing.assert_array_equal(np.nonzero(out)[0], nz)
            np.testing.assert_array_equal(out[nz], _old_sparse_probs(counts))
            assert out.sum() == pytest.approx(1.0)

    def test_all_zero_returns_zero(self):
        """全零输入返回全零向量，不抛异常、无 NaN。"""
        counts = np.zeros(20, dtype=np.float32)
        for temp, prune in [(1.0, 0.0), (0.5, 0.0), (1.0, 0.5), (0.3, 0.7)]:
            out = sharpen_counts(counts, temp, prune)
            assert out.shape == counts.shape
            assert np.all(out == 0.0)
            assert not np.any(np.isnan(out))

    def test_one_hot_is_identity(self):
        """one-hot（mate_shortcut 路径的 π）经任意锐化仍为 one-hot（恒等，不崩）。"""
        counts = np.zeros(30, dtype=np.float64)
        counts[7] = 5.0
        for temp, prune in [(1.0, 0.0), (0.5, 0.0), (1.0, 0.9), (0.2, 0.5)]:
            out = sharpen_counts(counts, temp, prune)
            assert out[7] == pytest.approx(1.0)
            assert out.sum() == pytest.approx(1.0)
            assert np.count_nonzero(out) == 1

    def test_temp_sharpens_and_preserves_order(self):
        """temp<1 锐化：p ∝ c^(1/T)，排序不变、熵下降。"""
        counts = np.array([4.0, 1.0, 2.0], dtype=np.float64)
        base = sharpen_counts(counts, 1.0, 0.0)
        sharp = sharpen_counts(counts, 0.5, 0.0)  # 1/T = 2 → c^2
        expected = np.array([16.0, 1.0, 4.0]) / 21.0
        np.testing.assert_allclose(sharp, expected)
        np.testing.assert_array_equal(np.argsort(base), np.argsort(sharp))
        h_base = -(base[base > 0] * np.log(base[base > 0])).sum()
        h_sharp = -(sharp[sharp > 0] * np.log(sharp[sharp > 0])).sum()
        assert h_sharp < h_base

    def test_temp_gt_one_flattens(self):
        """temp>1 平滑：熵上升（对称验证锐化公式方向）。"""
        counts = np.array([8.0, 2.0, 1.0], dtype=np.float64)
        base = sharpen_counts(counts, 1.0, 0.0)
        flat = sharpen_counts(counts, 2.0, 0.0)
        h_base = -(base * np.log(base)).sum()
        h_flat = -(flat * np.log(flat)).sum()
        assert h_flat > h_base

    def test_prune_removes_low_edges(self):
        """prune_frac 剔除 visits < frac*max 的边，再在存活边上归一。"""
        counts = np.array([10.0, 1.0, 4.0, 3.0], dtype=np.float64)
        out = sharpen_counts(counts, 1.0, 0.5)  # 阈值 = 5，剔除 <5 的 1/4/3
        assert out[0] == pytest.approx(1.0)
        assert np.count_nonzero(out) == 1

    def test_prune_boundary_kept(self):
        """恰好等于阈值 frac*max 的边保留（严格小于才剪）。"""
        counts = np.array([10.0, 5.0], dtype=np.float64)
        out = sharpen_counts(counts, 1.0, 0.5)  # 阈值 = 5，5 不 < 5 → 保留
        np.testing.assert_allclose(out, np.array([10.0, 5.0]) / 15.0)

    def test_prune_then_temp_order(self):
        """先剪后幂：被剪的边保持 0（0**正=0），存活边再锐化归一。"""
        counts = np.array([9.0, 3.0, 1.0], dtype=np.float64)
        out = sharpen_counts(counts, 0.5, 0.5)  # 阈值 4.5 → 剩 [9]，平方仍 one-hot
        assert out[0] == pytest.approx(1.0)
        assert out[1] == 0.0 and out[2] == 0.0

    def test_output_sums_to_one(self):
        """非全零输入锐化后概率和恒为 1。"""
        rng = np.random.default_rng(7)
        for _ in range(20):
            counts = rng.integers(0, 50, 40).astype(np.float64)
            if counts.sum() == 0:
                continue
            for temp, prune in [(1.0, 0.0), (0.5, 0.0), (1.0, 0.3), (0.7, 0.4)]:
                out = sharpen_counts(counts, temp, prune)
                assert out.sum() == pytest.approx(1.0)


# ======================================================================
# num_sims 按迭代调度：sims_schedule
# ======================================================================

class TestSimsSchedule:
    def test_disabled_default_returns_num_sims(self):
        """num_sims_late=0（默认）→ 任意迭代恒返回 num_sims。"""
        cfg = MCTSConfig(num_sims=400, num_sims_late=0, num_sims_late_start_iter=10)
        for it in [0, 5, 10, 100]:
            assert sims_schedule(it, cfg) == 400

    def test_before_start_iter_uses_num_sims(self):
        """late>0 但 iteration < start_iter → 仍用 num_sims。"""
        cfg = MCTSConfig(num_sims=400, num_sims_late=800, num_sims_late_start_iter=20)
        assert sims_schedule(19, cfg) == 400

    def test_at_start_iter_switches(self):
        """iteration == start_iter（边界含等号）→ 切到 num_sims_late。"""
        cfg = MCTSConfig(num_sims=400, num_sims_late=800, num_sims_late_start_iter=20)
        assert sims_schedule(20, cfg) == 800

    def test_after_start_iter_uses_late(self):
        cfg = MCTSConfig(num_sims=400, num_sims_late=800, num_sims_late_start_iter=20)
        assert sims_schedule(50, cfg) == 800

    def test_start_iter_zero_switches_immediately(self):
        """start_iter=0 且 late>0 → 从 iter 0 起即用 late。"""
        cfg = MCTSConfig(num_sims=400, num_sims_late=800, num_sims_late_start_iter=0)
        assert sims_schedule(0, cfg) == 800

    def test_engine_uses_scheduled_sims_in_meta(self):
        """_WorkerEngine 生效 sims 走调度，且 build_payload meta.sims 写实际值。"""
        cfg = Config()
        cfg.mcts.num_sims = 400
        cfg.mcts.num_sims_late = 800
        cfg.mcts.num_sims_late_start_iter = 5
        eng_early = _WorkerEngine(cfg, 4, None, np.random.default_rng(0), proc_id=0)
        eng_late = _WorkerEngine(cfg, 5, None, np.random.default_rng(0), proc_id=0)
        assert eng_early.num_sims == 400 and eng_early.target == 400
        assert eng_late.num_sims == 800 and eng_late.target == 800
        # 不 mutate cfg。
        assert cfg.mcts.num_sims == 400
        C = 14 * int(cfg.model.history_steps) + 3
        slot = _FakeSlot(sides=[RED], mat_reds=[0.0], result="1-0", channels=C)
        assert eng_late.build_payload(slot)["record"]["meta"]["sims"] == 800


# ======================================================================
# z 混根价值：build_payload（β=cfg.selfplay.root_value_weight）
# ======================================================================

class TestZRootValueBlend:
    def _engine(self, beta, iteration=100):
        """iteration=100 使默认 value_blend_iters=30 下 λ=0，隔离材料混合便于手算。"""
        cfg = Config()
        cfg.selfplay.root_value_weight = beta
        return _WorkerEngine(cfg, iteration, None, np.random.default_rng(0), proc_id=0)

    def _slot(self, eng, sides, mat_reds, roots, result):
        C = 14 * int(eng.cfg.model.history_steps) + 3
        slot = _FakeSlot(sides=sides, mat_reds=mat_reds, result=result, channels=C)
        slot.sample_root_values = list(roots)
        return slot

    def test_beta_zero_is_identity(self):
        """β=0 → values 与不含根价值的原 blend 逐 bit 相等（原行为）。"""
        eng = self._engine(0.0)
        slot = self._slot(eng, [RED, BLACK], [0.0, 0.0], [0.3, -0.4], "1-0")
        values = eng.build_payload(slot)["values"]
        lam = lambda_schedule(100, eng.cfg.selfplay)  # =0
        expected = np.array([
            float(blend_value(1.0, 0.0, lam)),
            float(blend_value(-1.0, 0.0, lam)),
        ], dtype=np.float32)
        np.testing.assert_array_equal(values, expected)

    def test_beta_half_convex_combo(self):
        """β=0.5 → values = 0.5·blend + 0.5·root_value（root 与 z 同视角）。"""
        eng = self._engine(0.5)
        slot = self._slot(eng, [RED, BLACK], [0.0, 0.0], [0.3, -0.4], "1-0")
        values = eng.build_payload(slot)["values"]
        expected = np.array([
            0.5 * 1.0 + 0.5 * 0.3,
            0.5 * (-1.0) + 0.5 * (-0.4),
        ], dtype=np.float32)
        np.testing.assert_allclose(values, expected, atol=1e-6)

    def test_length_mismatch_asserts(self):
        """逐步列表错位 → build_payload 断言失败（防静默污染）。"""
        eng = self._engine(0.5)
        slot = self._slot(eng, [RED, BLACK], [0.0, 0.0], [0.3], "1-0")  # roots 短一个
        with pytest.raises(AssertionError):
            eng.build_payload(slot)

    def test_draw_with_root_blend(self):
        """和棋：z=draw_penalty，β 混入根价值。"""
        eng = self._engine(0.5)
        slot = self._slot(eng, [RED], [0.0], [-0.2], "1/2-1/2")
        values = eng.build_payload(slot)["values"]
        dp = float(eng.cfg.selfplay.draw_penalty)
        expected = np.float32(0.5 * dp + 0.5 * (-0.2))
        np.testing.assert_allclose(values, [expected], atol=1e-6)
