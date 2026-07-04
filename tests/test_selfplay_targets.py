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
from rl.config import Config
from rl.selfplay import _WorkerEngine


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
