# -*- coding: utf-8 -*-
"""钉死 arena 胜负归属 / 晋升门控（rl/evaluate.py），弥补该模块此前零测试的缺口
（rl/evaluate.py:60-63 一带的 ``use_new``/``new_won`` 布尔逻辑一旦写反，门控会
静默颠倒——晋升表现更差的模型而不报错）。

全程 mock eval_fn / checkpoint 加载，不加载真实模型、不依赖 GPU：
- test_arena_* ：monkeypatch ``evaluate._play_one`` 返回预设结果序列，钉死
  ``arena()`` 本身的 wins/draws/losses/score 与红黑拆分统计（122 行 ``new_won``
  公式 + 108-134 行归属分支）。
- test_play_one_* ：不 mock ``search()``，只 mock eval_fn 本身（与真实调用方式
  一致），用极少 sims + 极短对局（``max_game_plies=2`` 强制 2 步内以和棋收场）
  记录每步实际调用了哪一方的 eval_fn，钉死 62 行 ``use_new`` 公式。
"""

from __future__ import annotations

import numpy as np
import pytest

from xiangqi.constants import ACTION_SIZE
from rl.config import Config
from rl import evaluate


class _FakeModel:
    """monkeypatch load_checkpoint 用的最小替身：只需支持 arena() 里调用的 .eval()。"""

    def eval(self) -> None:
        pass


def _fake_load_checkpoint(path, device=None):
    return _FakeModel(), {}


# --------------------------------------------------------------------------- (a)
def test_arena_attribution_and_score(monkeypatch):
    """monkeypatch ``_play_one`` 返回预设结果序列，钉死 arena() 的胜负归属
    （``new_won`` 公式，122 行）与红黑分拆统计、综合 score。

    6 盘：前 3 盘新模型执红（new_is_red=True，因 red_games = games - games//2
    = 3），后 3 盘执黑。依次给出结果 (1-0, 0-1, 1/2-1/2, 1-0, 0-1, 1/2-1/2)，
    恰好覆盖 (result, new_is_red) 的全部 6 种有意义组合各一次：
        (1-0, True)  -> 新模型胜（执红方赢）
        (0-1, True)  -> 新模型负（执红方输）
        (draw, True) -> 和
        (1-0, False) -> 新模型负（执黑方，红方赢不是新模型）
        (0-1, False) -> 新模型胜（执黑方，黑方赢是新模型）
        (draw, False)-> 和
    注意：本用例的胜负结果对 new_is_red 对称（每种执方各 1 胜 1 负 1 和），
    因此 ``new_won`` 整体对称调换（1-0/0-1 与 new_is_red 的配对同时取反）时
    总计数不变、本用例仍会通过——该 mutation 由下方非对称结果的
    ``test_arena_all_new_wins_gates_promotion`` 捕获，两个用例须配套存在。
    本用例钉死的是各计数桶的归属与 ``_score`` 的加权公式。
    """
    cfg = Config()
    cfg.arena.arena_games = 6
    cfg.train.device = "cpu"

    results = iter(["1-0", "0-1", "1/2-1/2", "1-0", "0-1", "1/2-1/2"])

    def fake_play_one(new_eval, best_eval, new_is_red, cfg_, rng):
        return next(results)

    monkeypatch.setattr(evaluate, "_play_one", fake_play_one)
    monkeypatch.setattr(evaluate, "load_checkpoint", _fake_load_checkpoint)

    stats = evaluate.arena("new.pt", "best.pt", cfg, device="cpu")

    assert stats["games"] == 6
    assert stats["wins"] == 2
    assert stats["losses"] == 2
    assert stats["draws"] == 2
    assert stats["red_wins"] == 1
    assert stats["red_losses"] == 1
    assert stats["red_draws"] == 1
    assert stats["black_wins"] == 1
    assert stats["black_losses"] == 1
    assert stats["black_draws"] == 1
    assert stats["score"] == pytest.approx(0.5)
    assert stats["red_score"] == pytest.approx(0.5)
    assert stats["black_score"] == pytest.approx(0.5)


def test_arena_all_new_wins_gates_promotion(monkeypatch):
    """补一组新模型全胜的场景：score 应为 1.0（>= 任意合理 gate_threshold），
    且 wins 应等于 games、losses/draws 均为 0——防止 ``new_won`` 公式在
    "全胜"这种最常见的门控通过场景下恰好因为写反而"偶然"算对分数。
    """
    cfg = Config()
    cfg.arena.arena_games = 4  # red_games = 4 - 2 = 2
    cfg.train.device = "cpu"

    # g=0,1 新模型执红 -> 红方赢(1-0)才算新模型赢；g=2,3 新模型执黑 -> 黑方赢(0-1)才算赢。
    results = iter(["1-0", "1-0", "0-1", "0-1"])

    def fake_play_one(new_eval, best_eval, new_is_red, cfg_, rng):
        return next(results)

    monkeypatch.setattr(evaluate, "_play_one", fake_play_one)
    monkeypatch.setattr(evaluate, "load_checkpoint", _fake_load_checkpoint)

    stats = evaluate.arena("new.pt", "best.pt", cfg, device="cpu")

    assert stats["wins"] == 4
    assert stats["losses"] == 0
    assert stats["draws"] == 0
    assert stats["red_wins"] == 2
    assert stats["black_wins"] == 2
    assert stats["score"] == pytest.approx(1.0)
    assert stats["score"] >= cfg.arena.gate_threshold  # 全胜必然通过晋升门控


# --------------------------------------------------------------------------- (b)
def _recording_eval(tag: str, log: list):
    """返回一个记录调用的 fake eval_fn：均匀策略 + 价值 0，足以让真实 search()
    正常跑完并选出一个合法着法；每次被调用时把 tag 追加到共享 log。"""

    def fn(states: np.ndarray):
        log.append(tag)
        b = states.shape[0]
        policy = np.full((b, ACTION_SIZE), 1.0 / ACTION_SIZE, dtype=np.float32)
        values = np.zeros(b, dtype=np.float32)
        return policy, values

    return fn


def _play_one_side_log(new_is_red: bool) -> list:
    """跑一盘真实 ``_play_one``（真实 Board + 真实 search()，只 mock eval_fn），
    ``max_game_plies=2`` 强制恰好 2 步（红方先手、黑方次之）后以和棋收场，
    返回按时间顺序记录的 "new"/"best" 调用日志（长度 >= 2，log[0] 对应红方那
    一步、log[-1] 对应黑方那一步——因为 search() 内部会把 num_sims 次评估都
    做完才返回，同一步内不会混用两个 eval_fn）。
    """
    log: list = []
    new_eval = _recording_eval("new", log)
    best_eval = _recording_eval("best", log)

    cfg = Config()
    cfg.arena.arena_sims = 8  # 极少 sims，几毫秒内跑完
    cfg.selfplay.max_game_plies = 2  # 2 步（红、黑各一步）后强制和棋，防止不必要地跑完整局
    rng = np.random.default_rng(0)

    result = evaluate._play_one(new_eval, best_eval, new_is_red, cfg, rng)
    assert result == "1/2-1/2"  # 2 步即触发 max_plies 和棋（sanity：确认场景确实只跑了 2 步）
    assert len(log) >= 2
    return log


def test_play_one_uses_new_eval_when_new_is_red_and_side_red():
    """new_is_red=True 时，红方（先手）那一步应调用新模型 eval_fn（62 行）。"""
    log = _play_one_side_log(new_is_red=True)
    assert log[0] == "new"


def test_play_one_uses_best_eval_when_new_is_red_and_side_black():
    """new_is_red=True 时，黑方（次手）那一步应调用 best 模型 eval_fn。"""
    log = _play_one_side_log(new_is_red=True)
    assert log[-1] == "best"


def test_play_one_uses_best_eval_when_new_is_black_and_side_red():
    """new_is_red=False（新模型执黑）时，红方（先手）那一步应调用 best 模型 eval_fn。"""
    log = _play_one_side_log(new_is_red=False)
    assert log[0] == "best"


def test_play_one_uses_new_eval_when_new_is_black_and_side_black():
    """new_is_red=False（新模型执黑）时，黑方（次手）那一步应调用新模型 eval_fn。"""
    log = _play_one_side_log(new_is_red=False)
    assert log[-1] == "new"
