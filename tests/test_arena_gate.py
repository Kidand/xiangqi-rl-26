# -*- coding: utf-8 -*-
"""LCB 门控（DESIGN §9.4）：``rl.evaluate.score_stats`` / ``passes_gate``。

- score_stats：手算对照几组 W/D/L，并覆盖全胜/全负边界（se 应为 0）。
- passes_gate：``gate_lcb_z<=0``（默认）时须与旧的 ``score>=gate_threshold``
  行为逐点一致；``gate_lcb_z>0`` 时验证 LCB 下界确实能把临界样本挡在门外，
  也能放行样本量更大/胜率更高的场景。
- 两条 arena 统计 dict 构造路径（串行 ~267-280 行、并行 ~196-201 行）都要带
  ``score_se`` 字段——分别驱动一次，钉死两处都没漏改。
- distinct_openings（DESIGN §9.4）：monkeypatch ``_play_one`` 返回可控的
  ``evaluate._PlayResult``（挂 ``.opening`` 指纹），驱动串行与并行两条路径，
  验证"全相同指纹→1"、"全不同指纹→N"两种边界，且口径一致。
"""

from __future__ import annotations

import math

import pytest

from rl.config import Config
from rl import evaluate
from rl.evaluate import passes_gate, score_stats


# --------------------------------------------------------------------------- score_stats
def test_score_stats_manual_example_borderline():
    """W40 D8 L32（N=80）：score=0.55，se 手算约 0.0527（独立于 score_stats 重新推导，
    避免测试与实现共用同一公式而失去意义）。"""
    w, d, l = 40, 8, 32
    n = w + d + l
    score, se = score_stats(w, d, l)

    assert score == pytest.approx(0.55)
    expected_se = math.sqrt(
        (w * (1 - 0.55) ** 2 + d * (0.5 - 0.55) ** 2 + l * (0 - 0.55) ** 2) / n / n
    )
    assert se == pytest.approx(expected_se)
    assert se == pytest.approx(0.0527, abs=2e-4)


def test_score_stats_manual_example_passing():
    """W48 D8 L24（N=80）：score=0.65，se 手算约 0.0503。"""
    score, se = score_stats(48, 8, 24)
    assert score == pytest.approx(0.65)
    assert se == pytest.approx(0.0503, abs=2e-4)


def test_score_stats_zero_games_is_zero():
    """无对局（N=0）：score 与 se 均为 0（不构成有效样本，不应除零）。"""
    assert score_stats(0, 0, 0) == (0.0, 0.0)


def test_score_stats_all_wins_se_is_zero():
    """全胜：每局得分恒为 1，样本方差为 0 → se=0。"""
    score, se = score_stats(80, 0, 0)
    assert score == pytest.approx(1.0)
    assert se == pytest.approx(0.0, abs=1e-12)


def test_score_stats_all_losses_se_is_zero():
    """全负：每局得分恒为 0，样本方差为 0 → se=0。"""
    score, se = score_stats(0, 0, 80)
    assert score == pytest.approx(0.0)
    assert se == pytest.approx(0.0, abs=1e-12)


def test_score_stats_all_draws_se_is_zero():
    """全和：每局得分恒为 0.5，样本方差为 0 → se=0。"""
    score, se = score_stats(0, 80, 0)
    assert score == pytest.approx(0.5)
    assert se == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- passes_gate
def test_passes_gate_lcb_off_matches_legacy_score_threshold():
    """gate_lcb_z<=0（默认）时必须与旧行为 ``score >= gate_threshold`` 逐点一致，
    不受 score_se 大小影响——覆盖阈值上下与恰好相等三种情形。"""
    cfg = Config().arena
    cfg.gate_threshold = 0.55
    cfg.gate_lcb_z = 0.0

    for wins, draws, losses in [(40, 8, 32), (48, 8, 24), (55, 0, 45), (54, 0, 46)]:
        score, se = score_stats(wins, draws, losses)
        stats = {"score": score, "score_se": se}
        assert passes_gate(stats, cfg) == (score >= cfg.gate_threshold)


def test_passes_gate_lcb_rejects_borderline_sample():
    """W40 D8 L32：score=0.55 达标，但 gate_lcb_z=1 时 LCB=score-se≈0.497<0.5，
    应拒绝晋升（抑制小样本噪声下的假晋升）。"""
    cfg = Config().arena
    cfg.gate_threshold = 0.55
    cfg.gate_lcb_z = 1.0

    score, se = score_stats(40, 8, 32)
    stats = {"score": score, "score_se": se}
    assert score >= cfg.gate_threshold  # 表面达标
    assert score - 1.0 * se < 0.5        # 但 LCB 未过半
    assert passes_gate(stats, cfg) is False


def test_passes_gate_lcb_accepts_stronger_sample():
    """W48 D8 L24：score=0.65，gate_lcb_z=1 时 LCB=score-se≈0.60≥0.5，应通过。"""
    cfg = Config().arena
    cfg.gate_threshold = 0.55
    cfg.gate_lcb_z = 1.0

    score, se = score_stats(48, 8, 24)
    stats = {"score": score, "score_se": se}
    assert passes_gate(stats, cfg) is True


def test_passes_gate_score_below_threshold_rejected_even_with_lcb_off():
    """score 本身未达阈值时，无论 gate_lcb_z 是否开启都必须拒绝。"""
    cfg = Config().arena
    cfg.gate_threshold = 0.55
    stats = {"score": 0.50, "score_se": 0.0}

    cfg.gate_lcb_z = 0.0
    assert passes_gate(stats, cfg) is False
    cfg.gate_lcb_z = 1.0
    assert passes_gate(stats, cfg) is False


# --------------------------------------------------------------------------- 两条 arena 路径都带 score_se
class _FakeModel:
    """monkeypatch load_checkpoint 用的最小替身：只需支持 .eval()。"""

    def eval(self) -> None:
        pass


def _fake_load_checkpoint(path, device=None):
    return _FakeModel(), {}


def test_arena_serial_path_includes_score_se(monkeypatch):
    """串行路径（workers=1，~267-280 行）：monkeypatch ``_play_one`` 钉死
    wins/draws/losses，验证聚合 dict 里的 score_se 与 score_stats 一致。"""
    cfg = Config()
    cfg.arena.arena_games = 6
    cfg.arena.arena_workers = 1  # 强制串行
    cfg.train.device = "cpu"

    results = iter(["1-0", "0-1", "1/2-1/2", "1-0", "0-1", "1/2-1/2"])

    def fake_play_one(new_eval, best_eval, new_is_red, cfg_, rng):
        return next(results)

    monkeypatch.setattr(evaluate, "_play_one", fake_play_one)
    monkeypatch.setattr(evaluate, "load_checkpoint", _fake_load_checkpoint)

    stats = evaluate.arena("new.pt", "best.pt", cfg, device="cpu")

    assert "score_se" in stats
    expected_score, expected_se = score_stats(stats["wins"], stats["draws"], stats["losses"])
    assert stats["score"] == pytest.approx(expected_score)
    assert stats["score_se"] == pytest.approx(expected_se)


def test_arena_parallel_path_includes_score_se(monkeypatch):
    """并行路径（~196-201 行）：monkeypatch 掉真实 spawn（进程池换成同进程直跑）、
    checkpoint 加载与 ``_play_one``，只验证聚合出的 dict 也带 score_se、且与
    score_stats 口径一致——不依赖真实多进程/真实模型（那部分由
    test_evaluate.py::test_arena_parallel_spawn_end_to_end 的 slow 用例兜底）。
    """
    import multiprocessing

    class _FakeQueue:
        def __init__(self):
            self._items = []

        def put(self, item):
            self._items.append(item)

        def get(self):
            return self._items.pop(0)

    class _FakePool:
        def __init__(self, processes=None, initializer=None, initargs=()):
            if initializer is not None:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def imap_unordered(self, fn, tasks):
            return [fn(t) for t in tasks]

    class _FakeCtx:
        def Queue(self):
            return _FakeQueue()

        def Pool(self, processes=None, initializer=None, initargs=()):
            return _FakePool(processes=processes, initializer=initializer, initargs=initargs)

    monkeypatch.setattr(multiprocessing, "get_context", lambda name: _FakeCtx())
    monkeypatch.setattr(evaluate, "load_checkpoint", _fake_load_checkpoint)
    # 恒返回 "1-0"（物理红方赢）：new_is_red=True 的对局记为新模型胜，
    # new_is_red=False 的对局记为新模型负——与切块方式无关，聚合结果确定。
    monkeypatch.setattr(evaluate, "_play_one", lambda new_eval, best_eval, new_is_red, cfg_, rng: "1-0")

    cfg = Config()
    cfg.arena.arena_games = 4
    cfg.arena.arena_workers = 2  # 显式并行，绕开 auto 的 cpu→串行判定
    cfg.train.device = "cpu"
    cfg.train.num_gpus = 0

    stats = evaluate.arena("new.pt", "best.pt", cfg, device="cpu")

    assert "score_se" in stats
    # red_games = 4 - 4//2 = 2 全胜，black_games = 2 全负。
    assert stats["wins"] == 2
    assert stats["losses"] == 2
    assert stats["draws"] == 0
    expected_score, expected_se = score_stats(2, 0, 2)
    assert stats["score"] == pytest.approx(expected_score)
    assert stats["score_se"] == pytest.approx(expected_se)


# --------------------------------------------------------------------------- distinct_openings（DESIGN §9.4）
def test_arena_serial_distinct_openings_all_same_is_one(monkeypatch):
    """串行路径：6 盘都返回同一开局指纹 → distinct_openings 应为 1。"""
    cfg = Config()
    cfg.arena.arena_games = 6
    cfg.arena.arena_workers = 1
    cfg.train.device = "cpu"

    same_opening = ("h2e2", "h9g7")

    def fake_play_one(new_eval, best_eval, new_is_red, cfg_, rng):
        return evaluate._PlayResult("1/2-1/2", same_opening)

    monkeypatch.setattr(evaluate, "_play_one", fake_play_one)
    monkeypatch.setattr(evaluate, "load_checkpoint", _fake_load_checkpoint)

    stats = evaluate.arena("new.pt", "best.pt", cfg, device="cpu")
    assert stats["distinct_openings"] == 1


def test_arena_serial_distinct_openings_all_different_is_n(monkeypatch):
    """串行路径：6 盘各返回互不相同的开局指纹 → distinct_openings 应等于盘数 6。"""
    cfg = Config()
    cfg.arena.arena_games = 6
    cfg.arena.arena_workers = 1
    cfg.train.device = "cpu"

    counter = iter(range(1000))

    def fake_play_one(new_eval, best_eval, new_is_red, cfg_, rng):
        i = next(counter)
        return evaluate._PlayResult("1/2-1/2", (f"a{i}b{i}",))

    monkeypatch.setattr(evaluate, "_play_one", fake_play_one)
    monkeypatch.setattr(evaluate, "load_checkpoint", _fake_load_checkpoint)

    stats = evaluate.arena("new.pt", "best.pt", cfg, device="cpu")
    assert stats["distinct_openings"] == 6


class _FakeQueueForOpenings:
    def __init__(self):
        self._items = []

    def put(self, item):
        self._items.append(item)

    def get(self):
        return self._items.pop(0)


class _FakePoolForOpenings:
    def __init__(self, processes=None, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def imap_unordered(self, fn, tasks):
        return [fn(t) for t in tasks]


class _FakeCtxForOpenings:
    def Queue(self):
        return _FakeQueueForOpenings()

    def Pool(self, processes=None, initializer=None, initargs=()):
        return _FakePoolForOpenings(processes=processes, initializer=initializer, initargs=initargs)


def test_arena_parallel_distinct_openings_matches_serial_semantics(monkeypatch):
    """并行路径（同进程假 Pool，不 spawn 真实进程）：验证 distinct_openings 的
    并集去重口径与串行路径一致——同一指纹在两个 chunk 里各出现一次算 1 个，
    每盘各自不同的指纹按盘数计数。"""
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_context", lambda name: _FakeCtxForOpenings())
    monkeypatch.setattr(evaluate, "load_checkpoint", _fake_load_checkpoint)

    cfg = Config()
    cfg.arena.arena_games = 4
    cfg.arena.arena_workers = 2  # 显式并行，绕开 auto 的 cpu→串行判定；切成 4 块（workers*2）
    cfg.train.device = "cpu"
    cfg.train.num_gpus = 0

    # 全部同一指纹 → 跨 chunk 求并集后仍应只有 1 个。
    same_opening = ("h2e2",)
    monkeypatch.setattr(
        evaluate, "_play_one",
        lambda new_eval, best_eval, new_is_red, cfg_, rng: evaluate._PlayResult("1/2-1/2", same_opening),
    )
    stats_same = evaluate.arena("new.pt", "best.pt", cfg, device="cpu")
    assert stats_same["distinct_openings"] == 1

    # 各盘指纹互不相同 → 应等于盘数（4）。
    counter = iter(range(1000))

    def fake_play_one_distinct(new_eval, best_eval, new_is_red, cfg_, rng):
        i = next(counter)
        return evaluate._PlayResult("1/2-1/2", (f"a{i}b{i}",))

    monkeypatch.setattr(evaluate, "_play_one", fake_play_one_distinct)
    stats_diff = evaluate.arena("new.pt", "best.pt", cfg, device="cpu")
    assert stats_diff["distinct_openings"] == 4
