# -*- coding: utf-8 -*-
"""A/B 终点裁决脚本（scripts/ab_judge.py）单测。

三块，按重要性倒序排列：
- **视角**（最重要）：mock ``rl.evaluate.arena``，钉死第一参数（= arena 的 "new" 方、
  也是返回 score 的视角方）恒为 **gumbel** checkpoint。这个方向写反不会报错，只会让
  最终决策整体反向（GO↔关闭），是本脚本唯一的致命 bug 面。同时钉死 score 未在
  main 内被 1-x 之类地翻转（高分→GO、低分→关闭）。
- **判据边界**：0.55 / 0.45 两个预注册端点的归属（闭开区间），并用真实 arena 的
  得分公式（rl.evaluate.score_stats）验证 200 盘下端点可被精确命中（浮点无偏移）。
- **窗口聚合**：summarize_metrics 的分桶、均值、晋升计数、缺列容错、重复 iter 去重。

全程 CPU、不加载真实模型、不跑对局（checkpoint 用空文件占位，history_steps 探测
会告警后回退默认值，不影响被 mock 的 arena 调用）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ 不是已安装包（pyproject packages.find 只含 xiangqi/rl/gui），把仓库根
# 加入 sys.path 才能 import scripts.ab_judge（与 tests/test_distill.py 同款）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import ab_judge


# ═══════════════════════════════════════════════════════════════
# (a) 视角 —— 最重要的一条
# ═══════════════════════════════════════════════════════════════
def _fake_stats(score: float, games: int = 200, distinct: int = 180) -> dict:
    """构造一份与 rl.evaluate.arena 返回值键齐全的假 stats（gumbel 视角）。"""
    wins = int(round(score * games))
    return {
        "games": games,
        "wins": wins, "draws": 0, "losses": games - wins,
        "red_wins": wins // 2, "red_draws": 0, "red_losses": (games - wins) // 2,
        "black_wins": wins - wins // 2, "black_draws": 0,
        "black_losses": (games - wins) - (games - wins) // 2,
        "score": score, "score_se": 0.035,
        "red_score": score, "black_score": score,
        "distinct_openings": distinct,
    }


def _run_arena_only(tmp_path, monkeypatch, score: float, distinct: int = 180,
                    games: int = 200):
    """建空 ckpt 占位文件 → mock rl.evaluate.arena → 只跑对抗环节。

    返回 (rc, captured_call)；captured_call 记录 arena 的位置参数与 cfg。
    """
    import rl.evaluate as evaluate

    g_ckpt = tmp_path / "ckpts_ab_gumbel" / "best.pt"
    p_ckpt = tmp_path / "ckpts_ab_puct" / "best.pt"
    for p in (g_ckpt, p_ckpt):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")  # 占位：arena 被 mock，权重不会被真正加载

    captured: dict = {}

    def fake_arena(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _fake_stats(score, games=games, distinct=distinct)

    monkeypatch.setattr(evaluate, "arena", fake_arena)

    rc = ab_judge.main([
        "--gumbel-ckpt", str(g_ckpt),
        "--puct-ckpt", str(p_ckpt),
        "--games", str(games), "--sims", "77",
        "--device", "cpu", "--num-gpus", "2",
        "--skip-metrics",
    ])
    return rc, captured, g_ckpt, p_ckpt


def test_arena_first_arg_is_gumbel_ckpt(tmp_path, monkeypatch, capsys):
    """★ 视角测试：arena 的第一位置参数（"new" 方 = score 的视角方）必须是 gumbel
    臂 checkpoint，第二个才是 puct 臂。写反 = 返回的 score 变成 puct 视角 = 判据
    整体反向，且不会有任何报错。"""
    rc, captured, g_ckpt, p_ckpt = _run_arena_only(tmp_path, monkeypatch, 0.62)
    assert rc == 0

    args = captured["args"]
    assert args[0] == str(g_ckpt), "arena 第一参数必须是 gumbel 臂（视角方）"
    assert args[1] == str(p_ckpt), "arena 第二参数必须是 puct 臂（对照方）"
    assert not captured["kwargs"], "四个参数应全部位置传入，避免关键字名漂移"

    # 输出必须显式标注视角，否则人读报告时同样会读反
    text = capsys.readouterr().out
    assert "gumbel 视角" in text
    assert "new" in text and "best" in text  # 打印了谁是 new / 谁是 best


def test_arena_cfg_overrides(tmp_path, monkeypatch):
    """arena 段按 CLI + 去相关固定值覆盖，train.device/num_gpus 透传。"""
    _, captured, _, _ = _run_arena_only(tmp_path, monkeypatch, 0.5, games=64)
    cfg = captured["args"][2]
    assert cfg.arena.arena_games == 64
    assert cfg.arena.arena_sims == 77
    assert cfg.arena.arena_temp_moves == 16      # 去相关，硬编码
    assert cfg.arena.arena_temperature == 0.8    # 去相关，硬编码
    assert cfg.arena.arena_workers == 0
    assert cfg.train.device == "cpu"
    assert cfg.train.num_gpus == 2
    assert captured["args"][3] == "cpu"          # device 第四参数


def test_high_score_gives_go_low_score_gives_kill(tmp_path, monkeypatch, capsys):
    """score 在 main 内不得被翻转：gumbel 高分 → GO，低分 → 关闭路线。"""
    _run_arena_only(tmp_path, monkeypatch, 0.62)
    assert ab_judge.VERDICT_GO in capsys.readouterr().out

    _run_arena_only(tmp_path, monkeypatch, 0.30)
    assert ab_judge.VERDICT_KILL in capsys.readouterr().out


def test_low_opening_diversity_warns(tmp_path, monkeypatch, capsys):
    """distinct_openings/games < 70% 必须告警（去相关不足 → 有效样本量被高估）。"""
    _run_arena_only(tmp_path, monkeypatch, 0.62, distinct=100, games=200)
    assert "去相关不足" in capsys.readouterr().out

    _run_arena_only(tmp_path, monkeypatch, 0.62, distinct=180, games=200)
    assert "去相关不足" not in capsys.readouterr().out


def test_history_steps_mismatch_aborts_before_arena(tmp_path, monkeypatch, capsys):
    """两臂 history_steps 不一致 → 中止且不调用 arena、不给裁决（退出码非 0）。

    arena 的 _play_one 只用一个 cfg.model.history_steps 编码双方（rl/evaluate.py:113），
    异构历史下第一次前向就会因输入通道数不符崩掉；这里提前给中文原因。
    """
    import torch

    import rl.evaluate as evaluate

    g_ckpt = tmp_path / "g.pt"
    p_ckpt = tmp_path / "p.pt"
    g_ckpt.write_bytes(b"")
    p_ckpt.write_bytes(b"")

    hist = {str(g_ckpt): 2, str(p_ckpt): 4}
    monkeypatch.setattr(
        torch, "load",
        lambda p, **kw: {"model_config": {"history_steps": hist[str(p)]}})

    called = []
    monkeypatch.setattr(evaluate, "arena", lambda *a, **k: called.append(a))

    rc = ab_judge.main([
        "--gumbel-ckpt", str(g_ckpt), "--puct-ckpt", str(p_ckpt),
        "--device", "cpu", "--skip-metrics",
    ])
    text = capsys.readouterr().out
    assert rc == 1
    assert not called, "history_steps 不一致时不得进入对局"
    assert "history_steps 不一致" in text
    assert "[2, 4]" in text
    for v in (ab_judge.VERDICT_GO, ab_judge.VERDICT_TIE, ab_judge.VERDICT_KILL):
        assert v not in text, "没打过就不能出裁决"


# ═══════════════════════════════════════════════════════════════
# (b) 预注册判据边界
# ═══════════════════════════════════════════════════════════════
def test_verdict_go_boundary_closed_at_055():
    """0.55 含在 GO 一侧（闭端点）。"""
    assert ab_judge.verdict(0.55) == ab_judge.VERDICT_GO
    assert ab_judge.verdict(0.5501) == ab_judge.VERDICT_GO
    assert ab_judge.verdict(1.0) == ab_judge.VERDICT_GO
    assert ab_judge.verdict(0.5499) == ab_judge.VERDICT_TIE  # 紧贴下方仍是平手


def test_verdict_tie_boundary_closed_at_045():
    """平手区间 = [0.45, 0.55)：0.45 含在平手一侧。"""
    assert ab_judge.verdict(0.45) == ab_judge.VERDICT_TIE
    assert ab_judge.verdict(0.50) == ab_judge.VERDICT_TIE
    assert ab_judge.verdict(0.4501) == ab_judge.VERDICT_TIE


def test_verdict_kill_below_045():
    """< 0.45 关闭路线（开端点）。"""
    assert ab_judge.verdict(0.4499) == ab_judge.VERDICT_KILL
    assert ab_judge.verdict(0.30) == ab_judge.VERDICT_KILL
    assert ab_judge.verdict(0.0) == ab_judge.VERDICT_KILL


def test_verdict_endpoints_hit_exactly_with_real_score_formula():
    """200 盘下端点可被真实 arena 得分公式精确命中（浮点不偏移）。

    若 (w+0.5d)/n 与字面量 0.55/0.45 存在 1ulp 差，边界局面会静默落到相邻分支，
    因此直接用 rl.evaluate.score_stats 交叉验证，而不是只信 verdict 的字面比较。
    """
    from rl.evaluate import score_stats

    s_go, _ = score_stats(110, 0, 90)          # 110/200 = 0.55
    assert s_go == 0.55
    assert ab_judge.verdict(s_go) == ab_judge.VERDICT_GO

    s_go_draws, _ = score_stats(100, 20, 80)   # (100+10)/200 = 0.55
    assert s_go_draws == 0.55
    assert ab_judge.verdict(s_go_draws) == ab_judge.VERDICT_GO

    s_tie, _ = score_stats(90, 0, 110)         # 90/200 = 0.45
    assert s_tie == 0.45
    assert ab_judge.verdict(s_tie) == ab_judge.VERDICT_TIE

    s_kill, _ = score_stats(89, 0, 111)        # 89/200 = 0.445
    assert ab_judge.verdict(s_kill) == ab_judge.VERDICT_KILL


# ═══════════════════════════════════════════════════════════════
# (c) 曲线聚合
# ═══════════════════════════════════════════════════════════════
def _row(it, **kw) -> dict:
    """构造一行 metrics（值均为字符串，与 csv.DictReader 的产出一致）。"""
    base = {
        "iter": str(it), "policy_loss": "1.0", "value_loss": "0.1",
        "value_mae": "0.2", "entropy": "2.0", "red_winrate": "0.3",
        "draw_rate": "0.4", "arena_score": "0.5", "promoted": "False",
    }
    base.update({k: str(v) for k, v in kw.items()})
    return base


def test_summarize_metrics_bucketing_and_means():
    """window=3 分两桶；均值/晋升计数/range 均按桶内实际行计算。"""
    rows = [
        _row(0, policy_loss=1.0, entropy=3.0, promoted="True"),
        _row(1, policy_loss=2.0, entropy=2.0, promoted="False"),
        _row(2, policy_loss=3.0, entropy=1.0, promoted="true"),   # 大小写不敏感
        _row(3, policy_loss=4.0, entropy=0.5, promoted="1"),      # "1" 也算晋升
        _row(4, policy_loss=6.0, entropy=0.5, promoted=""),
    ]
    out = ab_judge.summarize_metrics(rows, window=3)

    assert [w["bucket"] for w in out] == [0, 1]
    assert out[0]["range"] == "0-2" and out[0]["n"] == 3
    assert out[1]["range"] == "3-4" and out[1]["n"] == 2
    assert out[0]["policy_loss"] == pytest.approx(2.0)
    assert out[1]["policy_loss"] == pytest.approx(5.0)
    assert out[0]["entropy"] == pytest.approx(2.0)
    assert out[0]["promoted"] == 2      # True + true
    assert out[1]["promoted"] == 1      # "1"，空串不算
    # 其余列走同一条聚合路径，抽查两项
    assert out[0]["value_mae"] == pytest.approx(0.2)
    assert out[0]["draw_rate"] == pytest.approx(0.4)


def test_summarize_metrics_missing_values_and_unsorted_input():
    """缺列/空串不参与均值；整桶全缺 → None；输入乱序不影响 range（取 min/max）。"""
    rows = [
        _row(2, arena_score=""),
        _row(0, arena_score="0.60"),
        _row(1, arena_score="0.40"),
    ]
    del rows[0]["value_mae"]            # 整列缺失（旧版 csv）
    out = ab_judge.summarize_metrics(rows, window=10)
    assert len(out) == 1
    assert out[0]["range"] == "0-2"     # 乱序输入仍给出正确区间
    assert out[0]["arena_score"] == pytest.approx(0.5)   # 只平均两行
    assert out[0]["value_mae"] == pytest.approx(0.2)     # 另两行仍有值

    all_missing = ab_judge.summarize_metrics(
        [_row(0, arena_score=""), _row(1, arena_score="")], window=10)
    assert all_missing[0]["arena_score"] is None


def test_summarize_metrics_rejects_bad_window():
    with pytest.raises(ValueError):
        ab_judge.summarize_metrics([_row(0)], window=0)


def test_load_metrics_dedups_and_sorts(tmp_path):
    """防御性：同 iter 多行保留最后一行，输出按 iter 升序（从零训练本不该出现，
    但手工续跑会让窗口均值双计）。"""
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "iter,entropy,promoted\n"
        "1,2.0,False\n"
        "0,3.0,False\n"
        "1,9.0,True\n",        # 重复 iter=1，保留这行
        encoding="utf-8",
    )
    rows = ab_judge.load_metrics(csv_path)
    assert [r["iter"] for r in rows] == ["0", "1"]
    assert rows[1]["entropy"] == "9.0"
    assert ab_judge.load_metrics(tmp_path / "nope.csv") == []


# ═══════════════════════════════════════════════════════════════
# (d) 端到端：曲线环节、缺文件报错、不写文件
# ═══════════════════════════════════════════════════════════════
def _write_metrics(path: Path, n: int, entropy0: float, promo_every: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["iter,policy_loss,value_loss,value_mae,entropy,red_winrate,draw_rate,"
             "arena_score,promoted"]
    for i in range(n):
        lines.append(f"{i},1.5,0.1,0.2,{entropy0 - i * 0.01},0.3,0.4,0.52,"
                     f"{'True' if i % promo_every == 0 else 'False'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_curve_compare_runs_and_writes_no_files(tmp_path, capsys):
    """--skip-arena 只跑曲线：正常输出两臂对比与末窗口小结，且不新建任何文件。"""
    g = tmp_path / "logs_ab_gumbel" / "metrics.csv"
    p = tmp_path / "logs_ab_puct" / "metrics.csv"
    _write_metrics(g, 40, entropy0=2.0, promo_every=5)    # 熵更低、晋升更多
    _write_metrics(p, 35, entropy0=3.0, promo_every=10)

    before = sorted(x.relative_to(tmp_path).as_posix() for x in tmp_path.rglob("*"))
    rc = ab_judge.main([
        "--gumbel-metrics", str(g), "--puct-metrics", str(p),
        "--window", "30", "--skip-arena",
    ])
    after = sorted(x.relative_to(tmp_path).as_posix() for x in tmp_path.rglob("*"))

    assert rc == 0
    assert before == after, "脚本必须只读，不得产生任何文件"
    text = capsys.readouterr().out
    assert "窗口 iter 0-29" in text and "窗口 iter 30-" in text
    assert "末窗口小结" in text
    assert "gumbel 更低" in text        # 熵：2.0 段 < 3.0 段
    assert "gumbel 更多" in text        # 晋升：每 5 迭代 vs 每 10 迭代
    assert "两臂迭代数不等" in text     # 40 vs 35 行，等算力前提告警
    assert ab_judge.VERDICT_GO not in text  # --skip-arena 不得给出裁决


def test_missing_files_report_error_and_nonzero_exit(tmp_path, capsys):
    """两臂文件缺失 → 中文报错 + 提示两臂是否跑完 + 退出码非 0。"""
    rc = ab_judge.main([
        "--gumbel-ckpt", str(tmp_path / "ckpts_ab_gumbel" / "best.pt"),
        "--puct-ckpt", str(tmp_path / "ckpts_ab_puct" / "best.pt"),
        "--gumbel-metrics", str(tmp_path / "logs_ab_gumbel" / "metrics.csv"),
        "--puct-metrics", str(tmp_path / "logs_ab_puct" / "metrics.csv"),
    ])
    assert rc == 1
    text = capsys.readouterr().out
    assert "找不到" in text
    assert "两臂是否都已跑完" in text
    assert "ckpts_ab_gumbel" in text and "logs_ab_puct" in text


def test_init_placeholder_best_is_fatal(tmp_path):
    """启动占位的随机初始网（meta.init=True / iteration=-1）必须 fatal，不得进对局。

    背景：rl/train.py 启动时就把随机模型写成 best.pt；零晋升的臂拿它裁决会给出
    方向完全错误的 score（对抗验证实测隐患）。
    """
    import io

    import torch

    init_ck = tmp_path / "best_init.pt"
    torch.save({"model_state": {}, "model_config": {"history_steps": 2},
                "iteration": -1, "meta": {"init": True}}, init_ck)
    ok_ck = tmp_path / "best_ok.pt"
    torch.save({"model_state": {}, "model_config": {"history_steps": 2},
                "iteration": 42, "meta": {}}, ok_ck)

    out = io.StringIO()
    _, fatal = ab_judge._probe_history_steps([str(init_ck), str(ok_ck)], out)
    assert fatal is True
    assert "随机初始网" in out.getvalue()

    out2 = io.StringIO()
    hist2, fatal2 = ab_judge._probe_history_steps([str(ok_ck)], out2)
    assert fatal2 is False and hist2 == 2
