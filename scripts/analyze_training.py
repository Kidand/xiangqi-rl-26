# -*- coding: utf-8 -*-
"""
训练日志分析脚本：读取 logs/metrics.csv（+ 可选 records 采样、可选 checkpoint 天梯），
输出一份可直接粘贴的诊断报告，用于制定后续训练计划。

用法（在训练目录、即 logs/ ckpts/ records/ 的父目录下运行）：

    # 基础分析（秒级，纯 stdlib）
    python scripts/analyze_training.py

    # 默认只分析最后一次回滚之后的训练段；也可选第 0 段
    python scripts/analyze_training.py --segment 0

    # 加终局类型分布（采样 records/selfplay/*.jsonl，约几十秒）
    python scripts/analyze_training.py --records

    # 加 checkpoint 天梯（多进程吃满全部 GPU 实打对局；衡量 Elo 增益斜率）
    python scripts/analyze_training.py --ladder --ladder-games 200 --ladder-sims 400 \
        --config configs/cloud.yaml

    # 全量 + 存文件
    python scripts/analyze_training.py --records --ladder --out report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ────────────────────────────────────────────────────────────────
# 工具
# ────────────────────────────────────────────────────────────────
def _f(v, default=None):
    """安全转 float（metrics.csv 缺失列为空字符串）。"""
    try:
        number = float(v)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "1.0", "yes")


def _pct(x, digits=1) -> str:
    return f"{x * 100:.{digits}f}%" if x is not None else "-"


def _num(x, digits=3) -> str:
    return f"{x:.{digits}f}" if x is not None else "-"


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _elo_from_score(score: float, games: int) -> float:
    """由对局得分估算 Elo 差（得分方视角），钳制避免 0/1 发散。"""
    eps = 1.0 / (2.0 * max(1, games))
    s = min(max(score, eps), 1.0 - eps)
    return 400.0 * math.log10(s / (1.0 - s))


# ────────────────────────────────────────────────────────────────
# metrics.csv
# ────────────────────────────────────────────────────────────────
def load_metric_segments(csv_path: Path) -> list[list[dict]]:
    """按文件行序读取，迭代号下降时开启新段；仅在同段内去重。

    回滚后的同号迭代属于不同训练轨迹，不能全局合并或按迭代号排序。
    同段内相同迭代号保留最后一行；缺失、非有限或非整数迭代号跳过。
    """
    if not csv_path.exists():
        return []
    segments: list[list[dict]] = []
    by_iter: dict[int, dict] = {}
    previous_iter = None
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            it = _f(row.get("iter"))
            if it is None or not it.is_integer():
                continue
            it = int(it)
            if previous_iter is not None and it < previous_iter:
                segments.append(list(by_iter.values()))
                by_iter = {}
            by_iter[it] = row
            previous_iter = it
    if by_iter:
        segments.append(list(by_iter.values()))
    return segments


def _segment_index(segments: list[list[dict]], segment: int) -> int:
    if segment < -1 or segment >= len(segments) or not segments:
        raise IndexError(f"segment={segment} 越界：共 {len(segments)} 段，使用 0 起的段号或 -1（最后段）")
    return len(segments) - 1 if segment == -1 else segment


def load_metrics(csv_path: Path, segment: int = -1) -> list[dict]:
    """返回指定训练段（0 起；-1 为最后段），默认不混合回滚前后的日志。"""
    segments = load_metric_segments(csv_path)
    if not segments and segment == -1:
        return []
    return segments[_segment_index(segments, segment)]


def window_summary(rows: list[dict], window: int) -> list[dict]:
    """按迭代号分桶汇总（每 window 个迭代一行）。"""
    out = []
    buckets: dict[int, list[dict]] = {}
    for r in rows:
        b = int(_f(r["iter"])) // window
        buckets.setdefault(b, []).append(r)
    for b in sorted(buckets):
        rs = buckets[b]
        out.append({
            "range": f"{int(_f(rs[0]['iter']))}-{int(_f(rs[-1]['iter']))}",
            "n": len(rs),
            "draw": _mean(_f(r.get("draw_rate")) for r in rs),
            "red": _mean(_f(r.get("red_winrate")) for r in rs),
            "black": _mean(_f(r.get("black_winrate")) for r in rs),
            "plies": _mean(_f(r.get("avg_plies")) for r in rs),
            "resign": _mean(_f(r.get("resign_rate")) for r in rs),
            "loss": _mean(_f(r.get("loss")) for r in rs),
            "p_loss": _mean(_f(r.get("policy_loss")) for r in rs),
            "v_loss": _mean(_f(r.get("value_loss")) for r in rs),
            "v_mae": _mean(_f(r.get("value_mae")) for r in rs),
            "entropy": _mean(_f(r.get("entropy")) for r in rs),
            "target_entropy": _mean(_f(r.get("target_entropy")) for r in rs),
            "policy_kl": _mean(_f(r.get("policy_kl")) for r in rs),
            "lr": _mean(_f(r.get("lr")) for r in rs),
            "arena": _mean(_f(r.get("arena_score")) for r in rs),
            "promo": sum(1 for r in rs if _truthy(r.get("promoted"))),
            "sp_sec": _mean(_f(r.get("selfplay_sec")) for r in rs),
            "tr_sec": _mean(_f(r.get("train_sec")) for r in rs),
        })
    return out


def report_metrics(rows: list[dict], window: int, recent: int, lines: list[str]) -> None:
    first, last = int(_f(rows[0]["iter"])), int(_f(rows[-1]["iter"]))
    lines.append(f"## 1. 总览")
    lines.append(f"- 数据范围：iter {first} .. {last}（共 {len(rows)} 行）")
    lr_last = _f(rows[-1].get("lr"))
    lines.append(
        f"- 最新迭代：buffer={int(_f(rows[-1].get('buffer_size')) or 0):,}  "
        f"lr={_num(lr_last, 6)}  loss={_num(_f(rows[-1].get('loss')))}  "
        f"value_mae={_num(_f(rows[-1].get('value_mae')))}  "
        f"draw={_pct(_f(rows[-1].get('draw_rate')))}"
    )
    total_sec = [_f(r.get("total_sec")) for r in rows if _f(r.get("total_sec"))]
    if total_sec:
        lines.append(f"- 每迭代耗时：中位 {statistics.median(total_sec):.0f}s，最近 10 迭代均值 {_mean(total_sec[-10:]):.0f}s")
    lines.append("- 策略指标：entropy=预测熵 H(p)，target_entropy=目标熵 H(π)，"
                 "policy_kl=KL(π‖p)。缺失值显示为 -（未知），不能用 CE-H(p) 推算 KL。")

    lines.append("")
    lines.append(f"## 2. 分窗口汇总（每 {window} 迭代）")
    hdr = ("| iter 段 | 和棋 | 红胜 | 黑胜 | 均步数 | 认输率 | loss | p_loss | v_loss | v_mae "
           "| H(p) | H(π) | policy_kl | lr | arena均分 | 晋升 | selfplay_s |")
    lines.append(hdr)
    lines.append("|" + "---|" * 17)
    for w in window_summary(rows, window):
        lines.append(
            f"| {w['range']} | {_pct(w['draw'])} | {_pct(w['red'])} | {_pct(w['black'])} "
            f"| {_num(w['plies'], 0)} | {_pct(w['resign'])} | {_num(w['loss'])} | {_num(w['p_loss'])} "
            f"| {_num(w['v_loss'])} | {_num(w['v_mae'])} | {_num(w['entropy'])} "
            f"| {_num(w['target_entropy'])} | {_num(w['policy_kl'], 5)} | {_num(w['lr'], 6)} "
            f"| {_num(w['arena'])} | {w['promo']}/{w['n']} | {_num(w['sp_sec'], 0)} |"
        )

    # 晋升动态
    promo_iters = [int(_f(r["iter"])) for r in rows if _truthy(r.get("promoted"))]
    lines.append("")
    lines.append("## 3. 晋升动态")
    lines.append(f"- 总晋升 {len(promo_iters)} 次 / {len(rows)} 迭代")
    if promo_iters:
        gaps = [b - a for a, b in zip(promo_iters, promo_iters[1:])]
        lines.append(f"- 最近一次晋升：iter {promo_iters[-1]}（距最新 {last - promo_iters[-1]} 个迭代）")
        if gaps:
            lines.append(f"- 晋升间隔：中位 {statistics.median(gaps):.0f}，最近 5 次间隔 {gaps[-5:]}")
    recent_scores = [_f(r.get("arena_score")) for r in rows[-50:] if _f(r.get("arena_score")) is not None]
    if recent_scores:
        lines.append(
            f"- 最近 50 迭代 arena_score：min {min(recent_scores):.3f} / "
            f"中位 {statistics.median(recent_scores):.3f} / max {max(recent_scores):.3f}"
        )

    lines.append("")
    lines.append(f"## 4. 最近 {recent} 迭代明细")
    lines.append("| iter | 和棋 | 均步数 | loss | v_mae | H(p) | H(π) | policy_kl | arena | 晋升 |")
    lines.append("|" + "---|" * 10)
    for r in rows[-recent:]:
        lines.append(
            f"| {int(_f(r['iter']))} | {_pct(_f(r.get('draw_rate')))} | {_num(_f(r.get('avg_plies')), 0)} "
            f"| {_num(_f(r.get('loss')))} | {_num(_f(r.get('value_mae')))} | {_num(_f(r.get('entropy')))} "
            f"| {_num(_f(r.get('target_entropy')))} | {_num(_f(r.get('policy_kl')), 5)} "
            f"| {_num(_f(r.get('arena_score')))} | {'Y' if _truthy(r.get('promoted')) else ''} |"
        )


# ────────────────────────────────────────────────────────────────
# records 终局类型采样
# ────────────────────────────────────────────────────────────────
def report_records(records_dir: Path, n_files: int, n_lines: int, lines: list[str]) -> None:
    lines.append("")
    lines.append("## 5. 终局类型分布（records 采样）")
    files = sorted(records_dir.glob("iter_*.jsonl"))
    if not files:
        lines.append(f"- 未找到 {records_dir}/iter_*.jsonl，跳过")
        return
    # 均匀取 n_files 个文件（必含最后一个）
    if len(files) <= n_files:
        picked = files
    else:
        step = (len(files) - 1) / (n_files - 1)
        picked = [files[round(i * step)] for i in range(n_files - 1)] + [files[-1]]
    # 必须读整个文件：jsonl 按对局"完成顺序"追加，短局先完成先落盘，
    # 只读前 N 行会系统性漏掉文件尾部的长局/和棋（曾据此得出与
    # metrics.csv 的 draw_rate/avg_plies 自相矛盾的分布）。
    lines.append(f"- 采样 {len(picked)} 个文件（每文件全量统计，安全上限 {n_lines} 局）")
    known = ("checkmate", "stalemate", "repetition", "perpetual_check",
             "max_moves", "no_capture", "resign")
    lines.append("| 文件 | 局数 | " + " | ".join(known) + " | 其他 | plies p50/p90/max |")
    lines.append("|" + "---|" * (len(known) + 4))
    for p in picked:
        term: dict[str, int] = {}
        plies: list[int] = []
        n = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if n >= n_lines:   # 仅防极端超大文件，正常配置远达不到
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                term[str(rec.get("termination"))] = term.get(str(rec.get("termination")), 0) + 1
                if rec.get("plies") is not None:
                    plies.append(int(rec["plies"]))
                n += 1
        if n == 0:
            continue
        other = sum(v for k, v in term.items() if k not in known)
        cells = " | ".join(_pct(term.get(k, 0) / n) for k in known)
        p50 = statistics.median(plies) if plies else 0
        p90 = sorted(plies)[int(len(plies) * 0.9)] if plies else 0
        pmax = max(plies) if plies else 0
        lines.append(f"| {p.name} | {n} | {cells} | {_pct(other / n)} | {p50:.0f}/{p90}/{pmax} |")


# ────────────────────────────────────────────────────────────────
# checkpoint 天梯（多进程多 GPU 并行）
# ────────────────────────────────────────────────────────────────
_WORKER_DEVICE = None


def _ladder_init(dev_queue) -> None:
    """Pool initializer：每个 worker 进程启动时领取一个固定设备。

    设备绑定在进程而非任务上——每进程只在一张卡上建 CUDA 上下文，
    避免 48 worker × 8 卡交叉产生大量冗余上下文显存。
    """
    global _WORKER_DEVICE
    _WORKER_DEVICE = dev_queue.get()
    import torch

    torch.set_num_threads(1)  # 瓶颈在 Python MCTS，防多 worker CPU 超订


def _ladder_worker(task: dict) -> dict:
    """子进程：加载两个 checkpoint，在本 worker 绑定的设备上打一块对局，返回计数。"""
    from rl.evaluate import _play_one
    from rl.model import load_checkpoint
    from rl.selfplay import build_eval_fn

    dev = _WORKER_DEVICE or "cpu"
    new_model, new_ck = load_checkpoint(task["new_ckpt"], device=dev)
    old_model, _ = load_checkpoint(task["old_ckpt"], device=dev)
    new_model.eval()
    old_model.eval()
    new_eval = build_eval_fn(new_model, dev)
    old_eval = build_eval_fn(old_model, dev)

    cfg = task["cfg"]
    # history_steps 以 checkpoint 实际值为准（防止 cfg 与权重不一致）
    hist = (new_ck.get("model_config") or {}).get("history_steps")
    if hist is not None:
        cfg.model.history_steps = int(hist)

    import numpy as np

    rng = np.random.default_rng(task["seed"])
    w = d = l = 0
    rw = rd = rl_ = bw = bd = bl = 0
    for new_is_red in [True] * task["n_red"] + [False] * task["n_black"]:
        result = _play_one(new_eval, old_eval, new_is_red, cfg, rng)
        if result == "1/2-1/2":
            d += 1
            rd, bd = (rd + 1, bd) if new_is_red else (rd, bd + 1)
        elif (result == "1-0") == new_is_red:
            w += 1
            rw, bw = (rw + 1, bw) if new_is_red else (rw, bw + 1)
        else:
            l += 1
            rl_, bl = (rl_ + 1, bl) if new_is_red else (rl_, bl + 1)
    return {"pair": task["pair"], "wins": w, "draws": d, "losses": l,
            "red_w": rw, "red_d": rd, "red_l": rl_,
            "black_w": bw, "black_d": bd, "black_l": bl}


def _elo_ci95(wins: int, draws: int, losses: int) -> float:
    """Elo 差的 ±95% 置信半径（对局得分 1/0.5/0 的经验方差 + delta 法）。"""
    n = wins + draws + losses
    if n < 2:
        return float("inf")
    s = (wins + 0.5 * draws) / n
    var = (wins * (1 - s) ** 2 + draws * (0.5 - s) ** 2 + losses * s ** 2) / n
    se = math.sqrt(var / n)
    eps = 1.0 / (2.0 * n)
    s_c = min(max(s, eps), 1.0 - eps)
    # dElo/ds = 400 / (ln10 · s(1-s))
    return 1.96 * se * 400.0 / (math.log(10) * s_c * (1.0 - s_c))


def report_ladder(ckpt_dir: Path, offsets: list[int], games: int, sims: int,
                  device: str, workers: int, config_yaml: str | None,
                  lines: list[str]) -> None:
    import multiprocessing as mp

    lines.append("")
    ckpts = {}
    for p in sorted(ckpt_dir.glob("iter_*.pt")):
        try:
            ckpts[int(p.stem.split("_")[1])] = p
        except (IndexError, ValueError):
            continue
    if not ckpts:
        lines.append("## 6. checkpoint 天梯")
        lines.append(f"- 未找到 {ckpt_dir}/iter_*.pt，跳过")
        return

    from rl.config import Config          # 延迟导入（需要 torch）
    import torch

    # 设备列表与 worker 数：默认吃满全部 GPU，每卡多 worker（瓶颈在 CPU 端 MCTS）
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    from rl.selfplay import effective_cpu_count  # 容器感知（cgroup 配额/亲和性）

    cpu_n = effective_cpu_count()
    if workers <= 0:
        workers = min(max(4, cpu_n - 4), len(devices) * 6 if devices[0] != "cpu" else 8)

    cfg = Config.from_yaml(config_yaml) if config_yaml else Config()
    cfg.arena.arena_games = games
    cfg.arena.arena_sims = sims
    cfg.train.num_gpus = len(devices) if devices[0] != "cpu" else 0

    latest = max(ckpts)
    pairs = []
    for off in offsets:
        cands = [k for k in ckpts if k <= latest - off]
        if cands and max(cands) != latest:
            pairs.append((off, max(cands)))

    # 全局任务池：所有配对的对局切成 ~4 盘的小块混合调度，避免尾部 GPU 闲置
    tasks = []
    for pi, (off, old) in enumerate(pairs):
        red_total = games - games // 2
        black_total = games // 2
        n_chunks = max(1, min(workers, math.ceil(games / 4)))
        for c in range(n_chunks):
            n_red = red_total * (c + 1) // n_chunks - red_total * c // n_chunks
            n_black = black_total * (c + 1) // n_chunks - black_total * c // n_chunks
            if n_red + n_black == 0:
                continue
            tasks.append({
                "pair": (latest, old), "new_ckpt": str(ckpts[latest]), "old_ckpt": str(ckpts[old]),
                "n_red": n_red, "n_black": n_black,
                "seed": 20260704 + off * 1000 + c, "cfg": cfg,
            })

    lines.append(f"## 6. checkpoint 天梯（每对 {games} 盘 × {sims} sims，"
                 f"{workers} workers × {len(devices)} 设备并行）")
    lines.append(f"- 基准：iter_{latest:04d}（最新）为\"新\"方；score 是本批对局得分，"
                 "棋力差异需结合置信区间与独立复验判断。")
    print(f"[ladder] {len(pairs)} 对 × {games} 盘 → {len(tasks)} 个任务块，"
          f"workers={workers}，devices={devices}", file=sys.stderr)

    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    dev_queue = ctx.Queue()
    for i in range(workers):
        dev_queue.put(devices[i % len(devices)])   # worker 均匀铺满各卡
    agg: dict = {}
    with ctx.Pool(processes=workers, initializer=_ladder_init, initargs=(dev_queue,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_ladder_worker, tasks), 1):
            a = agg.setdefault(r["pair"], {k: 0 for k in r if k != "pair"})
            for k in a:
                a[k] += r[k]
            print(f"[ladder] {i}/{len(tasks)} 块完成", file=sys.stderr)
    wall = time.monotonic() - t0

    lines.append("| 对阵 | 新胜 | 和 | 新负 | score | 执红分 | 执黑分 | Elo 增益 (±95%CI) |")
    lines.append("|" + "---|" * 8)
    for (new_it, old_it), a in sorted(agg.items(), key=lambda kv: -kv[0][1]):
        n = a["wins"] + a["draws"] + a["losses"]
        s = (a["wins"] + 0.5 * a["draws"]) / n if n else 0.0
        red_n = a["red_w"] + a["red_d"] + a["red_l"]
        black_n = a["black_w"] + a["black_d"] + a["black_l"]
        red_s = (a["red_w"] + 0.5 * a["red_d"]) / red_n if red_n else 0.0
        black_s = (a["black_w"] + 0.5 * a["black_d"]) / black_n if black_n else 0.0
        elo = _elo_from_score(s, n)
        ci = _elo_ci95(a["wins"], a["draws"], a["losses"])
        lines.append(
            f"| iter_{new_it:04d} vs iter_{old_it:04d} | {a['wins']} | {a['draws']} | {a['losses']} "
            f"| {s:.3f} | {red_s:.3f} | {black_s:.3f} | {elo:+.0f} (±{ci:.0f}) |"
        )
    lines.append(f"- 总耗时 {wall:.0f}s（{len(tasks)} 块并行）")


# ────────────────────────────────────────────────────────────────
# 自动观察（启发式，仅陈述事实供人判断）
# ────────────────────────────────────────────────────────────────
def report_flags(rows: list[dict], lines: list[str]) -> None:
    lines.append("")
    lines.append("## 7. 自动观察")
    flags = []

    def tail_mean(key, n):
        return _mean(_f(r.get(key)) for r in rows[-n:])

    def head_mean(key, a, b):
        return _mean(_f(r.get(key)) for r in rows[a:b])

    draw = tail_mean("draw_rate", 25)
    if draw is not None and draw > 0.55:
        flags.append(f"最近 {min(25, len(rows))} 迭代和棋率 {_pct(draw)} 偏高——"
                     "先检查终局类型、对局长度与开局重复率；保持训练目标与搜索的零和语义一致")
    promo_recent = sum(1 for r in rows[-50:] if _truthy(r.get("promoted")))
    if promo_recent <= 2:
        flags.append(f"最近 {min(50, len(rows))} 迭代晋升 {promo_recent} 次——"
                     "晋升数受移动对手与门控噪声影响，不能据此认定棋力停滞；需对冻结基线独立复验")
    recent_rows = rows[-25:]
    kl_values = [_f(r.get("policy_kl")) for r in recent_rows]
    kl_values = [v for v in kl_values if v is not None]
    target_entropies = [_f(r.get("target_entropy")) for r in recent_rows]
    target_entropies = [v for v in target_entropies if v is not None]
    if kl_values:
        flags.append(f"真实 policy_kl 均值 {_num(_mean(kl_values), 5)}"
                     f"（最近 {len(recent_rows)} 行中 {len(kl_values)} 行有效）；"
                     "它衡量当前自博弈策略目标的拟合误差，不能直接判断棋力")
    else:
        flags.append("真实 policy_kl 未记录，KL 未知；policy_loss 接近预测熵 entropy 不能说明目标已学完")
    if target_entropies:
        flags.append(f"目标熵 target_entropy 均值 {_num(_mean(target_entropies), 5)}"
                     f"（最近 {len(recent_rows)} 行中 {len(target_entropies)} 行有效）")
    else:
        flags.append("目标熵 target_entropy 未记录（未知）；预测熵 H(p) 不能替代目标熵 H(π)")
    ent_now = tail_mean("entropy", 25)
    ent_prev = head_mean("entropy", -100, -75) if len(rows) >= 100 else None
    if ent_now is not None and ent_prev is not None and ent_now < ent_prev * 0.7:
        flags.append(f"预测熵从 {_num(ent_prev)} 降到 {_num(ent_now)}（-30%+）——"
                     "网络预测分布更集中，需结合实际走子分布与开局重复率判断探索情况")
    vmae_now = tail_mean("value_mae", 25)
    vmae_prev = head_mean("value_mae", -100, -75) if len(rows) >= 100 else None
    if vmae_now is not None and vmae_prev is not None and vmae_now > vmae_prev * 1.15:
        flags.append(f"value_mae 从 {_num(vmae_prev)} 升到 {_num(vmae_now)}——价值头拟合变差，检查和棋/长局占比")
    lr_now = _f(rows[-1].get("lr"))
    if lr_now is not None and lr_now <= 1.1e-4:
        flags.append(f"lr 已到退火地板（{_num(lr_now, 6)}）——续训需扩大 total_iterations（调度按 iter/total 取进度）")
    if not flags:
        flags.append("无异常信号")
    for f in flags:
        lines.append(f"- {f}")


# ────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练日志分析（在 logs/ ckpts/ records/ 的父目录运行）")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--segment", type=int, default=-1,
                    help="按 iter 回退切分日志，选择 0 起的训练段；默认 -1（最后段）")
    ap.add_argument("--records-dir", default="records/selfplay")
    ap.add_argument("--ckpt-dir", default="ckpts")
    ap.add_argument("--window", type=int, default=25, help="分桶窗口大小（迭代数）")
    ap.add_argument("--recent", type=int, default=20, help="明细表显示最近 N 迭代")
    ap.add_argument("--records", action="store_true", help="采样 records 统计终局类型（较慢）")
    ap.add_argument("--records-files", type=int, default=8)
    ap.add_argument("--records-lines", type=int, default=100000,
                    help="每文件统计的对局数安全上限（必须≥每迭代对局数，否则会漏掉写盘靠后的长局）")
    ap.add_argument("--ladder", action="store_true", help="checkpoint 天梯实战（多进程多 GPU 并行，需 torch）")
    ap.add_argument("--ladder-offsets", default="50,100,200", help="与最新 checkpoint 的迭代差，逗号分隔")
    ap.add_argument("--ladder-games", type=int, default=200, help="每对局数（并行后可放心加大以缩 CI）")
    ap.add_argument("--ladder-sims", type=int, default=400)
    ap.add_argument("--ladder-workers", type=int, default=0,
                    help="并行 worker 进程数；0=自动（GPU 数×6，受 CPU 核数约束）")
    ap.add_argument("--config", default=None,
                    help="训练 yaml（如 configs/cloud.yaml），使天梯的 c_puct/draw_penalty 等与训练一致")
    ap.add_argument("--device", default="auto", help="ladder 推理设备 auto/cuda/cpu")
    ap.add_argument("--out", default=None, help="同时写入文件")
    args = ap.parse_args(argv)

    lines: list[str] = []
    lines.append("# 训练日志分析报告")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}   参数：window={args.window} "
                 f"records={args.records} ladder={args.ladder}")
    lines.append("")

    segments = load_metric_segments(Path(args.log_dir) / "metrics.csv")
    if not segments:
        print(f"错误：找不到或无法解析 {args.log_dir}/metrics.csv（请在训练目录下运行）", file=sys.stderr)
        return 1
    try:
        selected_segment = _segment_index(segments, args.segment)
    except IndexError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    rows = segments[selected_segment]
    lines.append(f"训练分段：共 {len(segments)} 段，已选 segment={selected_segment}"
                 "（段号从 0 开始；每次 iter 下降开启新段）。")
    lines.append("")

    report_metrics(rows, args.window, args.recent, lines)
    if args.records:
        report_records(Path(args.records_dir), args.records_files, args.records_lines, lines)
    if args.ladder:
        offsets = [int(x) for x in args.ladder_offsets.split(",") if x.strip()]
        report_ladder(Path(args.ckpt_dir), offsets, args.ladder_games,
                      args.ladder_sims, args.device, args.ladder_workers,
                      args.config, lines)
    report_flags(rows, lines)
    lines.append("")
    lines.append("（请把以上完整报告粘贴回来，用于制定后续训练计划。）")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n已写入 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
