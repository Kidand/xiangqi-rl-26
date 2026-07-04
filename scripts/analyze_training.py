# -*- coding: utf-8 -*-
"""
训练日志分析脚本：读取 logs/metrics.csv（+ 可选 records 采样、可选 checkpoint 天梯），
输出一份可直接粘贴的诊断报告，用于制定后续训练计划。

用法（在训练目录、即 logs/ ckpts/ records/ 的父目录下运行）：

    # 基础分析（秒级，纯 stdlib）
    python scripts/analyze_training.py

    # 加终局类型分布（采样 records/selfplay/*.jsonl，约几十秒）
    python scripts/analyze_training.py --records

    # 加 checkpoint 天梯（加载模型实打对局，较慢；衡量 Elo 增益斜率）
    python scripts/analyze_training.py --ladder --ladder-games 60 --ladder-sims 400

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
        return float(v)
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
def load_metrics(csv_path: Path) -> list[dict]:
    """读取 metrics.csv；resume 重跑产生的重复迭代号只保留最后一行。"""
    if not csv_path.exists():
        return []
    by_iter: dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            it = _f(row.get("iter"))
            if it is None:
                continue
            by_iter[int(it)] = row
    return [by_iter[k] for k in sorted(by_iter)]


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

    lines.append("")
    lines.append(f"## 2. 分窗口汇总（每 {window} 迭代）")
    hdr = ("| iter 段 | 和棋 | 红胜 | 黑胜 | 均步数 | 认输率 | loss | p_loss | v_loss | v_mae "
           "| entropy | lr | arena均分 | 晋升 | selfplay_s |")
    lines.append(hdr)
    lines.append("|" + "---|" * 15)
    for w in window_summary(rows, window):
        lines.append(
            f"| {w['range']} | {_pct(w['draw'])} | {_pct(w['red'])} | {_pct(w['black'])} "
            f"| {_num(w['plies'], 0)} | {_pct(w['resign'])} | {_num(w['loss'])} | {_num(w['p_loss'])} "
            f"| {_num(w['v_loss'])} | {_num(w['v_mae'])} | {_num(w['entropy'])} | {_num(w['lr'], 6)} "
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
    lines.append("| iter | 和棋 | 均步数 | loss | v_mae | entropy | arena | 晋升 |")
    lines.append("|" + "---|" * 8)
    for r in rows[-recent:]:
        lines.append(
            f"| {int(_f(r['iter']))} | {_pct(_f(r.get('draw_rate')))} | {_num(_f(r.get('avg_plies')), 0)} "
            f"| {_num(_f(r.get('loss')))} | {_num(_f(r.get('value_mae')))} | {_num(_f(r.get('entropy')))} "
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
    lines.append(f"- 采样 {len(picked)} 个文件 × 每文件至多 {n_lines} 局")
    lines.append("| 文件 | 局数 | checkmate | repetition | max_moves | no_capture | resign | 其他 | plies p50/p90 |")
    lines.append("|" + "---|" * 9)
    for p in picked:
        term: dict[str, int] = {}
        plies: list[int] = []
        n = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if n >= n_lines:
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
        known = ("checkmate", "repetition", "max_moves", "no_capture", "resign")
        other = sum(v for k, v in term.items() if k not in known)
        cells = " | ".join(_pct(term.get(k, 0) / n) for k in known)
        p50 = statistics.median(plies) if plies else 0
        p90 = sorted(plies)[int(len(plies) * 0.9)] if plies else 0
        lines.append(f"| {p.name} | {n} | {cells} | {_pct(other / n)} | {p50:.0f}/{p90} |")


# ────────────────────────────────────────────────────────────────
# checkpoint 天梯
# ────────────────────────────────────────────────────────────────
def report_ladder(ckpt_dir: Path, offsets: list[int], games: int, sims: int,
                  device: str, lines: list[str]) -> None:
    lines.append("")
    lines.append(f"## 6. checkpoint 天梯（每对 {games} 盘 × {sims} sims）")
    ckpts = {}
    for p in sorted(ckpt_dir.glob("iter_*.pt")):
        try:
            ckpts[int(p.stem.split("_")[1])] = p
        except (IndexError, ValueError):
            continue
    if not ckpts:
        lines.append(f"- 未找到 {ckpt_dir}/iter_*.pt，跳过")
        return

    from rl.config import Config          # 延迟导入（需要 torch）
    from rl.evaluate import arena

    latest = max(ckpts)
    cfg = Config()
    cfg.arena.arena_games = games
    cfg.arena.arena_sims = sims
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.train.num_gpus = 1 if device == "cuda" else 0

    lines.append(f"- 基准：iter_{latest:04d}（最新）为\"新\"方；score>0.5 即最新更强")
    lines.append("| 对阵 | 新胜 | 和 | 新负 | score | Elo 增益估计 | 耗时 |")
    lines.append("|" + "---|" * 7)
    for off in offsets:
        target = latest - off
        # 找不大于 target 的最近存在的 checkpoint
        cands = [k for k in ckpts if k <= target]
        if not cands or max(cands) == latest:
            lines.append(f"| iter_{latest:04d} vs iter_{target:04d} | - | - | - | 无该检查点 | - | - |")
            continue
        old = max(cands)
        cfg.seed = 20260704 + off
        t0 = time.monotonic()
        st = arena(str(ckpts[latest]), str(ckpts[old]), cfg, device=device)
        dt = time.monotonic() - t0
        elo = _elo_from_score(st["score"], games)
        lines.append(
            f"| iter_{latest:04d} vs iter_{old:04d} | {st['wins']} | {st['draws']} | {st['losses']} "
            f"| {st['score']:.3f} | {elo:+.0f} | {dt:.0f}s |"
        )
        print(f"[ladder] iter_{latest:04d} vs iter_{old:04d}: "
              f"{st['wins']}W/{st['draws']}D/{st['losses']}L score={st['score']:.3f} ({dt:.0f}s)",
              file=sys.stderr)


# ────────────────────────────────────────────────────────────────
# 自动观察（启发式，仅陈述事实供人判断）
# ────────────────────────────────────────────────────────────────
def report_flags(rows: list[dict], lines: list[str]) -> None:
    lines.append("")
    lines.append("## 7. 自动观察")
    flags = []
    last = int(_f(rows[-1]["iter"]))

    def tail_mean(key, n):
        return _mean(_f(r.get(key)) for r in rows[-n:])

    def head_mean(key, a, b):
        return _mean(_f(r.get(key)) for r in rows[a:b])

    draw = tail_mean("draw_rate", 25)
    if draw is not None and draw > 0.55:
        flags.append(f"最近 25 迭代和棋率 {_pct(draw)} 偏高——考虑加大 draw_penalty 或提高 temp_final")
    promo_recent = sum(1 for r in rows[-50:] if _truthy(r.get("promoted")))
    if promo_recent <= 2:
        flags.append(f"最近 50 迭代仅晋升 {promo_recent} 次——同配方接近瓶颈或 arena 噪声淹没增益（建议加大 arena_games）")
    ent_now, ent_prev = tail_mean("entropy", 25), head_mean("entropy", -100, -75) if len(rows) >= 100 else (None,)
    if isinstance(ent_prev, tuple):
        ent_prev = None
    if ent_now is not None and ent_prev is not None and ent_now < ent_prev * 0.7:
        flags.append(f"策略熵从 {_num(ent_prev)} 降到 {_num(ent_now)}（-30%+）——探索收窄，注意开局多样性")
    vmae_now, vmae_prev = tail_mean("value_mae", 25), head_mean("value_mae", -100, -75) if len(rows) >= 100 else (None,)
    if isinstance(vmae_prev, tuple):
        vmae_prev = None
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
    ap.add_argument("--records-dir", default="records/selfplay")
    ap.add_argument("--ckpt-dir", default="ckpts")
    ap.add_argument("--window", type=int, default=25, help="分桶窗口大小（迭代数）")
    ap.add_argument("--recent", type=int, default=20, help="明细表显示最近 N 迭代")
    ap.add_argument("--records", action="store_true", help="采样 records 统计终局类型（较慢）")
    ap.add_argument("--records-files", type=int, default=8)
    ap.add_argument("--records-lines", type=int, default=500)
    ap.add_argument("--ladder", action="store_true", help="checkpoint 天梯实战（慢，需 torch）")
    ap.add_argument("--ladder-offsets", default="50,100,200", help="与最新 checkpoint 的迭代差，逗号分隔")
    ap.add_argument("--ladder-games", type=int, default=60)
    ap.add_argument("--ladder-sims", type=int, default=400)
    ap.add_argument("--device", default="auto", help="ladder 推理设备 auto/cuda/cpu")
    ap.add_argument("--out", default=None, help="同时写入文件")
    args = ap.parse_args(argv)

    lines: list[str] = []
    lines.append("# 训练日志分析报告")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}   参数：window={args.window} "
                 f"records={args.records} ladder={args.ladder}")
    lines.append("")

    rows = load_metrics(Path(args.log_dir) / "metrics.csv")
    if not rows:
        print(f"错误：找不到或无法解析 {args.log_dir}/metrics.csv（请在训练目录下运行）", file=sys.stderr)
        return 1

    report_metrics(rows, args.window, args.recent, lines)
    if args.records:
        report_records(Path(args.records_dir), args.records_files, args.records_lines, lines)
    if args.ladder:
        offsets = [int(x) for x in args.ladder_offsets.split(",") if x.strip()]
        report_ladder(Path(args.ckpt_dir), offsets, args.ladder_games,
                      args.ladder_sims, args.device, lines)
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
