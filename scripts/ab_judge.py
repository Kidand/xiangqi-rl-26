# -*- coding: utf-8 -*-
"""A/B 终点裁决：Gumbel 臂 vs PUCT 臂（从零、小尺度、等算力单变量对照）。

配套 configs/ab_scratch_gumbel.yaml / configs/ab_scratch_puct.yaml —— 两臂唯一差异
是 mcts 段与输出目录，各写 ckpts_ab_*/logs_ab_*/records_ab_*，不碰主 ckpts/。

本脚本干两件事，都只读不写（全程不落任何文件）：

1. **对抗**（可 --skip-arena 跳过）：200 盘去相关 arena（temp_moves=16、temperature=0.8）
   直接对打两臂的 best.pt，按预注册判据给三选一结论。
2. **曲线对比**（可 --skip-metrics 跳过）：并排读两臂 metrics.csv，每 30 迭代一窗口。

⚠ 视角（本脚本唯一的大坑）：``rl.evaluate.arena(new_ckpt, best_ckpt, ...)`` 返回的
wins/draws/losses/score 全部是**第一参数**的视角。这里第一参数恒为 **gumbel 臂**，
故所有 W/D/L 与 score 均为 gumbel 视角：score > 0.5 = gumbel 更强。视角搞反 = 决策
反向，所以输出的每一行分数都带"gumbel 视角"标注，且 tests/test_ab_judge.py 用 mock
钉死了调用方向。

预注册判据（跑前写死，见两个 yaml 的文件头；勿事后挪门柱）：
    gumbel 臂 score >= 0.55            → GO 全尺寸从零 Gumbel
    0.45 <= score <  0.55              → 平手，维持阶段六路线
    score < 0.45                       → 关闭从零 Gumbel 路线

用法（在训练目录、即 ckpts_ab_*/logs_ab_* 的父目录下运行；零参数即默认全量）：

    python scripts/ab_judge.py
    python scripts/ab_judge.py --games 400 --sims 400        # 加大样本缩 CI
    python scripts/ab_judge.py --skip-arena                  # 只看曲线（无需 GPU）
    python scripts/ab_judge.py --skip-metrics --device cpu   # 只跑对抗
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ────────────────────────────────────────────────────────────────
# 预注册判据（常量集中于此，改动 = 挪门柱，需在报告里显式声明）
# ────────────────────────────────────────────────────────────────
GO_THRESHOLD = 0.55       # 含端点：score == 0.55 判 GO
TIE_THRESHOLD = 0.45      # 含端点：score == 0.45 判平手
DIVERSITY_WARN = 0.70     # distinct_openings/games 低于此值 → 去相关不足告警

VERDICT_GO = "GO 全尺寸：启动全尺寸从零 Gumbel 训练"
VERDICT_TIE = "平手：维持阶段六路线（PUCT + π 锐化配方）"
VERDICT_KILL = "关闭从零 Gumbel 路线"

PREREG_NOTE = (
    "本判据于跑前预注册（见 configs/ab_scratch_*.yaml 文件头），勿事后挪门柱。"
)


def verdict(score: float) -> str:
    """预注册三分判据。

    ``score`` 必须是 **gumbel 臂视角**的 arena 综合得分（胜=1/和=0.5/负=0）。

    区间取闭开（边界归属写死于此，供单测钉死）：
        score >= 0.55        → GO 全尺寸        （0.55 含在 GO 一侧）
        0.45 <= score < 0.55 → 平手维持阶段六   （0.45 含在平手一侧）
        score <  0.45        → 关闭路线
    """
    s = float(score)
    if s >= GO_THRESHOLD:
        return VERDICT_GO
    if s >= TIE_THRESHOLD:
        return VERDICT_TIE
    return VERDICT_KILL


# ────────────────────────────────────────────────────────────────
# metrics.csv 读取与窗口聚合（纯函数，无 torch 依赖）
# ────────────────────────────────────────────────────────────────
def _f(v, default=None):
    """安全转 float（metrics.csv 缺列时为空字符串）。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "1.0", "yes")


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _num(x, digits=3) -> str:
    return f"{x:.{digits}f}" if x is not None else "-"


def _pct(x, digits=1) -> str:
    return f"{x * 100:.{digits}f}%" if x is not None else "-"


def _disp_w(s: str) -> int:
    """粗略终端显示宽度：CJK/全角算 2 列（表格对齐用，不追求 unicodedata 级精确）。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _lj(s: str, width: int) -> str:
    return s + " " * max(1, width - _disp_w(s))


def _rj(s: str, width: int) -> str:
    return " " * max(1, width - _disp_w(s)) + s


def load_metrics(csv_path) -> list[dict]:
    """读 metrics.csv，按 iter 升序返回原始行（字符串值）。

    从零训练不会 --resume，正常不该出现重复 iter；仍防御性地"同 iter 保留最后一行"
    （与 scripts/analyze_training.py:76 同口径），万一有人手工续跑也不会双计。
    """
    path = Path(csv_path)
    if not path.exists():
        return []
    by_iter: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            it = _f(row.get("iter"))
            if it is None:
                continue
            by_iter[int(it)] = row
    return [by_iter[k] for k in sorted(by_iter)]


def summarize_metrics(rows: list[dict], window: int = 30) -> list[dict]:
    """把 metrics 行按 ``iter // window`` 分桶聚合，返回按桶号升序的摘要列表。

    每项键：bucket(桶号)、range("0-29")、n(行数)、policy_loss/entropy/value_loss/
    value_mae/red_winrate/draw_rate/arena_score（窗口均值，全缺则 None）、
    promoted（窗口内晋升次数，整数）。

    纯函数：不读文件、不打印。桶内 range 由 min/max iter 求得，故传入行未排序也正确。
    """
    if int(window) <= 0:
        raise ValueError(f"window 必须为正整数，收到 {window}")
    window = int(window)
    buckets: dict[int, list[dict]] = {}
    for r in rows:
        it = _f(r.get("iter"))
        if it is None:
            continue
        buckets.setdefault(int(it) // window, []).append(r)

    out: list[dict] = []
    for b in sorted(buckets):
        rs = buckets[b]
        iters = [int(_f(r["iter"])) for r in rs]
        item = {
            "bucket": b,
            "range": f"{min(iters)}-{max(iters)}",
            "n": len(rs),
            "promoted": sum(1 for r in rs if _truthy(r.get("promoted"))),
        }
        for key in ("policy_loss", "entropy", "value_loss", "value_mae",
                    "red_winrate", "draw_rate", "arena_score"):
            item[key] = _mean(_f(r.get(key)) for r in rs)
        out.append(item)
    return out


# 曲线对比表的行定义：(显示名, 摘要键, 格式化器, 差值格式化器)
_CURVE_ROWS = [
    ("policy_loss", "policy_loss", lambda v: _num(v, 4), lambda d: f"{d:+.4f}"),
    ("entropy", "entropy", lambda v: _num(v, 4), lambda d: f"{d:+.4f}"),
    ("value_loss", "value_loss", lambda v: _num(v, 4), lambda d: f"{d:+.4f}"),
    ("value_mae", "value_mae", lambda v: _num(v, 4), lambda d: f"{d:+.4f}"),
    ("红胜率", "red_winrate", _pct, lambda d: f"{d * 100:+.1f}pt"),
    ("和率", "draw_rate", _pct, lambda d: f"{d * 100:+.1f}pt"),
    ("arena均分", "arena_score", lambda v: _num(v, 3), lambda d: f"{d:+.3f}"),
]


def _print_curves(g_rows: list[dict], p_rows: list[dict], window: int, out) -> None:
    """并排打印两臂窗口摘要 + 末窗口小结。

    两臂行数可能不等（某臂早停/仍在跑）：按桶号对齐，缺失方打 "-"，绝不 zip 截断。
    """
    g_sum = summarize_metrics(g_rows, window)
    p_sum = summarize_metrics(p_rows, window)
    g_by = {w["bucket"]: w for w in g_sum}
    p_by = {w["bucket"]: w for w in p_sum}

    print("", file=out)
    print("=" * 78, file=out)
    print(f"曲线对比（每 {window} 迭代一窗口；左 gumbel 臂 / 右 puct 臂，差 = gumbel - puct）",
          file=out)
    print("=" * 78, file=out)
    print(f"gumbel 臂：{len(g_rows)} 迭代（iter {int(_f(g_rows[0]['iter']))}"
          f"..{int(_f(g_rows[-1]['iter']))}）  |  "
          f"puct 臂：{len(p_rows)} 迭代（iter {int(_f(p_rows[0]['iter']))}"
          f"..{int(_f(p_rows[-1]['iter']))}）", file=out)
    if len(g_rows) != len(p_rows):
        print("  ⚠ 两臂迭代数不等 —— 等算力前提被破坏，曲线对比只作定性参考。", file=out)

    for b in sorted(set(g_by) | set(p_by)):
        g, p = g_by.get(b), p_by.get(b)
        # 窗口区间取两臂并集（某臂在窗口中途结束时，不能只报另一臂的区间）
        ends = [int(x) for w in (g, p) if w for x in w["range"].split("-")]
        rng = f"{min(ends)}-{max(ends)}"
        print("", file=out)
        print(f"── 窗口 iter {rng} "
              f"(gumbel {g['n'] if g else 0} 行 / puct {p['n'] if p else 0} 行) "
              + "─" * 18, file=out)
        print("  " + _lj("指标", 14) + _rj("gumbel", 12) + _rj("puct", 12)
              + _rj("差(g-p)", 14), file=out)
        for name, key, fmt, dfmt in _CURVE_ROWS:
            gv = g.get(key) if g else None
            pv = p.get(key) if p else None
            diff = dfmt(gv - pv) if (gv is not None and pv is not None) else "-"
            print("  " + _lj(name, 14) + _rj(fmt(gv), 12) + _rj(fmt(pv), 12)
                  + _rj(diff, 14), file=out)
        gp = f"{g['promoted']}/{g['n']}" if g else "-"
        pp = f"{p['promoted']}/{p['n']}" if p else "-"
        pdiff = f"{g['promoted'] - p['promoted']:+d}" if (g and p) else "-"
        print("  " + _lj("晋升次数", 14) + _rj(gp, 12) + _rj(pp, 12)
              + _rj(pdiff, 14), file=out)

    # ── 末窗口小结（两臂都有数据的最大桶号）──────────────────────────
    common = sorted(set(g_by) & set(p_by))
    print("", file=out)
    print("── 末窗口小结 " + "─" * 40, file=out)
    if not common:
        print("  两臂没有共同窗口，无法对比。", file=out)
    else:
        b = common[-1]
        g, p = g_by[b], p_by[b]
        rng = (g["range"] if g["range"] == p["range"]
               else f"gumbel {g['range']} / puct {p['range']}")
        print(f"  末共同窗口：iter {rng}", file=out)
        ge, pe = g.get("entropy"), p.get("entropy")
        if ge is not None and pe is not None:
            tail = ("两臂持平" if ge == pe
                    else f"{'gumbel' if ge < pe else 'puct'} 更低")
            print(f"  策略熵：gumbel {_num(ge, 4)} vs puct {_num(pe, 4)} → "
                  f"{tail}（熵低 = 策略更笃定，但不等于更强）", file=out)
        else:
            print("  策略熵：数据缺失，无法比较", file=out)
        gt = sum(w["promoted"] for w in g_sum)
        pt = sum(w["promoted"] for w in p_sum)
        tail2 = ("两臂相同" if gt == pt
                 else f"{'gumbel' if gt > pt else 'puct'} 更多")
        print(f"  晋升次数：末窗口 gumbel {g['promoted']}/{g['n']} vs "
              f"puct {p['promoted']}/{p['n']}；全程累计 gumbel {gt} vs puct {pt} → "
              f"{tail2}", file=out)
        print("  ⚠ 两臂的 arena_score/晋升各自对阵自己的 best，跨臂不可比（只反映"
              "各自内部改进速度）；跨臂强弱以上面的直接对抗为准。", file=out)


# ────────────────────────────────────────────────────────────────
# 对抗（arena）
# ────────────────────────────────────────────────────────────────
def _probe_history_steps(paths: list[str], out) -> tuple[int | None, bool]:
    """读两个 checkpoint 的 model_config.history_steps，返回 ``(值, 是否致命)``。

    arena 的 ``_play_one`` 用**同一个** ``cfg.model.history_steps`` 编码双方局面
    （rl/evaluate.py:113），两臂若不一致则结构上无法对打（第一次前向就会因输入通道
    数不符而崩）——返回 fatal=True 让调用方直接中止，给人看得懂的中文原因而不是
    一句 shape mismatch。

    同时检查 checkpoint 是否为训练启动时占位的**随机初始网**（rl/train.py 启动即把
    随机模型写成 best.pt，iteration=-1、meta.init=True）——零晋升的臂 best.pt 从未被
    真实模型覆盖，拿它裁决会给出方向完全错误的结论，故与 history_steps 不一致同级：fatal。

    读不到（文件损坏/旧格式）返回 ``(None, False)``：只告警，沿用 Config() 默认值。
    """
    vals = []
    for p in paths:
        try:
            import torch

            ck = torch.load(p, map_location="cpu", weights_only=False)
            meta = ck.get("meta") or {}
            if meta.get("init") is True or int(ck.get("iteration", 0)) < 0:
                print(f"错误：{p} 是训练启动时占位的随机初始网"
                      f"（iteration={ck.get('iteration')}, meta.init={meta.get('init')}）——"
                      f"该臂全程零晋升，best.pt 从未被真实模型覆盖，裁决无意义。"
                      f"先查该臂训练日志：150 迭代零晋升本身就是重大异常"
                      f"（主线从零首次晋升在 iter 2）。", file=out)
                return None, True
            vals.append(int((ck.get("model_config") or {})["history_steps"]))
        except Exception as e:  # 探测失败不致命，只告警
            print(f"  ⚠ 读不到 {p} 的 model_config.history_steps（{type(e).__name__}: {e}），"
                  f"沿用 Config() 默认值", file=out)
            return None, False
    if len(set(vals)) != 1:
        print(f"错误：两臂 history_steps 不一致：{vals} —— arena 只能用一个 history_steps "
              f"编码双方，这两个 checkpoint 结构上不可直接对打。两臂 yaml 的 model 段"
              f"应逐字段相同（单变量对照），请核对 configs/ab_scratch_*.yaml。", file=out)
        return None, True
    return vals[0], False


def run_arena(gumbel_ckpt: str, puct_ckpt: str, games: int, sims: int,
              device: str, num_gpus: int, out=sys.stdout) -> dict | None:
    """跑去相关 arena 并打印结果，返回 stats（gumbel 视角）；无法运行时返回 None。

    ⚠ 视角：第一参数 = arena 的 "new" 方 = **gumbel 臂**，返回的 wins/score 均为
    gumbel 视角。这个方向不可调换（调换 = 决策反向）。
    """
    # 延迟导入：--help / --skip-arena 完全不碰 torch
    from rl.config import Config
    import rl.evaluate as evaluate

    cfg = Config()
    # 只覆盖对抗必需的字段（两臂对称施加,不影响方向性结论）。
    # draw_penalty 显式对齐两臂训练值 -0.2（rl/evaluate.py 会注入树内和棋值）。
    cfg.selfplay.draw_penalty = -0.2
    cfg.arena.arena_games = int(games)
    cfg.arena.arena_sims = int(sims)
    cfg.arena.arena_temp_moves = 16      # 去相关：前 16 步温度采样
    cfg.arena.arena_temperature = 0.8
    cfg.arena.arena_workers = 0          # 0 = 自动并行（cuda 且 ≥16 盘）
    cfg.train.device = str(device)
    cfg.train.num_gpus = int(num_gpus)

    hist, fatal = _probe_history_steps([gumbel_ckpt, puct_ckpt], out)
    if fatal:
        return None
    if hist is not None:
        cfg.model.history_steps = hist

    print("", file=out)
    print("=" * 78, file=out)
    print("对抗（去相关 arena）", file=out)
    print("=" * 78, file=out)
    print(f"  新方(new) = gumbel 臂: {gumbel_ckpt}", file=out)
    print(f"  旧方(best)= puct   臂: {puct_ckpt}", file=out)
    print(f"  设置：{cfg.arena.arena_games} 盘 × {cfg.arena.arena_sims} sims，"
          f"红黑对调各半，无 Dirichlet 噪声", file=out)
    print(f"        去相关 arena_temp_moves={cfg.arena.arena_temp_moves} / "
          f"arena_temperature={cfg.arena.arena_temperature}，"
          f"arena_workers={cfg.arena.arena_workers}(0=自动)", file=out)
    print(f"        device={cfg.train.device} num_gpus={cfg.train.num_gpus} "
          f"history_steps={cfg.model.history_steps} "
          f"树内和棋值 draw_value={cfg.selfplay.draw_penalty}(与两臂训练一致)", file=out)
    print("  ★ 视角：以下 W/D/L 与 score 全部是 **gumbel 臂视角**"
          "（score > 0.5 = gumbel 更强）。", file=out)
    print(f"  ... 对局中（{cfg.arena.arena_games} 盘 × {cfg.arena.arena_sims} sims；"
          f"预注册的 400×200 在 8 卡上约二三十分钟）", file=out, flush=True)

    stats = evaluate.arena(gumbel_ckpt, puct_ckpt, cfg, cfg.train.device)

    n = int(stats.get("games", 0)) or 1
    score = float(stats["score"])
    se = float(stats.get("score_se", 0.0))
    lcb = score - 1.0 * se
    print("", file=out)
    print(f"  W/D/L（gumbel 视角）  : {stats['wins']} / {stats['draws']} / {stats['losses']}"
          f"  （共 {stats.get('games')} 盘）", file=out)
    print(f"  score（gumbel 视角）  : {score:.4f} ± {se:.4f} (1se)", file=out)
    print(f"  LCB(z=1)              : {lcb:.4f}   ← 仅供参考，判据用点估计 score", file=out)
    print(f"  gumbel 执红分         : {_num(stats.get('red_score'), 4)}"
          f"    执黑分: {_num(stats.get('black_score'), 4)}", file=out)
    frac = float(stats.get("distinct_openings", 0)) / n
    print(f"  开局多样性            : distinct_openings "
          f"{stats.get('distinct_openings')}/{stats.get('games')} = {_pct(frac)}", file=out)
    if frac < DIVERSITY_WARN:
        print(f"  ⚠ 多样性低于 {_pct(DIVERSITY_WARN, 0)}：对局高度重复，去相关不足，"
              f"有效样本量远小于盘数，score_se 被低估 —— 提高 arena_temp_moves/"
              f"arena_temperature 后重跑再下结论。", file=out)
    return stats


def _print_verdict(score: float, out=sys.stdout) -> str:
    v = verdict(score)
    print("", file=out)
    print("=" * 78, file=out)
    print("裁决", file=out)
    print("=" * 78, file=out)
    print(f"  gumbel 臂视角 score = {score:.4f}", file=out)
    print(f"  判据：>= {GO_THRESHOLD}(含) → GO；[{TIE_THRESHOLD}, {GO_THRESHOLD}) → 平手；"
          f"< {TIE_THRESHOLD} → 关闭", file=out)
    print(f"  ★ 结论：{v}", file=out)
    print(f"  {PREREG_NOTE}", file=out)
    return v


# ────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────
_MISSING_HINT = (
    "请确认：(1) 在两臂输出目录的父目录（仓库根/训练目录）下运行本脚本；"
    "(2) 两臂是否都已跑完 —— 每臂需按 configs/ab_scratch_{gumbel,puct}.yaml "
    "跑满 total_iterations=150；未跑完则 best.pt / metrics.csv 可能缺失。"
)


def _check_files(paths: list[str], what: str, out) -> bool:
    missing = [p for p in paths if not Path(p).exists()]
    if not missing:
        return True
    print("", file=out)
    print(f"错误：{what}所需文件缺失：", file=out)
    for p in missing:
        print(f"  - 找不到 {Path(p).resolve()}", file=out)
    print(f"  {_MISSING_HINT}", file=out)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="A/B 终点裁决：Gumbel 臂 vs PUCT 臂（对抗 + 曲线对比，只读不写）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--gumbel-ckpt", default="ckpts_ab_gumbel/best.pt",
                    help="gumbel 臂 best checkpoint（对抗中恒为 new 方 = 打分视角方）")
    ap.add_argument("--puct-ckpt", default="ckpts_ab_puct/best.pt",
                    help="puct 臂 best checkpoint（对抗中恒为 best 方 = 对照）")
    ap.add_argument("--gumbel-metrics", default="logs_ab_gumbel/metrics.csv",
                    help="gumbel 臂训练曲线 csv")
    ap.add_argument("--puct-metrics", default="logs_ab_puct/metrics.csv",
                    help="puct 臂训练曲线 csv")
    ap.add_argument("--games", type=int, default=400,
                    help="对抗盘数（预注册 400：低和棋率下 200 盘误 GO 率 ~8%%，400 盘 ~2%%）")
    ap.add_argument("--sims", type=int, default=200, help="每步 MCTS 次数")
    ap.add_argument("--window", type=int, default=30, help="曲线对比窗口（迭代数）")
    ap.add_argument("--device", default="cuda", help="推理设备 cuda/cpu")
    ap.add_argument("--num-gpus", type=int, default=8,
                    help="仅记录用途——arena 内部按实际 CUDA 设备数并行，不受此参数限制")
    ap.add_argument("--skip-arena", action="store_true", help="跳过对抗，只看曲线")
    ap.add_argument("--skip-metrics", action="store_true", help="跳过曲线，只跑对抗")
    args = ap.parse_args(argv)

    # Windows 默认 cp936 装不下 "⚠"（U+26A0）：重定向/管道时会在打印告警的瞬间
    # UnicodeEncodeError —— 那是 200 盘打完之后、裁决打印之前，代价是整轮重跑。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # capsys 等非标准 stdout 不支持 reconfigure，忽略即可

    out = sys.stdout
    print("=" * 78, file=out)
    print("A/B 终点裁决  ——  从零 Gumbel vs 现行 PUCT+锐化配方（等算力单变量对照）", file=out)
    print("=" * 78, file=out)
    print(f"  {PREREG_NOTE}", file=out)
    print("  本脚本只读不写，不产生任何文件。", file=out)

    if args.skip_arena and args.skip_metrics:
        print("", file=out)
        print("提示：--skip-arena 与 --skip-metrics 同时给出，无事可做。", file=out)
        return 0

    failed = False

    if not args.skip_arena:
        if _check_files([args.gumbel_ckpt, args.puct_ckpt], "对抗", out):
            stats = run_arena(args.gumbel_ckpt, args.puct_ckpt, args.games,
                              args.sims, args.device, args.num_gpus, out)
            if stats is not None:
                _print_verdict(float(stats["score"]), out)
            else:
                failed = True   # 对抗未能进行 → 无裁决，退出码非 0
        else:
            failed = True
    else:
        print("", file=out)
        print("（--skip-arena：跳过对抗，本次不产生裁决结论）", file=out)

    # 曲线环节失败只告警不改退出码：rc 语义 = "对抗/裁决是否成功"（rc=0 时上方
    # 裁决结论有效），避免外层自动化因 metrics.csv 缺失丢弃一条有效裁决。
    if not args.skip_metrics:
        if _check_files([args.gumbel_metrics, args.puct_metrics], "曲线对比", out):
            g_rows = load_metrics(args.gumbel_metrics)
            p_rows = load_metrics(args.puct_metrics)
            if not g_rows or not p_rows:
                empty = [p for p, r in ((args.gumbel_metrics, g_rows),
                                        (args.puct_metrics, p_rows)) if not r]
                print("", file=out)
                print(f"⚠ 曲线环节跳过：metrics.csv 存在但没有可解析的数据行：{empty}", file=out)
                print(f"  {_MISSING_HINT}", file=out)
            else:
                _print_curves(g_rows, p_rows, args.window, out)
        else:
            print("  ⚠ 曲线环节跳过（文件缺失）；不影响上方对抗/裁决的有效性。", file=out)
    else:
        print("", file=out)
        print("（--skip-metrics：跳过曲线对比）", file=out)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
