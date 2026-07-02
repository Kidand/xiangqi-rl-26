# -*- coding: utf-8 -*-
"""端到端冒烟测试（DESIGN.md §15）。

用 ``configs/smoke.yaml`` 跑完整流水线（self-play → 训练 → arena → 记录/CSV 落盘），
CPU < 5 分钟，结尾断言关键产物存在且 metrics.csv 行数正确，print PASS 并 exit 0。

为避免污染用户工作区，本脚本把输出目录重定向到临时目录（scratchpad/仓库 tmp）。

运行::

    .venv/Scripts/python.exe scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# 允许以脚本方式直接运行（把仓库根加入 sys.path）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.config import Config  # noqa: E402
from rl.logger import CSV_COLUMNS  # noqa: E402
from rl.train import run_training  # noqa: E402


def _read_csv_rows(csv_path: Path):
    import csv

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows


def main() -> int:
    config_path = _REPO_ROOT / "configs" / "smoke.yaml"
    if not config_path.exists():
        print(f"FAIL: 配置不存在 {config_path}")
        return 1

    cfg = Config.from_yaml(str(config_path))

    # 输出目录重定向到临时目录，跑完清理。
    out_dir = Path(tempfile.mkdtemp(prefix="xiangqi_smoke_"))
    cfg.log.log_dir = str(out_dir / "logs")
    cfg.log.records_dir = str(out_dir / "records")
    cfg.train.checkpoint_dir = str(out_dir / "checkpoints")
    cfg.log.tensorboard = False  # 冒烟无需 TB

    total_iters = int(cfg.train.total_iterations)
    print(f"[smoke] 输出目录: {out_dir}")
    print(f"[smoke] 配置: iters={total_iters} games/iter={cfg.selfplay.games_per_iteration} "
          f"model={cfg.model.blocks}b×{cfg.model.filters}f sims={cfg.mcts.num_sims}")

    t0 = time.monotonic()
    run_training(cfg, resume=False)
    elapsed = time.monotonic() - t0
    print(f"[smoke] 训练流程完成，用时 {elapsed:.1f}s")

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    records_dir = Path(cfg.log.records_dir) / "selfplay"
    logs_dir = Path(cfg.log.log_dir)

    failures = []

    # 1) best.pt 与逐迭代 checkpoint。
    best_pt = ckpt_dir / "best.pt"
    if not best_pt.exists():
        failures.append(f"缺少 best.pt: {best_pt}")
    for it in range(total_iters):
        p = ckpt_dir / f"iter_{it:04d}.pt"
        if not p.exists():
            failures.append(f"缺少 checkpoint: {p}")

    # 2) buffer 持久化目录。
    if not (ckpt_dir / "buffer" / "meta.json").exists():
        failures.append("缺少 buffer/meta.json")

    # 3) 每迭代 records/selfplay/iter_XXXX.jsonl，行数 = games_per_iteration。
    games_per_iter = int(cfg.selfplay.games_per_iteration)
    for it in range(total_iters):
        rp = records_dir / f"iter_{it:04d}.jsonl"
        if not rp.exists():
            failures.append(f"缺少对局记录: {rp}")
            continue
        n_lines = sum(1 for ln in rp.read_text(encoding="utf-8").splitlines() if ln.strip())
        if n_lines != games_per_iter:
            failures.append(
                f"记录行数错误 {rp.name}: 期望 {games_per_iter}，实际 {n_lines}"
            )

    # 4) metrics.csv：1 header + total_iters 数据行，列顺序符合契约。
    csv_path = logs_dir / "metrics.csv"
    if not csv_path.exists():
        failures.append(f"缺少 metrics.csv: {csv_path}")
    else:
        rows = _read_csv_rows(csv_path)
        if not rows:
            failures.append("metrics.csv 为空")
        else:
            header = rows[0]
            if header != CSV_COLUMNS:
                failures.append(f"CSV 列顺序不符\n期望 {CSV_COLUMNS}\n实际 {header}")
            data_rows = [r for r in rows[1:] if r]
            if len(data_rows) != total_iters:
                failures.append(
                    f"CSV 数据行数错误：期望 {total_iters}，实际 {len(data_rows)}"
                )

    # 5) train.log 存在。
    if not (logs_dir / "train.log").exists():
        failures.append("缺少 train.log")

    # 6) 用记录模块重演一局验证记录可读。
    try:
        from xiangqi.records import read_records, replay_record

        rp0 = records_dir / "iter_0000.jsonl"
        recs = read_records(rp0)
        if recs:
            replayed = replay_record(recs[0])
            if "fens" not in replayed or len(replayed["fens"]) != len(recs[0]["moves"]) + 1:
                failures.append("记录重演 fens 长度不符")
    except Exception as e:
        failures.append(f"记录重演失败: {e}")

    # 时限提示（非硬性失败，仅告警）。
    if elapsed > 300:
        print(f"[smoke][WARN] 用时 {elapsed:.1f}s 超过 5 分钟预期")

    # 清理临时目录（尽力而为）。
    try:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception:
        pass

    if failures:
        print("FAIL: 冒烟测试未通过：")
        for f in failures:
            print("  - " + f)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
