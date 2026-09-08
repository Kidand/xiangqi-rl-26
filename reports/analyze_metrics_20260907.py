"""Read-only training archive audit; writes only its JSON report in reports/."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.model import load_checkpoint

FIELDS = [
    "policy_loss", "value_loss", "entropy", "value_mae", "draw_rate",
    "avg_plies", "arena_score", "arena_red_score", "arena_black_score",
    "selfplay_sec", "train_sec", "total_sec", "resign_rate", "resign_fp_rate", "lr",
]


def describe(values: list[float]) -> dict:
    values = sorted(values)
    return {
        "n": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "min": values[0], "max": values[-1],
        "p90_nearest_rank": values[max(0, (9 * len(values) + 9) // 10 - 1)],
    }


def summarize(rows: list[dict]) -> dict:
    tail = rows[-100:]
    overhead = [float(r["total_sec"]) - float(r["selfplay_sec"]) - float(r["train_sec"]) for r in tail]
    return {
        "actual_iteration_range": [int(rows[0]["iter"]), int(rows[-1]["iter"])],
        "row_count": len(rows),
        "missing_inside_actual_range": sorted(set(range(int(rows[0]["iter"]), int(rows[-1]["iter"]) + 1)) - {int(r["iter"]) for r in rows}),
        "games": sum(int(r["games"]) for r in rows),
        "logged_total_hours": sum(float(r["total_sec"]) for r in rows) / 3600,
        "promotions": sum(r["promoted"] == "True" for r in rows),
        "last_promotion_iteration": next((int(r["iter"]) for r in reversed(rows) if r["promoted"] == "True"), None),
        "tail100": {
            "iteration_range": [int(tail[0]["iter"]), int(tail[-1]["iter"])],
            "row_count": len(tail),
            "promotions": sum(r["promoted"] == "True" for r in tail),
            "metrics": {f: describe([float(r[f]) for r in tail if r[f]]) for f in FIELDS},
            "resign_fp_nonzero_rows": sum(float(r["resign_fp_rate"]) > 0 for r in tail),
            "unallocated_total_minus_selfplay_minus_train_seconds": describe(overhead),
            "unallocated_time_fraction": sum(overhead) / sum(float(r["total_sec"]) for r in tail),
        },
    }


files = {p.name: list(csv.DictReader(p.open(encoding="utf-8", newline=""))) for p in sorted((ROOT / "logs").glob("metrics*.csv"))}
archive = files["metrics0729.csv"]
rollback_indices = [i for i in range(1, len(archive)) if int(archive[i]["iter"]) < int(archive[i - 1]["iter"])]
if len(rollback_indices) != 1:
    raise RuntimeError(f"Expected one rollback in archived metrics0729.csv, got {rollback_indices}")
cut = rollback_indices[0]
head, tail = archive[:cut], archive[cut:]
report = {
    "analysis_date": "2026-09-07",
    "method": "Split metrics0729.csv on physical row order when iteration decreases; split first monotonic segment using documented historical boundaries. Never deduplicate by iteration across phases.",
    "files": [], "stages": [], "checkpoints": [],
    "local_stage6_measured_metrics_present": False,
    "limitations": [
        "Historical stage boundaries: 0-599, 600-999, 1000-1499, 1500-1999; stage five is the appended rollback segment 1500-1753.",
        "Stage three has no row for iteration 1000; stage five has no row for iteration 1608. No imputation was performed.",
        "Archive files are overlapping cumulative snapshots, not independent runs. Their rows must not be summed.",
        "Tail100 statistics average per-iteration metrics; they are not global sample-weighted losses or conditional-rate ratios.",
        "policy_loss is CE(target,prediction); entropy is H(prediction). CE-H(prediction) is not KL(target||prediction). H(target) was not logged.",
        "Losses are computed on changing replay-buffer training batches. Lower losses alone do not establish better chess strength or generalization.",
        "Arena score compares each candidate with that iteration's moving incumbent. It is not a fixed-opponent ladder and cannot establish best0729 versus best0716 strength.",
        "CSV lacks per-game results, independent opening counts and score SE. Promotion counts and arena means alone cannot establish statistical significance.",
        "resign_fp_rate equals fp/calib_would per iteration (zero when denominator is zero); numerator and denominator are absent from CSV, so aggregate FP rate and confidence intervals cannot be reconstructed.",
        "total_sec includes stages beyond selfplay and training; their residual includes arena, checkpoint/buffer handling and other overhead, not a separately measured arena duration.",
        "No local training log or additional metrics rollback proves a stage-six run; a stage-six configuration and README statement alone are not outcome evidence.",
    ],
    "implementation_references": {
        "policy_CE": "rl/train.py:518", "value_MSE": "rl/train.py:519",
        "prediction_entropy": "rl/train.py:537", "resign_FP_denominator": "rl/train.py:429",
        "arena_then_total_time": "rl/train.py:819", "CSV_total_time": "rl/train.py:852",
        "legacy_iteration_deduplication": "scripts/analyze_training.py:86",
    },
}
for name, rows in files.items():
    cuts = [0] + [i for i in range(1, len(rows)) if int(rows[i]["iter"]) < int(rows[i - 1]["iter"])] + [len(rows)]
    segments = []
    for a, b in zip(cuts, cuts[1:]):
        segments.append({"data_row_range_1based": [a + 1, b], "csv_line_range_1based": [a + 2, b + 1], "iteration_range": [int(rows[a]["iter"]), int(rows[b - 1]["iter"])]})
    prefix_length = 0
    for a, b in zip(rows, head):
        if a != b:
            break
        prefix_length += 1
    report["files"].append({"path": f"logs/{name}", "row_count": len(rows), "monotonic_segments": segments, "identical_prefix_rows_vs_0729_first_segment": prefix_length})

for stage, lo, hi in [(1, 0, 599), (2, 600, 999), (3, 1000, 1499), (4, 1500, 1999)]:
    rows = [r for r in head if lo <= int(r["iter"]) <= hi]
    report["stages"].append({"stage": stage, "documented_iteration_range": [lo, hi], "source": "logs/metrics0729.csv first monotonic segment", **summarize(rows)})
report["stages"].append({"stage": 5, "documented_iteration_range": [1500, 1999], "source": "logs/metrics0729.csv second monotonic segment", **summarize(tail)})

for name in ["best0716", "best0720", "best0729"]:
    model, ckpt = load_checkpoint(ROOT / "ckpts" / f"{name}.pt")
    report["checkpoints"].append({
        "path": f"ckpts/{name}.pt", "iteration": ckpt["iteration"],
        "model_config": ckpt["model_config"], "meta": ckpt["meta"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "buffer_element_count": sum(p.numel() for p in model.buffers()),
        "state_dict_element_count": sum(p.numel() for p in model.state_dict().values()),
    })

destination = ROOT / "reports" / "training_metrics_20260907.json"
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
print(f"{len(report['files'])} files, {len(report['stages'])} phases, {len(report['checkpoints'])} checkpoints")
