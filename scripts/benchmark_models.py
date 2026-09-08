"""Evaluate a candidate against a frozen checkpoint on paired self-play openings."""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.benchmark import benchmark, load_openings
from rl.config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="candidate checkpoint .pt")
    parser.add_argument("--baseline", required=True, help="frozen baseline checkpoint, e.g. best0716.pt")
    parser.add_argument("--records", nargs="+", required=True, help="self-play JSON/JSONL files or globs")
    parser.add_argument("--config", help="shared MCTS defaults and game rules YAML")
    parser.add_argument("--candidate-config", help="optional candidate MCTS defaults YAML")
    parser.add_argument("--baseline-config", help="optional baseline MCTS defaults YAML")
    parser.add_argument("--candidate-draw-value", type=float, required=True)
    parser.add_argument("--baseline-draw-value", type=float, required=True)
    parser.add_argument("--opening-plies", type=int, default=12, help="fixed prefix length in half moves")
    parser.add_argument("--pairs", type=int, help="number of unique openings; default uses all")
    parser.add_argument("--seed", type=int, default=42, help="opening selection seed")
    parser.add_argument("--sims", type=int, default=400, help="same MCTS budget for both players")
    parser.add_argument("--max-plies", type=int, help="total game limit, including opening")
    parser.add_argument("--no-capture-plies", type=int, help="shared no-capture draw limit")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default 1)")
    parser.add_argument("--out", required=True, help="new output JSON (existing paths are refused)")
    args = parser.parse_args(argv)
    output = Path(args.out).resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    if args.threads < 1:
        parser.error("--threads must be positive")
    cfg = Config.from_yaml(args.config) if args.config else Config()
    candidate_cfg = Config.from_yaml(args.candidate_config) if args.candidate_config else cfg
    baseline_cfg = Config.from_yaml(args.baseline_config) if args.baseline_config else cfg
    cmcts = dataclasses.replace(candidate_cfg.mcts, draw_value=args.candidate_draw_value)
    bmcts = dataclasses.replace(baseline_cfg.mcts, draw_value=args.baseline_draw_value)
    max_plies = args.max_plies if args.max_plies is not None else cfg.selfplay.max_game_plies
    no_capture = args.no_capture_plies if args.no_capture_plies is not None else cfg.selfplay.no_capture_draw_plies
    files = []
    for pattern in args.records:
        matches = glob.glob(pattern)
        if not matches:
            parser.error(f"record path/glob matched no files: {pattern}")
        files.extend(matches)
    import torch

    torch.set_num_threads(args.threads)
    try:
        openings, selection = load_openings(
            files, prefix_plies=args.opening_plies, max_openings=args.pairs,
            seed=args.seed, max_plies=max_plies, no_capture_plies=no_capture,
        )
        print(f"Selected {len(openings)} unique openings, {2 * len(openings)} games; "
              f"{selection['duplicate_prefixes_excluded']} duplicate prefixes excluded.", flush=True)

        def progress(game):
            print(f"[{game['game_index'] + 1}/{2 * len(openings)}] "
                  f"pair={game['pair_index']} candidate={game['candidate_color']} "
                  f"result={game['result']} plies={game['plies']}", flush=True)

        result = benchmark(
            args.candidate, args.baseline, openings, candidate_mcts=cmcts, baseline_mcts=bmcts,
            sims=args.sims, max_plies=max_plies, no_capture_plies=no_capture,
            device=args.device, selection=selection, progress=progress,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids accidentally replacing another experiment's result.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    stats = result["statistics"]
    print(f"score={stats['score']:.4f}, pair_SE={stats['pair_score_se']}, "
          f"approx_95%_CI={stats['score_ci95_approx']}")
    for note in stats["limitations"]:
        print(note)
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
