"""Read-only probes for draw backup, arena gate, and historical Gumbel scheduling.

Run from the repository root:
    python reports/value_gate_probe_20260907.py

No model is loaded and no training is started. Only the adjacent JSON report is
written. Historical code is read from Git and executed in an in-memory module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rl.config import Config, MCTSConfig
from rl.evaluate import passes_gate, score_stats
from rl.mcts import SearchTree
from xiangqi.board import Board
from xiangqi.constants import ACTION_SIZE


def constant_eval(states, value=-0.2):
    return (
        np.full((len(states), ACTION_SIZE), 1 / ACTION_SIZE, np.float32),
        np.full(len(states), value, np.float32),
    )


def draw_probe():
    rows = []
    for max_plies in (1, 2, 400):
        tree = SearchTree(
            Board(max_plies=max_plies),
            MCTSConfig(batch_size=1, draw_value=-0.2),
            False,
        )
        states, tokens = tree.select_leaves(1)
        tree.apply_results(tokens, *constant_eval(states))
        states, tokens = tree.select_leaves(1)
        if tokens:
            tree.apply_results(tokens, *constant_eval(states))
        rows.append({
            "max_plies": max_plies,
            "network_leaf": bool(tokens),
            "root_q": tree.root_value(),
            "completed_root_visits": tree.total_visits,
        })
    assert rows[0]["root_q"] == -0.2
    assert abs(rows[1]["root_q"] - 0.2) < 1e-7
    return {
        "description": "Same predicted draw payoff becomes positive at a network leaf but remains negative at a terminal leaf.",
        "note": "With max_plies=2 every child is nonterminal but will draw after one further quiet initial-position move. This isolates backup semantics, not trained-model strength.",
        "rows": rows,
    }


def gate_probe():
    arena_cfg = Config().arena
    arena_cfg.gate_threshold = 0.55
    arena_cfg.gate_lcb_z = 1.0
    changed = 0
    total = 0
    for wins in range(161):
        for draws in range(161 - wins):
            score, se = score_stats(wins, draws, 160 - wins - draws)
            changed += passes_gate({"score": score, "score_se": se}, arena_cfg) != (score >= 0.55)
            total += 1
    assert changed == 0
    probability = sum(math.comb(160, wins) for wins in range(88, 161)) / 2**160
    return {
        "games": 160,
        "gate_threshold": 0.55,
        "gate_lcb_z": 1.0,
        "wdl_combinations_checked": total,
        "lcb_changed_decisions": changed,
        "false_promotion_probability_equal_strength_independent_no_draws": probability,
        "note": "The probability is an illustrative exact binomial calculation, not an estimate for actual correlated games with draws.",
    }


def historical_gumbel_probe():
    revision = "f7aee4d"
    source = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"{revision}:rl/mcts.py"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    module = ModuleType("historical_gumbel_probe")
    exec(compile(source, f"{revision}:rl/mcts.py", "exec"), module.__dict__)
    options = vars(MCTSConfig(num_sims=24, batch_size=9)).copy()
    options.update(algorithm="gumbel", gumbel_m=8, gumbel_c_visit=50.0, gumbel_c_scale=1.0)
    tree = module.SearchTree(
        Board(), SimpleNamespace(**options), True, rng=np.random.default_rng(17)
    )
    states, tokens = tree.select_leaves(1)
    tree.apply_results(tokens, *constant_eval(states, 0.0))
    tree._g_init()
    initial_candidates = tree._g_cand.copy()
    # Eight candidate leaves become pending. The ninth selection crosses the
    # halving boundary before any of those eight network values are returned.
    states, tokens = tree.select_leaves(9)
    candidates_before_results = tree._g_cand.copy()
    eliminated = sorted(set(map(int, initial_candidates)) - set(map(int, candidates_before_results)))
    snapshot = {
        "phase": tree._g_phase,
        "completed_root_visits": tree.total_visits,
        "pending_root_visits": int(tree._root.child_vloss.sum()),
        "initial_candidates": len(initial_candidates),
        "surviving_candidates": len(candidates_before_results),
        "pending_tokens": len(tokens),
    }
    assert eliminated and tree.total_visits == 0
    # Assign the highest root value to one already-eliminated candidate.
    delayed_best = eliminated[0]
    priors, values = constant_eval(states, 1.0)
    for i, token in enumerate(tokens):
        if token[1][0][1] == delayed_best:
            values[i] = -1.0
    tree.apply_results(tokens, priors, values)
    assert tree._root.child_W[delayed_best] / tree._root.child_N[delayed_best] == 1.0
    return {
        "revision": revision,
        "description": "Sequential Halving eliminates candidates using in-flight visits before their Q values are returned.",
        "before_results": snapshot,
        "best_delayed_candidate_was_eliminated": delayed_best not in tree._g_cand,
        "delayed_candidate_root_q": float(tree._root.child_W[delayed_best] / tree._root.child_N[delayed_best]),
        "note": "This proves a defect in the removed implementation; it does not establish the cause of previous training outcomes or predict a corrected implementation's strength.",
    }


if __name__ == "__main__":
    report = {
        "draw_backup": draw_probe(),
        "arena_gate": gate_probe(),
        "historical_gumbel": historical_gumbel_probe(),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    output = Path(__file__).with_suffix(".json")
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
