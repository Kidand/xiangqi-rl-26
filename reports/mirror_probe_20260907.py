"""Read-only model diagnostic; writes its JSON report, never trains or edits a model.

Run from the repository root:
    .venv/Scripts/python.exe reports/mirror_probe_20260907.py

Horizontal reflection is a Xiangqi symmetry, but policy/value disagreement alone
does not establish a strength defect or predict an Elo gain from augmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.encoding import encode_board, move_to_policy_index
from rl.heuristics import material_score
from rl.model import load_checkpoint
from xiangqi.board import Board
from xiangqi.constants import uci_to_move
from xiangqi.fen import format_fen


def mirror_square(square: int) -> int:
    return (square // 9) * 9 + 8 - square % 9


def mirror_move(move: tuple[int, int]) -> tuple[int, int]:
    return mirror_square(move[0]), mirror_square(move[1])


def mirror_action(action: int) -> int:
    return mirror_square(action // 90) * 90 + mirror_square(action % 90)


def mirror_start(board: Board) -> Board:
    squares = [board.squares[mirror_square(s)] for s in range(90)]
    return Board.from_fen(format_fen(squares, board.side_to_move, board.halfmove, board.fullmove))


def checked_case(board: Board, mirrored: Board, counts: dict, label: dict):
    legal = board.legal_moves()
    assert {mirror_move(m) for m in legal} == set(mirrored.legal_moves())
    assert board.result_and_termination() == mirrored.result_and_termination()
    assert material_score(board) == material_score(mirrored)
    state = encode_board(board, 2)
    mirrored_state = encode_board(mirrored, 2)
    assert np.array_equal(state[:, :, ::-1], mirrored_state)
    indices = [move_to_policy_index(m, board.side_to_move) for m in legal]
    assert [mirror_action(i) for i in indices] == [
        move_to_policy_index(mirror_move(m), mirrored.side_to_move) for m in legal
    ]
    counts["positions"] += 1
    counts["legal_action_index_checks"] += len(legal)
    counts["red_positions" if board.side_to_move == 1 else "black_positions"] += 1
    return (state, mirrored_state, indices, label)


def random_cases():
    rng = random.Random(20260907)
    counts = dict(positions=0, legal_action_index_checks=0, red_positions=0, black_positions=0)
    cases = []
    for game in range(8):
        board = Board()
        mirrored = mirror_start(board)
        for ply in range(60):
            case = checked_case(board, mirrored, counts, dict(game=game, ply=ply))
            legal = board.legal_moves()
            if ply % 10 == 5 and legal:
                cases.append(case)
            if not legal or board.result() is not None:
                break
            move = rng.choice(legal)
            board.push(move)
            mirrored.push(mirror_move(move))
    return cases, counts


def recorded_cases(path: Path):
    record = json.loads(path.read_text(encoding="utf-8"))
    board = Board.from_fen(record["start_fen"])
    mirrored = mirror_start(board)
    counts = dict(positions=0, legal_action_index_checks=0, red_positions=0, black_positions=0)
    cases = []
    for ply, text_move in enumerate(record["moves"]):
        cases.append(checked_case(board, mirrored, counts, dict(ply=ply)))
        move = uci_to_move(text_move)
        assert move in board.legal_moves()
        board.push(move)
        mirrored.push(mirror_move(move))
    assert board.result_and_termination() == mirrored.result_and_termination()
    metadata = {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "record_meta": record.get("meta"),
        "red": record.get("red"),
        "black": record.get("black"),
        "result": record.get("result"),
        "termination": record.get("termination"),
        "model_identity_caveat": "Record names both players ai and records mode/sims/time only; no model checkpoint identity or hash. Its date does not prove best0716 produced the game.",
    }
    return cases, counts, metadata


def probe(model, cases):
    states = np.stack([state for case in cases for state in case[:2]])
    logits_parts, values_parts = [], []
    with torch.inference_mode():
        for start in range(0, len(states), 32):
            logits, values = model(torch.from_numpy(states[start:start + 32]))
            logits_parts.append(logits.numpy())
            values_parts.append(values.numpy())
    logits = np.concatenate(logits_parts)
    values = np.concatenate(values_parts)
    rows = []
    for k, (_, _, indices, label) in enumerate(cases):
        indices = np.asarray(indices)
        reflected_indices = np.asarray([mirror_action(int(i)) for i in indices])
        p = logits[2*k, indices]
        mp = logits[2*k+1, reflected_indices]
        p = np.exp(p - p.max()); p /= p.sum()
        mp = np.exp(mp - mp.max()); mp /= mp.sum()
        rows.append({
            **label,
            "policy_tv": float(np.abs(p - mp).sum() / 2),
            "top1_agrees": bool(p.argmax() == mp.argmax()),
            "value": float(values[2*k]),
            "mirrored_value": float(values[2*k+1]),
            "value_absdiff": float(abs(values[2*k] - values[2*k+1])),
        })
    return {
        "positions": len(cases),
        "policy_tv_mean": float(np.mean([r["policy_tv"] for r in rows])),
        "policy_tv_median": float(np.median([r["policy_tv"] for r in rows])),
        "top1_agreement": float(np.mean([r["top1_agrees"] for r in rows])),
        "value_absdiff_mean": float(np.mean([r["value_absdiff"] for r in rows])),
        "value_absdiff_max": max(r["value_absdiff"] for r in rows),
        "per_position": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "ckpts/best0716.pt")
    parser.add_argument("--record", type=Path, default=ROOT / "records/gui/20260716-160758_ava_和棋.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/mirror_probe_20260907.json")
    args = parser.parse_args()
    model_path = args.model.resolve()
    report_model_path = (
        model_path.relative_to(ROOT).as_posix()
        if model_path.is_relative_to(ROOT) else model_path.name
    )
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(args.model, "cpu")
    assert checkpoint["model_config"]["history_steps"] == 2
    random_samples, random_checks = random_cases()
    record_samples, record_checks, record_metadata = recorded_cases(args.record)
    result = {
        "date": "2026-09-07",
        "model": {
            "path": report_model_path,
            "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "iteration": checkpoint["iteration"],
            "config": checkpoint["model_config"],
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "method": "Mirror all 31 input planes across columns; mirror both action endpoints; side and value target stay unchanged. Probabilities renormalized over legal moves. CPU float32 eval mode, no search or training.",
        "random_probe": {"checks": random_checks, "results": probe(model, random_samples)},
        "recorded_probe": {"source": record_metadata, "checks": record_checks, "results": probe(model, record_samples)},
        "limitations": [
            "Random legal trajectories may be outside the training distribution.",
            "The recorded sample is one 54-ply game, and adjacent positions are correlated.",
            "Policy asymmetry may choose equally good mirror alternatives; neither disagreement metric establishes Elo loss.",
            "No held-out game-strength test or augmentation-training experiment was performed.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {name: {k: v for k, v in result[name]["results"].items() if k != "per_position"} for name in ("random_probe", "recorded_probe")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
