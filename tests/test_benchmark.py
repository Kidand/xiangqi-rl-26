"""Paired attribution, historical openings, frozen search semantics, and CPU E2E."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from rl import benchmark as bm
from rl.config import MCTSConfig, ModelConfig
from xiangqi.board import Board
from xiangqi.constants import ACTION_SIZE, START_FEN, move_to_action, uci_to_move


def _record(moves=None, **overrides):
    return {"format": "xiangqi-record-v1", "start_fen": START_FEN,
            "red": "selfplay-iter100", "black": "selfplay-iter100",
            "moves": ["a3a4", "a6a5"] if moves is None else moves, **overrides}


def _records_file(tmp_path, records, name="openings.json"):
    path = tmp_path / name
    payload = "\n".join(json.dumps(r) for r in records) if name.endswith(".jsonl") else json.dumps(records)
    path.write_text(payload, encoding="utf-8")
    return path


@pytest.mark.parametrize("suffix", ["json", "jsonl"])
def test_openings_deduplicate_and_select_reproducibly(tmp_path, suffix):
    records = [_record(), _record(["a3a4", "a6a5", "c3c4"]),
               _record(["c3c4", "c6c5"]), _record(["e3e4", "e6e5"]),
               _record(["a3a4"])]
    path = _records_file(tmp_path, records, f"openings.{suffix}")
    first, selection = bm.load_openings([path], prefix_plies=2, max_openings=2, seed=17)
    again, _ = bm.load_openings([path], prefix_plies=2, max_openings=2, seed=17)
    assert first == again
    assert len({o.fingerprint for o in first}) == 2
    assert selection["unique_prefixes"] == 3
    assert selection["duplicate_prefixes_excluded"] == 1
    assert selection["short_records_excluded"] == 1
    assert selection["sources"][0]["sha256"] == bm.sha256_file(path)
    assert selection["selected_fingerprints"] == [o.fingerprint for o in first]


def test_replay_preserves_snapshots_repetition_and_counters(tmp_path):
    # Reversible horse moves revisit the initial position once; repetition and
    # historical input planes would be lost by reconstructing the final FEN.
    moves = ["b0c2", "b9c7", "c2b0", "c7b9"]
    path = _records_file(tmp_path, [_record(moves)])
    openings, _ = bm.load_openings([path], prefix_plies=4)
    board = openings[0].replay(400, 120)
    assert len(board._zhist) == 5
    assert len(board._undo) == 4
    assert board._zhist[0] == board._zhist[-1]
    assert board.snapshots(2)[1] is not None
    assert board.halfmove == 4
    assert board.last_move == uci_to_move("c7b9")
    assert Board(board.fen()).snapshots(2)[1] is None


@pytest.mark.parametrize("record, match", [
    (_record(format="other"), "expected xiangqi-record-v1"),
    (_record(red="human"), "source must be labeled"),
    (_record(start_fen=None), "start_fen is required"),
    (_record(start_fen="bad-fen"), "invalid opening"),
    (_record(start_fen="4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1"), "side not to move is in check"),
    (_record(moves="a3a4"), "moves must be a list"),
    (_record(["a0a9", "a6a5"]), "illegal opening move"),
    (_record(["xxxx", "a6a5"]), "invalid opening move"),
])
def test_invalid_openings_are_rejected(tmp_path, record, match):
    path = _records_file(tmp_path, [record])
    with pytest.raises(ValueError, match=match):
        bm.load_openings([path], prefix_plies=2)


def test_terminal_prefix_and_insufficient_unique_openings_rejected(tmp_path):
    path = _records_file(tmp_path, [_record()])
    with pytest.raises(ValueError, match="terminal opening"):
        bm.load_openings([path], prefix_plies=2, no_capture_plies=2)
    with pytest.raises(ValueError, match="only 1 unique"):
        bm.load_openings([path], prefix_plies=2, max_openings=2)
    with pytest.raises(ValueError, match="leave room"):
        bm.load_openings([path], prefix_plies=2, max_plies=2)
    with pytest.raises(ValueError, match="no usable"):
        bm.load_openings([path], prefix_plies=3)


def test_source_changes_during_read_are_rejected(tmp_path, monkeypatch):
    path = _records_file(tmp_path, [_record()])
    original_read = bm.read_records

    def changing_read(path_):
        records = original_read(path_)
        path_.write_text(json.dumps([_record(), _record()]), encoding="utf-8")
        return records

    monkeypatch.setattr(bm, "read_records", changing_read)
    with pytest.raises(ValueError, match="record file changed"):
        bm.load_openings([path], prefix_plies=2)


def test_pair_statistics_use_sample_variance_and_guard_degenerate_results():
    stats = bm.paired_score_stats([0.0, 0.5, 1.0])
    assert stats["score"] == 0.5
    assert stats["pair_score_se"] == pytest.approx(math.sqrt(0.25 / 3))
    assert stats["score_ci95_approx"] == [0.0, 1.0]
    assert stats["ci_estimable"] is True
    for scores in ([1.0], [1.0] * 100, [0.5] * 80):
        stats = bm.paired_score_stats(scores)
        assert stats["score_ci95_approx"] is None
        assert stats["ci_estimable"] is False
        assert "cannot reliably estimate" in stats["limitations"][-1]
    for scores in ([], [float("nan")], [-0.1], [1.1]):
        with pytest.raises(ValueError):
            bm.paired_score_stats(scores)


@pytest.mark.parametrize("candidate_is_red", [True, False])
def test_play_one_swaps_colors_preserves_history_and_independent_draw(tmp_path, monkeypatch, candidate_is_red):
    path = _records_file(tmp_path, [_record()])
    openings, _ = bm.load_openings([path], prefix_plies=2, max_plies=4)
    opening = openings[0]
    candidate_eval, baseline_eval = object(), object()
    cmcts = MCTSConfig(draw_value=0.0)
    bmcts = MCTSConfig(draw_value=-0.2)
    calls = []

    def fake_search(board, eval_fn, cfg, sims, add_noise, history_steps):
        calls.append((eval_fn, cfg.draw_value, sims, add_noise, history_steps,
                      len(board._zhist), board.snapshots(2)))
        counts = np.zeros(ACTION_SIZE)
        counts[move_to_action(min(board.legal_moves()))] = 1
        return counts, 0.0

    monkeypatch.setattr(bm, "search", fake_search)
    game = bm._play_one(opening, candidate_eval, baseline_eval, candidate_is_red,
                        cmcts, bmcts, 2, 3, 5, 4, 120)
    first = (candidate_eval, 0.0, 2) if candidate_is_red else (baseline_eval, -0.2, 3)
    second = (baseline_eval, -0.2, 3) if candidate_is_red else (candidate_eval, 0.0, 2)
    for call, expected in zip(calls, (first, second)):
        assert (call[0], call[1], call[4]) == expected
        assert call[2:4] == (5, False)
        assert call[-1][1] is not None
    assert [c[5] for c in calls] == [3, 4]
    assert game["moves"][:2] == list(opening.moves)
    assert game["plies"] == 4
    assert game["result"] == "1/2-1/2"
    assert game["candidate_score"] == 0.5
    assert len(opening.replay(4, 120)._zhist) == 3  # next paired game starts fresh


@pytest.mark.parametrize("result,is_red,score", [
    ("1-0", True, 1.0), ("1-0", False, 0.0),
    ("0-1", True, 0.0), ("0-1", False, 1.0),
    ("1/2-1/2", True, 0.5), ("1/2-1/2", False, 0.5),
])
def test_candidate_score_attribution(result, is_red, score):
    assert bm._candidate_score(result, is_red) == score


def test_benchmark_pairs_and_output_metadata(tmp_path, monkeypatch):
    path = _records_file(tmp_path, [_record(), _record(["c3c4", "c6c5"])])
    openings, selection = bm.load_openings([path], prefix_plies=2)
    candidate = tmp_path / "candidate.pt"
    baseline = tmp_path / "0716.pt"
    candidate.write_bytes(b"candidate")
    baseline.write_bytes(b"frozen-baseline")

    class FakeModel:
        def parameters(self):
            return []

    monkeypatch.setattr(bm, "load_checkpoint", lambda *a, **kw: (
        FakeModel(), {"model_config": dataclasses.asdict(ModelConfig()), "iteration": 123}))
    monkeypatch.setattr(bm, "build_eval_fn", lambda *a: object())
    calls = []
    results = iter(["1-0", "0-1", "0-1", "1/2-1/2"])

    def fake_game(opening, ceval, beval, candidate_is_red, cmcts, bmcts, *args):
        calls.append((opening.fingerprint, candidate_is_red, cmcts, bmcts))
        result = next(results)
        return {"result": result, "candidate_score": bm._candidate_score(result, candidate_is_red)}

    monkeypatch.setattr(bm, "_play_one", fake_game)
    progress = []
    original = MCTSConfig(draw_value=-0.2)
    report = bm.benchmark(candidate, baseline, openings,
                          candidate_mcts=MCTSConfig(draw_value=0), baseline_mcts=original,
                          selection=selection, sims=7, progress=progress.append)
    assert [(c[0], c[1]) for c in calls] == [(o.fingerprint, red) for o in openings for red in (True, False)]
    assert [p["score"] for p in report["pairs"]] == [1.0, 0.25]
    assert report["statistics"]["score"] == 0.625
    assert report["statistics"]["pair_score_se"] == pytest.approx(0.375)
    assert report["statistics"]["wins"] == 2
    assert report["statistics"]["losses"] == 1
    assert len(progress) == 4
    assert report["baseline"]["sha256"] == bm.sha256_file(baseline)
    assert report["search"]["candidate"]["draw_value"] == 0
    assert report["search"]["baseline"]["draw_value"] == -0.2
    assert all(c[2].num_sims == c[3].num_sims == 7 for c in calls)
    assert report["search"]["candidate"]["dirichlet_eps"] == 0
    assert original.dirichlet_eps == 0.25  # caller config is not mutated
    assert report["opening_selection"] == selection
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="duplicate opening"):
        bm.benchmark(candidate, baseline, [openings[0], openings[0]],
                     candidate_mcts=original, baseline_mcts=original)


def test_cli_real_tiny_model_end_to_end(tmp_path, capsys):
    import torch

    from rl.model import create_model, save_checkpoint
    from scripts.benchmark_models import main

    path = _records_file(tmp_path, [_record()])
    mc = ModelConfig(blocks=1, filters=8, history_steps=2)
    torch.manual_seed(7)
    model = create_model(mc)
    checkpoint = tmp_path / "tiny.pt"
    save_checkpoint(model, mc, checkpoint, iteration=716)
    output = tmp_path / "benchmark.json"
    previous_threads = torch.get_num_threads()
    try:
        assert main(["--candidate", str(checkpoint), "--baseline", str(checkpoint),
                     "--records", str(path), "--candidate-draw-value", "0",
                     "--baseline-draw-value", "-0.2", "--opening-plies", "2",
                     "--pairs", "1", "--sims", "2", "--max-plies", "4",
                     "--out", str(output)]) == 0
    finally:
        torch.set_num_threads(previous_threads)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["statistics"]["games"] == 2
    assert report["statistics"]["score"] == 0.5
    assert report["statistics"]["score_ci95_approx"] is None
    assert report["candidate"]["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert report["candidate"]["model_config"] == dataclasses.asdict(mc)
    assert report["candidate"]["sha256"] == report["baseline"]["sha256"]
    assert [g["candidate_color"] for g in report["games"]] == ["red", "black"]
    for game in report["games"]:
        assert game["moves"][:2] == ["a3a4", "a6a5"]
        board = Board(game["start_fen"], max_plies=4)
        for text in game["moves"]:
            move = uci_to_move(text)
            assert move in board.legal_moves()
            board.push(move)
        assert board.fen() == game["final_fen"]
        assert board.result() == game["result"]
    assert "[2/2]" in capsys.readouterr().out
