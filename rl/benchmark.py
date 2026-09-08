"""Frozen-baseline evaluation on paired, replayed self-play openings.

This is independent of the training promotion gate. Opening prefixes preserve
the full Board history; each prefix is played twice with candidate colors
swapped. Scores and uncertainty are aggregated by opening pair, not by game.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from rl.config import MCTSConfig
from rl.mcts import search
from rl.model import load_checkpoint
from rl.selfplay import build_eval_fn, sample_action_from_counts
from xiangqi.board import Board
from xiangqi.constants import RED, move_to_uci, uci_to_move
from xiangqi.records import RECORD_FORMAT, read_records


@dataclasses.dataclass(frozen=True)
class Opening:
    start_fen: str
    moves: tuple[str, ...]
    fingerprint: str
    source: dict

    def replay(self, max_plies: int, no_capture_plies: int) -> Board:
        """Rebuild history from the start, never from the prefix's final FEN."""
        board = Board(self.start_fen, max_plies=max_plies, no_capture_plies=no_capture_plies)
        if board.is_in_check(-board.side_to_move):
            raise ValueError("illegal opening start_fen: the side not to move is in check")
        for ply, text in enumerate(self.moves, 1):
            if board.result() is not None:
                raise ValueError(f"terminal opening before ply {ply}: {self.fingerprint}")
            try:
                move = uci_to_move(text)
            except (ValueError, TypeError, IndexError) as exc:
                raise ValueError(f"invalid opening move at ply {ply}: {text!r}") from exc
            if move not in board.legal_moves():
                raise ValueError(f"illegal opening move at ply {ply}: {text!r}")
            board.push(move)
        if board.result() is not None:
            raise ValueError(f"terminal opening: {self.fingerprint}")
        return board


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_rules(max_plies: int, no_capture_plies: int) -> None:
    if max_plies < 1 or no_capture_plies < 1:
        raise ValueError("max_plies and no_capture_plies must be positive")


def load_openings(
    paths: Sequence[str | Path],
    *,
    prefix_plies: int = 12,
    max_openings: int | None = None,
    seed: int = 42,
    max_plies: int = 400,
    no_capture_plies: int = 120,
) -> tuple[list[Opening], dict]:
    """Validate and deduplicate prefixes, then sample without replacement.

    Input records must carry the training writer's ``selfplay-*`` labels on
    both colors. These are a provenance declaration, not proof of origin.
    Short records are excluded and counted; malformed/illegal/terminal prefixes
    fail loudly. Only the requested prefix is used or validated, not its suffix.
    File order is canonicalized so shell glob ordering does not change selection.
    """
    _validate_rules(max_plies, no_capture_plies)
    if prefix_plies < 0 or (max_openings is not None and max_openings < 1):
        raise ValueError("prefix_plies must be nonnegative and max_openings positive")
    if prefix_plies >= max_plies:
        raise ValueError("prefix_plies must leave room below max_plies for evaluation")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    files = sorted({Path(path).resolve() for path in paths})
    if not files:
        raise ValueError("at least one self-play record file is required")
    unique: dict[str, Opening] = {}
    sources = []
    total = short = duplicate = 0
    for path in files:
        file_hash = sha256_file(path)
        records = read_records(path)
        if sha256_file(path) != file_hash:
            raise ValueError(f"record file changed while reading; use frozen self-play files: {path}")
        sources.append({"path": str(path), "sha256": file_hash, "records": len(records)})
        for index, record in enumerate(records):
            total += 1
            label = f"{path}, record {index}"
            if not isinstance(record, dict) or record.get("format") != RECORD_FORMAT:
                raise ValueError(f"expected {RECORD_FORMAT}: {label}")
            if not all(isinstance(record.get(color), str)
                       and record[color].startswith("selfplay-") for color in ("red", "black")):
                raise ValueError(f"opening source must be labeled selfplay-* for both colors: {label}")
            moves = record.get("moves")
            if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
                raise ValueError(f"moves must be a list of UCCI strings: {label}")
            if len(moves) < prefix_plies:
                short += 1
                continue
            try:
                if not isinstance(record.get("start_fen"), str):
                    raise ValueError("start_fen is required")
                board = Board(record["start_fen"], max_plies=max_plies,
                              no_capture_plies=no_capture_plies)
                start_fen = board.fen()
                prefix = tuple(moves[:prefix_plies])
                payload = json.dumps([start_fen, prefix], separators=(",", ":"))
                fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                opening = Opening(start_fen, prefix, fingerprint, {
                    "path": str(path), "file_sha256": file_hash, "record_index": index,
                    "red": record["red"], "black": record["black"],
                })
                opening.replay(max_plies, no_capture_plies)
            except (ValueError, TypeError, IndexError) as exc:
                raise ValueError(f"invalid opening in {label}: {exc}") from exc
            if fingerprint in unique:
                duplicate += 1
            else:
                unique[fingerprint] = opening
    pool = list(unique.values())
    if not pool:
        raise ValueError("no usable self-play opening prefixes")
    if max_openings is not None and max_openings > len(pool):
        raise ValueError(f"requested {max_openings} openings, only {len(pool)} unique prefixes available")
    size = len(pool) if max_openings is None else max_openings
    indices = np.random.default_rng(seed).choice(len(pool), size=size, replace=False).tolist()
    selected = [pool[i] for i in indices]
    selection = {
        "seed": seed, "prefix_plies": prefix_plies,
        "method": "numpy.default_rng.choice_without_replacement",
        "sources": sources, "total_records": total, "short_records_excluded": short,
        "duplicate_prefixes_excluded": duplicate, "unique_prefixes": len(pool),
        "selected_pool_indices": indices,
        "selected_fingerprints": [o.fingerprint for o in selected],
        "provenance": "both record player labels begin with selfplay-; metadata is not proof of origin",
    }
    return selected, selection


def paired_score_stats(pair_scores: Sequence[float]) -> dict:
    """Sample SE over pair means; avoid zero-width CI for degenerate samples."""
    scores = [float(x) for x in pair_scores]
    if not scores or any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
        raise ValueError("pair_scores must contain finite scores in [0, 1]")
    n = len(scores)
    score = statistics.mean(scores)
    variance = statistics.variance(scores) if n > 1 else None
    se = math.sqrt(variance / n) if variance is not None else None
    estimable = n > 1 and variance > 0
    ci = [max(0.0, score - 1.96 * se), min(1.0, score + 1.96 * se)] if estimable else None
    notes = ["Pairs are the sampling unit. The normal 95% interval assumes independent, representative openings.",
             "Shared prefixes, related self-play games, repeated candidate selection and hardware differences can invalidate that approximation."]
    if n < 30:
        notes.append("Fewer than 30 opening pairs: normal-interval accuracy is limited; do not infer superiority from this run alone.")
    if not estimable:
        notes.append("One pair or zero observed pair variance cannot reliably estimate uncertainty; no confidence interval or superiority conclusion is reported.")
    return {"pairs": n, "score": score, "pair_score_se": se,
            "score_ci95_approx": ci, "ci_estimable": estimable,
            "ci_method": "pair sample variance (ddof=1), normal mean +/- 1.96 SE",
            "limitations": notes}


def _candidate_score(result: str, candidate_is_red: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result not in ("1-0", "0-1"):
        raise ValueError(f"unexpected game result: {result!r}")
    return float((result == "1-0") == candidate_is_red)


def _play_one(
    opening: Opening,
    candidate_eval: Callable,
    baseline_eval: Callable,
    candidate_is_red: bool,
    candidate_mcts: MCTSConfig,
    baseline_mcts: MCTSConfig,
    candidate_history: int,
    baseline_history: int,
    sims: int,
    max_plies: int,
    no_capture_plies: int,
) -> dict:
    board = opening.replay(max_plies, no_capture_plies)
    moves = list(opening.moves)
    rng = np.random.default_rng(0)  # tau=0 never consumes RNG
    while board.result() is None:
        use_candidate = (board.side_to_move == RED) == candidate_is_red
        counts, _ = search(
            board, candidate_eval if use_candidate else baseline_eval,
            candidate_mcts if use_candidate else baseline_mcts, sims,
            add_noise=False,
            history_steps=candidate_history if use_candidate else baseline_history,
        )
        move = sample_action_from_counts(counts, 0.0, rng)
        legal = board.legal_moves()
        if move is None:
            if not legal:
                raise RuntimeError("nonterminal board has no legal moves")
            move = min(legal)  # deterministic tie-breaking, also for empty visits
        elif move not in legal:
            raise RuntimeError(f"search returned illegal move: {move}")
        moves.append(move_to_uci(move))
        board.push(move)
    result, termination = board.result_and_termination()
    return {
        "format": RECORD_FORMAT, "start_fen": opening.start_fen,
        "moves": moves, "result": result, "termination": termination, "plies": len(moves),
        "red": "candidate" if candidate_is_red else "baseline",
        "black": "baseline" if candidate_is_red else "candidate",
        "candidate_color": "red" if candidate_is_red else "black",
        "candidate_score": _candidate_score(result, candidate_is_red),
        "opening_fingerprint": opening.fingerprint, "opening_plies": len(opening.moves),
        "final_fen": board.fen(),
    }


def _effective_mcts(cfg: MCTSConfig, sims: int) -> MCTSConfig:
    if not math.isfinite(cfg.draw_value) or not -1 <= cfg.draw_value <= 1:
        raise ValueError("draw_value must be finite and in [-1, 1]")
    if cfg.batch_size < 1:
        raise ValueError("MCTS batch_size must be positive")
    return dataclasses.replace(cfg, num_sims=sims, dirichlet_eps=0.0,
                               temp_moves=0, temperature=0.0, temp_final=0.0,
                               num_sims_late=0, num_sims_late_start_iter=0)


def benchmark(
    candidate_ckpt: str | Path,
    baseline_ckpt: str | Path,
    openings: Sequence[Opening],
    *,
    candidate_mcts: MCTSConfig,
    baseline_mcts: MCTSConfig,
    sims: int = 400,
    max_plies: int = 400,
    no_capture_plies: int = 120,
    device: str = "cpu",
    selection: dict | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run serial paired games with independently configured model searches.

    Model history depth comes from each checkpoint, not a training YAML. Both
    players share sims and adjudication rules; draw_value can retain the frozen
    baseline's old semantics. The function never promotes or modifies models.
    """
    import torch

    _validate_rules(max_plies, no_capture_plies)
    if sims < 1 or not openings:
        raise ValueError("positive sims and at least one opening are required")
    if len({o.fingerprint for o in openings}) != len(openings):
        raise ValueError("duplicate opening fingerprints are not independent pairs")
    for opening in openings:
        opening.replay(max_plies, no_capture_plies)
    cmcts = _effective_mcts(candidate_mcts, sims)
    bmcts = _effective_mcts(baseline_mcts, sims)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    models = []
    evals = []
    histories = []
    for path in (candidate_ckpt, baseline_ckpt):
        path = Path(path).resolve()
        before = sha256_file(path)
        model, ckpt = load_checkpoint(path, device=str(dev))
        if sha256_file(path) != before:
            raise RuntimeError(f"checkpoint changed while loading: {path}")
        mc = dict(ckpt["model_config"])
        mc.setdefault("history_steps", 2)
        mc.setdefault("se", False)
        histories.append(int(mc["history_steps"]))
        models.append({"path": str(path), "sha256": before, "model_config": mc,
                       "parameter_count": sum(p.numel() for p in model.parameters()),
                       "iteration": ckpt.get("iteration")})
        evals.append(build_eval_fn(model, str(dev)))
    games, pairs = [], []
    for pair_index, opening in enumerate(openings):
        pair_games = []
        for candidate_is_red in (True, False):
            game = _play_one(opening, *evals, candidate_is_red, cmcts, bmcts,
                             *histories, sims, max_plies, no_capture_plies)
            game["game_index"] = len(games)
            game["pair_index"] = pair_index
            games.append(game)
            pair_games.append(game)
            if progress is not None:
                progress(game)
        pairs.append({"pair_index": pair_index,
                      "opening": dataclasses.asdict(opening),
                      "game_indices": [g["game_index"] for g in pair_games],
                      "candidate_scores": [g["candidate_score"] for g in pair_games],
                      "score": sum(g["candidate_score"] for g in pair_games) / 2})
    statistics_ = paired_score_stats([p["score"] for p in pairs])
    statistics_.update({"games": len(games),
                        "wins": sum(g["candidate_score"] == 1 for g in games),
                        "draws": sum(g["candidate_score"] == 0.5 for g in games),
                        "losses": sum(g["candidate_score"] == 0 for g in games)})
    return {
        "format": "xiangqi-paired-benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": models[0], "baseline": models[1],
        "search": {"candidate": dataclasses.asdict(cmcts), "baseline": dataclasses.asdict(bmcts),
                   "sims_per_move": sims, "add_noise": False, "move_selection": "visit_argmax",
                   "history_steps_source": "each checkpoint model_config"},
        "rules": {"max_game_plies": max_plies, "no_capture_draw_plies": no_capture_plies,
                  "opening_plies_count_toward_limit": True, "resign_enabled": False},
        "runtime": {"device": str(dev), "python": platform.python_version(),
                    "torch": str(torch.__version__), "numpy": np.__version__,
                    "execution": "serial", "precision": "float32, no autocast"},
        "opening_selection": selection,
        "pairs": pairs, "games": games, "statistics": statistics_,
    }
