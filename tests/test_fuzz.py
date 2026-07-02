# -*- coding: utf-8 -*-
"""Fuzz & boundary tests for the Xiangqi rule engine (DESIGN.md §4).

Marks
-----
``slow``  – tests that take more than a few seconds; excluded from fast suite.

Coverage (invariants verified on every random game position)
------------------------------------------------------------
* Zobrist: board._zobrist == board._compute_zobrist() after every push.
* FEN roundtrip: Board.from_fen(board.fen()).fen() == board.fen().
* push/pop restoration: fen / zobrist / halfmove / fullmove / side_to_move /
  last_move / king_pos / _snaps / _zhist all restored exactly.
* legal_moves soundness: every move in legal_moves() is legal (push → king
  not in check → pop). No move outside legal_moves() can keep king safe.
* is_in_check consistency: board.is_in_check(side) == opponent's pseudo-legal
  moves can reach side's king.
* Terminal idempotency: result() / termination() return the same value on
  repeated calls; legal_moves() still callable after terminal.
* Snapshot correctness: _snaps[-1] == bytes(p+7 for p in board.squares).

Boundary / unit tests (fast)
-----------------------------
* no_capture_plies parameter effect.
* Snapshots: initial, after push, after pop, after copy, n > history.
* Chinese notation: 前/中/后 (same-column 2 / 3 pawns, red and black).
* is_in_check vs brute-force for curated positions.
"""

from __future__ import annotations

import random

import pytest

from xiangqi.board import Board
from xiangqi.constants import (
    BLACK,
    CANNON,
    KING,
    PAWN,
    RED,
    ROOK,
    move_to_uci,
    uci_to_move,
)
from xiangqi.fen import format_fen
from xiangqi.notation import move_to_chinese

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brute_is_in_check(board: Board, side: int) -> bool:
    """Independent check: opponent pseudo-legal moves can reach side's king."""
    king_sq = board.king_pos.get(side)
    if king_sq is None:
        return False
    opp = -side
    saved = board.side_to_move
    board.side_to_move = opp
    opp_pseudo = board._pseudo_moves()
    board.side_to_move = saved
    return any(to == king_sq for _, to in opp_pseudo)


def _brute_legal_moves(board: Board) -> list[tuple[int, int]]:
    """Independent legal-move generator: push each pseudo-legal, check safety, pop."""
    me = board.side_to_move
    result = []
    for frm, to in board._pseudo_moves():
        board.push((frm, to))
        in_check = board.is_in_check(me)
        board.pop()
        if not in_check:
            result.append((frm, to))
    return result


def _empty_squares() -> list[int]:
    return [0] * 90


def _make_board_from_pieces(red_pieces, black_pieces, side=RED) -> Board:
    """Build a Board from {sq: piece_type} dicts (piece_type unsigned)."""
    squares = _empty_squares()
    for sq, pt in red_pieces.items():
        squares[sq] = pt
    for sq, pt in black_pieces.items():
        squares[sq] = -pt
    return Board.from_fen(format_fen(squares, side, 0, 1))


# ---------------------------------------------------------------------------
# Fuzz test (50 games, slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fuzz_50_games():
    """50 random games; invariants checked after every push."""
    rng = random.Random(20260626)
    games = 50
    max_plies_per_game = 200

    for game_idx in range(games):
        b = Board(max_plies=max_plies_per_game)

        for ply in range(max_plies_per_game + 5):
            # 1. Zobrist consistency
            assert b._zobrist == b._compute_zobrist(), (
                f"game {game_idx} ply {ply}: "
                f"stored={b._zobrist:#x} computed={b._compute_zobrist():#x}"
            )

            # 2. FEN roundtrip
            fen1 = b.fen()
            fen2 = Board.from_fen(fen1).fen()
            assert fen1 == fen2, f"game {game_idx} ply {ply}: FEN roundtrip failed"

            # 3. is_in_check consistency for both sides
            for side in (RED, BLACK):
                board_check = b.is_in_check(side)
                brute_check = _brute_is_in_check(b, side)
                assert board_check == brute_check, (
                    f"game {game_idx} ply {ply} side={side}: "
                    f"board={board_check} brute={brute_check} fen={b.fen()}"
                )

            # 4. Terminal idempotency
            if b.result() is not None:
                r1, r2 = b.result(), b.result()
                assert r1 == r2
                t1, t2 = b.termination(), b.termination()
                assert t1 == t2
                lm = b.legal_moves()  # must not raise
                if b.termination() in ("checkmate", "stalemate"):
                    assert lm == [], f"terminal but {len(lm)} legal moves"
                break

            # 5. legal_moves vs independent brute force
            lm_board = b.legal_moves()
            lm_brute = _brute_legal_moves(b)
            assert set(lm_board) == set(lm_brute), (
                f"game {game_idx} ply {ply} legal_moves mismatch:\n"
                f"  extra={set(lm_board)-set(lm_brute)}\n"
                f"  missing={set(lm_brute)-set(lm_board)}\n"
                f"  fen={b.fen()}"
            )

            if not lm_board:
                break

            # 6. push/pop full restoration
            move = rng.choice(lm_board)

            fen_before       = b.fen()
            z_before         = b._zobrist
            half_before      = b.halfmove
            full_before      = b.fullmove
            side_before      = b.side_to_move
            last_before      = b.last_move
            snaps_before     = b._snaps[:]
            zhist_before     = b._zhist[:]
            king_pos_before  = dict(b.king_pos)

            b.push(move)

            # After push: snapshot tail must match board state
            expected_snap = bytes(p + 7 for p in b.squares)
            assert b._snaps[-1] == expected_snap, (
                f"game {game_idx} ply {ply}: snapshot mismatch after push"
            )
            assert len(b._snaps) == len(snaps_before) + 1
            assert len(b._zhist) == len(zhist_before) + 1

            b.pop()

            assert b.fen()           == fen_before,      "fen mismatch after pop"
            assert b._zobrist        == z_before,         "zobrist mismatch after pop"
            assert b.halfmove        == half_before,      "halfmove mismatch after pop"
            assert b.fullmove        == full_before,      "fullmove mismatch after pop"
            assert b.side_to_move    == side_before,      "side_to_move mismatch"
            assert b.last_move       == last_before,      "last_move mismatch"
            assert b._snaps          == snaps_before,     "_snaps mismatch"
            assert b._zhist          == zhist_before,     "_zhist mismatch"
            assert b.king_pos        == king_pos_before,  "king_pos mismatch"

            # Replay the chosen move and continue
            b.push(move)


# ---------------------------------------------------------------------------
# no_capture_plies parameter
# ---------------------------------------------------------------------------

class TestNoCaptureParameter:
    def test_custom_no_capture_plies_triggers(self):
        """Board(no_capture_plies=10) terminates at halfmove==10."""
        b = Board.from_fen(
            "3k5/9/9/9/9/9/9/9/9/R3K4 w - - 9 1",
            no_capture_plies=10,
        )
        assert b.result() is None, "halfmove=9 should not yet be terminal"
        b.push(uci_to_move("a0a1"))  # non-capture: halfmove → 10
        assert b.result() == "1/2-1/2"
        assert b.termination() == "no_capture"

    def test_custom_no_capture_plies_not_triggered_early(self):
        """halfmove = threshold - 1 must NOT trigger."""
        b = Board.from_fen(
            "3k5/9/9/9/9/9/9/9/9/R3K4 w - - 8 1",
            no_capture_plies=10,
        )
        b.push(uci_to_move("a0a1"))  # halfmove → 9
        assert b.result() is None, f"halfmove=9 < 10 should be ongoing, got {b.result()}"

    def test_default_120_halfmoves(self):
        """Default limit of 120 half-moves triggers draw at exactly 120."""
        b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/R3K4 w - - 119 1")
        assert b.result() is None
        b.push(uci_to_move("a0a1"))  # → 120
        assert b.result() == "1/2-1/2"
        assert b.termination() == "no_capture"


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

class TestSnapshots:
    def test_initial_snapshot_is_correct(self):
        """Initial board snapshot matches board.squares encoding."""
        b = Board()
        snap = b.snapshots(1)[0]
        expected = bytes(p + 7 for p in b.squares)
        assert snap == expected

    def test_snapshots_returns_n_items(self):
        b = Board()
        for n in (1, 4, 8):
            out = b.snapshots(n)
            assert len(out) == n, f"snapshots({n}) returned {len(out)} items"

    def test_snapshots_none_for_insufficient_history(self):
        """From the start (1 history entry) snapshots(3) pads with None."""
        b = Board()
        out = b.snapshots(3)
        assert out[0] is not None, "index 0 (current) should be set"
        assert out[1] is None, "index 1 (one step back) should be None at start"
        assert out[2] is None, "index 2 should be None at start"

    def test_snapshot_after_push(self):
        b = Board()
        snap_initial = b.snapshots(1)[0]
        m = b.legal_moves()[0]
        b.push(m)
        out = b.snapshots(2)
        # index 0 = current (after push)
        expected_current = bytes(p + 7 for p in b.squares)
        assert out[0] == expected_current
        # index 1 = one step back = initial position
        assert out[1] == snap_initial

    def test_snapshot_after_pop(self):
        b = Board()
        snap_initial = b.snapshots(1)[0]
        m = b.legal_moves()[0]
        b.push(m)
        b.pop()
        out = b.snapshots(2)
        assert out[0] == snap_initial
        assert out[1] is None

    def test_snapshot_copy_independence(self):
        """Pushing on a copy does not affect the original's snapshots."""
        b = Board()
        snap_before = b.snapshots(3)[:]
        b2 = b.copy()
        b2.push(b2.legal_moves()[0])
        # Original must be unchanged
        assert b.snapshots(3) == snap_before


# ---------------------------------------------------------------------------
# Chinese notation: same-column pieces
# ---------------------------------------------------------------------------

class TestChineseNotationSameColumn:
    """前/中/后 disambiguation for pieces sharing a column."""

    def _board_with_red_pawns(self, rows, col=1):
        squares = _empty_squares()
        for r in rows:
            squares[r * 9 + col] = PAWN
        squares[0 * 9 + 4] = KING
        squares[9 * 9 + 3] = -KING
        return Board.from_fen(format_fen(squares, RED, 0, 1))

    def _board_with_black_pawns(self, rows, col=3):
        squares = _empty_squares()
        for r in rows:
            squares[r * 9 + col] = -PAWN
        squares[0 * 9 + 4] = KING
        squares[9 * 9 + 3] = -KING
        return Board.from_fen(format_fen(squares, BLACK, 0, 1))

    def test_two_red_pawns_front_back(self):
        """Two red pawns in same column → 前/后."""
        b = self._board_with_red_pawns([6, 7])  # row7 > row6, so row7 is front
        pawn_sqs = sorted([s for s, p in enumerate(b.squares) if p == PAWN])
        front = max(pawn_sqs)  # row7 (higher = closer to black = front for red)
        back  = min(pawn_sqs)  # row6

        cn_front = move_to_chinese(b, (front, front + 9))
        cn_back  = move_to_chinese(b, (back,  back + 9))

        assert cn_front.startswith("前"), f"front pawn got prefix: {cn_front}"
        assert cn_back.startswith("后"),  f"back pawn got prefix: {cn_back}"

    def test_three_red_pawns_front_mid_back(self):
        """Three red pawns in same column → 前/中/后."""
        b = self._board_with_red_pawns([5, 6, 7])
        pawn_sqs = sorted([s for s, p in enumerate(b.squares) if p == PAWN])
        front = max(pawn_sqs)   # row7
        mid   = sorted(pawn_sqs)[1]  # row6
        back  = min(pawn_sqs)   # row5

        cn_front = move_to_chinese(b, (front, front + 9))
        cn_mid   = move_to_chinese(b, (mid,   mid + 9))
        cn_back  = move_to_chinese(b, (back,  back + 9))

        assert cn_front.startswith("前"), f"front: {cn_front}"
        assert cn_mid.startswith("中"),   f"mid:   {cn_mid}"
        assert cn_back.startswith("后"),  f"back:  {cn_back}"

    def test_three_black_pawns_front_mid_back(self):
        """Three black pawns in same column → 前/中/后 (black perspective)."""
        b = self._board_with_black_pawns([2, 3, 4])  # all crossed
        pawn_sqs = sorted([s for s, p in enumerate(b.squares) if p == -PAWN])
        # For black: front = lowest row (closest to red)
        front = min(pawn_sqs)   # row2
        mid   = sorted(pawn_sqs)[1]  # row3
        back  = max(pawn_sqs)   # row4

        cn_front = move_to_chinese(b, (front, front - 9))
        cn_mid   = move_to_chinese(b, (mid,   mid - 9))
        cn_back  = move_to_chinese(b, (back,  back - 9))

        assert cn_front.startswith("前"), f"front: {cn_front}"
        assert cn_mid.startswith("中"),   f"mid:   {cn_mid}"
        assert cn_back.startswith("后"),  f"back:  {cn_back}"

    def test_two_black_pawns_front_back(self):
        """Two black pawns in same column → 前/后 (black perspective)."""
        b = self._board_with_black_pawns([2, 3])
        pawn_sqs = sorted([s for s, p in enumerate(b.squares) if p == -PAWN])
        front = min(pawn_sqs)  # row2 (lowest = front for black)
        back  = max(pawn_sqs)  # row3

        cn_front = move_to_chinese(b, (front, front - 9))
        cn_back  = move_to_chinese(b, (back,  back - 9))

        assert cn_front.startswith("前"), f"front: {cn_front}"
        assert cn_back.startswith("后"),  f"back:  {cn_back}"


# ---------------------------------------------------------------------------
# is_in_check vs brute-force for curated positions
# ---------------------------------------------------------------------------

class TestIsInCheckBruteForce:
    """Verify is_in_check() against an independent pseudo-move approach.

    Note: _brute_is_in_check() uses pseudo-legal moves and does NOT capture the
    飞将 (flying general) rule, which is a positional constraint not expressible
    as a single pseudo-legal capture.  Therefore tests that involve flying general
    check is_in_check() directly rather than comparing with the brute-force helper.
    """

    def test_rook_check(self):
        b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/3R1K3 b - - 0 1")
        assert b.is_in_check(BLACK) == _brute_is_in_check(b, BLACK)
        assert b.is_in_check(BLACK) is True

    def test_horse_check(self):
        # Horse at f7 (row7, col5), black king at e9 (row9, col4).
        # Horse move f7→e9: dr=+2, dc=-1; leg at f8 (row8, col5) must be empty.
        # Kings on different columns (e vs f) so no flying general.
        # FEN segments row9..row0: row9="4k4", row7="5N3", row0="5K3"
        b = Board.from_fen("4k4/9/5N3/9/9/9/9/9/9/5K3 b - - 0 1")
        assert b.is_in_check(BLACK) == _brute_is_in_check(b, BLACK)
        assert b.is_in_check(BLACK) is True

    def test_cannon_check(self):
        b = Board.from_fen("3k5/9/9/9/9/3P5/9/9/9/3C1K3 b - - 0 1")
        assert b.is_in_check(BLACK) == _brute_is_in_check(b, BLACK)
        assert b.is_in_check(BLACK) is True

    def test_flying_general_direct(self):
        """Flying general: two kings same file with no pieces between.

        This is a positional rule not captured by pseudo-legal moves, so we
        test is_in_check() directly without the brute-force helper.
        Both sides are in check simultaneously in this artificial position.
        """
        b = Board.from_fen("4k4/9/9/9/9/9/9/9/9/4K4 b - - 0 1")
        assert b.is_in_check(BLACK) is True, "BLACK should be in flying-general check"
        assert b.is_in_check(RED) is True,   "RED should be in flying-general check (symmetric)"

    def test_flying_general_blocked(self):
        """A pawn in the same file blocks the flying general.

        Red pawn at e5 (row5, col4) is between both kings in the e-file.
        Pawn distance from black king (e9) is 4 rows — not a direct pawn attack.
        No cannon is present, so neither king is in check.
        """
        # row9="4k4", row5="4P4", row0="4K4"
        b = Board.from_fen("4k4/9/9/9/4P4/9/9/9/9/4K4 b - - 0 1")
        assert b.is_in_check(BLACK) is False
        assert b.is_in_check(RED)   is False

    def test_not_in_check(self):
        b = Board.from_fen("3k5/9/3p5/9/9/9/9/9/9/3R1K3 b - - 0 1")
        assert b.is_in_check(BLACK) == _brute_is_in_check(b, BLACK)
        assert b.is_in_check(BLACK) is False

    def test_initial_no_check(self):
        b = Board()
        assert b.is_in_check(RED)   == _brute_is_in_check(b, RED)   is False
        assert b.is_in_check(BLACK) == _brute_is_in_check(b, BLACK) is False


# ---------------------------------------------------------------------------
# Terminal state machine (result/termination after game over)
# ---------------------------------------------------------------------------

class TestTerminalStateMachine:
    def test_result_idempotent_after_checkmate(self):
        b = Board.from_fen("R3k3R/9/9/9/9/9/9/9/9/4K4 b - - 0 1")
        assert b.result() == b.result() == "1-0"
        assert b.termination() == b.termination() == "checkmate"
        lm = b.legal_moves()   # must not raise
        assert lm == []

    def test_result_idempotent_after_stalemate(self):
        b = Board.from_fen("3k5/R8/9/9/4R4/9/9/9/9/5K3 b - - 0 1")
        assert b.result() == b.result() == "1-0"
        assert b.termination() == b.termination() == "stalemate"
        b.legal_moves()  # must not raise

    def test_result_idempotent_after_repetition(self):
        b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/R3K4 w - - 0 1")
        seq = ["a0a1", "d9d8", "a1a0", "d8d9"]
        for _ in range(4):
            for u in seq:
                if b.result() is not None:
                    break
                b.push(uci_to_move(u))
            if b.result() is not None:
                break
        assert b.result() == b.result() == "1/2-1/2"
        assert b.termination() == "repetition"
        b.legal_moves()  # must not raise after repetition

    def test_max_plies_exact_boundary(self):
        """At exactly max_plies moves: draw; one less: ongoing."""
        b_ongoing = Board.from_fen(
            "3k5/9/9/9/9/9/9/9/9/R3K4 w - - 0 1", max_plies=4
        )
        seq = ["a0a1", "d9d8", "a1a0"]
        for u in seq:
            b_ongoing.push(uci_to_move(u))
        assert b_ongoing.result() is None

        b_ongoing.push(uci_to_move("d8d9"))  # 4th move
        assert b_ongoing.result() == "1/2-1/2"
        assert b_ongoing.termination() == "max_moves"

    def test_repetition_2nd_occurrence_not_draw(self):
        """Second occurrence does NOT trigger draw."""
        b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/R3K4 w - - 0 1")
        seq = ["a0a1", "d9d8", "a1a0", "d8d9"]
        for u in seq:
            b.push(uci_to_move(u))
        assert b.result() is None
        assert b._zhist.count(b._zobrist) == 2
