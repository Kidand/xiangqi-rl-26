# -*- coding: utf-8 -*-
"""规则引擎冒烟测试（见 DESIGN.md §4 §12 §15）。

覆盖：FEN 往返 / 别名解析、44 着、perft、走法规则、将死/困毙即负、
帅对脸、长将判负、重复判和、Zobrist 稳定性、中文着法、记录往返。
"""

from __future__ import annotations

import os
import tempfile

import pytest

from xiangqi.board import Board
from xiangqi.constants import RED, START_FEN, move_to_uci, uci_to_move
from xiangqi.notation import move_to_chinese
import xiangqi.records as records


# --------------------------------------------------------------------------- FEN
def test_start_fen_roundtrip():
    b = Board()
    assert b.fen() == START_FEN
    assert Board.from_fen(START_FEN).fen() == START_FEN


def test_fen_alias_and_output():
    # 输入用 H/E 别名，输出必须规范化为 N/B。
    aliased = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"
    assert Board.from_fen(aliased).fen() == START_FEN


def test_bad_fen_raises():
    with pytest.raises(ValueError):
        Board.from_fen("this is not a fen")
    with pytest.raises(ValueError):
        Board.from_fen("rnbakabnr/9/1c5c1 w - - 0 1")  # 行数不足


def test_random_fen_roundtrip():
    import random

    rng = random.Random(7)
    b = Board()
    for _ in range(60):
        lm = b.legal_moves()
        if not lm or b.result() is not None:
            break
        # FEN 往返一致
        assert Board.from_fen(b.fen()).fen() == b.fen()
        b.push(rng.choice(lm))


# --------------------------------------------------------------------------- perft
def _perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for m in board.legal_moves():
        board.push(m)
        total += _perft(board, depth - 1)
        board.pop()
    return total


def test_initial_legal_move_count():
    assert len(Board().legal_moves()) == 44


def test_perft_1_2():
    b = Board()
    assert _perft(b, 1) == 44
    assert _perft(b, 2) == 1920


@pytest.mark.slow
def test_perft_3():
    assert _perft(Board(), 3) == 79666


# --------------------------------------------------------------------------- 规则
def test_push_pop_restores_state():
    b = Board()
    z0, fen0 = b.zobrist(), b.fen()
    for m in b.legal_moves():
        b.push(m)
        b.pop()
        assert b.zobrist() == z0
        assert b.fen() == fen0


def test_horse_leg_block():
    # 马 b0，其上方蹩腿点 b1 放一兵 -> 向上的两个马步 b0a2/b0c2 被蹩死；
    # 横向马步 b0d1 不受影响（其蹩腿点为 c0）。
    b = Board.from_fen("3k5/9/9/9/9/9/9/9/1P7/1N3K3 w - - 0 1")
    moves = {move_to_uci(m) for m in b.legal_moves()}
    assert "b0a2" not in moves
    assert "b0c2" not in moves
    assert "b0d1" in moves  # 横向马步蹩腿点 c0 为空


def test_flying_general_illegal():
    # 双帅同列且中间将变空 -> 该走法非法（白脸将）。
    # 红帅 e0, 黑将 e9, 红车 e5 挡着。移开红车暴露对脸即非法。
    b = Board.from_fen("4k4/9/9/9/4R4/9/9/9/9/4K4 w - - 0 1")
    moves = {move_to_uci(m) for m in b.legal_moves()}
    # 车沿 e 列上下移动仍挡着 -> 合法；车离开 e 列使双帅对脸 -> 非法
    assert "e5d5" not in moves  # 离开 e 列，暴露对脸
    assert "e5e6" in moves      # 仍在 e 列挡着


def test_checkmate():
    # 黑将 d9 被红车将死（对脸+车控制唯一逃格）。
    b = Board.from_fen("3k5/9/9/9/9/3R5/9/9/9/4K4 b - - 0 1")
    assert b.is_in_check(-RED)
    assert b.legal_moves() == []
    assert b.result() == "1-0"
    assert b.termination() == "checkmate"


def test_stalemate_is_loss():
    # 困毙即负：黑无着且不被将 -> 黑负（红胜）。
    b = Board.from_fen("3k5/R8/9/9/9/9/9/9/9/4RK3 b - - 0 1")
    assert not b.is_in_check(-RED)
    assert b.legal_moves() == []
    assert b.result() == "1-0"
    assert b.termination() == "stalemate"


def test_perpetual_check_loses():
    # 红车长将黑光帅：红每着将军 -> 红负（0-1）。
    b = Board.from_fen("4k4/9/9/9/9/4R4/9/9/9/5K3 b - - 0 1")
    seq = ["e9d9", "e4d4", "d9e9", "d4e4"]
    term = None
    for _ in range(3):
        for u in seq:
            if b.result() is not None:
                break
            b.push(uci_to_move(u))
        term = b.termination()
        if term is not None:
            break
    assert b.termination() == "perpetual_check"
    assert b.result() == "0-1"


def test_repetition_draw():
    # 双方无害往复 -> 第三次重复判和。
    b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/R3K4 w - - 0 1")
    seq = ["a0a1", "d9d8", "a1a0", "d8d9"]
    for _ in range(3):
        for u in seq:
            if b.result() is not None:
                break
            b.push(uci_to_move(u))
        if b.result() is not None:
            break
    assert b.termination() == "repetition"
    assert b.result() == "1/2-1/2"


def test_no_capture_draw():
    b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/R3K4 w - - 118 1")
    # 无吃子推进到 120 半回合判和
    b.push(uci_to_move("a0a1"))  # halfmove 119
    b.push(uci_to_move("d9d8"))  # halfmove 120
    assert b.termination() == "no_capture"
    assert b.result() == "1/2-1/2"


def test_max_plies_draw():
    b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/R3K4 w - - 0 1", max_plies=4)
    for u in ["a0a1", "d9d8", "a1a0", "d8d9"]:
        b.push(uci_to_move(u))
    assert b.termination() == "max_moves"
    assert b.result() == "1/2-1/2"


# --------------------------------------------------------------------------- 快照
def test_snapshots():
    b = Board()
    snaps = b.snapshots(3)
    assert len(snaps) == 3
    assert snaps[0] is not None and len(snaps[0]) == 90
    assert snaps[1] is None and snaps[2] is None  # 开局历史不足补 None
    b.push(uci_to_move("h2e2"))
    snaps = b.snapshots(3)
    assert snaps[0] is not None and snaps[1] is not None and snaps[2] is None
    # snapshot[sq] = piece + 7
    e2 = 2 * 9 + 4
    assert snaps[0][e2] == 6 + 7  # 红炮=6 -> 13


# --------------------------------------------------------------------------- 中文着法
def test_chinese_notation_examples():
    b = Board()
    assert move_to_chinese(b, uci_to_move("h2e2")) == "炮二平五"
    assert move_to_chinese(b, uci_to_move("b0c2")) == "马八进七"
    assert move_to_chinese(b, uci_to_move("h0g2")) == "马二进三"
    assert move_to_chinese(b, uci_to_move("b2e2")) == "炮八平五"


def test_chinese_front_back():
    # 同列两兵用 前/后。红兵 e5(row5) 与 e3(row3) 同列。红方 row 大者为“前”。
    b = Board.from_fen("4k4/9/9/9/4P4/9/4P4/9/9/4K4 w - - 0 1")
    assert move_to_chinese(b, uci_to_move("e5e6")).startswith("前兵")
    assert move_to_chinese(b, uci_to_move("e3e4")).startswith("后兵")


# --------------------------------------------------------------------------- 记录
def test_record_write_read_replay_roundtrip():
    import random

    rng = random.Random(11)
    b = Board()
    moves = []
    while len(moves) < 30:
        lm = b.legal_moves()
        if not lm or b.result() is not None:
            break
        m = rng.choice(lm)
        moves.append(move_to_uci(m))
        b.push(m)
    final_fen = b.fen()
    result = b.result()

    record = {
        "format": "xiangqi-record-v1",
        "start_fen": START_FEN,
        "moves": moves,
        "result": result,
        "termination": b.termination(),
        "plies": len(moves),
    }

    tmp = tempfile.mkdtemp()
    jpath = os.path.join(tmp, "g.json")
    records.write_record(jpath, record)
    loaded = records.read_records(jpath)
    assert len(loaded) == 1

    replay = records.replay_record(loaded[0])
    assert replay["moves"] == moves
    assert replay["fens"][-1] == final_fen  # 重演终局 FEN 一致
    assert len(replay["fens"]) == len(moves) + 1
    assert len(replay["chinese"]) == len(moves)

    # jsonl 追加读取
    lpath = os.path.join(tmp, "g.jsonl")
    records.append_jsonl(lpath, record)
    records.append_jsonl(lpath, record)
    assert len(records.read_records(lpath)) == 2


def test_parse_text_record():
    # 首行省略 FEN
    rec = records.parse_text_record("h2e2 h9g7\nb0c2")
    assert rec["start_fen"] == START_FEN
    assert rec["moves"] == ["h2e2", "h9g7", "b0c2"]
    replay = records.replay_record(rec)
    assert len(replay["fens"]) == 4
    # 首行带 FEN
    rec2 = records.parse_text_record(START_FEN + "\nh2e2")
    assert rec2["start_fen"] == START_FEN
    assert rec2["moves"] == ["h2e2"]


def test_replay_illegal_raises():
    bad = {"format": "xiangqi-record-v1", "start_fen": START_FEN, "moves": ["a0a5"]}
    with pytest.raises(ValueError):
        records.replay_record(bad)


# --------------------------------------------------------------------------- Zobrist
def test_zobrist_side_and_history():
    b = Board.from_fen(START_FEN)
    z_red = b.zobrist()
    b2 = Board.from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1")
    assert z_red != b2.zobrist()  # 走子方不同 -> 哈希不同


def test_copy_independent():
    b = Board()
    c = b.copy()
    c.push(uci_to_move("h2e2"))
    assert b.fen() == START_FEN  # 原局面不受影响
    assert c.fen() != START_FEN
