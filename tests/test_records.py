# -*- coding: utf-8 -*-
"""对局记录 xiangqi-record-v1 及中文着法的对抗性测试（DESIGN.md §12 §15）。

覆盖：
  records：写→读→重演往返；jsonl 多局追加；文本格式解析；非法着法报 ValueError
  notation：至少 8 例双色断言
"""

from __future__ import annotations

import os
import tempfile

import pytest

from xiangqi.board import Board
from xiangqi.constants import RED, BLACK, START_FEN, move_to_uci, uci_to_move
from xiangqi.notation import move_to_chinese
import xiangqi.records as records


# ─────────────────────────────────────────────────────────────────── 工具
def _play_n_moves(n: int, seed: int = 42) -> tuple[Board, list[str]]:
    """从初始局面随机走 n 步，返回 (board, ucci_moves)。"""
    import random
    rng = random.Random(seed)
    b = Board()
    moves: list[str] = []
    while len(moves) < n:
        lm = b.legal_moves()
        if not lm or b.result() is not None:
            break
        m = rng.choice(lm)
        moves.append(move_to_uci(m))
        b.push(m)
    return b, moves


# ═══════════════════════════════════════════════════════════════════
# 中文着法（notation）— 至少 8 例双色断言
# ═══════════════════════════════════════════════════════════════════
class TestChineseNotation:
    """覆盖红黑双方、各类棋子、各动词（进/退/平）。"""

    # ── 红方 ──────────────────────────────────────────────────────
    def test_red_cannon_ping_five(self):
        """炮二平五（h2e2）：红炮从九路平到五路。"""
        b = Board()
        assert move_to_chinese(b, uci_to_move("h2e2")) == "炮二平五"

    def test_red_horse_b0c2(self):
        """马八进七（b0c2）：红马从八路进到七路。"""
        b = Board()
        assert move_to_chinese(b, uci_to_move("b0c2")) == "马八进七"

    def test_red_horse_h0g2(self):
        """马二进三（h0g2）：红马从二路进到三路。"""
        b = Board()
        assert move_to_chinese(b, uci_to_move("h0g2")) == "马二进三"

    def test_red_cannon_b2e2(self):
        """炮八平五（b2e2）：红炮从八路平到五路。"""
        b = Board()
        assert move_to_chinese(b, uci_to_move("b2e2")) == "炮八平五"

    def test_red_rook_advance(self):
        """车进三：红车向前走三步。

        从初始局面走炮平五，再走马，再开车路。
        用确定局面：红车在 a0(row0 col0)，走 a0a3 = 进三。
        """
        fen = "3k5/9/9/9/9/9/9/9/9/R3K4 w - - 0 1"
        b = Board.from_fen(fen)
        # 车在 a0(col0)，走到 a3(row3 col0) = 进3步
        result = move_to_chinese(b, uci_to_move("a0a3"))
        assert "车" in result
        assert "进" in result

    def test_red_pawn_cross_river_advance(self):
        """兵进一：红兵向前。"""
        # 红兵在 e4(row4 col4)，进一步到 e5
        fen = "3k5/9/9/9/9/4P4/9/9/9/5K3 w - - 0 1"
        b = Board.from_fen(fen)
        result = move_to_chinese(b, uci_to_move("e4e5"))
        assert "兵" in result
        assert "进" in result

    def test_red_king_move(self):
        """帅平：帅横移。"""
        fen = "3k5/9/9/9/9/9/9/9/9/4K4 w - - 0 1"
        b = Board.from_fen(fen)
        # 帅在 e0，走到 d0（平）
        result = move_to_chinese(b, uci_to_move("e0d0"))
        assert "帅" in result
        assert "平" in result

    # ── 黑方 ──────────────────────────────────────────────────────
    def test_black_horse_h9g7(self):
        """马8进7（h9g7）：黑马从8路进到7路。"""
        b = Board()
        b.push(uci_to_move("h2e2"))  # 红先走
        assert move_to_chinese(b, uci_to_move("h9g7")) == "马8进7"

    def test_black_cannon_h7e7(self):
        """炮8平5（h7e7）：黑炮从h路（路8）平到五路。

        黑方列号：col0=1路…col7=8路。h7=col7=8路。
        """
        b = Board()
        b.push(uci_to_move("h2e2"))
        assert move_to_chinese(b, uci_to_move("h7e7")) == "炮8平5"

    def test_black_cannon_b7e7(self):
        """炮2平5（b7e7）：黑炮从b路（路2）平到五路。"""
        b = Board()
        b.push(uci_to_move("h2e2"))
        assert move_to_chinese(b, uci_to_move("b7e7")) == "炮2平5"

    def test_black_horse_b9c7(self):
        """马2进3（b9c7）：黑方马从2路进到3路（黑方列号用阿拉伯数字）。

        注意：黑方 col0=1路，col8=9路（与红方相反）。
        b9 = row9 col1 = 2路（col1+1=2），c7 = row7 col2 = 3路（col2+1=3）。
        """
        b = Board()
        b.push(uci_to_move("h2e2"))
        result = move_to_chinese(b, uci_to_move("b9c7"))
        assert result == "马2进3"

    def test_black_rook_retreat(self):
        """车退步：黑车向己方底线退。"""
        # 黑车从 a5 退到 a9
        fen = "4k4/9/9/9/r8/9/9/9/9/4K4 b - - 0 1"
        b = Board.from_fen(fen)
        result = move_to_chinese(b, uci_to_move("a5a9"))
        assert "车" in result
        assert "退" in result

    def test_black_advisor_move(self):
        """士斜走：黑士从 e8(row8 col4) 走到 d9(row9 col3)。

        FEN: row9="4k4", row8="4a4" -> 士在 e8(row8 col4)。
        """
        fen = "4k4/4a4/9/9/9/9/9/9/9/4K4 b - - 0 1"
        b = Board.from_fen(fen)
        result = move_to_chinese(b, uci_to_move("e8d9"))
        assert "士" in result

    def test_black_elephant_move(self):
        """象走斜：黑象从 c9 走到 e7。"""
        fen = "2b1k4/9/9/9/9/9/9/9/9/4K4 b - - 0 1"
        b = Board.from_fen(fen)
        result = move_to_chinese(b, uci_to_move("c9e7"))
        assert "象" in result

    def test_black_pawn_advance(self):
        """卒进：黑卒向前。"""
        # 黑卒在 a6(row6 col0)，向前到 a5
        fen = "4k4/9/9/p8/9/9/9/9/9/3K5 b - - 0 1"
        b = Board.from_fen(fen)
        result = move_to_chinese(b, uci_to_move("a6a5"))
        assert "卒" in result
        assert "进" in result

    # ── 前/后 消歧 ──────────────────────────────────────────────
    def test_red_front_back_pawn(self):
        """同列两红兵用前/后消歧：row 大者（靠前）= 前兵，row 小者 = 后兵。"""
        fen = "4k4/9/9/9/4P4/9/4P4/9/9/4K4 w - - 0 1"
        b = Board.from_fen(fen)
        # e5(row5) 是前兵，e3(row3) 是后兵（红方 row 越大越靠前）
        assert move_to_chinese(b, uci_to_move("e5e6")).startswith("前兵")
        assert move_to_chinese(b, uci_to_move("e3e4")).startswith("后兵")

    def test_red_front_back_cannon(self):
        """同列两红炮用前/后消歧。"""
        fen = "4k4/9/9/9/4C4/9/4C4/9/9/4K4 w - - 0 1"
        b = Board.from_fen(fen)
        # e5(row5) 是前炮，e3(row3) 是后炮
        front = move_to_chinese(b, uci_to_move("e5e4"))
        back = move_to_chinese(b, uci_to_move("e3e4"))
        assert front.startswith("前炮")
        assert back.startswith("后炮")


# ═══════════════════════════════════════════════════════════════════
# records 写读重演往返
# ═══════════════════════════════════════════════════════════════════
class TestRecordsRoundtrip:
    def test_write_read_single_json(self, tmp_path):
        """write_record + read_records 往返：单条 JSON。"""
        board, moves = _play_n_moves(20, seed=1)
        rec = {
            "format": records.RECORD_FORMAT,
            "start_fen": START_FEN,
            "moves": moves,
            "result": board.result(),
            "termination": board.termination(),
            "plies": len(moves),
        }
        path = str(tmp_path / "game.json")
        records.write_record(path, rec)
        loaded = records.read_records(path)
        assert len(loaded) == 1
        assert loaded[0]["moves"] == moves
        assert loaded[0]["format"] == records.RECORD_FORMAT

    def test_write_read_replay_final_fen(self, tmp_path):
        """重演后最终 FEN 与原局面一致。"""
        board, moves = _play_n_moves(30, seed=2)
        final_fen = board.fen()
        rec = {
            "format": records.RECORD_FORMAT,
            "start_fen": START_FEN,
            "moves": moves,
        }
        path = str(tmp_path / "game2.json")
        records.write_record(path, rec)
        loaded = records.read_records(path)[0]
        replay = records.replay_record(loaded)
        assert replay["fens"][-1] == final_fen

    def test_replay_fens_length(self, tmp_path):
        """replay fens 长度 = 着法数 + 1。"""
        board, moves = _play_n_moves(15, seed=3)
        rec = {"format": records.RECORD_FORMAT, "start_fen": START_FEN, "moves": moves}
        path = str(tmp_path / "game3.json")
        records.write_record(path, rec)
        replay = records.replay_record(records.read_records(path)[0])
        assert len(replay["fens"]) == len(moves) + 1

    def test_replay_chinese_length(self, tmp_path):
        """replay chinese 列表长度 = 着法数。"""
        board, moves = _play_n_moves(10, seed=4)
        rec = {"format": records.RECORD_FORMAT, "start_fen": START_FEN, "moves": moves}
        path = str(tmp_path / "game4.json")
        records.write_record(path, rec)
        replay = records.replay_record(records.read_records(path)[0])
        assert len(replay["chinese"]) == len(moves)

    def test_replay_first_fen_is_start(self):
        """replay fens[0] 是起始 FEN。"""
        rec = {"format": records.RECORD_FORMAT, "start_fen": START_FEN, "moves": ["h2e2"]}
        replay = records.replay_record(rec)
        assert replay["fens"][0] == START_FEN

    def test_replay_moves_preserved(self, tmp_path):
        """replay["moves"] 等于原始着法列表。"""
        board, moves = _play_n_moves(12, seed=5)
        rec = {"format": records.RECORD_FORMAT, "start_fen": START_FEN, "moves": moves}
        path = str(tmp_path / "game5.json")
        records.write_record(path, rec)
        replay = records.replay_record(records.read_records(path)[0])
        assert replay["moves"] == moves

    def test_jsonl_append_and_read(self, tmp_path):
        """append_jsonl + read_records 往返：多条 JSONL。"""
        path = str(tmp_path / "games.jsonl")
        for i in range(3):
            board, moves = _play_n_moves(5, seed=i + 10)
            rec = {
                "format": records.RECORD_FORMAT,
                "start_fen": START_FEN,
                "moves": moves,
                "result": board.result(),
                "plies": len(moves),
                "meta": {"seed": i},
            }
            records.append_jsonl(path, rec)
        loaded = records.read_records(path)
        assert len(loaded) == 3
        assert loaded[0]["meta"]["seed"] == 0
        assert loaded[2]["meta"]["seed"] == 2

    def test_jsonl_replay_all_games(self, tmp_path):
        """JSONL 中每条记录都可重演。"""
        path = str(tmp_path / "multi.jsonl")
        expected_finals = []
        for i in range(3):
            board, moves = _play_n_moves(8, seed=i + 20)
            expected_finals.append(board.fen())
            rec = {"format": records.RECORD_FORMAT, "start_fen": START_FEN, "moves": moves}
            records.append_jsonl(path, rec)
        for j, rec in enumerate(records.read_records(path)):
            replay = records.replay_record(rec)
            assert replay["fens"][-1] == expected_finals[j]

    def test_custom_start_fen_replay(self, tmp_path):
        """非初始 FEN 起点的记录可正确重演。"""
        start = "3k5/9/9/9/9/9/9/9/9/R3K4 w - - 0 1"
        b = Board.from_fen(start)
        moves = ["a0a1", "d9d8"]
        for u in moves:
            b.push(uci_to_move(u))
        final_fen = b.fen()
        rec = {"format": records.RECORD_FORMAT, "start_fen": start, "moves": moves}
        path = str(tmp_path / "custom.json")
        records.write_record(path, rec)
        replay = records.replay_record(records.read_records(path)[0])
        assert replay["fens"][0] == start
        assert replay["fens"][-1] == final_fen


# ═══════════════════════════════════════════════════════════════════
# parse_text_record
# ═══════════════════════════════════════════════════════════════════
class TestParseTextRecord:
    def test_no_fen_header_uses_start(self):
        """首行不含 '/' -> 使用初始局面。"""
        rec = records.parse_text_record("h2e2 h9g7\nb0c2")
        assert rec["start_fen"] == START_FEN
        assert rec["moves"] == ["h2e2", "h9g7", "b0c2"]

    def test_fen_header_used_as_start(self):
        """首行含 '/' -> 作为起始 FEN。"""
        start = "3k5/9/9/9/9/9/9/9/9/R3K4 w - - 0 1"
        rec = records.parse_text_record(start + "\na0a1 d9d8")
        assert rec["start_fen"] == start
        assert rec["moves"] == ["a0a1", "d9d8"]

    def test_initial_fen_as_header(self):
        """初始 FEN 作为首行。"""
        rec = records.parse_text_record(START_FEN + "\nh2e2")
        assert rec["start_fen"] == START_FEN
        assert rec["moves"] == ["h2e2"]

    def test_replay_after_text_parse(self):
        """parse_text_record 后可直接重演。"""
        rec = records.parse_text_record("h2e2 h9g7 b0c2")
        replay = records.replay_record(rec)
        assert len(replay["fens"]) == 4  # 起点 + 3步
        assert len(replay["chinese"]) == 3

    def test_multiple_lines_of_moves(self):
        """多行着法均被解析。"""
        text = "h2e2\nh9g7\nb0c2\nb9c7"
        rec = records.parse_text_record(text)
        assert rec["moves"] == ["h2e2", "h9g7", "b0c2", "b9c7"]

    def test_whitespace_handling(self):
        """多余空白和空行被正确跳过。"""
        text = "  h2e2   h9g7  \n\n  b0c2  "
        rec = records.parse_text_record(text)
        assert rec["moves"] == ["h2e2", "h9g7", "b0c2"]

    def test_empty_moves(self):
        """仅 FEN 行，无着法。"""
        rec = records.parse_text_record(START_FEN)
        assert rec["start_fen"] == START_FEN
        assert rec["moves"] == []


# ═══════════════════════════════════════════════════════════════════
# replay_record 错误处理
# ═══════════════════════════════════════════════════════════════════
class TestReplayErrors:
    def test_illegal_move_raises(self):
        """非法着法 -> ValueError。"""
        bad = {"format": records.RECORD_FORMAT, "start_fen": START_FEN, "moves": ["a0a5"]}
        with pytest.raises(ValueError):
            records.replay_record(bad)

    def test_illegal_move_in_middle_raises(self):
        """中途出现非法着法 -> ValueError。"""
        bad = {
            "format": records.RECORD_FORMAT,
            "start_fen": START_FEN,
            "moves": ["h2e2", "h9g7", "a0a5"],  # 第三步非法
        }
        with pytest.raises(ValueError):
            records.replay_record(bad)

    def test_out_of_turn_move_raises(self):
        """走错顺序（连续两步红方）-> ValueError。"""
        bad = {
            "format": records.RECORD_FORMAT,
            "start_fen": START_FEN,
            "moves": ["h2e2", "b2e2"],  # 两步均为红方炮，第二步非法
        }
        with pytest.raises(ValueError):
            records.replay_record(bad)


# ═══════════════════════════════════════════════════════════════════
# records 读取 .json 列表格式
# ═══════════════════════════════════════════════════════════════════
class TestReadRecordsList:
    def test_read_json_list(self, tmp_path):
        """JSON 文件为 list（多条）时可正确读取。"""
        import json
        recs = []
        for i in range(2):
            _, moves = _play_n_moves(5, seed=i + 30)
            recs.append({"format": records.RECORD_FORMAT, "start_fen": START_FEN, "moves": moves})
        path = str(tmp_path / "list.json")
        Path = __import__("pathlib").Path
        Path(path).write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
        loaded = records.read_records(path)
        assert len(loaded) == 2

    def test_read_json_single_dict(self, tmp_path):
        """JSON 文件为单条 dict 时包装为 list 返回。"""
        import json
        _, moves = _play_n_moves(5, seed=40)
        rec = {"format": records.RECORD_FORMAT, "start_fen": START_FEN, "moves": moves}
        path = str(tmp_path / "single.json")
        __import__("pathlib").Path(path).write_text(
            json.dumps(rec, ensure_ascii=False), encoding="utf-8"
        )
        loaded = records.read_records(path)
        assert isinstance(loaded, list)
        assert len(loaded) == 1
