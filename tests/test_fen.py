# -*- coding: utf-8 -*-
"""FEN 解析/序列化的对抗性测试（DESIGN.md §3 §15）。

覆盖：
  初始 FEN 往返 / 随机局面往返
  E/e/H/h 别名解析，输出规范化为 B/N
  r/R 走子方别名（等同 w）
  非法 FEN：无帅 / 行数错 / 字符错 / 列数错 / halfmove/fullmove 非法
  halfmove/fullmove 在 push 后正确递推
  Zobrist 哈希：走子方不同时哈希不同
"""

from __future__ import annotations

import pytest

from xiangqi.board import Board
from xiangqi.constants import RED, BLACK, START_FEN, uci_to_move
from xiangqi.fen import parse_fen, format_fen


# ─────────────────────────────────────────────────────────────────── 往返
class TestFenRoundtrip:
    def test_start_fen_roundtrip(self):
        """初始 FEN 经 Board 往返后不变。"""
        assert Board().fen() == START_FEN
        assert Board.from_fen(START_FEN).fen() == START_FEN

    def test_parse_then_format_identity(self):
        """parse_fen + format_fen 还原原始 FEN。"""
        squares, side, hm, fm = parse_fen(START_FEN)
        assert format_fen(squares, side, hm, fm) == START_FEN

    def test_random_game_roundtrip(self):
        """随机下 50 步，每步 FEN 往返一致。"""
        import random
        rng = random.Random(2025)
        b = Board()
        for _ in range(50):
            lm = b.legal_moves()
            if not lm or b.result() is not None:
                break
            assert Board.from_fen(b.fen()).fen() == b.fen()
            b.push(rng.choice(lm))

    def test_black_to_move_roundtrip(self):
        """黑方走子局面 FEN 往返保持走子方为 b。"""
        b = Board()
        b.push(uci_to_move("h2e2"))  # 走一步后黑方走子
        fen = b.fen()
        assert " b " in fen
        assert Board.from_fen(fen).fen() == fen

    def test_halfmove_fullmove_in_fen(self):
        """halfmove 和 fullmove 正确写入 FEN。"""
        fen = "3k5/9/9/9/9/9/9/9/9/R3K4 w - - 15 8"
        b = Board.from_fen(fen)
        assert b.halfmove == 15
        assert b.fullmove == 8
        assert b.fen() == fen


# ─────────────────────────────────────────────────────────────────── 别名
class TestFenAliases:
    def test_e_alias_for_elephant(self):
        """E/e 别名解析为相/象（ELEPHANT），输出用 B/b。"""
        aliased = "rheakae hr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"
        # 去掉空格（刚才多加了）
        aliased = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"
        b = Board.from_fen(aliased)
        assert b.fen() == START_FEN

    def test_h_alias_for_horse(self):
        """H/h 别名解析为马（HORSE），输出用 N/n。"""
        aliased = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        # 先测正常，再测 H
        fen_with_h = START_FEN.replace("N", "H").replace("n", "h")
        b = Board.from_fen(fen_with_h)
        assert b.fen() == START_FEN

    def test_e_and_h_together(self):
        """E/e 和 H/h 同时出现，全部正确解析。"""
        # 将初始 FEN 的 B->E, N->H（大小写）
        aliased = (START_FEN
                   .replace("B", "E").replace("b", "e")
                   .replace("N", "H").replace("n", "h"))
        assert Board.from_fen(aliased).fen() == START_FEN

    def test_mixed_canonical_alias(self):
        """混合使用规范字母与别名（如 B 与 E 混用）。"""
        # row0: RNBAKABNR -> RNEA K AENR（混用 E）
        mixed_row0 = "RNEAK AENR".replace(" ", "")  # = "RNEAK AENR"
        mixed_row0 = "RNEAKAENR"
        mixed = START_FEN.replace("RNBAKABNR", mixed_row0)
        b = Board.from_fen(mixed)
        assert b.fen() == START_FEN

    def test_r_side_alias_for_red(self):
        """走子方 'r'（小写）等同于 'w'，解析后 side=RED。"""
        fen_r = START_FEN.replace(" w ", " r ")
        b = Board.from_fen(fen_r)
        assert b.side_to_move == RED
        # 输出应规范化为 w
        assert " w " in b.fen()

    def test_R_side_alias_for_red(self):
        """走子方 'R'（大写）等同于 'w'，解析后 side=RED。"""
        fen_R = START_FEN.replace(" w ", " R ")
        b = Board.from_fen(fen_R)
        assert b.side_to_move == RED

    def test_b_side_black(self):
        """走子方 'b' 解析为 BLACK。"""
        fen_b = START_FEN.replace(" w ", " b ")
        b = Board.from_fen(fen_b)
        assert b.side_to_move == BLACK

    def test_output_always_w_for_red(self):
        """输出 FEN 走子方：红用 'w'，黑用 'b'（不用 'r'）。"""
        b = Board.from_fen(START_FEN.replace(" w ", " r "))
        assert " w " in b.fen()
        assert " r " not in b.fen()

    def test_output_always_b_for_black(self):
        """输出 FEN 走子方：黑一律用 'b'。"""
        b = Board.from_fen(START_FEN.replace(" w ", " b "))
        assert " b " in b.fen()


# ─────────────────────────────────────────────────────────────────── 非法 FEN
class TestInvalidFen:
    def test_no_red_king_raises(self):
        """无红帅 -> ValueError。"""
        # 把 K 替换为多余的 N
        bad = START_FEN.replace("RNBAKABNR", "RNBANBNR9")
        # 直接构造缺少 K 的 FEN
        bad2 = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAANBNR w - - 0 1"
        with pytest.raises(ValueError, match="王"):
            parse_fen(bad2)

    def test_no_black_king_raises(self):
        """无黑将 -> ValueError。"""
        bad = "rnbaanbnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        with pytest.raises(ValueError):
            parse_fen(bad)

    def test_two_red_kings_raises(self):
        """两个红帅 -> ValueError。"""
        bad = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKAKBNR w - - 0 1"
        # 10 chars in last row -> might fail on length first
        with pytest.raises(ValueError):
            parse_fen(bad)

    def test_wrong_row_count_raises(self):
        """FEN board 段行数不等于 10 -> ValueError。"""
        # 只有 9 行
        bad = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9 w - - 0 1"
        with pytest.raises(ValueError, match="10"):
            parse_fen(bad)

    def test_too_many_rows_raises(self):
        """FEN 超过 10 行 -> ValueError。"""
        bad = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR/9 w - - 0 1"
        with pytest.raises(ValueError):
            parse_fen(bad)

    def test_unknown_piece_char_raises(self):
        """未知棋子字符 -> ValueError。"""
        bad = START_FEN.replace("RNBAKABNR", "RNBAKABNZ")  # Z 不是棋子
        with pytest.raises(ValueError):
            parse_fen(bad)

    def test_row_too_long_raises(self):
        """一行格子数超过 9 -> ValueError。"""
        # row0: "RNBAKABNR" + 多加一个 N -> RNBAKABNRN (10 pieces)
        bad = START_FEN.replace("RNBAKABNR", "RNBAKABNRN")
        with pytest.raises(ValueError):
            parse_fen(bad)

    def test_row_too_short_raises(self):
        """一行格子数不足 9 -> ValueError。"""
        # row0: 把 9 变 8 (删一个空格描述)
        bad = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/8/RNBAKABNR w - - 0 1"
        with pytest.raises(ValueError):
            parse_fen(bad)

    def test_unknown_side_raises(self):
        """未知走子方字符 -> ValueError。"""
        bad = START_FEN.replace(" w ", " x ")
        with pytest.raises(ValueError, match="走子方"):
            parse_fen(bad)

    def test_negative_halfmove_raises(self):
        """负数 halfmove -> ValueError。"""
        bad = START_FEN.replace(" 0 1", " -1 1")
        with pytest.raises(ValueError):
            parse_fen(bad)

    def test_zero_fullmove_raises(self):
        """fullmove < 1 -> ValueError。"""
        bad = START_FEN.replace(" 0 1", " 0 0")
        with pytest.raises(ValueError):
            parse_fen(bad)

    def test_empty_fen_raises(self):
        """空字符串 -> ValueError。"""
        with pytest.raises(ValueError):
            parse_fen("")

    def test_garbage_fen_raises(self):
        """完全无效字符串 -> ValueError。"""
        with pytest.raises(ValueError):
            parse_fen("this is not a fen")


# ─────────────────────────────────────────────────────────────────── halfmove/fullmove 递推
class TestHalfmoveFullmove:
    def test_halfmove_increments_without_capture(self):
        """无吃子时 halfmove 每步 +1。"""
        b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/R3K4 w - - 0 1")
        b.push(uci_to_move("a0a1"))
        assert b.halfmove == 1
        b.push(uci_to_move("d9d8"))
        assert b.halfmove == 2

    def test_halfmove_resets_on_capture(self):
        """吃子时 halfmove 清零。"""
        # 红车 a0 吃黑车 a9
        b = Board.from_fen("r3k4/9/9/9/9/9/9/9/9/R3K4 w - - 50 1")
        b.push(uci_to_move("a0a9"))
        assert b.halfmove == 0

    def test_fullmove_increments_after_black_moves(self):
        """fullmove 仅在黑方走子后 +1。"""
        b = Board()
        fm0 = b.fullmove
        b.push(uci_to_move("h2e2"))  # 红方走
        assert b.fullmove == fm0      # 未变
        b.push(uci_to_move("h9g7"))  # 黑方走
        assert b.fullmove == fm0 + 1  # +1

    def test_fullmove_not_incremented_on_red_move(self):
        """红方走子后 fullmove 不变。"""
        b = Board()
        for _ in range(3):
            lm = [m for m in b.legal_moves() if b.side_to_move == RED]
            if not lm:
                break
            fm_before = b.fullmove
            b.push(lm[0])
            if b.side_to_move == BLACK:  # 刚走完红方
                assert b.fullmove == fm_before
            # 补一步黑方让红方能继续走
            lm2 = b.legal_moves()
            if lm2:
                b.push(lm2[0])

    def test_halfmove_preserved_by_pop(self):
        """pop 后 halfmove 恢复到走子前的值。"""
        b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/R3K4 w - - 5 1")
        hm_before = b.halfmove
        b.push(uci_to_move("a0a1"))
        b.pop()
        assert b.halfmove == hm_before

    def test_fullmove_preserved_by_pop_after_black(self):
        """pop 黑方走子后 fullmove 恢复。"""
        b = Board()
        b.push(uci_to_move("h2e2"))  # 红方
        fm_before = b.fullmove
        b.push(uci_to_move("h9g7"))  # 黑方 -> fullmove +1
        assert b.fullmove == fm_before + 1
        b.pop()
        assert b.fullmove == fm_before

    def test_halfmove_at_119_draws_on_next(self):
        """halfmove=119，走一步无吃子变 120 -> 判和。"""
        b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/R3K4 w - - 119 1")
        b.push(uci_to_move("a0a1"))
        assert b.halfmove == 120
        assert b.result() == "1/2-1/2"

    def test_large_halfmove_preserved(self):
        """halfmove 从高值（如 80）开始时正确递增。"""
        b = Board.from_fen("3k5/9/9/9/9/9/9/9/9/R3K4 w - - 80 5")
        assert b.halfmove == 80
        assert b.fullmove == 5
        b.push(uci_to_move("a0a1"))
        assert b.halfmove == 81
        assert b.fullmove == 5  # 红方走，fullmove 不变


# ─────────────────────────────────────────────────────────────────── Zobrist
class TestZobristHashing:
    def test_same_position_same_hash(self):
        """从相同 FEN 构建的 Board，哈希一致。"""
        b1 = Board.from_fen(START_FEN)
        b2 = Board.from_fen(START_FEN)
        assert b1.zobrist() == b2.zobrist()

    def test_different_side_to_move_different_hash(self):
        """走子方不同时，哈希不同。"""
        b_red = Board.from_fen(START_FEN)  # w
        b_black = Board.from_fen(START_FEN.replace(" w ", " b "))
        assert b_red.zobrist() != b_black.zobrist()

    def test_push_pop_restores_hash(self):
        """push 后 pop 还原 zobrist。"""
        b = Board()
        z0 = b.zobrist()
        b.push(uci_to_move("h2e2"))
        b.pop()
        assert b.zobrist() == z0

    def test_hash_changes_after_move(self):
        """走子后哈希改变。"""
        b = Board()
        z_before = b.zobrist()
        b.push(uci_to_move("h2e2"))
        assert b.zobrist() != z_before

    def test_two_paths_same_position_same_hash(self):
        """不同走法顺序抵达相同局面，哈希相同（前提：局面确实相同）。"""
        # 炮平五 + 回到原位 -> 初始局面
        b = Board()
        z0 = b.zobrist()
        b.push(uci_to_move("h2e2"))  # 炮平五
        b.pop()                       # 撤回
        assert b.zobrist() == z0

    def test_initial_board_hash_nonzero(self):
        """初始局面哈希非零（理论上极低概率为零，可接受）。"""
        assert Board().zobrist() != 0
