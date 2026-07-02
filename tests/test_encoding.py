"""编码模块测试（DESIGN.md §6，test_encoding.py）。

依赖 Board 的用例：若 xiangqi.board 尚未实现则 pytest.skip 跳过。
纯索引函数（move_to_policy_index / flip 往返等）必须无 Board 依赖并始终通过。
"""

from __future__ import annotations

import numpy as np
import pytest

# ------------------------------------------------------------
# 尝试导入 Board；若未实现则标记为不可用
# ------------------------------------------------------------
try:
    from xiangqi.board import Board  # type: ignore
    HAS_BOARD = True
except Exception:
    HAS_BOARD = False

# 始终可用的导入
from rl.encoding import (
    encode_board,
    legal_move_mask,
    move_to_policy_index,
    policy_index_to_move,
)
from xiangqi.constants import (
    ACTION_SIZE,
    BLACK,
    COLS,
    RED,
    ROWS,
    flip_action,
    flip_sq,
    sq,
)


# ============================================================
# 纯索引函数测试（无 Board 依赖）
# ============================================================

class TestFlipActionInvolution:
    """flip_action 是对合：flip_action(flip_action(a)) == a。"""

    def test_boundary_values(self):
        for a in [0, 1, 89, 90, 8099, 4500, 800]:
            assert flip_action(flip_action(a)) == a, (
                f"flip_action 不是对合，a={a}"
            )

    def test_all_values(self):
        for a in range(ACTION_SIZE):
            assert flip_action(flip_action(a)) == a


class TestFlipSqInvolution:
    """flip_sq 是对合：flip_sq(flip_sq(s)) == s。"""

    def test_all_squares(self):
        from xiangqi.constants import NUM_SQUARES
        for s in range(NUM_SQUARES):
            assert flip_sq(flip_sq(s)) == s, f"flip_sq 不是对合，s={s}"


class TestPolicyIndexRoundtrip:
    """move_to_policy_index / policy_index_to_move 互逆（任意 side）。"""

    # 典型着法（棋盘坐标）
    SAMPLE_MOVES = [
        (sq(0, 0), sq(0, 1)),   # 红方底线左侧
        (sq(9, 4), sq(7, 4)),   # 黑方将
        (sq(5, 2), sq(3, 2)),   # 过河棋子
        (sq(0, 7), sq(2, 6)),   # 马跳
        (sq(9, 0), sq(9, 8)),   # 黑车横扫
    ]

    @pytest.mark.parametrize("move", SAMPLE_MOVES)
    def test_red_roundtrip(self, move):
        idx = move_to_policy_index(move, RED)
        recovered = policy_index_to_move(idx, RED)
        assert recovered == move, f"RED 往返失败: {move} -> {idx} -> {recovered}"

    @pytest.mark.parametrize("move", SAMPLE_MOVES)
    def test_black_roundtrip(self, move):
        idx = move_to_policy_index(move, BLACK)
        recovered = policy_index_to_move(idx, BLACK)
        assert recovered == move, f"BLACK 往返失败: {move} -> {idx} -> {recovered}"

    def test_index_in_range(self):
        """策略索引必须在 [0, ACTION_SIZE)。"""
        for move in self.SAMPLE_MOVES:
            for side in [RED, BLACK]:
                idx = move_to_policy_index(move, side)
                assert 0 <= idx < ACTION_SIZE, (
                    f"策略索引越界: move={move}, side={side}, idx={idx}"
                )


class TestPolicyIndexFlipSymmetry:
    """黑方索引 == 对应翻转动作的红方索引。"""

    def test_black_equals_flipped_red(self):
        """move_to_policy_index(m, BLACK) == move_to_policy_index(flip_move(m), RED)"""
        test_moves = [
            (sq(0, 0), sq(1, 0)),
            (sq(9, 4), sq(8, 4)),
            (sq(4, 4), sq(5, 4)),
        ]
        for move in test_moves:
            flipped_move = (flip_sq(move[0]), flip_sq(move[1]))
            idx_black = move_to_policy_index(move, BLACK)
            idx_red_flipped = move_to_policy_index(flipped_move, RED)
            assert idx_black == idx_red_flipped, (
                f"翻转对称性失败: move={move}, "
                f"idx_black={idx_black}, idx_red_flipped={idx_red_flipped}"
            )


# ============================================================
# 依赖 Board 的测试（Board 不可用时 skip）
# ============================================================

SKIP_NO_BOARD = pytest.mark.skipif(
    not HAS_BOARD,
    reason="xiangqi.board 尚未实现，跳过 Board 相关测试",
)


@SKIP_NO_BOARD
class TestEncodeBoard:
    """encode_board 形状与基本属性测试。"""

    def _board(self, history_steps: int = 2):
        return Board.from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        )

    def test_output_shape_default(self):
        board = self._board()
        x = encode_board(board, history_steps=2)
        assert x.shape == (31, 10, 9), f"形状错误: {x.shape}"

    def test_output_dtype(self):
        board = self._board()
        x = encode_board(board, history_steps=2)
        assert x.dtype == np.float32

    def test_output_shape_history1(self):
        board = self._board()
        x = encode_board(board, history_steps=1)
        assert x.shape == (17, 10, 9), f"形状错误: {x.shape}"

    def test_output_shape_history4(self):
        board = self._board()
        x = encode_board(board, history_steps=4)
        assert x.shape == (59, 10, 9), f"形状错误: {x.shape}"

    def test_piece_planes_binary(self):
        """棋子平面值只含 0/1。"""
        board = self._board()
        x = encode_board(board, history_steps=2)
        C = x.shape[0]
        piece_planes = x[: C - 3]
        assert np.all((piece_planes == 0) | (piece_planes == 1)), (
            "棋子平面含非二值元素"
        )

    def test_side_channel_red(self):
        """红方走子时，C-3 平面全为 1。"""
        board = self._board()
        assert board.side_to_move == RED
        x = encode_board(board, history_steps=2)
        C = x.shape[0]
        assert np.all(x[C - 3] == 1.0), "红方走子时 side 平面应为全 1"

    def test_side_channel_black(self):
        """黑方走子时，C-3 平面全为 0。"""
        board = Board.from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1"
        )
        x = encode_board(board, history_steps=2)
        C = x.shape[0]
        assert np.all(x[C - 3] == 0.0), "黑方走子时 side 平面应为全 0"

    def test_last_move_plane_no_move(self):
        """初始局面无上一着，C-2 与 C-1 平面全零。"""
        board = self._board()
        x = encode_board(board, history_steps=2)
        C = x.shape[0]
        assert np.all(x[C - 2] == 0.0), "无着法时 from 平面应全 0"
        assert np.all(x[C - 1] == 0.0), "无着法时 to 平面应全 0"

    def test_last_move_plane_after_push(self):
        """走一步后，C-2 和 C-1 应为该着的起点/终点 one-hot。"""
        from xiangqi.constants import str_to_sq, sq_to_str

        board = self._board()
        # 红炮 h2e2
        move = (str_to_sq("h2"), str_to_sq("e2"))
        board.push(move)
        # 现在是黑方走，需翻转坐标
        x = encode_board(board, history_steps=2)
        C = x.shape[0]
        # 黑方视角下上一着被翻转
        f_from = flip_sq(move[0])
        f_to = flip_sq(move[1])
        assert x[C - 2, f_from // COLS, f_from % COLS] == 1.0, "from 平面位置错误"
        assert x[C - 1, f_to // COLS, f_to % COLS] == 1.0, "to 平面位置错误"


@SKIP_NO_BOARD
class TestLegalMoveMask:
    """legal_move_mask 与 board.legal_moves() 一致性。"""

    def _initial_board(self):
        return Board.from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        )

    def test_mask_shape(self):
        board = self._initial_board()
        mask = legal_move_mask(board)
        assert mask.shape == (ACTION_SIZE,)
        assert mask.dtype == bool

    def test_mask_count_initial(self):
        """初始局面红方有 44 个合法着（perft 1 = 44）。"""
        board = self._initial_board()
        mask = legal_move_mask(board)
        assert mask.sum() == 44, f"初始合法着数量错误: {mask.sum()}"

    def test_mask_matches_legal_moves(self):
        """掩码中 True 的索引集合 == legal_moves 对应索引集合。"""
        board = self._initial_board()
        side = board.side_to_move

        expected = {move_to_policy_index(m, side) for m in board.legal_moves()}
        actual = set(np.where(legal_move_mask(board))[0].tolist())
        assert actual == expected, (
            f"掩码与 legal_moves 不一致\n"
            f"  多余: {actual - expected}\n"
            f"  缺失: {expected - actual}"
        )


@SKIP_NO_BOARD
class TestEncodeViewFlipConsistency:
    """黑方局面编码 == 红黑互换镜像局面的红方编码（翻转一致性）。"""

    def test_flip_symmetry(self):
        """
        在初始局面，令 B_red = 红方走子，B_black = 黑方走子。
        encode_board(B_black) 的棋子平面应等于 B_red 编码红黑互换后的结果。

        简化验证：黑方局面编码中平面 0..6（自方）
        应为红方局面编码中平面 7..13（对手，原黑棋）经翻转后的镜像。
        """
        # 初始局面：红方走 & 黑方走（FEN 模式字段）
        board_red = Board.from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        )
        board_black = Board.from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1"
        )

        x_red = encode_board(board_red, history_steps=1)    # (17, 10, 9)
        x_black = encode_board(board_black, history_steps=1)  # (17, 10, 9)

        # 黑方编码的"自方"平面 0..6（黑棋，已翻转到底部）
        # 应等于红方编码的"对手"平面 7..13（黑棋，未翻转）经 180° 翻转
        for ch in range(7):
            black_self_plane = x_black[ch]             # (10, 9)
            red_opponent_plane = x_red[7 + ch]         # (10, 9)，对手即黑棋
            # 翻转红方视角的黑棋平面（180° 旋转）
            red_opp_flipped = red_opponent_plane[::-1, ::-1]
            np.testing.assert_array_equal(
                black_self_plane, red_opp_flipped,
                err_msg=f"翻转一致性失败，棋子类型通道 {ch}"
            )

    def test_double_flip_identity(self):
        """对初始局面（红方），翻转动作两次应为恒等。"""
        from xiangqi.constants import uci_to_move

        # 初始红方合法着中取几个
        board = Board.from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        )
        for move in board.legal_moves()[:10]:
            idx_black = move_to_policy_index(move, BLACK)
            recovered = policy_index_to_move(idx_black, BLACK)
            assert recovered == move, (
                f"双重翻转不是恒等: move={move}, "
                f"recovered={recovered}"
            )
