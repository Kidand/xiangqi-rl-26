"""Xiangqi FEN 解析 / 序列化，以及 UCCI 着法解析（见 DESIGN.md §3）。

坐标与棋子编码一律来自 xiangqi.constants，本模块不重复定义。

FEN 完整格式：``<board> <side> - - <halfmove> <fullmove>``
- board 段：从 row 9（黑底线）到 row 0，每行 col 0..8，数字表示连续空格，行间 ``/``。
- side：``w``=红（兼容 ``r``），``b``=黑；输出一律 ``w``/``b``。
- 第 3、4 段固定 ``-``。
解析时兼容棋子别名 ``E/e``（象）、``H/h``（马）；输出一律 ``B/N``。
"""

from __future__ import annotations

from .constants import (
    BLACK,
    CHAR_TO_PIECE,
    KING,
    PIECE_TO_CHAR,
    RED,
    START_FEN,
    uci_to_move,
)

__all__ = [
    "parse_fen",
    "format_fen",
    "parse_ucci_move",
    "START_FEN",
]


def parse_fen(fen: str) -> tuple[list[int], int, int, int]:
    """解析 FEN，返回 ``(squares, side, halfmove, fullmove)``。

    ``squares`` 为长度 90 的 list，``squares[row*9+col]`` 为有符号棋子值
    （正=红，负=黑，0=空）。非法 FEN 抛 ``ValueError``。
    """
    if not isinstance(fen, str):
        raise ValueError(f"FEN 必须是字符串: {fen!r}")
    parts = fen.strip().split()
    if not parts:
        raise ValueError("空 FEN")
    board_part = parts[0]
    side_part = parts[1] if len(parts) > 1 else "w"
    # 第 3、4 段（易位 / 吃过路兵占位符）忽略其内容，仅取 halfmove / fullmove。
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    if halfmove < 0 or fullmove < 1:
        raise ValueError(f"非法 halfmove/fullmove: {fen!r}")

    rows = board_part.split("/")
    if len(rows) != 10:
        raise ValueError(f"FEN board 段必须有 10 行: {board_part!r}")

    squares = [0] * 90
    for i, row_str in enumerate(rows):
        row = 9 - i  # rows[0] 对应 row 9（黑底线）
        col = 0
        for ch in row_str:
            if ch.isdigit():
                col += int(ch)
                if col > 9:
                    raise ValueError(f"FEN 行越界: {row_str!r}")
            else:
                if ch not in CHAR_TO_PIECE:
                    raise ValueError(f"未知棋子字符: {ch!r}")
                if col > 8:
                    raise ValueError(f"FEN 行越界: {row_str!r}")
                squares[row * 9 + col] = CHAR_TO_PIECE[ch]
                col += 1
        if col != 9:
            raise ValueError(f"FEN 行长度不足 9: {row_str!r}")

    if side_part in ("w", "W", "r", "R"):
        side = RED
    elif side_part in ("b", "B"):
        side = BLACK
    else:
        raise ValueError(f"未知走子方: {side_part!r}")

    # 基本合法性：双方各恰有一个王。
    red_kings = squares.count(KING)
    black_kings = squares.count(-KING)
    if red_kings != 1 or black_kings != 1:
        raise ValueError(f"FEN 双方必须各有一个王（红 {red_kings} 黑 {black_kings}）")

    return squares, side, halfmove, fullmove


def format_fen(squares: list[int], side: int, halfmove: int, fullmove: int) -> str:
    """把棋盘状态序列化为标准 FEN 字符串。"""
    rows = []
    for row in range(9, -1, -1):  # 从 row 9 到 row 0
        buf = []
        empty = 0
        for col in range(9):
            p = squares[row * 9 + col]
            if p == 0:
                empty += 1
            else:
                if empty:
                    buf.append(str(empty))
                    empty = 0
                ch = PIECE_TO_CHAR[abs(p)]
                buf.append(ch if p > 0 else ch.lower())
        if empty:
            buf.append(str(empty))
        rows.append("".join(buf))
    board_part = "/".join(rows)
    side_ch = "w" if side == RED else "b"
    return f"{board_part} {side_ch} - - {halfmove} {fullmove}"


def parse_ucci_move(text: str) -> tuple[int, int]:
    """解析 UCCI 着法字符串（如 ``h2e2``）为 ``(from_sq, to_sq)``。"""
    return uci_to_move(text)
