"""坐标系 / 棋子 / 动作编码常量与基础工具（全项目契约，见 DESIGN.md §2 §3 §5）。

- row 0 = 红方底线（下），row 9 = 黑方底线（上）
- col 0 = UCCI 列 'a'，col 8 = 'i'
- sq = row * 9 + col
- action = from_sq * 90 + to_sq
本模块零第三方依赖。
"""

from __future__ import annotations

ROWS = 10
COLS = 9
NUM_SQUARES = ROWS * COLS  # 90
ACTION_SIZE = NUM_SQUARES * NUM_SQUARES  # 8100

# 阵营
RED = 1
BLACK = -1

# 棋子类型（正=红，负=黑，0=空）
EMPTY = 0
KING = 1      # 帅/将
ADVISOR = 2   # 仕/士
ELEPHANT = 3  # 相/象
HORSE = 4     # 马
ROOK = 5      # 车
CANNON = 6    # 炮
PAWN = 7      # 兵/卒
PIECE_TYPES = (KING, ADVISOR, ELEPHANT, HORSE, ROOK, CANNON, PAWN)

# FEN 字母（红大写，黑小写）。解析时兼容 E/e->B/b（象）、H/h->N/n（马）。
PIECE_TO_CHAR = {
    KING: "K", ADVISOR: "A", ELEPHANT: "B", HORSE: "N",
    ROOK: "R", CANNON: "C", PAWN: "P",
}
CHAR_TO_PIECE = {}
for _pt, _ch in PIECE_TO_CHAR.items():
    CHAR_TO_PIECE[_ch] = _pt          # 红
    CHAR_TO_PIECE[_ch.lower()] = -_pt  # 黑
CHAR_TO_PIECE["E"] = ELEPHANT
CHAR_TO_PIECE["e"] = -ELEPHANT
CHAR_TO_PIECE["H"] = HORSE
CHAR_TO_PIECE["h"] = -HORSE

# 中文名（红, 黑）
PIECE_CN = {
    KING: ("帅", "将"), ADVISOR: ("仕", "士"), ELEPHANT: ("相", "象"),
    HORSE: ("马", "马"), ROOK: ("车", "车"), CANNON: ("炮", "炮"),
    PAWN: ("兵", "卒"),
}

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

# 启发式材料价值（单位：兵=1），过河兵另算
MATERIAL_VALUE = {
    KING: 0.0, ADVISOR: 2.0, ELEPHANT: 2.0, HORSE: 4.0,
    ROOK: 9.0, CANNON: 4.5, PAWN: 1.0,
}
PAWN_CROSSED_VALUE = 2.0  # 过河兵
MATERIAL_SCALE = 12.0     # tanh(material_diff / MATERIAL_SCALE)

# 河界 / 九宫
RED_RIVER_BANK = 4    # 红方本土 row <= 4
BLACK_RIVER_BANK = 5  # 黑方本土 row >= 5
PALACE_COLS = (3, 4, 5)
RED_PALACE_ROWS = (0, 1, 2)
BLACK_PALACE_ROWS = (7, 8, 9)

FILE_CHARS = "abcdefghi"


def sq(row: int, col: int) -> int:
    return row * 9 + col


def sq_row(s: int) -> int:
    return s // 9


def sq_col(s: int) -> int:
    return s % 9


def flip_sq(s: int) -> int:
    """180° 翻转（红黑视角互换）。"""
    return 89 - s


def sq_to_str(s: int) -> str:
    """UCCI 格子名，如 sq(2,7) -> 'h2'。"""
    return FILE_CHARS[s % 9] + str(s // 9)


def str_to_sq(text: str) -> int:
    col = FILE_CHARS.index(text[0].lower())
    row = int(text[1:])
    if not 0 <= row <= 9:
        raise ValueError(f"bad square: {text!r}")
    return row * 9 + col


def move_to_uci(move: tuple[int, int]) -> str:
    """(from_sq, to_sq) -> 'h2e2'"""
    return sq_to_str(move[0]) + sq_to_str(move[1])


def uci_to_move(text: str) -> tuple[int, int]:
    text = text.strip()
    if len(text) != 4:
        raise ValueError(f"bad move: {text!r}")
    return str_to_sq(text[:2]), str_to_sq(text[2:])


def move_to_action(move: tuple[int, int]) -> int:
    return move[0] * 90 + move[1]


def action_to_move(action: int) -> tuple[int, int]:
    return action // 90, action % 90


def flip_action(action: int) -> int:
    return flip_sq(action // 90) * 90 + flip_sq(action % 90)


def side_name(side: int) -> str:
    return "red" if side == RED else "black"
