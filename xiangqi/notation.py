"""中文纵线记法（如 ``炮二平五``、``马八进七``）。见 DESIGN.md §4 要点。

规则：
- 红方列号用汉字 九..一（col 0=九, col 8=一）；黑方列号用阿拉伯 1..9（col 0=1）。
- 动词：``平``（同行）、``进``（向敌方）、``退``（向己方）。
- 车/炮/兵/将 进退给步数；马/象/士 进退给目的列号。
- 同列多子用 前/后（三个用 前/中/后，更多用数字），此时省略起始列号。
"""

from __future__ import annotations

from .constants import (
    ADVISOR,
    CANNON,
    ELEPHANT,
    HORSE,
    KING,
    PAWN,
    PIECE_CN,
    RED,
    ROOK,
)

__all__ = ["move_to_chinese"]

_CN_NUM = "零一二三四五六七八九"
# 进退给步数 / 平给目的列的棋子（直线子）：车、炮、兵、将
_LINEAR = (ROOK, CANNON, PAWN, KING)


def _cn_col(col: int, side: int) -> str:
    """列号：红=汉字九..一，黑=阿拉伯 1..9。"""
    if side == RED:
        return _CN_NUM[9 - col]
    return str(col + 1)


def _cn_num(value: int, side: int) -> str:
    """步数：红用汉字，黑用阿拉伯。"""
    if side == RED:
        return _CN_NUM[value]
    return str(value)


def move_to_chinese(board, move: tuple[int, int]) -> str:
    """把着法转换为中文纵线记法字符串。``board`` 需暴露 ``squares``。"""
    frm, to = move
    squares = board.squares
    p = squares[frm]
    if p == 0:
        raise ValueError(f"起点无子: {move}")
    side = RED if p > 0 else -RED
    ptype = p if p > 0 else -p
    name = PIECE_CN[ptype][0 if side == RED else 1]

    fr_row, fr_col = divmod(frm, 9)
    to_row, to_col = divmod(to, 9)

    # 同类同列子（用于前/后消歧）
    same = [s for s in range(90) if squares[s] == p and s % 9 == fr_col]
    prefix = None
    if len(same) >= 2:
        # 按“前”（靠近敌方）到“后”排序：红方 row 越大越前，黑方 row 越小越前。
        same.sort(key=lambda s: (s // 9), reverse=(side == RED))
        idx = same.index(frm)
        count = len(same)
        if count == 2:
            labels = ["前", "后"]
        elif count == 3:
            labels = ["前", "中", "后"]
        else:
            labels = [_cn_num(i + 1, side) for i in range(count)]
        prefix = labels[idx] + name
    else:
        prefix = name + _cn_col(fr_col, side)

    # 动词 + 目的
    if to_row == fr_row:
        verb = "平"
        dest = _cn_col(to_col, side)
    else:
        forward = (to_row > fr_row) if side == RED else (to_row < fr_row)
        verb = "进" if forward else "退"
        if ptype in _LINEAR:
            dest = _cn_num(abs(to_row - fr_row), side)  # 步数
        else:
            dest = _cn_col(to_col, side)  # 马/象/士 给目的列号

    return prefix + verb + dest
