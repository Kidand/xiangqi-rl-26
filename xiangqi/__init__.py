"""中国象棋规则引擎（纯 Python，零 torch 依赖）。

导出 Board 及常用坐标/编码/FEN/记法/记录工具。坐标与动作编码定义于
``xiangqi.constants``（全项目契约，见 DESIGN.md §2 §3 §5）。
"""

from __future__ import annotations

from .board import Board
from .constants import (
    ACTION_SIZE,
    BLACK,
    EMPTY,
    RED,
    START_FEN,
    action_to_move,
    flip_action,
    flip_sq,
    move_to_action,
    move_to_uci,
    sq,
    sq_col,
    sq_row,
    sq_to_str,
    str_to_sq,
    uci_to_move,
)
from .fen import format_fen, parse_fen, parse_ucci_move
from .notation import move_to_chinese
from .records import (
    RECORD_FORMAT,
    append_jsonl,
    parse_text_record,
    read_records,
    replay_record,
    write_record,
)

__all__ = [
    "Board",
    # 常量 / 坐标
    "RED",
    "BLACK",
    "EMPTY",
    "ACTION_SIZE",
    "START_FEN",
    "sq",
    "sq_row",
    "sq_col",
    "flip_sq",
    "sq_to_str",
    "str_to_sq",
    "move_to_uci",
    "uci_to_move",
    "move_to_action",
    "action_to_move",
    "flip_action",
    # FEN
    "parse_fen",
    "format_fen",
    "parse_ucci_move",
    # 记法
    "move_to_chinese",
    # 记录
    "RECORD_FORMAT",
    "write_record",
    "append_jsonl",
    "read_records",
    "replay_record",
    "parse_text_record",
]
