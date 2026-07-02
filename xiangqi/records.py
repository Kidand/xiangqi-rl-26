"""对局记录 xiangqi-record-v1 读写与重演（见 DESIGN.md §12）。

记录格式（JSON 一局一条）：
    {"format": "xiangqi-record-v1",
     "start_fen": "...", "moves": ["h2e2", ...],
     "result": "1-0", "termination": "checkmate", "plies": 87,
     "red": "...", "black": "...", "meta": {...}}

文件 ``.jsonl``（多局）或 ``.json``（单局或列表），本模块自适应读取。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .board import Board
from .constants import START_FEN, uci_to_move
from .notation import move_to_chinese

__all__ = [
    "RECORD_FORMAT",
    "write_record",
    "append_jsonl",
    "read_records",
    "replay_record",
    "parse_text_record",
]

RECORD_FORMAT = "xiangqi-record-v1"


def write_record(path: str | Path, record: dict) -> None:
    """把单局记录写为 ``.json``（覆盖）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, record: dict) -> None:
    """把单局记录追加为 ``.jsonl`` 的一行。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path: str | Path) -> list[dict]:
    """读取记录文件，返回记录 list（``.json``/``.jsonl`` 自适应）。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    # .json：可能是单条 dict 或 list
    data = json.loads(text)
    if isinstance(data, list):
        return data
    return [data]


def replay_record(record: dict) -> dict:
    """重演记录，返回 ``{"fens", "moves", "chinese", "result"}``。

    ``fens`` 含起点及每步之后的 FEN（长度 = 着法数 + 1）。
    非法着法抛 ``ValueError``。
    """
    start_fen = record.get("start_fen") or START_FEN
    moves = list(record.get("moves") or [])
    board = Board.from_fen(start_fen)

    fens = [board.fen()]
    chinese: list[str] = []
    for i, ucci in enumerate(moves):
        mv = uci_to_move(ucci)
        legal = board.legal_moves()
        if mv not in legal:
            raise ValueError(f"第 {i + 1} 着非法: {ucci}（局面 {board.fen()}）")
        chinese.append(move_to_chinese(board, mv))
        board.push(mv)
        fens.append(board.fen())

    result = record.get("result")
    if result is None:
        result = board.result()
    return {"fens": fens, "moves": moves, "chinese": chinese, "result": result}


def parse_text_record(text: str) -> dict:
    """解析纯文本记录为 xiangqi-record-v1 dict。

    首行为 FEN（含 ``/`` 判定）时作为起点，否则默认初始局面；
    其余按空白/换行分割为 UCCI 着法。
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines and "/" in lines[0]:
        start_fen = lines[0]
        move_lines = lines[1:]
    else:
        start_fen = START_FEN
        move_lines = lines
    moves: list[str] = []
    for ln in move_lines:
        moves.extend(ln.split())
    return {"format": RECORD_FORMAT, "start_fen": start_fen, "moves": moves}
