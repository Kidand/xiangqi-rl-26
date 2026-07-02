# -*- coding: utf-8 -*-
"""
GUI FastAPI 后端（DESIGN.md §13）。

入口:
    python -m gui.server --port 8000 --model checkpoints/best.pt --sims 400 --device cpu

功能:
- 内存对局 store（dict, uuid game_id）
- hva 人走后自动 AI 回着；AI 计算放 asyncio.to_thread
- ava 用 /ai_move 单步推进
- 模型热加载（/api/model/load）；未加载时 hva/ava 返回 400
- hvh 与回放无需模型
- 静态托管 gui/web 于 /
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 导入规则引擎
from xiangqi.board import Board
from xiangqi.constants import (
    ACTION_SIZE,
    RED,
    BLACK,
    START_FEN,
    move_to_uci,
    uci_to_move,
    side_name,
)
from xiangqi.notation import move_to_chinese
from xiangqi.records import (
    RECORD_FORMAT,
    read_records,
    replay_record,
    parse_text_record,
)

# RL 模块（延迟 import 以免模型未装时报错）
from rl.config import MCTSConfig
from rl.mcts import search as mcts_search
from rl.model import XiangqiNet, load_checkpoint

# ===========================================================================
# 路径常量
# ===========================================================================
_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEB_DIR = Path(__file__).resolve().parent / "web"
_RECORDS_DIR = _REPO_ROOT / "records"

# ===========================================================================
# 全局模型状态
# ===========================================================================
@dataclass
class _ModelState:
    model: Optional[XiangqiNet] = None
    path: str = ""
    device: str = "cpu"
    blocks: int = 0
    filters: int = 0
    params: int = 0
    history_steps: int = 2

_model_state = _ModelState()

def _is_model_loaded() -> bool:
    return _model_state.model is not None

def _make_eval_fn(model: XiangqiNet, device: str, history_steps: int):
    """构造网络评估函数（numpy 接口）。"""
    dev = torch.device(device)

    def eval_fn(states: np.ndarray):
        # states: (B, C, 10, 9) float32
        with torch.no_grad():
            x = torch.from_numpy(states).to(dev)
            logits, values = model(x)
            policy_probs = F.softmax(logits, dim=-1).cpu().numpy()
            vals = values.cpu().numpy()
        return policy_probs, vals

    return eval_fn


# ===========================================================================
# 对局数据结构
# ===========================================================================
@dataclass
class _MoveEntry:
    move: str        # UCCI，如 "h2e2"
    chinese: str
    fen_after: str


@dataclass
class _GameData:
    game_id: str
    board: Board
    mode: str          # "hvh" | "hva" | "ava"
    human_side: str    # "red" | "black"（ava 模式无意义，默认 "red"）
    sims: int
    move_history: List[_MoveEntry] = field(default_factory=list)
    # 认输状态（规则引擎不支持认输，在数据层处理）
    resigned: bool = False
    resign_result: Optional[str] = None  # "1-0" or "0-1"


# 内存 store
_games: Dict[str, _GameData] = {}


def _new_game_id() -> str:
    return str(uuid.uuid4())


# ===========================================================================
# GameState 序列化
# ===========================================================================
def _game_is_over(game: _GameData) -> bool:
    """对局是否结束（含认输）。"""
    return game.resigned or (game.board.result() is not None)


def _build_game_state(
    game: _GameData,
    ai_move_str: Optional[str] = None,
    eval_info: Optional[dict] = None,
) -> dict:
    """把 _GameData 序列化为 DESIGN §13 规定的 GameState dict。"""
    board = game.board

    # 认输优先于规则引擎结果
    if game.resigned:
        result = game.resign_result
        termination = "resign"
        game_over = True
        legal_uci: list = []
    else:
        result = board.result()
        termination = board.termination() if result is not None else None
        game_over = result is not None
        legal_uci = [move_to_uci(m) for m in board.legal_moves()] if not game_over else []

    last_m = board.last_move
    last_move_uci = move_to_uci(last_m) if last_m else None

    gs: dict = {
        "game_id": game.game_id,
        "fen": board.fen(),
        "side_to_move": side_name(board.side_to_move),
        "legal_moves": legal_uci,
        "last_move": last_move_uci,
        "in_check": board.is_in_check(),
        "game_over": game_over,
        "result": result,
        "termination": termination,
        "move_history": [
            {"move": e.move, "chinese": e.chinese, "fen_after": e.fen_after}
            for e in game.move_history
        ],
        "eval": eval_info,
    }
    if ai_move_str is not None:
        gs["ai_move"] = ai_move_str
    return gs


# ===========================================================================
# MCTS 辅助
# ===========================================================================
def _run_mcts_sync(
    board: Board,
    sims: int,
) -> tuple[np.ndarray, float]:
    """同步跑 MCTS，返回 (visit_counts[8100], root_value 当前走子方视角)。"""
    ms = _model_state
    eval_fn = _make_eval_fn(ms.model, ms.device, ms.history_steps)
    cfg = MCTSConfig(
        num_sims=sims,
        batch_size=min(64, sims),
    )
    board_copy = board.copy()
    visits, root_val = mcts_search(
        board_copy, eval_fn, cfg, sims, add_noise=False,
        history_steps=ms.history_steps,
    )
    return visits, root_val


def _visits_to_top_moves(board: Board, visits: np.ndarray, top_n: int = 5) -> list:
    """把 visit_counts 转为 top_moves 列表（UCCI + 中文 + prob + visits）。"""
    legal = board.legal_moves()
    if not legal:
        return []
    legal_actions = {}
    for mv in legal:
        a = mv[0] * 90 + mv[1]
        legal_actions[a] = mv

    total_visits = float(visits.sum())
    if total_visits < 1:
        return []

    scored = []
    for a, mv in legal_actions.items():
        v = float(visits[a])
        if v > 0:
            scored.append((v, mv))
    scored.sort(key=lambda x: -x[0])
    top = scored[:top_n]

    result = []
    for v, mv in top:
        uci = move_to_uci(mv)
        try:
            cn = move_to_chinese(board, mv)
        except Exception:
            cn = uci
        result.append({
            "move": uci,
            "chinese": cn,
            "prob": v / total_visits,
            "visits": int(v),
        })
    return result


def _pick_best_move(board: Board, visits: np.ndarray) -> Optional[tuple]:
    """取 visit argmax 对应合法着法。"""
    legal = board.legal_moves()
    if not legal:
        return None
    best_move = None
    best_v = -1
    for mv in legal:
        a = mv[0] * 90 + mv[1]
        v = float(visits[a])
        if v > best_v:
            best_v = v
            best_move = mv
    return best_move


# ===========================================================================
# 执行走子并记录历史
# ===========================================================================
def _apply_move(game: _GameData, mv: tuple) -> None:
    board = game.board
    uci = move_to_uci(mv)
    try:
        cn = move_to_chinese(board, mv)
    except Exception:
        cn = uci
    board.push(mv)
    game.move_history.append(_MoveEntry(move=uci, chinese=cn, fen_after=board.fen()))


# ===========================================================================
# FastAPI 应用
# ===========================================================================
app = FastAPI(title="Xiangqi GUI Server", version="1.0.0")


# ── 统一错误格式：{"error": "..."}（DESIGN §13） ─────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """把 FastAPI HTTPException 转为 DESIGN §13 规定的 {"error": "..."} 格式。"""
    if isinstance(exc.detail, dict):
        content = exc.detail  # 已经是 dict，直接用
    else:
        content = {"error": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": f"请求参数错误: {exc.errors()}"},
    )


# ── Pydantic 请求模型 ─────────────────────────────────────────────────────────
class NewGameRequest(BaseModel):
    mode: str = "hvh"           # "hvh" | "hva" | "ava"
    human_side: str = "red"     # "red" | "black"
    fen: Optional[str] = None
    sims: int = 400


class MoveRequest(BaseModel):
    move: str                   # UCCI 着法如 "h2e2"


class AiMoveRequest(BaseModel):
    sims: Optional[int] = None


class ReplayLoadRequest(BaseModel):
    record: Optional[Any] = None  # v1 JSON dict
    text: Optional[str] = None    # 纯文本格式


class ModelLoadRequest(BaseModel):
    path: str


# ===========================================================================
# /api/game/new
# ===========================================================================
@app.post("/api/game/new")
async def new_game(req: NewGameRequest):
    mode = req.mode
    if mode not in ("hvh", "hva", "ava"):
        raise HTTPException(400, detail={"error": f"未知模式: {mode!r}"})

    # hva / ava 需要模型
    if mode in ("hva", "ava") and not _is_model_loaded():
        raise HTTPException(400, detail={"error": "未加载模型，请先通过 /api/model/load 加载模型"})

    try:
        board = Board.from_fen(req.fen) if req.fen else Board()
    except ValueError as e:
        raise HTTPException(400, detail={"error": f"非法 FEN: {e}"})

    game_id = _new_game_id()
    game = _GameData(
        game_id=game_id,
        board=board,
        mode=mode,
        human_side=req.human_side,
        sims=req.sims,
    )
    _games[game_id] = game

    ai_move_str = None
    eval_info = None

    # hva：若 AI 执红且初始局面红先，AI 先走
    if mode == "hva" and game.board.result() is None:
        ai_is_red = (req.human_side == "black")
        ai_side = RED if ai_is_red else BLACK
        if game.board.side_to_move == ai_side:
            visits, root_val = await asyncio.to_thread(
                _run_mcts_sync, game.board, game.sims
            )
            ai_mv = _pick_best_move(game.board, visits)
            if ai_mv:
                ai_move_str = move_to_uci(ai_mv)
                eval_info = {
                    "value": float(root_val),   # 当前走子方视角
                    "top_moves": _visits_to_top_moves(game.board, visits),
                }
                _apply_move(game, ai_mv)

    return _build_game_state(game, ai_move_str=ai_move_str, eval_info=eval_info)


# ===========================================================================
# GET /api/game/{id}
# ===========================================================================
@app.get("/api/game/{game_id}")
async def get_game(game_id: str):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})
    return _build_game_state(game)


# ===========================================================================
# POST /api/game/{id}/move
# ===========================================================================
@app.post("/api/game/{game_id}/move")
async def make_move(game_id: str, req: MoveRequest):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})

    if _game_is_over(game):
        raise HTTPException(400, detail={"error": "对局已结束"})

    # 解析着法
    try:
        mv = uci_to_move(req.move)
    except ValueError:
        raise HTTPException(400, detail={"error": f"着法格式错误: {req.move!r}"})

    # 合法性校验
    legal = game.board.legal_moves()
    if mv not in legal:
        raise HTTPException(400, detail={"error": f"非法着法: {req.move}"})

    # 应用人类走子
    _apply_move(game, mv)

    ai_move_str = None
    eval_info = None

    # hva 模式：AI 自动回着
    if game.mode == "hva" and game.board.result() is None:
        human_side_int = RED if game.human_side == "red" else BLACK
        # 轮到 AI 走
        if game.board.side_to_move != human_side_int:
            visits, root_val = await asyncio.to_thread(
                _run_mcts_sync, game.board, game.sims
            )
            ai_mv = _pick_best_move(game.board, visits)
            if ai_mv:
                ai_move_str = move_to_uci(ai_mv)
                eval_info = {
                    "value": float(root_val),
                    "top_moves": _visits_to_top_moves(game.board, visits),
                }
                _apply_move(game, ai_mv)

    return _build_game_state(game, ai_move_str=ai_move_str, eval_info=eval_info)


# ===========================================================================
# POST /api/game/{id}/ai_move
# ===========================================================================
@app.post("/api/game/{game_id}/ai_move")
async def ai_move(game_id: str, req: AiMoveRequest):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})

    if not _is_model_loaded():
        raise HTTPException(400, detail={"error": "未加载模型"})

    if _game_is_over(game):
        raise HTTPException(400, detail={"error": "对局已结束"})

    sims = req.sims if req.sims is not None else game.sims

    visits, root_val = await asyncio.to_thread(
        _run_mcts_sync, game.board, sims
    )
    ai_mv = _pick_best_move(game.board, visits)
    eval_info = {
        "value": float(root_val),
        "top_moves": _visits_to_top_moves(game.board, visits),
    }
    if ai_mv:
        _apply_move(game, ai_mv)

    return _build_game_state(game, eval_info=eval_info)


# ===========================================================================
# POST /api/game/{id}/undo
# ===========================================================================
@app.post("/api/game/{game_id}/undo")
async def undo(game_id: str):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})

    board = game.board

    # 认输后可撤销（还原认输状态，继续对局）
    if game.resigned:
        game.resigned = False
        game.resign_result = None
        return _build_game_state(game)

    if not game.move_history:
        raise HTTPException(400, detail={"error": "没有可撤销的着法"})

    # hva：撤两步（AI + 人），hvh：撤一步
    steps = 2 if game.mode == "hva" else 1
    steps = min(steps, len(game.move_history))

    for _ in range(steps):
        if game.move_history:
            board.pop()
            game.move_history.pop()

    return _build_game_state(game)


# ===========================================================================
# GET /api/game/{id}/hint
# ===========================================================================
@app.get("/api/game/{game_id}/hint")
async def hint(game_id: str, sims: int = Query(default=400)):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})

    if not _is_model_loaded():
        raise HTTPException(400, detail={"error": "未加载模型"})

    if _game_is_over(game):
        raise HTTPException(400, detail={"error": "对局已结束"})

    visits, root_val = await asyncio.to_thread(
        _run_mcts_sync, game.board, sims
    )
    best = _pick_best_move(game.board, visits)
    if best is None:
        raise HTTPException(400, detail={"error": "无合法着法"})

    best_uci = move_to_uci(best)
    try:
        best_cn = move_to_chinese(game.board, best)
    except Exception:
        best_cn = best_uci

    top_moves = _visits_to_top_moves(game.board, visits)

    return {
        "move": best_uci,
        "chinese": best_cn,
        "value": float(root_val),
        "top_moves": top_moves,
    }


# ===========================================================================
# GET /api/game/{id}/export
# ===========================================================================
@app.get("/api/game/{game_id}/export")
async def export_game(game_id: str):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})

    board = game.board
    moves = [e.move for e in game.move_history]

    # 从头重演得到每步 FEN
    start_fen = START_FEN  # 当前无法追溯初始 FEN，用标准开局
    # 实际上我们需要保存初始 FEN，重建以获得 fens 列表
    # 利用已有的 move_history 中的 fen_after（含起始 FEN）
    fens: list = []
    # 初始 FEN = 执行第一步之前的局面 = 逆推
    # 通过从头 replay 获取
    try:
        replay = replay_record({
            "format": RECORD_FORMAT,
            "start_fen": _get_game_start_fen(game),
            "moves": moves,
        })
        fens = replay["fens"]
    except Exception:
        fens = [e.fen_after for e in game.move_history]

    # 结果优先取认输，其次取规则引擎
    final_result = game.resign_result if game.resigned else board.result()
    final_term = "resign" if game.resigned else (board.termination() if board.result() else None)

    record = {
        "format": RECORD_FORMAT,
        "start_fen": _get_game_start_fen(game),
        "moves": moves,
        "result": final_result,
        "termination": final_term,
        "plies": len(moves),
        "red": "human" if game.human_side == "red" else ("ai" if game.mode != "hvh" else "human"),
        "black": "human" if game.human_side == "black" else ("ai" if game.mode != "hvh" else "human"),
        "meta": {
            "mode": game.mode,
            "sims": game.sims,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "fens": fens,
    }
    return record


# ===========================================================================
# POST /api/game/{id}/resign
# ===========================================================================
@app.post("/api/game/{game_id}/resign")
async def resign(game_id: str):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})

    if _game_is_over(game):
        raise HTTPException(400, detail={"error": "对局已结束"})

    # 确定认输方：hvh 当前走子方认输，hva 人类方认输
    if game.mode == "hva":
        resigning_side = RED if game.human_side == "red" else BLACK
    else:
        resigning_side = game.board.side_to_move

    winner = -resigning_side
    game.resigned = True
    game.resign_result = "1-0" if winner == RED else "0-1"

    return _build_game_state(game)


# ===========================================================================
# POST /api/replay/load
# ===========================================================================
@app.post("/api/replay/load")
async def replay_load(req: ReplayLoadRequest):
    if req.record is not None:
        record_dict = req.record if isinstance(req.record, dict) else dict(req.record)
    elif req.text is not None:
        record_dict = parse_text_record(req.text)
    else:
        raise HTTPException(400, detail={"error": "需要 record 或 text 字段"})

    try:
        result = replay_record(record_dict)
    except ValueError as e:
        raise HTTPException(400, detail={"error": f"重演失败: {e}"})

    return {
        "fens": result["fens"],
        "moves": result["moves"],
        "chinese": result["chinese"],
        "result": result["result"],
        "start_fen": record_dict.get("start_fen", START_FEN),
    }


# ===========================================================================
# GET /api/records/list
# ===========================================================================
@app.get("/api/records/list")
async def records_list():
    files_info = []
    if _RECORDS_DIR.exists():
        for p in sorted(_RECORDS_DIR.rglob("*.jsonl")) + sorted(_RECORDS_DIR.rglob("*.json")):
            if p.is_file():
                try:
                    recs = read_records(p)
                    game_count = len(recs)
                except Exception:
                    game_count = 0
                # 相对于 records/ 的路径，用正斜杠
                rel = p.relative_to(_RECORDS_DIR).as_posix()
                files_info.append({
                    "path": rel,
                    "games": game_count,
                    "mtime": p.stat().st_mtime,
                })
    return {"files": files_info}


# ===========================================================================
# GET /api/records/get
# ===========================================================================
@app.get("/api/records/get")
async def records_get(path: str = Query(...), index: int = Query(default=0)):
    # 防路径穿越：resolve 后必须在 records/ 内
    try:
        target = (_RECORDS_DIR / path).resolve()
    except Exception:
        raise HTTPException(400, detail={"error": "非法路径"})

    try:
        records_resolved = _RECORDS_DIR.resolve()
    except Exception:
        raise HTTPException(500, detail={"error": "服务器路径错误"})

    # 检查路径是否在 records/ 内
    try:
        target.relative_to(records_resolved)
    except ValueError:
        raise HTTPException(403, detail={"error": "路径超出 records/ 范围（路径穿越被拒绝）"})

    if not target.exists() or not target.is_file():
        raise HTTPException(404, detail={"error": "文件不存在"})

    try:
        recs = read_records(target)
    except Exception as e:
        raise HTTPException(400, detail={"error": f"读取失败: {e}"})

    if index < 0 or index >= len(recs):
        raise HTTPException(404, detail={"error": f"索引 {index} 超出范围（共 {len(recs)} 局）"})

    return recs[index]


# ===========================================================================
# POST /api/model/load
# ===========================================================================
@app.post("/api/model/load")
async def model_load(req: ModelLoadRequest):
    path = Path(req.path)
    if not path.is_absolute():
        path = _REPO_ROOT / path

    if not path.exists():
        raise HTTPException(404, detail={"error": f"文件不存在: {req.path}"})

    try:
        model, ckpt = await asyncio.to_thread(
            load_checkpoint, str(path), _model_state.device
        )
    except Exception as e:
        raise HTTPException(400, detail={"error": f"加载失败: {e}"})

    model.eval()
    mc = ckpt.get("model_config", {})

    _model_state.model = model
    _model_state.path = str(path)
    _model_state.blocks = mc.get("blocks", 0)
    _model_state.filters = mc.get("filters", 0)
    _model_state.history_steps = mc.get("history_steps", 2)
    _model_state.params = sum(p.numel() for p in model.parameters())

    info = {
        "path": str(path),
        "blocks": _model_state.blocks,
        "filters": _model_state.filters,
        "history_steps": _model_state.history_steps,
        "params": _model_state.params,
        "device": _model_state.device,
        "iteration": ckpt.get("iteration", 0),
    }
    return {"ok": True, "info": info}


# ===========================================================================
# GET /api/model/info
# ===========================================================================
@app.get("/api/model/info")
async def model_info():
    if not _is_model_loaded():
        return {
            "loaded": False,
            "path": None,
            "params": None,
            "blocks": None,
            "filters": None,
            "device": _model_state.device,
        }
    return {
        "loaded": True,
        "path": _model_state.path,
        "params": _model_state.params,
        "blocks": _model_state.blocks,
        "filters": _model_state.filters,
        "device": _model_state.device,
    }


# ===========================================================================
# 静态文件（最后挂载，以免覆盖 API 路由）
# ===========================================================================
if _WEB_DIR.exists():
    # 将 /index.html 映射到 /
    @app.get("/")
    async def index():
        return FileResponse(str(_WEB_DIR / "index.html"))

    # 挂载静态资源
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")


# ===========================================================================
# 辅助：推算游戏起始 FEN
# ===========================================================================
def _get_game_start_fen(game: _GameData) -> str:
    """从历史回退推算起始 FEN（简单版：从当前局面逆推着法数次 pop）。"""
    board = game.board
    # 通过把着法数从当前 board 逐步 pop 得到初始局面
    n = len(game.move_history)
    if n == 0:
        return board.fen()
    # 复制局面逆推
    tmp = board.copy()
    for _ in range(n):
        try:
            tmp.pop()
        except Exception:
            break
    return tmp.fen()


# ===========================================================================
# 命令行入口
# ===========================================================================
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Xiangqi GUI 后端")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--model", default=None, help="模型检查点路径（可选）")
    parser.add_argument("--sims", type=int, default=400, help="AI 默认模拟次数")
    parser.add_argument("--device", default="cpu", help="推理设备（cpu/cuda）")
    return parser.parse_args(argv)


def _load_model_at_startup(model_path: str, device: str) -> None:
    """启动时同步加载模型。"""
    path = Path(model_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        print(f"[WARNING] 模型文件不存在: {path}，跳过加载", file=sys.stderr)
        return
    try:
        model, ckpt = load_checkpoint(str(path), device)
        model.eval()
        mc = ckpt.get("model_config", {})
        _model_state.model = model
        _model_state.path = str(path)
        _model_state.device = device
        _model_state.blocks = mc.get("blocks", 0)
        _model_state.filters = mc.get("filters", 0)
        _model_state.history_steps = mc.get("history_steps", 2)
        _model_state.params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] 模型已加载: {path} ({_model_state.params:,} 参数)", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}", file=sys.stderr)


def main(argv=None):
    import uvicorn

    args = _parse_args(argv)

    # 设置设备
    _model_state.device = args.device

    # 启动时加载模型（如果指定）
    if args.model:
        _load_model_at_startup(args.model, args.device)

    # 注意：必须把 app 对象直接传给 uvicorn，而不是 "gui.server:app" 导入字符串。
    # 以 `python -m gui.server` 运行时本模块名为 __main__，_load_model_at_startup
    # 修改的是 __main__._model_state；若用导入字符串会让 uvicorn 另行 import 一个
    # 独立的 gui.server 模块（其 _model_state 为空），导致启动加载的模型对路由不可见。
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
