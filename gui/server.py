# -*- coding: utf-8 -*-
"""
GUI FastAPI 后端（DESIGN.md §13）。

入口:
    python -m gui.server --port 8000 --model ckpts/best.pt --sims 400 --device cpu

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
    normalize_evals,
    read_records,
    replay_record,
    parse_text_record,
    update_record,
)

# RL 模块（延迟 import 以免模型未装时报错）
from rl.config import MCTSConfig
from rl.mcts import search as mcts_search
from rl.model import XiangqiNet, load_checkpoint
from rl.selfplay import sample_action_from_counts

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
    default_sims: int = 400  # Default MCTS simulations, set from CLI --sims

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
    # 终局自动落盘的记录文件路径（每局至多一份；悔棋后再次终局以新文件替换旧文件）
    record_file: Optional[Path] = None
    # 每个局面一个胜率槽位（DESIGN §12 evals）：长度恒 = len(move_history)+1，
    # evals[i] 对应第 i 个局面（fens[i]），为 None 或 {"value_red","sims","source"}。
    evals: List[Optional[dict]] = field(default_factory=lambda: [None])
    # 每局一把异步锁：所有读写该局 board 的端点在锁内完成"读-算-写"全链路
    # （含跨 await asyncio.to_thread 的 MCTS 窗口），避免工作线程 copy()/MCTS 与
    # 事件循环线程 push/pop 交错撕裂局面；不同 game_id 之间互不阻塞。
    # 只在端点层获取；内部辅助函数（_apply_move/_get_game_start_fen 等）不得重复获取。
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False, repr=False)


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
) -> tuple[np.ndarray, float, np.ndarray]:
    """同步跑 MCTS，返回 (visit_counts[8100], root_value 当前走子方视角,
    q_values[8100] 根走子方视角、未访问边 NaN)。"""
    ms = _model_state
    eval_fn = _make_eval_fn(ms.model, ms.device, ms.history_steps)
    cfg = MCTSConfig(
        num_sims=sims,
        batch_size=min(64, sims),
    )
    board_copy = board.copy()
    visits, root_val, q_vals = mcts_search(
        board_copy, eval_fn, cfg, sims, add_noise=False,
        history_steps=ms.history_steps, return_q=True,
    )
    return visits, root_val, q_vals


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


# AI 落子采样参数（DESIGN §13）：GUI 全链路无 Dirichlet 噪声、前向确定，
# argmax 会使 ava 每局复读、hva 对相同人类走法回应固定。多样性由价值差
# 硬约束兜底：只在"Q 与最佳着法相差 ≤ dq"的着法中采样（Q 为根走子方视角，
# 与 root_value 同尺度 [-1,1]；纯 visits 截断会放进 Q 明显更差的离谱着法）。
_AI_OPENING_PLIES = 8     # 前 8 个半回合视为开局
_AI_OPENING_TAU = 1.0     # 开局温度
_AI_OPENING_KEEP = 0.15   # 开局 visits ≥ 最佳 15%（低访问边 Q 估计噪声大）
_AI_OPENING_DQ = 0.06     # 开局价值差上限（≈3% 胜率）
_AI_LATER_TAU = 0.35      # 开局后温度（近同分才换手）
_AI_LATER_KEEP = 0.50     # 开局后 visits ≥ 最佳 50%
_AI_LATER_DQ = 0.03       # 开局后价值差上限
_ai_rng = np.random.default_rng()


def _pick_ai_move(
    board: Board, visits: np.ndarray, q_vals: np.ndarray
) -> Optional[tuple]:
    """AI 落子选择：价值感知的截断温度采样；/hint 不走此路径（恒 argmax）。

    候选须同时满足 visits ≥ 最佳着法的 keep 倍、且 Q ≥ 参照 Q - dq
    （参照 = visits 最高着法的 Q）。ply 从局面 fullmove/side_to_move 推出：
    带着数字段的中局 FEN 不会被误判为开局高温；缺 fullmove 的短 FEN 默认 1
    会按开局对待——接受此边界（价值过滤已保证不会走明显劣着）。
    采样退化（截断后全零/数值异常）回退 argmax。
    """
    legal = board.legal_moves()
    if not legal:
        return None
    ply = (board.fullmove - 1) * 2 + (1 if board.side_to_move == BLACK else 0)
    opening = ply < _AI_OPENING_PLIES
    tau = _AI_OPENING_TAU if opening else _AI_LATER_TAU
    keep = _AI_OPENING_KEEP if opening else _AI_LATER_KEEP
    dq = _AI_OPENING_DQ if opening else _AI_LATER_DQ

    # 只在合法着法上取 visits（防御非法位残留）
    masked = np.zeros_like(visits, dtype=np.float64)
    for mv in legal:
        a = mv[0] * 90 + mv[1]
        masked[a] = visits[a]
    vmax = float(masked.max())
    if vmax <= 0:
        return _pick_best_move(board, visits)

    # 价值参照：visits 最高着法的 Q（低访问边 Q 噪声大，不作参照）
    a_ref = int(np.argmax(masked))
    q_ref = float(q_vals[a_ref])
    if np.isfinite(q_ref):
        # 双重截断：访问数下限 + 价值差上限（NaN 比较恒 False → 一并剔除）
        with np.errstate(invalid="ignore"):
            bad = ~(q_vals >= q_ref - dq)
        masked[bad] = 0.0
    masked[masked < vmax * keep] = 0.0

    mv = sample_action_from_counts(masked, tau, _ai_rng)
    return mv if mv is not None else _pick_best_move(board, visits)


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
    # 新局面尚无评估；保持 evals 与局面序列等长（DESIGN §12）
    game.evals.append(None)


def _eval_info_from_slot(game: _GameData) -> Optional[dict]:
    """当前局面若有已记录评估（evals 槽位），转为 GameState.eval。

    槽位存红方视角 value_red（DESIGN §12），故 side 恒为 "red"。
    悔棋响应用它让前端评估条恢复历史值而非回弹 50%。
    """
    idx = len(game.move_history)
    slot = game.evals[idx] if idx < len(game.evals) else None
    if slot is None:
        return None
    return {"value": float(slot["value_red"]), "side": "red", "top_moves": []}


def _record_position_eval(
    game: _GameData, value_mover: float, side_mover: int, sims: int
) -> None:
    """把一次 MCTS 评估写入当前局面的胜率槽位（须在 _apply_move 之前调用）。

    value 从走子方视角转为红方视角存储（DESIGN §12：value_red 恒红方视角）。
    已有更高 sims 的评估时不覆盖（hint/with_eval 可能用不同 sims 重评同一局面）。
    """
    idx = len(game.move_history)
    if idx >= len(game.evals):  # 防御性：理论上恒等长
        game.evals.extend([None] * (idx + 1 - len(game.evals)))
    old = game.evals[idx]
    if old is not None and old.get("sims", 0) >= sims:
        return
    value_red = float(value_mover) if side_mover == RED else -float(value_mover)
    game.evals[idx] = {"value_red": value_red, "sims": int(sims), "source": "play"}


# ===========================================================================
# 后台实时日志：AI 对局势的判断（打印到 stdout，供跑 gui.server 的终端观察）
# ===========================================================================
def _value_to_red_pct(value_mover: float, side_mover: int) -> float:
    """走子方视角 value ∈[-1,1] → 红方胜率百分比 ∈[0,100]。

    先把走子方视角转为红方视角（黑方走子取负），再线性映射到 [0,100]。
    """
    value_red = value_mover if side_mover == RED else -value_mover
    return (value_red + 1.0) / 2.0 * 100.0


def _log_eval(
    game: _GameData,
    tag: str,
    move_uci: Optional[str],
    chinese: Optional[str],
    value_mover: float,
    side_mover: int,
    sims: int,
    elapsed_ms: float,
    top_moves: Optional[list],
) -> None:
    """打印一行 AI 判断日志到 stdout（终端实时可见），失败不得影响接口。

    格式示例：
    [14:32:07] [对局 3f2a | 第12手] AI落子 炮二平五 | 红方胜率 63.2% (value=+0.264)
               | sims 400 | 耗时 1.8s | 备选: 马八进七 21% 车九平八 11%
    """
    try:
        pct = _value_to_red_pct(value_mover, side_mover)
        ts = time.strftime("%H:%M:%S")
        gid_short = game.game_id[:4]
        ply = len(game.move_history)
        move_part = f" {chinese}" if move_uci and chinese else ""
        elapsed_s = elapsed_ms / 1000.0

        # 备选：top_moves 剔除已落子（若有）后取前 3
        alts = [tm for tm in (top_moves or []) if tm.get("move") != move_uci][:3]
        alt_str = " ".join(f"{tm['chinese']} {tm['prob'] * 100:.0f}%" for tm in alts)

        line = (
            f"[{ts}] [对局 {gid_short} | 第{ply}手] {tag}{move_part} | "
            f"红方胜率 {pct:.1f}% (value={value_mover:+.3f}) | "
            f"sims {sims} | 耗时 {elapsed_s:.1f}s"
        )
        if alt_str:
            line += f" | 备选: {alt_str}"
        print(line, flush=True)
    except Exception:
        pass


def _log_game_over(game: _GameData) -> None:
    """对局结束时补一行结果日志（认输/自然终局均覆盖），失败不得影响接口。"""
    if not _game_is_over(game):
        return
    try:
        ts = time.strftime("%H:%M:%S")
        gid_short = game.game_id[:4]
        if game.resigned:
            result = game.resign_result
            termination = "resign"
        else:
            result = game.board.result()
            termination = game.board.termination()
        n = len(game.move_history)
        print(
            f"[{ts}] [对局 {gid_short}] 对局结束 result={result} termination={termination} 共{n}手",
            flush=True,
        )
    except Exception:
        pass


# ===========================================================================
# 对局记录构造与终局自动落盘（DESIGN §12）
# ===========================================================================
# 胜负中文映射（result → 文件名片段），用于自动落盘文件名。
_RESULT_ZH = {"1-0": "红胜", "0-1": "黑胜", "1/2-1/2": "和棋"}

# result → 红方视角 value（evals 终局槽位直接映射，DESIGN §12 source="result"）。
_RESULT_VALUE_RED = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


def _build_record(game: _GameData) -> dict:
    """把当前对局构造为 xiangqi-record-v1 记录 dict（含 fens）。

    导出端点与终局自动落盘共用本函数，保证两处内容逐字一致：
    起始 FEN 逆推、逐步重演得 fens、result/termination（认输优先）、
    red/black 身份三分支（ava→ai/ai，hvh→human/human，hva 按人执方）。
    """
    board = game.board
    moves = [e.move for e in game.move_history]
    start_fen = _get_game_start_fen(game)

    # 从头重演得到每步 FEN（失败时退回 move_history 里的 fen_after）
    fens: list = []
    try:
        replay = replay_record({
            "format": RECORD_FORMAT,
            "start_fen": start_fen,
            "moves": moves,
        })
        fens = replay["fens"]
    except Exception:
        fens = [e.fen_after for e in game.move_history]

    # 结果优先取认输，其次取规则引擎
    final_result = game.resign_result if game.resigned else board.result()
    final_term = "resign" if game.resigned else (board.termination() if board.result() else None)

    # 双方身份：ava 两方均为 ai；hvh 两方均为 human；hva 按人执方区分。
    if game.mode == "ava":
        red_who, black_who = "ai", "ai"
    elif game.mode == "hvh":
        red_who, black_who = "human", "human"
    else:  # hva
        red_who = "human" if game.human_side == "red" else "ai"
        black_who = "human" if game.human_side == "black" else "ai"

    # evals：与局面序列等长（防御性对齐）；终局局面若无评估按 result 直接映射。
    n_positions = len(moves) + 1
    evals: List[Optional[dict]] = list(game.evals[:n_positions])
    evals.extend([None] * (n_positions - len(evals)))
    if final_result in _RESULT_VALUE_RED and evals[-1] is None:
        evals[-1] = {
            "value_red": _RESULT_VALUE_RED[final_result],
            "sims": 0,
            "source": "result",
        }

    return {
        "format": RECORD_FORMAT,
        "start_fen": start_fen,
        "moves": moves,
        "result": final_result,
        "termination": final_term,
        "plies": len(moves),
        "red": red_who,
        "black": black_who,
        "meta": {
            "mode": game.mode,
            "sims": game.sims,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "fens": fens,
        "evals": evals,
    }


def _maybe_autosave_record(game: _GameData) -> None:
    """对局终局时自动把单局记录落盘到 ``records/gui/``（DESIGN §12）。

    - 未终局直接返回；
    - 运行时经模块全局 ``_RECORDS_DIR`` 取根目录（便于测试 monkeypatch）；
    - 文件名 ``YYYYMMDD-HHMMSS_<mode>_<胜负>.json``，同名冲突追加 -2/-3 序号；
    - 每局至多一份：悔棋后再次终局若新路径与旧文件不同，先删旧文件再写新文件；
    - 写入用 tmp + os.replace 原子换名；落盘失败不得让端点 5xx，仅打印警告。
    """
    if not _game_is_over(game):
        return
    try:
        record = _build_record(game)
        zh = _RESULT_ZH.get(record.get("result"), "未知")
        gui_dir = _RECORDS_DIR / "gui"
        gui_dir.mkdir(parents=True, exist_ok=True)

        base = f"{time.strftime('%Y%m%d-%H%M%S')}_{game.mode}_{zh}"
        # 同名冲突追加 -2/-3 序号；若冲突文件正是本局旧记录则复用（原地覆盖）。
        target = gui_dir / f"{base}.json"
        seq = 2
        while target.exists() and target != game.record_file:
            target = gui_dir / f"{base}-{seq}.json"
            seq += 1

        # 每局至多一份：新路径与旧文件不同则先删除旧文件。
        old = game.record_file
        if old is not None and old != target:
            old.unlink(missing_ok=True)

        # 原子写：tmp + os.replace（与仓库 checkpoint 保存习惯一致）。
        # tmp 名含随机后缀：与 /api/records/eval 写回（update_record 的 pid 后缀）
        # 及并发终局的其他对局互不冲突。
        data = json.dumps(record, ensure_ascii=False, indent=2)
        tmp = target.with_name(f"{target.name}.tmp-a{uuid.uuid4().hex[:8]}")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, target)
        game.record_file = target
    except Exception as e:
        print(f"[WARNING] 自动落盘对局记录失败: {e}", file=sys.stderr)


# ===========================================================================
# FastAPI 应用
# ===========================================================================
app = FastAPI(title="Xiangqi GUI Server", version="1.0.0")


# ── 静态资源 no-cache：前端 js/css/html 每次都走 etag 重新校验 ────────────────
# 否则浏览器会把 board.js/style.css 强缓存，改了 UI 普通刷新（F5）看不到更新，
# 必须硬刷新。no-cache 不等于不缓存：内容没变返回 304（省带宽），变了才下新的。
@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


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
    sims: Optional[int] = None  # 若 None，使用服务器默认值（从 CLI --sims 设置）
    ai_reply: bool = True       # False：即使 AI 先行也不含首着，客户端随后调 /ai_move


class MoveRequest(BaseModel):
    move: str                   # UCCI 着法如 "h2e2"
    ai_reply: bool = True       # False：hva 不自动回着（两段式渲染），客户端随后调 /ai_move


class AiMoveRequest(BaseModel):
    sims: Optional[int] = None


class ReplayLoadRequest(BaseModel):
    record: Optional[Any] = None  # v1 JSON dict
    text: Optional[str] = None    # 纯文本格式


class ModelLoadRequest(BaseModel):
    path: str


class RecordsEvalRequest(BaseModel):
    path: str                 # records/ 内的相对路径（同 /api/records/get）
    index: int = 0            # 文件内第几局
    sims: int = 128           # 每个局面的 MCTS 模拟次数（复盘补算默认低于对局）
    max_positions: int = 8    # 本批最多补算的局面数（分批防长阻塞/代理超时）


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

    # 使用请求中的 sims，若未指定则使用服务器默认值（从 CLI --sims 设置）
    sims = req.sims if req.sims is not None else _model_state.default_sims

    game_id = _new_game_id()
    game = _GameData(
        game_id=game_id,
        board=board,
        mode=mode,
        human_side=req.human_side,
        sims=sims,
    )
    _games[game_id] = game

    # game_id 在响应返回前对客户端不可见，理论上无并发；但发布到 _games 之后
    # 仍统一持锁完成"AI 首着 + 终局落盘"，与 DESIGN §13"锁内完成"的声明一致。
    async with game.lock:
        ai_move_str = None
        eval_info = None

        # hva：若 AI 执红且初始局面红先，AI 先走（ai_reply=false 时留给客户端 /ai_move）
        if mode == "hva" and req.ai_reply and game.board.result() is None:
            ai_is_red = (req.human_side == "black")
            ai_side = RED if ai_is_red else BLACK
            if game.board.side_to_move == ai_side:
                t0 = time.monotonic()
                visits, root_val, q_vals = await asyncio.to_thread(
                    _run_mcts_sync, game.board, game.sims
                )
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                ai_mv = _pick_ai_move(game.board, visits, q_vals)
                if ai_mv:
                    ai_move_str = move_to_uci(ai_mv)
                    top_moves = _visits_to_top_moves(game.board, visits)
                    eval_info = {
                        "value": float(root_val),   # AI 方（落子前的走子方）视角
                        "side": side_name(ai_side),  # 显式声明 value 是哪一方视角
                        "top_moves": top_moves,
                    }
                    _record_position_eval(game, float(root_val), ai_side, game.sims)
                    _apply_move(game, ai_mv)
                    _log_eval(
                        game, "AI落子", ai_move_str, game.move_history[-1].chinese,
                        float(root_val), ai_side, game.sims, elapsed_ms, top_moves,
                    )

        _log_game_over(game)
        _maybe_autosave_record(game)
        return _build_game_state(game, ai_move_str=ai_move_str, eval_info=eval_info)


# ===========================================================================
# GET /api/game/{id}
# ===========================================================================
@app.get("/api/game/{game_id}")
async def get_game(
    game_id: str,
    with_eval: bool = Query(default=False),
    sims: int = Query(default=400),
):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})

    # 全链路持锁：with_eval 的 MCTS 会 board.copy() + 数秒计算，须与 /move、/undo 等互斥。
    async with game.lock:
        eval_info = None
        # with_eval=true 且有模型且未终局时，跑一次 MCTS 得到当前走子方视角的评估。
        if with_eval and _is_model_loaded() and not _game_is_over(game):
            side_mover = game.board.side_to_move
            visits, root_val, _q = await asyncio.to_thread(
                _run_mcts_sync, game.board, sims
            )
            top_moves = _visits_to_top_moves(game.board, visits)
            eval_info = {
                "value": float(root_val),      # 当前走子方视角
                "side": side_name(side_mover),  # = 当前 side_to_move
                "top_moves": top_moves,
            }
            _record_position_eval(game, float(root_val), side_mover, sims)
        return _build_game_state(game, eval_info=eval_info)


# ===========================================================================
# POST /api/game/{id}/move
# ===========================================================================
@app.post("/api/game/{game_id}/move")
async def make_move(game_id: str, req: MoveRequest):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})

    # 全链路持锁：人类走子 + 可能的 AI 回着（含 to_thread 的数秒 MCTS）须原子完成，
    # 否则窗口内被 /undo 等改动局面会导致用旧 visits 匹配新局面而错配着法。
    async with game.lock:
        if _game_is_over(game):
            raise HTTPException(400, detail={"error": "对局已结束"})

        # hva 轮次守卫：两段式窗口内轮到 AI 时人类不得走子（否则会替 AI 落子）。
        if game.mode == "hva":
            human_side_int = RED if game.human_side == "red" else BLACK
            if game.board.side_to_move != human_side_int:
                raise HTTPException(409, detail={"error": "轮到 AI 走子，请等待 AI 回着"})

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

        # hva 模式：AI 自动回着（ai_reply=false 时留给客户端 /ai_move，两段式渲染）
        if game.mode == "hva" and req.ai_reply and game.board.result() is None:
            human_side_int = RED if game.human_side == "red" else BLACK
            # 轮到 AI 走
            if game.board.side_to_move != human_side_int:
                ai_side = game.board.side_to_move
                t0 = time.monotonic()
                visits, root_val, q_vals = await asyncio.to_thread(
                    _run_mcts_sync, game.board, game.sims
                )
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                ai_mv = _pick_ai_move(game.board, visits, q_vals)
                if ai_mv:
                    ai_move_str = move_to_uci(ai_mv)
                    top_moves = _visits_to_top_moves(game.board, visits)
                    eval_info = {
                        "value": float(root_val),
                        "side": side_name(ai_side),  # AI 方（落子前的走子方）视角
                        "top_moves": top_moves,
                    }
                    _record_position_eval(game, float(root_val), ai_side, game.sims)
                    _apply_move(game, ai_mv)
                    _log_eval(
                        game, "AI落子", ai_move_str, game.move_history[-1].chinese,
                        float(root_val), ai_side, game.sims, elapsed_ms, top_moves,
                    )

        _log_game_over(game)
        _maybe_autosave_record(game)
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

    # 全链路持锁：MCTS 计算与落子须原子完成，避免 visits 匹配到被改动后的局面。
    async with game.lock:
        if _game_is_over(game):
            raise HTTPException(400, detail={"error": "对局已结束"})

        # hva 两段式竞态守卫：/undo 若抢在 /ai_move 之前执行，轮次会回到人类方，
        # 此时 AI 不得代人走子（DESIGN §13）。
        if game.mode == "hva":
            human_side_int = RED if game.human_side == "red" else BLACK
            if game.board.side_to_move == human_side_int:
                raise HTTPException(409, detail={"error": "轮到人类走子，AI 不能代走"})

        sims = req.sims if req.sims is not None else game.sims

        ai_side = game.board.side_to_move
        t0 = time.monotonic()
        visits, root_val, q_vals = await asyncio.to_thread(
            _run_mcts_sync, game.board, sims
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        ai_mv = _pick_ai_move(game.board, visits, q_vals)
        top_moves = _visits_to_top_moves(game.board, visits)
        eval_info = {
            "value": float(root_val),
            "side": side_name(ai_side),  # AI 方（落子前的走子方）视角
            "top_moves": top_moves,
        }
        if ai_mv:
            ai_move_str = move_to_uci(ai_mv)
            _record_position_eval(game, float(root_val), ai_side, sims)
            _apply_move(game, ai_mv)
            _log_eval(
                game, "AI落子", ai_move_str, game.move_history[-1].chinese,
                float(root_val), ai_side, sims, elapsed_ms, top_moves,
            )

        _log_game_over(game)
        _maybe_autosave_record(game)
        return _build_game_state(game, eval_info=eval_info)


# ===========================================================================
# POST /api/game/{id}/undo
# ===========================================================================
@app.post("/api/game/{game_id}/undo")
async def undo(game_id: str):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})

    # 全链路持锁：pop 会改动 board，须与工作线程里的 copy()/MCTS 互斥。
    async with game.lock:
        board = game.board

        # 认输后可撤销（还原认输状态，继续对局）
        if game.resigned:
            game.resigned = False
            game.resign_result = None
            return _build_game_state(game, eval_info=_eval_info_from_slot(game))

        if not game.move_history:
            raise HTTPException(400, detail={"error": "没有可撤销的着法"})

        # hva：撤到轮到人类为止——AI 已回着时撤一整回合两步；两段式窗口
        # （人类刚落子、AI 未回着）只撤人类这一步。固定撤两步会把局面
        # 撤到 AI 轮次导致前端无路径恢复（DESIGN §13）。hvh：撤一步。
        if game.mode == "hva":
            human_side_int = RED if game.human_side == "red" else BLACK
            steps = 2 if game.board.side_to_move == human_side_int else 1
        else:
            steps = 1
        steps = min(steps, len(game.move_history))

        for _ in range(steps):
            if game.move_history:
                board.pop()
                game.move_history.pop()

        # evals 与局面序列保持等长（被撤销局面的评估一并丢弃）
        del game.evals[len(game.move_history) + 1:]

        # 撤后局面若曾被 AI 评估过，随响应带回，评估条恢复历史值
        return _build_game_state(game, eval_info=_eval_info_from_slot(game))


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

    # 全链路持锁：MCTS 用 board.copy() 起算，须与改动 board 的端点互斥，
    # 否则 await 返回后用旧局面的 visits 匹配新局面会给出错误提示。
    async with game.lock:
        if _game_is_over(game):
            raise HTTPException(400, detail={"error": "对局已结束"})

        side_mover = game.board.side_to_move
        t0 = time.monotonic()
        visits, root_val, _q = await asyncio.to_thread(
            _run_mcts_sync, game.board, sims
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        best = _pick_best_move(game.board, visits)
        if best is None:
            raise HTTPException(400, detail={"error": "无合法着法"})

        best_uci = move_to_uci(best)
        try:
            best_cn = move_to_chinese(game.board, best)
        except Exception:
            best_cn = best_uci

        top_moves = _visits_to_top_moves(game.board, visits)
        _record_position_eval(game, float(root_val), side_mover, sims)
        _log_eval(
            game, "提示", best_uci, best_cn,
            float(root_val), side_mover, sims, elapsed_ms, top_moves,
        )

        return {
            "move": best_uci,
            "chinese": best_cn,
            "value": float(root_val),
            "side": side_name(side_mover),  # value 为该走子方视角
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

    # 全链路持锁：_build_record 内 _get_game_start_fen 会对 board.copy() 逐步 pop
    # 逆推起始局面，且读取 board.result()/termination()，须与改动 board 的端点互斥。
    async with game.lock:
        return _build_record(game)


# ===========================================================================
# POST /api/game/{id}/resign
# ===========================================================================
@app.post("/api/game/{game_id}/resign")
async def resign(game_id: str):
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(404, detail={"error": "游戏不存在"})

    # 全链路持锁：读取 side_to_move 并写认输状态，须与改动 board 的端点互斥。
    async with game.lock:
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

        _log_game_over(game)
        _maybe_autosave_record(game)
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
        # 规范化 evals：与 fens 等长，缺失/非法项为 null（DESIGN §12/§13）
        "evals": normalize_evals(record_dict),
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
def _resolve_records_path(path: str) -> Path:
    """把请求里的相对路径解析到 records/ 内的既有文件；越界/缺失抛 HTTPException。"""
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
    return target


@app.get("/api/records/get")
async def records_get(path: str = Query(...), index: int = Query(default=0)):
    target = _resolve_records_path(path)

    try:
        recs = read_records(target)
    except Exception as e:
        raise HTTPException(400, detail={"error": f"读取失败: {e}"})

    if index < 0 or index >= len(recs):
        raise HTTPException(404, detail={"error": f"索引 {index} 超出范围（共 {len(recs)} 局）"})

    return recs[index]


# ===========================================================================
# POST /api/records/eval —— 复盘胜率补算（DESIGN §13）
# ===========================================================================
# 按 resolve 后的文件路径串行化补算，避免并发读-算-写回互相覆盖。
_records_eval_locks: Dict[str, asyncio.Lock] = {}


@app.post("/api/records/eval")
async def records_eval(req: RecordsEvalRequest):
    if not _is_model_loaded():
        raise HTTPException(400, detail={"error": "未加载模型，无法补算胜率"})

    target = _resolve_records_path(req.path)
    sims = max(1, min(int(req.sims), 1600))
    max_positions = max(1, min(int(req.max_positions), 64))

    lock = _records_eval_locks.setdefault(str(target), asyncio.Lock())
    async with lock:
        try:
            recs = await asyncio.to_thread(read_records, target)
        except Exception as e:
            raise HTTPException(400, detail={"error": f"读取失败: {e}"})

        if req.index < 0 or req.index >= len(recs):
            raise HTTPException(
                404, detail={"error": f"索引 {req.index} 超出范围（共 {len(recs)} 局）"}
            )
        record = recs[req.index]

        # 以 moves 重演为准得到局面序列（顺带校验记录合法性）
        try:
            fens = (await asyncio.to_thread(replay_record, record))["fens"]
        except ValueError as e:
            raise HTTPException(400, detail={"error": f"记录重演失败: {e}"})

        evals = normalize_evals(record)
        result = record.get("result")
        computed = 0
        changed = False

        for i, fen in enumerate(fens):
            if evals[i] is not None:
                continue
            board = Board.from_fen(fen)
            # 终局局面按结果直接映射（不跑 MCTS、不计补算额度）：
            # 末局面优先用记录 result（认输/长将等从 FEN 判不出），
            # 其余局面用 FEN 可判的规则终局（将死/困毙等）。
            mapped = None
            if i == len(fens) - 1 and result in _RESULT_VALUE_RED:
                mapped = _RESULT_VALUE_RED[result]
            else:
                r = board.result()
                if r is not None:
                    mapped = _RESULT_VALUE_RED.get(r, 0.0)
            if mapped is not None:
                evals[i] = {"value_red": mapped, "sims": 0, "source": "result"}
                changed = True
                continue

            if computed >= max_positions:
                continue  # 本批额度用完；后续槽位留待下一批
            side_mover = board.side_to_move
            _visits, root_val, _q = await asyncio.to_thread(_run_mcts_sync, board, sims)
            value_red = float(root_val) if side_mover == RED else -float(root_val)
            evals[i] = {"value_red": value_red, "sims": sims, "source": "replay"}
            computed += 1
            changed = True

        pending = sum(1 for e in evals if e is None)

        if changed:
            record = dict(record)
            record["evals"] = evals
            try:
                await asyncio.to_thread(update_record, target, req.index, record)
            except RuntimeError as e:
                # 文件被其他进程（如训练 append）改动 → 放弃写回，409 让前端停止循环
                raise HTTPException(409, detail={"error": str(e)})
            except Exception as e:
                raise HTTPException(500, detail={"error": f"记录写回失败: {e}"})

        return {"evals": evals, "pending": pending, "total": len(evals)}


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

    # 设置默认 MCTS 模拟次数（从 CLI --sims 参数）
    _model_state.default_sims = args.sims

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
