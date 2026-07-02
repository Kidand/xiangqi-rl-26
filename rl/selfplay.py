# -*- coding: utf-8 -*-
"""自我博弈 worker（多进程，批量叶子评估）——DESIGN.md §9-§11。

核心职责
--------
- ``selfplay_worker``：顶层函数（Windows/Linux spawn 兼容），加载 best 模型（eval），
  在单进程内并发 ``games_per_worker`` 盘对局，用 ``SearchTree.select_leaves`` 跨盘收集
  叶子 → stack 成一个大 batch → 一次网络前向 → 逐盘 ``apply_results``，最大化吞吐。
- 走子：温度采样 + 开局多样化（heuristics.opening_temperature）；一步杀捷径
  （mate_shortcut：遍历合法着法找立即将死则直接走，π 记该着 one-hot）；ResignTracker。
- 对局结束：组装训练样本（state uint8、稀疏 π=visits 归一后经 move_to_policy_index 转
  当前方视角、z 含和棋惩罚 + 材料混合 λ 且逐步符号 = 该步走子方视角）与
  xiangqi-record-v1 记录，放入 ``game_queue``，并附带统计（红/黑/和、plies、termination、
  resign 标记、nn_evals 等），供主进程实时聚合日志。

坐标 / 编码 / 材料 / 启发式一律 import 自 ``xiangqi.constants`` / ``rl.encoding`` /
``rl.heuristics``，本模块不重复定义。
"""

from __future__ import annotations

import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np

from xiangqi.board import Board
from xiangqi.constants import (
    BLACK,
    MATERIAL_SCALE,
    RED,
    START_FEN,
    action_to_move,
    move_to_uci,
)
from rl.config import Config
from rl.encoding import encode_board, move_to_policy_index
from rl.heuristics import (
    ResignTracker,
    apply_draw_penalty,
    blend_value,
    lambda_schedule,
    material_score,
)
from rl.mcts import SearchTree
from rl.model import load_checkpoint

__all__ = [
    "selfplay_worker",
    "build_eval_fn",
    "num_selfplay_workers",
    "worker_game_quota",
    "sample_action_from_counts",
]

# 记录用格式常量（与 xiangqi.records.RECORD_FORMAT 保持一致，避免在 worker 内引入
# 额外磁盘依赖——主进程负责实际写盘）。
_RECORD_FORMAT = "xiangqi-record-v1"


# ---------------------------------------------------------------------------
# 网络评估闭包：states[B,C,10,9] float32 -> (policy_probs[B,8100], values[B])
# ---------------------------------------------------------------------------
def build_eval_fn(model, device) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """把 torch 模型包装成 MCTS 需要的 net_eval_fn（softmax 概率 + 价值）。"""
    import torch

    dev = torch.device(device)

    @torch.no_grad()
    def fn(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32)).to(dev)
        logits, value = model(t)
        probs = torch.softmax(logits, dim=1)
        return (
            probs.detach().to("cpu").numpy(),
            value.detach().to("cpu").numpy().reshape(-1),
        )

    return fn


# ---------------------------------------------------------------------------
# worker 数量与配额（worker 与主进程须用同一公式，保证总局数恰为
# games_per_iteration，且 worker 签名无需额外参数）。
# ---------------------------------------------------------------------------
def num_selfplay_workers(cfg: Config) -> int:
    """本次自博弈的 worker 进程总数。

    - CUDA：``num_gpus * workers_per_gpu``（每 GPU round-robin 绑若干进程）。
    - CPU ：``max(1, workers_per_gpu)``（无 GPU 降级，本地冒烟）。
    """
    if str(cfg.train.device).startswith("cuda") and cfg.train.num_gpus > 0:
        return max(1, cfg.train.num_gpus * cfg.selfplay.workers_per_gpu)
    return max(1, cfg.selfplay.workers_per_gpu)


def worker_game_quota(cfg: Config, proc_id: int) -> int:
    """proc_id 号 worker 需产出的对局数（把 games_per_iteration 均摊到各 worker）。"""
    total = int(cfg.selfplay.games_per_iteration)
    n = num_selfplay_workers(cfg)
    base, rem = divmod(total, n)
    return base + (1 if proc_id < rem else 0)


# ---------------------------------------------------------------------------
# 走子采样：从访问计数（棋盘真实坐标 8100）按温度选一个着法。
# ---------------------------------------------------------------------------
def sample_action_from_counts(
    counts: np.ndarray, tau: float, rng: np.random.Generator
) -> Optional[Tuple[int, int]]:
    """按温度 tau 从访问计数采样一个着法。

    tau<=0（或极小）→ argmax；否则按 ``N^{1/tau}`` 归一后采样。
    counts 全零（无访问）时返回 None（调用方需回退到合法着法）。
    """
    nz = np.nonzero(counts)[0]
    if len(nz) == 0:
        return None
    c = counts[nz].astype(np.float64)
    if tau <= 1e-6:
        a = int(nz[int(np.argmax(c))])
    else:
        w = np.power(c, 1.0 / tau)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0.0:
            a = int(nz[int(np.argmax(c))])
        else:
            w /= s
            a = int(nz[int(rng.choice(len(nz), p=w))])
    return action_to_move(a)


# ---------------------------------------------------------------------------
# 单盘对局的运行时状态（一个 worker 内并发多个 slot）。
# ---------------------------------------------------------------------------
class _GameSlot:
    """一盘对局的搜索/采样状态机。

    状态流转：``new_move`` →（终局 finish / 一步杀直接走）→ ``searching`` → ``decide``
    →（走子 / 认输 finish）→ ``new_move`` …… 仅 ``searching`` 阶段需要网络前向，可跨盘凑批。
    """

    __slots__ = (
        "game_id",
        "board",
        "tree",
        "rng",
        "resign",
        "state",
        "ply",
        "nn_evals",
        "moves",
        "sample_states",
        "sample_pol_idx",
        "sample_pol_prob",
        "sample_sides",
        "sample_mat_red",
        "result",
        "termination",
        "emitted",
        "_last_visits",
        "_calib_consec_red",
        "_calib_consec_black",
        "_calib_resign_side",
    )

    def __init__(self, game_id: int, cfg: Config, mcts_cfg, rng: np.random.Generator) -> None:
        self.game_id = game_id
        self.board = Board(
            max_plies=int(cfg.selfplay.max_game_plies),
            no_capture_plies=int(cfg.selfplay.no_capture_draw_plies),
        )
        self.rng = rng
        self.tree = SearchTree(
            self.board,
            mcts_cfg,
            add_noise=True,  # 自博弈根节点加 Dirichlet 噪声
            history_steps=int(cfg.model.history_steps),
            rng=rng,
        )
        self.resign = ResignTracker(cfg.selfplay, game_id=game_id)
        self.state = "new_move"
        self.ply = 0
        self.nn_evals = 0
        self.moves: List[str] = []
        self.sample_states: List[np.ndarray] = []
        self.sample_pol_idx: List[np.ndarray] = []
        self.sample_pol_prob: List[np.ndarray] = []
        self.sample_sides: List[int] = []
        self.sample_mat_red: List[float] = []
        self.result: Optional[str] = None
        self.termination: Optional[str] = None
        self.emitted = False
        self._last_visits = -1
        # 认输误判（false-positive）校准所需的镜像计数（仅校准局用）。
        self._calib_consec_red = 0
        self._calib_consec_black = 0
        self._calib_resign_side: Optional[int] = None


def _find_mate_move(board: Board) -> Optional[Tuple[int, int]]:
    """遍历合法着法，返回第一个立即让对手无解（将死/困毙）的着法，否则 None。"""
    side = board.side_to_move
    win = "1-0" if side == RED else "0-1"
    for m in board.legal_moves():
        board.push(m)
        res = board.result()
        board.pop()
        if res == win:
            return m
    return None


def _record_sample(
    slot: _GameSlot,
    board: Board,
    pol_idx: np.ndarray,
    pol_prob: np.ndarray,
    history_steps: int,
) -> None:
    """记录一步训练样本（state uint8 + 稀疏 π + 走子方 + 红方视角材料分）。"""
    state = encode_board(board, history_steps).astype(np.uint8, copy=False)
    slot.sample_states.append(state)
    slot.sample_pol_idx.append(pol_idx)
    slot.sample_pol_prob.append(pol_prob)
    slot.sample_sides.append(board.side_to_move)
    slot.sample_mat_red.append(material_score(board.fen()))


class _WorkerEngine:
    """单进程内并发多盘对局的批量自博弈引擎。"""

    def __init__(self, cfg: Config, iteration: int, net_eval, base_rng: np.random.Generator, proc_id: int) -> None:
        self.cfg = cfg
        self.iteration = int(iteration)
        self.net_eval = net_eval
        self.rng = base_rng
        self.proc_id = int(proc_id)

        self.mcts_cfg = cfg.mcts
        self.history_steps = int(cfg.model.history_steps)
        self.num_sims = int(cfg.mcts.num_sims)
        self.batch_cap = max(1, int(cfg.mcts.batch_size))
        self.target = self.num_sims

        # 认输仅在 resign_enabled 且迭代 ≥ resign_start_iter 后启用（DESIGN §10.3）。
        self.resign_active = bool(cfg.selfplay.resign_enabled) and (
            self.iteration >= int(cfg.selfplay.resign_start_iter)
        )
        self.resign_threshold = float(cfg.selfplay.resign_threshold)
        self.resign_consecutive = int(cfg.selfplay.resign_consecutive)

    # -------------------------------------------------- slot 生命周期
    def new_slot(self, game_id: int) -> _GameSlot:
        child = np.random.default_rng(self.rng.integers(0, 2**63 - 1))
        slot = _GameSlot(game_id, self.cfg, self.mcts_cfg, child)
        self._advance_slot(slot)
        return slot

    def _advance_slot(self, slot: _GameSlot) -> None:
        """处理非搜索状态（new_move），把 slot 推进到 searching 或 finished。"""
        while True:
            if slot.state == "finished":
                return
            board = slot.board
            res = board.result()
            if res is not None:
                slot.result = res
                slot.termination = board.termination()
                slot.state = "finished"
                return
            # 一步杀捷径：直接走将死着，π 记 one-hot，不进搜索。
            if bool(self.cfg.selfplay.mate_shortcut):
                m = _find_mate_move(board)
                if m is not None:
                    self._play_mate(slot, m)
                    continue  # 回到 new_move（大概率下一轮即终局）
            # 需要搜索
            slot._last_visits = -1
            slot.state = "searching"
            return

    @staticmethod
    def _sparse_from_counts(counts: np.ndarray, side: int) -> Tuple[np.ndarray, np.ndarray]:
        """visits 归一 + move_to_policy_index 转当前方视角 → 稀疏 π (idx int16, prob f32)。"""
        nz = np.nonzero(counts)[0]
        probs = counts[nz].astype(np.float64)
        probs /= probs.sum()
        idx = np.fromiter(
            (move_to_policy_index(action_to_move(int(a)), side) for a in nz),
            dtype=np.int16,
            count=len(nz),
        )
        return idx, probs.astype(np.float32)

    def _push_move(self, slot: _GameSlot, move: Tuple[int, int]) -> None:
        """落子并同步树复用、步数、record 着法列表（不记样本）。"""
        slot.board.push(move)
        slot.tree.advance(move)
        slot.ply += 1
        slot.moves.append(move_to_uci(move))

    def _play_mate(self, slot: _GameSlot, move: Tuple[int, int]) -> None:
        """一步杀捷径：π one-hot 记样本并落子。"""
        board = slot.board
        side = board.side_to_move
        idx = np.array([move_to_policy_index(move, side)], dtype=np.int16)
        prob = np.array([1.0], dtype=np.float32)
        _record_sample(slot, board, idx, prob, self.history_steps)
        self._push_move(slot, move)

    def _decide(self, slot: _GameSlot) -> None:
        """搜索完成后：按温度选着、记样本、认输判定、落子。"""
        board = slot.board
        counts = slot.tree.visit_counts()
        root_value = slot.tree.root_value()
        side = board.side_to_move

        # 温度：开局多样化 > 温度采样步 > argmax。
        if slot.ply < int(self.cfg.selfplay.opening_random_plies):
            tau = float(self.cfg.selfplay.opening_temperature)
        elif slot.ply < int(self.mcts_cfg.temp_moves):
            tau = float(self.mcts_cfg.temperature)
        else:
            tau = 0.0

        move = sample_action_from_counts(counts, tau, slot.rng)
        if move is None:
            # 极端兜底：搜索无访问（不应发生），回退到首个合法着法（无样本）。
            legal = board.legal_moves()
            if not legal:
                slot.result = board.result()
                slot.termination = board.termination()
                slot.state = "finished"
                return
            self._push_move(slot, legal[0])
            self._advance_slot(slot)
            return

        # 记录当前局面训练样本（π = 归一化访问计数，当前方视角）。
        idx, prob = self._sparse_from_counts(counts, side)
        _record_sample(slot, board, idx, prob, self.history_steps)

        # 认输判定（在落子前）。校准镜像先更新。
        self._update_calibration(slot, side, root_value)
        if self.resign_active and slot.resign.update(side, root_value):
            # 当前走子方认输 → 判负，不再落子（当前局面样本已入库，携最终 z）。
            winner = -side
            slot.result = "1-0" if winner == RED else "0-1"
            slot.termination = "resign"
            slot.state = "finished"
            return

        self._push_move(slot, move)
        self._advance_slot(slot)

    def _update_calibration(self, slot: _GameSlot, side: int, root_value: float) -> None:
        """镜像跟踪校准局的"本应认输方"，供主进程统计误判率。"""
        if not self.resign_active or not slot.resign.is_calibration_game:
            return
        if root_value < self.resign_threshold:
            if side == RED:
                slot._calib_consec_red += 1
            else:
                slot._calib_consec_black += 1
        else:
            if side == RED:
                slot._calib_consec_red = 0
            else:
                slot._calib_consec_black = 0
        if slot._calib_resign_side is None:
            if slot._calib_consec_red >= self.resign_consecutive:
                slot._calib_resign_side = RED
            elif slot._calib_consec_black >= self.resign_consecutive:
                slot._calib_resign_side = BLACK

    # -------------------------------------------------- 批量搜索一轮
    def search_round(self, searching: List[_GameSlot]) -> None:
        """对所有 searching slot 各选一批叶子 → 合并前向 → 逐盘回填。"""
        batch_states: List[np.ndarray] = []
        layout: List[Tuple[_GameSlot, list, int]] = []

        for slot in searching:
            remaining = self.target - slot.tree.total_visits
            if remaining <= 0:
                slot.state = "decide"
                continue
            v0 = slot.tree.total_visits
            states, tokens = slot.tree.select_leaves(min(self.batch_cap, remaining))
            if tokens:
                batch_states.append(states)
                layout.append((slot, tokens, states.shape[0]))
            else:
                # 无待评估叶子（全终局/撞车）且无进展 → 结束本步搜索。
                if slot.tree.total_visits == v0:
                    slot.state = "decide"

        if not batch_states:
            return

        big = np.concatenate(batch_states, axis=0)
        probs, values = self.net_eval(big)
        off = 0
        for slot, tokens, n in layout:
            slot.tree.apply_results(tokens, probs[off:off + n], values[off:off + n])
            slot.nn_evals += n
            off += n
            if slot.tree.total_visits >= self.target:
                slot.state = "decide"

    # -------------------------------------------------- 组装 payload
    def build_payload(self, slot: _GameSlot) -> dict:
        """把结束的对局组装成 queue 消息（样本 + record + 统计）。"""
        cfg = self.cfg
        result = slot.result
        termination = slot.termination or "unknown"

        winner = 0
        if result == "1-0":
            winner = RED
        elif result == "0-1":
            winner = BLACK

        # 逐步价值：z 含和棋惩罚 + 材料混合 λ，符号 = 该步走子方视角。
        lam = lambda_schedule(self.iteration, cfg.selfplay)
        values = np.empty(len(slot.sample_sides), dtype=np.float32)
        for i, (side, mat_red) in enumerate(zip(slot.sample_sides, slot.sample_mat_red)):
            if winner == 0:
                z = apply_draw_penalty(0.0, cfg.selfplay)
            else:
                z = 1.0 if winner == side else -1.0
            mat_side = mat_red * side  # 红分转当前走子方视角
            mat_tanh = float(np.tanh(mat_side / MATERIAL_SCALE))
            values[i] = float(blend_value(z, mat_tanh, lam))

        if slot.sample_states:
            states = np.stack(slot.sample_states).astype(np.uint8, copy=False)
        else:
            C = 14 * self.history_steps + 3
            states = np.zeros((0, C, 10, 9), dtype=np.uint8)

        winner_name = "red" if winner == RED else "black" if winner == BLACK else "draw"

        # 认输误判：校准局中"本应认输方"最终未告负（和/胜）即为一次 false-positive。
        would_resign = slot._calib_resign_side is not None
        fp = bool(would_resign and winner != -slot._calib_resign_side)

        record = {
            "format": _RECORD_FORMAT,
            "start_fen": START_FEN,
            "moves": list(slot.moves),
            "result": result if result is not None else "1/2-1/2",
            "termination": termination,
            "plies": len(slot.moves),
            "red": f"selfplay-iter{self.iteration}",
            "black": f"selfplay-iter{self.iteration}",
            "meta": {
                "iteration": self.iteration,
                "sims": self.num_sims,
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
            },
        }

        stats = {
            "result": record["result"],
            "winner": winner_name,
            "plies": len(slot.moves),
            "termination": termination,
            "resigned": bool(slot.resign.resigned),
            "is_calibration": bool(slot.resign.is_calibration_game),
            "would_resign": bool(would_resign),
            "fp": fp,
            "nn_evals": int(slot.nn_evals),
        }

        return {
            "type": "game",
            "proc_id": self.proc_id,
            "states": states,
            "pol_idx": list(slot.sample_pol_idx),
            "pol_prob": list(slot.sample_pol_prob),
            "values": values,
            "record": record,
            "stats": stats,
        }


# ---------------------------------------------------------------------------
# 顶层 worker（spawn 目标函数）。
# ---------------------------------------------------------------------------
def selfplay_worker(
    proc_id: int,
    device: str,
    ckpt_path: str,
    cfg: Config,
    iteration: int,
    game_queue,
    stop_event,
    seed: int,
) -> None:
    """自我博弈 worker 进程入口（Windows/Linux spawn 兼容，顶层函数）。

    Parameters
    ----------
    proc_id     : worker 序号（用于 GPU round-robin 与配额计算）
    device      : "cuda" / "cpu"（cuda 时按 proc_id % num_gpus 绑卡）
    ckpt_path   : best 模型 checkpoint 路径（本 worker 加载并 eval）
    cfg         : 全局 Config
    iteration   : 当前迭代编号
    game_queue  : 多进程队列，逐盘上报 {"type":"game",...}，收尾上报 {"type":"done"}
    stop_event  : 多进程 Event，置位后尽快收尾
    seed        : 本 worker 随机种子
    """
    import torch

    torch.set_num_threads(1)  # 多进程避免线程超订
    try:
        # 设备解析：GPU round-robin cuda:{proc_id % num_gpus}，无 GPU 降级 cpu。
        if str(device).startswith("cuda") and torch.cuda.is_available() and cfg.train.num_gpus > 0:
            dev = f"cuda:{proc_id % max(1, cfg.train.num_gpus)}"
        else:
            dev = "cpu"

        torch.manual_seed(int(seed) & 0x7FFFFFFF)
        model, _ = load_checkpoint(ckpt_path, device=dev)
        model.eval()
        net_eval = build_eval_fn(model, dev)

        base_rng = np.random.default_rng(int(seed) & (2**63 - 1))
        engine = _WorkerEngine(cfg, iteration, net_eval, base_rng, proc_id)

        quota = worker_game_quota(cfg, proc_id)
        concurrency = min(int(cfg.selfplay.games_per_worker), quota) if quota > 0 else 0

        started = 0
        live: List[_GameSlot] = []
        for _ in range(concurrency):
            gid = proc_id * 1_000_000 + started
            live.append(engine.new_slot(gid))
            started += 1

        while live:
            if stop_event is not None and stop_event.is_set():
                break

            searching = [s for s in live if s.state == "searching"]
            if searching:
                engine.search_round(searching)

            # 决策阶段（搜索完成的 slot 选着/落子/终局）。
            for slot in live:
                if slot.state == "decide":
                    engine._decide(slot)

            # 结束的对局：上报并按配额补充新局。
            next_live: List[_GameSlot] = []
            for slot in live:
                if slot.state == "finished":
                    if not slot.emitted:
                        game_queue.put(engine.build_payload(slot))
                        slot.emitted = True
                    if started < quota and not (stop_event is not None and stop_event.is_set()):
                        gid = proc_id * 1_000_000 + started
                        next_live.append(engine.new_slot(gid))
                        started += 1
                    # 否则丢弃（该 slot 结束）
                else:
                    next_live.append(slot)
            live = next_live
    finally:
        try:
            game_queue.put({"type": "done", "proc_id": proc_id})
        except Exception:
            pass
