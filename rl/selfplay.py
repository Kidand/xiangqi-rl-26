# -*- coding: utf-8 -*-
"""自我博弈 worker（纯 CPU 进程，推理服务架构）——DESIGN.md §9-§11。

核心职责
--------
- ``selfplay_worker``：顶层函数（Windows/Linux spawn 兼容），**纯 CPU**（不加载模型、
  不碰 CUDA context，``torch.set_num_threads(1)``），在单进程内并发 ``games_per_worker``
  盘对局，用 ``SearchTree.select_leaves`` 跨盘收集叶子 → 转 uint8 + 取策略索引 →
  ``EvalClient.evaluate`` 向所属 EvalServer 请求批量前向 → 逐盘 ``apply_results_sparse``。
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

import dataclasses
import datetime
import math
import os
from pathlib import Path
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
    sims_schedule,
)
from rl.inference_server import EvalClient, WorkerStop
from rl.mcts import SearchTree

__all__ = [
    "selfplay_worker",
    "build_eval_fn",
    "num_selfplay_workers",
    "num_eval_servers",
    "worker_game_quota",
    "sample_action_from_counts",
    "sharpen_counts",
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
# worker / server 数量与配额（推理服务架构，DESIGN §9）。
# 瓶颈在 CPU 端 Python MCTS，故 worker 数量按 CPU 核数扩展（而非 GPU 数）。
# ---------------------------------------------------------------------------
def num_eval_servers(cfg: Config) -> int:
    """EvalServer 进程数：cuda 时每 GPU 一个，cpu 时共 1 个。"""
    if str(cfg.train.device).startswith("cuda") and cfg.train.num_gpus > 0:
        return max(1, int(cfg.train.num_gpus))
    return 1


def _read_cgroup_cpu_quota() -> Optional[int]:
    """读 cgroup CPU 配额（向上取整到核数）；无限额或不可读返回 None。

    K8s pod 的 limits.cpu 走 CFS 带宽限制：os.cpu_count() 与 sched_getaffinity
    都看不到它（返回宿主核数），只有 cgroup 配额文件是权威。兼容 v2 与 v1。
    """
    try:
        p = Path("/sys/fs/cgroup/cpu.max")                    # cgroup v2
        if p.exists():
            parts = p.read_text().split()
            if len(parts) >= 2 and parts[0] != "max":
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return max(1, math.ceil(quota / period))
            return None
        q = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")       # cgroup v1
        pd = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q.exists() and pd.exists():
            quota, period = int(q.read_text().strip()), int(pd.read_text().strip())
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def effective_cpu_count() -> int:
    """容器感知的有效核数：cpu_count / 调度亲和性 / cgroup CFS 配额三信号取最小。

    背景：曾在 192 核宿主 + 小 CFS 配额的 K8s pod 上因 os.cpu_count() 失真而
    spawn 180 个 worker（严重超订：CFS 节流 + 上下文切换烧掉配额本身）。
    裸机/本地（Windows/macOS 无 sched_getaffinity、无 cgroup 文件）自动回退 cpu_count。
    """
    signals = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            signals.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    quota = _read_cgroup_cpu_quota()
    if quota is not None:
        signals.append(quota)
    return max(1, min(signals))


def cpu_budget_summary() -> str:
    """人读的核数信号一览（写进日志，一眼看穿容器限额环境）。"""
    aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return (
        f"cpu_count={os.cpu_count()} affinity={aff} "
        f"cgroup_quota={_read_cgroup_cpu_quota()} -> effective={effective_cpu_count()}"
    )


def num_selfplay_workers(cfg: Config) -> int:
    """本次自博弈的纯 CPU worker 进程总数。

    - ``num_workers > 0``：直接使用该值（最高优先级）。
    - ``num_workers == 0``：自动，基于容器感知的有效核数 eff（effective_cpu_count）：
        - 核充裕（eff ≥ 4×(num_gpus+4)）：``max(8, eff - num_gpus - 4)``——裸机语义
          不变，为 EvalServer 与主进程留核；
        - 核受限（小配额容器）：``clamp(ceil(eff × 1.5), 8, 64)``——固定预留会吃掉
          大半预算；1.5× 轻度超订恰好覆盖 worker 同步等待 GPU 响应的空窗。
      cpu 设备 ``num_gpus`` 按 0 算。
    """
    n = int(cfg.selfplay.num_workers)
    if n > 0:
        return n
    ngpu = int(cfg.train.num_gpus) if str(cfg.train.device).startswith("cuda") else 0
    eff = effective_cpu_count()
    if eff >= 4 * (ngpu + 4):
        return max(8, eff - ngpu - 4)
    return max(8, min(64, math.ceil(eff * 1.5)))


def worker_game_quota(total_games: int, worker_id: int, n_workers: int) -> int:
    """worker_id 号 worker 需产出的对局数（把 total_games 均摊到 n_workers 个 worker）。"""
    n = max(1, int(n_workers))
    base, rem = divmod(int(total_games), n)
    return base + (1 if int(worker_id) < rem else 0)


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
# π 训练目标锐化（DESIGN §8）：只作用于训练样本 π 的构造，不影响落子采样
# （sample_action_from_counts）与 visit_counts 语义。
# ---------------------------------------------------------------------------
def sharpen_counts(counts: np.ndarray, temp: float, prune_frac: float) -> np.ndarray:
    """把访问计数锐化并归一，返回与 counts 同形的概率向量（float64）。

    先剪噪声边、再幂锐化、最后归一（DESIGN §8）：
      - prune_frac>0：剔除 visits < prune_frac*max 的边（削 Dirichlet 噪声边）；
      - temp!=1.0：p ∝ c^(1/T)（float64 中间量防溢出/下溢，T<1 锐化）。
    默认 (temp=1.0, prune_frac=0.0) 逐 bit 等价原始「counts[nz] 归一」：仅在 nonzero
    子集上按升序求和/除法，与旧路径同序同值。全零输入返回全零向量。
    """
    out = np.zeros(counts.shape, dtype=np.float64)
    nz = np.nonzero(counts)[0]
    if nz.size == 0:
        return out
    c = counts[nz].astype(np.float64)
    if prune_frac > 0.0:
        c[c < prune_frac * c.max()] = 0.0
    if temp != 1.0:
        c = np.power(c, 1.0 / temp)  # 0**正=0，被剪的边保持 0
    s = c.sum()
    if not np.isfinite(s) or s <= 0.0:
        # 极端兜底（全被剪/数值退化，不应发生）：退回未锐化归一。
        c = counts[nz].astype(np.float64)
        s = c.sum()
    out[nz] = c / s
    return out


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
        "sample_root_values",
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
        # 每步搜索根价值（当前走子方视角，与 z 同视角）；用于 z 混根价值（β>0）。
        self.sample_root_values: List[float] = []
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
    root_value: float,
) -> None:
    """记录一步训练样本（state uint8 + 稀疏 π + 走子方 + 红方视角材料分 + 根价值）。

    四个逐步列表（sides/mat_red/root_values 及稀疏 π）在此单点同步 append，保证等长
    （build_payload 断言依赖此不变式；错位会静默污染训练数据）。root_value 为当前
    走子方视角，与 z 同视角。
    """
    state = encode_board(board, history_steps).astype(np.uint8, copy=False)
    slot.sample_states.append(state)
    slot.sample_pol_idx.append(pol_idx)
    slot.sample_pol_prob.append(pol_prob)
    slot.sample_sides.append(board.side_to_move)
    # 直接传 Board 对象走 material_score 快路径（遍历 squares，免 fen() 字符串往返）。
    slot.sample_mat_red.append(material_score(board))
    slot.sample_root_values.append(float(root_value))


class _WorkerEngine:
    """单进程内并发多盘对局的批量自博弈引擎。"""

    def __init__(self, cfg: Config, iteration: int, client, base_rng: np.random.Generator, proc_id: int) -> None:
        self.cfg = cfg
        self.iteration = int(iteration)
        self.client = client          # EvalClient：向所属 EvalServer 请求批量前向
        self.rng = base_rng
        self.proc_id = int(proc_id)

        # 注入树内和棋惩罚（单一来源 selfplay.draw_penalty，避免与 arena 配置漂移）。
        # 树内终局和棋回传该值，与训练 z 目标（draw_penalty）一致。
        self.mcts_cfg = dataclasses.replace(
            cfg.mcts, draw_value=float(cfg.selfplay.draw_penalty)
        )
        self.history_steps = int(cfg.model.history_steps)
        # num_sims 按迭代调度（不 mutate cfg——cfg 引用会写进 checkpoint/meta）。
        self.num_sims = sims_schedule(self.iteration, cfg.mcts)
        self.batch_cap = max(1, int(cfg.mcts.batch_size))
        self.target = self.num_sims
        # π 训练目标锐化参数（DESIGN §8）；默认 (1.0, 0.0) = 不锐化。
        self.policy_temp = float(cfg.mcts.policy_target_temp)
        self.policy_prune_frac = float(cfg.mcts.policy_target_prune_frac)

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
            res, term = board.result_and_termination()
            if res is not None:
                slot.result = res
                slot.termination = term
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

    def _sparse_from_counts(self, counts: np.ndarray, side: int) -> Tuple[np.ndarray, np.ndarray]:
        """visits 锐化归一 + move_to_policy_index 转当前方视角 → 稀疏 π (idx int16, prob f32)。

        锐化（temp/prune_frac）后被剪的边概率为 0，不进稀疏输出。默认参数逐 bit 等价原行为。
        """
        probs_full = sharpen_counts(counts, self.policy_temp, self.policy_prune_frac)
        nz = np.nonzero(probs_full)[0]
        probs = probs_full[nz]
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
        """一步杀捷径：π one-hot 记样本并落子（无搜索，根价值取 +1.0：一步杀必胜）。"""
        board = slot.board
        side = board.side_to_move
        idx = np.array([move_to_policy_index(move, side)], dtype=np.int16)
        prob = np.array([1.0], dtype=np.float32)
        _record_sample(slot, board, idx, prob, self.history_steps, 1.0)
        self._push_move(slot, move)

    def _decide(self, slot: _GameSlot) -> None:
        """搜索完成后：按温度选着、记样本、认输判定、落子。"""
        board = slot.board
        counts = slot.tree.visit_counts()
        root_value = slot.tree.root_value()
        side = board.side_to_move

        # 温度：开局多样化 > 温度采样步 > 温度期后（temp_final>0 采样，否则 argmax）。
        if slot.ply < int(self.cfg.selfplay.opening_random_plies):
            tau = float(self.cfg.selfplay.opening_temperature)
        elif slot.ply < int(self.mcts_cfg.temp_moves):
            tau = float(self.mcts_cfg.temperature)
        else:
            # temp_final > 0：按 N^(1/temp_final) 归一采样（打破确定性循环、抑制重复判和）；
            # 否则 argmax（旧行为）。复用 sample_action_from_counts 同一温度采样路径。
            temp_final = float(self.mcts_cfg.temp_final)
            tau = temp_final if temp_final > 0.0 else 0.0

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
        _record_sample(slot, board, idx, prob, self.history_steps, root_value)

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
        """对所有 searching slot 各选一批叶子 → 合并成一个评估请求 → 逐盘回填。

        叶子 states（encode 输出 0/1 float32）直接 astype uint8；tokens 经
        tree.policy_indices 取当前方视角策略索引 → EvalClient.evaluate（稀疏返回）→
        逐 slot apply_results_sparse（与稠密 apply_results 数值等价）。
        """
        batch_states: List[np.ndarray] = []
        batch_idx: List[np.ndarray] = []      # 扁平：跨 slot 所有叶子的策略索引
        layout: List[Tuple[_GameSlot, list, int]] = []

        for slot in searching:
            remaining = self.target - slot.tree.total_visits
            if remaining <= 0:
                slot.state = "decide"
                continue
            v0 = slot.tree.total_visits
            states, tokens = slot.tree.select_leaves(min(self.batch_cap, remaining))
            if tokens:
                # encode_board 输出 0/1 float32，直接 astype uint8（数值无损）。
                batch_states.append(states.astype(np.uint8, copy=False))
                batch_idx.extend(slot.tree.policy_indices(tokens))
                layout.append((slot, tokens, states.shape[0]))
            else:
                # 无待评估叶子（全终局/撞车）且无进展 → 结束本步搜索。
                if slot.tree.total_visits == v0:
                    slot.state = "decide"

        if not batch_states:
            return

        big = np.concatenate(batch_states, axis=0)
        probs_list, values = self.client.evaluate(big, batch_idx)
        off = 0
        for slot, tokens, n in layout:
            slot.tree.apply_results_sparse(
                tokens, probs_list[off:off + n], values[off:off + n]
            )
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
        # 三个逐步列表必须等长（错位会静默污染训练数据，与 z 视角挂钩）。
        assert (
            len(slot.sample_root_values) == len(slot.sample_sides) == len(slot.sample_mat_red)
        ), "逐步列表长度错位：root_values/sides/mat_red 不等长"
        lam = lambda_schedule(self.iteration, cfg.selfplay)
        beta = float(cfg.selfplay.root_value_weight)  # z 混根价值权重，0=关（原行为）
        values = np.empty(len(slot.sample_sides), dtype=np.float32)
        for i, (side, mat_red) in enumerate(zip(slot.sample_sides, slot.sample_mat_red)):
            if winner == 0:
                z = apply_draw_penalty(0.0, cfg.selfplay)
            else:
                z = 1.0 if winner == side else -1.0
            mat_side = mat_red * side  # 红分转当前走子方视角
            mat_tanh = float(np.tanh(mat_side / MATERIAL_SCALE))
            blended = float(blend_value(z, mat_tanh, lam))
            if beta > 0.0:
                # root_value 与 z 同视角（当前走子方），直接凸组合无需翻符号。
                blended = (1.0 - beta) * blended + beta * float(slot.sample_root_values[i])
            values[i] = blended

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
    worker_id: int,
    cfg: Config,
    iteration: int,
    shm: dict,
    req_queue,
    resp_queue,
    game_queue,
    stop_event,
    seed: int,
    n_workers: int,
) -> None:
    """自我博弈 worker 进程入口（纯 CPU，Windows/Linux spawn 兼容，顶层函数）。

    不加载模型、不碰 CUDA context；评估请求经 ``EvalClient``（共享内存 + 队列）转发给
    所属 EvalServer。

    Parameters
    ----------
    worker_id   : worker 序号（用于配额计算与共享内存/队列定位）
    cfg         : 全局 Config
    iteration   : 当前迭代编号
    shm         : 本 worker 的共享张量字典（主进程创建并 share_memory_ 后传入）
    req_queue   : 所属 EvalServer 的请求队列
    resp_queue  : 本 worker 的响应队列
    game_queue  : 多进程队列，逐盘上报 {"type":"game",...}，收尾上报 {"type":"done"}
    stop_event  : 多进程 Event，置位后尽快收尾
    seed        : 本 worker 随机种子
    n_workers   : worker 总数（用于配额均摊）
    """
    import torch

    torch.set_num_threads(1)  # 纯 CPU，多进程避免线程超订
    try:
        client = EvalClient(worker_id, shm, req_queue, resp_queue, stop_event)

        base_rng = np.random.default_rng(int(seed) & (2**63 - 1))
        engine = _WorkerEngine(cfg, iteration, client, base_rng, worker_id)

        quota = worker_game_quota(int(cfg.selfplay.games_per_iteration), worker_id, n_workers)
        concurrency = min(int(cfg.selfplay.games_per_worker), quota) if quota > 0 else 0

        started = 0
        live: List[_GameSlot] = []
        for _ in range(concurrency):
            gid = worker_id * 1_000_000 + started
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
                        gid = worker_id * 1_000_000 + started
                        next_live.append(engine.new_slot(gid))
                        started += 1
                    # 否则丢弃（该 slot 结束）
                else:
                    next_live.append(slot)
            live = next_live
    except WorkerStop:
        # stop_event 置位或评估超时（server 疑似死亡）：尽快收尾。
        pass
    finally:
        try:
            game_queue.put({"type": "done", "proc_id": worker_id})
        except Exception:
            pass
