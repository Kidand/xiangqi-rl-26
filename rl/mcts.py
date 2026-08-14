# -*- coding: utf-8 -*-
"""批量化 PUCT MCTS（见 DESIGN.md §8）。

设计要点
--------
- 两阶段 API：``select_leaves`` 选出一批待评估叶子（施加 virtual loss），外部凑成
  大 batch 送网络前向后，再用 ``apply_results`` 回填先验并 backup。多棵树（多盘对局）
  可跨树 stack 成同一 batch，最大化 CPU/GPU 利用率。
- Q 存储为 **该节点走子方视角**；backup 逐层取负。
- virtual loss：select 下行时在所经边累加，apply / 终局回传时撤销；同一 batch 命中
  同一未展开节点靠 virtual loss 分流，若仍撞车则跳过该次模拟。
- 终局叶子（``board.result()`` 非 None）不进网络：直接用真实结果值 backup 并计 visits。
- 根节点 Dirichlet 噪声仅 ``add_noise=True`` 时混入先验；FPU = parent_Q - fpu_reduction。
- 节点用 ``__slots__``，children 按 ``legal_moves`` 顺序存 numpy 数组（P/N/W/vloss），
  不为每个节点分配 8100 维数组。
- ``visit_counts()`` 返回 **棋盘真实坐标** 下的 (8100,) 访问计数。
- Gumbel 训练搜索模式（DESIGN §8）：``cfg.algorithm == "gumbel"`` 且 ``add_noise=True``
  时，**根节点**改用 Gumbel-Top-m + Sequential Halving 调度（不混 Dirichlet），
  **非根节点仍走 PUCT**（工程简化，偏离论文的非根确定性规则）；训练 π 目标改由
  ``gumbel_policy_target()``（completed-Q）给出、落子由 ``gumbel_best_action()`` 给出。
  ``algorithm="puct"``（默认）时本模块行为与加入该模式前逐 bit 一致：所有 Gumbel 分支
  受 ``self._gumbel`` 单一开关保护，且**不从 self._rng 取任何随机数**（否则会平移后续
  Dirichlet 抽样序列）。

坐标 / 编码一律 import 自 ``xiangqi.constants`` 与 ``rl.encoding``，本模块不重复定义。
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from xiangqi.constants import (
    ACTION_SIZE,
    BLACK,
    COLS,
    RED,
    ROWS,
    move_to_action,
)
from rl.config import MCTSConfig
from rl.encoding import encode_board

__all__ = ["SearchTree", "search"]


# ---------------------------------------------------------------------------
# 单个搜索节点。children 相关数组按 legal_moves 顺序对齐，未展开时全部为 None。
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = (
        "expanded",
        "pending",
        "terminal",
        "terminal_value",
        "terminal_draw",  # 该终局是否和棋（决定回传是否翻符号）
        "net_value",     # 展开时网络给出的 value（**本节点走子方视角**），Gumbel v_mix 用
        "child_moves",   # list[(from_sq,to_sq)]，棋盘真实坐标，顺序 = legal_moves
        "child_nodes",   # list[_Node | None]，惰性创建
        "child_P",       # np.float32 (n,) 先验
        "child_N",       # np.int64  (n,) 访问计数
        "child_W",       # np.float64 (n,) 累计价值（本节点走子方视角）
        "child_vloss",   # np.int64  (n,) 当前挂起的 virtual loss 次数
    )

    def __init__(self) -> None:
        self.expanded = False
        self.pending = False
        self.terminal = False
        self.terminal_value = 0.0
        self.terminal_draw = False
        self.net_value = 0.0
        self.child_moves = None
        self.child_nodes = None
        self.child_P = None
        self.child_N = None
        self.child_W = None
        self.child_vloss = None


def _terminal_value(result: str, side_to_move: int) -> float:
    """终局结果串 -> 当前走子方视角价值（胜=+1 / 负=-1 / 和=0）。"""
    if result == "1/2-1/2":
        return 0.0
    winner = RED if result == "1-0" else BLACK
    return 1.0 if winner == side_to_move else -1.0


def _sh_schedule(m: int, num_sims: int) -> list:
    """Sequential Halving 调度表（DESIGN §8）。

    返回 [(m_phase, quota), ...]，长度 = phases = ceil(log2 m)（m=1 时 1）；
    quota = max(1, floor(num_sims / (phases * m_phase)))，即该 phase 内**每个存活候选**
    的模拟配额。下一 phase 的存活数 = ceil(m_phase/2)（至少 1）。
    例：m=8, N=24 -> [(8,1),(4,2),(2,4)]（恰好 24）；m=16, N=600 -> 586，余量由
    最终 phase 后的 round-robin 补齐。
    """
    m = max(1, int(m))
    num_sims = max(0, int(num_sims))
    phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
    out = []
    mp = m
    for _ in range(phases):
        out.append((mp, max(1, num_sims // (phases * mp))))
        mp = max(1, -(-mp // 2))
    return out


def _leaf_policy_indices(legal: list, side: int) -> np.ndarray:
    """把一个叶子的合法着法转换为**当前走子方视角**策略索引（int64，(n,)）。

    legal 拆成 from/to 两列：action = from*90 + to；BLACK 视角需 flip_action，即
    flip_sq(from)*90 + flip_sq(to) = (89-from)*90 + (89-to)。与
    ``rl.encoding.move_to_policy_index`` 逐项等价。稀疏评估协议（policy_indices /
    apply_results_sparse）与稠密路径（apply_results）复用同一计算，保证数值一致。
    """
    n = len(legal)
    lm = np.asarray(legal, dtype=np.int64).reshape(n, 2)
    frm = lm[:, 0]
    to = lm[:, 1]
    if side == BLACK:
        return (89 - frm) * 90 + (89 - to)
    return frm * 90 + to


class SearchTree:
    """单棵 MCTS 树（对应一盘对局），支持两阶段批量叶子评估。"""

    def __init__(
        self,
        board,
        cfg: MCTSConfig,
        add_noise: bool,
        history_steps: int = 2,
        rng: np.random.Generator | None = None,
        num_sims: int | None = None,
    ) -> None:
        # 复制一份棋盘作为持久工作局面；select 时 push/pop，advance 时永久 push。
        self._board = board.copy()
        self._cfg = cfg
        self._add_noise = bool(add_noise)
        self._history_steps = int(history_steps)
        self._rng = rng if rng is not None else np.random.default_rng()

        # 常用配置提前取出，避免热点路径反复属性查找。
        self._c_puct = float(cfg.c_puct)
        self._fpu = float(cfg.fpu_reduction)
        self._vl = float(cfg.virtual_loss)
        self._alpha = float(cfg.dirichlet_alpha)
        self._eps = float(cfg.dirichlet_eps)
        # 树内和棋终局回传值（不翻符号，双方同罚）。selfplay/arena 注入 draw_penalty。
        self._draw_value = float(cfg.draw_value)

        # ---- Gumbel 模式（DESIGN §8）：仅 add_noise=True 的训练路径生效；
        # arena/GUI（add_noise=False）恒走 PUCT。algorithm 默认 "puct" = 原行为。
        # 精确匹配 "gumbel"（与 rl/selfplay.py 的 self.gumbel 判定逐字一致）：
        # 两处判据必须完全同步，否则会出现"树跑 SH、selfplay 却按 visit 计数取目标"
        # 的静默错配（SH 的访问分布本身不是策略目标）。
        self._gumbel = str(getattr(cfg, "algorithm", "puct")) == "gumbel" and self._add_noise
        self._g_m = int(getattr(cfg, "gumbel_m", 16))
        self._g_c_visit = float(getattr(cfg, "gumbel_c_visit", 50.0))
        self._g_c_scale = float(getattr(cfg, "gumbel_c_scale", 1.0))
        # SH 总预算：外部驱动循环的 target（selfplay 的 sims_schedule 可能 != cfg.num_sims，
        # 故允许构造时覆盖）。
        self._g_num_sims = int(cfg.num_sims if num_sims is None else num_sims)
        self._g_reset_state()

        self._root = _Node()

    # ------------------------------------------------------------------ 属性
    @property
    def total_visits(self) -> int:
        """根已完成的模拟数（= 根各子边访问计数之和）。"""
        root = self._root
        if not root.expanded:
            return 0
        return int(root.child_N.sum())

    def root_value(self) -> float:
        """根 Q（当前走子方视角）。

        根为终局时返回其终局值：胜负 ±1、和棋返回 ``draw_value``（终局节点已把
        ``terminal_value`` 存为 draw_value）。根从未搜索但局面本身即终局时，惰性判定
        并返回对应终局值（同样区分和棋 draw_value）。
        """
        root = self._root
        if root.terminal:
            return float(root.terminal_value)
        if not root.expanded:
            res = self._board.result()
            if res is not None:
                if res == "1/2-1/2":
                    return self._draw_value
                return _terminal_value(res, self._board.side_to_move)
            return 0.0
        n = int(root.child_N.sum())
        if n <= 0:
            return 0.0
        return float(root.child_W.sum() / n)

    def visit_counts(self) -> np.ndarray:
        """(8100,) 根访问计数，棋盘真实坐标（非翻转视角）。"""
        counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        root = self._root
        if not root.expanded:
            return counts
        for i, move in enumerate(root.child_moves):
            counts[move_to_action(move)] = root.child_N[i]
        return counts

    def q_values(self) -> np.ndarray:
        """(8100,) 根各子边 Q（根走子方视角），棋盘真实坐标；未访问边为 NaN。

        child_W 挂在根上、按根走子方视角累计（backup 逐层取负后写入），
        故 W/N 即"根走子方走该着法的期望值"，与 root_value() 同一尺度。
        """
        q = np.full(ACTION_SIZE, np.nan, dtype=np.float32)
        root = self._root
        if not root.expanded:
            return q
        for i, move in enumerate(root.child_moves):
            n = float(root.child_N[i])
            if n > 0:
                q[move_to_action(move)] = float(root.child_W[i]) / n
        return q

    # ------------------------------------------------------------- Gumbel
    # 以下方法仅在 self._gumbel 为真时被调用（algorithm="gumbel" 且 add_noise=True）。
    # 关键视角约定：root.child_W 已按**根走子方视角**累计（backup 逐层取负后写入），
    # 故 q̂(a) = W/N 直接就是根走子方视角，与 _select_child / q_values() 取法完全同向，
    # **不得再取负**。node.net_value 同为该节点自身走子方视角；树复用后该节点成为根，
    # 其视角即根视角（勿"修正"）。
    def _g_reset_state(self) -> None:
        self._g_ready = False       # 是否已对当前根惰性初始化
        self._g_logits = None       # (n,) log(掩码归一先验 + 1e-12)
        self._g_gum = None          # (n,) i.i.d. Gumbel(0,1)
        self._g_cand = None         # 当前 phase 存活候选（root 子边索引，按分数降序）
        self._g_sched = None        # [(m_phase, quota), ...]
        self._g_phase = 0           # == len(_g_sched) 表示进入补齐（overflow）阶段
        self._g_base = None         # 本 phase 起始时的 child_N 快照
        self._g_block = None        # 本次 select_leaves 内各子边撞车次数（仅排序用）

    def _g_init(self) -> None:
        """根先验可用后的惰性初始化：采 Gumbel 噪声、取 top-m 候选、算 SH 调度表。"""
        root = self._root
        n = len(root.child_moves)
        self._g_logits = np.log(root.child_P.astype(np.float64) + 1e-12)
        self._g_gum = self._rng.gumbel(0.0, 1.0, size=n)
        m = max(1, min(self._g_m, n))
        # 候选按 g+logits 降序保留（预算不足以覆盖全部候选时，先访问分数高的）。
        order = np.argsort(-(self._g_gum + self._g_logits), kind="stable")
        self._g_cand = order[:m].copy()
        # 树复用时根上已有访问，SH 预算按"剩余"算（与驱动循环 target - total_visits 一致）。
        budget = max(1, self._g_num_sims - int(root.child_N.sum()))
        self._g_sched = _sh_schedule(m, budget)
        self._g_phase = 0
        self._g_base = root.child_N.copy()
        self._g_block = np.zeros(n, dtype=np.int64)
        self._g_ready = True

    def _g_v_mix(self) -> float:
        """v_mix = (v_net(root) + ΣN·q̄_π) / (1 + ΣN)，q̄_π = Σ_{N>0}π(a)Q(a) / Σ_{N>0}π(a)。

        π = 根先验（gumbel 模式下不含 Dirichlet），Q 与 v_net 均为根走子方视角。
        无任何已访问子边时退化为 v_net(root)。
        """
        root = self._root
        N = root.child_N
        sn = float(N.sum())
        v_net = float(root.net_value)
        if sn <= 0:
            return v_net
        vis = N > 0
        P = root.child_P.astype(np.float64)[vis]
        q = root.child_W[vis] / N[vis]
        pw = float(P.sum())
        qbar = float((P * q).sum() / pw) if pw > 1e-12 else 0.0
        return (v_net + sn * qbar) / (1.0 + sn)

    def _g_qhat(self) -> np.ndarray:
        """(n,) 各子边 q̂（根走子方视角）：已访问取 W/N，未访问用 v_mix 补全。"""
        root = self._root
        N = root.child_N
        q = np.full(N.shape, self._g_v_mix(), dtype=np.float64)
        vis = N > 0
        if vis.any():
            q[vis] = root.child_W[vis] / N[vis]
        return q

    def _g_sigma(self, qhat: np.ndarray) -> np.ndarray:
        """σ(q̂) = (c_visit + max_b N(b)) · c_scale · q̂。"""
        max_n = float(self._root.child_N.max()) if len(self._root.child_N) else 0.0
        return (self._g_c_visit + max_n) * self._g_c_scale * qhat

    def _g_scores(self) -> np.ndarray:
        """(n,) g + logits + σ(q̂)：候选排序 / 落子选择用（含 Gumbel 噪声）。"""
        return self._g_gum + self._g_logits + self._g_sigma(self._g_qhat())

    def _g_advance_phase(self) -> None:
        """当前 phase 所有存活候选均满配额时推进：折半保留前 ceil(m_phase/2)。

        末 phase 结束后不再折半（无"下一 phase"），直接进入补齐阶段，对该 phase 的
        存活候选 round-robin，直到外部驱动把预算用满。
        """
        root = self._root
        while self._g_phase < len(self._g_sched):
            cand = self._g_cand
            quota = self._g_sched[self._g_phase][1]
            # "实际访问 + 在途 virtual"（撞车会撤销 vloss，故自动重试）
            used = (root.child_N[cand] - self._g_base[cand]) + root.child_vloss[cand]
            if (used < quota).any():
                return
            self._g_phase += 1
            if self._g_phase < len(self._g_sched):
                keep = self._g_sched[self._g_phase][0]
                scores = self._g_scores()[cand]
                order = np.argsort(-scores, kind="stable")
                self._g_cand = cand[order[:max(1, keep)]].copy()
            self._g_base = root.child_N.copy()

    def _g_select_root(self) -> int:
        """根级 SH 选边（替代根 PUCT）：在当前 phase 内对未满配额的候选 round-robin。"""
        root = self._root
        self._g_advance_phase()
        cand = self._g_cand
        block = self._g_block[cand]
        if self._g_phase < len(self._g_sched):
            quota = self._g_sched[self._g_phase][1]
            used = (root.child_N[cand] - self._g_base[cand]) + root.child_vloss[cand]
            avail = np.nonzero(used < quota)[0]
            if avail.size:
                # 排序键：先按本次调用内的撞车次数（撞过的排后面，避免死磕同一候选），
                # 再按已用配额 —— 即 round-robin。
                j = avail[int(np.lexsort((used[avail], block[avail]))[0])]
                return int(cand[j])
        # 补齐阶段（或异常兜底）：对存活候选按有效访问数 round-robin。
        eff = root.child_N[cand] + root.child_vloss[cand]
        return int(cand[int(np.lexsort((eff, block))[0])])

    # ------------------------------------------------------- Gumbel 对外 API
    def gumbel_policy_target(self) -> np.ndarray:
        """(8100,) completed-Q 训练目标 π'(a) ∝ exp(logits(a) + σ(q̂(a)))。

        对**全部合法着法**归一（未访问着法 q̂ = v_mix 补全），棋盘真实坐标，和为 1。
        搜索未完成时也可调（用当前统计）；根未展开/终局时返回全零（无统计可报）。
        """
        out = np.zeros(ACTION_SIZE, dtype=np.float64)
        root = self._root
        if not root.expanded or root.terminal or not len(root.child_moves):
            return out
        logits = (
            self._g_logits
            if self._g_ready
            else np.log(root.child_P.astype(np.float64) + 1e-12)
        )
        x = logits + self._g_sigma(self._g_qhat())
        x = x - x.max()
        e = np.exp(x)
        e /= e.sum()
        for i, move in enumerate(root.child_moves):
            out[move_to_action(move)] = e[i]
        return out

    def gumbel_best_action(self) -> int:
        """最终存活候选中 argmax g+logits+σ(q̂) 对应的 action（棋盘真实坐标）。

        根未展开/终局时返回 -1；Gumbel 状态尚未初始化时退化为按 logits+σ(q̂) 取全部
        合法着法的 argmax（无噪声）。
        """
        root = self._root
        if not root.expanded or root.terminal or not len(root.child_moves):
            return -1
        if not self._g_ready:
            logits = np.log(root.child_P.astype(np.float64) + 1e-12)
            i = int(np.argmax(logits + self._g_sigma(self._g_qhat())))
            return int(move_to_action(root.child_moves[i]))
        cand = self._g_cand
        scores = self._g_scores()[cand]
        i = int(cand[int(np.argmax(scores))])
        return int(move_to_action(root.child_moves[i]))

    # ---------------------------------------------------------------- 选路
    def _select_child(self, node: _Node) -> int:
        """PUCT 选边：返回子节点索引。"""
        P = node.child_P
        N = node.child_N
        W = node.child_W
        VL = node.child_vloss

        ne = N + VL                          # 有效访问（含 virtual loss）
        # 父节点访问数 = 自身展开 1 次 + 所有子边访问（含挂起的 vloss）
        parent_visits = 1 + int(ne.sum())
        sqrt_parent = math.sqrt(parent_visits)

        real_n = int(N.sum())
        parent_q = (W.sum() / real_n) if real_n > 0 else 0.0

        # 未访问子节点 Q = parent_Q - fpu_reduction（First Play Urgency）
        q = np.full(P.shape, parent_q - self._fpu, dtype=np.float64)
        visited = ne > 0
        if visited.any():
            q[visited] = (W[visited] - self._vl * VL[visited]) / ne[visited]

        u = self._c_puct * P * sqrt_parent / (1.0 + ne)
        return int(np.argmax(q + u))

    def _pop_n(self, n: int) -> None:
        board = self._board
        for _ in range(n):
            board.pop()

    def _backup(self, path: list, value: float) -> None:
        """沿 path 逐层回传并撤销 virtual loss。

        value 为叶子（path 末端子节点）走子方视角。末边所属节点视角与叶子相反，
        故从末边起符号 -1、逐层取负。
        """
        sign = -1.0
        for node, i in reversed(path):
            node.child_N[i] += 1
            node.child_W[i] += sign * value
            node.child_vloss[i] -= 1
            sign = -sign

    def _backup_draw(self, path: list, value: float) -> None:
        """和棋回传：沿 path 每层 child_W 加 ``value`` **不翻符号**（双方同罚），
        照常 child_N += 1 与撤销 virtual loss。

        和棋对双方同为一个结果，若像胜负那样逐层取负，父节点会把 -draw_value 误读为
        "对我有利"（当 draw_value < 0 时尤甚），故此处必须保持符号不变。
        """
        for node, i in reversed(path):
            node.child_N[i] += 1
            node.child_W[i] += value
            node.child_vloss[i] -= 1

    def _backup_terminal(self, path: list, node: _Node) -> None:
        """按终局节点的和棋标志选择回传方式（和棋不翻符号、胜负逐层取负）。"""
        if node.terminal_draw:
            self._backup_draw(path, node.terminal_value)
        else:
            self._backup(path, node.terminal_value)

    def _descend(self):
        """从根下行一次。

        返回：
          ("leaf", (state, token))  待网络评估的叶子（已施加 virtual loss）
          ("terminal", None)        终局叶子已直接 backup 并计 visits
          ("collision", i)          撞到本 batch 已挂起的未展开节点，本次模拟跳过；
                                    i 为该路径的根子边索引（仅 gumbel 模式记录，否则 None）
        """
        node = self._root
        board = self._board
        path: list = []

        while True:
            if node.terminal:
                # 复访已解出的终局节点：再计一次访问，值固定，按和棋标志选回传方式。
                self._backup_terminal(path, node)
                self._pop_n(len(path))
                return ("terminal", None)

            if not node.expanded:
                if node.pending:
                    # 同一 batch 撞到未展开且已挂起的节点：撤销本路 vloss，跳过。
                    for n, i in path:
                        n.child_vloss[i] -= 1
                    self._pop_n(len(path))
                    root_i = path[0][1] if (self._gumbel and path) else None
                    return ("collision", root_i)

                # 单次生成合法着法，既用于终局判定又用于展开（消除双次 legal_moves）。
                legal = board.legal_moves()
                result, _term = board.status(legal)
                if result is not None:
                    node.terminal = True
                    if result == "1/2-1/2":
                        # 和棋：记 draw 标志，回传 draw_value 不翻符号（双方同罚）。
                        node.terminal_draw = True
                        node.terminal_value = self._draw_value
                    else:
                        node.terminal_draw = False
                        node.terminal_value = _terminal_value(result, board.side_to_move)
                    self._backup_terminal(path, node)
                    self._pop_n(len(path))
                    return ("terminal", None)

                # 待展开叶子：记录合法着法与视角，编码当前局面，标记挂起。
                node.pending = True
                side = board.side_to_move
                state = encode_board(board, self._history_steps)
                self._pop_n(len(path))
                token = (node, path, legal, side)
                return ("leaf", (state, token))

            # 已展开：选边并下行（施加 virtual loss）。
            # gumbel 模式下**根**用 Sequential Halving 调度替代 PUCT；非根仍走 PUCT。
            if self._gumbel and node is self._root:
                if not self._g_ready:
                    self._g_init()
                i = self._g_select_root()
            else:
                i = self._select_child(node)
            node.child_vloss[i] += 1
            path.append((node, i))
            board.push(node.child_moves[i])
            child = node.child_nodes[i]
            if child is None:
                child = _Node()
                node.child_nodes[i] = child
            node = child

    # ------------------------------------------------------------ 阶段一
    def select_leaves(self, max_leaves: int) -> tuple[np.ndarray, list]:
        """选出至多 max_leaves 个待评估叶子。

        返回 (states[K,C,10,9] float32, tokens)。终局叶子在内部直接回传、计入 visits，
        不出现在返回值里（K 可为 0）。
        """
        C = 14 * self._history_steps + 3
        empty = np.zeros((0, C, ROWS, COLS), dtype=np.float32)
        if max_leaves <= 0 or self._root.terminal:
            return empty, []

        if self._gumbel and self._g_block is not None:
            self._g_block[:] = 0    # 撞车计数只在本次调用内有效

        states: list = []
        tokens: list = []
        produced = 0            # 本次已产出的模拟数（叶子 + 终局）
        since_progress = 0      # 连续撞车计数
        attempts = 0
        max_attempts = max_leaves * 4 + 16
        stall_limit = max_leaves + 8

        while produced < max_leaves and attempts < max_attempts and since_progress <= stall_limit:
            attempts += 1
            kind, data = self._descend()
            if kind == "leaf":
                state, token = data
                states.append(state)
                tokens.append(token)
                produced += 1
                since_progress = 0
            elif kind == "terminal":
                produced += 1
                since_progress = 0
            else:  # collision
                if self._gumbel and data is not None:
                    self._g_block[data] += 1
                since_progress += 1

        if states:
            batch = np.stack(states).astype(np.float32, copy=False)
        else:
            batch = empty
        return batch, tokens

    # ------------------------------------------------------------ 阶段二
    def policy_indices(self, tokens: list) -> list:
        """每个叶子的合法着法策略索引（当前走子方视角，int64），供稀疏评估协议使用。

        返回 list[np.ndarray]，长度 = len(tokens)，第 i 个数组与 tokens[i] 的合法着法
        顺序一致，取值为网络策略输出的索引（0..8099）。它是稀疏返回协议
        （EvalServer 只在这些索引处 gather softmax 概率）的对齐依据。
        """
        return [_leaf_policy_indices(token[2], token[3]) for token in tokens]

    def _expand_and_backup(self, token, prior_unnorm: np.ndarray, value: float) -> None:
        """稠密 / 稀疏路径共享的展开逻辑：归一化先验→（根节点混噪声）→建子边→backup。

        prior_unnorm 为按合法着法顺序 gather 得到的先验（未归一化）；两条路径喂进的
        prior_unnorm 数值一致（同一 softmax 后在同一组索引处 gather），故结果严格等价。
        """
        node, path, legal, _side = token
        n = len(legal)
        prior = np.asarray(prior_unnorm, dtype=np.float64).reshape(-1)
        s = float(prior.sum())
        if s > 1e-12:
            prior = prior / s
        elif n > 0:
            prior = np.full(n, 1.0 / n, dtype=np.float64)

        # 根节点 Dirichlet 噪声（仅训练时）。gumbel 模式的探索源是 Gumbel 噪声，
        # 根不混 Dirichlet（DESIGN §8），且不得从 self._rng 取数。
        if node is self._root and self._add_noise and n > 0 and not self._gumbel:
            prior = self._mix_noise(prior)

        node.child_moves = list(legal)
        node.child_nodes = [None] * n
        node.child_P = prior.astype(np.float32)
        node.child_N = np.zeros(n, dtype=np.int64)
        node.child_W = np.zeros(n, dtype=np.float64)
        node.child_vloss = np.zeros(n, dtype=np.int64)
        node.expanded = True
        node.pending = False
        # 网络 value 存该节点自身走子方视角（= backup 的叶值视角）；树复用后此节点
        # 成为根时它就是根视角，Gumbel v_mix 直接取用，**勿再翻符号**。
        node.net_value = float(value)

        self._backup(path, float(value))

    def apply_results(
        self,
        tokens: list,
        policy_probs: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """回填网络先验并 backup（稠密路径）。

        policy_probs[K,8100] 为 softmax 后概率（未 mask），内部按合法着法 mask 归一后
        再展开到各子边。values[K] 为对应叶子走子方视角价值。
        """
        policy_probs = np.asarray(policy_probs, dtype=np.float32)
        values = np.asarray(values, dtype=np.float64).reshape(-1)

        for k, token in enumerate(tokens):
            _node, _path, legal, side = token
            idxs = _leaf_policy_indices(legal, side)
            prior = policy_probs[k][idxs].astype(np.float64)
            self._expand_and_backup(token, prior, float(values[k]))

    def apply_results_sparse(
        self,
        tokens: list,
        sparse_probs,
        values: np.ndarray,
    ) -> None:
        """回填网络先验并 backup（稀疏路径）。

        sparse_probs[i] 与 ``policy_indices(tokens)[i]`` 对齐（softmax 后在合法着法索引
        处 gather 所得，未归一化），values[i] 为对应叶子走子方视角价值。内部归一后展开，
        与 ``apply_results`` 数值严格等价。
        """
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        for k, token in enumerate(tokens):
            prior = np.asarray(sparse_probs[k], dtype=np.float64).reshape(-1)
            self._expand_and_backup(token, prior, float(values[k]))

    def _mix_noise(self, prior: np.ndarray) -> np.ndarray:
        """把 Dirichlet 噪声混入先验（float64 输入，返回 float64）。"""
        n = len(prior)
        if n <= 0:
            return prior
        noise = self._rng.dirichlet([self._alpha] * n)
        return (1.0 - self._eps) * prior + self._eps * noise

    # ------------------------------------------------------------ 树复用
    def advance(self, move) -> None:
        """走子并保留对应子树（树复用）。"""
        move = (int(move[0]), int(move[1]))
        reuse = None
        root = self._root
        if root.expanded and root.child_moves is not None:
            for i, m in enumerate(root.child_moves):
                if m == move:
                    reuse = root.child_nodes[i]
                    break

        self._board.push(move)
        # 换根 -> Gumbel 候选/调度全部作废，新根首次 select_leaves 时重新初始化。
        if self._gumbel:
            self._g_reset_state()
        if reuse is not None and (reuse.expanded or reuse.terminal):
            self._root = reuse
            # 树复用后新根需重新混入 Dirichlet 噪声（DESIGN §8：根节点每步加噪）。
            # 该子节点是在非根位置展开的，先验里尚无噪声，此处补上。
            # gumbel 模式不加 Dirichlet（探索源是每步重采的 Gumbel 噪声）。
            if self._add_noise and not self._gumbel and reuse.expanded and not reuse.terminal:
                reuse.child_P = self._mix_noise(
                    reuse.child_P.astype(np.float64)
                ).astype(np.float32)
        else:
            self._root = _Node()


# ---------------------------------------------------------------------------
# 单盘便捷封装（GUI / arena 用）。
# ---------------------------------------------------------------------------
def search(
    board,
    net_eval_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    cfg: MCTSConfig,
    num_sims: int,
    add_noise: bool = False,
    history_steps: int = 2,
    return_q: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
    """对单一局面跑 num_sims 次模拟。

    net_eval_fn: states[B,C,10,9] float32 -> (policy_probs[B,8100], values[B])。
    返回 (visit_counts[8100] 棋盘真实坐标, root_value 当前走子方视角)；
    return_q=True 时追加 q_values[8100]（根走子方视角，未访问边 NaN）。
    """
    # num_sims 透传给树（gumbel 的 SH 预算按它排；add_noise=False 时无影响）。
    tree = SearchTree(
        board, cfg, add_noise, history_steps=history_steps, num_sims=num_sims
    )

    def _result():
        if return_q:
            return tree.visit_counts(), tree.root_value(), tree.q_values()
        return tree.visit_counts(), tree.root_value()

    # 根即终局：无可搜索内容，直接返回真实结果值（root_value 惰性判定，和棋返回 draw_value）。
    if board.result() is not None:
        return _result()

    batch_cap = max(1, int(cfg.batch_size))
    while tree.total_visits < num_sims:
        before = tree.total_visits
        remaining = num_sims - before
        states, tokens = tree.select_leaves(min(batch_cap, remaining))
        if tokens:
            policy_probs, values = net_eval_fn(states)
            tree.apply_results(tokens, policy_probs, values)
        # 无 token 且毫无进展（仅根展开或全部撞车）：防止空转。
        if tree.total_visits == before and not tokens:
            break

    return _result()
