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
        """根 Q（当前走子方视角）。"""
        root = self._root
        if root.terminal:
            return float(root.terminal_value)
        if not root.expanded:
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

    def _descend(self):
        """从根下行一次。

        返回：
          ("leaf", (state, token))  待网络评估的叶子（已施加 virtual loss）
          ("terminal", None)        终局叶子已直接 backup 并计 visits
          ("collision", None)       撞到本 batch 已挂起的未展开节点，本次模拟跳过
        """
        node = self._root
        board = self._board
        path: list = []

        while True:
            if node.terminal:
                # 复访已解出的终局节点：再计一次访问，值固定。
                self._backup(path, node.terminal_value)
                self._pop_n(len(path))
                return ("terminal", None)

            if not node.expanded:
                if node.pending:
                    # 同一 batch 撞到未展开且已挂起的节点：撤销本路 vloss，跳过。
                    for n, i in path:
                        n.child_vloss[i] -= 1
                    self._pop_n(len(path))
                    return ("collision", None)

                # 单次生成合法着法，既用于终局判定又用于展开（消除双次 legal_moves）。
                legal = board.legal_moves()
                result, _term = board.status(legal)
                if result is not None:
                    node.terminal = True
                    node.terminal_value = _terminal_value(result, board.side_to_move)
                    self._backup(path, node.terminal_value)
                    self._pop_n(len(path))
                    return ("terminal", None)

                # 待展开叶子：记录合法着法与视角，编码当前局面，标记挂起。
                node.pending = True
                side = board.side_to_move
                state = encode_board(board, self._history_steps)
                self._pop_n(len(path))
                token = (node, path, legal, side)
                return ("leaf", (state, token))

            # 已展开：PUCT 选边并下行（施加 virtual loss）。
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

        # 根节点 Dirichlet 噪声（仅训练时）。
        if node is self._root and self._add_noise and n > 0:
            prior = self._mix_noise(prior)

        node.child_moves = list(legal)
        node.child_nodes = [None] * n
        node.child_P = prior.astype(np.float32)
        node.child_N = np.zeros(n, dtype=np.int64)
        node.child_W = np.zeros(n, dtype=np.float64)
        node.child_vloss = np.zeros(n, dtype=np.int64)
        node.expanded = True
        node.pending = False

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
        if reuse is not None and (reuse.expanded or reuse.terminal):
            self._root = reuse
            # 树复用后新根需重新混入 Dirichlet 噪声（DESIGN §8：根节点每步加噪）。
            # 该子节点是在非根位置展开的，先验里尚无噪声，此处补上。
            if self._add_noise and reuse.expanded and not reuse.terminal:
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
) -> tuple[np.ndarray, float]:
    """对单一局面跑 num_sims 次模拟。

    net_eval_fn: states[B,C,10,9] float32 -> (policy_probs[B,8100], values[B])。
    返回 (visit_counts[8100] 棋盘真实坐标, root_value 当前走子方视角)。
    """
    tree = SearchTree(board, cfg, add_noise, history_steps=history_steps)

    # 根即终局：无可搜索内容，直接返回真实结果值。
    if board.result() is not None:
        return tree.visit_counts(), _terminal_value(board.result(), board.side_to_move)

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

    return tree.visit_counts(), tree.root_value()
