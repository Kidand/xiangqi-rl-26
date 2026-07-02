"""局面张量编码与动作索引转换（DESIGN.md §5–§6）。

当前走子方视角：黑方走子时棋盘 180° 翻转且红黑角色互换，
使网络永远"从下往上打"。价值输出恒为当前走子方视角。

输入张量形状 (C, 10, 9)，C = 14 * history_steps + 3：
  t*14 + 0..6  : 第 t 步历史局面 (t=0=当前)，当前走子方 K,A,B,N,R,C,P one-hot
  t*14 + 7..13 : 对手 K,A,B,N,R,C,P one-hot
  C-3          : 走子方颜色（红=全1，黑=全0）
  C-2          : 上一着起点 one-hot（无则全0）
  C-1          : 上一着终点 one-hot（无则全0）
"""

from __future__ import annotations

import numpy as np

from xiangqi.constants import (
    ACTION_SIZE,
    NUM_SQUARES,
    ROWS,
    COLS,
    RED,
    BLACK,
    flip_sq,
    flip_action,
    move_to_action,
    action_to_move,
)

# 棋子类型索引（1=KING..7=PAWN），在平面中占 7 个通道（index 0..6）
_NUM_PIECE_TYPES = 7


def encode_board(board, history_steps: int = 2) -> np.ndarray:
    """将棋盘局面编码为神经网络输入张量。

    参数
    ----
    board         : 符合 DESIGN §4 API 的 Board 对象
    history_steps : 历史步数 T，输出通道数 C = 14*T + 3

    返回
    ----
    np.ndarray, shape=(C, 10, 9), dtype=float32
    """
    C = 14 * history_steps + 3
    planes = np.zeros((C, ROWS, COLS), dtype=np.float32)

    side = board.side_to_move
    flip = side == BLACK  # 黑方视角需要翻转

    # 获取历史快照列表（index 0=当前，1=上一步，...，不足补 None）
    snapshots = board.snapshots(history_steps)

    # 棋子平面向量化填充：worker 现在是纯 CPU 进程，encode_board 是热点。
    # planes 视作 (C, 90) 二维视图（planes 连续，reshape 为 view，写入穿透原数组）。
    flat = planes.reshape(C, NUM_SQUARES)
    for t, snap in enumerate(snapshots):
        if snap is None:
            # 历史不足，该时间步全零（已默认）
            continue

        base = t * 14

        # snap 为 90 字节（0..14，7=空）。frombuffer→int16 再 -7 得 piece（-7..7，0=空）。
        vals = np.frombuffer(snap, dtype=np.uint8).astype(np.int16) - 7  # (90,)
        occ = np.nonzero(vals)[0]                # 有子格子（raw 坐标）
        if occ.size == 0:
            continue
        pieces = vals[occ]                       # 各子有符号类型（±1..±7）

        # 编码坐标：黑方视角翻转 enc_sq = 89 - raw_sq，红方视角保持不变。
        enc_sq = (89 - occ) if flip else occ

        pt_idx = np.abs(pieces) - 1              # 0..6 棋子类型通道
        # 判定"自方"：黑方视角自方=黑子(<0)，红方视角自方=红子(>0)。
        is_self = (pieces < 0) if flip else (pieces > 0)
        chan = np.where(is_self, pt_idx, 7 + pt_idx)  # 0..13

        flat[base + chan, enc_sq] = 1.0

    # C-3：走子方颜色（红=全1，黑=全0）
    if side == RED:
        planes[C - 3, :, :] = 1.0

    # C-2, C-1：上一着起点/终点 one-hot
    last_move = getattr(board, "last_move", None)
    if last_move is not None:
        from_sq, to_sq = last_move
        if flip:
            from_sq = flip_sq(from_sq)
            to_sq = flip_sq(to_sq)
        planes[C - 2, from_sq // COLS, from_sq % COLS] = 1.0
        planes[C - 1, to_sq // COLS, to_sq % COLS] = 1.0

    return planes


def move_to_policy_index(move: tuple[int, int], side_to_move: int) -> int:
    """将棋盘坐标下的着法转换为网络策略输出的索引（当前走子方视角）。

    黑方走子时，动作编码自动做 180° 翻转，使索引恒为当前方视角。

    参数
    ----
    move          : (from_sq, to_sq)，棋盘真实坐标
    side_to_move  : RED(1) 或 BLACK(-1)

    返回
    ----
    int，范围 [0, ACTION_SIZE)
    """
    action = move_to_action(move)
    if side_to_move == BLACK:
        action = flip_action(action)
    return action


def policy_index_to_move(idx: int, side_to_move: int) -> tuple[int, int]:
    """将策略索引（当前走子方视角）还原为棋盘真实坐标下的着法。

    参数
    ----
    idx           : 策略索引，范围 [0, ACTION_SIZE)
    side_to_move  : RED(1) 或 BLACK(-1)

    返回
    ----
    (from_sq, to_sq)，棋盘真实坐标
    """
    if side_to_move == BLACK:
        idx = flip_action(idx)
    return action_to_move(idx)


def legal_move_mask(board) -> np.ndarray:
    """生成当前走子方视角下的合法着法掩码。

    参数
    ----
    board : 符合 DESIGN §4 API 的 Board 对象（需实现 legal_moves(), side_to_move）

    返回
    ----
    np.ndarray, shape=(8100,), dtype=bool
      True  → 该索引对应的动作合法（当前走子方视角）
    """
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    side = board.side_to_move
    for move in board.legal_moves():
        idx = move_to_policy_index(move, side)
        mask[idx] = True
    return mask
