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

    for t, snap in enumerate(snapshots):
        if snap is None:
            # 历史不足，该时间步全零（已默认）
            continue

        base = t * 14

        for raw_sq in range(NUM_SQUARES):
            raw_val = snap[raw_sq]       # 0..14，7 = 空格
            piece = raw_val - 7          # -7..7，0 = 空

            if piece == 0:
                continue  # 空格不处理

            # 计算编码坐标（黑方视角下翻转格子）
            enc_sq = flip_sq(raw_sq) if flip else raw_sq
            enc_row = enc_sq // COLS
            enc_col = enc_sq % COLS

            if flip:
                # 黑方视角：黑子（piece<0）变成"自方"，红子（piece>0）变成"对手"
                if piece < 0:
                    # 当前走子方（黑）自己的棋子
                    pt_idx = (-piece) - 1          # 0..6
                    planes[base + pt_idx, enc_row, enc_col] = 1.0
                else:
                    # 对手（红）的棋子
                    pt_idx = piece - 1             # 0..6
                    planes[base + 7 + pt_idx, enc_row, enc_col] = 1.0
            else:
                # 红方视角：红子（piece>0）为自方，黑子（piece<0）为对手
                if piece > 0:
                    pt_idx = piece - 1             # 0..6
                    planes[base + pt_idx, enc_row, enc_col] = 1.0
                else:
                    pt_idx = (-piece) - 1          # 0..6
                    planes[base + 7 + pt_idx, enc_row, enc_col] = 1.0

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
