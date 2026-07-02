"""启发式收敛加速（DESIGN §10）。

提供以下工具：
  material_score(board_or_fen)  — FEN 材料分（红正黑负，过河兵 2 分）
  blend_value(z, material_tanh, lam) — 材料混合
  lambda_schedule(iteration, cfg)    — λ 线性退火
  apply_draw_penalty(z, cfg)         — 和棋惩罚
  ResignTracker                      — 认输判定（连续阈值 + 校准采样）
  opening_temperature(ply, cfg)      — 开局多样化温度
"""

from __future__ import annotations

import math
from typing import Union

from xiangqi.constants import (
    BLACK,
    BLACK_RIVER_BANK,
    CHAR_TO_PIECE,
    KING,
    MATERIAL_VALUE,
    PAWN,
    PAWN_CROSSED_VALUE,
    RED,
    RED_RIVER_BANK,
    sq_row,
)

# ─────────────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────────────

def _parse_board_segment(board_segment: str) -> dict:
    """解析 FEN 棋盘段，返回 {sq_index: piece_int} 字典。

    piece_int 符合 constants 约定：正数=红方棋子（+KING=1 .. +PAWN=7），
    负数=黑方棋子（-KING=-1 .. -PAWN=-7）。

    Parameters
    ----------
    board_segment : FEN 棋盘段（不含走子方等其他字段），如
                    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
    """
    pieces: dict[int, int] = {}
    # FEN 从 row 9（黑底线）往 row 0（红底线）依次描述
    row = 9
    for rank_str in board_segment.split("/"):
        col = 0
        for ch in rank_str:
            if ch.isdigit():
                col += int(ch)
            else:
                piece = CHAR_TO_PIECE.get(ch)
                if piece is not None:
                    sq_idx = row * 9 + col
                    pieces[sq_idx] = piece
                col += 1
        row -= 1
    return pieces


# ─────────────────────────────────────────────────────────────────────────────
# §10.1 材料价值
# ─────────────────────────────────────────────────────────────────────────────

def material_score(board_or_fen: Union[str, object]) -> float:
    """计算局面材料分（红方视角：红正黑负）。

    材料价值参见 constants.MATERIAL_VALUE：
      车9  炮4.5  马4  象2  士2  兵1（过河 2）  帅/将 0

    过河判定（DESIGN §2）：
      红兵 row >= 5（BLACK_RIVER_BANK）→ 过河
      黑兵 row <= 4（RED_RIVER_BANK）   → 过河

    Parameters
    ----------
    board_or_fen : FEN 字符串，或任意带 .fen() 方法的 Board 对象
    """
    if isinstance(board_or_fen, str):
        fen_str = board_or_fen
    else:
        # Board 对象（须有 .fen() 方法）
        fen_str = board_or_fen.fen()

    board_segment = fen_str.split()[0]
    pieces = _parse_board_segment(board_segment)

    score = 0.0
    for sq_idx, piece in pieces.items():
        piece_type = abs(piece)
        side = RED if piece > 0 else BLACK  # 1 or -1

        base_val = MATERIAL_VALUE.get(piece_type, 0.0)

        # 过河兵额外加值（覆盖 MATERIAL_VALUE[PAWN]=1.0 → 2.0）
        if piece_type == PAWN:
            row = sq_row(sq_idx)
            if side == RED:
                # 红兵过河：到了黑方本土（row >= 5）
                if row >= BLACK_RIVER_BANK:
                    base_val = PAWN_CROSSED_VALUE
            else:
                # 黑兵过河：到了红方本土（row <= 4）
                if row <= RED_RIVER_BANK:
                    base_val = PAWN_CROSSED_VALUE

        # 红子得正分，黑子得负分
        score += base_val * side  # side=RED=+1, BLACK=-1

    return score


# ─────────────────────────────────────────────────────────────────────────────
# §10.1 材料混合
# ─────────────────────────────────────────────────────────────────────────────

def blend_value(z: float, material_tanh: float, lam: float) -> float:
    """混合自我博弈结果 z 与材料估值。

    z_target = (1 - λ) · z + λ · material_tanh

    Parameters
    ----------
    z             : 自我博弈结果（+1/-1/0 + 和棋惩罚）
    material_tanh : tanh(material_diff / MATERIAL_SCALE)
    lam           : 混合权重 λ ∈ [0, 1]
    """
    return (1.0 - lam) * z + lam * material_tanh


def lambda_schedule(iteration: int, cfg) -> float:
    """返回当前迭代的材料混合权重 λ（线性退火）。

    λ 从 cfg.value_blend_init（0.5）线性退火到 0，
    经过 cfg.value_blend_iters 个迭代后固定为 0。

    Parameters
    ----------
    iteration : 当前迭代编号（0-based）
    cfg       : SelfPlayConfig（须有 value_blend_init, value_blend_iters 字段）
    """
    init_lam: float = float(cfg.value_blend_init)
    blend_iters: int = int(cfg.value_blend_iters)

    if blend_iters <= 0 or iteration >= blend_iters:
        return 0.0

    # 线性退火：iter=0 → init_lam，iter=blend_iters → 0
    return init_lam * (1.0 - iteration / blend_iters)


# ─────────────────────────────────────────────────────────────────────────────
# §10.2 和棋惩罚
# ─────────────────────────────────────────────────────────────────────────────

def apply_draw_penalty(z: float, cfg) -> float:
    """若局面为和棋（z == 0.0），施加和棋惩罚。

    和棋双方均受同等惩罚（cfg.draw_penalty < 0），以提高进取性。

    Parameters
    ----------
    z   : 原始博弈结果（0.0 表示和棋）
    cfg : SelfPlayConfig（须有 draw_penalty 字段）

    Returns
    -------
    惩罚后的 z 值（非和棋时原样返回）
    """
    if z == 0.0:
        return float(cfg.draw_penalty)
    return z


# ─────────────────────────────────────────────────────────────────────────────
# §10.3 认输机制
# ─────────────────────────────────────────────────────────────────────────────

class ResignTracker:
    """认输判定器（DESIGN §10.3）。

    规则：
      - 连续 `resign_consecutive` 个该方回合，root value < resign_threshold → 认输
      - 红黑两方各自独立计数，任意一方达标即触发
      - `resign_disable_frac` 比例对局禁用认输（校准误判率）

    Parameters
    ----------
    cfg     : SelfPlayConfig
    game_id : 对局编号（0-based）；决定此局是否为校准局
    """

    def __init__(self, cfg, game_id: int = 0) -> None:
        self._threshold: float = float(cfg.resign_threshold)
        self._consecutive: int = int(cfg.resign_consecutive)
        self._enabled: bool = bool(cfg.resign_enabled)

        # 校准对局：按 resign_disable_frac 采样（固定周期）
        disable_frac: float = float(cfg.resign_disable_frac)
        if disable_frac > 0.0:
            period = max(1, round(1.0 / disable_frac))
            self._calibration_game: bool = (game_id % period == 0)
        else:
            self._calibration_game = False

        # 红黑各自的连续低价值计数
        self._consec_red: int = 0
        self._consec_black: int = 0

        # 状态标志
        self._resigned: bool = False       # 实际触发了认输
        self._would_resign: bool = False   # 校准局中本应认输

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def is_calibration_game(self) -> bool:
        """此局是否为禁用认输的校准对局。"""
        return self._calibration_game

    @property
    def resigned(self) -> bool:
        """是否已实际触发认输（非校准局）。"""
        return self._resigned

    @property
    def would_have_resigned(self) -> bool:
        """校准局中是否本应触发认输（用于统计 false-positive 率）。"""
        return self._would_resign

    # ── 核心逻辑 ──────────────────────────────────────────────────────────

    def update(self, side_to_move: int, root_value: float) -> bool:
        """更新当前走子方的认输状态并判断是否认输。

        每步只更新"当前走子方"的连续计数；另一方的计数不受影响。

        Parameters
        ----------
        side_to_move : RED(1) 或 BLACK(-1)
        root_value   : MCTS 根节点价值（当前走子方视角，范围 [-1, 1]）

        Returns
        -------
        True  → 应当结束对局（认输），非校准局且达到连续阈值
        False → 继续对局
        """
        if not self._enabled:
            return False

        # 更新对应方的连续计数
        if side_to_move == RED:
            if root_value < self._threshold:
                self._consec_red += 1
            else:
                self._consec_red = 0
        else:  # BLACK
            if root_value < self._threshold:
                self._consec_black += 1
            else:
                self._consec_black = 0

        # 判断是否达到连续阈值
        should_resign = (
            self._consec_red >= self._consecutive
            or self._consec_black >= self._consecutive
        )

        if should_resign:
            self._would_resign = True
            if not self._calibration_game:
                self._resigned = True
                return True

        return False

    def reset(self) -> None:
        """重置连续计数与状态标志（新对局开始时调用）。"""
        self._consec_red = 0
        self._consec_black = 0
        self._resigned = False
        self._would_resign = False


# ─────────────────────────────────────────────────────────────────────────────
# §10.4 开局多样化
# ─────────────────────────────────────────────────────────────────────────────

def opening_temperature(ply: int, cfg) -> float:
    """返回当前 ply 的采样温度。

    前 cfg.opening_random_plies 步使用较高温度（cfg.opening_temperature），
    防止开局塌缩到相同变例；之后返回标准温度 1.0（由调用方结合 MCTS 配置处理）。

    Parameters
    ----------
    ply : 当前半回合数（0-based）
    cfg : SelfPlayConfig（须有 opening_random_plies, opening_temperature 字段）
    """
    if ply < int(cfg.opening_random_plies):
        return float(cfg.opening_temperature)
    return 1.0
