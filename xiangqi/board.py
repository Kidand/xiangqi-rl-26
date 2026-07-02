"""中国象棋规则引擎：局面表示、走子生成、终局判定（见 DESIGN.md §2-§4）。

设计要点：
- 棋盘用长度 90 的 ``list[int]``，有符号棋子值（正=红 / 负=黑 / 0=空），索引 ``sq=row*9+col``。
- ``push`` / ``pop`` 采用 make-unmake（记录被吃子、halfmove、zobrist、快照、走子方、
  以及本着是否将军），**绝不 deepcopy**。
- ``legal_moves`` = 伪合法生成 + 过滤（走后己方不被将军、帅不对脸）。
- Zobrist：90 格 × 15 种棋子 + 走子方，固定 seed 预生成；重复检测用哈希历史栈。
- 终局：将死 / 困毙（即负）/ 重复（长将判负、否则和）/ 120 半回合无吃子 / 超 max_plies。

本模块零第三方依赖、零 torch 依赖，性能敏感。
"""

from __future__ import annotations

import random

from .constants import (
    ADVISOR,
    BLACK,
    CANNON,
    ELEPHANT,
    EMPTY,
    HORSE,
    KING,
    NUM_SQUARES,
    PAWN,
    RED,
    ROOK,
    START_FEN,
)
from .fen import format_fen, parse_fen

__all__ = ["Board"]

# ---------------------------------------------------------------------------
# Zobrist 预生成表：90 格 × 15 种棋子（索引 = piece+7，空=7）+ 走子方键。
# 固定 seed 保证跨进程一致（重复检测 / 缓存复用）。
# ---------------------------------------------------------------------------
_ZOBRIST_SEED = 20260626
_zrng = random.Random(_ZOBRIST_SEED)
# ZOBRIST[sq][piece+7]；空格（索引 7）永不参与 XOR。
ZOBRIST: list[list[int]] = [[_zrng.getrandbits(64) for _ in range(15)] for _ in range(NUM_SQUARES)]
SIDE_KEY: int = _zrng.getrandbits(64)

# 走子 / 攻击方向
_ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ELEPHANT_STEPS = ((2, 2), (2, -2), (-2, 2), (-2, -2))
# 马的 8 个 (dr, dc)：|2| 分量方向即蹩腿方向
_HORSE_DELTAS = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))

_NO_CAPTURE_DRAW_PLIES = 120  # 连续无吃子半回合数上限（自然限着）


class Board:
    """中国象棋局面。RED=1 在下，BLACK=-1 在上。"""

    __slots__ = (
        "squares",
        "side_to_move",
        "halfmove",
        "fullmove",
        "max_plies",
        "no_capture_plies",
        "last_move",
        "king_pos",
        "_zobrist",
        "_undo",
        "_zhist",
        "_snaps",
    )

    # ------------------------------------------------------------------ 构造
    def __init__(
        self,
        fen: str | None = None,
        max_plies: int = 400,
        no_capture_plies: int = _NO_CAPTURE_DRAW_PLIES,
    ) -> None:
        squares, side, halfmove, fullmove = parse_fen(fen if fen is not None else START_FEN)
        self._setup(squares, side, halfmove, fullmove, max_plies, no_capture_plies)

    def _setup(self, squares, side, halfmove, fullmove, max_plies, no_capture_plies) -> None:
        self.squares: list[int] = squares
        self.side_to_move: int = side
        self.halfmove: int = halfmove
        self.fullmove: int = fullmove
        self.max_plies: int = max_plies
        self.no_capture_plies: int = no_capture_plies
        self.last_move: tuple[int, int] | None = None
        self.king_pos: dict[int, int] = {}
        for s, p in enumerate(squares):
            if p == KING:
                self.king_pos[RED] = s
            elif p == -KING:
                self.king_pos[BLACK] = s
        self._zobrist: int = self._compute_zobrist()
        self._undo: list[tuple] = []
        self._zhist: list[int] = [self._zobrist]
        self._snaps: list[bytes] = [self._make_snapshot()]

    @classmethod
    def from_fen(
        cls, fen: str, max_plies: int = 400, no_capture_plies: int = _NO_CAPTURE_DRAW_PLIES
    ) -> "Board":
        return cls(fen, max_plies=max_plies, no_capture_plies=no_capture_plies)

    # ------------------------------------------------------------------ 基础
    def _compute_zobrist(self) -> int:
        z = 0
        sq = self.squares
        for s in range(NUM_SQUARES):
            p = sq[s]
            if p != 0:
                z ^= ZOBRIST[s][p + 7]
        if self.side_to_move == BLACK:
            z ^= SIDE_KEY
        return z

    def _make_snapshot(self) -> bytes:
        # snapshot[sq] = piece_at(sq) + 7（0..14，空=7）
        # 列表推导比生成器表达式喂给 bytes() 更快（~1.8x，语义不变）。
        return bytes([p + 7 for p in self.squares])

    def zobrist(self) -> int:
        return self._zobrist

    def piece_at(self, s: int) -> int:
        return self.squares[s]

    def fen(self) -> str:
        return format_fen(self.squares, self.side_to_move, self.halfmove, self.fullmove)

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.squares = self.squares[:]
        b.side_to_move = self.side_to_move
        b.halfmove = self.halfmove
        b.fullmove = self.fullmove
        b.max_plies = self.max_plies
        b.no_capture_plies = self.no_capture_plies
        b.last_move = self.last_move
        b.king_pos = dict(self.king_pos)
        b._zobrist = self._zobrist
        b._undo = list(self._undo)
        b._zhist = list(self._zhist)
        b._snaps = list(self._snaps)
        return b

    def snapshots(self, n: int) -> list[bytes | None]:
        """最近 n 个局面快照，index 0=当前局面，1=一步前……不足补 None。"""
        snaps = self._snaps
        length = len(snaps)
        out: list[bytes | None] = []
        for i in range(n):
            idx = length - 1 - i
            out.append(snaps[idx] if idx >= 0 else None)
        return out

    # ------------------------------------------------------------ 攻击检测
    def _square_attacked(self, target: int, by_side: int) -> bool:
        """target 格是否被 by_side 一方的棋子攻击（含帅对脸/白脸将）。

        仕/相永不可能攻击到对方九宫内的将，故只检测 马/车/炮/兵/帅对脸。
        """
        sq = self.squares
        tr, tc = divmod(target, 9)

        # 马（含蹩腿）
        for dr, dc in _HORSE_DELTAS:
            orow = tr - dr
            ocol = tc - dc
            if 0 <= orow < 10 and 0 <= ocol < 9:
                if sq[orow * 9 + ocol] == HORSE * by_side:
                    if abs(dr) == 2:
                        leg = (orow + dr // 2) * 9 + ocol
                    else:
                        leg = orow * 9 + (ocol + dc // 2)
                    if sq[leg] == 0:
                        return True

        rook_p = ROOK * by_side
        king_p = KING * by_side
        pawn_p = PAWN * by_side
        cannon_p = CANNON * by_side

        for dr, dc in _ORTHO:
            r = tr + dr
            c = tc + dc
            dist = 1
            found = False
            while 0 <= r < 10 and 0 <= c < 9:
                pc = sq[r * 9 + c]
                if pc != 0:
                    found = True
                    break
                r += dr
                c += dc
                dist += 1
            if not found:
                continue
            # 第一个棋子
            if pc == rook_p:
                return True
            if pc == king_p:
                # 相邻的帅可攻击；同列（dc==0）且中间无子即帅对脸
                if dist == 1 or dc == 0:
                    return True
            if dist == 1 and pc == pawn_p:
                if _pawn_attacks(by_side, r, c, tr, tc):
                    return True
            # 越过第一个子找炮架后的炮
            r += dr
            c += dc
            while 0 <= r < 10 and 0 <= c < 9:
                pc2 = sq[r * 9 + c]
                if pc2 != 0:
                    if pc2 == cannon_p:
                        return True
                    break
                r += dr
                c += dc
        return False

    def is_in_check(self, side: int | None = None) -> bool:
        if side is None:
            side = self.side_to_move
        ksq = self.king_pos.get(side)
        if ksq is None:
            return False
        return self._square_attacked(ksq, -side)

    # ------------------------------------------------------------ 走法生成
    def _pseudo_moves(self) -> list[tuple[int, int]]:
        me = self.side_to_move
        sq = self.squares
        red = me > 0
        moves: list[tuple[int, int]] = []
        ap_append = moves.append

        for frm in range(NUM_SQUARES):
            p = sq[frm]
            if p == 0 or (p > 0) != red:
                continue
            ap = p if p > 0 else -p
            r, c = divmod(frm, 9)

            if ap == ROOK:
                for dr, dc in _ORTHO:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < 10 and 0 <= nc < 9:
                        t = nr * 9 + nc
                        v = sq[t]
                        if v == 0:
                            ap_append((frm, t))
                        else:
                            if (v > 0) != red:
                                ap_append((frm, t))
                            break
                        nr += dr
                        nc += dc

            elif ap == CANNON:
                for dr, dc in _ORTHO:
                    nr, nc = r + dr, c + dc
                    jumped = False
                    while 0 <= nr < 10 and 0 <= nc < 9:
                        t = nr * 9 + nc
                        v = sq[t]
                        if not jumped:
                            if v == 0:
                                ap_append((frm, t))
                            else:
                                jumped = True
                        else:
                            if v != 0:
                                if (v > 0) != red:
                                    ap_append((frm, t))
                                break
                        nr += dr
                        nc += dc

            elif ap == HORSE:
                for dr, dc in _HORSE_DELTAS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < 10 and 0 <= nc < 9):
                        continue
                    if abs(dr) == 2:
                        leg = (r + dr // 2) * 9 + c
                    else:
                        leg = r * 9 + (c + dc // 2)
                    if sq[leg] != 0:
                        continue
                    t = nr * 9 + nc
                    v = sq[t]
                    if v == 0 or (v > 0) != red:
                        ap_append((frm, t))

            elif ap == KING:
                for dr, dc in _ORTHO:
                    nr, nc = r + dr, c + dc
                    if 3 <= nc <= 5 and (0 <= nr <= 2 if red else 7 <= nr <= 9):
                        t = nr * 9 + nc
                        v = sq[t]
                        if v == 0 or (v > 0) != red:
                            ap_append((frm, t))

            elif ap == ADVISOR:
                for dr, dc in _DIAG:
                    nr, nc = r + dr, c + dc
                    if 3 <= nc <= 5 and (0 <= nr <= 2 if red else 7 <= nr <= 9):
                        t = nr * 9 + nc
                        v = sq[t]
                        if v == 0 or (v > 0) != red:
                            ap_append((frm, t))

            elif ap == ELEPHANT:
                for dr, dc in _ELEPHANT_STEPS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < 10 and 0 <= nc < 9):
                        continue
                    if (nr <= 4) if red else (nr >= 5):  # 不过河
                        eye = (r + dr // 2) * 9 + (c + dc // 2)
                        if sq[eye] != 0:  # 塞象眼
                            continue
                        t = nr * 9 + nc
                        v = sq[t]
                        if v == 0 or (v > 0) != red:
                            ap_append((frm, t))

            elif ap == PAWN:
                fdr = 1 if red else -1
                nr = r + fdr
                if 0 <= nr < 10:
                    t = nr * 9 + c
                    v = sq[t]
                    if v == 0 or (v > 0) != red:
                        ap_append((frm, t))
                crossed = (r >= 5) if red else (r <= 4)
                if crossed:
                    for dc in (1, -1):
                        nc = c + dc
                        if 0 <= nc < 9:
                            t = r * 9 + nc
                            v = sq[t]
                            if v == 0 or (v > 0) != red:
                                ap_append((frm, t))

        return moves

    def _leaves_king_safe(self, frm: int, to: int, me: int) -> bool:
        """轻量 make-unmake：试走 (frm,to) 后 me 方的帅是否安全（不被将、不对脸）。"""
        sq = self.squares
        moving = sq[frm]
        captured = sq[to]
        sq[to] = moving
        sq[frm] = EMPTY
        saved = None
        if moving == KING or moving == -KING:
            saved = self.king_pos[me]
            self.king_pos[me] = to
        ok = not self._square_attacked(self.king_pos[me], -me)
        sq[frm] = moving
        sq[to] = captured
        if saved is not None:
            self.king_pos[me] = saved
        return ok

    def legal_moves(self) -> list[tuple[int, int]]:
        me = self.side_to_move
        res = []
        for frm, to in self._pseudo_moves():
            if self._leaves_king_safe(frm, to, me):
                res.append((frm, to))
        return res

    # ------------------------------------------------------------ push / pop
    def push(self, move: tuple[int, int]) -> None:
        frm, to = move
        sq = self.squares
        moving = sq[frm]
        captured = sq[to]
        mover = self.side_to_move
        prev_halfmove = self.halfmove
        prev_zobrist = self._zobrist
        prev_last_move = self.last_move

        # 增量更新 zobrist
        z = self._zobrist
        z ^= ZOBRIST[frm][moving + 7]
        if captured != 0:
            z ^= ZOBRIST[to][captured + 7]
        z ^= ZOBRIST[to][moving + 7]
        z ^= SIDE_KEY
        self._zobrist = z

        # 落子
        sq[to] = moving
        sq[frm] = EMPTY
        if moving == KING or moving == -KING:
            self.king_pos[mover] = to

        # halfmove（无吃子累加，吃子清零）
        self.halfmove = 0 if captured != 0 else prev_halfmove + 1
        # 走子方 / 回合数
        self.side_to_move = -mover
        if mover == BLACK:
            self.fullmove += 1
        self.last_move = (frm, to)

        # 历史栈
        self._zhist.append(z)
        self._snaps.append(self._make_snapshot())

        # 本着是否将军（对手王被 mover 攻击）
        opp_king = self.king_pos.get(-mover)
        gave_check = opp_king is not None and self._square_attacked(opp_king, mover)

        self._undo.append(
            (frm, to, captured, prev_halfmove, prev_zobrist, prev_last_move, gave_check, mover)
        )

    def pop(self) -> None:
        frm, to, captured, prev_halfmove, prev_zobrist, prev_last_move, _gc, mover = self._undo.pop()
        sq = self.squares
        moving = sq[to]
        sq[frm] = moving
        sq[to] = captured
        if moving == KING or moving == -KING:
            self.king_pos[mover] = frm
        self.side_to_move = mover
        self.halfmove = prev_halfmove
        self._zobrist = prev_zobrist
        self.last_move = prev_last_move
        if mover == BLACK:
            self.fullmove -= 1
        self._zhist.pop()
        self._snaps.pop()

    # ------------------------------------------------------------ 终局判定
    def _perpetual_loser(self) -> int | None:
        """当前局面第 3 次出现时，判定最近一个重复周期内是否有一方每着都将军。

        返回长将方（判负方）RED/BLACK，或 None（普通重复 -> 和棋）。
        """
        z = self._zobrist
        zhist = self._zhist
        idxs = [j for j, zz in enumerate(zhist) if zz == z]
        if len(idxs) < 2:
            return None
        i1, i2 = idxs[-2], idxs[-1]  # 一个完整周期：位置 i1 -> i2
        undo = self._undo
        red_any = red_all = False
        black_any = black_all = False
        red_all = black_all = True
        for k in range(i1, i2):
            entry = undo[k]
            gave_check = entry[6]
            emover = entry[7]
            if emover == RED:
                red_any = True
                if not gave_check:
                    red_all = False
            else:
                black_any = True
                if not gave_check:
                    black_all = False
        red_perp = red_any and red_all
        black_perp = black_any and black_all
        if red_perp and not black_perp:
            return RED
        if black_perp and not red_perp:
            return BLACK
        return None

    @staticmethod
    def _winner_result(winner: int) -> str:
        return "1-0" if winner == RED else "0-1"

    def status(self, legal: list[tuple[int, int]] | None = None) -> tuple[str | None, str | None]:
        """终局判定：返回 (result, termination)，进行中为 (None, None)。

        判定优先级（不可变）：将死 / 困毙（即负）> 重复（长将判负 / 否则和）
        > 120 半回合无吃子 > 超 max_plies。与旧 ``_status`` 逐条一致。

        ``legal`` 可传入预先算好的 ``legal_moves()`` 结果以避免重复生成（MCTS 叶子
        处已生成一次）；为 None 时内部自行调用，语义完全相同。
        """
        moves = self.legal_moves() if legal is None else legal
        if not moves:
            # 无合法着法：将死或困毙，走子方均判负（困毙即负）。
            winner = -self.side_to_move
            if self.is_in_check(self.side_to_move):
                return self._winner_result(winner), "checkmate"
            return self._winner_result(winner), "stalemate"

        # 重复：当前局面第 3 次出现
        if self._zhist.count(self._zobrist) >= 3:
            loser = self._perpetual_loser()
            if loser is not None:
                return self._winner_result(-loser), "perpetual_check"
            return "1/2-1/2", "repetition"

        if self.halfmove >= self.no_capture_plies:
            return "1/2-1/2", "no_capture"
        if len(self._undo) >= self.max_plies:
            return "1/2-1/2", "max_moves"
        return None, None

    def result(self) -> str | None:
        return self.status()[0]

    def termination(self) -> str | None:
        return self.status()[1]

    def result_and_termination(self) -> tuple[str | None, str | None]:
        """一次返回 (result, termination)，避免 result()+termination() 双次判定。"""
        return self.status()

    # ------------------------------------------------------------ 杂项
    def __repr__(self) -> str:
        return f"Board(fen={self.fen()!r})"


def _pawn_attacks(pawn_side: int, pr: int, pc: int, tr: int, tc: int) -> bool:
    """位于 (pr,pc) 的 pawn_side 兵是否能攻击（走到）目标 (tr,tc)。"""
    if pawn_side == RED:
        if tr == pr + 1 and tc == pc:  # 向前（上）
            return True
        if pr >= 5 and tr == pr and abs(tc - pc) == 1:  # 过河后平移
            return True
    else:
        if tr == pr - 1 and tc == pc:  # 向前（下）
            return True
        if pr <= 4 and tr == pr and abs(tc - pc) == 1:
            return True
    return False
