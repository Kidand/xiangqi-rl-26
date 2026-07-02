# -*- coding: utf-8 -*-
"""规则引擎性能基准（见 DESIGN.md §15）。

- 随机对局测 movegen + push/pop 速度（moves/s）。
- perft(1..depth) 计时与 nodes/s。

用法:
    .venv/Scripts/python.exe scripts/bench_engine.py [--games 300] [--perft 3] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# 允许直接以脚本方式运行（无需安装包）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiangqi.board import Board  # noqa: E402


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for m in board.legal_moves():
        board.push(m)
        total += perft(board, depth - 1)
        board.pop()
    return total


def bench_random_games(games: int, max_plies: int, seed: int) -> None:
    rng = random.Random(seed)
    total_moves = 0
    total_gens = 0
    finished = 0
    t0 = time.perf_counter()
    for _ in range(games):
        b = Board(max_plies=max_plies)
        for _ in range(max_plies):
            lm = b.legal_moves()
            total_gens += 1
            if not lm:
                finished += 1
                break
            b.push(rng.choice(lm))
            total_moves += 1
        else:
            # 达到 max_plies 未分胜负
            pass
    dt = time.perf_counter() - t0
    print("=== 随机对局基准 ===")
    print(f"对局数           : {games}")
    print(f"总着法数         : {total_moves}")
    print(f"legal_moves 调用 : {total_gens}")
    print(f"耗时             : {dt:.3f}s")
    if dt > 0:
        print(f"moves/s          : {total_moves / dt:,.0f}")
        print(f"legal_moves/s    : {total_gens / dt:,.0f}")


def bench_perft(max_depth: int) -> None:
    print("=== perft 基准（初始局面）===")
    for depth in range(1, max_depth + 1):
        b = Board()
        t0 = time.perf_counter()
        nodes = perft(b, depth)
        dt = time.perf_counter() - t0
        rate = f"{nodes / dt:,.0f}" if dt > 0 else "inf"
        print(f"perft({depth}) = {nodes:>8}  |  {dt:8.3f}s  |  {rate:>12} nodes/s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Xiangqi 规则引擎基准")
    ap.add_argument("--games", type=int, default=300, help="随机对局数")
    ap.add_argument("--max-plies", type=int, default=400, help="每局最大半回合")
    ap.add_argument("--perft", type=int, default=3, help="perft 最大深度")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    args = ap.parse_args()

    bench_random_games(args.games, args.max_plies, args.seed)
    print()
    bench_perft(args.perft)


if __name__ == "__main__":
    main()
