"""测试训练支撑模块（DESIGN §10 §11）。

覆盖：
  rl.replay_buffer : 存取往返、环形覆盖、save-load
  rl.logger        : CSV 列头顺序
  rl.heuristics    : 材料分手算对照、ResignTracker 逻辑、lambda/blend/draw/temp
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

# ── xiangqi 引擎 / 常量 ──────────────────────────────────────────────────
from xiangqi.board import Board
from xiangqi.constants import (
    ACTION_SIZE,
    ADVISOR,
    action_to_move,
    CANNON,
    ELEPHANT,
    HORSE,
    KING,
    MATERIAL_VALUE,
    PAWN,
    PAWN_CROSSED_VALUE,
    RED,
    BLACK,
    ROOK,
    START_FEN,
)

# ── 待测模块 ──────────────────────────────────────────────────────────────
from rl.config import Config, SelfPlayConfig
from rl.heuristics import (
    ResignTracker,
    apply_draw_penalty,
    blend_value,
    lambda_schedule,
    material_score,
    opening_temperature,
)
from rl.logger import CSV_COLUMNS, TrainLogger
from rl.replay_buffer import ReplayBuffer
from rl.selfplay import sample_action_from_counts
from rl.train import resolve_train_steps


# ======================================================================
# 辅助函数
# ======================================================================

_STATE_SHAPE = (31, 10, 9)  # 默认 history_steps=2


def _make_sparse_policy(k: int = 5):
    """生成 k 个随机动作的稀疏策略 (indices_int16, probs_float32)。"""
    indices = np.random.choice(ACTION_SIZE, k, replace=False).astype(np.int16)
    probs = np.ones(k, dtype=np.float32) / k
    return indices, probs


def _make_game(n: int, state_shape=_STATE_SHAPE):
    """生成 n 步的伪对局样本（states, sparse_policies, values）。"""
    states = np.random.randint(0, 2, (n, *state_shape), dtype=np.uint8)
    sp = [_make_sparse_policy(5) for _ in range(n)]
    values = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
    return states, sp, values


# ======================================================================
# ReplayBuffer 测试
# ======================================================================

class TestReplayBufferBasic:
    def test_empty_len(self):
        buf = ReplayBuffer(capacity=100, state_shape=_STATE_SHAPE)
        assert len(buf) == 0

    def test_add_single_game_len(self):
        buf = ReplayBuffer(capacity=1000, state_shape=_STATE_SHAPE)
        states, sp, values = _make_game(10)
        buf.add_game(states, sp, values)
        assert len(buf) == 10

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=1000, state_shape=_STATE_SHAPE)
        states, sp, values = _make_game(20)
        buf.add_game(states, sp, values)

        s_out, pi_out, v_out = buf.sample(8)
        assert s_out.shape == (8, *_STATE_SHAPE)
        assert s_out.dtype == np.float32
        assert pi_out.shape == (8, ACTION_SIZE)
        assert pi_out.dtype == np.float32
        assert v_out.shape == (8,)
        assert v_out.dtype == np.float32

    def test_policy_dense_row_sum_equals_one(self):
        """稀疏策略稠密化后每行之和应等于 1（probs 已归一化）。"""
        buf = ReplayBuffer(capacity=200, state_shape=_STATE_SHAPE)
        states, sp, values = _make_game(50)
        buf.add_game(states, sp, values)

        _, pi_out, _ = buf.sample(50)
        row_sums = pi_out.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-5,
            err_msg="稠密化策略行和不等于 1"
        )

    def test_sample_dense_matches_naive(self, monkeypatch):
        """sample() 稀疏→稠密结果与逐样本朴素参考实现 bit-exact 一致（densification 契约）。

        固定采样索引（monkeypatch np.random.randint），覆盖变长稀疏策略、空策略、
        重复采样同一样本、int16 大索引等情形，作为稠密化行为的回归保护，并同时校验
        states/values 与采样索引对齐。
        """
        buf = ReplayBuffer(capacity=50, state_shape=(4, 10, 9))
        states = np.random.randint(0, 2, (6, 4, 10, 9), dtype=np.uint8)
        sps = [
            (np.array([1, 5, 7], dtype=np.int16),
             np.array([0.2, 0.5, 0.3], dtype=np.float32)),
            (np.array([0], dtype=np.int16),
             np.array([1.0], dtype=np.float32)),
            (np.array([3, 8], dtype=np.int16),
             np.array([0.4, 0.6], dtype=np.float32)),
            (np.array([], dtype=np.int16),
             np.array([], dtype=np.float32)),                    # 空策略
            (np.array([100, 200, 300, 8099], dtype=np.int16),
             np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)),
            (np.array([42], dtype=np.int16),
             np.array([1.0], dtype=np.float32)),
        ]
        values = np.arange(6, dtype=np.float32)
        buf.add_game(states, sps, values)

        # 固定采样索引：含空策略(3)、重复(4 出现两次)、边界大索引(4)。
        fixed_idx = np.array([0, 3, 4, 1, 4, 2, 5, 3], dtype=np.int64)
        monkeypatch.setattr(np.random, "randint", lambda *a, **k: fixed_idx)

        s_out, pi_out, v_out = buf.sample(len(fixed_idx))

        # 朴素参考实现（旧逐样本赋值）。
        ref = np.zeros((len(fixed_idx), ACTION_SIZE), dtype=np.float32)
        for j, si in enumerate(fixed_idx):
            pidx = buf._pol_indices[si]
            pprob = buf._pol_probs[si]
            if pidx is not None and len(pidx) > 0:
                ref[j, pidx.astype(np.int32)] = pprob

        np.testing.assert_array_equal(pi_out, ref)
        np.testing.assert_array_equal(s_out, buf._states[fixed_idx].astype(np.float32))
        np.testing.assert_array_equal(v_out, buf._values[fixed_idx])

    def test_state_values_are_float32(self):
        """states 输出为 float32（从 uint8 转换）。"""
        buf = ReplayBuffer(capacity=50, state_shape=(4, 10, 9))
        s = np.ones((5, 4, 10, 9), dtype=np.uint8)
        sp = [_make_sparse_policy() for _ in range(5)]
        v = np.zeros(5, dtype=np.float32)
        buf.add_game(s, sp, v)
        s_out, _, _ = buf.sample(3)
        assert s_out.dtype == np.float32

    def test_sample_empty_raises(self):
        buf = ReplayBuffer(capacity=10, state_shape=(2, 10, 9))
        with pytest.raises(RuntimeError, match="空"):
            buf.sample(1)

    def test_length_mismatch_raises(self):
        buf = ReplayBuffer(capacity=100, state_shape=_STATE_SHAPE)
        states = np.zeros((5, *_STATE_SHAPE), dtype=np.uint8)
        sp = [_make_sparse_policy() for _ in range(3)]  # 长度不匹配
        values = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="长度不一致"):
            buf.add_game(states, sp, values)


class TestReplayBufferRingOverwrite:
    def test_ring_size_capped_at_capacity(self):
        """添加超过容量的样本后 len == capacity。"""
        capacity = 10
        buf = ReplayBuffer(capacity=capacity, state_shape=(3, 10, 9))
        # 添加 27 个样本（超过容量 2.7 倍）
        states, sp, values = _make_game(27, state_shape=(3, 10, 9))
        buf.add_game(states, sp, values)
        assert len(buf) == capacity

    def test_ring_pos_wraps_correctly(self):
        """环形指针 pos 应等于 (total_samples % capacity)。"""
        capacity = 10
        buf = ReplayBuffer(capacity=capacity, state_shape=(3, 10, 9))
        n = 27  # 27 % 10 = 7
        states, sp, values = _make_game(n, state_shape=(3, 10, 9))
        buf.add_game(states, sp, values)
        assert buf._pos == n % capacity

    def test_ring_sample_after_overflow(self):
        """满溢后仍可正常采样（不崩）。"""
        capacity = 5
        buf = ReplayBuffer(capacity=capacity, state_shape=(2, 10, 9))
        states, sp, values = _make_game(12, state_shape=(2, 10, 9))
        buf.add_game(states, sp, values)
        assert len(buf) == capacity
        s, pi, v = buf.sample(3)
        assert s.shape == (3, 2, 10, 9)
        assert pi.shape == (3, ACTION_SIZE)

    def test_ring_overwrites_oldest(self):
        """环形缓冲区确保新数据覆盖了旧数据。

        验证方法：容量 3，写入 6 步（步值 0..5），
        采样所有槽位后，槽位中的价值应为 3,4,5 之一（最新 3 个）。
        """
        capacity = 3
        buf = ReplayBuffer(capacity=capacity, state_shape=(1, 2, 2))
        for i in range(6):
            s = np.zeros((1, 1, 2, 2), dtype=np.uint8)
            pol = [(np.array([i % ACTION_SIZE], dtype=np.int16),
                    np.array([1.0], dtype=np.float32))]
            v = np.array([float(i)], dtype=np.float32)
            buf.add_game(s, pol, v)

        # 直接检查底层价值数组——有效位置应为 3,4,5
        valid = set(buf._values[: capacity].tolist())
        assert valid == {3.0, 4.0, 5.0}, f"期望 {{3,4,5}}，实际 {valid}"


class TestReplayBufferSaveLoad:
    def test_roundtrip_basic(self, tmp_path):
        """保存/加载基本往返：len、state_shape、capacity 一致。"""
        buf = ReplayBuffer(capacity=50, state_shape=_STATE_SHAPE)
        states, sp, values = _make_game(20)
        buf.add_game(states, sp, values)

        save_dir = tmp_path / "buf1"
        buf.save(save_dir)
        buf2 = ReplayBuffer.load(save_dir)

        assert len(buf2) == len(buf) == 20
        assert buf2.capacity == buf.capacity
        assert buf2.state_shape == buf.state_shape

    def test_roundtrip_states_equal(self, tmp_path):
        """保存/加载后状态数组内容一致。"""
        buf = ReplayBuffer(capacity=30, state_shape=(5, 10, 9))
        states, sp, values = _make_game(15, state_shape=(5, 10, 9))
        buf.add_game(states, sp, values)

        save_dir = tmp_path / "buf_states"
        buf.save(save_dir)
        buf2 = ReplayBuffer.load(save_dir)

        np.testing.assert_array_equal(
            buf2._states[:15], buf._states[:15],
            err_msg="加载后状态不一致"
        )

    def test_roundtrip_values_equal(self, tmp_path):
        """保存/加载后价值数组内容一致。"""
        buf = ReplayBuffer(capacity=30, state_shape=_STATE_SHAPE)
        states, sp, values = _make_game(10)
        buf.add_game(states, sp, values)

        save_dir = tmp_path / "buf_vals"
        buf.save(save_dir)
        buf2 = ReplayBuffer.load(save_dir)

        np.testing.assert_allclose(
            buf2._values[:10], buf._values[:10],
            atol=1e-6, err_msg="加载后价值不一致"
        )

    def test_roundtrip_policy_equal(self, tmp_path):
        """保存/加载后稀疏策略稠密化结果一致。"""
        buf = ReplayBuffer(capacity=20, state_shape=_STATE_SHAPE)
        states, sp, values = _make_game(8)
        buf.add_game(states, sp, values)

        save_dir = tmp_path / "buf_policy"
        buf.save(save_dir)
        buf2 = ReplayBuffer.load(save_dir)

        # 对所有有效位置比较稠密策略
        for i in range(8):
            pi1 = buf._pol_indices[i].astype(np.int32)
            pp1 = buf._pol_probs[i]
            pi2 = buf2._pol_indices[i].astype(np.int32)
            pp2 = buf2._pol_probs[i]
            np.testing.assert_array_equal(pi1, pi2, err_msg=f"索引不一致 slot {i}")
            np.testing.assert_allclose(pp1, pp2, atol=1e-6, err_msg=f"概率不一致 slot {i}")

    def test_save_load_ring_overflow(self, tmp_path):
        """满溢后保存/加载，环形顺序和 pos 正确。"""
        capacity = 8
        buf = ReplayBuffer(capacity=capacity, state_shape=(2, 10, 9))
        states, sp, values = _make_game(12, state_shape=(2, 10, 9))
        buf.add_game(states, sp, values)
        assert len(buf) == capacity

        save_dir = tmp_path / "buf_ring"
        buf.save(save_dir)
        buf2 = ReplayBuffer.load(save_dir)

        assert len(buf2) == capacity
        assert buf2._pos == buf._pos

        for i in range(capacity):
            np.testing.assert_array_equal(
                buf2._states[i], buf._states[i],
                err_msg=f"满溢后槽位 {i} 数据不一致"
            )

    def test_save_load_empty(self, tmp_path):
        """空 buffer 保存/加载不崩，且 len==0。"""
        buf = ReplayBuffer(capacity=10, state_shape=(2, 10, 9))
        save_dir = tmp_path / "buf_empty"
        buf.save(save_dir)
        buf2 = ReplayBuffer.load(save_dir)
        assert len(buf2) == 0


# ======================================================================
# TrainLogger CSV 测试
# ======================================================================

class TestTrainLoggerCSV:
    def test_csv_column_order(self, tmp_path):
        """metrics.csv 列名顺序必须与 DESIGN §11 完全一致。"""
        logger = TrainLogger(log_dir=str(tmp_path / "logs"), tensorboard=False)
        row = {col: 0.0 for col in CSV_COLUMNS}
        row["iter"] = 1
        row["promoted"] = False
        logger.iteration_summary(row)
        logger.close()

        csv_path = tmp_path / "logs" / "metrics.csv"
        assert csv_path.exists(), "metrics.csv 未被创建"
        with open(str(csv_path), "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CSV_COLUMNS, (
            f"列顺序不符\n期望: {CSV_COLUMNS}\n实际: {header}"
        )

    def test_csv_header_written_once(self, tmp_path):
        """多次调用 iteration_summary 时 header 仅写一次。"""
        logger = TrainLogger(log_dir=str(tmp_path / "logs2"), tensorboard=False)
        for i in range(3):
            row = {col: float(i) for col in CSV_COLUMNS}
            row["iter"] = i
            row["promoted"] = (i % 2 == 0)
            logger.iteration_summary(row)
        logger.close()

        csv_path = tmp_path / "logs2" / "metrics.csv"
        with open(str(csv_path), "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        # 应有 1 header + 3 data 行 = 4 行
        assert len(all_lines) == 4, (
            f"期望 4 行（1 header + 3 data），实际 {len(all_lines)} 行"
        )

    def test_csv_column_count(self):
        """CSV_COLUMNS 应包含 DESIGN §11 规定的 25 列。"""
        assert len(CSV_COLUMNS) == 25, (
            f"CSV_COLUMNS 应有 25 列，实际 {len(CSV_COLUMNS)}"
        )

    def test_csv_required_columns_present(self):
        """关键列必须存在。"""
        required = [
            "iter", "red_winrate", "black_winrate", "draw_rate",
            "avg_plies", "buffer_size", "loss", "arena_score",
            "promoted", "selfplay_sec", "train_sec", "total_sec",
        ]
        for col in required:
            assert col in CSV_COLUMNS, f"CSV_COLUMNS 缺少列 '{col}'"

    def test_logger_info_warn_no_crash(self, tmp_path):
        """info/warn 接口不崩溃。"""
        logger = TrainLogger(log_dir=str(tmp_path / "logs_iw"), tensorboard=False)
        logger.info("测试 info 消息")
        logger.warn("测试 warn 消息")
        logger.close()

    def test_train_log_created(self, tmp_path):
        """logs/train.log 文件被创建。"""
        logger = TrainLogger(log_dir=str(tmp_path / "logs_tl"), tensorboard=False)
        logger.info("hello")
        logger.close()
        assert (tmp_path / "logs_tl" / "train.log").exists()


# ======================================================================
# material_score 测试
# ======================================================================

class TestMaterialScore:
    def test_start_fen_zero(self):
        """初始局面红黑对称，材料分应为 0.0。"""
        score = material_score(START_FEN)
        assert score == pytest.approx(0.0), f"初始局面材料分应为 0，实际: {score}"

    def test_manual_start_fen_components(self):
        """手算初始局面红方总材料 = 黑方总材料（对称验证）。"""
        expected_one_side = (
            2 * MATERIAL_VALUE[ROOK]     +  # 车×2 = 18
            2 * MATERIAL_VALUE[HORSE]    +  # 马×2 = 8
            2 * MATERIAL_VALUE[ELEPHANT] +  # 相×2 = 4
            2 * MATERIAL_VALUE[ADVISOR]  +  # 仕×2 = 4
            MATERIAL_VALUE[KING]         +  # 帅 = 0
            2 * MATERIAL_VALUE[CANNON]   +  # 炮×2 = 9
            5 * MATERIAL_VALUE[PAWN]        # 兵×5（未过河）= 5
        )
        # 红 + 黑 = expected - expected = 0
        assert expected_one_side == pytest.approx(48.0)
        assert material_score(START_FEN) == pytest.approx(0.0)

    def test_red_rook_advantage(self):
        """去掉黑方一辆车，材料分应 = +9.0。"""
        # 初始 FEN 黑方左车在 a9（FEN 首段第一字符 'r'）
        # "rnbakabnr" → 去掉第一个 'r' → "1nbakabnr"
        fen = "1nbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        score = material_score(fen)
        assert score == pytest.approx(MATERIAL_VALUE[ROOK]), (
            f"红多车材料分应为 {MATERIAL_VALUE[ROOK]}，实际: {score}"
        )

    def test_black_horse_advantage(self):
        """去掉红方一匹马，材料分应 = -4.0。"""
        # 初始 FEN row 0 = "RNBAKABNR"，去掉第二个 N → "R1BAKABNR"
        fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/R1BAKABNR w - - 0 1"
        score = material_score(fen)
        assert score == pytest.approx(-MATERIAL_VALUE[HORSE]), (
            f"黑多马材料分应为 {-MATERIAL_VALUE[HORSE]}，实际: {score}"
        )

    def test_crossed_red_pawn_value_2(self):
        """红兵过河（row >= 5）值 2 分。

        FEN 从 row 9 往下解析：段位置 5（从 1 起数）对应 row 5（BLACK_RIVER_BANK=5，过河）。
        "4k4/9/9/9/P8/9/9/9/9/4K4"：第 5 段 "P8" 对应 row 5 = 过河红兵。
        """
        fen = "4k4/9/9/9/P8/9/9/9/9/4K4 w - - 0 1"
        score = material_score(fen)
        assert score == pytest.approx(PAWN_CROSSED_VALUE), (
            f"过河红兵（row 5）应值 {PAWN_CROSSED_VALUE}，实际: {score}"
        )

    def test_uncrossed_red_pawn_value_1(self):
        """红兵未过河（row <= 4）值 1 分。

        "4k4/9/9/9/9/P8/9/9/9/4K4"：第 6 段 "P8" 对应 row 4 = 未过河红兵。
        """
        fen = "4k4/9/9/9/9/P8/9/9/9/4K4 w - - 0 1"
        score = material_score(fen)
        assert score == pytest.approx(MATERIAL_VALUE[PAWN]), (
            f"未过河红兵（row 4）应值 {MATERIAL_VALUE[PAWN]}，实际: {score}"
        )

    def test_crossed_black_pawn_value_neg2(self):
        """黑兵过河（row <= 4）值 -2 分。

        "4K4/9/9/9/9/p8/9/9/9/4k4"：第 6 段 "p8" 对应 row 4 = 过河黑兵（RED_RIVER_BANK=4）。
        """
        fen = "4K4/9/9/9/9/p8/9/9/9/4k4 b - - 0 1"
        score = material_score(fen)
        assert score == pytest.approx(-PAWN_CROSSED_VALUE), (
            f"过河黑兵（row 4）应值 {-PAWN_CROSSED_VALUE}，实际: {score}"
        )

    def test_uncrossed_black_pawn_value_neg1(self):
        """黑兵未过河（row >= 5）值 -1 分。

        "4K4/9/9/9/p8/9/9/9/9/4k4"：第 5 段 "p8" 对应 row 5 = 未过河黑兵。
        """
        fen = "4K4/9/9/9/p8/9/9/9/9/4k4 b - - 0 1"
        score = material_score(fen)
        assert score == pytest.approx(-MATERIAL_VALUE[PAWN]), (
            f"未过河黑兵（row 5）应值 {-MATERIAL_VALUE[PAWN]}，实际: {score}"
        )

    def test_board_object_with_fen_method(self):
        """board_or_fen 接受带 .fen() 方法的 Board 对象。"""
        class FakeBoardObj:
            def fen(self):
                return START_FEN

        score = material_score(FakeBoardObj())
        assert score == pytest.approx(0.0)

    def test_cannon_values(self):
        """炮价值 4.5：去掉双方各一炮，分数仍为 0。"""
        # 去掉红方一炮（row 2 col 7 = 'C' → h2）和黑方一炮（row 7 col 1 = 'c' → b7）
        # 原 FEN row2 = "1C5C1"，去掉右边的 C → "1C6"
        # 原 FEN row7 = "1c5c1"，去掉右边的 c → "1c6"
        # 对称去掉，分数应仍为 0
        fen = "rnbakabnr/9/1c6/p1p1p1p1p/9/9/P1P1P1P1P/1C6/9/RNBAKABNR w - - 0 1"
        score = material_score(fen)
        assert score == pytest.approx(0.0), (
            f"对称去掉各一炮后材料分应为 0，实际: {score}"
        )

    def test_board_squares_fastpath_matches_fen_start(self):
        """Board 对象快路径（遍历 squares）与 FEN 字符串路径在初始局面结果相等。"""
        board = Board()
        assert material_score(board) == material_score(board.fen()) == 0.0

    def test_board_squares_fastpath_matches_fen_random(self):
        """随机走子后的若干局面：快路径（board）与 FEN 路径逐局面结果完全相等。

        覆盖含吃子、过河兵、红黑双方视角的多样局面。两路径取值一致，结果应 bit-exact。
        """
        rng = np.random.default_rng(20260702)
        board = Board()
        checked = 0
        for _ in range(400):
            legal = board.legal_moves()
            res, _term = board.result_and_termination()
            if not legal or res is not None:
                board = Board()  # 终局则重开新局继续采样
                continue
            move = legal[int(rng.integers(0, len(legal)))]
            board.push(move)
            fast = material_score(board)          # squares 快路径
            slow = material_score(board.fen())    # FEN 字符串路径
            assert fast == slow, (
                f"两路径材料分不一致: fast={fast} slow={slow} fen={board.fen()!r}"
            )
            checked += 1
        assert checked >= 50  # 确保确有足够多样本被比较


# ======================================================================
# 启发式函数测试
# ======================================================================

class TestBlendValue:
    def test_lam_zero(self):
        """λ=0 时纯用 z。"""
        assert blend_value(0.7, 0.3, 0.0) == pytest.approx(0.7)

    def test_lam_one(self):
        """λ=1 时纯用 material_tanh。"""
        assert blend_value(0.7, 0.3, 1.0) == pytest.approx(0.3)

    def test_lam_half(self):
        """λ=0.5 时各取一半。"""
        assert blend_value(1.0, -1.0, 0.5) == pytest.approx(0.0)

    def test_formula(self):
        """验证公式 (1-λ)·z + λ·m。"""
        z, m, lam = 0.8, -0.4, 0.3
        expected = (1 - lam) * z + lam * m
        assert blend_value(z, m, lam) == pytest.approx(expected)


class TestLambdaSchedule:
    def _cfg(self, init=0.5, iters=30):
        return SelfPlayConfig(value_blend_init=init, value_blend_iters=iters)

    def test_iter_zero(self):
        """iter=0 时 λ = value_blend_init。"""
        cfg = self._cfg(0.5, 30)
        assert lambda_schedule(0, cfg) == pytest.approx(0.5)

    def test_iter_half(self):
        """iter=15 时 λ = 0.25（线性退火中点）。"""
        cfg = self._cfg(0.5, 30)
        assert lambda_schedule(15, cfg) == pytest.approx(0.25)

    def test_iter_end(self):
        """iter=blend_iters 时 λ = 0。"""
        cfg = self._cfg(0.5, 30)
        assert lambda_schedule(30, cfg) == pytest.approx(0.0)

    def test_iter_beyond(self):
        """iter > blend_iters 时 λ = 0。"""
        cfg = self._cfg(0.5, 30)
        assert lambda_schedule(100, cfg) == pytest.approx(0.0)

    def test_zero_iters(self):
        """blend_iters=0 时 λ 恒为 0。"""
        cfg = self._cfg(0.5, 0)
        assert lambda_schedule(0, cfg) == pytest.approx(0.0)

    def test_monotone_decreasing(self):
        """λ 随 iteration 单调非增。"""
        cfg = self._cfg(0.8, 20)
        lambdas = [lambda_schedule(i, cfg) for i in range(25)]
        for i in range(len(lambdas) - 1):
            assert lambdas[i] >= lambdas[i + 1], (
                f"λ 在 iter {i}→{i+1} 不单调：{lambdas[i]:.4f} < {lambdas[i+1]:.4f}"
            )


class TestApplyDrawPenalty:
    def _cfg(self, penalty=-0.1):
        return SelfPlayConfig(draw_penalty=penalty)

    def test_draw_gets_penalty(self):
        """和棋（z=0）应得到 draw_penalty。"""
        cfg = self._cfg(-0.1)
        assert apply_draw_penalty(0.0, cfg) == pytest.approx(-0.1)

    def test_win_unchanged(self):
        """胜利（z=1.0）不受影响。"""
        cfg = self._cfg(-0.1)
        assert apply_draw_penalty(1.0, cfg) == pytest.approx(1.0)

    def test_loss_unchanged(self):
        """失败（z=-1.0）不受影响。"""
        cfg = self._cfg(-0.1)
        assert apply_draw_penalty(-1.0, cfg) == pytest.approx(-1.0)

    def test_different_penalty(self):
        """验证不同 penalty 值被正确应用。"""
        cfg = self._cfg(-0.3)
        assert apply_draw_penalty(0.0, cfg) == pytest.approx(-0.3)


class TestOpeningTemperature:
    def _cfg(self, plies=8, temp=1.25):
        return SelfPlayConfig(opening_random_plies=plies, opening_temperature=temp)

    def test_early_plies_use_high_temp(self):
        """前 opening_random_plies 步温度 = opening_temperature。"""
        cfg = self._cfg(8, 1.25)
        for ply in range(8):
            assert opening_temperature(ply, cfg) == pytest.approx(1.25), (
                f"ply={ply} 应为 1.25"
            )

    def test_later_plies_use_standard_temp(self):
        """ply >= opening_random_plies 时温度 = 1.0。"""
        cfg = self._cfg(8, 1.25)
        for ply in [8, 10, 100]:
            assert opening_temperature(ply, cfg) == pytest.approx(1.0), (
                f"ply={ply} 应为 1.0"
            )

    def test_zero_plies_always_standard(self):
        """opening_random_plies=0 时任何 ply 都返回 1.0。"""
        cfg = self._cfg(0, 1.25)
        for ply in range(5):
            assert opening_temperature(ply, cfg) == pytest.approx(1.0)


# ======================================================================
# ResignTracker 测试
# ======================================================================

def _resign_cfg(**kwargs):
    defaults = dict(
        resign_enabled=True,
        resign_threshold=-0.92,
        resign_consecutive=4,
        resign_disable_frac=0.1,
    )
    defaults.update(kwargs)
    return SelfPlayConfig(**defaults)


class TestResignTracker:
    def test_trigger_red_four_consecutive(self):
        """红方连续 4 回合低于阈值 → 认输触发。"""
        cfg = _resign_cfg()
        tracker = ResignTracker(cfg, game_id=1)  # game_id=1 非校准局

        results = [tracker.update(RED, -0.95) for _ in range(4)]
        assert results[-1] is True, "第 4 次更新应触发认输"
        assert tracker.resigned is True

    def test_trigger_black_four_consecutive(self):
        """黑方连续 4 回合低于阈值 → 认输触发。"""
        cfg = _resign_cfg()
        tracker = ResignTracker(cfg, game_id=1)

        results = [tracker.update(BLACK, -0.95) for _ in range(4)]
        assert results[-1] is True
        assert tracker.resigned is True

    def test_no_trigger_below_three(self):
        """连续 3 次不触发（需 4 次）。"""
        cfg = _resign_cfg()
        tracker = ResignTracker(cfg, game_id=1)

        results = [tracker.update(RED, -0.95) for _ in range(3)]
        assert not any(results)
        assert tracker.resigned is False

    def test_reset_on_good_value(self):
        """中间出现高于阈值的分数，计数重置，不触发认输。"""
        cfg = _resign_cfg()
        tracker = ResignTracker(cfg, game_id=1)

        tracker.update(RED, -0.95)  # consec_red=1
        tracker.update(RED, -0.95)  # consec_red=2
        tracker.update(RED, 0.1)   # consec_red=0（重置）
        tracker.update(RED, -0.95)  # consec_red=1
        result = tracker.update(RED, -0.95)  # consec_red=2

        assert result is False
        assert tracker.resigned is False

    def test_calibration_game_no_actual_resign(self):
        """校准对局不触发实际认输，但记录 would_have_resigned。"""
        cfg = _resign_cfg(resign_disable_frac=0.1)
        # game_id=0 → 0 % round(1/0.1=10) == 0 → 校准局
        tracker = ResignTracker(cfg, game_id=0)
        assert tracker.is_calibration_game is True

        for _ in range(4):
            tracker.update(RED, -0.95)

        assert tracker.resigned is False, "校准局不应实际认输"
        assert tracker.would_have_resigned is True, "校准局应记录本应认输"

    def test_non_calibration_game_resigns(self):
        """非校准对局正常触发认输。"""
        cfg = _resign_cfg(resign_disable_frac=0.1)
        # game_id=1 → 1 % 10 != 0 → 非校准
        tracker = ResignTracker(cfg, game_id=1)
        assert tracker.is_calibration_game is False

        results = [tracker.update(RED, -0.95) for _ in range(4)]
        assert results[-1] is True
        assert tracker.resigned is True

    def test_disabled_never_resigns(self):
        """resign_enabled=False 时永远不认输。"""
        cfg = _resign_cfg(resign_enabled=False)
        tracker = ResignTracker(cfg, game_id=1)

        for _ in range(20):
            result = tracker.update(RED, -0.99)
        assert result is False
        assert tracker.resigned is False

    def test_sides_independent(self):
        """红黑两方计数独立：交替各 2 次不触发（各方连续数 = 2 < 4）。"""
        cfg = _resign_cfg()
        tracker = ResignTracker(cfg, game_id=1)

        results = []
        for _ in range(2):
            results.append(tracker.update(RED, -0.95))
            results.append(tracker.update(BLACK, -0.95))

        assert all(not r for r in results)
        assert tracker.resigned is False

    def test_red_reaches_four_black_unchanged(self):
        """红方连续 4 次低值，黑方无影响，仍触发认输。"""
        cfg = _resign_cfg()
        tracker = ResignTracker(cfg, game_id=1)

        # 黑方先走一次（高值）
        tracker.update(BLACK, 0.5)
        # 红方 4 次连续低值
        results = [tracker.update(RED, -0.95) for _ in range(4)]
        assert results[-1] is True

    def test_reset_clears_state(self):
        """reset() 后状态归零，之前计数清除。"""
        cfg = _resign_cfg()
        tracker = ResignTracker(cfg, game_id=1)

        # 让红方积累 3 次
        for _ in range(3):
            tracker.update(RED, -0.95)

        tracker.reset()
        # 重置后再走 3 次不触发（连续数从 0 开始）
        results = [tracker.update(RED, -0.95) for _ in range(3)]
        assert not any(results)
        assert tracker.resigned is False

    def test_threshold_boundary(self):
        """恰好等于阈值（-0.92 == threshold）不触发，低于才触发。"""
        cfg = _resign_cfg(resign_threshold=-0.92)
        tracker = ResignTracker(cfg, game_id=1)

        # 恰好等于阈值：不计入连续（root_value < threshold 才计）
        for _ in range(4):
            result = tracker.update(RED, -0.92)
        assert result is False, "等于阈值不应触发认输"

        # 低于阈值才触发
        tracker.reset()
        for _ in range(4):
            result = tracker.update(RED, -0.921)
        assert result is True, "低于阈值应触发认输"


# ======================================================================
# 优化器参数分组测试（DESIGN §9：L2 正则不应作用于 BN 与 bias）
# ======================================================================

class TestOptimizerWeightDecay:
    """weight decay 只能作用于卷积/全连接权重（2D+），BN 与所有 bias 不做 wd。"""

    def _build(self, optimizer_name):
        from rl.config import Config
        from rl.model import create_model
        from rl.train import _build_optimizer

        cfg = Config()
        cfg.model.se = True  # 连 SE 的 Linear 一起覆盖
        cfg.train.optimizer = optimizer_name
        cfg.train.weight_decay = 1e-4
        model = create_model(cfg.model)
        return model, _build_optimizer(model, cfg)

    @pytest.mark.parametrize("optimizer_name", ["adamw", "sgd"])
    def test_bn_and_bias_excluded_from_weight_decay(self, optimizer_name):
        model, opt = self._build(optimizer_name)

        decay_params, nodecay_params = [], []
        for g in opt.param_groups:
            if g["weight_decay"] == 0.0:
                nodecay_params.extend(g["params"])
            else:
                assert g["weight_decay"] == pytest.approx(1e-4)
                decay_params.extend(g["params"])

        # decay 组必须全是 2D+ 权重张量；no-decay 组必须全是 1D（BN weight/bias、各层 bias）
        assert decay_params and nodecay_params
        assert all(p.ndim >= 2 for p in decay_params)
        assert all(p.ndim <= 1 for p in nodecay_params)

        # 覆盖所有参数，无遗漏无重复
        total = sum(p.numel() for p in model.parameters())
        grouped = sum(p.numel() for p in decay_params + nodecay_params)
        assert grouped == total

        # 至少存在若干 1D 张量（BN），确保测试有意义
        assert sum(1 for p in model.parameters() if p.ndim <= 1) == len(nodecay_params)


# ======================================================================
# temp_final：温度期后走子采样（复用 sample_action_from_counts 温度路径）
# ======================================================================

class TestTempFinal:
    """走子选择在温度期后按 temp_final 采样：0 → argmax；>0 → N^(1/temp_final) 采样。

    _decide 温度期后取 tau=temp_final（>0）或 0（argmax），再交给
    sample_action_from_counts，故此处直接以伪 counts 验证该采样路径。
    """

    # 伪访问计数：argmax 明显（action 10），另有两个次优着。
    def _counts(self):
        counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        counts[10] = 10.0   # argmax
        counts[20] = 5.0
        counts[30] = 3.0
        return counts

    def test_temp_final_zero_is_argmax(self):
        """temp_final=0（tau=0）恒返回 argmax 着。"""
        counts = self._counts()
        argmax_move = action_to_move(10)
        rng = np.random.default_rng(0)
        for _ in range(1000):
            assert sample_action_from_counts(counts, 0.0, rng) == argmax_move

    def test_temp_final_positive_samples_non_argmax(self):
        """temp_final=0.25：1000 次采样（固定 seed）非 argmax 着频次 > 0，且 argmax 仍占多数。"""
        counts = self._counts()
        argmax_move = action_to_move(10)
        rng = np.random.default_rng(12345)
        picks = [sample_action_from_counts(counts, 0.25, rng) for _ in range(1000)]
        n_argmax = sum(1 for m in picks if m == argmax_move)
        n_other = 1000 - n_argmax
        assert n_other > 0, "temp_final>0 应能采到非 argmax 着"
        assert n_argmax > 500, f"argmax 仍应占多数，实际 {n_argmax}/1000"
        # 只采到了合法（有访问）着法。
        legal = {action_to_move(a) for a in (10, 20, 30)}
        assert set(picks).issubset(legal)


# ======================================================================
# resolve_train_steps：自动 / 固定训练步数（DESIGN §9.3）
# ======================================================================

class TestResolveTrainSteps:
    def _cfg(self, fixed=1000, reuse=3.0, batch=4096):
        cfg = Config()
        cfg.train.train_steps_per_iteration = fixed
        cfg.train.sample_reuse = reuse
        cfg.train.batch_size = batch
        return cfg

    def test_fixed_returns_as_is(self):
        """train_steps_per_iteration > 0 时原样返回，忽略新样本数。"""
        cfg = self._cfg(fixed=1000)
        assert resolve_train_steps(0, cfg) == 1000
        assert resolve_train_steps(10_000_000, cfg) == 1000

    def test_auto_lower_clamp(self):
        """自动模式下限 10：新样本极少时 ceil(...) < 10 → 返回 10。"""
        cfg = self._cfg(fixed=0, reuse=3.0, batch=4096)
        # ceil(100*3/4096)=ceil(0.073)=1 -> clamp 10
        assert resolve_train_steps(100, cfg) == 10
        assert resolve_train_steps(0, cfg) == 10

    def test_auto_upper_clamp(self):
        """自动模式上限 5000：新样本极多时 ceil(...) > 5000 → 返回 5000。"""
        cfg = self._cfg(fixed=0, reuse=3.0, batch=4096)
        # ceil(1e8*3/4096) 远大于 5000
        assert resolve_train_steps(100_000_000, cfg) == 5000

    def test_auto_typical_hand_calc(self):
        """典型值手算对照：ceil(1_000_000*3/4096)=733，落在 [10,5000] 内。"""
        cfg = self._cfg(fixed=0, reuse=3.0, batch=4096)
        assert resolve_train_steps(1_000_000, cfg) == 733

    def test_auto_boundary_exact_5000(self):
        """恰好边界：构造 ceil 结果正好 5000。"""
        cfg = self._cfg(fixed=0, reuse=1.0, batch=100)
        # ceil(500_000*1/100)=5000
        assert resolve_train_steps(500_000, cfg) == 5000

    def test_auto_reuse_scales_steps(self):
        """sample_reuse 线性缩放步数：reuse 翻倍步数翻倍（未触边界时）。"""
        cfg1 = self._cfg(fixed=0, reuse=2.0, batch=1000)
        cfg2 = self._cfg(fixed=0, reuse=4.0, batch=1000)
        # ceil(100_000*2/1000)=200 ; ceil(100_000*4/1000)=400
        assert resolve_train_steps(100_000, cfg1) == 200
        assert resolve_train_steps(100_000, cfg2) == 400
