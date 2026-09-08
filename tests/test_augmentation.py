"""左右镜像必须保持真实局面、历史和策略目标的对应关系。"""

from __future__ import annotations

import numpy as np
import pytest

from rl.augmentation import augment_horizontal
from rl.encoding import encode_board, legal_move_mask, move_to_policy_index
from xiangqi.board import Board
from xiangqi.constants import ACTION_SIZE, BLACK, COLS, NUM_SQUARES, RED, uci_to_move


def _mirror_move(move):
    return tuple((square // COLS) * COLS + COLS - 1 - square % COLS for square in move)


def _trajectory(history_steps):
    """两块真实棋盘各走原着和镜像着，保留完整的 Board 历史。"""
    board, mirror = Board(), Board()
    opening = ["h2e2", "h9g7", "h0g2", "b7e7", "i0h0", "i9h9", "g3g4", "g6g5", "g4g5"]
    states, mirrored_states, policies, mirrored_policies, sides = [], [], [], [], []
    for ply in range(len(opening) + 1):
        legal = board.legal_moves()
        assert set(mirror.legal_moves()) == {_mirror_move(move) for move in legal}
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        mirrored_policy = np.zeros_like(policy)
        weights = np.arange(1, len(legal) + 1, dtype=np.float32)
        weights /= weights.sum()
        for move, weight in zip(legal, weights):
            policy[move_to_policy_index(move, board.side_to_move)] = weight
            mirrored_policy[move_to_policy_index(_mirror_move(move), mirror.side_to_move)] = weight
        np.testing.assert_array_equal(mirrored_policy > 0, legal_move_mask(mirror))
        states.append(encode_board(board, history_steps=history_steps))
        mirrored_states.append(encode_board(mirror, history_steps=history_steps))
        policies.append(policy)
        mirrored_policies.append(mirrored_policy)
        sides.append(board.side_to_move)
        if ply < len(opening):
            move = uci_to_move(opening[ply])
            assert move in legal
            board.push(move)
            mirror.push(_mirror_move(move))
    return tuple(map(np.stack, (states, policies, mirrored_states, mirrored_policies))), sides


@pytest.mark.parametrize("history_steps", [1, 2, 4])
def test_real_trajectory_matches_mirrored_board_and_all_legal_actions(history_steps):
    (states, policies, expected_states, expected_policies), sides = _trajectory(history_steps)
    assert set(sides) == {RED, BLACK}
    # 初始补零历史与随后完整历史都在本批次内；后续轨迹是非对称的。
    assert np.any(states[1:] != expected_states[1:])
    actual_states, actual_policies = augment_horizontal(states, policies, 1.0)
    np.testing.assert_array_equal(actual_states, expected_states)
    np.testing.assert_array_equal(actual_policies, expected_policies)
    np.testing.assert_array_equal(actual_states[:, -3], states[:, -3])
    np.testing.assert_array_equal(np.sort(actual_policies, axis=1), np.sort(policies, axis=1))
    np.testing.assert_allclose(actual_policies.sum(axis=1), 1.0, atol=1e-7)
    assert np.isfinite(actual_policies).all()


@pytest.mark.parametrize("side", [RED, BLACK])
def test_mirrors_both_endpoints_for_every_action(side):
    states = np.zeros((1, 17, 10, 9), dtype=np.uint8)
    policies = np.arange(ACTION_SIZE, dtype=np.float32)[None, :]
    _, mirrored = augment_horizontal(states, policies, 1.0)
    for from_sq in range(NUM_SQUARES):
        for to_sq in range(NUM_SQUARES):
            move = (from_sq, to_sq)
            original_index = move_to_policy_index(move, side)
            mirrored_index = move_to_policy_index(_mirror_move(move), side)
            assert mirrored[0, mirrored_index] == policies[0, original_index]


def test_double_mirror_restores_exact_samples_and_never_modifies_inputs():
    (states, policies, _, _), _ = _trajectory(history_steps=4)
    original_states, original_policies = states.copy(), policies.copy()
    mirrored = augment_horizontal(states, policies, 1.0)
    restored = augment_horizontal(*mirrored, 1.0)
    for actual, expected in zip(restored, (original_states, original_policies)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(states, original_states)
    np.testing.assert_array_equal(policies, original_policies)
    for actual, original in zip(mirrored, (states, policies)):
        assert not np.shares_memory(actual, original)


class _FixedRng:
    def random(self, size):
        assert size == 4
        return np.array([0.0, 0.25, 0.5, 0.999])


def test_probability_selects_same_individual_samples_in_states_and_policies():
    (states, policies, expected_states, expected_policies), _ = _trajectory(history_steps=2)
    # 排除左右对称的初始局面，使每个样本是否翻转都可观察。
    states, policies = states[1:5], policies[1:5]
    actual_states, actual_policies = augment_horizontal(states, policies, 0.5, rng=_FixedRng())
    np.testing.assert_array_equal(actual_states[:2], expected_states[1:3])
    np.testing.assert_array_equal(actual_policies[:2], expected_policies[1:3])
    np.testing.assert_array_equal(actual_states[2:], states[2:])
    np.testing.assert_array_equal(actual_policies[2:], policies[2:])


@pytest.mark.parametrize("rng_factory", [np.random.default_rng, np.random.RandomState])
def test_explicit_rng_reproducibility(rng_factory):
    (states, policies, _, _), _ = _trajectory(history_steps=2)
    first = augment_horizontal(states, policies, 0.5, rng=rng_factory(716))
    second = augment_horizontal(states, policies, 0.5, rng=rng_factory(716))
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)


def test_default_rng_honors_global_numpy_seed():
    (states, policies, _, _), _ = _trajectory(history_steps=2)
    previous_random_state = np.random.get_state()
    try:
        np.random.seed(716)
        first = augment_horizontal(states, policies, 0.5)
        np.random.seed(716)
        second = augment_horizontal(states, policies, 0.5)
        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)
    finally:
        np.random.set_state(previous_random_state)


class _NoRng:
    def random(self, size):
        pytest.fail("deterministic augmentation must not consume randomness")


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_deterministic_probability_does_not_consume_randomness(probability, monkeypatch):
    states = np.zeros((2, 31, 10, 9), dtype=np.uint8)
    policies = np.zeros((2, ACTION_SIZE), dtype=np.float32)
    monkeypatch.setattr(np.random, "random", _NoRng().random)
    augment_horizontal(states, policies, probability)
    actual = augment_horizontal(states, policies, probability, rng=_NoRng())
    if probability == 0.0:
        assert actual[0] is states and actual[1] is policies


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_empty_batch_preserves_shape_and_does_not_consume_randomness(probability):
    states = np.empty((0, 31, 10, 9), dtype=np.uint8)
    policies = np.empty((0, ACTION_SIZE), dtype=np.float32)
    actual = augment_horizontal(states, policies, probability, rng=_NoRng())
    assert actual[0].shape == states.shape
    assert actual[1].shape == policies.shape


@pytest.mark.parametrize("state_dtype,policy_dtype", [(np.uint8, np.float32), (np.float32, np.float64)])
def test_readonly_noncontiguous_inputs_preserve_dtype_and_are_not_modified(state_dtype, policy_dtype):
    states = np.arange(4 * 31 * 10 * 9).reshape(4, 31, 10, 9).astype(state_dtype)[::2, :, :, ::-1]
    policies = np.arange(4 * ACTION_SIZE).reshape(4, ACTION_SIZE).astype(policy_dtype)[::2, ::-1]
    originals = (states.copy(), policies.copy())
    assert not states.flags.c_contiguous and not policies.flags.c_contiguous
    states.setflags(write=False)
    policies.setflags(write=False)
    actual = augment_horizontal(states, policies, 1.0)
    reference = augment_horizontal(*originals, 1.0)
    for result, expected, original, source in zip(actual, reference, originals, (states, policies)):
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(source, original)
        assert result.dtype == source.dtype
        assert result.flags.c_contiguous
        assert result.flags.writeable
        assert not np.shares_memory(result, source)


@pytest.mark.parametrize("probability", [-0.001, 1.001, np.nan, np.inf, -np.inf, None, "invalid"])
def test_invalid_probability_rejected(probability):
    with pytest.raises(ValueError, match="probability"):
        augment_horizontal(np.zeros((1, 31, 10, 9)), np.zeros((1, ACTION_SIZE)), probability)


@pytest.mark.parametrize(
    "state_shape,policy_shape,error",
    [
        ((31, 10, 9), (1, ACTION_SIZE), "states"),
        ((1, 31, 9, 10), (1, ACTION_SIZE), "states"),
        ((1, 0, 10, 9), (1, ACTION_SIZE), "states"),
        ((1, 31, 10, 9), (ACTION_SIZE,), "policies"),
        ((1, 31, 10, 9), (1, ACTION_SIZE - 1), "policies"),
        ((1, 31, 10, 9), (2, ACTION_SIZE), "policies"),
    ],
)
def test_invalid_shape_rejected_even_when_disabled(state_shape, policy_shape, error):
    with pytest.raises(ValueError, match=error):
        augment_horizontal(np.zeros(state_shape), np.zeros(policy_shape), 0.0)
