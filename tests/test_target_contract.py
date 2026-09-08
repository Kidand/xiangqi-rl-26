"""价值标签契约：新实验不能静默混入未知或不同尺度的旧样本。"""

from __future__ import annotations

from copy import deepcopy
import json
import shutil

import numpy as np
import pytest

from rl.config import Config
from rl.replay_buffer import ReplayBuffer
from rl.targets import value_target_metadata


def _metadata(**changes):
    cfg = Config()
    cfg.selfplay.draw_penalty = 0.0
    cfg.selfplay.root_value_weight = 0.0
    cfg.selfplay.value_blend_init = 0.0
    cfg.selfplay.value_blend_iters = 0
    for key, value in changes.items():
        setattr(cfg.selfplay, key, value)
    return value_target_metadata(cfg)


def _saved_buffer(path, metadata):
    buf = ReplayBuffer(4, (2, 3, 4), target_metadata=metadata)
    states = np.arange(6 * 2 * 3 * 4, dtype=np.uint8).reshape(6, 2, 3, 4) % 2
    policies = [
        (np.array([i, i + 10], dtype=np.int16), np.array([0.25, 0.75], dtype=np.float32))
        for i in range(6)
    ]
    buf.add_game(states, policies, np.linspace(-1, 1, 6, dtype=np.float32))
    buf.save(path)
    return buf


def _remove_metadata(path):
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("target_metadata", None)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def test_value_target_metadata_schema_and_legacy_negative_draw():
    assert _metadata() == {
        "version": 1,
        "draw_penalty": 0.0,
        "root_value_weight": 0.0,
        "value_blend_init": 0.0,
        "value_blend_iters": 0,
    }
    metadata = _metadata(draw_penalty=-0.2, root_value_weight=0.3,
                         value_blend_init=0.5, value_blend_iters=30)
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["draw_penalty"] == -0.2
    assert metadata["value_blend_iters"] == 30


@pytest.mark.parametrize("field", [
    "draw_penalty", "root_value_weight", "value_blend_init", "value_blend_iters",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, True, "0.5"])
def test_non_finite_or_non_numeric_parameters_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        _metadata(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("draw_penalty", -1.01), ("draw_penalty", 1.01),
    ("root_value_weight", -0.01), ("root_value_weight", 1.01),
    ("value_blend_init", -0.01), ("value_blend_init", 1.01),
    ("value_blend_iters", -1), ("value_blend_iters", 0.5),
])
def test_out_of_range_parameters_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        _metadata(**{field: value})


def test_metadata_roundtrip_preserves_ring_contents(tmp_path):
    expected = _metadata()
    original = _saved_buffer(tmp_path / "buffer", expected)
    loaded = ReplayBuffer.load(tmp_path / "buffer", expected_target_metadata=expected)
    assert loaded.target_metadata == expected
    assert loaded._size == original._size
    assert loaded._pos == original._pos
    np.testing.assert_array_equal(loaded._states, original._states)
    np.testing.assert_array_equal(loaded._values, original._values)
    for left, right in zip(loaded._pol_probs, original._pol_probs):
        np.testing.assert_array_equal(left, right)
    meta = json.loads((tmp_path / "buffer" / "meta.json").read_text(encoding="utf-8"))
    assert meta["target_metadata"] == expected


def test_metadata_constructor_and_snapshot_copy_nested_data(tmp_path):
    metadata = {**_metadata(), "audit": {"sources": ["0716"]}}
    original = deepcopy(metadata)
    buf = ReplayBuffer(2, (1, 2, 2), metadata)
    metadata["audit"]["sources"].append("caller mutation")
    assert buf.target_metadata == original
    snap = buf.snapshot()
    buf.target_metadata["audit"]["sources"].append("later mutation")
    assert snap.target_metadata == original
    snap.save(tmp_path / "snapshot")
    loaded = ReplayBuffer.load(tmp_path / "snapshot", expected_target_metadata=original)
    assert loaded.target_metadata == original


@pytest.mark.parametrize("remove_key", [False, True])
def test_missing_metadata_rejected_before_array_allocation_or_loading(tmp_path, monkeypatch, remove_key):
    path = tmp_path / "buffer"
    _saved_buffer(path, None)
    if remove_key:
        _remove_metadata(path)

    def must_not_load(*args, **kwargs):
        pytest.fail("标签契约必须在分配/读取大数组之前检查")

    monkeypatch.setattr(np, "load", must_not_load)
    monkeypatch.setattr(np, "zeros", must_not_load)
    # capacity/state_shape 可用也不能把契约失败当成损坏并重建空 buffer。
    with pytest.raises(ValueError, match="缺少 target_metadata"):
        ReplayBuffer.load(path, capacity=4, state_shape=(2, 3, 4),
                          expected_target_metadata=_metadata())


@pytest.mark.parametrize("change", [
    {"version": 2}, {"draw_penalty": -0.2}, {"root_value_weight": 0.1},
    {"value_blend_init": 0.1}, {"value_blend_iters": 30},
])
def test_known_mismatch_never_falls_back_or_honors_legacy_override(tmp_path, monkeypatch, change):
    path = tmp_path / "buffer"
    expected = _metadata()
    _saved_buffer(path, expected)
    backup = tmp_path / "buffer.old"
    shutil.copytree(path, backup)
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["target_metadata"].update(change)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("不应读取样本数组"))

    with pytest.raises(ValueError, match="标签契约不匹配"):
        ReplayBuffer.load(path, capacity=4, state_shape=(2, 3, 4),
                          expected_target_metadata=expected,
                          allow_missing_target_metadata=True)
    assert backup.exists(), "契约失败不能删除可恢复的旧份"


def test_fallback_backup_must_also_match_contract(tmp_path):
    path = tmp_path / "buffer"
    _saved_buffer(tmp_path / "buffer.old", _metadata(draw_penalty=-0.2))
    with pytest.raises(ValueError, match="标签契约不匹配"):
        ReplayBuffer.load(path, capacity=4, state_shape=(2, 3, 4),
                          expected_target_metadata=_metadata())


def test_legacy_reader_without_expected_contract_remains_compatible(tmp_path):
    path = tmp_path / "buffer"
    original = _saved_buffer(path, None)
    _remove_metadata(path)
    loaded = ReplayBuffer.load(path)
    assert loaded.target_metadata is None
    np.testing.assert_array_equal(loaded._values, original._values)


def test_explicit_legacy_override_warns_and_preserves_unknown_metadata(tmp_path):
    path = tmp_path / "buffer"
    _saved_buffer(path, None)
    _remove_metadata(path)
    warnings = []
    loaded = ReplayBuffer.load(path, expected_target_metadata=_metadata(),
                               allow_missing_target_metadata=True, warn=warnings.append)
    assert loaded.target_metadata is None
    assert len(warnings) == 1 and "契约仍未知" in warnings[0]
    loaded.save(path)
    assert ReplayBuffer.load(path).target_metadata is None


def test_missing_or_corrupt_storage_rebuild_keeps_expected_contract(tmp_path):
    expected = _metadata()
    loaded = ReplayBuffer.load(tmp_path / "absent", capacity=4, state_shape=(2, 3, 4),
                               expected_target_metadata=expected, warn=lambda msg: None)
    assert len(loaded) == 0
    assert loaded.target_metadata == expected


def test_requested_state_shape_checked_before_array_loading(tmp_path, monkeypatch):
    path = tmp_path / "buffer"
    _saved_buffer(path, _metadata())
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("不应读取样本数组"))
    with pytest.raises(ValueError, match="state_shape 不匹配"):
        ReplayBuffer.load(path, capacity=4, state_shape=(3, 3, 4),
                          expected_target_metadata=_metadata())


@pytest.mark.parametrize("corrupt", ["states", "values", "sparse_policy"])
def test_payload_shapes_cannot_silently_broadcast_or_truncate(tmp_path, corrupt):
    path = tmp_path / "buffer"
    _saved_buffer(path, _metadata())
    if corrupt == "states":
        # 广播赋值本来会将单一平面复制到全部通道，应被拒绝。
        np.savez_compressed(path / "states_0000.npz", states=np.zeros((4, 1, 3, 4), dtype=np.uint8))
    elif corrupt == "values":
        np.save(path / "values.npy", np.array([0.5], dtype=np.float32))
    else:
        np.savez_compressed(path / "sparse_policy.npz", indices=np.array([0]),
                            probs=np.array([1.0]), lengths=np.array([1, 1, 1, 1]))
    with pytest.raises(ValueError, match="shape"):
        ReplayBuffer.load(path, expected_target_metadata=_metadata())
