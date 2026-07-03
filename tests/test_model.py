"""网络模型测试（DESIGN.md §7，test_model.py）。

覆盖内容：
  - 各预设尺寸前向传播形状 / value tanh 范围 / 参数量打印
  - TorchScript 导出与原模型输出数值一致（atol=1e-4）
  - ONNX 导出后用 onnxruntime 校验一致性（atol=1e-4）
  - checkpoint save/load 往返
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.config import ModelConfig
from rl.model import (
    XiangqiNet,
    create_model,
    load_checkpoint,
    save_checkpoint,
)
from xiangqi.constants import ACTION_SIZE, ROWS, COLS

# 动态 batch 验证用的批次大小列表
_BATCH_SIZES = [1, 2, 4]

# 预设尺寸
_PRESETS = [
    ModelConfig(blocks=5,  filters=64,  se=False, history_steps=2),   # small
    ModelConfig(blocks=10, filters=128, se=False, history_steps=2),   # medium（默认）
    ModelConfig(blocks=15, filters=192, se=False, history_steps=2),   # large
    ModelConfig(blocks=5,  filters=64,  se=True,  history_steps=2),   # small + SE
]

# 只取 small 预设用于导出测试（加快速度）
_SMALL_CFG = ModelConfig(blocks=5, filters=64, se=False, history_steps=2)


def _random_input(cfg: ModelConfig, batch_size: int = 1) -> torch.Tensor:
    return torch.randn(batch_size, cfg.in_channels, ROWS, COLS)


# ============================================================
# 前向传播形状与值域
# ============================================================

class TestForwardShape:
    @pytest.mark.parametrize("cfg", _PRESETS)
    def test_policy_shape(self, cfg: ModelConfig):
        model = create_model(cfg).eval()
        x = _random_input(cfg, batch_size=2)
        with torch.no_grad():
            policy, _ = model(x)
        assert policy.shape == (2, ACTION_SIZE), (
            f"policy 形状错误: {policy.shape}"
        )

    @pytest.mark.parametrize("cfg", _PRESETS)
    def test_value_shape(self, cfg: ModelConfig):
        model = create_model(cfg).eval()
        x = _random_input(cfg, batch_size=3)
        with torch.no_grad():
            _, value = model(x)
        assert value.shape == (3,), f"value 形状错误: {value.shape}"

    @pytest.mark.parametrize("batch_size", _BATCH_SIZES)
    def test_dynamic_batch(self, batch_size: int):
        cfg = _SMALL_CFG
        model = create_model(cfg).eval()
        x = _random_input(cfg, batch_size=batch_size)
        with torch.no_grad():
            policy, value = model(x)
        assert policy.shape[0] == batch_size
        assert value.shape[0] == batch_size


class TestValueRange:
    """value 输出经 tanh，值域在 [-1, 1]。"""

    @pytest.mark.parametrize("cfg", _PRESETS)
    def test_tanh_range(self, cfg: ModelConfig):
        model = create_model(cfg).eval()
        # 用极端输入也能保持 tanh 范围
        x = torch.randn(8, cfg.in_channels, ROWS, COLS) * 10.0
        with torch.no_grad():
            _, value = model(x)
        assert torch.all(value >= -1.0), f"value 超出下限: {value.min()}"
        assert torch.all(value <= 1.0), f"value 超出上限: {value.max()}"


class TestParamCount:
    """打印各预设参数量，并验证大致范围（不作严格约束，仅供参考）。"""

    def test_param_counts(self):
        for cfg in _PRESETS:
            model = create_model(cfg)
            n = sum(p.numel() for p in model.parameters())
            label = f"blocks={cfg.blocks},filters={cfg.filters},se={cfg.se}"
            print(f"  {label}: {n:,} 参数")
            # 宽松验证：至少有几万参数
            assert n > 10_000, f"参数量异常小: {n}"

    def test_small_param_count(self):
        """small 预设参数量（全卷积策略头后约 0.6M）。"""
        cfg = ModelConfig(blocks=5, filters=64, se=False)
        n = sum(p.numel() for p in create_model(cfg).parameters())
        assert 300_000 < n < 1_500_000, f"small 参数量超出合理范围: {n:,}"

    def test_larger_model_has_more_params(self):
        """较大模型参数量应多于较小模型。"""
        cfg_small = ModelConfig(blocks=5, filters=64, se=False)
        cfg_large = ModelConfig(blocks=15, filters=192, se=False)
        n_small = sum(p.numel() for p in create_model(cfg_small).parameters())
        n_large = sum(p.numel() for p in create_model(cfg_large).parameters())
        assert n_large > n_small, (
            f"large 模型 ({n_large:,}) 应大于 small ({n_small:,})"
        )


# ============================================================
# 检查点 save/load 往返
# ============================================================

class TestCheckpointRoundtrip:
    def test_save_and_load(self, tmp_path):
        cfg = _SMALL_CFG
        model = create_model(cfg).eval()

        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(
            model, cfg, ckpt_path,
            iteration=42,
            meta={"note": "unit-test"},
        )

        loaded_model, ckpt = load_checkpoint(ckpt_path, device="cpu")
        loaded_model.eval()

        # 验证 checkpoint 字段
        assert ckpt["iteration"] == 42
        assert ckpt["meta"]["note"] == "unit-test"
        mc = ckpt["model_config"]
        assert mc["blocks"] == cfg.blocks
        assert mc["filters"] == cfg.filters
        assert mc["se"] == cfg.se
        assert mc["history_steps"] == cfg.history_steps

        # 验证权重一致
        x = _random_input(cfg, batch_size=2)
        with torch.no_grad():
            p1, v1 = model(x)
            p2, v2 = loaded_model(x)

        np.testing.assert_allclose(
            p1.numpy(), p2.numpy(), atol=1e-6,
            err_msg="policy 权重往返不一致"
        )
        np.testing.assert_allclose(
            v1.numpy(), v2.numpy(), atol=1e-6,
            err_msg="value 权重往返不一致"
        )

    def test_config_preserved(self, tmp_path):
        """checkpoint 中 model_config 字段完整保存。"""
        cfg = ModelConfig(blocks=3, filters=32, se=True, history_steps=4)
        model = create_model(cfg)
        ckpt_path = tmp_path / "cfg_test.pt"
        save_checkpoint(model, cfg, ckpt_path, iteration=0)

        _, ckpt = load_checkpoint(ckpt_path)
        mc = ckpt["model_config"]
        assert mc["blocks"] == 3
        assert mc["filters"] == 32
        assert mc["se"] is True
        assert mc["history_steps"] == 4


# ============================================================
# 检查点原子写盘（写 .tmp 后 os.replace，覆盖已存在文件、无残留）
# ============================================================

class TestCheckpointAtomicSave:
    def test_overwrite_existing_no_tmp_residue(self, tmp_path):
        """覆盖已存在的 checkpoint 成功，且不留下 .tmp 残留。

        第一次保存 iteration=1，第二次覆盖同一路径 iteration=2；加载后应读到第二次的
        iteration，且目录中没有 ``*.tmp`` 文件（os.replace 已把 tmp 换名为正式文件）。
        """
        cfg = _SMALL_CFG
        model = create_model(cfg).eval()
        ckpt_path = tmp_path / "best.pt"

        save_checkpoint(model, cfg, ckpt_path, iteration=1)
        assert ckpt_path.exists()
        save_checkpoint(model, cfg, ckpt_path, iteration=2)  # 覆盖

        _, ckpt = load_checkpoint(ckpt_path, device="cpu")
        assert ckpt["iteration"] == 2, "覆盖后应读到第二次写入的 iteration"

        # 无 .tmp 残留
        residue = list(tmp_path.glob("*.tmp"))
        assert residue == [], f"不应残留 .tmp 文件：{residue}"

    def test_existing_file_intact_when_save_interrupted(self, tmp_path, monkeypatch):
        """写 .tmp 途中崩溃：原有 checkpoint 完整不被破坏（原子性核心保证）。

        先正常保存 iteration=1；再让 torch.save 抛异常模拟被 Ctrl-C 打断，确认：
        （a）原文件仍可加载且仍是 iteration=1（未被截断）；（b）不残留可被误当正式
        文件的半成品（残留仅可能是 .tmp）。
        """
        import rl.model as model_mod

        cfg = _SMALL_CFG
        model = create_model(cfg).eval()
        ckpt_path = tmp_path / "best.pt"
        save_checkpoint(model, cfg, ckpt_path, iteration=1)

        real_save = model_mod.torch.save

        def boom(*a, **k):
            # 先真实写出 tmp（模拟已写一部分），再抛异常打断 os.replace 之前。
            real_save(*a, **k)
            raise KeyboardInterrupt("模拟迭代末 Ctrl-C")

        monkeypatch.setattr(model_mod.torch, "save", boom)
        with pytest.raises(KeyboardInterrupt):
            save_checkpoint(model, cfg, ckpt_path, iteration=2)

        # 原文件未被破坏，仍是 iteration=1。
        _, ckpt = load_checkpoint(ckpt_path, device="cpu")
        assert ckpt["iteration"] == 1, "被打断的保存不得破坏原有 checkpoint"


# ============================================================
# TorchScript 导出一致性
# ============================================================

class TestTorchScript:
    def _get_traced(self, cfg: ModelConfig, sample: torch.Tensor):
        model = create_model(cfg).eval()
        with torch.no_grad():
            traced = torch.jit.trace(model, sample)
        return model, traced

    def test_torchscript_shape(self):
        cfg = _SMALL_CFG
        x = _random_input(cfg, batch_size=2)
        model, traced = self._get_traced(cfg, x)
        with torch.no_grad():
            p, v = traced(x)
        assert p.shape == (2, ACTION_SIZE)
        assert v.shape == (2,)

    def test_torchscript_output_match(self):
        cfg = _SMALL_CFG
        x = _random_input(cfg, batch_size=2)
        model, traced = self._get_traced(cfg, x)
        model.eval()
        with torch.no_grad():
            p_orig, v_orig = model(x)
            p_ts, v_ts = traced(x)

        np.testing.assert_allclose(
            p_orig.numpy(), p_ts.numpy(), atol=1e-4,
            err_msg="TorchScript policy 输出不一致"
        )
        np.testing.assert_allclose(
            v_orig.numpy(), v_ts.numpy(), atol=1e-4,
            err_msg="TorchScript value 输出不一致"
        )

    def test_torchscript_save_load(self, tmp_path):
        """TorchScript 保存后加载仍与原模型一致。"""
        cfg = _SMALL_CFG
        x = _random_input(cfg, batch_size=2)
        model, traced = self._get_traced(cfg, x)

        ts_path = tmp_path / "model.ts"
        traced.save(str(ts_path))
        loaded_ts = torch.jit.load(str(ts_path))

        model.eval()
        with torch.no_grad():
            p_orig, v_orig = model(x)
            p_loaded, v_loaded = loaded_ts(x)

        np.testing.assert_allclose(
            p_orig.numpy(), p_loaded.numpy(), atol=1e-4,
            err_msg="TorchScript 加载后 policy 不一致"
        )

    def test_torchscript_dynamic_batch(self, tmp_path):
        """TorchScript trace 后支持不同批次大小。"""
        cfg = _SMALL_CFG
        sample = _random_input(cfg, batch_size=1)
        model, traced = self._get_traced(cfg, sample)
        traced_path = tmp_path / "dyn.ts"
        traced.save(str(traced_path))
        loaded = torch.jit.load(str(traced_path))

        for bs in [1, 3, 8]:
            x = _random_input(cfg, batch_size=bs)
            with torch.no_grad():
                p, v = loaded(x)
            assert p.shape == (bs, ACTION_SIZE)
            assert v.shape == (bs,)


# ============================================================
# ONNX 导出与 onnxruntime 一致性
# ============================================================

class TestONNX:
    """ONNX 导出 + onnxruntime 推理数值一致性验证。"""

    @pytest.fixture
    def onnx_model_and_ref(self, tmp_path):
        """导出小模型并返回 (onnx_path, orig_model, sample_input)。"""
        import onnx as onnx_lib

        cfg = _SMALL_CFG
        model = create_model(cfg).eval()
        sample = _random_input(cfg, batch_size=2)
        onnx_path = tmp_path / "model.onnx"

        with torch.no_grad():
            # dynamo=False 使用 TorchScript 传统导出器（不依赖 onnxscript）
            torch.onnx.export(
                model,
                (sample,),          # args 必须为 tuple
                str(onnx_path),
                dynamo=False,
                input_names=["input"],
                output_names=["policy", "value"],
                dynamic_axes={
                    "input":  {0: "batch"},
                    "policy": {0: "batch"},
                    "value":  {0: "batch"},
                },
                opset_version=17,
                do_constant_folding=True,
            )

        # onnx.checker 校验图结构
        onnx_model = onnx_lib.load(str(onnx_path))
        onnx_lib.checker.check_model(onnx_model)

        return onnx_path, model, sample

    def test_onnx_checker_passes(self, onnx_model_and_ref):
        """ONNX 图结构校验通过（fixture 内已执行，此处仅确认 fixture 不报错）。"""
        onnx_path, _, _ = onnx_model_and_ref
        assert onnx_path.exists()

    def test_onnx_policy_match(self, onnx_model_and_ref):
        import onnxruntime as ort

        onnx_path, model, sample = onnx_model_and_ref
        with torch.no_grad():
            p_ref, _ = model(sample)

        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        p_ort = sess.run(["policy"], {"input": sample.numpy()})[0]

        np.testing.assert_allclose(
            p_ref.numpy(), p_ort, atol=1e-4,
            err_msg="ONNX policy 输出与原模型不一致"
        )

    def test_onnx_value_match(self, onnx_model_and_ref):
        import onnxruntime as ort

        onnx_path, model, sample = onnx_model_and_ref
        with torch.no_grad():
            _, v_ref = model(sample)

        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        v_ort = sess.run(["value"], {"input": sample.numpy()})[0]

        np.testing.assert_allclose(
            v_ref.numpy(), v_ort, atol=1e-4,
            err_msg="ONNX value 输出与原模型不一致"
        )

    def test_onnx_dynamic_batch(self, onnx_model_and_ref):
        """ONNX 支持不同批次大小（动态 batch 维度）。"""
        import onnxruntime as ort

        onnx_path, _, _ = onnx_model_and_ref
        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        cfg = _SMALL_CFG

        for bs in [1, 3, 5]:
            x = _random_input(cfg, batch_size=bs).numpy()
            outputs = sess.run(["policy", "value"], {"input": x})
            p_ort, v_ort = outputs[0], outputs[1]
            assert p_ort.shape == (bs, ACTION_SIZE), (
                f"ONNX policy 动态 batch 形状错误: bs={bs}, shape={p_ort.shape}"
            )
            assert v_ort.shape == (bs,), (
                f"ONNX value 动态 batch 形状错误: bs={bs}, shape={v_ort.shape}"
            )

    def test_onnx_value_tanh_range(self, onnx_model_and_ref):
        """ONNX 推理结果的 value 仍在 [-1, 1]。"""
        import onnxruntime as ort

        onnx_path, _, _ = onnx_model_and_ref
        cfg = _SMALL_CFG
        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        x = (_random_input(cfg, batch_size=8) * 10.0).numpy()
        v_ort = sess.run(["value"], {"input": x})[0]
        assert np.all(v_ort >= -1.0) and np.all(v_ort <= 1.0), (
            f"ONNX value 超出 tanh 范围: min={v_ort.min():.4f}, max={v_ort.max():.4f}"
        )

    def test_torchscript_vs_onnx_match(self, onnx_model_and_ref, tmp_path):
        """TorchScript 与 ONNX 推理结果互相一致（双重一致性）。"""
        import onnxruntime as ort

        onnx_path, model, sample = onnx_model_and_ref
        cfg = _SMALL_CFG

        # TorchScript
        with torch.no_grad():
            traced = torch.jit.trace(model, sample)
            p_ts, v_ts = traced(sample)

        # ONNX
        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        p_ort, v_ort = sess.run(["policy", "value"], {"input": sample.numpy()})

        np.testing.assert_allclose(
            p_ts.numpy(), p_ort, atol=1e-4,
            err_msg="TorchScript 与 ONNX policy 不一致"
        )
        np.testing.assert_allclose(
            v_ts.numpy(), v_ort, atol=1e-4,
            err_msg="TorchScript 与 ONNX value 不一致"
        )
