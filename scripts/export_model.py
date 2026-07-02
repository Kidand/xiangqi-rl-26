"""checkpoint 导出工具：TorchScript / ONNX（DESIGN.md §7 末段）。

用法示例：
    python scripts/export_model.py checkpoints/best.pt --torchscript --onnx
    python scripts/export_model.py checkpoints/best.pt --onnx --out-dir exports/

导出后自动用随机输入对比原模型与导出模型的输出，atol=1e-4 验证一致性。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# 确保仓库根目录在 sys.path，无论从何处调用脚本
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.model import load_checkpoint, XiangqiNet
from rl.config import ModelConfig


# ---------------------------------------------------------------------------
# 导出函数
# ---------------------------------------------------------------------------

def _make_sample_input(cfg: ModelConfig, batch_size: int = 1) -> torch.Tensor:
    """生成用于导出和验证的随机样本输入。"""
    C_in = cfg.in_channels
    return torch.randn(batch_size, C_in, 10, 9, dtype=torch.float32)


def export_torchscript(
    model: XiangqiNet,
    sample_input: torch.Tensor,
    out_path: Path,
) -> torch.jit.ScriptModule:
    """用 torch.jit.trace 导出 TorchScript 模型并保存到 out_path。"""
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_input)
    traced.save(str(out_path))
    print(f"[export] TorchScript 已保存: {out_path}")
    return traced


def export_onnx(
    model: XiangqiNet,
    cfg: ModelConfig,
    sample_input: torch.Tensor,
    out_path: Path,
    opset_version: int = 17,
) -> None:
    """导出 ONNX 模型（动态 batch 维度）到 out_path。"""
    import onnx

    model.eval()
    with torch.no_grad():
        # dynamo=False 使用 TorchScript 传统导出路径，不依赖 onnxscript
        torch.onnx.export(
            model,
            (sample_input,),    # args 必须为 tuple
            str(out_path),
            dynamo=False,
            input_names=["input"],
            output_names=["policy", "value"],
            dynamic_axes={
                "input":  {0: "batch"},
                "policy": {0: "batch"},
                "value":  {0: "batch"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    # 简单校验 ONNX 图结构
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    print(f"[export] ONNX 已保存并通过 onnx.checker: {out_path}")


# ---------------------------------------------------------------------------
# 验证函数
# ---------------------------------------------------------------------------

def _compare_outputs(
    ref_policy: np.ndarray,
    ref_value: np.ndarray,
    tgt_policy: np.ndarray,
    tgt_value: np.ndarray,
    label: str,
    atol: float = 1e-4,
) -> bool:
    """对比两组输出，打印结果，返回是否一致。"""
    p_ok = np.allclose(ref_policy, tgt_policy, atol=atol)
    v_ok = np.allclose(ref_value, tgt_value, atol=atol)
    p_max_diff = float(np.abs(ref_policy - tgt_policy).max())
    v_max_diff = float(np.abs(ref_value - tgt_value).max())
    status = "通过" if (p_ok and v_ok) else "不一致"
    print(
        f"[verify {label}] {status} | "
        f"policy 最大误差={p_max_diff:.2e} | "
        f"value  最大误差={v_max_diff:.2e} (atol={atol})"
    )
    return p_ok and v_ok


def verify_torchscript(
    model: XiangqiNet,
    traced: torch.jit.ScriptModule,
    sample_input: torch.Tensor,
    atol: float = 1e-4,
) -> bool:
    """对比原模型与 TorchScript 模型的输出。"""
    model.eval()
    with torch.no_grad():
        p_ref, v_ref = model(sample_input)
        p_ts, v_ts = traced(sample_input)

    return _compare_outputs(
        p_ref.numpy(), v_ref.numpy(),
        p_ts.numpy(), v_ts.numpy(),
        label="TorchScript",
        atol=atol,
    )


def verify_onnx(
    model: XiangqiNet,
    onnx_path: Path,
    sample_input: torch.Tensor,
    atol: float = 1e-4,
) -> bool:
    """对比原模型与 ONNX Runtime 推理的输出。"""
    import onnxruntime as ort

    model.eval()
    with torch.no_grad():
        p_ref, v_ref = model(sample_input)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        ["policy", "value"],
        {"input": sample_input.numpy()},
    )
    p_ort, v_ort = ort_out[0], ort_out[1]

    return _compare_outputs(
        p_ref.numpy(), v_ref.numpy(),
        p_ort, v_ort,
        label="ONNX",
        atol=atol,
    )


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="将 checkpoint 导出为 TorchScript 和/或 ONNX 格式。"
    )
    parser.add_argument("checkpoint", help="输入 checkpoint 路径（.pt）")
    parser.add_argument(
        "--torchscript", action="store_true", help="导出 TorchScript (.ts)"
    )
    parser.add_argument(
        "--onnx", action="store_true", help="导出 ONNX (.onnx)"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="导出目录（默认与 checkpoint 同目录）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="验证时使用的批次大小（默认 1）",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset 版本（默认 17）",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="输出一致性验证的绝对误差阈值（默认 1e-4）",
    )
    args = parser.parse_args()

    if not args.torchscript and not args.onnx:
        parser.error("请至少指定 --torchscript 或 --onnx 之一")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[error] checkpoint 文件不存在: {ckpt_path}")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = ckpt_path.stem

    print(f"[export] 加载 checkpoint: {ckpt_path}")
    model, ckpt = load_checkpoint(ckpt_path, device="cpu")
    model.eval()

    # 从 checkpoint 中取回 ModelConfig（已由 load_checkpoint 解析）
    mc = ckpt["model_config"]
    cfg = ModelConfig(
        blocks=int(mc["blocks"]),
        filters=int(mc["filters"]),
        se=bool(mc.get("se", False)),
        history_steps=int(mc.get("history_steps", 2)),
    )

    print(
        f"[export] 模型配置: blocks={cfg.blocks}, filters={cfg.filters}, "
        f"se={cfg.se}, history_steps={cfg.history_steps}, "
        f"C_in={cfg.in_channels}"
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] 参数量: {n_params:,}")

    # 随机验证输入（多批次以覆盖动态 batch）
    sample = _make_sample_input(cfg, batch_size=args.batch_size)

    all_ok = True

    # TorchScript 导出
    if args.torchscript:
        ts_path = out_dir / f"{stem}.ts"
        traced = export_torchscript(model, sample, ts_path)
        ok = verify_torchscript(model, traced, sample, atol=args.atol)
        all_ok = all_ok and ok

    # ONNX 导出
    if args.onnx:
        onnx_path = out_dir / f"{stem}.onnx"
        export_onnx(model, cfg, sample, onnx_path, opset_version=args.opset)
        ok = verify_onnx(model, onnx_path, sample, atol=args.atol)
        all_ok = all_ok and ok

    if all_ok:
        print("[export] 全部验证通过。")
        return 0
    else:
        print("[export] 存在验证失败，请检查上方输出。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
