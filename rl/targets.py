"""价值标签的版本化契约（DESIGN §9–§10）。

契约描述样本落盘时采用的标签生成规则，用于阻止续训混用不同价值尺度。
历史配置仍可记录负的和棋值；严格零和实验应显式使用 draw_penalty=0。
"""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.config import Config


def value_target_metadata(cfg: "Config") -> dict:
    """返回 JSON 可序列化的标签契约，先验证所有影响价值标签的参数。

    混合权重均须在 [0, 1]，和棋值须在 [-1, 1]，材料退火迭代数须为
    非负整数。验证在分配 replay buffer 或开始自博弈之前进行。
    """
    sp = cfg.selfplay

    def finite_number(name: str) -> float:
        raw = getattr(sp, name)
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(f"selfplay.{name} 必须为有限数值，得到 {raw!r}")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"selfplay.{name} 必须为有限数值，得到 {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"selfplay.{name} 必须为有限数值，得到 {raw!r}")
        return value

    draw = finite_number("draw_penalty")
    root_weight = finite_number("root_value_weight")
    blend_init = finite_number("value_blend_init")
    blend_iters = finite_number("value_blend_iters")
    if not -1.0 <= draw <= 1.0:
        raise ValueError(f"selfplay.draw_penalty 必须在 [-1, 1]，得到 {draw}")
    for name, value in (("root_value_weight", root_weight), ("value_blend_init", blend_init)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"selfplay.{name} 必须在 [0, 1]，得到 {value}")
    if blend_iters < 0 or not blend_iters.is_integer():
        raise ValueError(
            f"selfplay.value_blend_iters 必须为非负整数，得到 {blend_iters}"
        )

    return {
        "version": 1,
        "draw_penalty": draw,
        "root_value_weight": root_weight,
        "value_blend_init": blend_init,
        "value_blend_iters": int(blend_iters),
    }
