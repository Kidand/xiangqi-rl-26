"""训练样本的左右镜像增强（DESIGN.md §6）。"""

from __future__ import annotations

import numpy as np

from xiangqi.constants import ACTION_SIZE, COLS, NUM_SQUARES, ROWS


# action = from_sq * NUM_SQUARES + to_sq；左右镜像是对合，所以同一映射
# 既可把原动作送到镜像动作，也可从镜像输出位置反查原概率。
_MIRRORED_SQUARES = np.arange(NUM_SQUARES).reshape(ROWS, COLS)[:, ::-1].ravel()
_MIRRORED_ACTIONS = (
    _MIRRORED_SQUARES[:, None] * NUM_SQUARES + _MIRRORED_SQUARES[None, :]
).ravel()
_MIRRORED_ACTIONS.setflags(write=False)


def augment_horizontal(
    states: np.ndarray,
    policies: np.ndarray,
    probability: float,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """逐样本同步镜像所有输入平面与策略动作的起终点。

    ``states`` 为 ``(B, C, 10, 9)``，``policies`` 为 ``(B, 8100)``。
    左右镜像不交换红黑方，历史、上一着和颜色平面随棋盘一起翻转；
    价值目标不变，因此本函数不接收也不修改价值目标。

    ``probability`` 必须有限且在 [0, 1]。省略 ``rng`` 时使用
    ``np.random``，保持 ``np.random.seed`` 的可复现性；也可传入
    ``np.random.Generator`` 或 ``RandomState``。概率 0、1 和空批次
    不消耗随机数。概率 0 或空批次直接返回输入；其余情况返回保持
    dtype 的 C 连续副本，支持非连续和只读输入，始终不修改调用方数组。
    """
    try:
        probability = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability must be finite and in [0, 1]") from exc
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")

    states = np.asarray(states)
    policies = np.asarray(policies)
    if states.ndim != 4 or states.shape[1] < 1 or states.shape[2:] != (ROWS, COLS):
        raise ValueError(f"states must have shape (B, C, {ROWS}, {COLS}), C > 0")
    if policies.ndim != 2 or policies.shape != (states.shape[0], ACTION_SIZE):
        raise ValueError(f"policies must have shape (B, {ACTION_SIZE}) matching states")

    batch_size = states.shape[0]
    if probability == 0.0 or batch_size == 0:
        return states, policies

    if probability == 1.0:
        selected = np.arange(batch_size)
    else:
        random_source = np.random if rng is None else rng
        selected = np.flatnonzero(random_source.random(batch_size) < probability)

    augmented_states = states.copy(order="C")
    augmented_policies = policies.copy(order="C")
    if selected.size:
        augmented_states[selected] = states[selected, :, :, ::-1]
        augmented_policies[selected] = policies[np.ix_(selected, _MIRRORED_ACTIONS)]
    return augmented_states, augmented_policies
