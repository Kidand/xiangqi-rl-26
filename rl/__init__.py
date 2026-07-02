"""rl 包公共 API 导出。

训练系统子模块：
  - config    : 配置数据类
  - encoding  : 局面张量编码 / 动作索引转换
  - model     : 策略价值网络（ResNet）
"""

from rl.config import (
    ModelConfig,
    MCTSConfig,
    SelfPlayConfig,
    TrainConfig,
    ArenaConfig,
    LogConfig,
    Config,
)
from rl.encoding import (
    encode_board,
    move_to_policy_index,
    policy_index_to_move,
    legal_move_mask,
)
from rl.model import (
    XiangqiNet,
    create_model,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    # 配置
    "ModelConfig",
    "MCTSConfig",
    "SelfPlayConfig",
    "TrainConfig",
    "ArenaConfig",
    "LogConfig",
    "Config",
    # 编码
    "encode_board",
    "move_to_policy_index",
    "policy_index_to_move",
    "legal_move_mask",
    # 模型
    "XiangqiNet",
    "create_model",
    "load_checkpoint",
    "save_checkpoint",
]
