"""训练配置体系（契约，见 DESIGN.md §7–§11）。YAML 预设位于 configs/。

用法:
    cfg = Config.from_yaml("configs/cloud.yaml")
    cfg = Config()  # 全默认
YAML 顶层键与各子配置字段同名，未出现的字段用默认值；未知键报错（防拼写错误静默失效）。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    blocks: int = 10            # 残差块数
    filters: int = 128          # 通道数
    se: bool = False            # Squeeze-Excitation
    history_steps: int = 2      # 输入历史步数 T，C_in = 14*T + 3

    @property
    def in_channels(self) -> int:
        return 14 * self.history_steps + 3


@dataclass
class MCTSConfig:
    num_sims: int = 400
    c_puct: float = 1.8
    dirichlet_alpha: float = 0.2
    dirichlet_eps: float = 0.25
    fpu_reduction: float = 0.2       # 未访问子节点 Q = parent_Q - fpu_reduction
    virtual_loss: float = 1.0
    temp_moves: int = 24             # 前 N ply 温度采样
    temperature: float = 1.0
    temp_final: float = 0.0          # temp_moves 之后的采样温度；0 = argmax（旧行为）
    draw_value: float = 0.0          # 树内和棋终局回传值（不翻符号，双方同罚）；
                                     # selfplay/arena 会注入 selfplay.draw_penalty
    batch_size: int = 64             # 单进程叶子评估凑批上限
    # π 训练目标锐化（DESIGN §8）：只作用于训练样本 π 的构造，不影响落子采样
    # 与 visit_counts 语义。默认值 = 原行为（不锐化）。
    policy_target_temp: float = 1.0       # 目标温度 T，p ∝ c^(1/T)；<1 锐化
    policy_target_prune_frac: float = 0.0 # 剪掉 visits < frac*max 的边再归一；0=关
    # num_sims 按迭代调度（DESIGN §8）：iteration ≥ late_start_iter 时改用 num_sims_late
    #（>0 时生效），用于收尾阶段灌高质量 π buffer（蒸馏热启动数据）。
    num_sims_late: int = 0
    num_sims_late_start_iter: int = 0
    # Gumbel 训练搜索（DESIGN §8）：仅 selfplay 训练路径（add_noise=True）生效；
    # arena/GUI 恒走 PUCT。"gumbel" 下 Dirichlet/温度采样/π 锐化均不使用。
    algorithm: str = "puct"          # "puct" | "gumbel"
    gumbel_m: int = 16               # 根候选数（Gumbel-Top-m）
    gumbel_c_visit: float = 50.0     # σ(q̂) 变换的 visit 项
    gumbel_c_scale: float = 1.0      # σ(q̂) 变换缩放


@dataclass
class SelfPlayConfig:
    games_per_iteration: int = 2000
    # 推理服务架构（DESIGN §9）：worker 为纯 CPU 进程，数量按 CPU 核数扩展
    num_workers: int = 0             # 0 = 自动 max(8, cpu_count - num_gpus - 4)
    workers_per_gpu: int = 4         # [已弃用] 仅为兼容旧 YAML 保留，新架构不使用
    games_per_worker: int = 16       # 每 worker 并发对局数
    # EvalServer（每 GPU 一个批量推理服务进程）
    eval_max_batch: int = 2048       # 单次前向 batch 上限
    eval_wait_ms: float = 2.0        # 聚合请求的等待窗口（毫秒）
    eval_fp16: bool = True           # cuda 上用 bf16 autocast 推理
    max_game_plies: int = 400
    no_capture_draw_plies: int = 120
    # 启发式（DESIGN §10）
    value_blend_init: float = 0.5    # 材料混合初始 λ
    value_blend_iters: int = 30      # λ 线性退火到 0 的迭代数
    draw_penalty: float = -0.1       # 和棋 z 值（双方）
    resign_enabled: bool = True
    resign_start_iter: int = 10
    resign_threshold: float = -0.92
    resign_consecutive: int = 4
    resign_disable_frac: float = 0.1  # 禁用认输校准对局比例
    opening_random_plies: int = 8
    opening_temperature: float = 1.25
    mate_shortcut: bool = True       # 一步杀捷径
    # z 目标与 MCTS 根价值混合（DESIGN §10）：z' = (1-β)·blend(z,材料,λ) + β·root_value，
    # root_value 为该样本落子前搜索根价值（当前走子方视角，与 z 同视角无需翻转）。
    # 压低对等对局中 ±1 结果的方差。0 = 关（原行为）。
    root_value_weight: float = 0.0


@dataclass
class TrainConfig:
    device: str = "cuda"             # "cuda" / "cpu"
    num_gpus: int = 8
    batch_size: int = 4096
    train_steps_per_iteration: int = 1000  # 0 = 自动：clamp(ceil(新样本×sample_reuse/batch), 10, 5000)
    sample_reuse: float = 3.0        # 自动步数下每个新样本的期望学习次数
    buffer_window: int = 1_500_000   # 保留最近样本数
    min_buffer_to_train: int = 20_000
    optimizer: str = "adamw"         # "adamw" / "sgd"
    lr: float = 1e-3
    lr_min: float = 1e-4             # 余弦退火下限
    weight_decay: float = 1e-4
    momentum: float = 0.9            # sgd 用
    value_loss_weight: float = 1.0
    grad_clip: float = 5.0
    total_iterations: int = 200
    checkpoint_dir: str = "ckpts"
    amp: bool = True                 # 混合精度（bf16）
    # buffer 落盘策略：snapshot=快照+后台线程（默认，save 隐藏进下一迭代，RAM 峰值
    # +1 份 states 拷贝）；freeze_window=对只读活 buffer 起线程（零额外 RAM，只藏进
    # train+arena）；sync=同步保存（原行为）。
    async_buffer_save: str = "snapshot"


@dataclass
class ArenaConfig:
    arena_games: int = 40
    arena_sims: int = 200
    gate_threshold: float = 0.55     # 胜率（和=0.5）≥ 此值则晋升 best
    # LCB 门控（DESIGN §9）：>0 时额外要求 score - z·score_se ≥ 0.5（score_se 为
    # 逐局得分 {0,0.5,1} 的样本标准误），抑制假晋升导致的 best 随机漂移。0 = 关。
    gate_lcb_z: float = 0.0
    arena_temp_moves: int = 8        # 前几步轻微随机避免完全重复对局
    arena_temperature: float = 0.5
    arena_workers: int = 0           # 并行对弈进程数；0=自动（cuda 且 ≥16 盘时 GPU 数×4），1=串行


@dataclass
class LogConfig:
    log_dir: str = "logs"
    records_dir: str = "records"
    log_interval_sec: float = 30.0   # selfplay 实时统计间隔
    train_log_interval: int = 100    # 每 N 训练步打印
    tensorboard: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        import yaml

        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = cls()
        for section, values in raw.items():
            if section == "seed":
                cfg.seed = int(values)
                continue
            if not hasattr(cfg, section):
                raise KeyError(f"unknown config section: {section}")
            sub = getattr(cfg, section)
            if not dataclasses.is_dataclass(sub):
                raise KeyError(f"not a config section: {section}")
            valid = {f.name for f in dataclasses.fields(sub)}
            for k, v in (values or {}).items():
                if k not in valid:
                    raise KeyError(f"unknown key {section}.{k}")
                setattr(sub, k, v)
        # algorithm 拼写校验：mcts 与 selfplay 两处判据均为精确匹配 "gumbel"，
        # 若不在白名单会全局静默退回 PUCT——在加载期直接报错（防"看似在跑 gumbel"）。
        if cfg.mcts.algorithm not in ("puct", "gumbel"):
            raise ValueError(
                f"mcts.algorithm={cfg.mcts.algorithm!r} 不合法，可选 'puct'/'gumbel'（区分大小写）"
            )
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
