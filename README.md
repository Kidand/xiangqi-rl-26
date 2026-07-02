# 中国象棋 AlphaZero 式强化学习系统

纯自我博弈、无人类数据、端到端可复现的中国象棋 AlphaZero 实现。云端 8×H100 训练，笔记本 CPU/核显推理与 Web GUI 对战。

## 特性

- **自我博弈训练**：PUCT MCTS + ResNet 策略价值网络，从初始局面零知识开始学习
- **启发式加速**：材料价值混合、和棋惩罚、认输机制、一步杀捷径
- **完整规则引擎**：象棋全规则（马腿、象眼、士宫、帅对脸、兵规则、炮打、困毙即负、长将判负、重复判和）
- **本地 GUI**：Web 界面人机对战、观战、复盘、FEN 导入导出、着法回放
- **云端容易部署**：CUDA 12.8 自动配置脚本，tmux 日志监控，续训支持
- **模型导出**：TorchScript + ONNX 跨平台推理，笔记本可离线使用

## 目录结构

```
xiangqi-rl-26/
├── DESIGN.md                  # 全项目设计契约（必读）
├── README.md                  # 本文档
├── pyproject.toml
├── configs/                   # YAML 训练预设
│   ├── smoke.yaml            # CPU 冒烟（2-3 分钟）
│   ├── laptop.yaml           # 笔记本 CPU（迭代耗时 10-30 min）
│   └── cloud_8xh100.yaml     # 云端正式（迭代耗时 30-60 min）
├── xiangqi/                   # 规则引擎（零 torch 依赖）
│   ├── board.py              # 局面、走子、终局判定
│   ├── fen.py                # FEN 解析、UCCI 着法
│   └── notation.py           # 中文着法（炮二平五）
├── rl/                        # 训练系统（依赖 torch）
│   ├── config.py             # 配置体系
│   ├── encoding.py           # 局面→张量、动作转换
│   ├── model.py              # ResNet 策略价值网络
│   ├── mcts.py               # PUCT 蒙特卡洛树搜索
│   ├── selfplay.py           # 自我博弈 worker（多进程、多 GPU）
│   ├── replay_buffer.py      # 经验回放（磁盘持久化）
│   ├── heuristics.py         # 启发式（材料混合、和棋惩罚等）
│   ├── evaluate.py           # Arena 对抗门控（新 vs best）
│   ├── logger.py             # 日志系统（控制台 + 文件 + TensorBoard）
│   └── train.py              # 主训练循环（入口）
├── gui/                       # Web 对战界面
│   ├── server.py             # FastAPI 后端
│   └── web/                  # HTML/JS/CSS（无构建工具）
├── scripts/                   # 辅助脚本
│   ├── setup_env.sh          # 云端 Linux 环境配置
│   ├── setup_env_local.ps1   # 本地 Windows 配置
│   ├── train_cloud.sh        # 云端训练启动
│   ├── smoke_test.py         # 端到端冒烟验证
│   ├── bench_engine.py       # 引擎性能基准
│   └── export_model.py       # 导出 TorchScript/ONNX
├── records/                   # 对局记录（selfplay/ 与 gui/ 子目录）
├── ckpts/                    # 模型检查点（best.pt, iter_XXXX.pt）
├── logs/                     # 训练日志（train.log, metrics.csv, tb/）
└── tests/                    # pytest
```

## 本地快速开始

### 1. 环境配置（Windows）

```powershell
# PowerShell 中运行
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\scripts\setup_env_local.ps1
```

完成后虚拟环境 `.venv` 已激活，PyTorch CPU 版本已安装。

### 2. 启动 Web GUI

```bash
# 激活虚拟环境（若未激活）
.\.venv\Scripts\Activate.ps1

# 启动服务
python -m gui.server --port 8000
```

打开浏览器访问 `http://127.0.0.1:8000`。

### 3. 加载预训练模型（可选）

GUI 中点击「模型」→「加载」，选择 `ckpts/best.pt`（若存在）。若无模型，仍可进行人机互动（hvh=人vs人，hvh 无需 AI）。

### 4. 对战操作

#### 新对局

1. 点击「新对局」
2. 选择模式：
   - **hva**（人 vs AI）：选择执子方（红/黑）、难度（sims 档位）
   - **ava**（AI vs AI）：观看两个 AI 对战
   - **hvh**（人 vs 人）：本地两人对战
3. 可选：导入自定义 FEN 开局

#### 走子

- 点击棋子选中（高亮绿色）
- 点击合法落点（绿圆点）走子
- AI 自动回应（hva 模式）

#### 辅助功能

- **评估条**：实时显示局面评估值（红方视角，绿色=优势）
- **提示**：点击「提示」获得 AI 推荐着法（Top 5 + 箭头指示）
- **悔棋**：撤销最后一回合（hva 撤销两步）
- **认输**：主动认输
- **翻转棋盘**：调整视角
- **着法列表**：点击跳转到该步局面

### 5. 复盘与导入

#### 导入对局

点击「复盘」→「导入」，支持三种格式：

**格式 1：xiangqi-record-v1 JSON**
```json
{
  "format": "xiangqi-record-v1",
  "start_fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
  "moves": ["h2e2", "h9g7"],
  "result": "1-0",
  "termination": "checkmate"
}
```

**格式 2：纯文本**
```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
h2e2 h9g7 e2e9
```
或省略初始 FEN（默认标准开局）：
```
h2e2 h9g7 e2e9
```

#### 浏览对局库

点击「复盘」→「浏览库」，查看 `records/` 下的自我博弈对局（按迭代号组织）。

#### 播放控制

- `|< < > >|` 导航
- 播放/暂停、速度调节
- 点击着法列表跳转

#### 导出对局

点击「导出」下载当前对局为 JSON 文件。

## 云端训练全流程

### 云端环境（/home/jovyan/LLM/yanjz/projects/xiangqi-rl-26）

#### 初次配置

```bash
# 1. 克隆或进入项目目录
cd /home/jovyan/LLM/yanjz/projects/xiangqi-rl-26

# 2. 运行环境配置脚本（首次或重新配置时）
bash scripts/setup_env.sh
# 脚本会：
#   - 检查/安装 uv
#   - 创建虚拨虚拟环境（Python 3.11）
#   - 安装 PyTorch CUDA 12.8（支持 H100）
#   - 安装其他依赖（numpy, pyyaml, fastapi, tensorboard 等）
#   - 验证环境（CUDA 可用性、GPU 数量）
```

#### 开始训练

```bash
# 3. 启动 tmux 会话
tmux new-session -s xiangqi

# 4. 在 tmux 中启动训练
cd /home/jovyan/LLM/yanjz/projects/xiangqi-rl-26
bash scripts/train_cloud.sh
# 或自定义配置：
# bash scripts/train_cloud.sh --config configs/custom.yaml

# 5. 在另一个终端观察日志
tail -f /home/jovyan/LLM/yanjz/projects/xiangqi-rl-26/logs/train.log

# 6. TensorBoard 实时监控（可选，在有图形界面的环境）
tensorboard --logdir /home/jovyan/LLM/yanjz/projects/xiangqi-rl-26/logs/tb --port 6006
```

#### 续训（中断恢复）

```bash
# 在 tmux 中，Ctrl+C 停止训练后
bash scripts/train_cloud.sh --resume
# 脚本自动从最新 checkpoint 和 buffer 续训
```

#### 监控训练进度

**实时统计**（每 30 秒更新一次，汇总所有 worker）：
```
[iter 12 | selfplay] games 356/2000 (17.8%) | R/B/D 142/128/86 | 红胜率 39.9% 黑胜率 36.0% 和率 24.2%
  | avg_plies 121.4 | moves/s 84.2 | nn_evals/s 21030 | resign 31.5% | buffer 812k | elapsed 06:12
```

**训练阶段**（每 100 步）：
```
[iter 12 train] step 234/1000 | loss 0.234 (policy 0.156 + value 0.078) | entropy 4.21 | value_mae 0.12
  | lr 8.9e-4 | samples/s 18234 | grad_norm 1.23
```

**Arena 评测**（迭代结束时）：
```
[iter 12 arena] new vs best: 24W 6D 10L (score 67.5%) | 执红 14W2D4L 执黑 10W4D6L | GATE: PROMOTED ✓
```

**指标汇总行**写入 `logs/metrics.csv`，可用 Excel 或 Python 绘图分析。

#### 导出模型

```bash
# 训练完成后
python scripts/export_model.py ckpts/best.pt --torchscript --onnx
# 生成 best.ts（TorchScript）和 best.onnx（ONNX）
```

#### 拷回笔记本

```bash
# 笔记本上（sftp/scp 或云存储）
scp -r /path/to/remote/ckpts/best.pt ~/xiangqi-rl-26/ckpts/
scp -r /path/to/remote/ckpts/best.onnx ~/xiangqi-rl-26/ckpts/
```

然后笔记本 GUI 加载 `ckpts/best.pt` 即可推理。

> **迁移提示**：检查点统一存放于 `ckpts/`；如需从旧训练目录续训，把历史检查点整理进 `ckpts/` 后 `--resume` 可无缝续训。

## 配置说明

### 三个预设对比

| 预设 | 场景 | 迭代耗时 | 棋力 | 适用 |
|------|------|---------|------|------|
| **smoke.yaml** | CPU 端到端验证 | 2-3 分钟 | 极弱 | 代码检查、ci/cd |
| **laptop.yaml** | 本地试训（收敛趋势） | 10-30 min/iter | 弱（但可看趋势） | 参数调试、本地开发 |
| **cloud_8xh100.yaml** | 云端正式 | 30-60 min/iter | 强（与强 AI 竞争级别） | 实际训练 |

#### smoke.yaml（冒烟测试）

```yaml
model: {blocks: 2, filters: 32}          # 2 个残差块，32 通道，~0.1M 参数
mcts: {num_sims: 24}                     # 少量模拟
selfplay: {games_per_iteration: 4}       # 仅 4 盘对局
train: {device: cpu, total_iterations: 2} # CPU，2 个迭代
```

用途：验证代码正确性、环境配置、完整流水线。

#### laptop.yaml（笔记本试训）

```yaml
model: {blocks: 5, filters: 64}          # 5 个残差块，64 通道，~0.5M 参数
mcts: {num_sims: 120}                    # 中等模拟
selfplay: {games_per_iteration: 40}      # 40 盘对局
train: {device: cpu, batch_size: 256, total_iterations: 50}
```

用途：本地调参、观察收敛趋势、学习过程。预期棋力有限（不与云端模型对齐），但可看网络学习的曲线。

#### cloud_8xh100.yaml（云端正式）

```yaml
model: {blocks: 10, filters: 128}        # 10 个残差块，128 通道，~7M 参数
mcts: {num_sims: 600}                    # 大量模拟（覆盖更多变化）
selfplay:
  games_per_iteration: 2000              # 2000 盘对局
  num_workers: 0                         # 0 = 自动 max(8, cpu_count - num_gpus - 4)
  eval_max_batch: 2048                   # EvalServer 单次前向 batch 上限
  eval_wait_ms: 2.0                      # 聚合请求的等待窗口（毫秒）
  eval_fp16: true                        # cuda 上 bf16 autocast 推理
train:
  device: cuda
  num_gpus: 8                            # 分布式（DDP 或 DataParallel）
  batch_size: 4096                       # 大 batch
  total_iterations: 300                  # 300 个迭代
```

用途：正式训练，充分利用 8×H100 算力，预期达到竞争级棋力。

### 关键超参调整指南

根据训练日志（`logs/metrics.csv`）调参：

#### 1. **红黑胜率不均衡** → 检查视角翻转

症状：红黑胜率长期五五开，和率极低（< 5%）。

原因：常见的 MCTS 视角翻转 bug（黑方 move mask 未正确应用）。

检查：
- `encoding.py` 的 `legal_move_mask()` 是否仅在黑方时翻转
- `mcts.py` 的 `apply_results()` 是否正确处理策略概率的 mask 与视角

#### 2. **和率过高** → 增加进取性

症状：和率 > 50%，红黑胜率都较低。

对策：
- 降低 `draw_penalty`（默认 -0.1），如改为 -0.2 或 -0.3。该值既作为训练 z 目标的和棋惩罚，
  也会被注入 MCTS 作为**树内和棋终局回传值**（见下文「树内和棋惩罚」），双管齐下抑制塌缩。
- 开启 `mcts.temp_final`（默认 0=关），如改为 0.25：温度期后不再纯 argmax，而是按访问计数
  轻度采样，打破确定性循环、抑制重复判和（见下文「温度期后采样」）。
- 提早启用认输 `resign_start_iter`（默认 10），改为 5
- 检查是否进入随机走子的循环（MCTS 搜索不足）

#### 3. **学习曲线停滞** → 调整材料混合

症状：loss 下降缓慢，value_mae 居高不下（> 0.3）。

对策：
- 延长 `value_blend_iters`（默认 30），如改为 50-100
- 增加初始 `value_blend_init`（默认 0.5），如改为 0.7
- 查看 `resign_fp_rate`（认输误判率）是否过高（>5%），若是降低 `resign_threshold`

#### 4. **内存溢出或 OOM** → 减少并发

症状：训练中 GPU 报 out of memory。

对策：
- 减少 `batch_size`（默认 4096），如改为 2048
- 减少 `eval_max_batch`（默认 2048），如改为 1024（减少 EvalServer 单次前向显存占用）
- 减少 `games_per_worker`（默认 16），如改为 8

#### 5. **吞吐过低** → 增加并发

症状：`games/hour` < 1000（预期 2000-5000）。

对策：
- 增加 `num_workers`（默认 0=自动，手动指定如 64+，按 CPU 核数扩展）
- 日志中观察 `evals/s`（EvalServer 每秒评估数）和 `avg_batch`（平均 batch 大小）：
  - `avg_batch` 过低（< 64）→ 增加 `num_workers` 或调大 `eval_wait_ms`（延长聚合窗口）
  - `evals/s` 瓶颈 → 检查 GPU 利用率，适当增大 `eval_max_batch`
- 检查 I/O 瓶颈：buffer 写盘是否阻塞（看 `moves/s`）
- 确认无 GPU 显存溢出导致的降速

> **注**：`workers_per_gpu` 已弃用（新架构按 CPU 核数而非 GPU 数扩展 worker），保留该字段仅为兼容旧 YAML，新配置请改用 `num_workers`。

#### 7. **三项训练优化参数**（收敛加速）

以下三个参数用于抑制和棋塌缩、加速小 buffer 期收敛：

- **温度期后采样（`mcts.temp_final`，默认 0.0）**：`ply >= temp_moves`（温度期结束）后的走子选择。
  为 0 时 argmax（旧行为）；> 0 时按 `N^(1/temp_final)` 归一采样（`temp_final` 越小越接近
  argmax）。推荐 0.25 左右，用于打破确定性循环、抑制重复判和。开局多样化与温度期逻辑不变；
  **arena 不受影响**（arena 仍是 `arena_temp_moves` 之后 argmax）。

- **树内和棋惩罚（`draw_value` 注入机制）**：`MCTSConfig.draw_value`（默认 0.0）是 MCTS 树内
  和棋终局的回传值。selfplay worker 与 arena 在构造搜索树前，用 `dataclasses.replace(cfg.mcts,
  draw_value=cfg.selfplay.draw_penalty)` 注入——**单一来源为 `selfplay.draw_penalty`**，避免两处
  配置漂移。和棋回传时**不翻符号**（双方同罚），与逐层取负的胜负回传区分。GUI 不注入（默认 0）。
  效果：搜索阶段就把和棋当作（轻微）负收益，与训练 z 目标一致，减少 AI 主动求和。

- **自动训练步数（`train.train_steps_per_iteration=0` + `train.sample_reuse`）**：
  `train_steps_per_iteration > 0` 时为固定步数（默认 1000）；**为 0 时自动**：
  `clamp(ceil(本迭代新样本数 × sample_reuse / batch_size), 10, 5000)`。`sample_reuse`（默认 3.0）
  是每个新样本的期望学习次数。自动模式可防止小 buffer 期步数固定导致每样本被重复学习数十遍而
  过拟合；日志每迭代打印一行「本迭代新样本 X，训练步数 Y（自动/固定）」。

#### 6. **entropy 过低** → 增加温度或噪声

症状：策略 entropy < 2.0，AI 走子过于确定性，容易陷入局部最优。

对策：
- 增加 `opening_temperature`（默认 1.25），如改为 1.5-2.0
- 增加 Dirichlet `dirichlet_alpha`（默认 0.2），如改为 0.5
- 检查根节点噪声 `dirichlet_eps` 是否为 0（训练时应为 0.25）

## 训练日志字段解释

详见 `logs/train.log` 和 `logs/metrics.csv`。

### train.log（实时日志）

**Self-play 聚合行**（每 30s）：
```
[iter 12 | selfplay] games 356/2000 (17.8%) | R/B/D 142/128/86 | 红胜率 39.9% 黑胜率 36.0% 和率 24.2%
  | avg_plies 121.4 | moves/s 84.2 | nn_evals/s 21030 | resign 31.5% | buffer 812k | elapsed 06:12
```

| 字段 | 含义 |
|------|------|
| `iter 12` | 第 12 个迭代 |
| `games 356/2000` | 已完成 356 盘，目标 2000 |
| `R/B/D 142/128/86` | 红胜 142 盘、黑胜 128 盘、和棋 86 盘 |
| `红胜率 39.9%` | 红胜 / (红胜 + 黑胜 + 和) = 红方胜率 |
| `黑胜率 36.0%` | 黑方胜率 |
| `和率 24.2%` | 和棋比例 |
| `avg_plies 121.4` | 平均对局长度（半回合数） |
| `moves/s 84.2` | 每秒走子数（反映 CPU 走子生成速度） |
| `nn_evals/s 21030` | 每秒网络评估数（GPU 吞吐） |
| `resign 31.5%` | 启用认输的对局比例 |
| `buffer 812k` | 回放缓冲样本数（850k = 850,000） |
| `elapsed 06:12` | 本迭代已耗时 |

**训练阶段行**（每 100 步）：
```
[iter 12 train] step 234/1000 | loss 0.234 (policy 0.156 + value 0.078) | entropy 4.21 | value_mae 0.12
  | lr 8.9e-4 | samples/s 18234 | grad_norm 1.23
```

| 字段 | 含义 |
|------|------|
| `step 234/1000` | 当前 training step / 目标 steps |
| `loss 0.234` | 总损失（策略+价值+L2 正则） |
| `policy 0.156` | 策略交叉熵损失 |
| `value 0.078` | 价值 MSE 损失 |
| `entropy 4.21` | 策略输出熵（越高越多样；过低表示过度确定） |
| `value_mae 0.12` | 价值 MAE（0~1 范围，越小越准） |
| `lr 8.9e-4` | 当前学习率（余弦退火） |
| `samples/s 18234` | 每秒处理样本数 |
| `grad_norm 1.23` | 梯度范数（过大可能导致不稳定） |

**Arena 最终行**（迭代末）：
```
[iter 12 arena] new vs best: 24W 6D 10L (score 67.5%) | 执红 14W2D4L 执黑 10W4D6L | GATE: PROMOTED ✓
```

| 字段 | 含义 |
|------|------|
| `new vs best` | 新模型 vs 历史最佳 |
| `24W 6D 10L` | 新模型胜 24、和 6、负 10（总 40 盘） |
| `score 67.5%` | 新模型得分 = (W + 0.5×D) / 总盘数 |
| `执红 14W2D4L` | 新模型执红时表现 |
| `执黑 10W4D6L` | 新模型执黑时表现 |
| `GATE: PROMOTED ✓` | 新模型超过 gate_threshold(55%)，晋升为 best；否则 REJECTED |

### metrics.csv

汇总表，每迭代一行。列名固定：

| 列名 | 含义 |
|------|------|
| `iter` | 迭代号 |
| `games` | 完成对局数 |
| `red_win, black_win, draw` | 各类胜负数 |
| `red_winrate, black_winrate, draw_rate` | 各类比例 |
| `avg_plies` | 平均对局长度 |
| `resign_rate` | 认输比例 |
| `resign_fp_rate` | 认输误判率（翻盘对局 / 认输对局） |
| `buffer_size` | 缓冲样本数 |
| `loss, policy_loss, value_loss` | 损失（总、策略、价值） |
| `entropy, value_mae` | 指标 |
| `lr` | 学习率 |
| `arena_score, arena_red_score, arena_black_score` | Arena 评分 |
| `promoted` | 1=晋升，0=未晋升 |
| `selfplay_sec, train_sec, total_sec` | 耗时统计 |

用途：用 pandas/Excel 绘制曲线，观察收敛趋势。

## FEN 与对局记录格式

### FEN（Xiangqi Forsyth-Edwards Notation）

标准中国象棋 FEN，兼容 UCCI。

```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
```

**字段**（空格分隔）：

1. **棋盘**：从 row 9（黑底线）到 row 0，每行 col 0→8，数字表示连续空格，`/` 分行
   - 红棋大写：`K(帅) A(仕) B(相) N(马) R(车) C(炮) P(兵)`
   - 黑棋小写：对应小写
   - 兼容别名：`E/e(象) H/h(马)`，输出统一用 `B/N`

2. **走子方**：`w`(红) 或 `b`(黑)，兼容输入 `r`，输出用 `w`

3. **易位与吃过路**：象棋无此概念，固定 `-`

4. **未吃子半回合计数**：0-120（用于 60 回合限着规则），字符 `-` 表示 0

5. **整数半回合计数**：从 0 起

6. **整数全回合数**：从 1 起

### UCCI 着法

4 字符，起点+终点：

```
h2e2  # 红方炮从 h2 走到 e2（平移）
h9g7  # 黑方马从 h9 走到 g7
```

列 a-i，行 0-9。坐标（row, col）映射见 `xiangqi/constants.py`。

### 对局记录（xiangqi-record-v1）

JSON 格式，一局一条（`.json` 单局或 `.jsonl` 多局）。

```json
{
  "format": "xiangqi-record-v1",
  "start_fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
  "moves": ["h2e2", "h9g7", "e2e9"],
  "result": "1-0",
  "termination": "checkmate",
  "plies": 87,
  "red": "selfplay-iter12",
  "black": "selfplay-iter12",
  "meta": {
    "iteration": 12,
    "sims": 400,
    "time": "2026-07-02T10:00:00"
  }
}
```

**字段**：

| 字段 | 含义 |
|------|------|
| `format` | 版本标识，固定 `xiangqi-record-v1` |
| `start_fen` | 开局 FEN（或 null=标准开局） |
| `moves` | UCCI 着法数组 |
| `result` | `"1-0"`(红胜) / `"0-1"`(黑胜) / `"1/2-1/2"`(和) |
| `termination` | 终局方式：`"checkmate"`, `"stalemate"`, `"resignation"`, `"repetition"`, `"perpetual_check"`, `"no_capture"`, `"max_moves"` |
| `plies` | 总半回合数 |
| `red`, `black` | 执子者描述（如 `"selfplay-iter12"`, `"ai-human"`) |
| `meta` | 可选元数据（迭代、模拟数、时间戳等） |

**可选字段**（GUI 导出时附加）：

```json
{
  ...
  "fens": ["起始 FEN", "第 1 着后 FEN", ...]
}
```

允许快速回放无需重新执行着法。

## FAQ

### Q: 本地笔记本可以单独运行 GUI 吗？

**A:** 可以。GUI 支持三种模式：

1. **hvh**（人 vs 人）：无需 AI/模型，纯本地对战
2. **hva/ava**（需模型）：加载 `ckpts/best.pt`（从云端拷回）后可用

若无模型，hvh 仍完全可用。

### Q: smoke_test.py 怎么跑？

**A:** 
```bash
python scripts/smoke_test.py
```

应在 < 5 分钟内完成，验证：
- 4 盘 selfplay（100 sims）
- 50 步训练
- 4 盘 arena 对抗

退出码 0 = 通过。若失败，检查：
- Python 版本 >= 3.9
- numpy, torch 版本
- 是否有 NaN/Inf 导致的数值错误

### Q: 怎样导出模型给笔记本？

**A:**
```bash
# 云端
python scripts/export_model.py ckpts/best.pt --torchscript --onnx

# 拷回笔记本
scp best.pt your-laptop:~/xiangqi-rl-26/ckpts/
scp best.onnx your-laptop:~/xiangqi-rl-26/ckpts/

# 笔记本 GUI 加载
# POST /api/model/load {"path": "ckpts/best.pt"}
```

支持 TorchScript（`.pt`）和 ONNX（`.onnx`，推理加速，需装 onnxruntime）。

### Q: 如何调试长将判负逻辑？

**A:** 检查 `xiangqi/board.py` 的 `_is_perpetual_check()` 方法。单元测试见 `tests/test_rules.py`。

关键点：
- 重复周期内该方每着都是将军 → 判负
- 否则判和（棋力不足的防守）

### Q: Arena 门控的 gate_threshold 怎么选？

**A:** 默认 0.55（胜率 55% = 优势但不压倒）。

- **激进**（0.50）：任何赢面即晋升，快速迭代，可能波动
- **保守**（0.60）：需明显领先，确保稳定进步，迭代偏慢

通常 0.55-0.60 较平衡。看 metrics.csv 的 promoted 列，若经常被拒，可下调至 0.52。

### Q: 怎样在 tmux 外观察日志？

**A:**
```bash
# 终端 1：启动训练
tmux new -s train
bash scripts/train_cloud.sh

# 终端 2：实时查看日志
tail -f logs/train.log

# 终端 3：监控 metrics
watch -n 10 'tail -5 logs/metrics.csv'

# TensorBoard（若有图形环境）
tensorboard --logdir logs/tb --port 6006
```

### Q: 云端突然断开连接，模型丢了怗？

**A:** 不会。训练脚本每迭代末自动保存：

- `ckpts/best.pt`：历史最佳（Arena 晋升）
- `ckpts/iter_XXXX.pt`：最新迭代（中间存档）
- `rl/replay_buffer/` 下 `.npz` 文件：经验缓冲

重新连接后 `bash scripts/train_cloud.sh --resume` 从最新 checkpoint 续训，buffer 自动加载。

### Q: 本地 Windows 虚拟环境激活失败？

**A:** 检查执行策略：
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope CurrentUser
```

或直接调用 Python：
```powershell
.\.venv\Scripts\python.exe -m rl.train --config configs/smoke.yaml
```

### Q: 推理时 CPU 和 ONNX 推理速度差多少？

**A:** 机器学习的推理速度取决于硬件与模型。一般：

- **TorchScript**：与原模型等价，慢 5-10%（JIT 编译）
- **ONNX + onnxruntime**：快 20-50%（针对 CPU 优化）

笔记本 CPU 上预期：`onnxruntime` > `TorchScript` > 原 PyTorch。

### Q: 能否换其他优化器（如 SGD）？

**A:** 可以，修改 `configs/xxx.yaml`：

```yaml
train:
  optimizer: sgd          # 默认 adamw
  lr: 0.1                 # 典型 SGD 学习率
  momentum: 0.9
  weight_decay: 1e-4
```

SGD+Momentum 常见于象棋 RL，可能收敛更稳定，但需要调参。

### Q: 遇到 NaN loss 怎么办？

**A:** 常见原因与解决：

1. **学习率过高**：降低 `train.lr`（默认 0.001），如改为 0.0005
2. **价值目标异常**：检查 `heuristics.py` 的材料价值计算，或禁用 `value_blend`
3. **梯度爆炸**：增加 `grad_clip`（默认 5.0），如改为 1.0
4. **网络初始化**：确保 `model.py` 的权重初始化正确（一般框架默认即可）

检查 `logs/train.log` 何时首次出现 NaN，倒推条件并微调。

## 参考

- **完整设计文档**：见 `DESIGN.md`（全项目契约）
- **配置详解**：见 `rl/config.py` 及各预设 YAML
- **测试用例**：见 `tests/` 目录，覆盖规则、编码、MCTS、模型
- **API 文档**：见 `DESIGN.md` §13（GUI FastAPI）

---

**作者与维护**：AlphaZero 式中国象棋强化学习系统  
**最后更新**：2026-07-02  
**许可证**：见 pyproject.toml（若有）
