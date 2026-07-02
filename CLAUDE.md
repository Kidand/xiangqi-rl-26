# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目

AlphaZero 式中国象棋强化学习系统：纯自我博弈训练（无人类数据）+ Web GUI 对战/复盘。
**DESIGN.md 是全项目唯一契约**（坐标系、FEN、动作编码、网络输入、API、记录格式）——改任何跨模块接口必须先改 DESIGN.md，代码与它冲突时以它为准。

## 常用命令

本地 Windows 开发环境的解释器固定为 `.venv/Scripts/python.exe`（CPU torch）；云端 Linux 为 `.venv/bin/python`。

```bash
# 测试（slow 标记 = perft(3)、50 盘 fuzz 等长测试）
.venv/Scripts/python.exe -m pytest tests/ -q -m "not slow"
.venv/Scripts/python.exe -m pytest tests/ -q -m slow
.venv/Scripts/python.exe -m pytest tests/test_mcts.py -q          # 单文件
.venv/Scripts/python.exe -m pytest tests/test_rules.py -q -k horse  # 单用例

# 端到端冒烟（CPU 完整流水线：自博弈→训练→arena→记录/CSV 校验，~30s，改动 rl/ 后必跑）
.venv/Scripts/python.exe scripts/smoke_test.py

# 训练（云端 8×H100 用 cloud_8xh100.yaml；--resume 续训）
python -m rl.train --config configs/smoke.yaml

# GUI（浏览器开 http://127.0.0.1:8000；hvh 与复盘不需要模型）
.venv/Scripts/python.exe -m gui.server --model ckpts/best.pt --port 8000 --device cpu

# 引擎基准 / 模型导出
.venv/Scripts/python.exe scripts/bench_engine.py
.venv/Scripts/python.exe scripts/export_model.py ckpts/best.pt --torchscript --onnx
```

## 架构要点（跨文件才能看懂的部分）

依赖方向：`xiangqi/`（规则引擎，零 torch）← `rl/`（训练，torch）← `gui/`（FastAPI + 原生 JS）。

**全局约定（constants.py，所有模块共享，不得重复定义）**：
- `sq = row*9 + col`，row 0 = 红方底线；`action = from_sq*90 + to_sq`，`ACTION_SIZE=8100`
- 180° 翻转：`flip_sq(s)=89-s`，用于黑方视角

**当前走子方视角（最大的 bug 陷阱，横跨 encoding/mcts/selfplay/gui）**：
网络输入与策略索引恒为当前走子方视角——黑方走子时棋盘 180° 翻转+红黑互换（`rl/encoding.py`）；
value 恒为当前走子方期望。因此：
- `SearchTree.visit_counts()` 返回**棋盘真实坐标**，入训练样本时才经 `move_to_policy_index` 转视角
- MCTS backup 逐层取负（叶值为叶子走子方视角）
- selfplay 的 z 目标符号 = 该步走子方视角（`mat_red * side` 转材料分）
- GUI 前端评估条要按 `side_to_move` 换算红方视角（此处曾出过符号颠倒 bug）

**MCTS 两阶段 API**（DESIGN §8）：`select_leaves()`（施加 virtual loss，终局叶子内部直接回传）→ 跨多盘 stack 成大 batch 一次前向 → `apply_results()`。selfplay worker 按此跨盘合批。树复用 `advance()` 后必须给新根重新混 Dirichlet 噪声（曾修复过丢失 bug）。

**训练循环**（`rl/train.py`，同步迭代制，推理服务架构见 DESIGN §9）：spawn 每 GPU 一个 EvalServer（`rl/inference_server.py`，聚合请求成大 batch、bf16 前向、稀疏返回合法着法概率）+ `num_workers` 个纯 CPU 自博弈 worker（0=按核数自动；共享内存传状态，`torch.multiprocessing` spawn，Windows 兼容）→ queue 聚合实时统计（红黑胜率是核心观测指标；`evals/s`/`avg_batch` 看推理吞吐）→ replay buffer（states uint8、π 稀疏存储）→ GPU 0 训练（BN/bias 不做 weight decay）→ arena 门控晋升 best。检查点目录 `ckpts/`。

**规则引擎注意**：困毙即负（≠国象和棋）；长将判负；`Board.status(legal=None)` 接受预计算 legal_moves 避免重复生成（MCTS 热路径，`legal_moves` 占 CPU ~47%，勿在同一叶子重复调用）；`push/pop` 是 make-unmake，禁止 deepcopy。

**格式契约**：checkpoint = `{"model_state", "model_config", "iteration", ...}`（加载方必须用 model_config 重建网络）；对局记录 = `xiangqi-record-v1` JSON（起始 FEN + UCCI 着法），selfplay 与 GUI 通用，`xiangqi/records.py` 读写。

## 验证习惯

- 改 `xiangqi/` → 跑 `test_rules.py`/`test_fuzz.py`（含与暴力实现对拍的 fuzz）；perft 基准：初始局面 44 / 1920 / 79666
- 改 `rl/` 数学逻辑 → 不要只跑单测，写一次性脚本实证符号/翻转自洽（此类 bug 不报错只让训练悄悄变弱），再跑 smoke_test
- 改 GUI API → 前后端都要对 DESIGN §13 核对（前端 `gui/web/app.js` 的 fetch 与 `gui/server.py` 路由）
