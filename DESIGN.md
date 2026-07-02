# 中国象棋 AlphaZero 式强化学习系统 — 设计文档（契约版）

> 本文档是全项目的**唯一事实来源（single source of truth）**。所有模块必须严格遵守本文档定义的
> 坐标系、FEN、动作编码、网络输入编码、API 契约与记录格式。任何偏离都视为 bug。

## 1. 总览

- **目标**：无人类数据，纯自我博弈（AlphaZero 范式）训练中国象棋模型。
- **训练环境**：云端 8×H100，CUDA 12.8，项目路径 `/home/jovyan/LLM/yanjz/projects/xiangqi-rl-26`。
- **推理环境**：Windows / macOS 笔记本（CPU 或核显），模型导出 TorchScript + ONNX。
- **界面**：本地 Web GUI（FastAPI + 原生 JS 单页），人机对战 / 观战 / 复盘 / FEN 导入导出。
- **对局记录**：Xiangqi FEN + UCCI 着法列表（JSONL），自我博弈与 GUI 对战共用同一格式，GUI 可加载回放。

### 目录结构

```
xiangqi-rl-26/
├── DESIGN.md                  # 本文档
├── pyproject.toml
├── configs/                   # YAML 训练预设: smoke.yaml / laptop.yaml / cloud_8xh100.yaml
├── xiangqi/                   # 纯 Python 规则引擎（零 torch 依赖）
│   ├── __init__.py            # 导出 Board, Move 等公共 API
│   ├── constants.py           # [已定稿] 坐标/棋子/动作编码
│   ├── board.py               # 局面表示、走子生成、终局判定
│   ├── fen.py                 # FEN 解析/序列化、UCCI 着法解析
│   └── notation.py            # 中文着法（炮二平五）转换
├── rl/                        # 训练系统（依赖 torch）
│   ├── config.py              # [已定稿] 配置体系
│   ├── encoding.py            # 局面→张量、动作↔策略索引（含视角翻转）
│   ├── model.py               # 策略价值网络（ResNet）
│   ├── mcts.py                # 批量化 PUCT MCTS
│   ├── selfplay.py            # 自我博弈 worker（多进程，每进程绑 1 GPU）
│   ├── replay_buffer.py       # 经验回放（磁盘持久化 .npz）
│   ├── heuristics.py          # 启发式：材料价值混合、和棋惩罚、认输、开局多样化
│   ├── evaluate.py            # Arena：新旧模型对抗门控
│   ├── logger.py              # 日志：控制台 rich + train.log + metrics.csv + TensorBoard
│   └── train.py               # 主训练循环（入口）
├── gui/
│   ├── server.py              # FastAPI 后端（入口: python -m gui.server）
│   └── web/                   # index.html / app.js / board.js / style.css（无构建工具）
├── scripts/
│   ├── setup_env.sh           # 云端 uv 环境（CUDA 12.8）
│   ├── setup_env_local.ps1    # 本地 Windows CPU 环境
│   ├── train_cloud.sh         # 8×H100 训练启动（tmux 内）
│   ├── smoke_test.py          # 端到端冒烟：tiny 模型自我博弈→训练→arena（CPU 可跑）
│   ├── bench_engine.py        # 引擎性能基准（movegen/s, perft）
│   └── export_model.py        # checkpoint → TorchScript / ONNX
├── records/                   # 对局记录（selfplay/ 与 gui/ 子目录）
├── checkpoints/               # 模型检查点（best.pt, iter_XXXX.pt）
├── logs/                      # train.log / metrics.csv / tb/
└── tests/                     # pytest
```

## 2. 坐标系契约（所有模块必须一致）

- 棋盘 10 行（rank）× 9 列（file）。**红方在下**。
- `row ∈ [0,9]`：row 0 = 红方底线（红帅初始行），row 9 = 黑方底线。
- `col ∈ [0,8]`：col 0 = 红方视角最左路（UCCI 列 'a'），col 8 = 'i'。
- 格子索引 `sq = row * 9 + col`，`sq ∈ [0, 89]`。
- UCCI 格子字符串 = 列字母 + 行数字：`sq_to_str(sq(2,7)) == "h2"`。
- UCCI 着法 = 起点+终点，如开局中炮 `"h2e2"`。
- 180° 翻转（黑方视角）：`flip_sq(sq) = 89 - sq`。
- 楚河汉界：row 4 与 row 5 之间。红过河 = row ≥ 5；黑过河 = row ≤ 4。
- 九宫：红 rows 0–2 × cols 3–5；黑 rows 7–9 × cols 3–5。

## 3. FEN 契约（Xiangqi FEN，UCCI 标准）

- 棋子字母：`K帅 A仕 B相 N马 R车 C炮 P兵`（红大写 / 黑小写）。解析时**兼容** `E/e`（象）与 `H/h`（马）作为输入别名，输出一律用 B/N。
- 棋盘段：从 row 9（黑底线）到 row 0，每行从 col 0 到 col 8，数字表示连续空格，行间 `/`。
- 走子方：`w` = 红（兼容解析 `r`），`b` = 黑。输出用 `w`。
- 完整格式：`<board> <side> - - <halfmove> <fullmove>`，第 3、4 段固定 `-`（象棋无易位/吃过路兵），halfmove = 未吃子半回合数（用于 60 回合自然限着），fullmove 从 1 起。
- 初始局面：
  `rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1`

## 4. 规则要点（board.py 必须实现）

1. 马蹩腿、象塞眼、象不过河、士/帅不出九宫、帅不对脸（白脸将，作为合法性过滤）。
2. 兵过河前只能进，过河后可平；永不后退。到底线仍可平移。
3. 炮隔一子吃子，平移不吃子。
4. **困毙即负**：无合法着法的一方输（与国象和棋不同！terminal value = 对走子方 -1）。
5. 将军/将死检测；`Board.result()` 返回进行中/红胜/黑胜/和。
6. 重复局面：同一局面（棋盘+走子方）第 3 次出现 → 判定：若循环中一方每着都在将军（长将），该方判负；否则判和。实现用 Zobrist 哈希 + 历史栈；长将检测取最近一个重复周期内该方所有着法是否均为将军。
7. 和棋条件：3 次重复（非长将）、连续 120 半回合无吃子、总长超过 `max_game_plies`（默认 400 半回合，训练配置可调低）。
8. `Board` 关键 API（供 MCTS / GUI 使用，必须按此签名实现）：
   ```python
   class Board:
       side_to_move: int              # RED=1 / BLACK=-1
       def legal_moves(self) -> list[tuple[int,int]]   # [(from_sq,to_sq)]
       def push(self, move) -> None;  def pop(self) -> None
       def is_in_check(self, side=None) -> bool
       def result(self) -> str | None # None=进行中, "1-0","0-1","1/2-1/2"
       def termination(self) -> str | None  # "checkmate","stalemate","repetition","perpetual_check","no_capture","max_moves"
       def fen(self) -> str;  @classmethod def from_fen(cls, fen)
       def zobrist(self) -> int
       def copy(self) -> "Board"
       last_move: tuple[int,int] | None
       def snapshots(self, n: int) -> list[bytes | None]
           # 最近 n 个局面快照，index 0=当前局面，1=一步前……不足补 None。
           # 快照为 90 字节：snapshot[sq] = piece_at(sq) + 7（取值 0..14，空=7）。
           # push 时追加记录、pop 时回退，供 encoding 历史平面使用。
   ```
   性能要求：`legal_moves` 走 pseudo-legal + 过滤（自杀/对脸），初始局面 `legal_moves` 恰好 44 个。perft(2)=1920, perft(3)=79666。

## 5. 动作编码契约

- `ACTION_SIZE = 8100`，`action = from_sq * 90 + to_sq`。非法动作靠 mask 屏蔽。
- 视角翻转：`flip_action(a) = flip_sq(from) * 90 + flip_sq(to)`。
- 网络策略输出恒为**当前走子方视角**（黑走时棋盘与动作都翻转），见 §6。

## 6. 网络输入编码契约（rl/encoding.py）

**当前走子方视角**：黑方走子时，棋盘 180° 翻转且红黑角色互换，使网络永远"从下往上打"。价值输出恒为**当前走子方**的期望 ∈ [-1,1]。

输入张量 `(C, 10, 9)`，`C = 14 * history_steps + 3`（默认 history_steps=2 → C=31）：

| 平面 | 内容 |
|---|---|
| `t*14 + 0..6` | 第 t 步前的历史局面（t=0 为当前）：**当前走子方**的 K,A,B,N,R,C,P 位置 one-hot |
| `t*14 + 7..13` | 对手的 K,A,B,N,R,C,P |
| `C-3` | 走子方颜色：红=全1，黑=全0（翻转后仍保留先后手信息） |
| `C-2` | 上一着起点 one-hot（无则全0） |
| `C-1` | 上一着终点 one-hot |

历史不足时（开局）补零。`encoding.py` 必须提供：
```python
def encode_board(board, history_steps=2) -> np.ndarray            # float32 (C,10,9)
def move_to_policy_index(move, side_to_move) -> int               # 黑方自动 flip
def policy_index_to_move(idx, side_to_move) -> tuple[int,int]
def legal_move_mask(board) -> np.ndarray                          # bool (8100,)
```

## 7. 网络结构（rl/model.py）

AlphaZero 风格 ResNet，尺寸由 `ModelConfig` 决定：

- Stem: Conv3×3(C_in→F) + BN + ReLU
- N × 残差块 [Conv3×3 + BN + ReLU + Conv3×3 + BN +（可选 SE）+ skip + ReLU]
- Policy head（全卷积结构化头）: Conv1×1(F→F) + BN + ReLU → Conv1×1(F→90)。输出 (B,90,10,9)，
  通道 = 终点格 to_sq、空间位置 = 起点格 from_sq；`flatten(2).transpose(1,2).reshape(B,8100)`
  得到 action = from_sq*90 + to_sq 的 logits。（相比 Flatten→FC(8100) 省 ~23M 参数且保留空间结构。）
- Value head: Conv1×1(F→8) + BN + ReLU → Flatten(720) → FC(256) + ReLU → FC(1) → tanh

预设：`small` 5b×64f（~0.6M 参数，冒烟/调试）、`medium` 10b×128f（~3.2M，**默认**，笔记本可跑）、`large` 15b×192f（~10M）。
`forward(x) -> (policy_logits[B,8100], value[B])`。导出支持 TorchScript 与 ONNX（scripts/export_model.py，动态 batch）。

**Checkpoint 格式**（train / export / GUI 共用）：`torch.save` 的 dict：
`{"model_state": state_dict, "model_config": {"blocks","filters","se","history_steps"}, "iteration": int, "optimizer_state": 可选, "meta": dict}`。
加载方必须用 `model_config` 重建网络结构后再 load_state_dict（CPU map_location）。

## 8. MCTS 规范（rl/mcts.py）

- PUCT：`U = c_puct * P * sqrt(N_parent) / (1 + N_child)`，`c_puct` 默认 1.8（配置）。
- Q 存储为**该节点走子方**视角；回传逐层取负。未访问子节点 Q 初始化 = FPU（父 Q - 0.2，配置）。
- 根节点 Dirichlet 噪声：`α=0.2, ε=0.25`（训练时开，评测/GUI 关）。
- **批量叶子评估**：单进程内跨多盘对局收集叶子，凑 batch 送 GPU/CPU 前向；用 virtual loss（默认 1.0）允许同一棵树并发选路。
- 温度：ply < `temp_moves`（默认 24）时按 `N^{1/τ}` 采样（τ=1），之后 argmax。
- 终局叶子直接用真实结果（±1 / 和棋值），不进网络。
- 树复用：走子后子树保根。
- 接口（两阶段 API，支持跨对局合并 batch）：
  ```python
  class SearchTree:                       # 单棵树（一盘对局）
      def __init__(self, board, cfg: MCTSConfig, add_noise: bool)
      def select_leaves(self, max_leaves: int) -> tuple[np.ndarray, list]
          # 返回 (states[K,C,10,9] float32, tokens)。选叶子时施加 virtual loss；
          # 终局叶子内部直接回传真实值、计入 visits，不出现在返回值里（K 可为 0）。
      def apply_results(self, tokens, policy_probs, values) -> None
          # policy_probs[K,8100] 为 softmax 后概率（未 mask），内部按合法着法 mask 归一再展开
      @property
      def total_visits(self) -> int       # 根已完成模拟数
      def visit_counts(self) -> "np.ndarray"  # (8100,) 根访问计数，**棋盘真实坐标**（非翻转视角）
      def root_value(self) -> float       # 根 Q，当前走子方视角
      def advance(self, move) -> None     # 走子并保留对应子树（树复用）
  def search(board, net_eval_fn, cfg, num_sims, add_noise=False) -> tuple[np.ndarray, float]
      # 单盘便捷封装（GUI/arena 用）；net_eval_fn: states[B,C,10,9] -> (policy_probs[B,8100], values[B])
  ```
  selfplay 驱动方式：对每盘未满 sims 的对局调 select_leaves，跨盘 stack 成大 batch 一次前向，再逐盘 apply_results。
  训练样本 π 入库时用 move_to_policy_index 转为当前方视角；visit_counts 本身保持棋盘真实坐标。

## 9. 自我博弈与训练流水线（同步迭代制，便于看日志调参）

每次迭代（iteration）：
1. **Self-play**：8 个 GPU × `workers_per_gpu`（默认 4）个进程，每进程并发 `games_per_worker`（默认 16）盘，共产生 `games_per_iteration` 盘（默认 2000）。当前 best 模型执红黑双方。
2. **写盘**：样本 (state, π, z) 追加进 replay buffer（保留最近 `buffer_window` 个位置，默认 1,500,000）；对局记录写 `records/selfplay/iter_XXXX.jsonl`。
3. **训练**：GPU 0（或 DDP）在 buffer 上采样 `train_steps_per_iteration`（默认 1000）步，batch 4096。损失 `CE(π) + MSE(z) + L2(1e-4)`，AdamW lr 1e-3 余弦退火（配置可换 SGD+momentum）。
4. **Arena 门控**：新模型 vs best，`arena_games`（默认 40，红黑各半）盘，200 sims、无噪声、低温。胜率 ≥ `gate_threshold`（默认 0.55，和棋计 0.5）则晋升 best。
5. 保存 checkpoint、打日志、进入下一迭代。

多进程结构：`train.py` 主进程 spawn selfplay workers（`torch.multiprocessing`，每 worker `CUDA_VISIBLE_DEVICES` 绑卡），worker 通过 `Queue` 上报完成的对局（样本+统计），主进程聚合。**必须支持 `device=cpu` 无 GPU 降级**（本地冒烟）。中断恢复：`--resume` 从最新 checkpoint + buffer 续训。

## 10. 启发式收敛加速（rl/heuristics.py，全部可配置开关）

1. **材料价值混合**：`z_target = (1-λ)·z + λ·tanh(material_diff/材料尺度)`，λ 从 `value_blend_init`（0.5）线性退火到 0（前 `value_blend_iters` 个迭代）。材料分：车9 炮4.5 马4 象2 士2 兵1(过河2)。
2. **和棋惩罚**：和棋的 z = `draw_penalty`（默认 -0.1，双方同罚），提高进取性（象棋和棋率高，防止塌缩到互相闲逛）。
3. **认输机制**：迭代 ≥ `resign_start_iter`（默认 10）后启用；双方连续 `resign_consecutive`（4）个己方回合 root value < `resign_threshold`（-0.92）即判负提前结束；10% 对局禁用认输以校准误判率（日志输出 false-positive 率）。
4. **开局多样化**：前 `opening_random_plies`（8）步温度 1.25 采样，防止开局塌缩。
5. **一步杀捷径**：self-play 每步先查一步将死，存在则直接走（目标 π 为该着 one-hot），加速终局信号传播。

## 11. 日志契约（rl/logger.py）—— 用户核心需求

**Self-play 实时**（每 `log_interval_sec`=30s，聚合所有 worker）：
```
[iter 12 | selfplay] games 356/2000 (17.8%) | R/B/D 142/128/86 | 红胜率 39.9% 黑胜率 36.0% 和率 24.2%
  | avg_plies 121.4 | moves/s 84.2 | nn_evals/s 21030 | resign 31.5% | buffer 812k | elapsed 06:12
```
**训练**（每 100 步）：loss 总/policy/value、policy entropy、value MAE、lr、samples/s、grad_norm。
**Arena**（结束时）：`new vs best: 24W 6D 10L (score 67.5%) | 执红 14W2D4L 执黑 10W4D6L | GATE: PROMOTED`。
**每迭代汇总行**写入 `logs/metrics.csv`（列固定：iter, games, red_win, black_win, draw, red_winrate, black_winrate, draw_rate, avg_plies, resign_rate, resign_fp_rate, buffer_size, loss, policy_loss, value_loss, entropy, value_mae, lr, arena_score, arena_red_score, arena_black_score, promoted, selfplay_sec, train_sec, total_sec）。
同时输出 TensorBoard（logs/tb/）。所有实时行同步写 `logs/train.log`（带时间戳）。
**红黑胜率是核心观测指标**：象棋红先手优势明显，健康训练应看到红胜率 > 黑胜率且和率随棋力上升；若红黑胜率长期五五开且和率极低，多半有视角翻转 bug —— 日志必须让人一眼看出。

## 12. 对局记录格式（唯一格式，selfplay 与 GUI 通用）

JSON 一局一条；文件为 `.jsonl`（多局）或 `.json`（单局）。GUI 两种都能加载。

```json
{"format": "xiangqi-record-v1",
 "start_fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
 "moves": ["h2e2", "h9g7", "..."],
 "result": "1-0",
 "termination": "checkmate",
 "plies": 87,
 "red": "selfplay-iter12", "black": "selfplay-iter12",
 "meta": {"iteration": 12, "sims": 400, "time": "2026-07-02T10:00:00"}}
```

回放时逐着重演得到每步 FEN。GUI 导出时额外提供含每步 FEN 的 `fens` 数组（可选字段）。GUI 还兼容导入纯文本格式：首行 FEN（或省略=初始局面），其后每行/空格分隔的 UCCI 着法。

## 13. GUI API 契约（FastAPI，前后端共同遵守）

Base: `http://127.0.0.1:8000`。静态页面挂 `/`。所有响应 JSON；错误 `{"error": "..."}`
+ 4xx。`GameState` 对象（多接口共用）：

```json
{"game_id": "g1", "fen": "...", "side_to_move": "red",
 "legal_moves": ["a0a1", "..."], "last_move": "h2e2",
 "in_check": false, "game_over": false, "result": null, "termination": null,
 "move_history": [{"move": "h2e2", "chinese": "炮二平五", "fen_after": "..."}],
 "eval": {"value": 0.12, "top_moves": [{"move": "b2e2", "prob": 0.31, "visits": 124}]} }
```
`eval` 仅在有模型且请求带 `with_eval=true` 或走完 AI 着法后附带。

| 方法/路径 | 请求体 | 响应 |
|---|---|---|
| `POST /api/game/new` | `{"mode":"hva"\|"ava"\|"hvh", "human_side":"red"\|"black", "fen":可选, "sims":400}` | GameState（若 AI 执红先走，响应已含 AI 首着） |
| `GET /api/game/{id}` | – | GameState |
| `POST /api/game/{id}/move` | `{"move":"h2e2"}` | GameState；hva 模式下已包含 AI 回着（`ai_move` 字段同时单列） |
| `POST /api/game/{id}/ai_move` | `{"sims":可选}` | GameState（ava 单步推进用） |
| `POST /api/game/{id}/undo` | – | GameState（hva 撤销一整回合两步） |
| `GET /api/game/{id}/hint?sims=400` | – | `{"move","chinese","value","top_moves":[...]}` |
| `GET /api/game/{id}/export` | – | xiangqi-record-v1 JSON（含 fens 数组） |
| `POST /api/replay/load` | `{"record": <v1 JSON>}` 或 `{"text": "FEN\n着法..."}` | `{"fens":[逐步FEN], "moves":[...], "chinese":[...], "result", "start_fen"}` |
| `GET /api/records/list` | – | `{"files":[{"path","games","mtime"}]}`（扫描 records/ 递归） |
| `GET /api/records/get?path=...&index=0` | – | 该文件第 index 局的 v1 JSON（路径必须限制在 records/ 内） |
| `POST /api/model/load` | `{"path":"checkpoints/best.pt"}` | `{"ok":true,"info":{...}}` |
| `GET /api/model/info` | – | `{"loaded":bool,"path","params","blocks","filters","device"}` |
| `POST /api/game/{id}/resign` | – | GameState（人认输） |

无模型加载时 hva/ava 返回 400（提示先加载模型）；hvh 与回放不需要模型。AI 走子在线程池执行避免阻塞事件循环。启动参数：`python -m gui.server --model checkpoints/best.pt --port 8000 --sims 400 --device cpu`。

## 14. GUI 前端功能清单（gui/web/，原生 JS，无构建）

- SVG/Canvas 棋盘：木纹配色、楚河汉界、九宫斜线、圆形棋子（中文字），红黑分色。
- 交互：点选走子（选中高亮 + 合法落点圆点提示）、最后一着高亮、将军红光提示、吃子/走子音效可关。
- 对战：新对局对话框（模式 hva/ava/hvh、执红/执黑、难度=sims 档位：快 100/标准 400/强 1200、自定义 FEN 开局）。
- 辅助：评估条（红方胜率视角）、提示按钮（箭头显示 AI 推荐着 + top-5 列表）、悔棋、认输、翻转棋盘、着法列表（中文纵队记法，点击跳转局面）。
- 复盘：导入（文件选择 .json/.jsonl/.txt 或粘贴文本）、records/ 浏览器（列出自我博弈记录选局）、播放控制（|< < 播放/暂停 > >|、速度）、导出当前对局（下载 .json）。
- AI vs AI 观战：自动逐步推进 + 暂停。
- 状态栏：模型信息、当前评估值、思考中指示。

## 15. 测试与验收（tests/）

- `test_fen.py`：初始 FEN 往返、随机局面往返、别名 E/H 解析、非法 FEN 报错。
- `test_rules.py`：马腿/象眼/士宫/帅对脸/兵规则/炮打；将军、将死、困毙即负；长将判负；重复判和；perft(1)=44, perft(2)=1920, perft(3)=79666（初始局面）。
- `test_encoding.py`：红黑视角翻转一致性（翻转两次=恒等；黑方局面编码 == 红黑互换镜像局面的红方编码）、mask 与 legal_moves 一致。
- `test_model.py`：前向 shape、tanh 范围、TorchScript/ONNX 导出数值一致（atol 1e-4）。
- `test_mcts.py`：一步杀局面 MCTS(100 sims) 必选杀着；访问计数归一。
- `test_records.py`：记录写→读→重演到终局 FEN 一致。
- `scripts/smoke_test.py`：small 模型 CPU 端到端（4 盘 selfplay → 50 步训练 → 4 盘 arena），< 5 分钟，退出码 0 即通过。

## 16. 云端部署（8×H100, CUDA 12.8）

```bash
cd /home/jovyan/LLM/yanjz/projects/xiangqi-rl-26
bash scripts/setup_env.sh          # uv venv + torch cu128 + 依赖
source .venv/bin/activate
tmux new -s xiangqi
python -m rl.train --config configs/cloud_8xh100.yaml   # 或 bash scripts/train_cloud.sh
# 观察: tail -f logs/train.log ; tensorboard --logdir logs/tb
```
中断续训：`python -m rl.train --config configs/cloud_8xh100.yaml --resume`。
训好后：`python scripts/export_model.py checkpoints/best.pt --onnx --torchscript`，把 `best.pt`/`best.onnx` 拷回笔记本，GUI 加载即可。
