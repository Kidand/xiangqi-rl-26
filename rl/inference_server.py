# -*- coding: utf-8 -*-
"""批量推理服务（EvalServer）——DESIGN.md §9 推理服务架构。

进程拓扑
--------
瓶颈在 CPU 端 Python MCTS，故 selfplay worker 是纯 CPU 进程（不持有模型 / CUDA
context），每 GPU 一个 ``eval_server_proc``（``device=cpu`` 时共 1 个）持有模型，循环：

  1. 在 ``eval_wait_ms`` 窗口内聚合多个 worker 的评估请求（req_queue 只传元数据
     ``(worker_id, n)``）；
  2. 从各 worker 的**共享内存**读出叶子局面，拼成一个大 batch（上限 ``eval_max_batch``）
     一次前向（cuda 上 bf16 autocast，``eval_fp16`` 可关）；
  3. softmax（float32）→ 按各叶子合法着法索引矩阵 ``torch.gather``（稀疏返回，避免 8100
     维全量拷贝）→ 结果写回各 worker 的共享内存 → 逐个通知其响应队列。

共享内存布局（每 worker 一套，主进程创建后 ``share_memory_()`` 传参，
Windows spawn 下必须作为进程参数传递，不能用全局）：
  - ``states``  uint8  (MAX_LEAVES, C, 10, 9)
  - ``idx``     int16  (MAX_LEAVES, MAX_MOVES)
  - ``idx_len`` int16  (MAX_LEAVES,)
  - ``probs``   float32 (MAX_LEAVES, MAX_MOVES)   —— 响应（稀疏概率）
  - ``values``  float32 (MAX_LEAVES,)             —— 响应（价值）

``MAX_LEAVES = games_per_worker × mcts.batch_size``；``MAX_MOVES = 120``（象棋合法
着法上限富余值，超过即 assert）。

统计（``stats_shm`` 共享 float64 张量，形状 (n_servers, 3)，列 =
``[n_evals, n_batches, sum_batch_leaves]``）由 server 累加，主进程算 evals/s 与 avg_batch。
"""

from __future__ import annotations

import time
from queue import Empty
from typing import Dict, List, Tuple

import numpy as np

from xiangqi.constants import ACTION_SIZE, ROWS, COLS

__all__ = [
    "MAX_MOVES",
    "WorkerStop",
    "create_shared_buffers",
    "run_eval_batch",
    "eval_server_proc",
    "EvalClient",
]

# 象棋合法着法数上限富余值（DESIGN §9）。填充共享内存时超过即 assert。
MAX_MOVES = 120


class WorkerStop(Exception):
    """worker 侧信号：收到 stop_event 或等待响应超时（server 疑似死亡），应尽快收尾。"""


# ---------------------------------------------------------------------------
# 共享内存分配（主进程调用；每 worker 一套，share_memory_() 后作为参数传给子进程）。
# ---------------------------------------------------------------------------
def create_shared_buffers(n_workers: int, max_leaves: int, in_channels: int) -> Dict[int, dict]:
    """为 ``n_workers`` 个 worker 各分配一套共享张量，返回 ``{worker_id: {name: tensor}}``。

    Windows spawn 兼容：这些张量必须由主进程创建并作为进程参数传递（不能置于全局
    再靠 fork 共享）。
    """
    import torch

    max_leaves = max(1, int(max_leaves))
    bufs: Dict[int, dict] = {}
    for w in range(int(n_workers)):
        d = {
            "states": torch.zeros((max_leaves, in_channels, ROWS, COLS), dtype=torch.uint8),
            "idx": torch.zeros((max_leaves, MAX_MOVES), dtype=torch.int16),
            "idx_len": torch.zeros((max_leaves,), dtype=torch.int16),
            "probs": torch.zeros((max_leaves, MAX_MOVES), dtype=torch.float32),
            "values": torch.zeros((max_leaves,), dtype=torch.float32),
        }
        for t in d.values():
            t.share_memory_()
        bufs[w] = d
    return bufs


# ---------------------------------------------------------------------------
# 纯函数：batch 组装→前向→gather→拆分（单元可测，不起进程）。
# ---------------------------------------------------------------------------
def run_eval_batch(
    model,
    device,
    states_u8: np.ndarray,
    idx: np.ndarray,
    idx_len: np.ndarray,
    fp16: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """对一个大 batch 前向并按合法着法索引 gather 稀疏概率。

    参数
    ----
    states_u8 : (N, C, 10, 9) uint8，encode_board 输出的 0/1 平面
    idx       : (N, MAX_MOVES) int16，每叶子合法着法策略索引（pad 位置任意）
    idx_len   : (N,) int16，每叶子实际合法着法数（≤ MAX_MOVES）
    fp16      : cuda 上是否用 bf16 autocast 做卷积前向（softmax 恒在 float32）

    返回
    ----
    (probs, values)
      probs  : (N, MAX_MOVES) float32，[:, :idx_len[i]] 为 softmax 后合法着法概率，
               其余 pad 位置置 0
      values : (N,) float32，当前走子方视角价值
    """
    import torch

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    n = int(states_u8.shape[0])

    x = torch.from_numpy(np.ascontiguousarray(states_u8)).to(torch.float32)
    idx_t = torch.from_numpy(np.ascontiguousarray(idx)).to(torch.long)
    # pad 位置索引可能为 0（合法），此处再 clamp 一次防越界（pad 位置结果无害，稍后清零）。
    idx_t = idx_t.clamp_(0, ACTION_SIZE - 1)

    if dev.type == "cuda":
        x = x.pin_memory().to(dev, non_blocking=True)
        idx_t = idx_t.to(dev, non_blocking=True)
    else:
        x = x.to(dev)
        idx_t = idx_t.to(dev)

    with torch.inference_mode():
        if dev.type == "cuda" and fp16:
            # bf16 只用于卷积前向；autocast 会自动处理，softmax 放 autocast 外（float32）。
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, values = model(x)
        else:
            logits, values = model(x)

        # bf16 输出转 float32 再算 softmax / 写共享内存。
        logits = logits.float()
        probs_full = torch.softmax(logits, dim=1)           # (N, 8100) float32
        gathered = torch.gather(probs_full, 1, idx_t)       # (N, MAX_MOVES)
        values = values.float().reshape(-1)

    probs_np = gathered.detach().to("cpu").numpy().astype(np.float32, copy=False)
    values_np = values.detach().to("cpu").numpy().astype(np.float32, copy=False)

    # pad 位置清零（worker 只读 [:idx_len]，清零使返回干净、便于测试断言截断正确）。
    lens = np.asarray(idx_len).reshape(-1).astype(np.int64)
    for i in range(n):
        L = int(lens[i])
        if 0 <= L < MAX_MOVES:
            probs_np[i, L:] = 0.0
    return probs_np, values_np


# ---------------------------------------------------------------------------
# EvalServer 进程入口（顶层函数，Windows/Linux spawn 兼容）。
# ---------------------------------------------------------------------------
def eval_server_proc(
    server_id: int,
    device: str,
    ckpt_path: str,
    cfg,
    shm_dict: Dict[int, dict],
    req_queue,
    resp_queues: list,
    stop_event,
    stats_shm,
) -> None:
    """批量推理服务进程。

    Parameters
    ----------
    server_id   : 服务序号（cuda 时对应 GPU 号；cpu 时恒 0）
    device      : "cuda:{id}" / "cpu"
    ckpt_path   : best 模型 checkpoint 路径
    cfg         : 全局 Config（读取 eval_max_batch / eval_wait_ms / eval_fp16）
    shm_dict    : {worker_id: {name: 共享张量}}，本 server 负责的各 worker 共享内存
    req_queue   : 本 server 的请求队列，元素为 (worker_id, n)
    resp_queues : 按 worker_id 索引的响应队列列表（server 只 put，worker 只 get）
    stop_event  : 多进程 Event，置位后 server 收尾退出
    stats_shm   : 共享 float64 张量 (n_servers, 3)，列 [n_evals, n_batches, sum_batch_leaves]
    """
    import torch

    from rl.model import load_checkpoint

    torch.set_num_threads(1)

    dev = torch.device(device)
    model, _ = load_checkpoint(ckpt_path, device=str(dev))
    model.eval()

    fp16 = bool(cfg.selfplay.eval_fp16)
    max_batch = max(1, int(cfg.selfplay.eval_max_batch))
    wait_s = max(0.0, float(cfg.selfplay.eval_wait_ms) / 1000.0)

    stats_row = stats_shm[server_id]

    while not stop_event.is_set():
        # 拿到首个请求（超时后回头查 stop_event）。
        try:
            first = req_queue.get(timeout=0.1)
        except Empty:
            continue

        batch: List[Tuple[int, int]] = [first]
        total_leaves = int(first[1])

        # 在 eval_wait_ms 窗口内继续 get_nowait 聚合，直到无请求或总叶子达上限。
        deadline = time.monotonic() + wait_s
        while total_leaves < max_batch and time.monotonic() < deadline:
            try:
                wid, n = req_queue.get_nowait()
            except Empty:
                continue
            batch.append((wid, int(n)))
            total_leaves += int(n)

        # 从各 worker 共享内存读出叶子，拼成大 batch。
        states_list: List[np.ndarray] = []
        idx_list: List[np.ndarray] = []
        len_list: List[np.ndarray] = []
        layout: List[Tuple[int, int, int]] = []  # (worker_id, offset, n)
        off = 0
        for wid, n in batch:
            buf = shm_dict[wid]
            states_list.append(buf["states"][:n].numpy())
            idx_list.append(buf["idx"][:n].numpy())
            len_list.append(buf["idx_len"][:n].numpy())
            layout.append((wid, off, n))
            off += n

        big_states = np.concatenate(states_list, axis=0)
        big_idx = np.concatenate(idx_list, axis=0)
        big_len = np.concatenate(len_list, axis=0)

        probs, values = run_eval_batch(model, dev, big_states, big_idx, big_len, fp16)

        # 结果按 worker 写回其共享内存并逐个通知。
        for wid, offset, n in layout:
            buf = shm_dict[wid]
            buf["probs"][:n].copy_(torch.from_numpy(probs[offset:offset + n]))
            buf["values"][:n].copy_(torch.from_numpy(values[offset:offset + n]))
            resp_queues[wid].put(n)

        # 统计累加：n_evals（总叶子）/ n_batches（前向次数）/ sum_batch_leaves（批大小和）。
        stats_row[0] += float(total_leaves)
        stats_row[1] += 1.0
        stats_row[2] += float(total_leaves)


# ---------------------------------------------------------------------------
# worker 侧客户端。
# ---------------------------------------------------------------------------
class EvalClient:
    """worker 侧评估客户端：填共享内存 → 发请求 → 等响应 → 读回稀疏结果。"""

    def __init__(
        self,
        worker_id: int,
        shm: dict,
        req_queue,
        resp_queue,
        stop_event,
        timeout: float = 30.0,
    ) -> None:
        self.worker_id = int(worker_id)
        self.shm = shm
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.stop_event = stop_event
        self.timeout = float(timeout)
        self._max_leaves = int(shm["states"].shape[0])

    def evaluate(
        self, states_u8: np.ndarray, idx_list: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """评估一批叶子。

        参数
        ----
        states_u8 : (N, C, 10, 9) uint8，本轮全部叶子（跨 slot 拼接）
        idx_list  : 长度 N 的列表，第 i 项为该叶子合法着法策略索引（int，(Li,)）

        返回
        ----
        (probs_list, values)
          probs_list : 长度 N，第 i 项为 (Li,) float32 稀疏概率（与 idx_list[i] 对齐）
          values     : (N,) float32
        """
        import torch

        n = int(states_u8.shape[0])
        assert n == len(idx_list), f"states/idx 数量不一致: {n} vs {len(idx_list)}"
        assert n <= self._max_leaves, f"叶子数 {n} 超过 MAX_LEAVES {self._max_leaves}"

        shm = self.shm
        if n > 0:
            shm["states"][:n].copy_(torch.from_numpy(np.ascontiguousarray(states_u8)))
            for i, ind in enumerate(idx_list):
                L = len(ind)
                assert L <= MAX_MOVES, f"合法着法数 {L} 超过 MAX_MOVES {MAX_MOVES}"
                shm["idx_len"][i] = L
                if L > 0:
                    shm["idx"][i, :L].copy_(
                        torch.from_numpy(np.asarray(ind, dtype=np.int16))
                    )

        # 发请求 → 阻塞等响应（带 timeout + stop_event 检查防死锁）。
        self.req_queue.put((self.worker_id, n))
        deadline = time.monotonic() + self.timeout
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise WorkerStop("stop_event set while waiting for eval response")
            try:
                self.resp_queue.get(timeout=0.2)
                break
            except Empty:
                if time.monotonic() > deadline:
                    # 超时且未收到响应：server 疑似死亡，抛 WorkerStop 让 worker 收尾。
                    raise WorkerStop("eval response timeout")

        # 读回稀疏结果（copy 出共享内存，避免后续覆盖）。
        probs_list: List[np.ndarray] = []
        for i, ind in enumerate(idx_list):
            L = len(ind)
            probs_list.append(shm["probs"][i, :L].numpy().copy())
        values = shm["values"][:n].numpy().copy()
        return probs_list, values
