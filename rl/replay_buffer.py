"""经验回放缓冲区（DESIGN §9，磁盘持久化 .npz 分片 + meta.json）。

存储格式：
  - states  : np.uint8 (0/1 平面，节省内存)
  - policy  : 稀疏存储（每样本 int16 索引数组 + float32 概率数组）
  - value   : float32

环形覆盖策略：写入位置循环至 capacity 后从头覆盖旧样本。

save/load：
  meta.json          — capacity / state_shape / size / pos
  states_NNNN.npz    — 状态分片（每片至多 SHARD_SIZE 条）
  values.npy         — 价值数组（有效顺序）
  sparse_policy.npz  — flat 索引+概率+lengths（有效顺序）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from xiangqi.constants import ACTION_SIZE

# 每个保存分片最多包含的样本数（避免单文件过大）
_SHARD_SIZE = 100_000


class ReplayBuffer:
    """环形经验回放缓冲区。

    Parameters
    ----------
    capacity    : 最大样本数（满后环形覆盖最旧样本）
    state_shape : 单个状态的形状，如 (31, 10, 9)
    """

    def __init__(self, capacity: int, state_shape: tuple) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity 必须 > 0，得到 {capacity}")
        self.capacity: int = capacity
        self.state_shape: tuple = tuple(state_shape)

        # 当前有效样本数（≤ capacity）
        self._size: int = 0
        # 下一个写入位置（环形指针）
        self._pos: int = 0

        # 状态池（uint8 节省内存：0/1 平面存 0 或 1）
        self._states: np.ndarray = np.zeros(
            (capacity, *self.state_shape), dtype=np.uint8
        )
        # 价值池
        self._values: np.ndarray = np.zeros(capacity, dtype=np.float32)
        # 稀疏策略池：per-slot 存 (int16 indices, float32 probs)
        # 初始化为 None，写入时替换
        self._pol_indices: List = [None] * capacity
        self._pol_probs: List = [None] * capacity

    # ── 写入 ──────────────────────────────────────────────────────────────

    def add_game(
        self,
        states: np.ndarray,
        sparse_policies: List[Tuple[np.ndarray, np.ndarray]],
        values: np.ndarray,
    ) -> None:
        """将一盘棋的所有时间步样本追加到缓冲区（环形覆盖）。

        Parameters
        ----------
        states          : (T, *state_shape)，uint8 或 float32（自动转 uint8）
        sparse_policies : list of (indices: int16 array, probs: float32 array)
                          长度 = T，每个元素对应一步的稀疏策略
        values          : (T,) float32，每步的目标价值
        """
        n = len(states)
        if len(sparse_policies) != n or len(values) != n:
            raise ValueError(
                f"长度不一致：states={n}, "
                f"sparse_policies={len(sparse_policies)}, "
                f"values={len(values)}"
            )

        for i in range(n):
            pos = self._pos

            # 状态存为 uint8（0/1 平面）
            self._states[pos] = np.asarray(states[i], dtype=np.uint8)

            # 价值
            self._values[pos] = float(values[i])

            # 稀疏策略
            idx_arr, prb_arr = sparse_policies[i]
            self._pol_indices[pos] = np.asarray(idx_arr, dtype=np.int16)
            self._pol_probs[pos] = np.asarray(prb_arr, dtype=np.float32)

            # 推进环形指针
            self._pos = (pos + 1) % self.capacity
            if self._size < self.capacity:
                self._size += 1

    # ── 采样 ──────────────────────────────────────────────────────────────

    def sample(
        self, batch: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """随机均匀采样 batch 个样本。

        Returns
        -------
        states       : float32 (batch, *state_shape)  — uint8 → float32
        policy_dense : float32 (batch, ACTION_SIZE)   — 稀疏策略稠密化
        values       : float32 (batch,)
        """
        if self._size == 0:
            raise RuntimeError("ReplayBuffer 为空，无法采样")
        if batch <= 0:
            raise ValueError(f"batch 必须 > 0，得到 {batch}")

        # 在有效槽位中随机采样（有效槽位 = [0, self._size)，环形有效区）
        idx = np.random.randint(0, self._size, size=batch)

        # states uint8 → float32：花式索引后单次向量化转换。
        states_out = self._states[idx].astype(np.float32)
        values_out = self._values[idx].copy()

        # 稀疏 → 稠密策略：逐样本行内花式赋值。
        # 说明：此处的成本几乎全部来自向 (batch, ACTION_SIZE) 稠密数组散射写入
        # （batch=4096 时约 132MB，缺页/内存带宽受限，~44ms），Python 循环本身仅
        # ~2ms。用 np.repeat + np.concatenate 做全局二维/扁平花式赋值经实测反而更慢
        # （4096 下 0.68~0.78x，额外的索引数组构造与二维高级索引开销 > 省下的循环），
        # 故保留逐行写入这一最快实现；行内 int16 索引 → int32 避免负数问题。
        policy_dense = np.zeros((batch, ACTION_SIZE), dtype=np.float32)
        for j, si in enumerate(idx):
            pi = self._pol_indices[si]
            pp = self._pol_probs[si]
            if pi is not None and len(pi) > 0:
                policy_dense[j, pi.astype(np.int32)] = pp

        return states_out, policy_dense, values_out

    # ── 长度 ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    # ── 磁盘持久化 ────────────────────────────────────────────────────────

    def save(self, dir: "str | Path") -> None:
        """将 buffer 保存到目录（分片 .npz + meta.json）。

        文件布局：
          <dir>/meta.json
          <dir>/states_NNNN.npz    （每片至多 _SHARD_SIZE 条）
          <dir>/values.npy
          <dir>/sparse_policy.npz
        """
        save_dir = Path(dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        n = self._size

        # meta
        meta = {
            "capacity": self.capacity,
            "state_shape": list(self.state_shape),
            "size": n,
            "pos": self._pos,
        }
        (save_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        if n == 0:
            return

        # 按环形顺序排列的有效槽位索引
        valid_idx = self._ordered_valid_indices()

        # ── 状态分片 ─────────────────────────────────────────────────────
        num_shards = (n + _SHARD_SIZE - 1) // _SHARD_SIZE
        for shard in range(num_shards):
            start = shard * _SHARD_SIZE
            end = min(start + _SHARD_SIZE, n)
            chunk = self._states[valid_idx[start:end]]
            np.savez_compressed(
                str(save_dir / f"states_{shard:04d}.npz"),
                states=chunk,
            )

        # ── 价值 ─────────────────────────────────────────────────────────
        np.save(str(save_dir / "values.npy"), self._values[valid_idx])

        # ── 稀疏策略（flat 存储）─────────────────────────────────────────
        flat_idx_list: list[np.ndarray] = []
        flat_prb_list: list[np.ndarray] = []
        pol_lens = np.zeros(n, dtype=np.int32)

        for j, si in enumerate(valid_idx):
            pi = self._pol_indices[si]
            pp = self._pol_probs[si]
            if pi is not None and len(pi) > 0:
                flat_idx_list.append(pi.astype(np.int16))
                flat_prb_list.append(pp.astype(np.float32))
                pol_lens[j] = len(pi)
            # else: pol_lens[j] stays 0

        flat_indices = (
            np.concatenate(flat_idx_list)
            if flat_idx_list
            else np.array([], dtype=np.int16)
        )
        flat_probs = (
            np.concatenate(flat_prb_list)
            if flat_prb_list
            else np.array([], dtype=np.float32)
        )

        np.savez_compressed(
            str(save_dir / "sparse_policy.npz"),
            indices=flat_indices,
            probs=flat_probs,
            lengths=pol_lens,
        )

    @classmethod
    def load(cls, dir: "str | Path") -> "ReplayBuffer":
        """从目录加载 buffer（与 save() 对应）。"""
        load_dir = Path(dir)
        meta_path = load_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json 不存在：{meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        capacity: int = meta["capacity"]
        state_shape: tuple = tuple(meta["state_shape"])
        n: int = meta["size"]
        pos: int = meta["pos"]

        buf = cls(capacity, state_shape)
        buf._size = n
        buf._pos = pos

        if n == 0:
            return buf

        # 恢复环形顺序的有效槽位索引（与 save() 时一致）
        valid_idx = buf._ordered_valid_indices()

        # ── 读取状态分片 ─────────────────────────────────────────────────
        num_shards = (n + _SHARD_SIZE - 1) // _SHARD_SIZE
        all_states: list[np.ndarray] = []
        for shard in range(num_shards):
            shard_path = load_dir / f"states_{shard:04d}.npz"
            if not shard_path.exists():
                raise FileNotFoundError(f"状态分片不存在：{shard_path}")
            data = np.load(str(shard_path))
            all_states.append(data["states"])
        states_ordered = np.concatenate(all_states, axis=0)  # (n, *state_shape)

        # ── 读取价值 ─────────────────────────────────────────────────────
        values_ordered = np.load(str(load_dir / "values.npy"))

        # ── 读取稀疏策略 ─────────────────────────────────────────────────
        sp = np.load(str(load_dir / "sparse_policy.npz"))
        flat_indices: np.ndarray = sp["indices"]
        flat_probs: np.ndarray = sp["probs"]
        pol_lens: np.ndarray = sp["lengths"]

        # 写回正确槽位
        buf._states[valid_idx] = states_ordered
        buf._values[valid_idx] = values_ordered

        offset = 0
        for j, si in enumerate(valid_idx):
            length = int(pol_lens[j])
            if length > 0:
                buf._pol_indices[si] = flat_indices[offset : offset + length].astype(
                    np.int16
                )
                buf._pol_probs[si] = flat_probs[offset : offset + length].astype(
                    np.float32
                )
            else:
                buf._pol_indices[si] = np.array([], dtype=np.int16)
                buf._pol_probs[si] = np.array([], dtype=np.float32)
            offset += length

        return buf

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _ordered_valid_indices(self) -> np.ndarray:
        """返回有效槽位的原始索引，按"最旧→最新"的写入顺序排列。

        - 未满（size < capacity）：索引 [0 .. size-1]（写入顺序即存储顺序）
        - 已满（size == capacity）：从 pos（最旧）绕一圈到 pos-1（最新）
        """
        n = self._size
        if n == 0:
            return np.array([], dtype=np.int64)
        if n < self.capacity:
            return np.arange(n, dtype=np.int64)
        # 已满：从 _pos 开始环绕
        return np.array(
            [(self._pos + i) % self.capacity for i in range(n)], dtype=np.int64
        )
