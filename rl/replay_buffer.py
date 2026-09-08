"""经验回放缓冲区（DESIGN §9，磁盘持久化 .npz 分片 + meta.json）。

存储格式：
  - states  : np.uint8 (0/1 平面，节省内存)
  - policy  : 稀疏存储（每样本 int16 索引数组 + float32 概率数组）
  - value   : float32

环形覆盖策略：写入位置循环至 capacity 后从头覆盖旧样本。

save/load：
  meta.json          — capacity / state_shape / size / pos / target_metadata
  states_NNNN.npz    — 状态分片（每片至多 SHARD_SIZE 条）
  values.npy         — 价值数组（有效顺序）
  sparse_policy.npz  — flat 索引+概率+lengths（有效顺序）
"""

from __future__ import annotations

import json
from copy import deepcopy
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from xiangqi.constants import ACTION_SIZE

# 每个保存分片最多包含的样本数（避免单文件过大）
_SHARD_SIZE = 100_000

_LOG = logging.getLogger(__name__)


class ReplayBufferCompatibilityError(ValueError):
    """数据存在但与调用方的契约不兼容，不得当作磁盘损坏而静默重建。"""


class ReplayBuffer:
    """环形经验回放缓冲区。

    Parameters
    ----------
    capacity    : 最大样本数（满后环形覆盖最旧样本）
    state_shape : 单个状态的形状，如 (31, 10, 9)
    target_metadata : 价值标签契约；None 表示旧格式、标签来源未知
    """

    def __init__(
        self, capacity: int, state_shape: tuple, target_metadata: Optional[dict] = None
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity 必须 > 0，得到 {capacity}")
        self.capacity: int = capacity
        self.state_shape: tuple = tuple(state_shape)
        self.target_metadata: Optional[dict] = deepcopy(target_metadata)

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

    # ── 一致性快照（供后台线程 save 用）─────────────────────────────────────

    def snapshot(self) -> "ReplayBuffer":
        """返回当前 buffer 的一致性快照，可直接 ``.save(dir)``（供后台保存线程用）。

        动机：``save()``（npz 压缩 ~1.5M 样本）耗时 ~100-150s，同步执行会让整机空转。
        取快照后可把压缩落盘挪到后台线程，与下一迭代 selfplay 并行；快照冻结「取快照那
        一刻」的数据，后续 ``add_game`` 不会污染正在被压缩的内容。

        深浅拷贝依据（已核实全仓 add_game 的写入方式）：
          - ``_states``：**实体拷贝**（``.copy()``）。add_game 对其做原地写
            ``self._states[pos] = ...``，若仅共享引用，后台 save 期间下一迭代的 add_game
            会改写正在压缩的行。
          - ``_values``：**实体拷贝**（同理，add_game 原地写 ``self._values[pos] = ...``）。
          - ``_pol_indices`` / ``_pol_probs``：**浅拷贝**（``list(...)``）。add_game 只做整槽
            引用替换（``self._pol_indices[pos] = np.asarray(...)``），全仓无任何对已存槽位
            数组的原地元素改写（``_pol_indices[si][...] = ...`` 之类，已 grep 确认为空），故
            浅拷贝捕获的数组对象引用在后台 save 期间保持不变——add_game 只替换 list 槽位、
            绝不动被引用的旧数组，快照读到的始终是取快照时的一致数据。
          - ``_size`` / ``_pos``：**冻结标量**。``save()`` 的 valid_idx 依赖二者确定有效槽位
            与环形顺序，必须一并冻结。
          - ``target_metadata``：**深拷贝**，避免后续修改污染已捕获样本的标签契约。

        RAM 峰值：快照对整块 ``_states`` 做实体拷贝，体量≈活 buffer 的 states
        （configs/cloud 满 buffer ~1.5M×2790B≈4.2GB）。故后台保存期间系统 RAM
        峰值 ≈ 活 buffer states + 一份快照 states（约 +4.2GB）；多 GPU 服务器 RAM 通常 TB
        级，可忽略。RAM 紧张时改用 rl/train.py `_BufferSaver` 的 "freeze_window" 降级模式
        （零额外 RAM、部分隐藏）。
        """
        # 用 __new__ 跳过 __init__ 的 zeros 分配（否则会先无谓分配一整块 capacity 再丢弃）。
        snap = object.__new__(ReplayBuffer)
        snap.capacity = self.capacity
        snap.state_shape = self.state_shape
        snap.target_metadata = deepcopy(self.target_metadata)
        snap._size = self._size          # 冻结（valid_idx 依赖）
        snap._pos = self._pos            # 冻结（valid_idx 依赖）
        snap._states = self._states.copy()   # 实体拷贝（add_game 原地写）
        snap._values = self._values.copy()   # 实体拷贝（add_game 原地写）
        snap._pol_indices = list(self._pol_indices)  # 浅拷贝（整槽替换，无原地改写）
        snap._pol_probs = list(self._pol_probs)      # 浅拷贝（同上）
        return snap

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
        """将 buffer 原子地保存到目录（分片 .npz + meta.json）。

        文件布局：
          <dir>/meta.json         （最后写，作为「本份已完整」的完成标记）
          <dir>/states_NNNN.npz    （每片至多 _SHARD_SIZE 条）
          <dir>/values.npy
          <dir>/sparse_policy.npz

        目录级原子换名（多分片必须同代，不能出现新旧分片混装）：

          1. 写入全新 ``<dir>.tmp``（meta.json 最后写，作为完成标记）
          2. rmtree(``<dir>.old``, ignore_errors)  —— 确保下一步 rename 目标不存在
          3. 若 ``<dir>`` 存在 → os.rename(dir, ``<dir>.old``)
          4. os.rename(``<dir>.tmp``, dir)
          5. rmtree(``<dir>.old``)

        **任意时刻杀进程的四个窗口都能恢复到某份完整 buffer**（load 只看
        ``<dir>`` 与 ``<dir>.old``，忽略残留 ``<dir>.tmp``）：

          - 窗口 A（步骤 1 写 tmp 途中）：``<dir>`` 仍是上一份完整数据（未被触碰），
            ``<dir>.tmp`` 不完整但被 load 忽略 → 回到 ``<dir>``。
          - 窗口 B（步骤 3 与 4 之间，两次 rename 之间）：``<dir>`` 缺失，
            ``<dir>.old`` = 旧的完整数据，``<dir>.tmp`` = 新的完整数据；load 主目录缺失
            → 回退 ``<dir>.old`` 完整数据。
          - 窗口 C（步骤 4 之后、步骤 5 rmtree 之前）：``<dir>`` = 新完整数据，
            ``<dir>.old`` = 旧完整数据 → 直接用 ``<dir>``。
          - 窗口 D（步骤 5 rmtree 途中）：``<dir>`` = 新完整数据（已就位），
            ``<dir>.old`` 正被删；load 主目录即完整 → 用 ``<dir>``。

        因 meta.json 最后写，只要 ``<dir>/meta.json`` 存在且各分片可解压，即视为该份完整。
        """
        target = Path(dir)
        tmp_dir = target.with_name(target.name + ".tmp")
        old_dir = target.with_name(target.name + ".old")

        # 1) 写入全新 <dir>.tmp（清掉可能残留的上次 .tmp）。
        shutil.rmtree(tmp_dir, ignore_errors=True)
        self._write_payload(tmp_dir)

        # 2) 清掉旧 .old（保证下一步 rename 目标不存在——Windows 上 rename 到已存在目录会失败）。
        shutil.rmtree(old_dir, ignore_errors=True)

        # 3) 若主目录存在，先移到 .old。
        if target.exists():
            os.rename(target, old_dir)

        # 4) tmp 就位为主目录（此刻主目录必不存在）。
        os.rename(tmp_dir, target)

        # 5) 删除旧份。
        shutil.rmtree(old_dir, ignore_errors=True)

    def _write_payload(self, dir: "str | Path") -> None:
        """把当前 buffer 完整写入给定目录（meta.json 最后写，作为完成标记）。

        供 save() 写 ``<dir>.tmp`` 用；不做任何换名，纯落盘。
        """
        save_dir = Path(dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        n = self._size

        meta = {
            "capacity": self.capacity,
            "state_shape": list(self.state_shape),
            "size": n,
            "pos": self._pos,
            "target_metadata": deepcopy(self.target_metadata),
        }

        if n == 0:
            # 空 buffer：只有 meta.json（最后写）。
            (save_dir / "meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
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

        # meta.json 最后写：它的存在标志本份数据分片已全部落盘（完成标记）。
        (save_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        dir: "str | Path",
        capacity: Optional[int] = None,
        state_shape: Optional[tuple] = None,
        warn: Optional[Callable[[str], None]] = None,
        expected_target_metadata: Optional[dict] = None,
        allow_missing_target_metadata: bool = False,
    ) -> "ReplayBuffer":
        """容错加载 buffer。

        容错链（对应 save() 的原子换名，覆盖被 Ctrl-C 打断的各窗口）：

          1. 优先读 ``<dir>``；
          2. 失败（缺失 / 解压错 / meta 缺失或损坏）→ 尝试 ``<dir>.old``，
             成功则打 WARN「主 buffer 损坏，已回退上一份」；
          3. 两者都失败 → 若给了 ``capacity`` 与 ``state_shape`` 则新建空 buffer 并
             打 WARN，否则重新抛出主目录的异常（无法凭空重建）。

        加载成功后清理遗留的 ``<dir>.tmp`` / ``<dir>.old``（上次被打断的残留）。

        Parameters
        ----------
        warn : 容错告警回调（单参数字符串消息），默认 None 时回退 ``_LOG.warning``。
               供调用方接入自身日志系统（如 rl/train.py 的 TrainLogger.warn），
               否则告警只落 stderr、进不了 train.log。
        expected_target_metadata : 非 None 时，在分配/加载样本数组之前检查标签契约。
               已知不匹配立即抛 ValueError，不回退旧份或重建空 buffer。
        allow_missing_target_metadata : 显式兼容缺少契约的历史 buffer 并告警；
               保留 target_metadata=None，不给未知标签补写预期契约。
               对已有契约但不匹配的 buffer 不生效。
        state_shape : 提供时同时校验磁盘状态形状；仍用于损坏时重建空 buffer。
        """
        if warn is None:
            warn = _LOG.warning
        load_dir = Path(dir)
        old_dir = load_dir.with_name(load_dir.name + ".old")
        tmp_dir = load_dir.with_name(load_dir.name + ".tmp")

        buf: Optional["ReplayBuffer"] = None
        try:
            buf = cls._load_from_dir(
                load_dir, expected_target_metadata, state_shape,
                allow_missing_target_metadata,
            )
        except ReplayBufferCompatibilityError:
            raise
        except Exception as e_primary:
            # 回退到上一份完整数据 <dir>.old。
            try:
                buf = cls._load_from_dir(
                    old_dir, expected_target_metadata, state_shape,
                    allow_missing_target_metadata,
                )
                warn(
                    f"主 buffer 损坏（{type(e_primary).__name__}: {e_primary}），"
                    f"已回退上一份 {old_dir.name}"
                )
            except ReplayBufferCompatibilityError:
                raise
            except Exception as e_old:
                if capacity is not None and state_shape is not None:
                    warn(
                        f"主 buffer 与备份均不可用（主: {e_primary}；备: {e_old}），"
                        f"新建空 buffer"
                    )
                    buf = cls(int(capacity), tuple(state_shape), expected_target_metadata)
                else:
                    # 无法重建：抛出主目录的原始异常。
                    raise e_primary

        if expected_target_metadata is not None and buf.target_metadata is None:
            warn("已显式允许加载缺少 target_metadata 的历史 buffer；标签契约仍未知")

        # 清理遗留 .tmp / .old（数据已在内存中，下次 save 会原子重建各份）。
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(old_dir, ignore_errors=True)
        return buf

    @classmethod
    def _load_from_dir(
        cls,
        dir: "str | Path",
        expected_target_metadata: Optional[dict] = None,
        expected_state_shape: Optional[tuple] = None,
        allow_missing_target_metadata: bool = False,
    ) -> "ReplayBuffer":
        """严格加载给定目录（任一环节缺失/损坏均抛异常）。供 load() 的容错链调用。"""
        load_dir = Path(dir)
        meta_path = load_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json 不存在：{meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        target_metadata = meta.get("target_metadata")
        if expected_target_metadata is not None:
            if target_metadata is None:
                if not allow_missing_target_metadata:
                    raise ReplayBufferCompatibilityError(
                        f"buffer 缺少 target_metadata，无法验证标签契约：{load_dir}"
                    )
            elif target_metadata != expected_target_metadata:
                raise ReplayBufferCompatibilityError(
                    f"buffer target_metadata 与预期标签契约不匹配：{load_dir}；"
                    f"磁盘={target_metadata!r}，预期={expected_target_metadata!r}"
                )
        capacity: int = meta["capacity"]
        state_shape: tuple = tuple(meta["state_shape"])
        n: int = meta["size"]
        pos: int = meta["pos"]

        if expected_state_shape is not None and state_shape != tuple(expected_state_shape):
            raise ReplayBufferCompatibilityError(
                f"buffer state_shape 不匹配：磁盘={state_shape}，"
                f"预期={tuple(expected_state_shape)}"
            )
        if not (0 <= n <= capacity and 0 <= pos < capacity):
            raise ValueError(f"buffer size/pos 超出 capacity：{n=}, {pos=}, {capacity=}")
        if n < capacity and pos != n:
            raise ValueError(f"未满 buffer 的 pos 必须等于 size：{n=}, {pos=}")

        buf = cls(capacity, state_shape, target_metadata)
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
            with np.load(str(shard_path)) as data:
                chunk = data["states"]
            expected_shape = (min(_SHARD_SIZE, n - shard * _SHARD_SIZE), *state_shape)
            if chunk.shape != expected_shape:
                raise ValueError(
                    f"状态分片 shape 不匹配：{shard_path}，"
                    f"得到 {chunk.shape}，预期 {expected_shape}"
                )
            all_states.append(chunk)
        states_ordered = np.concatenate(all_states, axis=0)  # (n, *state_shape)

        # ── 读取价值 ─────────────────────────────────────────────────────
        values_ordered = np.load(str(load_dir / "values.npy"))
        if values_ordered.shape != (n,):
            raise ValueError(f"values shape 不匹配：得到 {values_ordered.shape}，预期 {(n,)}")

        # ── 读取稀疏策略 ─────────────────────────────────────────────────
        with np.load(str(load_dir / "sparse_policy.npz")) as sp:
            flat_indices: np.ndarray = sp["indices"]
            flat_probs: np.ndarray = sp["probs"]
            pol_lens: np.ndarray = sp["lengths"]
        if (
            pol_lens.shape != (n,)
            or not np.issubdtype(pol_lens.dtype, np.integer)
            or np.any(pol_lens < 0)
            or flat_indices.ndim != 1
            or flat_probs.shape != flat_indices.shape
            or int(pol_lens.sum(dtype=np.int64)) != len(flat_indices)
        ):
            raise ValueError("sparse_policy shape/lengths 不匹配")

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
