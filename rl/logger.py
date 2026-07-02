"""训练日志模块（DESIGN §11）。

三路输出：
  1. rich 控制台（无 rich 环境降级 print）
  2. logs/train.log（带时间戳）
  3. TensorBoard（可关，logs/tb/）

Windows 控制台 UTF-8 容错（sys.stdout.reconfigure）。

CSV 列顺序固定（DESIGN §11）：
  iter, games, red_win, black_win, draw,
  red_winrate, black_winrate, draw_rate, avg_plies,
  resign_rate, resign_fp_rate, buffer_size,
  loss, policy_loss, value_loss, entropy, value_mae,
  lr, arena_score, arena_red_score, arena_black_score,
  promoted, selfplay_sec, train_sec, total_sec
"""

from __future__ import annotations

import csv
import datetime
import sys
from pathlib import Path
from typing import Any

# ── Windows 控制台 UTF-8 容错 ─────────────────────────────────────────────
try:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# ── rich 可选导入 ─────────────────────────────────────────────────────────
try:
    from rich.console import Console as _RichConsole

    _console = _RichConsole(highlight=False)
    _HAS_RICH = True
except ImportError:
    _console = None  # type: ignore[assignment]
    _HAS_RICH = False

# ── metrics.csv 列顺序（DESIGN §11，严格固定）─────────────────────────────
CSV_COLUMNS: list[str] = [
    "iter",
    "games",
    "red_win",
    "black_win",
    "draw",
    "red_winrate",
    "black_winrate",
    "draw_rate",
    "avg_plies",
    "resign_rate",
    "resign_fp_rate",
    "buffer_size",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "value_mae",
    "lr",
    "arena_score",
    "arena_red_score",
    "arena_black_score",
    "promoted",
    "selfplay_sec",
    "train_sec",
    "total_sec",
]


def _fmt_elapsed(seconds: float) -> str:
    """把秒数格式化为 MM:SS（< 1h）或 HH:MM:SS。"""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


class TrainLogger:
    """训练日志：控制台 + train.log + metrics.csv + TensorBoard。

    Parameters
    ----------
    log_dir:
        日志目录（默认 "logs"）。会自动创建。
    tensorboard:
        是否启用 TensorBoard writer（需 torch.utils.tensorboard）。
    """

    def __init__(self, log_dir: str = "logs", tensorboard: bool = True) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # ── 文件日志（带时间戳，直接 open() 写入，绕过 logging 模块捕获问题）──
        self._log_file_path = self._log_dir / "train.log"
        # 以追加模式打开，确保在 Windows 上立即创建文件
        try:
            self._log_file_handle = open(
                str(self._log_file_path), "a", encoding="utf-8", errors="replace"
            )
        except OSError:
            self._log_file_handle = None

        # ── metrics.csv（追加模式，首次写入时写 header）──────────────────
        self._csv_path = self._log_dir / "metrics.csv"
        # 若文件已存在，认为 header 已写
        self._csv_initialized: bool = self._csv_path.exists()

        # ── TensorBoard（可选）────────────────────────────────────────────
        self._tb_writer: Any = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]

                tb_dir = self._log_dir / "tb"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(str(tb_dir))
            except Exception:
                # torch.utils.tensorboard 不可用时静默降级
                pass

    # ── 内部辅助 ──────────────────────────────────────────────────────────

    def _print(self, msg: str) -> None:
        """输出到控制台（rich 或 print 降级）。"""
        if _HAS_RICH and _console is not None:
            try:
                _console.print(msg)
            except Exception:
                print(msg)
        else:
            try:
                print(msg)
            except UnicodeEncodeError:
                # Windows 终端编码异常时用替换字符
                print(msg.encode("utf-8", errors="replace").decode("ascii", errors="replace"))

    def _write_log(self, msg: str) -> None:
        """写入 train.log（直接文件写入，带时间戳）。"""
        if self._log_file_handle is None:
            return
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log_file_handle.write(f"{ts} {msg}\n")
            self._log_file_handle.flush()
        except Exception:
            pass

    def _emit(self, msg: str) -> None:
        """同时输出到控制台和日志文件。"""
        self._print(msg)
        self._write_log(msg)

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def selfplay_progress(self, stats: dict) -> None:
        """按 DESIGN §11 格式输出 self-play 实时进度行。

        stats 期望键（值缺失时用 0/0.0 默认）：
          iter            当前迭代编号
          games           已完成对局数
          total_games     本迭代目标对局数
          red_win         红方胜局数
          black_win       黑方胜局数
          draw            和局数
          avg_plies       平均步数
          moves_per_sec   走子速度（moves/s）
          nn_evals_per_sec 网络评估速度（evals/s）
          resign_rate     认输率（0.0~1.0）
          buffer_size     buffer 中样本总数
          elapsed_sec     已用秒数
        """
        it = stats.get("iter", 0)
        games = stats.get("games", 0)
        total = stats.get("total_games", 2000)
        red_win = stats.get("red_win", 0)
        black_win = stats.get("black_win", 0)
        draw = stats.get("draw", 0)
        avg_plies = stats.get("avg_plies", 0.0)
        moves_s = stats.get("moves_per_sec", 0.0)
        nn_s = stats.get("nn_evals_per_sec", 0.0)
        resign_rate = stats.get("resign_rate", 0.0)
        buf = stats.get("buffer_size", 0)
        elapsed = stats.get("elapsed_sec", 0.0)

        pct = games / total * 100.0 if total > 0 else 0.0
        finished = red_win + black_win + draw
        red_wr = red_win / finished * 100.0 if finished > 0 else 0.0
        black_wr = black_win / finished * 100.0 if finished > 0 else 0.0
        draw_r = draw / finished * 100.0 if finished > 0 else 0.0
        buf_k = buf / 1000.0

        elapsed_str = _fmt_elapsed(elapsed)

        line = (
            f"[iter {it} | selfplay] "
            f"games {games}/{total} ({pct:.1f}%) | "
            f"R/B/D {red_win}/{black_win}/{draw} | "
            f"红胜率 {red_wr:.1f}% 黑胜率 {black_wr:.1f}% 和率 {draw_r:.1f}%\n"
            f"  | avg_plies {avg_plies:.1f} | moves/s {moves_s:.1f} | "
            f"nn_evals/s {nn_s:.0f} | resign {resign_rate * 100:.1f}% | "
            f"buffer {buf_k:.0f}k | elapsed {elapsed_str}"
        )
        self._emit(line)

    def train_step(
        self,
        step: int,
        loss: float,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        value_mae: float,
        lr: float,
        samples_per_sec: float = 0.0,
        grad_norm: float = 0.0,
        iteration: int = 0,
    ) -> None:
        """输出训练步日志（DESIGN §11，每 `train_log_interval` 步调用一次）。"""
        line = (
            f"[iter {iteration} | train step {step}] "
            f"loss {loss:.4f} = policy {policy_loss:.4f} + value {value_loss:.4f} | "
            f"entropy {entropy:.3f} | value_mae {value_mae:.4f} | "
            f"lr {lr:.2e} | samples/s {samples_per_sec:.0f} | grad_norm {grad_norm:.3f}"
        )
        self._emit(line)

        # TensorBoard 标量
        if self._tb_writer is not None:
            global_step = iteration * 100_000 + step
            try:
                self._tb_writer.add_scalar("train/loss", loss, global_step)
                self._tb_writer.add_scalar("train/policy_loss", policy_loss, global_step)
                self._tb_writer.add_scalar("train/value_loss", value_loss, global_step)
                self._tb_writer.add_scalar("train/entropy", entropy, global_step)
                self._tb_writer.add_scalar("train/value_mae", value_mae, global_step)
                self._tb_writer.add_scalar("train/lr", lr, global_step)
                self._tb_writer.add_scalar("train/grad_norm", grad_norm, global_step)
                self._tb_writer.add_scalar("train/samples_per_sec", samples_per_sec, global_step)
            except Exception:
                pass

    def arena_result(
        self,
        wins: int,
        draws: int,
        losses: int,
        red_wins: int,
        red_draws: int,
        red_losses: int,
        black_wins: int,
        black_draws: int,
        black_losses: int,
        score: float,
        threshold: float,
        promoted: bool,
        iteration: int = 0,
    ) -> None:
        """输出 Arena 门控结果（DESIGN §11 格式）。

        Parameters
        ----------
        wins/draws/losses : 总体战绩（新 vs 旧）
        red_wins/… : 新模型执红时的战绩
        black_wins/… : 新模型执黑时的战绩
        score : 综合得分（胜=1, 和=0.5, 负=0）占比
        threshold : 晋升阈值（ArenaConfig.gate_threshold）
        promoted : 是否晋升 best
        iteration : 当前迭代编号
        """
        gate = "PROMOTED" if promoted else "REJECTED"
        total = wins + draws + losses
        score_pct = score * 100.0

        line = (
            f"[iter {iteration} | arena] new vs best: "
            f"{wins}W {draws}D {losses}L (score {score_pct:.1f}%) | "
            f"执红 {red_wins}W{red_draws}D{red_losses}L "
            f"执黑 {black_wins}W{black_draws}D{black_losses}L | "
            f"GATE: {gate}"
        )
        self._emit(line)

        if self._tb_writer is not None:
            try:
                self._tb_writer.add_scalar("arena/score", score, iteration)
                self._tb_writer.add_scalar("arena/promoted", float(promoted), iteration)
            except Exception:
                pass

    def iteration_summary(self, row: dict) -> None:
        """将本迭代汇总行写入 metrics.csv（列顺序与 DESIGN §11 完全一致）。

        Parameters
        ----------
        row : 包含 CSV_COLUMNS 中各键的字典（缺失键填空字符串）。
        """
        # 按固定顺序构造行，缺失字段填空
        ordered_row = {col: row.get(col, "") for col in CSV_COLUMNS}

        write_header = not self._csv_initialized
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                    self._csv_initialized = True
                writer.writerow(ordered_row)
        except Exception as e:
            self._write_log(f"[WARN] 写 metrics.csv 失败: {e}")

        # TensorBoard 标量（数值字段）
        if self._tb_writer is not None:
            iteration = row.get("iter", 0)
            scalar_keys = [
                "red_winrate", "black_winrate", "draw_rate", "avg_plies",
                "resign_rate", "resign_fp_rate", "buffer_size",
                "loss", "policy_loss", "value_loss", "entropy", "value_mae",
                "lr", "arena_score", "arena_red_score", "arena_black_score",
                "selfplay_sec", "train_sec", "total_sec",
            ]
            for k in scalar_keys:
                v = row.get(k)
                if v is not None and v != "":
                    try:
                        self._tb_writer.add_scalar(f"iter/{k}", float(v), int(iteration))
                    except Exception:
                        pass
            try:
                self._tb_writer.add_scalar(
                    "iter/promoted", float(bool(row.get("promoted", False))), int(iteration)
                )
            except Exception:
                pass

        # 控制台 + 文件 简短汇总提示
        it = row.get("iter", "?")
        loss = row.get("loss", "?")
        arena = row.get("arena_score", "?")
        prom = row.get("promoted", "?")
        self._emit(f"[iter {it} | summary] loss={loss} arena_score={arena} promoted={prom}")

    def info(self, msg: str) -> None:
        """普通信息日志。"""
        self._emit(f"[INFO] {msg}")

    def warn(self, msg: str) -> None:
        """警告日志。"""
        self._emit(f"[WARN] {msg}")

    def close(self) -> None:
        """释放资源（关闭 TensorBoard writer 和日志文件句柄）。"""
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
            self._tb_writer = None
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.flush()
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None

    def __enter__(self) -> "TrainLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
