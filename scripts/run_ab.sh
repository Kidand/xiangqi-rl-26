#!/bin/bash
# A/B 预研一键执行：两臂训练 + 终点裁决（scripts/ab_judge.py）
#
# 用法（tmux 内）：
#   bash scripts/run_ab.sh                 # 默认：4+4 卡并行（两臂同时跑）
#   MODE=seq bash scripts/run_ab.sh        # 顺序模式：每臂独占全机，先 puct 后 gumbel
#   RESUME=1 bash scripts/run_ab.sh        # 中断后续跑（两臂各自从自己的 checkpoint 续）
#
# 分配逻辑（par 模式）：
#   - GPU：puct 臂 CUDA_VISIBLE_DEVICES=0-3，gumbel 臂 4-7；两臂 yaml 的 num_gpus
#     被改写为 4（EvalServer 每 GPU 一个，见 rl/selfplay.py:97）。
#   - CPU：显式对半分 num_workers=(有效核数-12)/2（容器感知 effective_cpu_count；
#     预留 8 个 EvalServer + 2 个主进程的余量）。不能留 0 让两臂各自动检测——
#     那会各抢满核、2× 超订反而变慢。瓶颈在 CPU 端 Python MCTS，故并行模式的
#     墙钟与顺序模式大致相当，收益是"一条命令、两臂同时出结果、单臂崩溃不拖累另一臂"。
# 改写通过临时 yaml 副本完成，configs/ 下的契约文件不被修改。
set -e
set -o pipefail   # seq 模式经 tee 管道，缺它训练失败会被 tee 的退出码吞掉

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

MODE="${MODE:-par}"
RESUME_ARGS=()
[ "${RESUME:-0}" = "1" ] && RESUME_ARGS+=("--resume")

mkdir -p logs_ab_meta
echo "=========================================="
echo "A/B 预研：从零 Gumbel vs 现行 PUCT 配方"
echo "模式: $MODE ${RESUME:+(RESUME=1 续跑)}"
echo "=========================================="

if [ "$MODE" = "par" ]; then
    EFF=$(python -c "from rl.selfplay import effective_cpu_count; print(effective_cpu_count())")
    W=$(( (EFF - 12) / 2 ))
    [ "$W" -lt 8 ] && W=8
    TMP_AB=$(mktemp -d)
    for arm in puct gumbel; do
        sed -e "s/num_workers: 0/num_workers: $W/" \
            -e "s/num_gpus: 8/num_gpus: 4/" \
            "configs/ab_scratch_${arm}.yaml" > "$TMP_AB/ab_${arm}.yaml"
    done
    echo "有效核数 $EFF → 每臂 $W 个 selfplay worker + 4 GPU"
    echo "日志: logs_ab_meta/train_{puct,gumbel}.log（tail -f 观察）"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m rl.train \
        --config "$TMP_AB/ab_puct.yaml" "${RESUME_ARGS[@]}" \
        > logs_ab_meta/train_puct.log 2>&1 &
    P_PUCT=$!
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m rl.train \
        --config "$TMP_AB/ab_gumbel.yaml" "${RESUME_ARGS[@]}" \
        > logs_ab_meta/train_gumbel.log 2>&1 &
    P_GUMBEL=$!
    echo "puct 臂 PID=$P_PUCT (GPU 0-3) | gumbel 臂 PID=$P_GUMBEL (GPU 4-7)"

    FAIL=0
    if ! wait "$P_PUCT"; then
        echo "!! puct 臂失败，见 logs_ab_meta/train_puct.log"; FAIL=1
    fi
    if ! wait "$P_GUMBEL"; then
        echo "!! gumbel 臂失败，见 logs_ab_meta/train_gumbel.log"; FAIL=1
    fi
    rm -rf "$TMP_AB"
    if [ "$FAIL" -ne 0 ]; then
        echo "有臂失败：修复后 RESUME=1 bash scripts/run_ab.sh 续跑（已完成的臂会秒过）。"
        exit 1
    fi
else
    # 顺序模式：每臂独占全机（num_workers/num_gpus 用 yaml 原值：自动/8）
    python -m rl.train --config configs/ab_scratch_puct.yaml "${RESUME_ARGS[@]}" \
        2>&1 | tee logs_ab_meta/train_puct.log
    python -m rl.train --config configs/ab_scratch_gumbel.yaml "${RESUME_ARGS[@]}" \
        2>&1 | tee logs_ab_meta/train_gumbel.log
fi

echo ""
echo "两臂训练完成，终点裁决（400 盘去相关 arena，此时全部 8 卡空闲可用）："
python scripts/ab_judge.py 2>&1 | tee logs_ab_meta/judge.out
