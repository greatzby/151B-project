#!/bin/bash
# v3 版 RL：pass rate 过滤 + GRPO + val 评测
#
# 前置：
#   ckpts_v3/sft_merged 已存在
#   splits_v3/train.jsonl 已存在
#   splits_v3/val.jsonl   已存在
#
# 资源：
#   pass rate 计算：双卡 vLLM TP=2
#   RL 训练：       双卡 DDP（默认）或单卡
#   val 评测：      双卡 vLLM TP=2

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$(python -c "import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ─── 可调参数 ──────────────────────────────────────────────────
RECOMPUTE_PASS_RATE="${RECOMPUTE_PASS_RATE:-0}"   # 1=强制重算 pass rate
PASS_RATE_N="${PASS_RATE_N:-6}"                   # 每题采样数（建议 6）
RL_DDP="${RL_DDP:-1}"                             # 1=双卡 DDP
RL_LR="${RL_LR:-1e-5}"                            # ↑ 默认从 5e-6 提到 1e-5
RL_TEMP="${RL_TEMP:-1.0}"                         # ↑ 默认从 0.9 提到 1.0
RL_BETA_OVERRIDE="${RL_BETA_OVERRIDE:-0.02}"      # ↓ 默认从 0.04 降到 0.02
RL_MAX_STEPS="${RL_MAX_STEPS:-50}"                # 默认 70 步

FILTERED_DATA="splits_v3/train_filtered.jsonl"

mkdir -p ckpts_v3 results_v3 logs

# ─── 工具函数 ──────────────────────────────────────────────────
cleanup_vllm() {
    echo "🧹 清理 vLLM 进程..."
    pkill -9 -f "VllmWorker" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    sleep 8
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
    echo ""
}

run_eval() {
    local model="$1"
    local data="$2"
    local out="$3"

    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "⏭️  跳过：$out 已存在"
        return 0
    fi

    echo ">>> 评测：$model"
    CUDA_VISIBLE_DEVICES=0,1 python evaluate.py \
        --model "$model" \
        --data "$data" \
        --output "$out"
    cleanup_vllm
}

# ─── 启动前检查 ─────────────────────────────────────────────────
echo "================================================"
echo "启动前检查"
echo "================================================"
echo "RL_LR        = $RL_LR"
echo "RL_TEMP      = $RL_TEMP"
echo "RL_BETA      = $RL_BETA_OVERRIDE"
echo "RL_MAX_STEPS = $RL_MAX_STEPS"
echo "PASS_RATE_N  = $PASS_RATE_N"
echo "RL_DDP       = $RL_DDP"
echo ""

if [ ! -d "ckpts_v3/sft_merged" ] || [ -z "$(ls -A ckpts_v3/sft_merged 2>/dev/null)" ]; then
    echo "❌ 缺少 ckpts_v3/sft_merged"
    exit 1
fi
echo "✅ ckpts_v3/sft_merged 存在"

if [ ! -f "splits_v3/train.jsonl" ]; then
    echo "❌ 缺少 splits_v3/train.jsonl"
    exit 1
fi
N_TRAIN=$(wc -l < splits_v3/train.jsonl)
echo "✅ splits_v3/train.jsonl: $N_TRAIN 条"

if [ ! -f "splits_v3/val.jsonl" ]; then
    echo "❌ 缺少 splits_v3/val.jsonl"
    exit 1
fi
N_VAL=$(wc -l < splits_v3/val.jsonl)
echo "✅ splits_v3/val.jsonl:   $N_VAL 条"

cleanup_vllm

# ─── [1/3] 计算 pass rate 并过滤 ────────────────────────────────
echo ""
echo "================================================"
echo "[1/3] 计算 pass rate 并过滤训练数据"
echo "================================================"

if [ "$RECOMPUTE_PASS_RATE" = "0" ] && [ -f "$FILTERED_DATA" ] && [ -s "$FILTERED_DATA" ]; then
    N_FILT=$(wc -l < "$FILTERED_DATA")
    echo "⏭️  跳过：$FILTERED_DATA 已存在（$N_FILT 条）"
    echo "   如需重算: RECOMPUTE_PASS_RATE=1 bash $0"
else
    cleanup_vllm

    CUDA_VISIBLE_DEVICES=0,1 python compute_pass_rate.py \
        --model ckpts_v3/sft_merged \
        --input splits_v3/train.jsonl \
        --output "$FILTERED_DATA" \
        --meta_output splits_v3/pass_rate.jsonl \
        --n_samples "$PASS_RATE_N" \
        --tensor_parallel_size 2

    cleanup_vllm
fi

if [ ! -s "$FILTERED_DATA" ]; then
    echo "❌ 过滤后数据为空，请检查 pass rate 分布"
    exit 1
fi
N_FILT=$(wc -l < "$FILTERED_DATA")
echo "✅ 过滤后训练数据：$N_FILT 条"

# ─── [2/3] RL 训练 ──────────────────────────────────────────────
echo ""
echo "================================================"
echo "[2/3] RL 训练（GRPO，使用过滤后数据）"
echo "================================================"

if [ -d "ckpts_v3/rl_merged" ] && [ -n "$(ls -A ckpts_v3/rl_merged 2>/dev/null)" ]; then
    echo "⏭️  跳过：ckpts_v3/rl_merged 已存在"
    echo "   如需重训: rm -rf ckpts_v3/rl ckpts_v3/rl_merged"
else
    cleanup_vllm

    set +e
    if [ "$RL_DDP" = "1" ]; then
        echo ">>> 双卡 DDP 训练"
        CUDA_VISIBLE_DEVICES=0,1 torchrun \
            --standalone --nproc_per_node=2 \
            rl_train.py \
            --no_vllm \
            --train_data "$FILTERED_DATA" \
            --lr "$RL_LR" \
            --temperature "$RL_TEMP" \
            --beta "$RL_BETA_OVERRIDE" \
            --max_steps "$RL_MAX_STEPS"
    else
        echo ">>> 单卡训练"
        CUDA_VISIBLE_DEVICES=0 python rl_train.py \
            --no_vllm \
            --train_data "$FILTERED_DATA" \
            --lr "$RL_LR" \
            --temperature "$RL_TEMP" \
            --beta "$RL_BETA_OVERRIDE" \
            --max_steps "$RL_MAX_STEPS"
    fi
    TRAIN_EXIT=$?
    set -e

    cleanup_vllm

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "❌ rl_train.py 退出码：$TRAIN_EXIT"
        if [ $TRAIN_EXIT -eq 137 ]; then
            echo "💡 137 = OOM kill"
            echo "   dmesg -T | grep -iE 'killed|oom' | tail -10"
        fi
        exit $TRAIN_EXIT
    fi
fi

if [ ! -d "ckpts_v3/rl_merged" ] || [ -z "$(ls -A ckpts_v3/rl_merged 2>/dev/null)" ]; then
    echo "❌ ckpts_v3/rl_merged 不存在"
    exit 1
fi

# ─── [3/3] val 评测 ─────────────────────────────────────────────
echo ""
echo "================================================"
echo "[3/3] 评测 RL 模型 on val"
echo "================================================"

run_eval "ckpts_v3/rl_merged" \
         "splits_v3/val.jsonl" \
         "results_v3/eval_rl_val.jsonl"

echo ""
echo "🎉 完成！"
echo "结果：results_v3/eval_rl_val.jsonl"
echo "对比：results_v3/eval_sft_val.jsonl（baseline）"