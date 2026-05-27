#!/bin/bash
# v2 版：仅 RL 训练 + RL 评测（步骤 5/6）
# 前置条件：已完成 SFT 训练，ckpts_v2/sft_merged 存在
#                  已有 splits_v2/val.jsonl
#
# 资源分配：
#   - RL 训练：默认双卡 DDP，GPU 0 + GPU 1，--no_vllm（HF generate）
#   - RL 评测：双卡 TP=2，vLLM mp backend

set -e

# 确保脚本从项目根目录执行
cd "$(dirname "$0")"

source .venv/bin/activate

# 默认全局只用 GPU 0（需要双卡的步骤会 inline 覆盖）
export CUDA_VISIBLE_DEVICES=0

# cuDNN 路径
export LD_LIBRARY_PATH=$(python -c "import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH

# 让 PyTorch 显存分配更宽松
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 让 vLLM 在多卡情况下用 mp 后端，不用 Ray
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# RL 是否用双卡 DDP
# 1 = torchrun 双卡
# 0 = 单卡 python rl_train.py --no_vllm
RL_DDP="${RL_DDP:-1}"

mkdir -p ckpts_v2 results_v2 logs

# ============ 工具函数 ============

cleanup_vllm() {
    echo ""
    echo "🧹 清理残留 vLLM / GPU 进程..."
    pkill -9 -f "VllmWorker" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    sleep 8
    echo "📊 当前 GPU 状态："
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv
    echo ""
}

run_eval() {
    local model="$1"
    local data="$2"
    local out="$3"
    local extra="${4:-}"

    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "⏭️  跳过：$out 已存在。如要重跑，请先删除该文件。"
        return 0
    fi

    echo ">>> 评测：model=$model"
    echo ">>> 数据：$data"
    echo ">>> 输出：$out"

    CUDA_VISIBLE_DEVICES=0,1 python evaluate.py \
        --model "$model" \
        --data "$data" \
        --output "$out" \
        $extra

    cleanup_vllm
}

check_file_exists() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo "❌ 缺少文件：$f"
        exit 1
    fi
}

check_dir_nonempty() {
    local d="$1"
    if [ ! -d "$d" ] || [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
        echo "❌ 缺少目录或目录为空：$d"
        exit 1
    fi
}

# ============ 启动前检查 ============

echo "================================================"
echo "启动前检查（仅 RL pipeline）"
echo "================================================"

# RL 训练需要 SFT 模型作为起点
check_dir_nonempty "ckpts_v2/sft_merged"
echo "✅ SFT 模型存在：ckpts_v2/sft_merged"

# 评测需要 val 数据
check_file_exists "splits_v2/val.jsonl"
N_VAL=$(wc -l < splits_v2/val.jsonl)
echo "✅ val 数据存在：splits_v2/val.jsonl（$N_VAL 条）"

echo ""

cleanup_vllm

# ============ [1/2] RL 训练 ============

echo ""
echo "================================================"
echo "[1/2] RL 训练：HF generate（不用 vLLM）"
echo "================================================"

if [ -d "ckpts_v2/rl_merged" ] && [ -n "$(ls -A ckpts_v2/rl_merged 2>/dev/null)" ]; then
    echo "⏭️  跳过：ckpts_v2/rl_merged 已存在。"
    echo "如需重新 RL 训练，请先删除："
    echo "    rm -rf ckpts_v2/rl ckpts_v2/rl_merged"
else
    cleanup_vllm

    set +e

    if [ "$RL_DDP" = "1" ]; then
        echo ">>> 使用双卡 DDP 训练 RL：GPU 0 + GPU 1"
        CUDA_VISIBLE_DEVICES=0,1 torchrun \
            --standalone \
            --nproc_per_node=2 \
            rl_train.py \
            --no_vllm
    else
        echo ">>> 使用单卡训练 RL：GPU 0"
        CUDA_VISIBLE_DEVICES=0 python rl_train.py --no_vllm
    fi

    TRAIN_EXIT=$?
    set -e

    cleanup_vllm

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "❌ rl_train.py 退出码：$TRAIN_EXIT"

        if [ $TRAIN_EXIT -eq 137 ]; then
            echo "💡 137 = 进程被 SIGKILL，通常是系统 RAM OOM 或 cgroup 限制。"
            echo "💡 可检查："
            echo "    dmesg -T | grep -iE 'killed|oom' | tail -10"
            echo "    free -h"
        else
            echo "💡 可尝试降低："
            echo "    RL_BATCH_SIZE"
            echo "    RL_NUM_GENERATIONS"
            echo "    RL_MAX_COMPLETION_LEN"
            echo "    RL_MAX_PROMPT_LEN"
        fi

        exit $TRAIN_EXIT
    fi
fi

check_dir_nonempty "ckpts_v2/rl_merged"

# ============ [2/2] 评测 RL ============

echo ""
echo "================================================"
echo "[2/2] 评测 RL 模型"
echo "================================================"

run_eval "ckpts_v2/rl_merged" \
         "splits_v2/val.jsonl" \
         "results_v2/eval_rl_val.jsonl"

echo ""
echo "🎉 RL 训练与 val 评测流程完成（v2）！"
echo ""
echo "生成的主要文件："
echo "  results_v2/eval_rl_val.jsonl"
echo ""
echo "private 推理 / Kaggle submission 请运行 run_voted_private_only.sh"