#!/bin/bash
# v2 版：从拒绝采样到 base/sft/rl 在 val 上的评测
# 所有产物输出到 *_v2 目录，旧目录数据保持不动
#
# 资源分配：
#   - 拒绝采样 prepare_sft_data.py：双卡 TP=2，vLLM mp backend
#   - Base/SFT/RL 评测：双卡 TP=2，vLLM mp backend
#   - SFT 训练：单卡 GPU 0
#   - RL 训练：默认双卡 DDP，GPU 0 + GPU 1，--no_vllm（HF generate）

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

# 是否强制重新生成 SFT 数据
# 1 = 每次都重新跑 prepare_sft_data.py
# 0 = 已有 splits_v2/train.jsonl、splits_v2/val.jsonl、sft_data_v2/sft_train.jsonl 则跳过
REBUILD_SFT_DATA="${REBUILD_SFT_DATA:-1}"

# RL 是否用双卡 DDP
# 1 = torchrun 双卡
# 0 = 单卡 python rl_train.py --no_vllm
RL_DDP="${RL_DDP:-1}"

mkdir -p splits_v2 sft_data_v2 ckpts_v2 results_v2 logs

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
echo "启动前检查（v2 pipeline）"
echo "================================================"

PUBLIC_PATH=$(python - <<'PY'
from config import PUBLIC_DATA_PATH
print(PUBLIC_DATA_PATH)
PY
)

echo "当前 config.py 里的 PUBLIC_DATA_PATH = $PUBLIC_PATH"

if [ ! -f "$PUBLIC_PATH" ]; then
    echo "❌ 找不到 public 数据文件：$PUBLIC_PATH"
    echo "请确认 data/public_v2.jsonl 存在，并且 config.py 路径已切到 v2。"
    exit 1
fi

echo "✅ public 数据文件存在"
echo ""

cleanup_vllm

# ============ [1/6] 拒绝采样：生成 SFT 训练数据 ============

echo ""
echo "================================================"
echo "[1/6] 拒绝采样：生成 SFT 训练数据（v2）"
echo "================================================"

if [ "$REBUILD_SFT_DATA" = "0" ] \
   && [ -f "splits_v2/train.jsonl" ] \
   && [ -f "splits_v2/val.jsonl" ] \
   && [ -f "sft_data_v2/sft_train.jsonl" ]; then

    echo "⏭️  跳过 prepare_sft_data.py：已有以下文件："
    echo "    splits_v2/train.jsonl"
    echo "    splits_v2/val.jsonl"
    echo "    sft_data_v2/sft_train.jsonl"
    echo ""
    echo "如需重新生成，请使用："
    echo "    REBUILD_SFT_DATA=1 bash run_train_pipeline.sh"

else
    cleanup_vllm

    CUDA_VISIBLE_DEVICES=0,1 python prepare_sft_data.py \
        --tensor_parallel_size 2

    cleanup_vllm
fi

# ============ Sanity Check：确认步骤1产物齐全 ============

echo ""
echo "================================================"
echo "Sanity check：确认步骤1产物齐全"
echo "================================================"

check_file_exists "splits_v2/train.jsonl"
check_file_exists "splits_v2/val.jsonl"
check_file_exists "sft_data_v2/sft_train.jsonl"

N_TRAIN=$(wc -l < splits_v2/train.jsonl)
N_VAL=$(wc -l < splits_v2/val.jsonl)
N_SFT=$(wc -l < sft_data_v2/sft_train.jsonl)

echo "✅ train: $N_TRAIN 条"
echo "✅ val:   $N_VAL 条"
echo "✅ SFT:   $N_SFT 条（每题保留所有正确回答）"
echo ""

# ============ [2/6] Base 模型在 val 上的 baseline ============

echo ""
echo "================================================"
echo "[2/6] 评测 base 模型 baseline"
echo "================================================"

run_eval "Qwen/Qwen3-4B-Thinking-2507" \
         "splits_v2/val.jsonl" \
         "results_v2/eval_base_val.jsonl"

# ============ [3/6] SFT 训练 ============

echo ""
echo "================================================"
echo "[3/6] SFT 训练：单卡 GPU 0"
echo "================================================"

if [ -d "ckpts_v2/sft_merged" ] && [ -n "$(ls -A ckpts_v2/sft_merged 2>/dev/null)" ]; then
    echo "⏭️  跳过：ckpts_v2/sft_merged 已存在。"
    echo "如需重新 SFT 训练，请先删除："
    echo "    rm -rf ckpts_v2/sft ckpts_v2/sft_merged"
else
    CUDA_VISIBLE_DEVICES=0 python sft_train.py
fi

check_dir_nonempty "ckpts_v2/sft_merged"

# ============ [4/6] 评测 SFT ============

echo ""
echo "================================================"
echo "[4/6] 评测 SFT 模型"
echo "================================================"

run_eval "ckpts_v2/sft_merged" \
         "splits_v2/val.jsonl" \
         "results_v2/eval_sft_val.jsonl"

# ============ [5/6] RL 训练 ============

echo ""
echo "================================================"
echo "[5/6] RL 训练：HF generate（不用 vLLM）"
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

# ============ [6/6] 评测 RL ============

echo ""
echo "================================================"
echo "[6/6] 评测 RL 模型"
echo "================================================"

run_eval "ckpts_v2/rl_merged" \
         "splits_v2/val.jsonl" \
         "results_v2/eval_rl_val.jsonl"

echo ""
echo "🎉 训练与 val 评测流程完成（v2）！"
echo ""
echo "生成的主要文件："
echo "  results_v2/eval_base_val.jsonl"
echo "  results_v2/eval_sft_val.jsonl"
echo "  results_v2/eval_rl_val.jsonl"
echo ""
echo "private 推理 / Kaggle submission 请运行 run_voted_private_only.sh"