#!/bin/bash
# 从 SFT 训练开始，到生成 3 份 Kaggle 提交结束
# 假设 prepare_sft_data.py（步骤1）已经跑完
#
# 资源分配：
#   - SFT 训练：单卡（GPU 0）
#   - 评测：双卡 TP=2（vLLM, mp backend，不用 Ray）
#   - RL 训练：双卡 DDP（GPU 0 + GPU 1），不用 vLLM

set -e
source .venv/bin/activate

# 默认全局只用 GPU 0（适用于 SFT 训练这种单卡步骤）
# 评测和 RL 训练会 inline 覆盖为 0,1
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$(python -c "import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH

# 让 PyTorch 显存分配更宽松
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 让 vLLM 在多卡情况下用 mp 后端（不用 Ray）
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p splits sft_data ckpts results logs

# ============ 工具函数 ============

# 杀掉残留 vLLM 子进程并等显存释放
cleanup_vllm() {
    echo ""
    echo "🧹 清理残留 vLLM / GPU 进程..."
    pkill -9 -f "VllmWorker"   2>/dev/null || true
    pkill -9 -f "EngineCore"   2>/dev/null || true
    pkill -9 -f "multiproc"    2>/dev/null || true
    pkill -9 -f "rl_train.py"  2>/dev/null || true
    sleep 8
    echo "📊 当前 GPU 状态："
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv
    echo ""
}

# 安全的评测调用：跑完后自动清理；输出文件已存在则跳过
# 评测用 vLLM TP=2，需要两张卡都可见
run_eval() {
    local model="$1"
    local data="$2"
    local out="$3"
    local extra="$4"   # 比如 --no_eval

    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "⏭️  跳过：$out 已存在（如要重跑请先删除该文件）"
        return 0
    fi

    echo ">>> 评测：model=$model"
    echo ">>> 数据：$data → 输出：$out"
    CUDA_VISIBLE_DEVICES=0,1 python evaluate.py \
        --model "$model" \
        --data  "$data"  \
        --output "$out" \
        $extra

    cleanup_vllm
}

# ============ Sanity Check：确认步骤1已经跑完 ============
echo "Sanity check..."
for f in splits/train.jsonl splits/val.jsonl sft_data/sft_train.jsonl; do
    if [ ! -f "$f" ]; then
        echo "❌ 缺少 $f，请先跑完 prepare_sft_data.py！"
        exit 1
    fi
done
N_SFT=$(wc -l < sft_data/sft_train.jsonl)
echo "✅ 步骤1产物齐全（$N_SFT 条 SFT 数据），开始训练"
echo ""

# 启动前先清一遍，防止上一轮失败留下的残尸占显存
cleanup_vllm

# ============ [2/9] Base 模型在 val 上的 baseline ============
echo "================================================"
echo "[2/9] 评测 base 模型 baseline (~10 min)"
echo "================================================"
run_eval "Qwen/Qwen3-4B-Thinking-2507" \
         "splits/val.jsonl" \
         "results/eval_base_val.jsonl"

# ============ [3/9] SFT 训练（单卡）============
echo ""
echo "================================================"
echo "[3/9] SFT 训练 (~2h)"
echo "================================================"
if [ -d "ckpts/sft_merged" ] && [ -n "$(ls -A ckpts/sft_merged 2>/dev/null)" ]; then
    echo "⏭️  跳过：ckpts/sft_merged 已存在"
else
    CUDA_VISIBLE_DEVICES=0 python sft_train.py
fi

# ============ [4/9] 评测 SFT ============
echo ""
echo "================================================"
echo "[4/9] 评测 SFT 模型 (~10 min)"
echo "================================================"
run_eval "ckpts/sft_merged" \
         "splits/val.jsonl" \
         "results/eval_sft_val.jsonl"

# ============ [5/9] RL 训练（双卡 DDP，不用 vLLM）============
echo ""
echo "================================================"
echo "[5/9] RL (GRPO) 训练 - 双卡 DDP，HF generate（约 2-3h）"
echo "================================================"
if [ -d "ckpts/rl_merged" ] && [ -n "$(ls -A ckpts/rl_merged 2>/dev/null)" ]; then
    echo "⏭️  跳过：ckpts/rl_merged 已存在"
else
    cleanup_vllm

    set +e
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --multi_gpu \
        --num_processes 2 \
        --num_machines 1 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        rl_train.py --no_vllm
    TRAIN_EXIT=$?
    set -e

    cleanup_vllm

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "❌ rl_train.py 退出码 $TRAIN_EXIT"
        echo "💡 提示：可降低 RL_BATCH_SIZE 或 RL_MAX_COMPLETION_LEN 后重试"
        exit $TRAIN_EXIT
    fi
fi

# ============ [6/9] 评测 RL ============
echo ""
echo "================================================"
echo "[6/9] 评测 RL 模型 (~10 min)"
echo "================================================"
run_eval "ckpts/rl_merged" \
         "splits/val.jsonl" \
         "results/eval_rl_val.jsonl"

# ============ [7-9] 生成 3 份 Kaggle 提交 ============
echo ""
echo "================================================"
echo "[7-9] 在 private.jsonl 上生成 3 份提交"
echo "================================================"

for variant in base sft rl; do
    case $variant in
        base) MODEL="Qwen/Qwen3-4B-Thinking-2507" ;;
        sft)  MODEL="ckpts/sft_merged" ;;
        rl)   MODEL="ckpts/rl_merged" ;;
    esac

    echo ""
    echo ">>> [$variant] 推理 private.jsonl"
    run_eval "$MODEL" \
             "data/private.jsonl" \
             "results/eval_${variant}_private.jsonl" \
             "--no_eval"

    echo ">>> [$variant] 转 CSV"
    if [ -f "results/submission_${variant}.csv" ]; then
        echo "⏭️  跳过：results/submission_${variant}.csv 已存在"
    else
        python convert_to_csv.py \
            --input  "results/eval_${variant}_private.jsonl" \
            --output "results/submission_${variant}.csv"
    fi
done

# ============ 总结 ============
echo ""
echo "🎉 全流程完成！"
echo ""
echo "============ 验证集准确率对比 ============"
python - <<'PY'
import json, os
print(f"{'Model':<8} | {'Total':>6} | {'Correct':>8} | {'Acc':>7}")
print("-" * 40)
for v in ["base", "sft", "rl"]:
    p = f"results/eval_{v}_val.jsonl"
    if not os.path.exists(p):
        print(f"{v:<8} | (not found)")
        continue
    with open(p) as f:
        data = [json.loads(l) for l in f if l.strip()]
    n  = len(data)
    nc = sum(d.get("correct", False) for d in data)
    acc = nc / n * 100 if n else 0.0
    print(f"{v:<8} | {n:>6d} | {nc:>8d} | {acc:>6.2f}%")
PY

echo ""
echo "============ 三份 Kaggle 提交 ============"
ls -lh results/submission_*.csv