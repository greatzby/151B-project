#!/bin/bash
# 等当前 run 跑完后启动：
# 只跑 private 集（n=5 多数投票），生成 3 份 voted CSV
# 不再跑 val，节省时间，全部火力给 private 推理

set -e
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p results logs

cleanup_vllm() {
    echo ""
    echo "🧹 清理残留 vLLM 进程..."
    pkill -9 -f "VllmWorker"  2>/dev/null || true
    pkill -9 -f "EngineCore"  2>/dev/null || true
    sleep 8
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
}

run_voted() {
    local model="$1"
    local out="$2"

    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "⏭️  跳过：$out 已存在"
        return 0
    fi

    echo ""
    echo ">>> 投票推理：model=$model → 输出：$out"
    CUDA_VISIBLE_DEVICES=0,1 python evaluate.py \
        --model "$model" \
        --data  "data/private.jsonl" \
        --output "$out" \
        --n 5 \
        --max_tokens 32768 \
        --temperature 0.7 \
        --top_p 0.95 \
        --top_k 20 \
        --min_p 0.0 \
        --presence_penalty 0.5 \
        --repetition_penalty 1.0 \
        --no_eval

    cleanup_vllm
}

cleanup_vllm

# ============ 三个模型分别跑 voted private ============
echo ""
echo "============================================================"
echo "[1/3] base 模型 voted private 推理"
echo "============================================================"
run_voted "Qwen/Qwen3-4B-Thinking-2507" \
          "results/eval_base_private_voted.jsonl"

echo ""
echo "============================================================"
echo "[2/3] sft 模型 voted private 推理"
echo "============================================================"
run_voted "ckpts/sft_merged" \
          "results/eval_sft_private_voted.jsonl"

echo ""
echo "============================================================"
echo "[3/3] rl 模型 voted private 推理"
echo "============================================================"
run_voted "ckpts/rl_merged" \
          "results/eval_rl_private_voted.jsonl"

# ============ 转 CSV ============
echo ""
echo "============================================================"
echo "转 3 份 voted CSV"
echo "============================================================"
for variant in base sft rl; do
    if [ -f "results/submission_${variant}_voted.csv" ]; then
        echo "⏭️  跳过：results/submission_${variant}_voted.csv 已存在"
    else
        python convert_to_csv.py \
            --input  "results/eval_${variant}_private_voted.jsonl" \
            --output "results/submission_${variant}_voted.csv"
    fi
done

# ============ 总结 ============
echo ""
echo "🎉 全部完成！"
echo ""
echo "============ 当前所有候选 CSV（共 6 份可提交）============"
ls -lh results/submission_*.csv
echo ""
echo "→ baseline (n=1):"
echo "    submission_base.csv  /  submission_sft.csv  /  submission_rl.csv"
echo "→ voted (n=5, 强采样):"
echo "    submission_base_voted.csv  /  submission_sft_voted.csv  /  submission_rl_voted.csv"
echo ""
echo "全部提交到 Kaggle 看哪个分最高即可。"