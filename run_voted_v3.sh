#!/bin/bash
# v2 版（用新 judger + ckpts_v2 模型重跑）
# private 推理 + 多数投票（n=5），生成 3 份 voted CSV
# 输出全部进 results_v2/，不会动 results/ 下的旧产物

set -e
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p results_v3 logs

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

# ============ 三个模型分别跑 voted private（v2） ============
echo ""
echo "============================================================"
echo "[1/3] base 模型 voted private 推理（v2）"
echo "============================================================"
run_voted "Qwen/Qwen3-4B-Thinking-2507" \
          "results_v3/eval_base_private_voted_long_v3.jsonl"

echo ""
echo "============================================================"
echo "[2/3] sft 模型 voted private 推理（v2，ckpts_v2/sft_merged）"
echo "============================================================"
run_voted "ckpts_v3/sft_merged" \
          "results_v3/eval_sft_private_voted_long_v3.jsonl"

echo ""
echo "============================================================"
echo "[3/3] rl 模型 voted private 推理（v2，ckpts_v2/rl_merged）"
echo "============================================================"
run_voted "ckpts_v3/rl_merged" \
          "results_v3/eval_rl_private_voted_long_v3.jsonl"

# ============ 转 CSV ============
echo ""
echo "============================================================"
echo "转 3 份 voted CSV（v2）"
echo "============================================================"
for variant in base sft rl; do
    csv_out="results_v3/submission_${variant}_voted_long_v3.csv"
    jsonl_in="results_v3/eval_${variant}_private_voted_long_v3.jsonl"
    if [ -f "$csv_out" ]; then
        echo "⏭️  跳过：$csv_out 已存在"
    else
        python convert_to_csv.py \
            --input  "$jsonl_in" \
            --output "$csv_out"
    fi
done

# ============ 总结 ============
echo ""
echo "🎉 全部完成！"
echo ""
echo "全部提交 Kaggle 看哪个分最高即可。"