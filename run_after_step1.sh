#!/bin/bash
# 假设 prepare_sft_data.py（步骤1）已经跑完
# 从 SFT 训练开始，到生成 3 份 Kaggle 提交结束

set -e
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH=$(python -c "import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH


mkdir -p splits sft_data ckpts results

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

# ============ [2/9] Base 模型在 val 上的 baseline ============
echo "================================================"
echo "[2/9] 评测 base 模型 baseline (~10 min)"
echo "================================================"
python evaluate.py \
    --model "Qwen/Qwen3-4B-Thinking-2507" \
    --data splits/val.jsonl \
    --output results/eval_base_val.jsonl

# ============ [3/9] SFT 训练 ============
echo ""
echo "================================================"
echo "[3/9] SFT 训练 (~2h)"
echo "================================================"
CUDA_VISIBLE_DEVICES=0 python sft_train.py

# ============ [4/9] 评测 SFT ============
echo ""
echo "================================================"
echo "[4/9] 评测 SFT 模型 (~10 min)"
echo "================================================"
python evaluate.py \
    --model ckpts/sft_merged \
    --data splits/val.jsonl \
    --output results/eval_sft_val.jsonl

# ============ [5/9] RL 训练 ============
echo ""
echo "================================================"
echo "[5/9] RL (GRPO) 训练 (~8-10h)"
echo "================================================"
python rl_train.py

# ============ [6/9] 评测 RL ============
echo ""
echo "================================================"
echo "[6/9] 评测 RL 模型 (~10 min)"
echo "================================================"
python evaluate.py \
    --model ckpts/rl_merged \
    --data splits/val.jsonl \
    --output results/eval_rl_val.jsonl

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
    python evaluate.py \
        --model "$MODEL" \
        --data data/private.jsonl \
        --output "results/eval_${variant}_private.jsonl" \
        --no_eval

    echo ">>> [$variant] 转 CSV"
    python convert_to_csv.py \
        --input "results/eval_${variant}_private.jsonl" \
        --output "results/submission_${variant}.csv"
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
    print(f"{v:<8} | {n:>6d} | {nc:>8d} | {nc/n*100:>6.2f}%")
PY

echo ""
echo "============ 三份 Kaggle 提交 ============"
ls -lh results/submission_*.csv