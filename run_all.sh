#!/bin/bash
set -e   # 出错就退出
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

mkdir -p splits sft_data ckpts results

echo ""
echo "================================================"
echo "[1/7] 拒绝采样：生成 SFT 训练数据 (~1.5h)"
echo "================================================"
python prepare_sft_data.py

echo ""
echo "================================================"
echo "[2/7] 评测 base 模型（baseline 准确率）"
echo "================================================"
python evaluate.py \
    --model "Qwen/Qwen3-4B-Thinking-2507" \
    --data splits/val.jsonl \
    --output results/eval_base_val.jsonl

echo ""
echo "================================================"
echo "[3/7] SFT 训练 (~2h)"
echo "================================================"
CUDA_VISIBLE_DEVICES=0 python sft_train.py

echo ""
echo "================================================"
echo "[4/7] 评测 SFT 模型"
echo "================================================"
python evaluate.py \
    --model ckpts/sft_merged \
    --data splits/val.jsonl \
    --output results/eval_sft_val.jsonl

echo ""
echo "================================================"
echo "[5/7] RL (GRPO) 训练 (~8-10h)"
echo "================================================"
python rl_train.py

echo ""
echo "================================================"
echo "[6/7] 评测 RL 模型"
echo "================================================"
python evaluate.py \
    --model ckpts/rl_merged \
    --data splits/val.jsonl \
    --output results/eval_rl_val.jsonl

echo ""
echo "================================================"
echo "[7/7] 用最好的模型在 private.jsonl 上跑预测 + 转 CSV"
echo "================================================"
# 默认用 RL 模型，如果 SFT 更好可改成 ckpts/sft_merged
python evaluate.py \
    --model ckpts/rl_merged \
    --data data/private.jsonl \
    --output results/eval_rl_private.jsonl \
    --no_eval

python convert_to_csv.py \
    --input results/eval_rl_private.jsonl \
    --output results/submission.csv

echo ""
echo "🎉 全流程完成！"
echo "提交文件: results/submission.csv"