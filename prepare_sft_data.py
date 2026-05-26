"""
步骤1：拒绝采样生成 SFT 训练数据。

1. 将 public.jsonl 切成 train/val
2. 用 base 模型对训练集每题生成 N 个回答
3. 用 judger 筛选 **所有** 正确的回答作为 SFT 数据（每题保留全部正确样本）
"""
import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from data_utils import load_jsonl, save_jsonl, split_data, build_prompt
from judger import Judger


def score_response(response, item, judger):
    """评判单个回答是否正确。"""
    gold   = item["answer"]
    is_mcq = bool(item.get("options"))
    try:
        if is_mcq:
            m = re.search(r"\\boxed\{([A-Za-z])\}", response)
            pred = m.group(1).upper() if m else ""
            return pred == str(gold).strip().upper()
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            return judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES_PER_QUESTION)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    args = parser.parse_args()

    # 1. 切分数据
    data = load_jsonl(PUBLIC_DATA_PATH)
    train_data, val_data = split_data(data, val_ratio=VAL_RATIO, seed=SEED)
    print(f"Total: {len(data)} | Train: {len(train_data)} | Val: {len(val_data)}")
    save_jsonl(TRAIN_DATA_PATH, train_data)
    save_jsonl(VAL_DATA_PATH, val_data)
    print(f"已保存切分到: {SPLITS_DIR}")

    # 2. 加载模型 (BF16, 双卡 TP)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        dtype="bfloat16",
        distributed_executor_backend="mp",   # ← 新增
        disable_custom_all_reduce=True,      # ← 新增（关键）
    )

    sampling_params = SamplingParams(
        n=args.n_samples,
        max_tokens=MAX_TOKENS,
        temperature=RS_TEMPERATURE,
        top_p=RS_TOP_P,
    )

    # 3. 构造 prompt
    prompts = []
    for item in train_data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # 4. 批量生成
    print(f"正在为 {len(prompts)} 题各生成 {args.n_samples} 个候选回答...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # 5. 筛选所有正确的回答
    judger  = Judger(strict_extract=False)
    sft_data = []
    n_correct_q   = 0          # 至少有 1 个正确回答的题目数
    n_total_correct = 0        # 总共保留下来的正确样本数
    per_q_correct = []         # 每题正确数的分布

    for item, output in tqdm(zip(train_data, outputs), total=len(outputs), desc="筛选"):
        system, user = build_prompt(item["question"], item.get("options"))

        correct_responses = []
        for sample in output.outputs:
            response = sample.text.strip()
            if score_response(response, item, judger):
                correct_responses.append(response)

        per_q_correct.append(len(correct_responses))

        if correct_responses:
            n_correct_q += 1
            for response in correct_responses:
                sft_data.append({
                    "id":        item.get("id"),
                    "system":    system,
                    "user":      user,
                    "assistant": response,
                    "answer":    item["answer"],
                    "is_mcq":    bool(item.get("options")),
                })
                n_total_correct += 1

    # 统计分布
    from collections import Counter
    dist = Counter(per_q_correct)
    print("=" * 60)
    print(f"📊 拒绝采样统计:")
    print(f"  题目总数              : {len(train_data)}")
    print(f"  至少有一个正确回答的题目: {n_correct_q} ({100*n_correct_q/len(train_data):.1f}%)")
    print(f"  保留 SFT 样本总数      : {n_total_correct}")
    print(f"  每题正确数分布        : {dict(sorted(dist.items()))}")
    if n_correct_q:
        avg = n_total_correct / n_correct_q
        print(f"  有正确回答题目的平均样本数: {avg:.2f}")
    print("=" * 60)

    save_jsonl(SFT_DATA_PATH, sft_data)
    print(f"已保存 {len(sft_data)} 条 SFT 训练数据到: {SFT_DATA_PATH}")


if __name__ == "__main__":
    main()