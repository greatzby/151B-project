"""
用 SFT merged 模型对 train set 每题采样 N 次，算 pass rate，
过滤出难度适中（既不全对也不全错）的题作为 RL 训练数据。
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data_utils import load_jsonl, save_jsonl, build_prompt
from judger import Judger


def score_response(response, item, judger):
    gold = item["answer"]
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
    parser.add_argument("--model", type=str, default=str(SFT_MERGED_DIR),
                        help="算 pass rate 用的模型（默认 SFT merged）")
    parser.add_argument("--input", type=str, default=str(TRAIN_DATA_PATH))
    parser.add_argument("--output", type=str,
                        default=str(SPLITS_DIR / "train_filtered.jsonl"))
    parser.add_argument("--meta_output", type=str,
                        default=str(SPLITS_DIR / "pass_rate.jsonl"))
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--max_tokens", type=int, default=RL_MAX_COMPLETION_LEN,
                        help="生成最大 token 数（默认与 RL 训练一致，保证分布匹配）")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--keep_min_correct", type=int, default=1,
                        help="至少正确 N 次才保留（默认 1）")
    parser.add_argument("--keep_min_wrong", type=int, default=1,
                        help="至少错误 N 次才保留（默认 1）")
    args = parser.parse_args()

    # 1. 加载数据
    train_data = load_jsonl(args.input)
    print(f"已加载 {len(train_data)} 条训练数据")

    # 2. 加载模型
    print(f"加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        n=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # 3. 构造 prompt
    prompts = []
    for item in train_data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # 4. 批量生成
    print(f"为 {len(prompts)} 题各生成 {args.n_samples} 个候选...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # 5. 评分 + 过滤
    judger = Judger(strict_extract=False)
    n_correct_dist = Counter()
    meta_records = []
    filtered_data = []

    for item, output in tqdm(zip(train_data, outputs), total=len(outputs), desc="评分"):
        n_correct = sum(
            1 for sample in output.outputs
            if score_response(sample.text.strip(), item, judger)
        )
        n_wrong = args.n_samples - n_correct
        pass_rate = n_correct / args.n_samples

        n_correct_dist[n_correct] += 1
        meta_records.append({
            "id": item.get("id"),
            "n_correct": n_correct,
            "n_total": args.n_samples,
            "pass_rate": pass_rate,
        })

        if n_correct >= args.keep_min_correct and n_wrong >= args.keep_min_wrong:
            filtered_data.append(item)

    # 6. 输出统计
    print("=" * 60)
    print(f"📊 Pass rate 分布 (n={args.n_samples}):")
    for k in sorted(n_correct_dist.keys()):
        bar = "█" * int(40 * n_correct_dist[k] / len(train_data))
        print(f"  {k}/{args.n_samples} 对 : {n_correct_dist[k]:4d} 题 "
              f"({100 * n_correct_dist[k] / len(train_data):5.1f}%) {bar}")
    print("─" * 60)
    print(f"  原始题数 : {len(train_data)}")
    print(f"  过滤后  : {len(filtered_data)} "
          f"({100 * len(filtered_data) / len(train_data):.1f}%)")
    print(f"  过滤条件: ≥{args.keep_min_correct} 对 且 ≥{args.keep_min_wrong} 错")
    print("=" * 60)

    save_jsonl(args.output, filtered_data)
    save_jsonl(args.meta_output, meta_records)
    print(f"✅ 过滤后数据保存到: {args.output}")
    print(f"✅ pass rate 元信息保存到: {args.meta_output}")


if __name__ == "__main__":
    main()