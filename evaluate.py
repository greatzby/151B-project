"""
用 vLLM 加载模型，跑数据集生成 + 评分。

支持任意模型路径：base / SFT merged / RL merged。
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
from data_utils import load_jsonl, save_jsonl, build_prompt
from judger import Judger


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    return ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == str(gold_letter).strip().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default=str(MODEL_ID),
                        help="模型名或本地路径")
    parser.add_argument("--data",   type=str, default=str(VAL_DATA_PATH),
                        help="JSONL 数据文件")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSONL 路径")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_tokens",  type=int,   default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=EVAL_TEMPERATURE)
    parser.add_argument("--top_p",       type=float, default=EVAL_TOP_P)
    parser.add_argument("--n",           type=int,   default=EVAL_N)
    parser.add_argument("--no_eval",     action="store_true",
                        help="不评分（用于 private.jsonl 没有 gold answer 的情况）")
    args = parser.parse_args()

    # 1. 加载数据
    data = load_jsonl(args.data)
    print(f"已加载 {len(data)} 道题 ({args.data})")

    # 2. 加载模型
    print(f"正在加载模型: {args.model}")
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
    )

    sampling_params = SamplingParams(
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=20,
        repetition_penalty=1.0,
    )

    # 3. 构造 prompt
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # 4. 批量生成
    print(f"正在生成 {len(prompts)} 个回答...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = [out.outputs[0].text.strip() for out in outputs]

    # 5. 评分 (如果有 gold)
    judger = Judger(strict_extract=False) if not args.no_eval else None
    results = []

    for item, response in tqdm(zip(data, responses), total=len(data), desc="评分"):
        is_mcq = bool(item.get("options"))
        record = {"id": item.get("id"), "is_mcq": is_mcq, "response": response}

        if not args.no_eval:
            gold = item["answer"]
            record["gold"] = gold
            try:
                if is_mcq:
                    correct = score_mcq(response, str(gold))
                else:
                    gold_list = gold if isinstance(gold, list) else [gold]
                    correct = judger.auto_judge(
                        pred=response,
                        gold=gold_list,
                        options=[[]] * len(gold_list),
                    )
            except Exception:
                correct = False
            record["correct"] = correct

        results.append(record)

    save_jsonl(args.output, results)
    print(f"✅ 已保存 {len(results)} 条结果到: {args.output}")

    # 6. 打印准确率
    if not args.no_eval:
        mcq  = [r for r in results if r["is_mcq"]]
        free = [r for r in results if not r["is_mcq"]]

        def acc(s):
            return sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0

        print("=" * 50)
        print("评测结果")
        print("=" * 50)
        if mcq:
            print(f"  MCQ        : {sum(r['correct'] for r in mcq):4d} / {len(mcq):4d}  ({acc(mcq):.2f}%)")
        if free:
            print(f"  Free-form  : {sum(r['correct'] for r in free):4d} / {len(free):4d}  ({acc(free):.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
        print("=" * 50)


if __name__ == "__main__":
    main()