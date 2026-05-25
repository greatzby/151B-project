"""
用 vLLM 加载模型，跑数据集生成 + 多数投票 + （可选）评分。

输出 JSONL 每行 1 个 record：
  {
    "id":             ...,
    "is_mcq":         ...,
    "response":       <多数投票胜出的那一条完整回答>,   # ← 提交 CSV 用这个
    "all_responses":  [<n 个候选>...],                  # ← 备查
    "vote_counts":    {归一化答案: 票数, ...},
    "winning_answer": <胜出的归一化答案>,
    "gold":           ... (有 gold 时),
    "correct":        ... (有 gold 时,基于胜出 response 评分)
  }

⚠ 输出每个 id 只占 1 行 → CSV 转换天然正确。
"""
import argparse
import json
import os
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


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    return ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == str(gold_letter).strip().upper()


def canonical_answer(response: str, judger: Judger, is_mcq: bool) -> str:
    """把 response 归一化成可哈希的答案字符串，用于投票分组。"""
    try:
        if is_mcq:
            return extract_letter(response)
        raw = judger.extract_ans(response)
        if not raw:
            return ""
        parts = judger.split_by_comma(raw)
        normalized = []
        for p in parts:
            try:
                normalized.append(judger.norm_ans_str(p))
            except Exception:
                normalized.append(p.strip())
        return " || ".join(normalized)
    except Exception:
        return ""


def majority_vote(responses, judger, is_mcq):
    """对 n 个候选 response 做多数投票，返回 (winning_response, canon, vote_counts)。"""
    if len(responses) <= 1:
        canon = canonical_answer(responses[0], judger, is_mcq) if responses else ""
        return (responses[0] if responses else ""), canon, ({canon: 1} if canon else {})

    canons = [canonical_answer(r, judger, is_mcq) for r in responses]
    valid = [c for c in canons if c]
    if not valid:
        # 所有候选都没提取出答案，退而求其次
        return responses[0], "", {}

    counter = Counter(valid)
    winning_canon, _ = counter.most_common(1)[0]

    # 在原顺序里找第一个 canon == winning_canon 的 response（它的解题过程通常完整）
    for r, c in zip(responses, canons):
        if c == winning_canon:
            return r, winning_canon, dict(counter)

    return responses[0], winning_canon, dict(counter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default=str(MODEL_ID))
    parser.add_argument("--data",   type=str, default=str(VAL_DATA_PATH))
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_tokens",         type=int,   default=MAX_TOKENS)
    parser.add_argument("--temperature",        type=float, default=EVAL_TEMPERATURE)
    parser.add_argument("--top_p",              type=float, default=EVAL_TOP_P)
    parser.add_argument("--top_k",              type=int,   default=EVAL_TOP_K)
    parser.add_argument("--min_p",              type=float, default=EVAL_MIN_P)
    parser.add_argument("--presence_penalty",   type=float, default=EVAL_PRESENCE_PENALTY)
    parser.add_argument("--repetition_penalty", type=float, default=EVAL_REPETITION_PENALTY)
    parser.add_argument("--n",                  type=int,   default=EVAL_N)
    parser.add_argument("--no_eval", action="store_true")
    args = parser.parse_args()

    print(f"🔎 CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"🔎 tensor_parallel_size = {args.tensor_parallel_size}")
    print(f"🔎 sampling: n={args.n}, max_tokens={args.max_tokens}, "
          f"T={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, "
          f"min_p={args.min_p}, pres_pen={args.presence_penalty}, rep_pen={args.repetition_penalty}")

    # 1. 加载数据
    data = load_jsonl(args.data)
    print(f"已加载 {len(data)} 道题 ({args.data})")

    # 2. 加载模型（双卡 mp 后端，绕开 Ray）
    print(f"正在加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    if args.tensor_parallel_size > 1:
        llm_kwargs["distributed_executor_backend"] = "mp"
        llm_kwargs["disable_custom_all_reduce"] = True

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
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

    # 4. 批量生成（每题 n 个候选）
    print(f"正在为 {len(prompts)} 道题各生成 {args.n} 个候选回答...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # 5. 投票 + 评分
    judger = Judger(strict_extract=False)
    results = []
    n_unanimous = 0
    n_majority  = 0
    n_plurality = 0

    for item, output in tqdm(zip(data, outputs), total=len(data), desc="投票&评分"):
        is_mcq = bool(item.get("options"))
        all_responses = [o.text.strip() for o in output.outputs]

        winning_response, winning_canon, vote_counts = majority_vote(
            all_responses, judger, is_mcq
        )

        if vote_counts:
            top_count = max(vote_counts.values())
            if top_count == args.n:
                n_unanimous += 1
            elif top_count > args.n / 2:
                n_majority += 1
            else:
                n_plurality += 1

        record = {
            "id":             item.get("id"),
            "is_mcq":         is_mcq,
            "response":       winning_response,
            "all_responses":  all_responses,
            "vote_counts":    vote_counts,
            "winning_answer": winning_canon,
        }

        if not args.no_eval:
            gold = item["answer"]
            record["gold"] = gold
            try:
                if is_mcq:
                    correct = score_mcq(winning_response, str(gold))
                else:
                    gold_list = gold if isinstance(gold, list) else [gold]
                    correct = judger.auto_judge(
                        pred=winning_response,
                        gold=gold_list,
                        options=[[]] * len(gold_list),
                    )
            except Exception:
                correct = False
            record["correct"] = correct

        results.append(record)

    save_jsonl(args.output, results)
    print(f"✅ 已保存 {len(results)} 条结果到: {args.output}")

    # 投票质量分布
    print(f"\n📊 投票质量分布（n={args.n}）：")
    print(f"  全票一致      : {n_unanimous} / {len(results)}")
    print(f"  绝对多数      : {n_majority} / {len(results)}")
    print(f"  相对多数(分歧): {n_plurality} / {len(results)}")

    # 准确率（如果有 gold）
    if not args.no_eval:
        mcq  = [r for r in results if r["is_mcq"]]
        free = [r for r in results if not r["is_mcq"]]
        def acc(s):
            return sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0
        print("=" * 50)
        print(f"评测结果（多数投票, n={args.n}）")
        print("=" * 50)
        if mcq:
            print(f"  MCQ        : {sum(r['correct'] for r in mcq):4d} / {len(mcq):4d}  ({acc(mcq):.2f}%)")
        if free:
            print(f"  Free-form  : {sum(r['correct'] for r in free):4d} / {len(free):4d}  ({acc(free):.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
        print("=" * 50)


if __name__ == "__main__":
    main()