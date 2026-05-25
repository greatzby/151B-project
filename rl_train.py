"""
GRPO RL 训练，用 judger 作为 reward function。

资源分配（双卡）：
- GPU 0：训练
- GPU 1：vLLM 加速采样
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data_utils import load_jsonl, build_prompt
from judger import Judger


# 全局 judger（用于 reward）
_judger = Judger(strict_extract=False)


def reward_correctness(completions, **kwargs):
    """主 reward：答对 +1，答错 0"""
    answers  = kwargs.get("answer", [])
    is_mcqs  = kwargs.get("is_mcq", [])

    rewards = []
    for i, completion in enumerate(completions):
        gold   = answers[i]   if i < len(answers)  else None
        is_mcq = is_mcqs[i]   if i < len(is_mcqs)  else False

        # 现在 gold 永远是 list[str]（在 main 里统一过了）
        if gold is None or (isinstance(gold, list) and len(gold) == 0):
            rewards.append(0.0)
            continue
        try:
            if is_mcq:
                m = re.search(r"\\boxed\{([A-Za-z])\}", completion)
                pred = m.group(1).upper() if m else ""
                # MCQ 题：gold 是 list，取第一个元素作为正确选项
                gold_str = gold[0] if isinstance(gold, list) else gold
                correct = pred == str(gold_str).strip().upper()
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                correct = _judger.auto_judge(
                    pred=completion,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            rewards.append(1.0 if correct else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_format(completions, **kwargs):
    """副 reward：有正确格式的 \\boxed{} 给 0.1 分（鼓励格式规范）"""
    rewards = []
    for c in completions:
        if re.search(r"\\boxed\{[^}]+\}", c):
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(SFT_MERGED_DIR),
                        help="SFT merged 模型路径")
    parser.add_argument("--max_steps", type=int, default=RL_MAX_STEPS)
    parser.add_argument("--lr", type=float, default=RL_LR)
    parser.add_argument("--num_generations", type=int, default=RL_NUM_GENERATIONS)
    parser.add_argument("--no_vllm", action="store_true",
                        help="禁用 vLLM 加速（fallback 到 HF 生成，会慢很多）")
    args = parser.parse_args()

    # 1. 加载训练数据
    train_data = load_jsonl(TRAIN_DATA_PATH)
    print(f"已加载 {len(train_data)} 条训练数据")

    # 2. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 数据集
    examples = []
    for item in train_data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # 🔧 统一 answer 为 list[str]，避免 PyArrow 类型混乱：
        # MCQ 题原本是 "A"，free-form 是 ["xxx", "yyy"]，
        # PyArrow 不允许同一列既有 str 又有 list，会报
        # "cannot mix list and non-list, non-null values"
        ans = item["answer"]
        if ans is None:
            ans = []
        elif not isinstance(ans, list):
            ans = [str(ans)]
        else:
            ans = [str(x) for x in ans]

        examples.append({
            "prompt": prompt,
            "answer": ans,                       # 永远是 list[str]
            "is_mcq": bool(item.get("options")),
        })
    dataset = Dataset.from_list(examples)

    # 4. LoRA（在 SFT merged 模型基础上训练新的 LoRA）
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. GRPO 配置
    grpo_config = GRPOConfig(
        output_dir=str(RL_CKPT_DIR),
        max_steps=args.max_steps,
        per_device_train_batch_size=RL_BATCH_SIZE,
        gradient_accumulation_steps=RL_GRAD_ACCUM,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        num_generations=args.num_generations,
        max_prompt_length=RL_MAX_PROMPT_LEN,
        max_completion_length=RL_MAX_COMPLETION_LEN,
        temperature=0.9,
        beta=RL_BETA,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=not args.no_vllm,
        vllm_device="cuda:1",                # vLLM 用 GPU 1
        vllm_gpu_memory_utilization=0.6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
    )

    # 6. Trainer
    trainer = GRPOTrainer(
        model=args.model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_correctness, reward_format],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # 7. 训练
    trainer.train()
    trainer.save_model(str(RL_CKPT_DIR))
    tokenizer.save_pretrained(str(RL_CKPT_DIR))
    print(f"✅ RL LoRA adapter 保存到: {RL_CKPT_DIR}")

    # 8. Merge LoRA → 完整模型
    print("正在 merge RL LoRA 进 SFT merged 模型...")
    merged_model = trainer.model.merge_and_unload()
    RL_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(RL_MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(RL_MERGED_DIR))
    print(f"✅ 完整 RL 模型保存到: {RL_MERGED_DIR}")


if __name__ == "__main__":
    main()