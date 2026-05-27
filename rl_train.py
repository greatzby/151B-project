"""
GRPO RL 训练。本版本支持：
  --train_data  指定训练数据路径（默认 TRAIN_DATA_PATH）
  --temperature 采样温度（默认 0.9）
  --beta        KL 系数（默认 RL_BETA）
"""
import argparse
import inspect
import json
import os
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


_judger = Judger(strict_extract=False)


def reward_correctness(completions, **kwargs):
    answers = kwargs.get("answer", [])
    is_mcqs = kwargs.get("is_mcq", [])

    rewards = []
    for i, completion in enumerate(completions):
        gold = answers[i] if i < len(answers) else None
        is_mcq = is_mcqs[i] if i < len(is_mcqs) else False

        if gold is None or (isinstance(gold, list) and len(gold) == 0):
            rewards.append(0.0)
            continue
        try:
            if is_mcq:
                m = re.search(r"\\boxed\{([A-Za-z])\}", completion)
                pred = m.group(1).upper() if m else ""
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
    rewards = []
    for c in completions:
        if re.search(r"\\boxed\{[^}]+\}", c):
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def build_grpo_config(args, per_device_bs):
    valid = set(inspect.signature(GRPOConfig.__init__).parameters)
    print(f"🔎 GRPOConfig 检测："
          f" use_vllm={'use_vllm' in valid},"
          f" vllm_mode={'vllm_mode' in valid},"
          f" vllm_device={'vllm_device' in valid}")

    kwargs = dict(
        output_dir=str(RL_CKPT_DIR),
        max_steps=args.max_steps,
        per_device_train_batch_size=per_device_bs,
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
        temperature=args.temperature,                # ← 改：从 args 读
        beta=args.beta,                              # ← 改：从 args 读
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
    )

    if not args.no_vllm:
        if "use_vllm" in valid:
            kwargs["use_vllm"] = True
        if "vllm_mode" in valid:
            kwargs["vllm_mode"] = "colocate"
            print("✅ 使用 TRL 新版 vLLM API: vllm_mode='colocate'")
        elif "vllm_device" in valid:
            kwargs["vllm_device"] = args.vllm_device
            print(f"✅ 使用 TRL 旧版 vLLM API: vllm_device='{args.vllm_device}'")
        if "vllm_gpu_memory_utilization" in valid:
            kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_mem
        if "vllm_dtype" in valid:
            kwargs["vllm_dtype"] = "bfloat16"
        if "vllm_max_model_len" in valid:
            kwargs["vllm_max_model_len"] = RL_MAX_PROMPT_LEN + RL_MAX_COMPLETION_LEN
    else:
        if "use_vllm" in valid:
            kwargs["use_vllm"] = False
        print("⚠️  --no_vllm 已开启，将使用 HuggingFace generate")

    dropped = [k for k in kwargs if k not in valid]
    if dropped:
        print(f"⚠️  当前 TRL 不支持以下参数，将被忽略: {dropped}")
    kwargs = {k: v for k, v in kwargs.items() if k in valid}

    return GRPOConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(SFT_MERGED_DIR))
    parser.add_argument("--train_data", type=str, default=str(TRAIN_DATA_PATH),  # ← 新增
                        help="训练数据路径，默认用 config 里的 TRAIN_DATA_PATH")
    parser.add_argument("--max_steps", type=int, default=RL_MAX_STEPS)
    parser.add_argument("--lr", type=float, default=RL_LR)
    parser.add_argument("--num_generations", type=int, default=RL_NUM_GENERATIONS)
    parser.add_argument("--temperature", type=float, default=0.9)        # ← 新增
    parser.add_argument("--beta", type=float, default=RL_BETA)           # ← 新增
    parser.add_argument("--no_vllm", action="store_true")
    parser.add_argument("--vllm_gpu_mem", type=float, default=0.30)
    parser.add_argument("--vllm_device", type=str, default="cuda:0")
    args = parser.parse_args()

    num_procs = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_bs = max(RL_BATCH_SIZE, args.num_generations)
    while (per_device_bs * num_procs) % args.num_generations != 0:
        per_device_bs += 1
    global_bs = per_device_bs * num_procs
    unique_prompts_per_step = global_bs // args.num_generations
    print(f"📐 num_processes        = {num_procs}")
    print(f"📐 per_device_batch_size = {per_device_bs}")
    print(f"📐 global_batch_size    = {global_bs}")
    print(f"📐 num_generations      = {args.num_generations}")
    print(f"📐 unique_prompts/step  = {unique_prompts_per_step}")
    print(f"📐 lr={args.lr}, temperature={args.temperature}, beta={args.beta}")

    # 1. 加载训练数据 ← 改：从 args 读
    train_data = load_jsonl(args.train_data)
    print(f"已加载 {len(train_data)} 条训练数据 from {args.train_data}")

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
             {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ans = item["answer"]
        if ans is None:
            ans = []
        elif not isinstance(ans, list):
            ans = [str(ans)]
        else:
            ans = [str(x) for x in ans]
        examples.append({
            "prompt": prompt,
            "answer": ans,
            "is_mcq": bool(item.get("options")),
        })
    dataset = Dataset.from_list(examples)

    # 4. LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_config = build_grpo_config(args, per_device_bs)

    trainer = GRPOTrainer(
        model=args.model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_correctness, reward_format],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(RL_CKPT_DIR))
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(str(RL_CKPT_DIR))
        print(f"✅ RL LoRA adapter 保存到: {RL_CKPT_DIR}")

    trainer.accelerator.wait_for_everyone()

    if trainer.accelerator.is_main_process:
        print("正在 merge RL LoRA 进 SFT merged 模型...")
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        merged_model = unwrapped.merge_and_unload()
        RL_MERGED_DIR.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(RL_MERGED_DIR), safe_serialization=True)
        tokenizer.save_pretrained(str(RL_MERGED_DIR))
        print(f"✅ 完整 RL 模型保存到: {RL_MERGED_DIR}")


if __name__ == "__main__":
    main()