"""
SFT 训练，使用 LoRA。训练完会自动 merge LoRA 进 base 模型，保存到 SFT_MERGED_DIR。
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=SFT_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=SFT_BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int,   default=SFT_GRAD_ACCUM)
    parser.add_argument("--lr",         type=float, default=SFT_LR)
    parser.add_argument("--max_seq_len", type=int,  default=SFT_MAX_SEQ_LEN)
    args = parser.parse_args()

    # 1. 加载 SFT 数据
    sft_data = []
    with open(SFT_DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                sft_data.append(json.loads(line))
    print(f"已加载 {len(sft_data)} 条 SFT 样本")

    # 2. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 数据集（用 messages 格式，TRL 会自动只对 assistant 部分计算 loss）
    def to_messages(ex):
        return {
            "messages": [
                {"role": "system",    "content": ex["system"]},
                {"role": "user",      "content": ex["user"]},
                {"role": "assistant", "content": ex["assistant"]},
            ]
        }

    dataset = Dataset.from_list([to_messages(ex) for ex in sft_data])

    # 4. 模型
    print(f"正在加载 base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # 5. LoRA 配置
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. 训练参数
    sft_config = SFTConfig(
        output_dir=str(SFT_CKPT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        max_length=args.max_seq_len,
        packing=False,
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        lr_scheduler_type="cosine",
        dataset_text_field=None,
    )

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # 8. 训练
    trainer.train()
    trainer.save_model(str(SFT_CKPT_DIR))
    tokenizer.save_pretrained(str(SFT_CKPT_DIR))
    print(f"✅ SFT LoRA adapter 保存到: {SFT_CKPT_DIR}")

    # 9. Merge LoRA → 完整模型，方便 RL 和评测加载
    print("正在 merge LoRA 进 base 模型...")
    merged_model = trainer.model.merge_and_unload()
    SFT_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(SFT_MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(SFT_MERGED_DIR))
    print(f"✅ 完整 SFT 模型保存到: {SFT_MERGED_DIR}")


if __name__ == "__main__":
    main()