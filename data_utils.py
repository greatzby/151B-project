"""数据加载和 prompt 构造。"""
import json
import random
from pathlib import Path
from typing import Optional


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_data(data, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_val = int(len(data) * val_ratio)
    val_idx = set(indices[:n_val])
    train = [d for i, d in enumerate(data) if i not in val_idx]
    val   = [d for i, d in enumerate(data) if i in val_idx]
    return train, val


# ── Prompt 模板（保留你原版的） ─────────────────────────────────────
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem carefully but concisely. "
    "The problem may contain one or multiple blanks marked [ANS]. "
    "Return the final answer values in the same order as the blanks. "
    "If there are multiple answers, separate them by commas inside one single \\boxed{}. "
    "Do not include labels like a) or b) inside the box. "
    "End your response with exactly one final answer in the form \\boxed{...}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output only the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list]):
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question