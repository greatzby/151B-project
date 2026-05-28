"""
run_inference.py — Single entry point for the full inference pipeline.

CLI:
    python run_inference.py --output submission.csv

Programmatic:
    from run_inference import run_inference
    run_inference(output_csv="submission.csv")
"""
import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PRIVATE_DATA_PATH, MAX_TOKENS, MAX_MODEL_LEN, MODEL_ID,
    EVAL_TEMPERATURE, EVAL_TOP_P, EVAL_TOP_K, EVAL_MIN_P,
    EVAL_PRESENCE_PENALTY, EVAL_REPETITION_PENALTY, EVAL_N,
)
from data_utils import load_jsonl, build_prompt
from judger import Judger
from evaluate import majority_vote


# ----------------------------------------------------------------------
# >>> EDIT THESE TWO IDS to your HuggingFace Hub paths after uploading <<<
# ----------------------------------------------------------------------
SFT_MODEL_ID = "greatzby123/qwen3-4b-sft-merged"
RL_MODEL_ID  = "greatzby123/qwen3-4b-rl-merged"

DEFAULT_MODELS = [
    MODEL_ID,        # Qwen/Qwen3-4B-Thinking-2507  (base)
    SFT_MODEL_ID,    # SFT merged
    RL_MODEL_ID,     # RL  merged
]


def _release_vllm(llm):
    """Best-effort GPU cleanup so the next model can be loaded in-process."""
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel, destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def _generate_with_model(model_id, prompts, n, tensor_parallel_size):
    from vllm import LLM, SamplingParams
    print(f"\n========== Generating with {model_id} (n={n}) ==========")

    llm_kwargs = dict(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    if tensor_parallel_size > 1:
        llm_kwargs["distributed_executor_backend"] = "mp"
        llm_kwargs["disable_custom_all_reduce"] = True

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        n=n,
        max_tokens=MAX_TOKENS,
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        top_k=EVAL_TOP_K,
        min_p=EVAL_MIN_P,
        presence_penalty=EVAL_PRESENCE_PENALTY,
        repetition_penalty=EVAL_REPETITION_PENALTY,
    )
    outputs = llm.generate(prompts, sampling_params=sampling)
    all_texts = [[o.text.strip() for o in out.outputs] for out in outputs]
    _release_vllm(llm)
    return all_texts


def run_inference(
    data_path: str = str(PRIVATE_DATA_PATH),
    output_csv: str = "submission.csv",
    model_ids=None,
    tensor_parallel_size: int = 2,
    n_per_model: int = EVAL_N,
):
    """End-to-end inference.

    Loads each model in `model_ids` sequentially, samples `n_per_model`
    completions per question, pools all candidates across all models,
    runs majority vote, and writes the final CSV.
    """
    if model_ids is None:
        model_ids = DEFAULT_MODELS

    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} questions from {data_path}")
    print(f"Models in ensemble: {model_ids}")
    print(f"Total candidates per question = {n_per_model} x {len(model_ids)} "
          f"= {n_per_model * len(model_ids)}")

    # Tokenizer for building prompts (chat template is shared across base/SFT/RL).
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        ))

    # Generate with each model and pool all candidates.
    pooled = [[] for _ in data]
    for mid in model_ids:
        gens = _generate_with_model(mid, prompts, n_per_model, tensor_parallel_size)
        for i, lst in enumerate(gens):
            pooled[i].extend(lst)

    # Majority vote across the full pool.
    judger = Judger(strict_extract=False)
    rows = []
    for item, candidates in tqdm(zip(data, pooled), total=len(data), desc="vote"):
        is_mcq = bool(item.get("options"))
        winning, _, _ = majority_vote(candidates, judger, is_mcq)
        rows.append({"id": item.get("id"), "response": winning})

    df = pd.DataFrame(rows)
    out_path = Path(output_csv)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, default=str(PRIVATE_DATA_PATH))
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--n_per_model", type=int, default=EVAL_N)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override the default model list (HF Hub IDs or local paths)")
    args = parser.parse_args()

    run_inference(
        data_path=args.data,
        output_csv=args.output,
        model_ids=args.models,
        tensor_parallel_size=args.tensor_parallel_size,
        n_per_model=args.n_per_model,
    )