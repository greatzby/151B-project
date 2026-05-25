"""所有配置集中在这里，方便调整。"""
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.resolve()
DATA_DIR     = ROOT_DIR / "data"
SPLITS_DIR   = ROOT_DIR / "splits"
SFT_DATA_DIR = ROOT_DIR / "sft_data"
CKPT_DIR     = ROOT_DIR / "ckpts"
RESULTS_DIR  = ROOT_DIR / "results"

PUBLIC_DATA_PATH  = DATA_DIR / "public.jsonl"
PRIVATE_DATA_PATH = DATA_DIR / "private.jsonl"

TRAIN_DATA_PATH = SPLITS_DIR / "train.jsonl"
VAL_DATA_PATH   = SPLITS_DIR / "val.jsonl"
SFT_DATA_PATH   = SFT_DATA_DIR / "sft_train.jsonl"

SFT_CKPT_DIR    = CKPT_DIR / "sft"
SFT_MERGED_DIR  = CKPT_DIR / "sft_merged"
RL_CKPT_DIR     = CKPT_DIR / "rl"
RL_MERGED_DIR   = CKPT_DIR / "rl_merged"

# ── 模型 ──────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
MAX_TOKENS    = 16384      # 单次生成最大 token
MAX_MODEL_LEN = 16384      # vLLM 最大序列长度

# ── 数据划分 ──────────────────────────────────────────────────────
VAL_RATIO = 0.10           # 10% 验证集
SEED      = 42

# ── 拒绝采样 ──────────────────────────────────────────────────────
N_SAMPLES_PER_QUESTION = 4
RS_TEMPERATURE         = 0.8
RS_TOP_P               = 0.95

# ── SFT ───────────────────────────────────────────────────────────
SFT_EPOCHS       = 2
SFT_BATCH_SIZE   = 1
SFT_GRAD_ACCUM   = 8       # 等效 batch_size = 8
SFT_LR           = 1e-4
SFT_MAX_SEQ_LEN  = 8192
LORA_R           = 16
LORA_ALPHA       = 32

# ── RL (GRPO) ─────────────────────────────────────────────────────
RL_MAX_STEPS         = 200      # 总步数（按时间调整）
RL_BATCH_SIZE        = 2
RL_GRAD_ACCUM        = 2
RL_LR                = 5e-6
RL_NUM_GENERATIONS   = 4        # 每个 prompt 生成几个采样
RL_MAX_PROMPT_LEN    = 1024
RL_MAX_COMPLETION_LEN = 1024
RL_BETA              = 0.04     # KL 惩罚系数

# ── 评测 ──────────────────────────────────────────────────────────
EVAL_TEMPERATURE = 0.7
EVAL_TOP_P       = 0.95
EVAL_N           = 1            # 单次生成