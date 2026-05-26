"""所有配置集中在这里，方便调整。"""
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.resolve()
DATA_DIR     = ROOT_DIR / "data"
SPLITS_DIR   = ROOT_DIR / "splits_v2"
SFT_DATA_DIR = ROOT_DIR / "sft_data_v2"
CKPT_DIR     = ROOT_DIR / "ckpts_v2"
RESULTS_DIR  = ROOT_DIR / "results_v2"

PUBLIC_DATA_PATH  = DATA_DIR / "public_v2.jsonl"
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
MAX_TOKENS    = 32768       # 评测 / 拒绝采样的最大生成长度
MAX_MODEL_LEN = 32768

# ── 数据划分 ──────────────────────────────────────────────────────
VAL_RATIO = 0.10
SEED      = 42

# ── 拒绝采样 ──────────────────────────────────────────────────────
N_SAMPLES_PER_QUESTION = 3      # 每题采样 3 个候选
RS_TEMPERATURE         = 0.8
RS_TOP_P               = 0.95

# ── SFT ───────────────────────────────────────────────────────────
SFT_EPOCHS       = 2
SFT_BATCH_SIZE   = 1
SFT_GRAD_ACCUM   = 8
SFT_LR           = 1e-4
SFT_MAX_SEQ_LEN  = 16384        # 16K，覆盖大部分 thinking trajectory
LORA_R           = 16
LORA_ALPHA       = 32

# ── RL (GRPO) ─────────────────────────────────────────────────────
RL_MAX_STEPS         = 150      # 12h 预算下减半
RL_BATCH_SIZE        = 2
RL_GRAD_ACCUM        = 2
RL_LR                = 5e-6
RL_NUM_GENERATIONS   = 2        # 显存换长度（注意：=2 时 advantage 方差大但能跑）
RL_MAX_PROMPT_LEN    = 2048     # 防题目截断
RL_MAX_COMPLETION_LEN = 8192    # thinking 模型必须给足空间
RL_BETA              = 0.04

# ── 评测（n=5 多数投票）─────────────────────────────────────────
EVAL_TEMPERATURE        = 0.7
EVAL_TOP_P              = 0.95
EVAL_TOP_K              = 20
EVAL_MIN_P              = 0.0
EVAL_PRESENCE_PENALTY   = 0.5
EVAL_REPETITION_PENALTY = 1.0
EVAL_N                  = 5      # 多数投票