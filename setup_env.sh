#!/bin/bash
set -e

echo "========== Step 1: 安装 uv =========="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "========== Step 2: 创建虚拟环境 (Python 3.10) =========="
uv venv .venv --python 3.10 --seed

# 激活
source .venv/bin/activate

echo "========== Step 3: 安装 PyTorch (CUDA 12.1) =========="
uv pip install --upgrade pip wheel setuptools
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "========== Step 4: 安装 HuggingFace + TRL + vLLM =========="
uv pip install \
    "transformers==4.46.3" \
    "accelerate==1.1.1" \
    "peft==0.13.2" \
    "trl==0.13.0" \
    "datasets==3.1.0" \
    "bitsandbytes==0.44.1" \
    sentencepiece \
    safetensors

uv pip install "vllm==0.6.6"

echo "========== Step 5: 安装其他依赖 =========="
uv pip install \
    "sympy==1.13.3" \
    "numpy<2" \
    pandas \
    tqdm \
    "antlr4-python3-runtime==4.11.1"

echo ""
echo "================================================"
echo "✅ 环境安装完成！"
echo "下次开始前先运行：source .venv/bin/activate"
echo "================================================"