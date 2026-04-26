#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/thesis/gpm

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# Install torch combo that works with the available mamba prebuilt wheels
pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118

# Resolve ABI and Python tag for matching mamba wheel
ABI=$(python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
PYTAG=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
WHEEL_URL="https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu11torch2.6cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"

pip uninstall -y mamba-ssm causal-conv1d || true
pip install --no-cache-dir "${WHEEL_URL}"

# Runtime dependencies used by gpm
pip install --no-cache-dir natsort tensorflow-cpu tensorflow-datasets wandb psutil

# Quick environment verification
python -c "import torch, tensorflow as tf, tensorflow_datasets as tfds, natsort; from mamba_ssm import Mamba; print('python ok'); print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch._C._GLIBCXX_USE_CXX11_ABI); print('tensorflow', tf.__version__); print('mamba_ssm OK')"

# Freeze exact working environment
pip freeze > requirements-lock.txt

echo "Done. Environment is ready and pinned in requirements-lock.txt"
