#!/bin/bash
# install_training.sh
# Installs Unsloth + training dependencies for CUDA 12.0 on RTX 4090.
# Run once before finetune.py.

set -e

echo "Installing PyTorch with CUDA 12.1 support..."
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing Unsloth + training stack..."
uv add "unsloth[cu121-torch260] @ git+https://github.com/unslothai/unsloth.git"
uv add trl peft accelerate bitsandbytes datasets

echo "Verifying GPU access..."
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "All set. Run training with:"
echo "  uv run python finetune.py"
echo "  uv run python finetune.py --max-steps 100   # smoke test"
