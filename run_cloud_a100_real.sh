#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found"
  exit 1
fi

if [[ -z "${IMAGENET_PATH:-}" ]]; then
  echo "Please set IMAGENET_PATH before running this script."
  echo "Example: export IMAGENET_PATH=/data/imagenet"
  exit 1
fi

if [[ ! -d "${IMAGENET_PATH}/train" ]]; then
  echo "ImageNet train directory not found at ${IMAGENET_PATH}/train"
  exit 1
fi

if [[ ! -d "${IMAGENET_PATH}/val" ]]; then
  echo "ImageNet val directory not found at ${IMAGENET_PATH}/val"
  exit 1
fi

echo "GPU summary:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "Checking torch CUDA visibility..."
conda run -n jit-local python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

export PYTHONUNBUFFERED=1

conda run -n jit-local python main_jit.py \
  --model JiT-B/32 \
  --img_size 256 \
  --noise_scale 1.0 \
  --batch_size 8 \
  --lr 1e-4 \
  --epochs 10 \
  --warmup_epochs 0 \
  --num_workers 8 \
  --class_num 1000 \
  --data_path "${IMAGENET_PATH}" \
  --output_dir ./output_cloud_a100 \
  --resume ./output_cloud_a100 \
  --save_last_freq 1 \
  --device cuda
