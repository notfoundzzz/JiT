#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

if [[ -z "${IMAGENET_PATH:-}" ]]; then
  echo "Please set IMAGENET_PATH before running this script."
  echo "Example: export IMAGENET_PATH=/path/to/imagenet"
  exit 1
fi

if [[ ! -d "${IMAGENET_PATH}/train" ]]; then
  echo "ImageNet train directory not found at ${IMAGENET_PATH}/train"
  exit 1
fi

conda run -n jit-local python main_jit.py \
  --model JiT-B/32 \
  --img_size 256 \
  --noise_scale 1.0 \
  --batch_size 8 \
  --lr 1e-4 \
  --epochs 5 \
  --warmup_epochs 0 \
  --num_workers 8 \
  --class_num 1000 \
  --data_path "${IMAGENET_PATH}" \
  --output_dir ./output_imagenet_a100 \
  --resume ./output_imagenet_a100 \
  --save_last_freq 1 \
  --device cuda
