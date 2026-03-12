#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

if [[ ! -d toy_data/train/class_0 ]]; then
  python create_toy_data.py
fi

conda run -n jit-local python main_jit.py \
  --model JiT-B/32 \
  --img_size 64 \
  --noise_scale 1.0 \
  --epochs 1 \
  --warmup_epochs 0 \
  --batch_size 1 \
  --blr 1e-5 \
  --num_workers 0 \
  --class_num 1 \
  --data_path ./toy_data \
  --output_dir ./output_local \
  --save_last_freq 1 \
  --device cuda
