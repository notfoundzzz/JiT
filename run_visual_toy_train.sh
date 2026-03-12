#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

rm -rf toy_visual_data

conda run -n jit-local python create_toy_data.py \
  --output_dir ./toy_visual_data \
  --img_size 64 \
  --train_per_class 16 \
  --val_per_class 4 \
  --seed 7

conda run -n jit-local python main_jit.py \
  --model JiT-B/32 \
  --img_size 64 \
  --noise_scale 1.0 \
  --epochs 8 \
  --warmup_epochs 0 \
  --batch_size 8 \
  --lr 1e-4 \
  --num_workers 0 \
  --class_num 4 \
  --data_path ./toy_visual_data \
  --output_dir ./output_visual_toy \
  --save_last_freq 2 \
  --device cuda
