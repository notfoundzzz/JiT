#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

rm -rf toy_3060_data

conda run -n jit-local python create_toy_data.py \
  --output_dir ./toy_3060_data \
  --img_size 64 \
  --train_per_class 64 \
  --val_per_class 8 \
  --seed 11

conda run -n jit-local python main_jit.py \
  --model JiT-Tiny/16 \
  --img_size 64 \
  --noise_scale 1.0 \
  --epochs 60 \
  --warmup_epochs 0 \
  --batch_size 16 \
  --lr 3e-4 \
  --num_workers 0 \
  --class_num 4 \
  --data_path ./toy_3060_data \
  --output_dir ./output_visual_3060 \
  --save_last_freq 10 \
  --device cuda
