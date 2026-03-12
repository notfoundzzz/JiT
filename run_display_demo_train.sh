#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

rm -rf toy_display_data

conda run -n jit-local python create_toy_data.py \
  --output_dir ./toy_display_data \
  --img_size 64 \
  --train_per_class 64 \
  --val_per_class 8 \
  --seed 13

conda run -n jit-local python train_display_demo.py \
  --model JiT-Tiny/16 \
  --img_size 64 \
  --class_num 4 \
  --data_path ./toy_display_data \
  --output_dir ./output_display_demo \
  --epochs 80 \
  --batch_size 16 \
  --lr 3e-4 \
  --noise_scale 1.0 \
  --t_min 0.2 \
  --t_max 0.7 \
  --device cuda
