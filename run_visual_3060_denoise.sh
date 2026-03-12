#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

conda run -n jit-local python visual_toy_denoise_demo.py \
  --checkpoint ./output_visual_3060/checkpoint-last.pth \
  --data_dir ./toy_3060_data/train \
  --output_dir ./denoise_visual_3060 \
  --device cuda \
  --timestep 0.35
