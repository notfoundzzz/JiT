#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

conda run -n jit-local python minimal_sample.py \
  --checkpoint ./output_visual_toy/checkpoint-last.pth \
  --output_dir ./samples_visual_toy \
  --labels 0,0,1,1,2,2,3,3 \
  --steps 20 \
  --cfg 1.5 \
  --device cuda
