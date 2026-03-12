#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

conda run -n jit-local python display_recon_demo.py \
  --checkpoint ./output_display_demo/checkpoint-last.pth \
  --data_dir ./toy_display_data/train \
  --output_dir ./display_demo_outputs \
  --device cuda \
  --timestep 0.35
