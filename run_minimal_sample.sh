#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

conda run -n jit-local python minimal_sample.py \
  --checkpoint ./output_local/checkpoint-last.pth \
  --output_dir ./samples_local \
  --num_samples 4 \
  --device cuda
