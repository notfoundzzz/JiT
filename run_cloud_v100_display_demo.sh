#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

JIT_PYTHON="${JIT_PYTHON:-/data/Shenzhen/zhahongli/envs/jit-local/bin/python}"
DISPLAY_DATA_DIR="${DISPLAY_DATA_DIR:-./toy_display_data}"
DISPLAY_OUTPUT_DIR="${DISPLAY_OUTPUT_DIR:-./output_display_demo}"
DISPLAY_RENDER_DIR="${DISPLAY_RENDER_DIR:-./display_demo_outputs}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/cloud_v100_display_demo_${RUN_ID}.log"

echo "Display demo starting..."
echo "Full log: ${LOG_FILE}"

if [[ ! -x "${JIT_PYTHON}" ]]; then
  echo "Python executable not found: ${JIT_PYTHON}"
  exit 1
fi

echo "Checking environment..."
"${JIT_PYTHON}" - <<'PY' | tee -a "${LOG_FILE}"
import cv2, torch
print("cv2", cv2.__version__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the uploaded environment.")
print("device", torch.cuda.get_device_name(0))
PY

if [[ ! -d "${DISPLAY_DATA_DIR}/train/class_0" ]]; then
  echo "Creating display demo toy data at ${DISPLAY_DATA_DIR}"
  "${JIT_PYTHON}" create_toy_data.py \
    --output_dir "${DISPLAY_DATA_DIR}" \
    --img_size 64 \
    --train_per_class 64 \
    --val_per_class 8 \
    --seed 13 >>"${LOG_FILE}" 2>&1
fi

echo "Training display-first checkpoint..."
if ! "${JIT_PYTHON}" train_display_demo.py \
  --model JiT-Tiny/16 \
  --img_size 64 \
  --class_num 4 \
  --data_path "${DISPLAY_DATA_DIR}" \
  --output_dir "${DISPLAY_OUTPUT_DIR}" \
  --epochs 80 \
  --batch_size 16 \
  --lr 3e-4 \
  --noise_scale 1.0 \
  --t_min 0.2 \
  --t_max 0.7 \
  --device cuda >>"${LOG_FILE}" 2>&1; then
  echo "Display demo training failed. Last log lines:"
  tail -n 20 "${LOG_FILE}"
  exit 1
fi

echo "Rendering display-first triptychs..."
if ! "${JIT_PYTHON}" display_recon_demo.py \
  --checkpoint "${DISPLAY_OUTPUT_DIR}/checkpoint-last.pth" \
  --data_dir "${DISPLAY_DATA_DIR}/train" \
  --output_dir "${DISPLAY_RENDER_DIR}" \
  --device cuda \
  --timestep 0.35 >>"${LOG_FILE}" 2>&1; then
  echo "Display demo render failed. Last log lines:"
  tail -n 20 "${LOG_FILE}"
  exit 1
fi

echo "Display demo succeeded."
echo "Checkpoint: ${DISPLAY_OUTPUT_DIR}/checkpoint-last.pth"
echo "Triptychs: ${DISPLAY_RENDER_DIR}"
echo "Full log: ${LOG_FILE}"
