#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

JIT_PYTHON="${JIT_PYTHON:-/data/Shenzhen/zhahongli/envs/jit-local/bin/python}"
VISUAL_DATA_DIR="${VISUAL_DATA_DIR:-./toy_cloud_visual}"
VISUAL_OUTPUT_DIR="${VISUAL_OUTPUT_DIR:-./output_cloud_visual}"
VISUAL_SAMPLE_DIR="${VISUAL_SAMPLE_DIR:-./samples_cloud_visual}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/cloud_v100_visual_toy_${RUN_ID}.log"

echo "Visual toy run starting..."
echo "Full log: ${LOG_FILE}"

if [[ ! -x "${JIT_PYTHON}" ]]; then
  echo "Python executable not found: ${JIT_PYTHON}"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found"
  exit 1
fi

echo "GPU summary:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

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

if [[ ! -d "${VISUAL_DATA_DIR}/train/class_0" ]]; then
  echo "Creating toy visual dataset at ${VISUAL_DATA_DIR}"
  "${JIT_PYTHON}" create_toy_data.py \
    --output_dir "${VISUAL_DATA_DIR}" \
    --img_size 64 \
    --train_per_class 16 \
    --val_per_class 4 \
    --seed 0 >>"${LOG_FILE}" 2>&1
fi

mkdir -p "${VISUAL_OUTPUT_DIR}"

echo "Training visual toy checkpoint..."
if ! "${JIT_PYTHON}" main_jit.py \
  --model JiT-B/32 \
  --img_size 64 \
  --noise_scale 1.0 \
  --epochs 20 \
  --warmup_epochs 0 \
  --batch_size 4 \
  --blr 1e-3 \
  --num_workers 0 \
  --class_num 4 \
  --data_path "${VISUAL_DATA_DIR}" \
  --output_dir "${VISUAL_OUTPUT_DIR}" \
  --resume "${VISUAL_OUTPUT_DIR}" \
  --save_last_freq 5 \
  --device cuda >>"${LOG_FILE}" 2>&1; then
  echo "Visual toy training failed. Last log lines:"
  tail -n 20 "${LOG_FILE}"
  exit 1
fi

if [[ ! -f "${VISUAL_OUTPUT_DIR}/checkpoint-last.pth" ]]; then
  echo "Visual toy run finished but checkpoint-last.pth was not created."
  tail -n 20 "${LOG_FILE}"
  exit 1
fi

mkdir -p "${VISUAL_SAMPLE_DIR}/inputs" "${VISUAL_SAMPLE_DIR}/generated"

echo "Saving input examples..."
for class_dir in "${VISUAL_DATA_DIR}"/train/class_*; do
  class_name="$(basename "${class_dir}")"
  first_image="$(find "${class_dir}" -maxdepth 1 -type f -name '*.png' | sort | head -n 1)"
  if [[ -n "${first_image}" ]]; then
    cp "${first_image}" "${VISUAL_SAMPLE_DIR}/inputs/${class_name}_input.png"
  fi
done

echo "Generating visual toy samples..."
if ! "${JIT_PYTHON}" minimal_sample.py \
  --checkpoint "${VISUAL_OUTPUT_DIR}/checkpoint-last.pth" \
  --output_dir "${VISUAL_SAMPLE_DIR}/generated" \
  --labels 0,1,2,3 \
  --cfg 1.0 \
  --device cuda >>"${LOG_FILE}" 2>&1; then
  echo "Visual toy sample generation failed. Last log lines:"
  tail -n 20 "${LOG_FILE}"
  exit 1
fi

echo "Visual toy run succeeded."
echo "Checkpoint: ${VISUAL_OUTPUT_DIR}/checkpoint-last.pth"
echo "Inputs: ${VISUAL_SAMPLE_DIR}/inputs"
echo "Generated: ${VISUAL_SAMPLE_DIR}/generated"
echo "Full log: ${LOG_FILE}"
