#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

JIT_PYTHON="${JIT_PYTHON:-/data/Shenzhen/zhahongli/envs/jit-local/bin/python}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-./toy_cloud_smoke}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-./output_cloud_smoke}"
SAMPLE_OUTPUT_DIR="${SAMPLE_OUTPUT_DIR:-./samples_cloud_smoke}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/cloud_v100_smoke_sample_${RUN_ID}.log"

echo "Smoke sample starting..."
echo "Full log: ${LOG_FILE}"

if [[ ! -x "${JIT_PYTHON}" ]]; then
  echo "Python executable not found: ${JIT_PYTHON}"
  exit 1
fi

if [[ ! -f "${SMOKE_OUTPUT_DIR}/checkpoint-last.pth" ]]; then
  echo "Checkpoint not found: ${SMOKE_OUTPUT_DIR}/checkpoint-last.pth"
  exit 1
fi

if [[ ! -d "${SMOKE_DATA_DIR}/train" ]]; then
  echo "Toy smoke dataset not found: ${SMOKE_DATA_DIR}/train"
  exit 1
fi

mkdir -p "${SAMPLE_OUTPUT_DIR}/inputs" "${SAMPLE_OUTPUT_DIR}/generated"

echo "Saving input examples..."
for class_dir in "${SMOKE_DATA_DIR}"/train/class_*; do
  class_name="$(basename "${class_dir}")"
  first_image="$(find "${class_dir}" -maxdepth 1 -type f -name '*.png' | sort | head -n 1)"
  if [[ -n "${first_image}" ]]; then
    cp "${first_image}" "${SAMPLE_OUTPUT_DIR}/inputs/${class_name}_input.png"
  fi
done

echo "Generating sample images..."
if ! "${JIT_PYTHON}" minimal_sample.py \
  --checkpoint "${SMOKE_OUTPUT_DIR}/checkpoint-last.pth" \
  --output_dir "${SAMPLE_OUTPUT_DIR}/generated" \
  --labels 0,1,2,3 \
  --device cuda >>"${LOG_FILE}" 2>&1; then
  echo "Sample generation failed. Last log lines:"
  tail -n 20 "${LOG_FILE}"
  exit 1
fi

echo "Smoke samples ready."
echo "Inputs: ${SAMPLE_OUTPUT_DIR}/inputs"
echo "Generated: ${SAMPLE_OUTPUT_DIR}/generated"
echo "Full log: ${LOG_FILE}"
