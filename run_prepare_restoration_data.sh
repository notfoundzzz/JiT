#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

JIT_PYTHON="${JIT_PYTHON:-/data/Shenzhen/zhahongli/envs/jit-local/bin/python}"
SOURCE_DIR="${SOURCE_DIR:-${REPO_DIR}/source_images_kodak}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/paired_JiT-image-to-image}"
IMG_SIZE="${IMG_SIZE:-512}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
SEED="${SEED:-0}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"

mkdir -p "${LOG_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/prepare_restoration_data_${RUN_ID}.log"
LATEST_LOG_LINK="${LOG_DIR}/prepare_restoration_data_latest.log"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG_LINK}"

echo "Preparing paired restoration data..."
echo "Source dir: ${SOURCE_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Image size: ${IMG_SIZE}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Full log: ${LOG_FILE}"

if [[ ! -x "${JIT_PYTHON}" ]]; then
  echo "Python executable not found: ${JIT_PYTHON}"
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Source image directory not found: ${SOURCE_DIR}"
  exit 1
fi

if ! "${JIT_PYTHON}" create_paired_restoration_data.py \
  --input_dir "${SOURCE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --img_size "${IMG_SIZE}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" | tee -a "${LOG_FILE}"; then
  echo "Data preparation failed. Last log lines:"
  tail -n 30 "${LOG_FILE}"
  exit 1
fi

echo "Paired data ready: ${OUTPUT_DIR}"
echo "Latest log: ${LATEST_LOG_LINK}"
