#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

JIT_PYTHON="${JIT_PYTHON:-/data/Shenzhen/zhahongli/envs/jit-local/bin/python}"
MODEL_NAME="${MODEL_NAME:-JiT-H/32}"
IMG_SIZE="${IMG_SIZE:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/output_JiT-image-to-image}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${OUTPUT_DIR}/checkpoint-last.pth}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/data/Shenzhen/zhahongli/models/Qwen2-VL-2B-Instruct}"
INPUT_PATH="${INPUT_PATH:-${REPO_DIR}/paired_JiT-image-to-image/lq/sample_00000.png}"
TARGET_PATH="${TARGET_PATH:-${REPO_DIR}/paired_JiT-image-to-image/hq/sample_00000.png}"
RESTORED_PATH="${RESTORED_PATH:-${OUTPUT_DIR}/sample_00000_restored.png}"
TRIPTYCH_PATH="${TRIPTYCH_PATH:-${OUTPUT_DIR}/sample_00000_triptych.png}"
DEVICE="${DEVICE:-cuda}"
TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib64}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
DISABLE_AMP="${DISABLE_AMP:-1}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/infer_restoration_triptych_${RUN_ID}.log"
LATEST_LOG_LINK="${LOG_DIR}/infer_restoration_triptych_latest.log"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG_LINK}"

unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH
export TRITON_LIBCUDA_PATH

echo "JiT image-to-image inference starting..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Qwen model path: ${QWEN_MODEL_PATH}"
echo "Input: ${INPUT_PATH}"
echo "Target: ${TARGET_PATH}"
echo "Restored output: ${RESTORED_PATH}"
echo "Triptych output: ${TRIPTYCH_PATH}"
echo "Disable AMP: ${DISABLE_AMP}"
echo "LoRA rank: ${LORA_RANK}"
echo "LoRA alpha: ${LORA_ALPHA}"
echo "LoRA dropout: ${LORA_DROPOUT}"
echo "Full log: ${LOG_FILE}"

if [[ ! -x "${JIT_PYTHON}" ]]; then
  echo "Python executable not found: ${JIT_PYTHON}"
  exit 1
fi

for required_path in "${CHECKPOINT_PATH}" "${INPUT_PATH}" "${TARGET_PATH}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required file not found: ${required_path}"
    exit 1
  fi
done

if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
  echo "Qwen model path not found: ${QWEN_MODEL_PATH}"
  exit 1
fi

EXTRA_ARGS=()
if [[ "${DISABLE_AMP}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_amp)
fi

if ! "${JIT_PYTHON}" infer_jit_restoration.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --input "${INPUT_PATH}" \
  --output "${RESTORED_PATH}" \
  --model "${MODEL_NAME}" \
  --img_size "${IMG_SIZE}" \
  --qwen_model_path "${QWEN_MODEL_PATH}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
  echo "Restoration inference failed. Last log lines:"
  tail -n 30 "${LOG_FILE}"
  exit 1
fi

if ! "${JIT_PYTHON}" infer_jit_restoration_triptych.py \
  --input "${INPUT_PATH}" \
  --restored "${RESTORED_PATH}" \
  --target "${TARGET_PATH}" \
  --output "${TRIPTYCH_PATH}" \
  --img_size "${IMG_SIZE}" 2>&1 | tee -a "${LOG_FILE}"; then
  echo "Triptych creation failed. Last log lines:"
  tail -n 30 "${LOG_FILE}"
  exit 1
fi

echo "Latest log: ${LATEST_LOG_LINK}"
