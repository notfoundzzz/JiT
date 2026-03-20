#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

JIT_PYTHON="${JIT_PYTHON:-/data/Shenzhen/zhahongli/envs/jit-local/bin/python}"
MODEL_NAME="${MODEL_NAME:-JiT-H/32}"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/paired_JiT-image-to-image}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/output_JiT-image-to-image}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${REPO_DIR}/JiT-H-32/checkpoint-last.pth}"
EMA_KEY="${EMA_KEY:-model_ema1}"
IMG_SIZE="${IMG_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib64}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_restoration_h32_${RUN_ID}.log"
LATEST_LOG_LINK="${LOG_DIR}/train_restoration_h32_latest.log"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG_LINK}"

unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH
export TRITON_LIBCUDA_PATH

echo "JiT image-to-image restoration training starting..."
echo "Model: ${MODEL_NAME}"
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Pretrained checkpoint: ${PRETRAINED_CHECKPOINT}"
echo "TRITON_LIBCUDA_PATH: ${TRITON_LIBCUDA_PATH}"
echo "Full log: ${LOG_FILE}"

if [[ ! -x "${JIT_PYTHON}" ]]; then
  echo "Python executable not found: ${JIT_PYTHON}"
  exit 1
fi

if [[ ! -d "${DATA_PATH}/lq" || ! -d "${DATA_PATH}/hq" ]]; then
  echo "Paired dataset not found under ${DATA_PATH}"
  exit 1
fi

if [[ ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "Pretrained checkpoint not found: ${PRETRAINED_CHECKPOINT}"
  exit 1
fi

if ! "${JIT_PYTHON}" - <<'PY' | tee -a "${LOG_FILE}"
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the uploaded environment.")
print("device 0", torch.cuda.get_device_name(0))
PY
then
  echo "Environment check failed. Last log lines:"
  tail -n 30 "${LOG_FILE}"
  exit 1
fi

if ! "${JIT_PYTHON}" main_jit_restoration.py \
  --model "${MODEL_NAME}" \
  --img_size "${IMG_SIZE}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --ema_key "${EMA_KEY}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" 2>&1 | tee -a "${LOG_FILE}"; then
  echo "Restoration training failed. Last log lines:"
  tail -n 30 "${LOG_FILE}"
  exit 1
fi

echo "Restoration training finished."
echo "Output dir: ${OUTPUT_DIR}"
echo "Latest log: ${LATEST_LOG_LINK}"
