#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

JIT_PYTHON="${JIT_PYTHON:-/data/Shenzhen/zhahongli/envs/jit-local/bin/python}"
MODEL_NAME="${MODEL_NAME:-JiT-L/32}"
case "${MODEL_NAME}" in
  "JiT-H/32")
    DEFAULT_CHECKPOINT_PATH="${REPO_DIR}/JiT-H-32/checkpoint-last.pth"
    DEFAULT_CFG_SCALE="2.3"
    ;;
  *)
    DEFAULT_CHECKPOINT_PATH="${REPO_DIR}/JiT-l-32/checkpoint-last.pth"
    DEFAULT_CFG_SCALE="2.5"
    ;;
esac

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/generated_pretrained_cloud}"
LOG_DIR="${LOG_DIR:-./logs}"
DEVICE="${DEVICE:-cuda}"
IMG_SIZE="${IMG_SIZE:-512}"
NOISE_SCALE="${NOISE_SCALE:-2.0}"
CFG_SCALE="${CFG_SCALE:-${DEFAULT_CFG_SCALE}}"
NUM_STEPS="${NUM_STEPS:-50}"
EMA_KEY="${EMA_KEY:-model_ema1}"
LABELS="${LABELS:-0,207,281}"
SEEDS="${SEEDS:-0,1,2}"
TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib64}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/cloud_pretrained_generate_${RUN_ID}.log"
LATEST_LOG_LINK="${LOG_DIR}/cloud_pretrained_generate_latest.log"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG_LINK}"

unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH
export TRITON_LIBCUDA_PATH

echo "Pretrained generation run starting..."
echo "Model: ${MODEL_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output root: ${OUTPUT_ROOT}"
echo "TRITON_LIBCUDA_PATH: ${TRITON_LIBCUDA_PATH}"
echo "Full log: ${LOG_FILE}"

if [[ ! -x "${JIT_PYTHON}" ]]; then
  echo "Python executable not found: ${JIT_PYTHON}"
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found"
  exit 1
fi

echo "GPU summary:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "${LOG_FILE}"

echo "Checking Python/CUDA environment..."
"${JIT_PYTHON}" - <<'PY' | tee -a "${LOG_FILE}"
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the uploaded environment.")
print("device 0", torch.cuda.get_device_name(0))
PY

IFS=',' read -ra LABEL_ARRAY <<< "${LABELS}"
IFS=',' read -ra SEED_ARRAY <<< "${SEEDS}"

for raw_label in "${LABEL_ARRAY[@]}"; do
  label="$(echo "${raw_label}" | xargs)"
  for raw_seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${raw_seed}" | xargs)"
    RUN_OUT_DIR="${OUTPUT_ROOT}/label_${label}_seed_${seed}"
    mkdir -p "${RUN_OUT_DIR}"
    echo "Generating label=${label}, seed=${seed} ..."
    if ! "${JIT_PYTHON}" generate_pretrained_samples.py \
      --checkpoint "${CHECKPOINT_PATH}" \
      --output_dir "${RUN_OUT_DIR}" \
      --model "${MODEL_NAME}" \
      --img_size "${IMG_SIZE}" \
      --noise_scale "${NOISE_SCALE}" \
      --cfg "${CFG_SCALE}" \
      --num_sampling_steps "${NUM_STEPS}" \
      --ema_key "${EMA_KEY}" \
      --device "${DEVICE}" \
      --labels "${label}" \
      --seed "${seed}" >>"${LOG_FILE}" 2>&1; then
      echo "Generation failed for label=${label}, seed=${seed}. Last log lines:"
      tail -n 20 "${LOG_FILE}"
      exit 1
    fi
  done
done

echo "Pretrained generation run finished."
echo "Outputs: ${OUTPUT_ROOT}"
echo "Latest log: ${LATEST_LOG_LINK}"
