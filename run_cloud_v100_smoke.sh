#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

JIT_PYTHON="${JIT_PYTHON:-/data/Shenzhen/zhahongli/envs/jit-local/bin/python}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-./toy_cloud_smoke}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-./output_cloud_smoke}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/cloud_v100_smoke_${RUN_ID}.log"
LATEST_LOG_LINK="${LOG_DIR}/cloud_v100_smoke_latest.log"

unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH

echo "Smoke test starting..."
echo "Full log: ${LOG_FILE}"

if [[ ! -x "${JIT_PYTHON}" ]]; then
  echo "Python executable not found: ${JIT_PYTHON}"
  echo "Set JIT_PYTHON to your uploaded environment, for example:"
  echo "  export JIT_PYTHON=/data/Shenzhen/zhahongli/envs/jit-local/bin/python"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found"
  exit 1
fi

echo "GPU summary:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "Checking uploaded environment..."
"${JIT_PYTHON}" - <<'PY' | tee -a "${LOG_FILE}"
import cv2, numpy, scipy, torch, torchvision, torch_fidelity
print("cv2", cv2.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torch_fidelity", torch_fidelity.__file__)
print("cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the uploaded environment.")
print("device", torch.cuda.get_device_name(0))
PY
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG_LINK}"

if [[ ! -d "${SMOKE_DATA_DIR}/train/class_0" ]]; then
  echo "Creating toy smoke dataset at ${SMOKE_DATA_DIR}"
  "${JIT_PYTHON}" create_toy_data.py \
    --output_dir "${SMOKE_DATA_DIR}" \
    --img_size 64 \
    --train_per_class 4 \
    --val_per_class 1 \
    --seed 0 >>"${LOG_FILE}" 2>&1
fi

mkdir -p "${SMOKE_OUTPUT_DIR}"

echo "Starting cloud smoke training run..."
if ! "${JIT_PYTHON}" main_jit.py \
  --model JiT-B/32 \
  --img_size 64 \
  --noise_scale 1.0 \
  --epochs 1 \
  --warmup_epochs 0 \
  --batch_size 1 \
  --blr 1e-5 \
  --num_workers 0 \
  --class_num 4 \
  --data_path "${SMOKE_DATA_DIR}" \
  --output_dir "${SMOKE_OUTPUT_DIR}" \
  --resume "${SMOKE_OUTPUT_DIR}" \
  --save_last_freq 1 \
  --device cuda >>"${LOG_FILE}" 2>&1; then
  echo "Smoke run failed. Last log lines:"
  tail -n 20 "${LOG_FILE}"
  exit 1
fi

if [[ ! -f "${SMOKE_OUTPUT_DIR}/checkpoint-last.pth" ]]; then
  echo "Smoke run finished but checkpoint-last.pth was not created."
  tail -n 20 "${LOG_FILE}"
  exit 1
fi

echo "Smoke run succeeded."
echo "Checkpoint: ${SMOKE_OUTPUT_DIR}/checkpoint-last.pth"
echo "Full log: ${LOG_FILE}"
