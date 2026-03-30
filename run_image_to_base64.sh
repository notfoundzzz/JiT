#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

JIT_PYTHON="${JIT_PYTHON:-/data/Shenzhen/zhahongli/envs/jit-local/bin/python}"
INPUT_PATH="${INPUT_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
WRAP="${WRAP:-0}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"

mkdir -p "${LOG_DIR}"

if [[ -z "${INPUT_PATH}" ]]; then
  echo "Please provide INPUT_PATH=/path/to/image"
  exit 1
fi

if [[ ! -x "${JIT_PYTHON}" ]]; then
  echo "Python executable not found: ${JIT_PYTHON}"
  exit 1
fi

CMD=(
  "${JIT_PYTHON}" -u image_to_base64.py
  --input "${INPUT_PATH}"
  --wrap "${WRAP}"
)

if [[ -n "${OUTPUT_PATH}" ]]; then
  CMD+=(--output "${OUTPUT_PATH}")
fi

"${CMD[@]}"
