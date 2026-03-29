#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
INTERVAL="${INTERVAL:-10}"
RUN_SECS="${RUN_SECS:-0}"
KEEP_LAST="${KEEP_LAST:-0}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/gpu_monitor_${RUN_ID}.log"
LATEST_LOG_LINK="${LOG_DIR}/gpu_monitor_latest.log"

mkdir -p "${LOG_DIR}"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG_LINK}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found"
  exit 1
fi

echo "GPU monitor starting..."
echo "Interval: ${INTERVAL}s"
echo "Run seconds: ${RUN_SECS} (0 means unlimited)"
echo "Keep last snapshots: ${KEEP_LAST} (0 means unlimited)"
echo "Log file: ${LOG_FILE}"

start_ts="$(date +%s)"
snapshot_count=0

while true; do
  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  if [[ "${RUN_SECS}" != "0" && "${elapsed}" -ge "${RUN_SECS}" ]]; then
    echo "Reached run limit (${RUN_SECS}s). Exiting." | tee -a "${LOG_FILE}"
    break
  fi

  snapshot_file="$(mktemp)"
  {
    echo "===== $(date '+%F %T') ====="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    echo "--- processes ---"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
    echo
  } > "${snapshot_file}"

  cat "${snapshot_file}" | tee -a "${LOG_FILE}"
  snapshot_count=$((snapshot_count + 1))

  if [[ "${KEEP_LAST}" != "0" && "${snapshot_count}" -gt "${KEEP_LAST}" ]]; then
    python3 - "$LOG_FILE" "$KEEP_LAST" <<'PY'
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
keep_last = int(sys.argv[2])
text = log_path.read_text()
parts = [p for p in text.split("===== ") if p.strip()]
if len(parts) > keep_last:
    parts = parts[-keep_last:]
    log_path.write_text("".join("===== " + p for p in parts))
PY
  fi

  rm -f "${snapshot_file}"

  sleep "${INTERVAL}"
done

echo "Latest log symlink: ${LATEST_LOG_LINK}"
