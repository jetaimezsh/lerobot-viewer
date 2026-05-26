#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

cd "${PROJECT_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3 and python3-venv first." >&2
  exit 1
fi

if [ ! -d "${VENV_PATH}" ]; then
  python3 -m venv "${VENV_PATH}"
fi

"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements.txt"

echo ""
echo "Virtual environment is ready: ${VENV_PATH}"
echo "Start server: bash scripts/start_backend.sh"
