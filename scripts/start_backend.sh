#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
  echo "Virtual environment not found. Creating it first..."
  bash "${SCRIPT_DIR}/setup_venv.sh"
fi

cd "${PROJECT_ROOT}"
"${VENV_PYTHON}" -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}"
