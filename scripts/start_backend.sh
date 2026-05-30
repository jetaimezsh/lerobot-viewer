#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONDA_PYTHON=""

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  CONDA_PYTHON="${CONDA_PREFIX}/bin/python"
fi

if [ -n "${CONDA_PYTHON}" ]; then
  PYTHON="${CONDA_PYTHON}"
  echo "Using active conda Python: ${PYTHON}"
else
  if [ ! -x "${VENV_PYTHON}" ]; then
    echo "Virtual environment not found. Creating it first..."
    bash "${SCRIPT_DIR}/setup_venv.sh"
  fi
  PYTHON="${VENV_PYTHON}"
  echo "Using project venv Python: ${PYTHON}"
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}"
