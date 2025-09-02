#!/usr/bin/env bash
set -euo pipefail

# Create .venv if missing, install requirements, and set PYTHONPATH.
# Usage: bash scripts/dev_env.sh

ROOT="$PWD"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Python 3 not found. Install python3 and retry." >&2
  exit 1
fi

VENV="$ROOT/.venv"
VENVPY="$VENV/bin/python"

if [ ! -x "$VENVPY" ]; then
  echo "Creating virtual environment at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

"$VENVPY" -m pip install -r "$ROOT/requirements.txt"

export PYTHONPATH="$PWD/src/main"
echo "$PYTHONPATH"
