#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src/main"
PY="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then PY="python"; fi
exec "$PY" -m engine.entrypoint "$@"

