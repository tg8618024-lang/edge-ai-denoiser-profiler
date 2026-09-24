#!/usr/bin/env bash
set -e

echo "==============================================================================="
echo "  NVIDIA-STYLE REAL-TIME EDGE AI AUDIO DENOISER & HARDWARE PROFILER"
echo "==============================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    if ! command -v python3 &> /dev/null; then
        echo "[ERROR] python3 could not be found. Please install Python 3.10+."
        exit 1
    fi
    echo "[INFO] Creating virtual environment at .venv ..."
    python3 -m venv .venv
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
    echo "[INFO] Installing project requirements ..."
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install -r requirements.txt
fi

echo "[INFO] Starting interactive studio dashboard on http://127.0.0.1:8000 ..."
exec "$PYTHON" run_dashboard.py "$@"
