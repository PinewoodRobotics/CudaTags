#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/CudaTags

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -v .

SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

# Make sure pybind11-stubgen is found via the current environment's bin
PYTHON_BIN=$(which python)
PYBIND11_STUBGEN_BIN="$(dirname "$PYTHON_BIN")/pybind11-stubgen"
"$PYBIND11_STUBGEN_BIN" cuda_tags --output-dir "$SITE_PACKAGES"