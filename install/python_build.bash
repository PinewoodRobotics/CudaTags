#!/usr/bin/env bash
# Builds and installs the Python package into .venv at the repository root.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/cuda/bin:$PATH"
git submodule update --init detection/third_party/apriltag

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -v .
