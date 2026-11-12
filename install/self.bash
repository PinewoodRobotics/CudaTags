#!/usr/bin/env bash
# Builds CudaTags with CUDA, runs the tests, and installs libcuda_tags.so and
# cuda_tags/cuda_tags.h (the C ABI) under PREFIX (default /usr/local).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-/usr/local}"
ARCH="${CMAKE_CUDA_ARCHITECTURES:-87}"
export PATH="/usr/local/cuda/bin:$PATH"

git -C "$ROOT" submodule update --init detection/third_party/apriltag
cmake -S "$ROOT" -B "$ROOT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDATAGS_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$ROOT/build"
(cd "$ROOT/build" && ctest --output-on-failure)
sudo cmake --install "$ROOT/build"
sudo ldconfig
