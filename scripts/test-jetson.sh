#!/usr/bin/env bash
# Full test run on a Jetson (JetPack 6, CUDA 12.2+): the CUDA build, ctest
# (including the golden image on the GPU), the Rust crate with CUDA, and the
# Python module.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${CMAKE_CUDA_ARCHITECTURES:-87}"
BUILD="$ROOT/build/jetson"
export PATH="/usr/local/cuda/bin:$PATH"

git -C "$ROOT" submodule update --init detection/third_party/apriltag

PYBIND_ARGS=()
if python3 -m pybind11 --cmakedir >/dev/null 2>&1; then
  PYBIND_ARGS=(-DCUDATAGS_BUILD_PYTHON=ON "-Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)"
               "-DPYTHON_EXECUTABLE=$(command -v python3)")
fi
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCUDATAGS_WITH_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" "${PYBIND_ARGS[@]}"
cmake --build "$BUILD" --parallel
ctest --test-dir "$BUILD" --output-on-failure

(cd "$ROOT" && CUDA_TAGS_REQUIRE_CUDA=1 CMAKE_CUDA_ARCHITECTURES="$ARCH" \
  cargo test --release --features cuda)

if compgen -G "$BUILD/cuda_tags*.so" >/dev/null; then
  (cd "$ROOT" && CUDA_TAGS_EXTENSION="$(ls "$BUILD"/cuda_tags*.so)" \
    python3 -m unittest discover -s tests -v)
else
  echo "pybind11 not installed: skipped the Python module" >&2
fi
echo "All Jetson tests passed."
