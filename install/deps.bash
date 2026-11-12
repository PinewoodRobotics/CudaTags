#!/usr/bin/env bash
# Build dependencies on JetPack 6 (Ubuntu 22.04). CUDA comes with JetPack.
# Java (openjdk-17-jdk) is only needed for -DCUDATAGS_BUILD_JAVA=ON.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git pkg-config \
  libopencv-dev python3-dev python3-pip python3-venv
