#! /bin/bash

set -e

cd /opt

if [ ! -d allwpilib ]; then
  git clone https://github.com/wpilibsuite/allwpilib.git
fi

cd allwpilib

apt-get update
apt-get install -y \
  openjdk-17-jdk \
  ninja-build \
  protobuf-compiler \
  libxrandr-dev \
  libssh-dev \
  gcc-12 \
  g++-12

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

cmake -S . -B build-cmake \
  -DWITH_GUI=OFF \
  -DWITH_JAVA=OFF \
  -DWITH_SIMULATION_MODULES=OFF \
  -DWITH_TESTS=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_C_FLAGS="-Wno-error=restrict -Wno-restrict -Wno-psabi" \
  -DCMAKE_CXX_FLAGS="-Wno-error=restrict -Wno-restrict -Wno-psabi" \

cmake --build build-cmake --target install --parallel $(nproc)