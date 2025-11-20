#! /bin/bash

set -e

# Work from the project root at /opt (this repository)
cd /opt

cd detection
cd third_party/apriltag

# Build apriltag with CMake in a dedicated build directory
mkdir -p build
cd build
cmake ..
make

cp libapriltag.a /usr/lib

# Go back to project root and build the rest of CudaTags
cd ../../..
mkdir -p build
cd build
cmake ..
make

cp lib971apriltag.so /usr/lib
SYSTEM_ARCH=$(uname -m)

cd ../../
mkdir -p lib/linux/$SYSTEM_ARCH
cp /usr/lib/libapriltag.a lib/linux/$SYSTEM_ARCH/libapriltag.a
cp /usr/lib/lib971apriltag.so lib/linux/$SYSTEM_ARCH/lib971apriltag.so