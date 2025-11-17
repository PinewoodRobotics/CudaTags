#! /bin/bash

cd ~/Documents/CudaTags


cd detection
cd third_party/apriltag
mkdir build
cd build
cmake ..
make
cd ../../..
mkdir build
cd build
cmake ..
make
sudo cp lib971apriltag.so /usr/lib 