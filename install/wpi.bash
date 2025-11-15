#! /bin/bash

cd ~/Documents/

git clone https://github.com/wpilibsuite/allwpilib.git
cd allwpilib
sudo apt install openjdk-17-jdk
sudo apt install ninja-build
sudo apt install protobuf-compiler
sudo apt install libxrandr-dev
sudo apt install libssh-dev
sudo apt install libopencv4.5-java
cmake --preset default -DWITH_GUI=OFF -DWITH_JAVA=ON -DWITH_SIMULATION_MODULES=OFF -DWITH_TESTS=OFF -DOPENCV_JAR_FILE=/usr/share/java/opencv.jar
cd build-cmake 
cmake --build . --parallel 4  #may run out of memory compiling wpimath if too high
sudo cmake --build . --target install