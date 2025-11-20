#! /bin/bash

sudo apt update && sudo apt install -y openjdk-17-jdk

echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64/' >> ~/.bashrc

source ~/.bashrc