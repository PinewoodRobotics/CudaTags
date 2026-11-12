# JetPack 6 (CUDA 12.6) build environment for CudaTags.
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libopencv-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-numpy \
    pybind11-dev \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV PATH=$PATH:/usr/local/cuda/bin
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
