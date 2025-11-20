FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies, CUDA toolkit (for nvcc), and required libraries
# nvidia-cuda-toolkit provides the 'fake' CUDA environment (compiler + headers) on non-Jetson hardware
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    cmake \
    ninja-build \
    openjdk-17-jdk \
    coreutils \
    ncurses-bin \
    sudo \
    procps \
    findutils \
    git \
    gcc-12 \
    g++-12 \
    nvidia-cuda-toolkit \
    libopencv-dev \
    protobuf-compiler \
    libxrandr-dev \
    libssh-dev \
    pkg-config \
    libclang-dev \
    python3 \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Create symlinks to simulate the standard CUDA path (/usr/local/cuda)
# usage by CMakeLists.txt: include_directories(/usr/local/cuda/include)
RUN mkdir -p /usr/local/cuda \
    && ln -s /usr/include /usr/local/cuda/include \
    && ln -s /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 \
    && ln -s /usr/lib/nvidia-cuda-toolkit/bin /usr/local/cuda/bin

# Set working directory
WORKDIR /opt

# Set environment variables to match the simulated CUDA paths
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV PATH=$PATH:/usr/lib/nvidia-cuda-toolkit/bin
