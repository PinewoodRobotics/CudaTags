from skbuild import setup

setup(
    name="cuda_tags",
    version="0.1.0",
    description="CUDA-accelerated AprilTag detection",
    author="FRC 971",
    license="BSD-3-Clause",
    packages=["cuda_tags"],
    package_dir={"cuda_tags": "src/main/python/cuda_tags"},
    cmake_install_dir="src/main/python/cuda_tags",
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=["numpy>=1.21"],
    cmake_args=[
        "-DCUDATAGS_BUILD_JAVA=OFF",
        "-DCUDATAGS_BUILD_PYTHON=ON",
        "-DCMAKE_CXX_FLAGS=-w",
        "-DCMAKE_CUDA_FLAGS=-w",
    ],
)
