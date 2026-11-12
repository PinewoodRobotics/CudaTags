import os

if os.environ.get("CUDATAGS_TYPES_ONLY") == "1":
    from setuptools import setup
    native_options = {}
else:
    from skbuild import setup
    native_options = dict(
        cmake_install_dir="src/main/python/cuda_tags",
        cmake_args=["-DCUDATAGS_BUILD_JAVA=OFF", "-DCUDATAGS_BUILD_PYTHON=ON"],
    )

setup(
    name="cuda_tags", version="0.1.0", description="CUDA-accelerated AprilTag detection",
    author="FRC 971", license="BSD-3-Clause", packages=["cuda_tags"],
    package_dir={"cuda_tags": "src/main/python/cuda_tags"},
    package_data={"cuda_tags": ["*.pyi", "py.typed"]},
    python_requires=">=3.10", include_package_data=True,
    install_requires=["numpy>=1.23,<3"], **native_options,
)
