CudaTags – CUDA-Accelerated AprilTag Library for PhotonVision
============================================================

Overview
--------

CudaTags is a CUDA-accelerated AprilTag detector based on team 971's work, packaged as a shared library (`lib971apriltag.so`) and JNI wrapper for use with PhotonVision on NVIDIA Jetson platforms (e.g., Orin) running JetPack 6.2.

The project:
- Uses CUDA and custom kernels (`frc971/orin/*.cu`) to speed up AprilTag detection.
- Wraps the detector with JNI (`GpuDetectorJNI.cc`) so it can be called from Java (e.g., PhotonVision / WPILib ecosystem).
- Links against the upstream `apriltag` library in `third_party/apriltag`.

How It Works
------------

At a high level, CudaTags replaces the CPU-heavy parts of the AprilTag pipeline with CUDA kernels and exposes the result to Java:

1. A camera image (typically from OpenCV in Java) is passed into the JNI layer (`GpuDetectorJNI.cc`).
2. The JNI layer transfers the image into GPU memory and calls CUDA code in `frc971/orin/*.cu`.
3. CUDA kernels handle thresholding, connected-components labeling, line fitting, and other steps needed for AprilTag detection.
4. The GPU results are fed into the upstream `apriltag` library (`third_party/apriltag`) to decode and refine the final tag poses.
5. The decoded tags (IDs, corners, and pose information) are returned back through JNI to Java, where they can be used by PhotonVision or other WPILib-based applications.

This design lets you keep your robot code and vision logic in Java while offloading the heavy image processing steps to the Jetson GPU.

Requirements
------------

Tested environment:
- NVIDIA Jetson running **JetPack 6.2**
- CUDA toolkit installed at `/usr/local/cuda`
- **OpenJDK 17**
- OpenCV with Java bindings (e.g., `libopencv4.5-java`)
- WPILib toolchain installed system-wide

Key system packages (most are handled by the install scripts below):
- `openjdk-17-jdk`
- `ninja-build`
- `protobuf-compiler`
- `libxrandr-dev`
- `libssh-dev`
- `libopencv4.5-java`

Quick Start – How to Build Using `install` Scripts
--------------------------------------------------

From a fresh JetPack 6.2 install on your Jetson:

1. Clone this repository:
   - `cd ~/Documents`
   - `git clone <this-repo-url> CudaTags`
   - `cd CudaTags`

2. Run the combined installer (this will update packages, install dependencies, set up environment variables, build WPILib, and build this project):
   - `bash install/build.bash`

What `install/build.bash` does:
- Runs `install/pre.bash` to:
  - Install Java 17.
  - Append the following to your `~/.bashrc`:
    - `export PATH=$PATH:/usr/local/cuda/bin`
    - `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`
    - `export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64/`
- Runs `install/wpi.bash` to:
  - Clone `https://github.com/wpilibsuite/allwpilib.git`.
  - Configure and build WPILib with CMake (no GUI, Java enabled, some modules/tests disabled).
  - Install WPILib system-wide.
- Runs `install/self.bash` to:
  - Build `third_party/apriltag`.
  - Build this project (CudaTags).
  - Copy `lib971apriltag.so` to `/usr/lib`.

After the script completes successfully:
- Your environment variables will be set on new shells (`PATH`, `LD_LIBRARY_PATH`, `JAVA_HOME`).
- WPILib will be installed and available.
- `lib971apriltag.so` will be installed to `/usr/lib` for system-wide use.

Manual Install – Step-by-Step
-----------------------------

If you prefer to do everything manually (or adapt for a slightly different environment), follow these steps.

1. Install Java and set environment variables:
   - `sudo apt install openjdk-17-jdk`
   - Add the following to your `~/.bashrc`:
     - `export PATH=$PATH:/usr/local/cuda/bin`
     - `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`
     - `export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64/`
   - Reload your shell:
     - `source ~/.bashrc`

2. Install and build WPILib:
   - `cd ~/Documents`
   - `git clone https://github.com/wpilibsuite/allwpilib.git`
   - `cd allwpilib`
   - `sudo apt install openjdk-17-jdk`
   - `sudo apt install ninja-build`
   - `sudo apt install protobuf-compiler`
   - `sudo apt install libxrandr-dev`
   - `sudo apt install libssh-dev`
   - `sudo apt install libopencv4.5-java`
   - `cmake --preset default -DWITH_GUI=OFF -DWITH_JAVA=ON -DWITH_SIMULATION_MODULES=OFF -DWITH_TESTS=OFF -DOPENCV_JAR_FILE=/usr/share/java/opencv.jar`
   - `cd build-cmake`
   - `cmake --build . --parallel 4`  # reduce `--parallel` if you hit memory limits
   - `sudo cmake --build . --target install`

3. Build CudaTags and AprilTag:
   - `cd ~/Documents/CudaTags`
   - `cd third_party/apriltag`
   - `mkdir build`
   - `cd build`
   - `cmake ..`
   - `make`
   - `cd ../../..`
   - `mkdir build`
   - `cd build`
   - `cmake ..`
   - `make`
   - `sudo cp lib971apriltag.so /usr/lib`

Project Layout
--------------

- `frc971/orin/` – CUDA implementation and helpers for AprilTag detection (thresholding, labeling, line fitting, etc.).
- `third_party/apriltag/` – Upstream AprilTag library (built as `libapriltag.a`).
- `GpuDetectorJNI.cc` – JNI bridge exposing the CUDA detector to Java.
- `install/` – Helper scripts to set up the environment, WPILib, and build this project.
- `CMakeLists.txt` – CMake configuration for the shared library `lib971apriltag.so`.

Building Directly with CMake
----------------------------

In most cases you should use the `install` scripts above. If you already have all dependencies installed (including WPILib, CUDA, JNI, and OpenCV), you can build directly:

1. `cd ~/Documents/CudaTags`
2. `mkdir -p build`
3. `cd build`
4. `cmake ..`
5. `make`

The resulting `lib971apriltag.so` will be placed in the `build` directory; you can then install or copy it to a suitable library path (e.g., `/usr/lib`).

Using from Java / PhotonVision
------------------------------

The library is intended to be used via the JNI wrapper from Java code (e.g., inside PhotonVision or other WPILib-based Java applications).

Make sure that:
- `lib971apriltag.so` is on the system library path (for example, installed in `/usr/lib`).
- Your Java process can load the JNI library, either by:
  - Calling `System.loadLibrary("971apriltag")`, or
  - Calling `System.load("/full/path/to/lib971apriltag.so")`.

How you wire this into PhotonVision or your robot project depends on your existing codebase, but typically:
- A camera frame is acquired into an OpenCV `Mat`.
- The `Mat` (or equivalent image buffer) is passed into native code via JNI.
- The tags and poses returned by CudaTags are used for robot localization or target tracking.

License
-------

This repository includes the upstream AprilTag library under its own license (`third_party/apriltag/LICENSE.md`).  
See `LICENSE.txt` in this repository for the overall project licensing details.
