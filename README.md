## CudaTags – CUDA-Accelerated AprilTag Detection

Note: ai readme. 

### What this repo is

- **CUDA AprilTag engine**: GPU-accelerated AprilTag detector based on Team 971’s implementation (`frc971/orin/*.cu` + `third_party/apriltag`).
- **Shared library**: Builds `lib971apriltag.so` via CMake.
- **Bindings**:
  - **Java/JNI** for PhotonVision / WPILib (`GpuDetectorJNI.cc`).
  - **Rust** wrapper using `cxx` (`src/bridge.rs`, `src/cpp/cuda_tags_wrapper.*`, `src/main.rs` demo).

If you just want to **see tags being detected**, the Rust demo is the simplest way to test once the CUDA library is built.

---

## Requirements

- **GPU platform**
  - Tested on NVIDIA Jetson (e.g., Orin) with **JetPack 6.2**.
  - CUDA toolkit installed at `/usr/local/cuda`.
- **System packages (typical Jetson setup)**
  - `build-essential cmake ninja-build`
  - `openjdk-17-jdk`
  - `libopencv-dev` (and/or `libopencv4.5-java` if using Java)
  - `protobuf-compiler libxrandr-dev libssh-dev`
- **Rust toolchain (for the demo / crate)**
  - `rustup` with a recent stable toolchain (edition 2024 is used).

Most of these are handled automatically if you use the `install` scripts below.

---

## Quick start: build everything and run a live demo

The fastest “it actually works” path on a fresh Jetson with JetPack:

1. **Clone the repo**
   - `cd ~/Documents`
   - `git clone <this-repo-url> CudaTags`
   - `cd CudaTags`

2. **Run the installer (builds dependencies + CUDA library)**
   - `bash install/build.bash`

   This will:
   - Install Java 17 and basic dependencies.
   - Set CUDA / Java environment variables in `~/.bashrc`.
   - Clone and build WPILib.
   - Build `third_party/apriltag` and this project.
   - Install `lib971apriltag.so` to `/usr/lib`.

3. **Open a new shell** so the environment from `~/.bashrc` is picked up.

4. **Run the Rust camera demo**
   - `cd ~/Documents/CudaTags`
   - `cargo run --release`

   What you should see:
   - A window named `cam` from OpenCV.
   - Live camera frames from `/dev/video0`.
   - AprilTags outlined in red with their IDs drawn on the image.
   - FPS and timing stats printed to stderr in the form:
     - `[CudaTags-Rust] frame 640x480 channels=3 stride=... detections=N time_ms=...`

If this works, you have a fully functioning CUDA + AprilTag + OpenCV stack.

---

## Building the CUDA AprilTag library only

If you want to build the GPU library separately (without using `install/build.bash`):

1. **Build the upstream AprilTag library**
   - `cd ~/Documents/CudaTags/third_party/apriltag`
   - `mkdir -p build`
   - `cd build`
   - `cmake ..`
   - `make`

2. **Build CudaTags via CMake**
   - `cd ~/Documents/CudaTags`
   - `mkdir -p build`
   - `cd build`
   - `cmake ..`
   - `make`

   This produces `build/lib971apriltag.so`.

3. **Install the shared library (so Rust/Java can link it)**
   - `sudo cp lib971apriltag.so /usr/lib`

Once `lib971apriltag.so` is installed into `/usr/lib`, you can:
- Use the **Rust demo** (`cargo run --release`).
- Use the **Java/JNI integration** as described below.

---

## Rust crate: how to use it and what the demo does

The Rust crate in this repo (`Cargo.toml`, `src/lib.rs`, `src/bridge.rs`, `src/main.rs`) wraps the CUDA detector via `cxx` and runs a simple camera demo.

- **Entry point**: `src/main.rs`
  - Opens camera index `0` with OpenCV (MJPG at 640×480 @ 30 FPS).
  - Calls `make_cuda_tag_detector` from the C++ layer to create a GPU detector.
  - Sends frames into the GPU, retrieves `DetectionResult` values, and draws:
    - Tag outline (quadrilateral).
    - Tag center.
    - Tag ID text near the center.
  - Overlays **FPS**, **processing time (ms)**, and **detections count** on the frame.
  - Press `Esc` to exit the window.

- **Bridge types**: `src/bridge.rs`
  - Defines `CameraMatrix`, `DistCoeffs`, `RustImage`, `TagType`, and `DetectionResult`.
  - Exposes `make_cuda_tag_detector` and `CudaTagsWrapper::process` from C++.

### Running on a different camera or resolution

- To use a different camera index, change the line in `src/main.rs`:
  - `let mut cam = videoio::VideoCapture::new(0, videoio::CAP_V4L2)?;`
- To change resolution or FPS, adjust:
  - `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT`, `CAP_PROP_FPS`.
- Make sure the width/height passed to `make_cuda_tag_detector` match the actual camera resolution.

---

## Java / PhotonVision usage (JNI layer)

For Java/WPILib/PhotonVision integration, the key file is `GpuDetectorJNI.cc`. It:

- Wraps the CUDA-based `frc971::apriltag::GpuDetector`.
- Exposes JNI methods for creating/destroying detectors and processing `cv::Mat` images.
- Produces an array of `edu.wpi.first.apriltag.AprilTagDetection` objects with ID, Hamming distance, decision margin, homography, corners, and center.

To use it:

1. **Ensure `lib971apriltag.so` is built and installed** (see sections above).
2. **Make sure your Java classpath and `java.library.path` can find everything**.
   - Example (from `test.sh`):
     - `java -cp GpuDetectorJNI.jar -Djava.library.path=. org.photonvision.jni.GpuDetectorJNI`
3. **From Java**, you typically:
   - Capture frames into an OpenCV `Mat`.
   - Pass the image data to the JNI layer.
   - Use the returned detections for localization or targeting.

The exact wiring depends on your PhotonVision / robot codebase, but this library is designed to drop into that ecosystem with minimal changes.

---

## Notes / troubleshooting

- **CUDA / OpenCV include paths**
  - `CMakeLists.txt` and `build.rs` assume CUDA headers at `/usr/local/cuda/include` and OpenCV at `/usr/include/opencv4`.
  - Adjust those paths if your distro uses different locations.
- **LLVM / Clang for Rust `cxx` builds**
  - On some systems you may need:
    - `export LLVM_CONFIG_PATH=/usr/bin/llvm-config-14`
  - This is hinted at in `test.sh`.
- **Architecture-specific libraries**
  - `build.rs` looks for `lib/linux/<arch>` when giving Rust a link search path, but installing `lib971apriltag.so` to `/usr/lib` is usually enough.

---

## Project layout

- **CUDA + AprilTag core**
  - `frc971/orin/` – GPU implementation and helpers (thresholding, labeling, line fitting, etc.).
  - `third_party/apriltag/` – Upstream AprilTag library.
- **Bindings**
  - `GpuDetectorJNI.cc` – Java/WPILib JNI bindings.
  - `src/cpp/cuda_tags_wrapper.*` – C++ bridge for the Rust `cxx` layer.
  - `src/bridge.rs` – Rust side of the `cxx` bridge.
- **Rust crate**
  - `Cargo.toml`, `src/lib.rs`, `src/main.rs`.
- **Build / install**
  - `CMakeLists.txt` – CMake config for `lib971apriltag.so`.
  - `install/*.bash` – Helper scripts to set up dependencies and build everything.

---

## License

This repository includes the upstream AprilTag library under its own license (`third_party/apriltag/LICENSE.md`).  
See `LICENSE.txt` in this repository for the overall project licensing details.

