# CudaTags

GPU AprilTag (tag36h11) detection for NVIDIA Jetson, built on
[Team 971's CUDA detector](https://github.com/frc971/971-Robot-Code) and the
upstream [AprilTag](https://github.com/AprilRobotics/apriltag) library, with
an optional per-tag 6-DoF pose (OpenCV `SOLVEPNP_IPPE_SQUARE`).

One C++ core is shared by every interface:

| Interface | Where | Use it from |
| --- | --- | --- |
| C++ core | `src/main/cpp/include/cuda_tags/detector.hpp` | C++ (link `cuda_tags_core`) |
| C ABI | `src/main/cpp/include/cuda_tags/cuda_tags.h`, `libcuda_tags.so` | any FFI, `dlopen` |
| Rust crate `cuda_tags` | `src/main/rust` | Cargo path/git dependency |
| Python module `cuda_tags` | `src/main/python` | `pip install .` |
| Java `frc4765.cuda.tags.CudaTags` | `src/main/java` | JNI (`libcuda_tags_jni.so`) |

## Supported platforms

- **GPU detection:** NVIDIA Jetson Orin on JetPack 6 (Ubuntu 22.04, CUDA 12.2
  or newer), GPU architecture `sm_87`. Other CUDA GPUs work by setting
  `CMAKE_CUDA_ARCHITECTURES` (for example `native`).
- **Everywhere else** (macOS, x86 Linux without CUDA): everything builds and
  the CPU tests run, but creating a detector fails with a clear "built without
  CUDA" error. This lets projects that depend on CudaTags build and test on
  developer machines.

Requirements: CMake 3.22+, a C++20 compiler, OpenCV 4 (core, imgproc,
calib3d; `libopencv-dev`), and the CUDA toolkit for GPU builds. Only the
`detection/third_party/apriltag` submodule is needed:

```sh
git submodule update --init detection/third_party/apriltag
```

(`detection/third_party/allwpilib` is no longer used by any build.)

## Detector behaviour (all interfaces)

- **One image size per detector.** Width and height must be multiples of 8
  with `width * height < 2^22` (1920x1080 is fine). A frame of another size is
  an error; the detector never reallocates GPU memory silently. Build a new
  detector for a new size.
- **Threading.** A detector owns its CUDA stream, GPU buffers and decode
  thread pool. Use it from the thread that created it, one call at a time.
  Separate detectors (one per camera) may run in parallel.
- **Lifetime.** Results are copied out of the detector; the C ABI's results
  are valid until the next `ct_detect`.
- **Errors** never cross a language boundary as C++ exceptions: Rust gets
  `Result`, Python `ValueError`/`RuntimeError`, Java
  `IllegalArgumentException`/`RuntimeException`, C a `NULL`/`-1` return plus
  `ct_last_error()`.
- **Coordinates.** Corners and centers are pixels in the input image, which
  is distorted, not undistorted. The detector uses the distortion model only to
  refine edges. They follow AprilTag's convention: the center of pixel `(x, y)`
  is at `(x + 0.5, y + 0.5)`. Corner order is AprilTag's: bottom-left,
  bottom-right, top-right, top-left of an upright tag.
- **Pose** (when a tag size is given): camera-from-tag rotation and
  translation in meters, `p_camera = R * p_tag + t`, with the tag in AprilTag's
  frame (origin at the center, x right, y down, z into the tag). It is solved
  with the camera's intrinsics *and* distortion after shifting corners to
  OpenCV's pixel convention.
- **Tunables:** `threads` (decode threads), `decimate` (1 or 2, default 2),
  `min_white_black_diff` (default 5), `max_hamming` (0 to 2, default 1),
  `refine_edges` (default on), `decode_sharpening` (default 0.25), and
  `tag_size_m` (0 or unset disables the pose).

## Rust

```toml
[dependencies]
cuda_tags = { path = "path/to/CudaTags/src/main/rust", features = ["cuda"] }
```

```rust
use cuda_tags::{CameraIntrinsics, Detector, Distortion, Options};

let camera = CameraIntrinsics { fx: 600.0, fy: 600.0, cx: 320.0, cy: 240.0 };
let options = Options { threads: 2, tag_size_m: Some(0.1651), ..Options::default() };
let mut detector = Detector::with_options(camera, Distortion::default(), 640, 480, &options)?;
for tag in detector.detect(&gray_frame)? {          // &[u8], width * height bytes
    println!("{} {:?} {:?}", tag.id, tag.corners, tag.pose);
}
```

`Detector::new(camera, distortion, width, height, threads)` is the short
form, with default options and no pose. `Detector` is neither `Send` nor
`Sync`.

`build.rs` behaves as follows:

- **With the `cuda` feature, on Linux, with a CUDA toolkit:** it builds the
  C++/CUDA sources with CMake and links them statically. OpenCV and
  `libstdc++` stay dynamic, and the CUDA runtime is linked statically. It
  finds CUDA through `CUDA_HOME`, `CUDA_PATH` or `CUDAToolkit_ROOT`, then
  `/usr/local/cuda`, then `nvcc` on `PATH`. It finds OpenCV with
  `pkg-config opencv4`. It honours `CMAKE_CUDA_ARCHITECTURES` (default `87`).
- **Without the feature, off Linux, or without a toolkit:** it builds a stub,
  and `Detector::new` returns `Error::Unavailable`.
  `cuda_tags::cuda_available()` and `stub_reason()` report which build you
  got. Set `CUDA_TAGS_REQUIRE_CUDA=1` to turn a missing toolkit into a build
  error, for deployment builds.

Example (uses the golden test image by default):

```sh
cargo run --release --features cuda --example detect_pgm -- [image.pgm] [fx fy cx cy] [tag_size_m]
```

## C ABI

`libcuda_tags.so` exports only the `ct_*` functions declared in
`cuda_tags/cuda_tags.h`, ABI version 1:

- `ct_create`, `ct_detect`, `ct_detection_at`, `ct_destroy`, `ct_last_error`
  and `ct_abi_version`, compatible with the drone project's `cuda_tags_ffi`
  shim.
- Additions: `ct_options_init` with `ct_create_with_options` for every
  tunable, `ct_detection_homography`, and `ct_cuda_available`.

```sh
./install/self.bash   # build, test, install to /usr/local (PREFIX=... to change)
```

This installs `lib/libcuda_tags.so` and `include/cuda_tags/cuda_tags.h`.

## Python

```sh
pip install .                               # on the Jetson; builds with CUDA
CUDATAGS_TYPES_ONLY=1 pip install .         # type stubs only, any machine
```

```python
import cuda_tags, numpy as np

detector = cuda_tags.CudaTagsWrapper(
    cuda_tags.tag36h11,
    cuda_tags.CameraMatrix(fx=600, cx=320, fy=600, cy=240),
    cuda_tags.DistCoeffs(0, 0, 0, 0, 0),     # k1, k2, p1, p2, k3
    2, 640, 480,                             # threads, width, height
    tag_size_m=0.1651)
for tag in detector.process(frame):          # (h, w), (h, w, 3) or (h, w, 4) uint8
    print(tag.id, tag.corners, tag.pose and tag.pose.translation)
```

- `process` accepts any uint8 layout. Padded rows are read in place, and
  other strided views are copied first.
- Three- and four-channel images are BGR/BGRA by default. Pass
  `color_order="rgb"` for RGB/RGBA.
- `process` releases the GIL while it detects.
- `cuda_tags.to_gray(image)` returns the gray image the detector sees.

## Java

Configure with `-DCUDATAGS_BUILD_JAVA=ON` to build `cuda-tags.jar` and
`libcuda_tags_jni.so`. Then:

```java
try (CudaTags detector = new CudaTags(640, 480, fx, fy, cx, cy, distortion, 2, 0.1651)) {
  CudaTags.Detection[] tags = detector.detect(grayBytes);    // or detect(pixels, stride, channels), detectMat(mat.getNativeObjAddr())
}
```

## Building and testing with CMake

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # CUDA is used when nvcc is found
cmake --build build
ctest --test-dir build --output-on-failure
```

Options:

- `CUDATAGS_WITH_CUDA`: on when a CUDA compiler is found.
- `CUDATAGS_BUILD_C_LIBRARY`: default on.
- `CUDATAGS_BUILD_PYTHON`: on under scikit-build.
- `CUDATAGS_BUILD_JAVA`: default off.
- `CUDATAGS_BUILD_TESTS`: on for top-level builds.
- `CMAKE_CUDA_ARCHITECTURES`: default `87`.

Tests:

- **`core_cpu`** (any machine): option validation, color conversion with
  strides and channel orders, the pose solve with distortion, and C ABI error
  handling.
- **`golden_cpu`** (any machine): `tests/golden/tags_640x480.pgm`, three tags
  at known poses rendered by `tests/golden/generate_golden.cpp`, run through
  AprilTag's CPU detector. This uses the patched `apriltag_971/apriltag.c`
  that the GPU detector links against. Ids, sub-pixel corners and poses must
  match the ground truth in `tags_640x480.txt`.
- **`golden_gpu`** (CUDA builds): the same image and checks through the GPU
  detector, plus rejection of a wrongly sized frame.
- **`tests/test_binding.py`**: the Python module. It runs on a stub build
  too; set `CUDA_TAGS_EXTENSION` to the built `cuda_tags*.so`.
- **`cargo test`**: the Rust crate. On a stub build it checks the
  `Unavailable` path; with `--features cuda` on a Jetson it runs the golden
  image.

CI (`.github/workflows/ci.yml`) runs everything that needs no GPU on Linux
and macOS. On a Jetson, run everything:

```sh
./scripts/test-jetson.sh
```

## Repository layout

```
detection/
  CMakeLists.txt        apriltag (static) + 971apriltag (static, CUDA)
  frc971/orin/          Team 971's GPU detector
  apriltag_971/         upstream apriltag.c with quad_decode_index() and
                        reconcile_detections() exported for the GPU detector
                        (its SHA-256 pins the upstream file it was copied from)
  third_party/apriltag  upstream AprilTag (submodule)
src/main/cpp/           shared C++ core and C ABI
src/main/rust/          Rust crate (build.rs drives CMake)
src/main/python/        Python module and package
src/main/java/          Java class and JNI
tests/                  C++/Python tests, golden image
install/                Jetson setup scripts
```

## License

Apache-2.0 (`LICENSE.txt`). The upstream AprilTag library and its derived
`detection/apriltag_971/apriltag.c` are under AprilTag's BSD 2-Clause license
(`detection/third_party/apriltag/LICENSE.md`).
