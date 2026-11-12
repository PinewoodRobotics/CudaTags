// CudaTags C++ core: the one wrapper around Team 971's GPU AprilTag
// detector that every binding (C ABI, Rust, Python, Java) uses.
//
// Threading and lifetime
// ----------------------
// * A Detector owns its own CUDA stream, device buffers, AprilTag decode
//   worker pool and result vector. Calls on one Detector must not overlap.
//   Treat it as bound to the thread that created it: build one detector per
//   camera on that camera's thread. Separate detectors may run concurrently.
// * The vector returned by Detect() is owned by the detector and is valid
//   until the next Detect() call or until the detector is destroyed. Copy
//   what you need to keep.
// * Detect() never reallocates GPU memory: a frame whose size differs from
//   the size the detector was built for is rejected with
//   std::invalid_argument. Build a new detector for a new size.
//
// Errors are reported with exceptions: std::invalid_argument for bad
// arguments, std::runtime_error for CUDA or detector failures. The C ABI in
// cuda_tags.h catches them; nothing throws across it.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cuda_tags {

// Pinhole intrinsics in pixels, at the detector's image size.
struct Camera {
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

// OpenCV (Brown-Conrady) distortion coefficients. All zero is an ideal
// pinhole camera.
struct Distortion {
  double k1 = 0.0;
  double k2 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double k3 = 0.0;
};

// Detector configuration. Only tag36h11 is supported.
struct Options {
  // Image size in pixels. Both must be multiples of 8 and
  // width * height must be below 2^22 (e.g. 1920x1080 is fine).
  uint32_t width = 0;
  uint32_t height = 0;
  // CPU threads that decode candidate quads (AprilTag's nthreads), 1..64.
  int threads = 1;
  // Quad detection runs on an image decimated by this factor: 1 or 2.
  // AprilTag's quad_decimate.
  int decimate = 2;
  // Minimum intensity difference between the white and black side of an
  // edge for it to be considered (AprilTag's qtp.min_white_black_diff),
  // 0..255.
  int min_white_black_diff = 5;
  // Maximum number of bit errors corrected while decoding, 0..2.
  int max_hamming = 1;
  // Refine quad edges against the full-resolution image (recommended).
  bool refine_edges = true;
  // Sharpening applied to decoded bit samples (AprilTag's
  // decode_sharpening), >= 0.
  double decode_sharpening = 0.25;
  // Edge length of the tag's black border in meters. When > 0, every
  // detection carries a pose; 0 disables the pose solve.
  double tag_size_m = 0.0;
};

// Camera-from-tag transform: p_camera = rotation * p_tag + translation.
// The tag frame is AprilTag's: origin at the tag center, x to the right,
// y down, z into the tag (away from the camera when the tag faces it).
struct Pose {
  std::array<double, 9> rotation{};     // row-major 3x3
  std::array<double, 3> translation{};  // meters
};

struct Detection {
  int id = 0;
  int hamming = 0;
  float decision_margin = 0.0f;
  // Pixels in the input image (distorted, not undistorted), in AprilTag's
  // convention: the center of pixel (x, y) is at (x + 0.5, y + 0.5). Subtract
  // 0.5 to compare with OpenCV, whose pixel centers are at integers.
  std::array<double, 2> center{};
  // AprilTag's corner order: for an upright tag, bottom-left, bottom-right,
  // top-right, top-left. These are the tag-frame points (-s, s), (s, s),
  // (s, -s), (-s, -s) with s = tag_size_m / 2.
  std::array<std::array<double, 2>, 4> corners{};
  // Row-major homography from the ideal tag square [-1, 1]^2 to pixels.
  std::array<double, 9> homography{};
  // From OpenCV's SOLVEPNP_IPPE_SQUARE with the detector's camera and
  // distortion. Empty when tag_size_m is 0 or the solve failed.
  std::optional<Pose> pose;
};

enum class PixelFormat { kGray, kBgr, kRgb, kBgra, kRgba };

int Channels(PixelFormat format);

// A borrowed, possibly padded, 8-bit image. Row r starts at
// data + r * stride; stride >= width * channels.
struct ImageView {
  const uint8_t *data = nullptr;
  size_t size = 0;  // bytes readable from data
  uint32_t width = 0;
  uint32_t height = 0;
  size_t stride = 0;
  PixelFormat format = PixelFormat::kGray;
};

// True when this build contains the CUDA detector. Without it, Detector's
// constructor throws std::runtime_error.
bool CudaAvailable();

// Empty when the configuration is usable, otherwise the reason it is not.
std::string ValidateOptions(const Camera &camera, const Distortion &distortion,
                            const Options &options);

// Checks the view's geometry against its buffer and returns the error, or
// an empty string.
std::string ValidateImage(const ImageView &image);

// Converts any supported view to a contiguous width * height gray image.
// Throws std::invalid_argument for an invalid view.
void ConvertToGray(const ImageView &image, std::vector<uint8_t> *gray);

// Solves the camera-from-tag pose of one tag from its corners exactly as a
// Detection reports them (corner order, distorted pixels, AprilTag's +0.5
// pixel-center convention, which this converts to OpenCV's) with OpenCV's
// SOLVEPNP_IPPE_SQUARE and the camera's distortion. Empty if the solve fails
// or the result is not finite.
std::optional<Pose> SolveTagPose(
    const std::array<std::array<double, 2>, 4> &corners, const Camera &camera,
    const Distortion &distortion, double tag_size_m);

class Detector {
 public:
  // Throws std::invalid_argument for a bad configuration and
  // std::runtime_error when CUDA is unavailable or allocation fails.
  Detector(const Camera &camera, const Distortion &distortion,
           const Options &options);
  ~Detector();

  Detector(const Detector &) = delete;
  Detector &operator=(const Detector &) = delete;

  // Detects tags in an image of exactly options().width x height. Color
  // images are converted to gray on the CPU first.
  const std::vector<Detection> &Detect(const ImageView &image);

  // Contiguous gray image; length must be width * height.
  const std::vector<Detection> &DetectGray(const uint8_t *pixels,
                                           size_t length);

  const Options &options() const;
  const Camera &camera() const;
  const Distortion &distortion() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cuda_tags
