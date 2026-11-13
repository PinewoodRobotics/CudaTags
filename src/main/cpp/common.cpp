// CPU-side parts of the core (validation, color conversion, pose). They do
// not need CUDA, so they build and are unit-tested on any machine.
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "cuda_tags/detector.hpp"

namespace cuda_tags {
namespace {

bool Finite(double value) { return std::isfinite(value); }

constexpr double kAprilTagPixelOffset = 0.5;

}  // namespace

int Channels(PixelFormat format) {
  switch (format) {
    case PixelFormat::kGray:
      return 1;
    case PixelFormat::kBgr:
    case PixelFormat::kRgb:
      return 3;
    case PixelFormat::kBgra:
    case PixelFormat::kRgba:
      return 4;
  }
  return 0;
}

std::string ValidateOptions(const Camera &camera, const Distortion &distortion,
                            const Options &options) {
  std::ostringstream error;
  if (options.width == 0 || options.height == 0 || options.width % 8 != 0 ||
      options.height % 8 != 0) {
    error << "image size " << options.width << "x" << options.height
          << " is invalid: width and height must be positive multiples of 8";
  } else if (static_cast<uint64_t>(options.width) * options.height >=
             (uint64_t{1} << 22)) {
    error << "image size " << options.width << "x" << options.height
          << " is too large: width * height must be below 4194304";
  } else if (options.threads < 1 || options.threads > 64) {
    error << "threads must be between 1 and 64, got " << options.threads;
  } else if (options.decimate != 1 && options.decimate != 2) {
    error << "decimate must be 1 or 2, got " << options.decimate;
  } else if (options.min_white_black_diff < 0 ||
             options.min_white_black_diff > 255) {
    error << "min_white_black_diff must be between 0 and 255, got "
          << options.min_white_black_diff;
  } else if (options.max_hamming < 0 || options.max_hamming > 2) {
    error << "max_hamming must be between 0 and 2, got "
          << options.max_hamming;
  } else if (!Finite(options.decode_sharpening) ||
             options.decode_sharpening < 0.0) {
    error << "decode_sharpening must be finite and >= 0";
  } else if (!Finite(options.tag_size_m) || options.tag_size_m < 0.0) {
    error << "tag_size_m must be finite and >= 0 (0 disables pose)";
  } else if (options.color_conversion != ColorConversion::kWhileDetecting &&
             options.color_conversion != ColorConversion::kBeforeDetecting) {
    error << "unknown color_conversion";
  } else if (!Finite(camera.fx) || !Finite(camera.fy) || camera.fx <= 0.0 ||
             camera.fy <= 0.0) {
    error << "camera fx and fy must be finite and > 0";
  } else if (!Finite(camera.cx) || !Finite(camera.cy)) {
    error << "camera cx and cy must be finite";
  } else if (!Finite(distortion.k1) || !Finite(distortion.k2) ||
             !Finite(distortion.p1) || !Finite(distortion.p2) ||
             !Finite(distortion.k3)) {
    error << "distortion coefficients must be finite";
  }
  return error.str();
}

std::string ValidateImage(const ImageView &image) {
  const int channels = Channels(image.format);
  if (channels == 0) return "unsupported pixel format";
  if (image.data == nullptr) return "image data is null";
  if (image.width == 0 || image.height == 0) return "image is empty";
  const uint64_t row_bytes = uint64_t{image.width} * channels;
  if (image.stride < row_bytes) {
    return "image stride is smaller than one row of pixels";
  }
  const uint64_t required =
      uint64_t{image.height - 1} * image.stride + row_bytes;
  if (image.size < required) {
    std::ostringstream error;
    error << "image buffer holds " << image.size << " bytes but a "
          << image.width << "x" << image.height << "x" << channels
          << " image with stride " << image.stride << " needs " << required;
    return error.str();
  }
  return {};
}

void ConvertToGray(const ImageView &image, std::vector<uint8_t> *gray) {
  // Validate before sizing the output from the image's dimensions.
  if (const std::string error = ValidateImage(image); !error.empty()) {
    throw std::invalid_argument(error);
  }
  gray->resize(static_cast<size_t>(image.width) * image.height);
  ConvertToGray(image, gray->data());
}

void ConvertToGray(const ImageView &image, uint8_t *gray) {
  if (const std::string error = ValidateImage(image); !error.empty()) {
    throw std::invalid_argument(error);
  }
  if (image.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      image.height > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("image is too large");
  }
  const int width = static_cast<int>(image.width);
  const int height = static_cast<int>(image.height);
  cv::Mat out(height, width, CV_8UC1, gray);
  const cv::Mat in(height, width, CV_MAKETYPE(CV_8U, Channels(image.format)),
                   const_cast<uint8_t *>(image.data), image.stride);
  switch (image.format) {
    case PixelFormat::kGray:
      in.copyTo(out);
      break;
    case PixelFormat::kBgr:
      cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
      break;
    case PixelFormat::kRgb:
      cv::cvtColor(in, out, cv::COLOR_RGB2GRAY);
      break;
    case PixelFormat::kBgra:
      cv::cvtColor(in, out, cv::COLOR_BGRA2GRAY);
      break;
    case PixelFormat::kRgba:
      cv::cvtColor(in, out, cv::COLOR_RGBA2GRAY);
      break;
  }
  // cvtColor must have written into the caller's buffer, not reallocated.
  if (out.data != gray) {
    throw std::runtime_error("color conversion reallocated its output");
  }
}

std::optional<Pose> SolveTagPose(
    const std::array<std::array<double, 2>, 4> &corners, const Camera &camera,
    const Distortion &distortion, double tag_size_m) {
  if (!(Finite(tag_size_m) && tag_size_m > 0.0)) return std::nullopt;
  const double s = tag_size_m / 2.0;
  // AprilTag's tag frame, in the detector's corner order.
  const std::vector<cv::Point3d> object_points = {
      {-s, s, 0.0}, {s, s, 0.0}, {s, -s, 0.0}, {-s, -s, 0.0}};
  std::vector<cv::Point2d> image_points;
  image_points.reserve(4);
  for (const auto &corner : corners) {
    if (!Finite(corner[0]) || !Finite(corner[1])) return std::nullopt;
    // AprilTag puts pixel centers at +0.5; OpenCV calibrations put them at
    // integers.
    image_points.emplace_back(corner[0] - kAprilTagPixelOffset,
                              corner[1] - kAprilTagPixelOffset);
  }
  const cv::Matx33d camera_matrix(camera.fx, 0.0, camera.cx, 0.0, camera.fy,
                                  camera.cy, 0.0, 0.0, 1.0);
  // The detector reports corners in the original (distorted) image, so the
  // solve must undistort them with the same model the detector used.
  const cv::Matx<double, 1, 5> coefficients(distortion.k1, distortion.k2,
                                            distortion.p1, distortion.p2,
                                            distortion.k3);
  cv::Mat rvec, tvec;
  try {
    if (!cv::solvePnP(object_points, image_points, camera_matrix, coefficients,
                      rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE)) {
      return std::nullopt;
    }
  } catch (const cv::Exception &) {
    return std::nullopt;
  }
  cv::Matx33d rotation;
  cv::Rodrigues(rvec, rotation);
  Pose pose;
  for (int i = 0; i < 9; ++i) {
    if (!Finite(rotation.val[i])) return std::nullopt;
    pose.rotation[i] = rotation.val[i];
  }
  for (int i = 0; i < 3; ++i) {
    pose.translation[i] = tvec.at<double>(i);
    if (!Finite(pose.translation[i])) return std::nullopt;
  }
  return pose;
}

}  // namespace cuda_tags
