// CPU-side tests of the core and the C ABI: option validation, image
// conversion (strides, channel orders), the shared gray formula against
// OpenCV, the pose solve (with distortion) and the C ABI's error handling.
// Runs with or without CUDA.
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "apriltag_971/ct_pixel_layout.h"

#include "cuda_tags/cuda_tags.h"
#include "cuda_tags/detector.hpp"
#include "test_util.h"

namespace {

using cuda_tags::Camera;
using cuda_tags::Distortion;
using cuda_tags::ImageView;
using cuda_tags::Options;
using cuda_tags::PixelFormat;

Options ValidOptions() {
  Options options;
  options.width = 640;
  options.height = 480;
  return options;
}

const Camera kCamera{600.0, 610.0, 320.0, 240.0};

void TestValidateOptions() {
  EXPECT_TRUE(cuda_tags::ValidateOptions(kCamera, {}, ValidOptions()).empty());
  auto invalid = [](auto mutate) {
    Options options = ValidOptions();
    Camera camera = kCamera;
    Distortion distortion;
    mutate(options, camera, distortion);
    return !cuda_tags::ValidateOptions(camera, distortion, options).empty();
  };
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) { o.width = 0; }));
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) { o.width = 642; }));
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) { o.height = 484 + 1; }));
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) {
    o.width = 2048;
    o.height = 2048;
  }));
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) { o.threads = 0; }));
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) { o.decimate = 3; }));
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) { o.max_hamming = 3; }));
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) {
    o.min_white_black_diff = 256;
  }));
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) { o.tag_size_m = -1; }));
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) {
    o.tag_size_m = std::nan("");
  }));
  EXPECT_TRUE(invalid([](Options &, Camera &c, Distortion &) { c.fx = 0; }));
  EXPECT_TRUE(invalid([](Options &, Camera &c, Distortion &) {
    c.cy = INFINITY;
  }));
  EXPECT_TRUE(invalid([](Options &, Camera &, Distortion &d) {
    d.k3 = std::nan("");
  }));
  EXPECT_TRUE(invalid([](Options &o, Camera &, Distortion &) {
    o.color_conversion = static_cast<cuda_tags::ColorConversion>(7);
  }));
  EXPECT_TRUE(!invalid([](Options &o, Camera &, Distortion &) {
    o.color_conversion = cuda_tags::ColorConversion::kBeforeDetecting;
  }));
  // 1920x1080 and decimate 1 are fine.
  EXPECT_TRUE(!invalid([](Options &o, Camera &, Distortion &) {
    o.width = 1920;
    o.height = 1080;
    o.decimate = 1;
  }));
}

void TestConvertToGray() {
  // 2x8 BGR image, padded to a 32-byte stride.
  const uint32_t width = 8, height = 2;
  const size_t stride = 32;
  std::vector<uint8_t> bgr(stride * height, 0xEE);  // padding is garbage
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      uint8_t *pixel = &bgr[y * stride + x * 3];
      pixel[0] = 255;  // blue
      pixel[1] = 0;
      pixel[2] = 0;
    }
  }
  std::vector<uint8_t> gray;
  ImageView view{bgr.data(), bgr.size(), width, height, stride,
                 PixelFormat::kBgr};
  cuda_tags::ConvertToGray(view, &gray);
  EXPECT_TRUE(gray.size() == width * height);
  // BT.601: blue weighs 0.114, red 0.299.
  EXPECT_NEAR(gray[0], 29, 1);
  EXPECT_NEAR(gray[width * height - 1], 29, 1);
  view.format = PixelFormat::kRgb;  // same bytes read as RGB: pure red
  cuda_tags::ConvertToGray(view, &gray);
  EXPECT_NEAR(gray[0], 76, 1);

  // Padded gray rows are copied without their padding.
  std::vector<uint8_t> padded(16 * 2, 9);
  for (int x = 0; x < 8; ++x) {
    padded[x] = static_cast<uint8_t>(x);
    padded[16 + x] = static_cast<uint8_t>(100 + x);
  }
  cuda_tags::ConvertToGray({padded.data(), padded.size(), 8, 2, 16,
                            PixelFormat::kGray},
                           &gray);
  EXPECT_TRUE(gray[7] == 7 && gray[8] == 100 && gray[15] == 107);

  // BGRA and RGBA.
  std::vector<uint8_t> bgra = {0, 0, 255, 255};  // 1x1 red pixel
  cuda_tags::ConvertToGray({bgra.data(), 4, 1, 1, 4, PixelFormat::kBgra}, &gray);
  EXPECT_NEAR(gray[0], 76, 1);
  cuda_tags::ConvertToGray({bgra.data(), 4, 1, 1, 4, PixelFormat::kRgba}, &gray);
  EXPECT_NEAR(gray[0], 29, 1);

  // Short buffers and bad strides are rejected.
  auto rejects = [&](ImageView bad) {
    try {
      cuda_tags::ConvertToGray(bad, &gray);
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  };
  EXPECT_TRUE(rejects({bgr.data(), bgr.size() - 9, width, height, stride,
                       PixelFormat::kBgr}));
  EXPECT_TRUE(rejects({bgr.data(), bgr.size(), width, height, 20,
                       PixelFormat::kBgr}));
  EXPECT_TRUE(rejects({nullptr, 0, width, height, stride, PixelFormat::kBgr}));
  // The last row needs only width * channels bytes, not a full stride.
  EXPECT_TRUE(!rejects({bgr.data(), stride + width * 3, width, height, stride,
                        PixelFormat::kBgr}));
}

// The GPU kernels and the CPU decode read color through ct_pixel_gray; it
// must equal cv::cvtColor for every color, in both channel orders, so every
// color path gives the same detections as converting with OpenCV first.
void TestPixelGrayMatchesOpenCv() {
  cv::Mat colors(4096, 4096, CV_8UC3);
  for (int i = 0; i < (1 << 24); ++i) {
    colors.data[3 * i] = static_cast<uint8_t>(i);
    colors.data[3 * i + 1] = static_cast<uint8_t>(i >> 8);
    colors.data[3 * i + 2] = static_cast<uint8_t>(i >> 16);
  }
  const ct_pixel_layout bgr{3, {2, 1, 0}, {9798, 19235, 3735}};
  const ct_pixel_layout rgb{3, {0, 1, 2}, {9798, 19235, 3735}};
  for (const bool rgb_order : {false, true}) {
    cv::Mat gray;
    cv::cvtColor(colors, gray,
                 rgb_order ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY);
    long mismatches = 0;
    for (int i = 0; i < (1 << 24); ++i) {
      mismatches += gray.data[i] !=
                    ct_pixel_gray(colors.data, i, rgb_order ? &rgb : &bgr);
    }
    EXPECT_TRUE(mismatches == 0);
  }
  // Gray (NULL layout) reads the byte as is.
  const uint8_t row[] = {7, 200};
  EXPECT_TRUE(ct_pixel_gray(row, 1, nullptr) == 200);
}

void TestSolveTagPose() {
  const double tag_size = 0.1651, s = tag_size / 2;
  const Distortion distortion{-0.31, 0.12, 0.0012, -0.0008, -0.02};
  const cv::Matx33d camera_matrix(kCamera.fx, 0, kCamera.cx, 0, kCamera.fy,
                                  kCamera.cy, 0, 0, 1);
  const cv::Vec3d rvec(0.2, -0.5, 0.15), tvec(0.3, -0.12, 1.4);
  const std::vector<cv::Point3d> object = {
      {-s, s, 0}, {s, s, 0}, {s, -s, 0}, {-s, -s, 0}};
  std::vector<cv::Point2d> projected;
  cv::projectPoints(object, rvec, tvec, camera_matrix,
                    cv::Matx<double, 1, 5>(distortion.k1, distortion.k2,
                                           distortion.p1, distortion.p2,
                                           distortion.k3),
                    projected);
  std::array<std::array<double, 2>, 4> corners;
  // As the detector reports them: AprilTag's pixel centers are at +0.5.
  for (int i = 0; i < 4; ++i) {
    corners[i] = {projected[i].x + 0.5, projected[i].y + 0.5};
  }

  const auto pose = cuda_tags::SolveTagPose(corners, kCamera, distortion, tag_size);
  EXPECT_TRUE(pose.has_value());
  if (pose) {
    cv::Matx33d expected;
    cv::Rodrigues(rvec, expected);
    for (int i = 0; i < 9; ++i) EXPECT_NEAR(pose->rotation[i], expected.val[i], 1e-6);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(pose->translation[i], tvec[i], 1e-6);
  }
  // Ignoring the distortion the corners were projected with is measurably
  // wrong: the detector reports distorted corners, so the solve must use it.
  const auto ignoring = cuda_tags::SolveTagPose(corners, kCamera, {}, tag_size);
  EXPECT_TRUE(ignoring.has_value());
  if (ignoring) EXPECT_TRUE(std::abs(ignoring->translation[0] - tvec[0]) > 1e-3);

  EXPECT_TRUE(!cuda_tags::SolveTagPose(corners, kCamera, distortion, 0.0));
  corners[2][0] = std::nan("");
  EXPECT_TRUE(!cuda_tags::SolveTagPose(corners, kCamera, distortion, tag_size));
}

void TestCAbi() {
  EXPECT_TRUE(ct_abi_version() == CT_ABI_VERSION);
  EXPECT_TRUE(ct_abi_version() == 1);
  EXPECT_TRUE(ct_last_error() != nullptr);

  ct_options options;
  ct_options_init(&options, 640, 480);
  EXPECT_TRUE(options.width == 640 && options.height == 480);
  EXPECT_TRUE(options.threads == 1 && options.decimate == 2);
  EXPECT_TRUE(options.min_white_black_diff == 5 && options.max_hamming == 1);
  EXPECT_TRUE(options.refine_edges == 1 && options.tag_size_m == 0.0);
  EXPECT_NEAR(options.decode_sharpening, 0.25, 0.0);

  const ct_camera camera{600, 600, 320, 240, 0, 0, 0, 0, 0};
  // Invalid arguments: NULL plus a message, no exception.
  EXPECT_TRUE(ct_create(&camera, 641, 480, 1, 0.1) == nullptr);
  EXPECT_TRUE(std::string(ct_last_error()).find("multiples of 8") !=
              std::string::npos);
  EXPECT_TRUE(ct_create(nullptr, 640, 480, 1, 0.1) == nullptr);
  EXPECT_TRUE(ct_create(&camera, 640, 480, 0, 0.1) == nullptr);
  EXPECT_TRUE(ct_create_with_options(&camera, nullptr) == nullptr);
  EXPECT_TRUE(ct_detect(nullptr, nullptr, 0) == -1);
  ct_detection detection;
  EXPECT_TRUE(ct_detection_at(nullptr, 0, &detection) == -1);
  double homography[9];
  EXPECT_TRUE(ct_detection_homography(nullptr, 0, homography) == -1);
  ct_destroy(nullptr);
  const uint8_t pixel = 0;
  EXPECT_TRUE(ct_detect_image(nullptr, &pixel, 1, 1, CT_FORMAT_RGB) == -1);
  EXPECT_TRUE(ct_input_buffer(nullptr, nullptr) == nullptr);
  size_t size = 123;
  EXPECT_TRUE(ct_input_buffer(nullptr, &size) == nullptr && size == 0);
  EXPECT_TRUE(ct_set_color_conversion(nullptr, CT_COLOR_WHILE_DETECTING) ==
              -1);

  // The error is thread-local and survives successful calls.
  ct_create(&camera, 641, 480, 1, 0.1);
  const std::string error = ct_last_error();
  EXPECT_TRUE(ct_abi_version() == 1);
  EXPECT_TRUE(error == ct_last_error());

  EXPECT_TRUE(ct_cuda_available() == (cuda_tags::CudaAvailable() ? 1 : 0));
  if (!ct_cuda_available()) {
    EXPECT_TRUE(ct_create(&camera, 640, 480, 1, 0.1) == nullptr);
    EXPECT_TRUE(std::string(ct_last_error()).find("without CUDA") !=
                std::string::npos);
  }
}

void TestDetectorWithoutCuda() {
  if (cuda_tags::CudaAvailable()) return;
  bool threw = false;
  try {
    cuda_tags::Detector detector(kCamera, {}, ValidOptions());
  } catch (const std::runtime_error &) {
    threw = true;
  }
  EXPECT_TRUE(threw);
  // Configuration errors are still reported as invalid arguments.
  threw = false;
  try {
    Options options = ValidOptions();
    options.width = 1;
    cuda_tags::Detector detector(kCamera, {}, options);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

}  // namespace

int main() {
  TestValidateOptions();
  TestConvertToGray();
  TestPixelGrayMatchesOpenCv();
  TestSolveTagPose();
  TestCAbi();
  TestDetectorWithoutCuda();
  return test::Finish("core_cpu_test");
}
