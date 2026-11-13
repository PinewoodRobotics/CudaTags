// Detects the golden image and checks ids, corners and poses against the
// ground truth it was rendered from.
//
//   golden_test <image.pgm> <scene.txt> cpu   AprilTag's CPU detector, built
//                                             from apriltag_971/apriltag.c
//                                             (runs anywhere; validates the
//                                             golden data, the apriltag split
//                                             and the core's pose solve)
//   golden_test <image.pgm> <scene.txt> gpu   cuda_tags::Detector (CUDA),
//                                             also checking that color and
//                                             zero-copy frames give exactly
//                                             the gray frame's detections
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "cuda_tags/detector.hpp"
#include "golden/golden_scene.h"
#include "test_util.h"

namespace {

// Corners of a synthetic, noise-free image: sub-pixel accuracy is expected.
constexpr double kCornerTolerancePx = 0.3;
// Detections use AprilTag's pixel convention (centers at +0.5); the scene's
// ground truth uses OpenCV's (centers at integers).
constexpr double kAprilTagPixelOffset = 0.5;
constexpr double kTranslationTolerance = 0.02;   // fraction of distance
constexpr double kRotationToleranceDeg = 1.0;

double RotationErrorDeg(const std::array<double, 9> &a,
                        const std::array<double, 9> &b) {
  // angle of a^T b
  double trace = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) trace += a[k * 3 + i] * b[k * 3 + i];
  }
  const double c = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(c) * 180.0 / M_PI;
}

std::vector<cuda_tags::Detection> DetectCpu(const cv::Mat &gray) {
  apriltag_family_t *family = tag36h11_create();
  apriltag_detector_t *detector = apriltag_detector_create();
  apriltag_detector_add_family_bits(detector, family, 1);
  detector->quad_decimate = 2.0f;
  detector->nthreads = 1;
  image_u8_t image{gray.cols, gray.rows, static_cast<int32_t>(gray.step[0]),
                   gray.data};
  zarray_t *found = apriltag_detector_detect(detector, &image);
  std::vector<cuda_tags::Detection> detections;
  for (int i = 0; i < zarray_size(found); ++i) {
    apriltag_detection_t *tag;
    zarray_get(found, i, &tag);
    cuda_tags::Detection out;
    out.id = tag->id;
    out.hamming = tag->hamming;
    out.decision_margin = tag->decision_margin;
    out.center = {tag->c[0], tag->c[1]};
    for (int c = 0; c < 4; ++c) out.corners[c] = {tag->p[c][0], tag->p[c][1]};
    detections.push_back(out);
  }
  apriltag_detections_destroy(found);
  apriltag_detector_destroy(detector);
  tag36h11_destroy(family);
  return detections;
}

bool Same(const std::vector<cuda_tags::Detection> &a,
          const std::vector<cuda_tags::Detection> &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].id != b[i].id || a[i].hamming != b[i].hamming ||
        a[i].decision_margin != b[i].decision_margin ||
        a[i].center != b[i].center || a[i].corners != b[i].corners ||
        a[i].homography != b[i].homography ||
        a[i].pose.has_value() != b[i].pose.has_value() ||
        (a[i].pose && (a[i].pose->rotation != b[i].pose->rotation ||
                       a[i].pose->translation != b[i].pose->translation))) {
      return false;
    }
  }
  return true;
}

cuda_tags::ImageView View(const cv::Mat &image, cuda_tags::PixelFormat format) {
  cuda_tags::ImageView view;
  view.data = image.data;
  view.width = static_cast<uint32_t>(image.cols);
  view.height = static_cast<uint32_t>(image.rows);
  view.stride = image.step[0];
  view.size = (image.rows - 1) * image.step[0] + image.cols * image.elemSize();
  view.format = format;
  return view;
}

// Color frames (every format; packed and padded; from the caller's memory
// and from the detector's input buffer; both conversions) must give exactly
// the detections of the gray frame cv::cvtColor makes from them.
void CheckColorFrames(cuda_tags::Detector &detector, const cv::Mat &gray) {
  // A warm cast with uneven channels and a little noise, so every channel's
  // weight matters.
  std::vector<cv::Mat> planes = {gray * 0.61, gray * 0.83, gray * 1.0};
  cv::Mat noise(gray.size(), CV_8UC1);
  cv::randu(noise, 0, 6);
  planes[0] += noise;
  cv::Mat bgr, rgb, bgra, rgba, expected;
  cv::merge(planes, bgr);
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
  cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
  cv::cvtColor(bgr, expected, cv::COLOR_BGR2GRAY);
  const std::vector<cuda_tags::Detection> reference =
      detector.DetectGray(expected.data, expected.total());
  EXPECT_TRUE(reference.size() == 3);

  struct Case {
    const cv::Mat *image;
    cuda_tags::PixelFormat format;
  };
  const Case cases[] = {{&bgr, cuda_tags::PixelFormat::kBgr},
                        {&rgb, cuda_tags::PixelFormat::kRgb},
                        {&bgra, cuda_tags::PixelFormat::kBgra},
                        {&rgba, cuda_tags::PixelFormat::kRgba}};
  uint8_t *buffer = detector.InputBuffer();
  EXPECT_TRUE(buffer != nullptr);
  for (const auto conversion : {cuda_tags::ColorConversion::kWhileDetecting,
                                cuda_tags::ColorConversion::kBeforeDetecting}) {
    detector.SetColorConversion(conversion);
    for (const Case &c : cases) {
      // The caller's memory, packed and with padded rows.
      EXPECT_TRUE(Same(detector.Detect(View(*c.image, c.format)), reference));
      cv::Mat padded(c.image->rows, c.image->cols + 13, c.image->type());
      cv::Mat inner = padded.colRange(0, c.image->cols);
      c.image->copyTo(inner);
      EXPECT_TRUE(Same(detector.Detect(View(inner, c.format)), reference));
      // The input buffer, read in place: packed, and padded where that
      // fits (the buffer holds width * height * 4 bytes).
      const size_t row = c.image->cols * c.image->elemSize();
      for (const size_t stride : {row, row + 40}) {
        if ((c.image->rows - 1) * stride + row > detector.InputBufferSize()) {
          continue;
        }
        for (int y = 0; y < c.image->rows; ++y) {
          std::memcpy(buffer + y * stride, c.image->ptr(y), row);
        }
        cv::Mat in_buffer(c.image->rows, c.image->cols, c.image->type(),
                          buffer, stride);
        EXPECT_TRUE(
            Same(detector.Detect(View(in_buffer, c.format)), reference));
      }
    }
  }
  detector.SetColorConversion(cuda_tags::ColorConversion::kBeforeDetecting);

  // Gray in the input buffer, read in place.
  std::memcpy(buffer, expected.data, expected.total());
  EXPECT_TRUE(Same(detector.DetectGray(buffer, expected.total()), reference));
}

}  // namespace

int Run(int argc, char **argv) {
  if (argc != 4 || (std::strcmp(argv[3], "cpu") != 0 &&
                    std::strcmp(argv[3], "gpu") != 0)) {
    std::cerr << "usage: golden_test <image.pgm> <scene.txt> cpu|gpu\n";
    return 2;
  }
  const bool gpu = std::strcmp(argv[3], "gpu") == 0;
  const golden::Scene scene = golden::ReadScene(argv[2]);
  golden::GrayImage pgm = golden::ReadPgm(argv[1]);
  const cv::Mat gray(pgm.height, pgm.width, CV_8UC1, pgm.pixels.data());
  if (gray.cols != scene.width || gray.rows != scene.height) {
    std::cerr << "cannot read " << argv[1] << " at the scene's size\n";
    return 1;
  }
  const cuda_tags::Camera camera{scene.fx, scene.fy, scene.cx, scene.cy};
  const cuda_tags::Distortion distortion;

  std::vector<cuda_tags::Detection> detections;
  if (gpu) {
    cuda_tags::Options options;
    options.width = scene.width;
    options.height = scene.height;
    options.threads = 2;
    options.tag_size_m = scene.tag_size_m;
    cuda_tags::Detector detector(camera, distortion, options);
    // Twice: the second run must not depend on state from the first.
    detector.DetectGray(gray.data, gray.total());
    detections = detector.DetectGray(gray.data, gray.total());

    // A frame of another size is rejected, not reallocated for.
    bool rejected = false;
    try {
      std::vector<uint8_t> small(64 * 64);
      detector.DetectGray(small.data(), small.size());
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    EXPECT_TRUE(rejected);

    // Detections of the golden gray frame from the input buffer are
    // identical to those from the caller's memory.
    std::memcpy(detector.InputBuffer(), gray.data, gray.total());
    EXPECT_TRUE(Same(detector.DetectGray(detector.InputBuffer(), gray.total()),
                     detections));
    CheckColorFrames(detector, gray);
  } else {
    detections = DetectCpu(gray);
    for (auto &detection : detections) {
      detection.pose = cuda_tags::SolveTagPose(detection.corners, camera,
                                               distortion, scene.tag_size_m);
    }
  }

  EXPECT_TRUE(detections.size() == scene.tags.size());
  std::map<int, const cuda_tags::Detection *> by_id;
  for (const auto &detection : detections) {
    EXPECT_TRUE(by_id.count(detection.id) == 0);
    by_id[detection.id] = &detection;
    EXPECT_TRUE(detection.hamming == 0);
  }
  for (const golden::Tag &expected : scene.tags) {
    const auto it = by_id.find(expected.id);
    if (it == by_id.end()) {
      test::Fail(__FILE__, __LINE__,
                 "tag " + std::to_string(expected.id) + " not detected");
      continue;
    }
    const cuda_tags::Detection &actual = *it->second;
    std::cout << "tag " << actual.id << ": margin " << actual.decision_margin;
    double worst = 0.0;
    for (int c = 0; c < 4; ++c) {
      for (int k = 0; k < 2; ++k) {
        const double corner = actual.corners[c][k] - kAprilTagPixelOffset;
        worst = std::max(worst, std::abs(corner - expected.corners[c][k]));
        EXPECT_NEAR(corner, expected.corners[c][k], kCornerTolerancePx);
      }
    }
    std::cout << ", worst corner error " << worst << " px";
    if (!actual.pose) {
      test::Fail(__FILE__, __LINE__,
                 "tag " + std::to_string(expected.id) + " has no pose");
      std::cout << "\n";
      continue;
    }
    const double distance =
        std::sqrt(expected.translation[0] * expected.translation[0] +
                  expected.translation[1] * expected.translation[1] +
                  expected.translation[2] * expected.translation[2]);
    for (int k = 0; k < 3; ++k) {
      EXPECT_NEAR(actual.pose->translation[k], expected.translation[k],
                  kTranslationTolerance * distance);
    }
    const double angle =
        RotationErrorDeg(actual.pose->rotation, expected.rotation);
    std::cout << ", rotation error " << angle << " deg\n";
    EXPECT_TRUE(angle <= kRotationToleranceDeg);
  }
  return test::Finish(gpu ? "golden_test gpu" : "golden_test cpu");
}

int main(int argc, char **argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "golden_test: " << error.what() << "\n";
    return 1;
  }
}
