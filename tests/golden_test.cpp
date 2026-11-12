// Detects the golden image and checks ids, corners and poses against the
// ground truth it was rendered from.
//
//   golden_test <image.pgm> <scene.txt> cpu   AprilTag's CPU detector, built
//                                             from apriltag_971/apriltag.c
//                                             (runs anywhere; validates the
//                                             golden data, the apriltag split
//                                             and the core's pose solve)
//   golden_test <image.pgm> <scene.txt> gpu   cuda_tags::Detector (CUDA)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

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
