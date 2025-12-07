#include "cuda_tags.h"
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "frc971/orin/971apriltag.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

using namespace frc971::apriltag;

apriltag_detector_t *maketagdetector(apriltag_family_t *tag_family,
                                     int nthreads, bool debug) {
  apriltag_detector_t *tag_detector = apriltag_detector_create();

  apriltag_detector_add_family_bits(tag_detector, tag_family, 1);

  tag_detector->nthreads = nthreads;
  tag_detector->wp = workerpool_create(tag_detector->nthreads);
  tag_detector->qtp.min_white_black_diff = 5;
  tag_detector->debug = debug;

  return tag_detector;
}

CudaTagsWrapper::CudaTagsWrapper(TagType tagType, CameraMatrix camMatrix,
                                 DistCoeffs camDistCoeffs, int nthreads,
                                 int imgWidth, int imgHeight) {
  this->camera_matrix =
      CameraMatrix{camMatrix.fx, camMatrix.cx, camMatrix.fy, camMatrix.cy};
  this->dist_coeffs =
      DistCoeffs{camDistCoeffs.k1, camDistCoeffs.k2, camDistCoeffs.p1,
                 camDistCoeffs.p2, camDistCoeffs.k3};

  auto tag_family = tag36h11_create();
  switch (tagType) {
  case TagType::tag36h11:
    tag_family = tag36h11_create();
  }

  this->tags_detector = maketagdetector(tag_family, nthreads, false);
  this->gpu_processor =
      new GpuDetector(imgWidth, imgHeight, this->tags_detector,
                      this->camera_matrix, this->dist_coeffs);
}

CudaTagsWrapper::~CudaTagsWrapper() {
  delete this->gpu_processor;
  if (this->tags_detector != nullptr) {
    apriltag_detector_destroy(this->tags_detector);
    this->tags_detector = nullptr;
  }
}

std::vector<DetectionResult>
CudaTagsWrapper::process(uint8_t *data, int width, int height, int channels) {
  if (width != this->gpu_processor->width() ||
      height != this->gpu_processor->height()) {
    delete this->gpu_processor;
    this->gpu_processor = new frc971::apriltag::GpuDetector(
        width, height, this->tags_detector, this->camera_matrix,
        this->dist_coeffs);
  }

  // Always convert to grayscale before detection
  cv::Mat input_img(height, width, (channels == 1 ? CV_8UC1 : CV_8UC3), data);

  cv::Mat gray;
  if (channels == 1) {
    gray = input_img;
  } else {
    cv::cvtColor(input_img, gray, cv::COLOR_BGR2GRAY);
  }

  // Detect in grayscale
  this->gpu_processor->DetectGrayHost(gray.data);

  const zarray_t *detections = this->gpu_processor->Detections();
  return to_detection_results(detections);
}

std::vector<DetectionResult> to_detection_results(const zarray_t *detections) {
  std::vector<DetectionResult> results;
  for (int i = 0; i < zarray_size(detections); ++i) {
    apriltag_detection_t *detection;
    zarray_get(detections, i, &detection);
    results.push_back(
        DetectionResult{detection->id, detection->hamming,
                        detection->decision_margin, to_3x3_array(detection->H),
                        to_2d_array(detection->c), to_4x2_array(detection->p)});
  }

  return results;
}

std::array<std::array<double, 3>, 3> to_3x3_array(const double mat[3][3]) {
  return {{{{mat[0][0], mat[0][1], mat[0][2]}},
           {{mat[1][0], mat[1][1], mat[1][2]}},
           {{mat[2][0], mat[2][1], mat[2][2]}}}};
}

std::array<double, 2> to_2d_array(const double array[2]) {
  return {{array[0], array[1]}};
}

std::array<std::array<double, 2>, 4> to_4x2_array(const double array[4][2]) {
  return {{{array[0][0], array[0][1]},
           {array[1][0], array[1][1]},
           {array[2][0], array[2][1]},
           {array[3][0], array[3][1]}}};
}

std::array<std::array<double, 3>, 3> to_3x3_array(const matd_t *mat) {
  return {{{{mat->data[0], mat->data[1], mat->data[2]}},
           {{mat->data[3], mat->data[4], mat->data[5]}},
           {{mat->data[6], mat->data[7], mat->data[8]}}}};
}