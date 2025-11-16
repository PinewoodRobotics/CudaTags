#include "cuda_tags_wrapper.h"
#include "frc971/orin/971apriltag.h"
#include "target/cxxbridge/CudaTags/src/bridge.rs.h"
#include "third_party/apriltag/apriltag.h"
#include "third_party/apriltag/tag36h11.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

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

rust::Vec<DetectionResult> to_rust_vec(const zarray_t *detections) {
  rust::Vec<DetectionResult> result;
  for (int i = 0; i < zarray_size(detections); ++i) {
    apriltag_detection_t *detection;
    zarray_get(detections, i, &detection);
    result.push_back(
        DetectionResult{detection->id, detection->hamming,
                        detection->decision_margin, to_3x3_array(detection->H),
                        to_2d_array(detection->c), to_4x2_array(detection->p)});
  }
  return result;
}

rust::Vec<DetectionResult> CudaTagsWrapper::process(RustImage image) {
  auto cvImage = process_with_opencv(image.pixels, image.width, image.height,
                                     image.stride, image.channels);
  if (cvImage.cols != this->gpu_processor->width() ||
      cvImage.rows != this->gpu_processor->height()) {
    throw std::runtime_error("Image size mismatch");
  }

  if (cvImage.type() != CV_8UC1) {
    this->gpu_processor->DetectColor(cvImage.ptr());
  } else {
    this->gpu_processor->DetectGrayHost(cvImage.ptr());
  }

  return to_rust_vec(this->gpu_processor->Detections());
}

CudaTagsWrapper::CudaTagsWrapper(TagType tagType, CameraMatrix camMatrix,
                                 DistCoeffs camDistCoeffs, int nthreads,
                                 int imgWidth, int imgHeight) {
  this->camera_matrix = frc971::apriltag::CameraMatrix{
      camMatrix.fx, camMatrix.cx, camMatrix.fy, camMatrix.cy};
  this->dist_coeffs = frc971::apriltag::DistCoeffs{
      camDistCoeffs.k1, camDistCoeffs.k2, camDistCoeffs.p1, camDistCoeffs.p2,
      camDistCoeffs.k3};

  auto tag_family = tag36h11_create();
  switch (tagType) {
  case TagType::tag36h11:
    tag_family = tag36h11_create();
  }

  this->tags_detector = maketagdetector(tag_family, nthreads, false);
  this->gpu_processor = new frc971::apriltag::GpuDetector(
      imgWidth, imgHeight, this->tags_detector, this->camera_matrix,
      this->dist_coeffs);
}

CudaTagsWrapper::~CudaTagsWrapper() {
  delete this->gpu_processor;
  delete this->tags_detector;
}

cv::Mat process_with_opencv(rust::Slice<std::uint8_t> pixels,
                            std::int32_t width, std::int32_t height,
                            std::int32_t stride, std::int32_t channels) {
  auto *data = pixels.data();

  int type;
  switch (channels) {
  case 1:
    type = CV_8UC1;
    break;
  case 3:
    type = CV_8UC3;
    break;
  case 4:
    type = CV_8UC4;
    break;
  default:
    throw std::runtime_error("Unsupported channel count");
  }

  cv::Mat img(height, width, type, static_cast<void *>(data), stride);
  return img;
}

std::unique_ptr<CudaTagsWrapper>
make_cuda_tag_detector(TagType tagType, CameraMatrix camMatrix,
                       DistCoeffs camDistCoeffs, int nthreads, int imgWidth,
                       int imgHeight) {
  return std::make_unique<CudaTagsWrapper>(tagType, camMatrix, camDistCoeffs,
                                           nthreads, imgWidth, imgHeight);
}