#pragma once

#include "frc971/orin/971apriltag.h"
#include "rust/cxx.h"
#include "third_party/apriltag/apriltag.h"
#include <memory>
#include <opencv2/core.hpp>

enum class TagType : uint8_t;

struct CameraMatrix;
struct DistCoeffs;
struct DetectionResult;
struct RustImage;

class CudaTagsWrapper {
public:
  CudaTagsWrapper(TagType tagType, CameraMatrix camMatrix,
                  DistCoeffs camDistCoeffs, int nthreads, int imgHeight,
                  int imgWidth);
  ~CudaTagsWrapper();

  rust::Vec<DetectionResult> process(RustImage image);

private:
  frc971::apriltag::GpuDetector *gpu_processor;
  apriltag_detector_t *tags_detector;
  frc971::apriltag::CameraMatrix camera_matrix;
  frc971::apriltag::DistCoeffs dist_coeffs;
};

std::unique_ptr<CudaTagsWrapper>
make_cuda_tag_detector(TagType tagType, CameraMatrix camMatrix,
                       DistCoeffs camDistCoeffs, int nthreads, int imgWidth,
                       int imgHeight);

cv::Mat process_with_opencv(rust::Slice<std::uint8_t> pixels,
                            std::int32_t width, std::int32_t height,
                            std::int32_t stride, std::int32_t channels);

rust::Vec<DetectionResult> to_rust_vec(const zarray_t *detections);

std::array<std::array<double, 3>, 3> to_3x3_array(const double mat[3][3]);

std::array<double, 2> to_2d_array(const double array[2]);

std::array<std::array<double, 2>, 4> to_4x2_array(const double array[4][2]);

std::array<std::array<double, 3>, 3> to_3x3_array(const matd_t *mat);