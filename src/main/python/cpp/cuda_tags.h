#include "apriltag/apriltag.h"
#include "frc971/orin/971apriltag.h"
#include <memory>
#include <opencv2/core.hpp>

enum class TagType {
  tag36h11 = 1,
};

struct DetectionResult {
  int id;
  int hamming;
  float decision_margin;
  std::array<std::array<double, 3>, 3> homography;
  std::array<double, 2> center;
  std::array<std::array<double, 2>, 4> corners;
};

class CudaTagsWrapper {
public:
  CudaTagsWrapper(TagType tagType, frc971::apriltag::CameraMatrix camMatrix,
                  frc971::apriltag::DistCoeffs camDistCoeffs, int nthreads,
                  int imgHeight, int imgWidth);
  ~CudaTagsWrapper();

  std::vector<DetectionResult> process(uint8_t *data, int width, int height,
                                       int channels);

private:
  frc971::apriltag::GpuDetector *gpu_processor;
  apriltag_detector_t *tags_detector;
  frc971::apriltag::CameraMatrix camera_matrix;
  frc971::apriltag::DistCoeffs dist_coeffs;
};

std::array<std::array<double, 3>, 3> to_3x3_array(const double mat[3][3]);
std::array<double, 2> to_2d_array(const double array[2]);
std::array<std::array<double, 2>, 4> to_4x2_array(const double array[4][2]);
std::array<std::array<double, 3>, 3> to_3x3_array(const matd_t *mat);
std::vector<DetectionResult> to_detection_results(const zarray_t *detections);