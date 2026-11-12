// Detector for builds without CUDA: configuration is still validated, but
// construction always fails. Lets the bindings and the C ABI build and be
// tested on machines without a GPU.
#include <stdexcept>
#include <string>

#include "cuda_tags/detector.hpp"

namespace cuda_tags {

bool CudaAvailable() { return false; }

struct Detector::Impl {
  Options options;
  Camera camera;
  Distortion distortion;
  std::vector<Detection> detections;
};

Detector::Detector(const Camera &camera, const Distortion &distortion,
                   const Options &options) {
  if (const std::string error = ValidateOptions(camera, distortion, options);
      !error.empty()) {
    throw std::invalid_argument(error);
  }
  throw std::runtime_error(
      "CudaTags was built without CUDA; GPU detection is unavailable");
}

Detector::~Detector() = default;

const Options &Detector::options() const { return impl_->options; }
const Camera &Detector::camera() const { return impl_->camera; }
const Distortion &Detector::distortion() const { return impl_->distortion; }

const std::vector<Detection> &Detector::Detect(const ImageView &) {
  throw std::runtime_error("CudaTags was built without CUDA");
}

const std::vector<Detection> &Detector::DetectGray(const uint8_t *, size_t) {
  throw std::runtime_error("CudaTags was built without CUDA");
}

}  // namespace cuda_tags
