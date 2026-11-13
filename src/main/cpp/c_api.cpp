// C ABI over the core. Every entry point catches all exceptions.
#include "cuda_tags/cuda_tags.h"

#include <exception>
#include <new>
#include <string>

#include "cuda_tags/detector.hpp"

namespace {

thread_local std::string last_error;

int32_t Fail(std::string message) {
  last_error = std::move(message);
  return -1;
}

void FailFromException(const char *where) {
  try {
    throw;
  } catch (const std::exception &error) {
    last_error = std::string(where) + ": " + error.what();
  } catch (...) {
    last_error = std::string(where) + ": unknown C++ exception";
  }
}

}  // namespace

struct ct_detector {
  explicit ct_detector(const cuda_tags::Camera &camera,
                       const cuda_tags::Distortion &distortion,
                       const cuda_tags::Options &options)
      : detector(camera, distortion, options) {}
  cuda_tags::Detector detector;
  const std::vector<cuda_tags::Detection> *detections = nullptr;
};

extern "C" {

uint32_t ct_abi_version(void) { return CT_ABI_VERSION; }

const char *ct_last_error(void) { return last_error.c_str(); }

int32_t ct_cuda_available(void) { return cuda_tags::CudaAvailable() ? 1 : 0; }

void ct_options_init(ct_options *options, uint32_t width, uint32_t height) {
  if (options == nullptr) return;
  const cuda_tags::Options defaults;
  options->width = width;
  options->height = height;
  options->threads = defaults.threads;
  options->decimate = defaults.decimate;
  options->min_white_black_diff = defaults.min_white_black_diff;
  options->max_hamming = defaults.max_hamming;
  options->refine_edges = defaults.refine_edges ? 1 : 0;
  options->decode_sharpening = defaults.decode_sharpening;
  options->tag_size_m = defaults.tag_size_m;
}

ct_detector *ct_create_with_options(const ct_camera *camera,
                                    const ct_options *options) {
  if (camera == nullptr || options == nullptr) {
    Fail("ct_create: null argument");
    return nullptr;
  }
  try {
    const cuda_tags::Camera intrinsics{camera->fx, camera->fy, camera->cx,
                                       camera->cy};
    const cuda_tags::Distortion distortion{camera->k1, camera->k2, camera->p1,
                                           camera->p2, camera->k3};
    cuda_tags::Options core;
    core.width = options->width;
    core.height = options->height;
    core.threads = options->threads;
    core.decimate = options->decimate;
    core.min_white_black_diff = options->min_white_black_diff;
    core.max_hamming = options->max_hamming;
    core.refine_edges = options->refine_edges != 0;
    core.decode_sharpening = options->decode_sharpening;
    core.tag_size_m = options->tag_size_m;
    return new ct_detector(intrinsics, distortion, core);
  } catch (...) {
    FailFromException("ct_create");
  }
  return nullptr;
}

ct_detector *ct_create(const ct_camera *camera, uint32_t width, uint32_t height,
                       int32_t threads, double tag_size_m) {
  ct_options options;
  ct_options_init(&options, width, height);
  options.threads = threads;
  options.tag_size_m = tag_size_m;
  return ct_create_with_options(camera, &options);
}

void ct_destroy(ct_detector *detector) {
  try {
    delete detector;
  } catch (...) {
    // Destructors do not throw; never let one escape regardless.
  }
}

int32_t ct_detect(ct_detector *detector, const uint8_t *pixels,
                  size_t length) {
  if (detector == nullptr || pixels == nullptr) {
    return Fail("ct_detect: null argument");
  }
  detector->detections = nullptr;
  try {
    detector->detections = &detector->detector.DetectGray(pixels, length);
    return static_cast<int32_t>(detector->detections->size());
  } catch (...) {
    FailFromException("ct_detect");
  }
  return -1;
}

int32_t ct_detect_image(ct_detector *detector, const uint8_t *pixels,
                        size_t length, size_t stride, int32_t format) {
  if (detector == nullptr || pixels == nullptr) {
    return Fail("ct_detect_image: null argument");
  }
  static constexpr cuda_tags::PixelFormat kFormats[] = {
      cuda_tags::PixelFormat::kGray, cuda_tags::PixelFormat::kBgr,
      cuda_tags::PixelFormat::kRgb, cuda_tags::PixelFormat::kBgra,
      cuda_tags::PixelFormat::kRgba};
  if (format < CT_FORMAT_GRAY || format > CT_FORMAT_RGBA) {
    return Fail("ct_detect_image: unknown pixel format " +
                std::to_string(format));
  }
  detector->detections = nullptr;
  try {
    const cuda_tags::Options &options = detector->detector.options();
    cuda_tags::ImageView view;
    view.data = pixels;
    view.size = length;
    view.width = options.width;
    view.height = options.height;
    view.stride = stride;
    view.format = kFormats[format];
    detector->detections = &detector->detector.Detect(view);
    return static_cast<int32_t>(detector->detections->size());
  } catch (...) {
    FailFromException("ct_detect_image");
  }
  return -1;
}

uint8_t *ct_input_buffer(ct_detector *detector, size_t *size) {
  if (size != nullptr) *size = 0;
  if (detector == nullptr) {
    Fail("ct_input_buffer: null argument");
    return nullptr;
  }
  uint8_t *buffer = detector->detector.InputBuffer();
  if (buffer == nullptr) {
    Fail("ct_input_buffer: this build has no GPU input buffer");
    return nullptr;
  }
  if (size != nullptr) *size = detector->detector.InputBufferSize();
  return buffer;
}

int32_t ct_set_color_conversion(ct_detector *detector, int32_t conversion) {
  if (detector == nullptr) return Fail("ct_set_color_conversion: null argument");
  if (conversion != CT_COLOR_WHILE_DETECTING &&
      conversion != CT_COLOR_BEFORE_DETECTING) {
    return Fail("ct_set_color_conversion: unknown value " +
                std::to_string(conversion));
  }
  detector->detector.SetColorConversion(
      conversion == CT_COLOR_BEFORE_DETECTING
          ? cuda_tags::ColorConversion::kBeforeDetecting
          : cuda_tags::ColorConversion::kWhileDetecting);
  return 0;
}

int32_t ct_detection_at(const ct_detector *detector, size_t index,
                        ct_detection *out) {
  if (detector == nullptr || out == nullptr ||
      detector->detections == nullptr ||
      index >= detector->detections->size()) {
    return Fail("ct_detection_at: index out of range");
  }
  const cuda_tags::Detection &tag = (*detector->detections)[index];
  *out = ct_detection{};
  out->id = tag.id;
  out->hamming = tag.hamming;
  out->decision_margin = tag.decision_margin;
  out->center[0] = tag.center[0];
  out->center[1] = tag.center[1];
  for (int corner = 0; corner < 4; ++corner) {
    out->corners[corner][0] = tag.corners[corner][0];
    out->corners[corner][1] = tag.corners[corner][1];
  }
  if (tag.pose) {
    out->pose_valid = 1;
    for (int k = 0; k < 9; ++k) out->rotation[k] = tag.pose->rotation[k];
    for (int k = 0; k < 3; ++k) out->translation[k] = tag.pose->translation[k];
  }
  return 0;
}

int32_t ct_detection_homography(const ct_detector *detector, size_t index,
                                double out[9]) {
  if (detector == nullptr || out == nullptr ||
      detector->detections == nullptr ||
      index >= detector->detections->size()) {
    return Fail("ct_detection_homography: index out of range");
  }
  const cuda_tags::Detection &tag = (*detector->detections)[index];
  for (int k = 0; k < 9; ++k) out[k] = tag.homography[k];
  return 0;
}

}  // extern "C"
