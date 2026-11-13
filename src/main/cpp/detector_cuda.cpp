// Detector backed by Team 971's GPU detector (frc971/orin).
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "cuda_tags/detector.hpp"
#include "frc971/orin/971apriltag.h"

namespace cuda_tags {
namespace {

// frc971's CHECK_CUDA ignores errors, so check the sticky per-thread error
// after each GPU call sequence instead.
void ThrowOnCudaError(const char *what) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": CUDA error: " +
                             cudaGetErrorString(error));
  }
}

// Byte offsets of R, G and B, and OpenCV's BT.601 weights (bit-exact with
// cv::cvtColor; see ct_pixel_layout.h).
ct_pixel_layout LayoutFor(PixelFormat format) {
  ct_pixel_layout layout{};
  layout.channels = Channels(format);
  const bool rgb_order =
      format == PixelFormat::kRgb || format == PixelFormat::kRgba;
  layout.offset[0] = rgb_order ? 0 : 2;
  layout.offset[1] = 1;
  layout.offset[2] = rgb_order ? 2 : 0;
  layout.weight[0] = 9798;
  layout.weight[1] = 19235;
  layout.weight[2] = 3735;
  return layout;
}

// A pinned host buffer the GPU reads and writes in place.
struct MappedBuffer {
  uint8_t *host = nullptr;
  uint8_t *device = nullptr;
  size_t size = 0;

  void Allocate(size_t bytes) {
    void *memory = nullptr;
    if (cudaHostAlloc(&memory, bytes, cudaHostAllocMapped) != cudaSuccess) {
      cudaGetLastError();
      throw std::runtime_error("could not allocate a pinned buffer");
    }
    host = static_cast<uint8_t *>(memory);
    size = bytes;
    void *mapped = nullptr;
    if (cudaHostGetDevicePointer(&mapped, memory, 0) != cudaSuccess) {
      cudaGetLastError();
      throw std::runtime_error("could not map a pinned buffer for the GPU");
    }
    device = static_cast<uint8_t *>(mapped);
  }
  bool Contains(const uint8_t *data, size_t bytes) const {
    return host != nullptr && data >= host && bytes <= size &&
           static_cast<size_t>(data - host) <= size - bytes;
  }
  ~MappedBuffer() {
    if (host != nullptr) cudaFreeHost(host);
  }
};

}  // namespace

bool CudaAvailable() { return true; }

struct Detector::Impl {
  Camera camera;
  Distortion distortion;
  Options options;
  apriltag_family_t *family = nullptr;
  apriltag_detector_t *tags = nullptr;
  frc971::apriltag::GpuDetector *gpu = nullptr;
  std::vector<uint8_t> gray;
  std::vector<Detection> detections;
  // Pinned host memory the GPU reads and writes in place (zero-copy).
  MappedBuffer input;      // width * height * 4: frames the caller writes
  MappedBuffer gray_host;  // width * height: gray frames the CPU decode reads
  ct_pixel_layout layout{};

  ~Impl() {
    // The GPU detector holds a pointer to tags, which holds the family.
    delete gpu;
    if (tags != nullptr) apriltag_detector_destroy(tags);  // and its pool
    if (family != nullptr) tag36h11_destroy(family);
  }
};

Detector::Detector(const Camera &camera, const Distortion &distortion,
                   const Options &options)
    : impl_(std::make_unique<Impl>()) {
  if (const std::string error = ValidateOptions(camera, distortion, options);
      !error.empty()) {
    throw std::invalid_argument(error);
  }
  int devices = 0;
  const cudaError_t device_error = cudaGetDeviceCount(&devices);
  if (device_error != cudaSuccess || devices == 0) {
    cudaGetLastError();  // clear it
    throw std::runtime_error(
        std::string("no usable CUDA device: ") +
        (device_error != cudaSuccess ? cudaGetErrorString(device_error)
                                     : "no devices found"));
  }
  ThrowOnCudaError("before creating the detector");

  Impl &impl = *impl_;
  impl.camera = camera;
  impl.distortion = distortion;
  impl.options = options;

  impl.family = tag36h11_create();
  impl.tags = apriltag_detector_create();
  if (impl.family == nullptr || impl.tags == nullptr) {
    throw std::runtime_error("could not allocate the AprilTag detector");
  }
  apriltag_detector_add_family_bits(impl.tags, impl.family,
                                    options.max_hamming);
  impl.tags->nthreads = options.threads;
  impl.tags->wp = workerpool_create(options.threads);
  if (impl.tags->wp == nullptr) {
    throw std::runtime_error("could not create the AprilTag worker pool");
  }
  impl.tags->quad_decimate = static_cast<float>(options.decimate);
  impl.tags->refine_edges = options.refine_edges;
  impl.tags->decode_sharpening = options.decode_sharpening;
  impl.tags->qtp.min_white_black_diff = options.min_white_black_diff;
  impl.tags->qtp.deglitch = false;  // unsupported by the GPU detector
  // Debug mode writes images to /tmp on every frame; never enable it here.
  impl.tags->debug = false;

  impl.gpu = new frc971::apriltag::GpuDetector(
      options.width, options.height, impl.tags,
      frc971::apriltag::CameraMatrix{camera.fx, camera.cx, camera.fy,
                                     camera.cy},
      frc971::apriltag::DistCoeffs{distortion.k1, distortion.k2,
                                   distortion.p1, distortion.p2,
                                   distortion.k3},
      static_cast<size_t>(options.decimate));
  ThrowOnCudaError("allocating the GPU detector");
  impl.gray.reserve(static_cast<size_t>(options.width) * options.height);
  const size_t pixels = static_cast<size_t>(options.width) * options.height;
  impl.input.Allocate(pixels * 4);
  impl.gray_host.Allocate(pixels);
}

Detector::~Detector() = default;

uint8_t *Detector::InputBuffer() { return impl_->input.host; }
size_t Detector::InputBufferSize() const { return impl_->input.size; }

void Detector::SetColorConversion(ColorConversion conversion) {
  if (conversion != ColorConversion::kWhileDetecting &&
      conversion != ColorConversion::kBeforeDetecting) {
    throw std::invalid_argument("unknown color conversion");
  }
  impl_->options.color_conversion = conversion;
}

const Options &Detector::options() const { return impl_->options; }
const Camera &Detector::camera() const { return impl_->camera; }
const Distortion &Detector::distortion() const { return impl_->distortion; }

const std::vector<Detection> &Detector::Detect(const ImageView &image) {
  Impl &impl = *impl_;
  if (const std::string error = ValidateImage(image); !error.empty()) {
    throw std::invalid_argument(error);
  }
  if (image.width != impl.options.width ||
      image.height != impl.options.height) {
    throw std::invalid_argument(
        "frame is " + std::to_string(image.width) + "x" +
        std::to_string(image.height) + " but the detector was built for " +
        std::to_string(impl.options.width) + "x" +
        std::to_string(impl.options.height));
  }
  impl.detections.clear();
  ThrowOnCudaError("before detection");
  if (impl.input.Contains(image.data, image.size)) {
    // The frame is in the input buffer: the GPU reads it in place.
    const uint8_t *device =
        impl.input.device + (image.data - impl.input.host);
    if (image.format == PixelFormat::kGray) {
      if (image.stride == image.width) {
        impl.gpu->DetectGrayMapped(device, image.data);
      } else {
        ConvertToGray(image, impl.gray_host.host);
        impl.gpu->DetectGrayMapped(impl.gray_host.device, impl.gray_host.host);
      }
    } else {
      impl.layout = LayoutFor(image.format);
      if (impl.options.color_conversion == ColorConversion::kBeforeDetecting) {
        impl.gpu->DetectColorConvertFirst(device, image.stride, impl.layout,
                                          impl.gray_host.device,
                                          impl.gray_host.host);
      } else {
        impl.gpu->DetectColorFused(device, image.data, image.stride,
                                   impl.layout);
      }
    }
  } else if (image.format == PixelFormat::kGray &&
             image.stride == image.width) {
    impl.gpu->DetectGrayHost(const_cast<uint8_t *>(image.data),
                             /*copy_back=*/false);
  } else {
    // Any other frame: converting (or packing) it on the CPU straight into
    // GPU-visible memory beats copying 3-4 bytes per pixel there for the GPU
    // to convert.
    ConvertToGray(image, impl.gray_host.host);
    impl.gpu->DetectGrayMapped(impl.gray_host.device, impl.gray_host.host);
  }
  ThrowOnCudaError("detection");
  return CollectDetections();
}

const std::vector<Detection> &Detector::DetectGray(const uint8_t *pixels,
                                                   size_t length) {
  Impl &impl = *impl_;
  if (pixels == nullptr) throw std::invalid_argument("image data is null");
  const size_t expected =
      static_cast<size_t>(impl.options.width) * impl.options.height;
  if (length != expected) {
    throw std::invalid_argument(
        "gray image has " + std::to_string(length) +
        " bytes but the detector was built for " +
        std::to_string(impl.options.width) + "x" +
        std::to_string(impl.options.height) + " (" + std::to_string(expected) +
        " bytes)");
  }
  ImageView view;
  view.data = pixels;
  view.size = length;
  view.width = impl.options.width;
  view.height = impl.options.height;
  view.stride = impl.options.width;
  view.format = PixelFormat::kGray;
  return Detect(view);
}

const std::vector<Detection> &Detector::CollectDetections() {
  Impl &impl = *impl_;
  const zarray_t *found = impl.gpu->Detections();
  impl.detections.reserve(zarray_size(found));
  for (int i = 0; i < zarray_size(found); ++i) {
    apriltag_detection_t *tag;
    zarray_get(found, i, &tag);
    Detection out;
    out.id = tag->id;
    out.hamming = tag->hamming;
    out.decision_margin = tag->decision_margin;
    out.center = {tag->c[0], tag->c[1]};
    for (int corner = 0; corner < 4; ++corner) {
      out.corners[corner] = {tag->p[corner][0], tag->p[corner][1]};
    }
    if (tag->H != nullptr && tag->H->nrows == 3 && tag->H->ncols == 3) {
      for (int k = 0; k < 9; ++k) out.homography[k] = tag->H->data[k];
    }
    if (impl.options.tag_size_m > 0.0) {
      out.pose = SolveTagPose(out.corners, impl.camera, impl.distortion,
                              impl.options.tag_size_m);
    }
    impl.detections.push_back(out);
  }
  return impl.detections;
}

}  // namespace cuda_tags
