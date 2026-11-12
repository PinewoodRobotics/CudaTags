// Python module over the shared C++ core (src/main/cpp).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_tags/detector.hpp"

namespace py = pybind11;

namespace {

// Python-facing parameter types. Field order matches the 0.1 API:
// CameraMatrix(fx, cx, fy, cy).
struct CameraMatrix {
  double fx = 0.0;
  double cx = 0.0;
  double fy = 0.0;
  double cy = 0.0;
};

struct DistCoeffs {
  double k1 = 0.0;
  double k2 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double k3 = 0.0;
};

enum class TagType { tag36h11 = 1 };

cuda_tags::PixelFormat FormatFor(int channels, const std::string &order) {
  const bool rgb = order == "rgb";
  if (!rgb && order != "bgr") {
    throw py::value_error("color_order must be 'bgr' or 'rgb'");
  }
  switch (channels) {
    case 1:
      return cuda_tags::PixelFormat::kGray;
    case 3:
      return rgb ? cuda_tags::PixelFormat::kRgb : cuda_tags::PixelFormat::kBgr;
    case 4:
      return rgb ? cuda_tags::PixelFormat::kRgba
                 : cuda_tags::PixelFormat::kBgra;
  }
  throw py::value_error("image must have 1, 3 or 4 channels, got " +
                        std::to_string(channels));
}

// Describes a uint8 array as an ImageView, without copying when its rows
// are padded but its pixels are packed; otherwise through a contiguous
// copy held in `holder`. Needs the GIL.
cuda_tags::ImageView ViewOf(const py::array &input, const std::string &order,
                            py::array *holder) {
  if (!input.dtype().is(py::dtype::of<uint8_t>())) {
    throw py::type_error("image must be a numpy array of dtype uint8, got " +
                         std::string(py::str(input.dtype())));
  }
  if (input.ndim() != 2 && input.ndim() != 3) {
    throw py::value_error(
        "image must have shape (height, width) or (height, width, channels)");
  }
  const py::ssize_t height = input.shape(0);
  const py::ssize_t width = input.shape(1);
  const py::ssize_t channels = input.ndim() == 3 ? input.shape(2) : 1;
  if (height <= 0 || width <= 0) throw py::value_error("image is empty");
  if (height > (py::ssize_t{1} << 24) || width > (py::ssize_t{1} << 24)) {
    throw py::value_error("image is too large");
  }
  const auto format = FormatFor(static_cast<int>(channels), order);

  const bool packed_pixels =
      input.strides(1) == channels &&
      (input.ndim() == 2 || input.strides(2) == 1) &&
      input.strides(0) >= width * channels;
  py::array array = input;
  if (!packed_pixels) {
    // Negative, transposed or sliced strides: copy into C order.
    array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::
        ensure(input);
    if (!array) throw py::error_already_set();
  }
  *holder = array;
  cuda_tags::ImageView view;
  view.data = static_cast<const uint8_t *>(array.data());
  view.width = static_cast<uint32_t>(width);
  view.height = static_cast<uint32_t>(height);
  view.stride = static_cast<size_t>(array.strides(0));
  view.size = static_cast<size_t>(height - 1) * view.stride +
              static_cast<size_t>(width * channels);
  view.format = format;
  return view;
}

py::list ToList(const double *values, int count) {
  py::list result;
  for (int i = 0; i < count; ++i) result.append(values[i]);
  return result;
}

py::list ToRows(const double *values, int rows, int columns) {
  py::list result;
  for (int r = 0; r < rows; ++r) result.append(ToList(values + r * columns, columns));
  return result;
}

class PyDetector {
 public:
  PyDetector(TagType tag_type, const CameraMatrix &camera,
             const DistCoeffs &distortion, int threads, int width, int height,
             double tag_size_m, int decimate, int min_white_black_diff,
             int max_hamming, bool refine_edges, double decode_sharpening) {
    if (tag_type != TagType::tag36h11) {
      throw py::value_error("only tag36h11 is supported");
    }
    if (width <= 0 || height <= 0) {
      throw py::value_error("imgWidth and imgHeight must be positive");
    }
    cuda_tags::Options options;
    options.width = static_cast<uint32_t>(width);
    options.height = static_cast<uint32_t>(height);
    options.threads = threads;
    options.decimate = decimate;
    options.min_white_black_diff = min_white_black_diff;
    options.max_hamming = max_hamming;
    options.refine_edges = refine_edges;
    options.decode_sharpening = decode_sharpening;
    options.tag_size_m = tag_size_m;
    detector_ = std::make_unique<cuda_tags::Detector>(
        cuda_tags::Camera{camera.fx, camera.fy, camera.cx, camera.cy},
        cuda_tags::Distortion{distortion.k1, distortion.k2, distortion.p1,
                              distortion.p2, distortion.k3},
        options);
  }

  std::vector<cuda_tags::Detection> Process(const py::array &image,
                                            const std::string &order) {
    py::array holder;
    const cuda_tags::ImageView view = ViewOf(image, order, &holder);
    // Detection never touches Python objects (holder keeps the buffer
    // alive), so other Python threads run meanwhile.
    py::gil_scoped_release release;
    return detector_->Detect(view);
  }

  uint32_t width() const { return detector_->options().width; }
  uint32_t height() const { return detector_->options().height; }
  double tag_size_m() const { return detector_->options().tag_size_m; }

 private:
  std::unique_ptr<cuda_tags::Detector> detector_;
};

}  // namespace

PYBIND11_MODULE(cuda_tags, m) {
  m.doc() = "CUDA-accelerated tag36h11 AprilTag detection (CudaTags).";

  // pybind11 raises ValueError for std::invalid_argument and RuntimeError
  // for std::runtime_error.

  py::class_<CameraMatrix>(m, "CameraMatrix")
      .def(py::init<>())
      .def(py::init<double, double, double, double>(), py::arg("fx"),
           py::arg("cx"), py::arg("fy"), py::arg("cy"))
      .def_readwrite("fx", &CameraMatrix::fx)
      .def_readwrite("cx", &CameraMatrix::cx)
      .def_readwrite("fy", &CameraMatrix::fy)
      .def_readwrite("cy", &CameraMatrix::cy)
      .def("__repr__", [](const CameraMatrix &c) {
        return "CameraMatrix(fx=" + std::to_string(c.fx) +
               ", cx=" + std::to_string(c.cx) + ", fy=" + std::to_string(c.fy) +
               ", cy=" + std::to_string(c.cy) + ")";
      });

  py::class_<DistCoeffs>(m, "DistCoeffs")
      .def(py::init<>())
      .def(py::init<double, double, double, double, double>(), py::arg("k1"),
           py::arg("k2"), py::arg("p1"), py::arg("p2"), py::arg("k3"))
      .def_readwrite("k1", &DistCoeffs::k1)
      .def_readwrite("k2", &DistCoeffs::k2)
      .def_readwrite("p1", &DistCoeffs::p1)
      .def_readwrite("p2", &DistCoeffs::p2)
      .def_readwrite("k3", &DistCoeffs::k3);

  py::enum_<TagType>(m, "TagType")
      .value("tag36h11", TagType::tag36h11)
      .export_values();

  py::class_<cuda_tags::Pose>(m, "Pose")
      .def_property_readonly("rotation",
                             [](const cuda_tags::Pose &self) {
                               return ToRows(self.rotation.data(), 3, 3);
                             })
      .def_property_readonly("translation", [](const cuda_tags::Pose &self) {
        return ToList(self.translation.data(), 3);
      });

  py::class_<cuda_tags::Detection>(m, "DetectionResult")
      .def(py::init<>())
      .def_readonly("id", &cuda_tags::Detection::id)
      .def_readonly("hamming", &cuda_tags::Detection::hamming)
      .def_readonly("decision_margin", &cuda_tags::Detection::decision_margin)
      .def_property_readonly("homography",
                             [](const cuda_tags::Detection &self) {
                               return ToRows(self.homography.data(), 3, 3);
                             })
      .def_property_readonly("center",
                             [](const cuda_tags::Detection &self) {
                               return ToList(self.center.data(), 2);
                             })
      .def_property_readonly("corners",
                             [](const cuda_tags::Detection &self) {
                               return ToRows(self.corners[0].data(), 4, 2);
                             })
      .def_readonly("pose", &cuda_tags::Detection::pose);

  py::class_<PyDetector>(m, "CudaTagsWrapper")
      .def(py::init<TagType, CameraMatrix, DistCoeffs, int, int, int, double,
                    int, int, int, bool, double>(),
           py::arg("tagType"), py::arg("camMatrix"), py::arg("camDistCoeffs"),
           py::arg("nthreads"), py::arg("imgWidth"), py::arg("imgHeight"),
           py::kw_only(), py::arg("tag_size_m") = 0.0,
           py::arg("decimate") = 2, py::arg("min_white_black_diff") = 5,
           py::arg("max_hamming") = 1, py::arg("refine_edges") = true,
           py::arg("decode_sharpening") = 0.25)
      .def_property_readonly("width", &PyDetector::width)
      .def_property_readonly("height", &PyDetector::height)
      .def_property_readonly("tag_size_m", &PyDetector::tag_size_m)
      .def("process", &PyDetector::Process, py::arg("image"),
           py::arg("color_order") = "bgr",
           "Detect tags in a uint8 image of shape (height, width) or "
           "(height, width, 3|4). The size must match the detector's.");

  m.def(
      "to_gray",
      [](const py::array &image, const std::string &order) {
        py::array holder;
        const cuda_tags::ImageView view = ViewOf(image, order, &holder);
        py::array_t<uint8_t> out(
            {static_cast<py::ssize_t>(view.height),
             static_cast<py::ssize_t>(view.width)});
        std::vector<uint8_t> gray;
        cuda_tags::ConvertToGray(view, &gray);
        std::copy(gray.begin(), gray.end(), out.mutable_data());
        return out;
      },
      py::arg("image"), py::arg("color_order") = "bgr",
      "Convert an image to the contiguous gray image the detector sees.");

  m.def("cuda_available", &cuda_tags::CudaAvailable,
        "True when this build contains the CUDA detector.");
}
