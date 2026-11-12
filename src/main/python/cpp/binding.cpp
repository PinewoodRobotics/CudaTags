#include "cuda_tags.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>
#include <climits>

using namespace frc971::apriltag;
namespace py = pybind11;

PYBIND11_MODULE(cuda_tags, m) {
  py::class_<CameraMatrix>(m, "CameraMatrix")
      .def(py::init<>())
      .def(py::init<float, float, float, float>(), py::arg("fx"), py::arg("cx"),
           py::arg("fy"), py::arg("cy")) // parameter constructor
      .def_readwrite("fx", &CameraMatrix::fx)
      .def_readwrite("cx", &CameraMatrix::cx)
      .def_readwrite("fy", &CameraMatrix::fy)
      .def_readwrite("cy", &CameraMatrix::cy);

  py::class_<DistCoeffs>(m, "DistCoeffs")
      .def(py::init<>())
      .def(py::init<float, float, float, float, float>(), py::arg("k1"),
           py::arg("k2"), py::arg("p1"), py::arg("p2"),
           py::arg("k3")) // parameter constructor
      .def_readwrite("k1", &DistCoeffs::k1)
      .def_readwrite("k2", &DistCoeffs::k2)
      .def_readwrite("p1", &DistCoeffs::p1)
      .def_readwrite("p2", &DistCoeffs::p2)
      .def_readwrite("k3", &DistCoeffs::k3);

  py::enum_<TagType>(m, "TagType")
      .value("tag36h11", TagType::tag36h11)
      .export_values();

  py::class_<DetectionResult>(m, "DetectionResult")
      .def(py::init<>())
      .def_readwrite("id", &DetectionResult::id)
      .def_readwrite("hamming", &DetectionResult::hamming)
      .def_readwrite("decision_margin", &DetectionResult::decision_margin)
      .def_property_readonly("homography",
                             [](const DetectionResult &self) {
                               py::list result;
                               for (const auto &row : self.homography) {
                                 py::list row_list;
                                 for (double val : row) {
                                   row_list.append(val);
                                 }
                                 result.append(row_list);
                               }
                               return result;
                             })
      .def_property_readonly("center",
                             [](const DetectionResult &self) {
                               py::list result;
                               for (double val : self.center) {
                                 result.append(val);
                               }
                               return result;
                             })
      .def_property_readonly("corners", [](const DetectionResult &self) {
        py::list result;
        for (const auto &corner : self.corners) {
          py::list corner_list;
          for (double val : corner) {
            corner_list.append(val);
          }
          result.append(corner_list);
        }
        return result;
      });

  py::class_<CudaTagsWrapper>(m, "CudaTagsWrapper", py::dynamic_attr())
      .def(py::init<TagType, CameraMatrix, DistCoeffs, int, int, int>(),
           py::arg("tagType"), py::arg("camMatrix"), py::arg("camDistCoeffs"),
           py::arg("nthreads"), py::arg("imgWidth"), py::arg("imgHeight"))
      .def(
          "process",
          [](CudaTagsWrapper &self, py::array_t<uint8_t, py::array::c_style> arr) {
            auto buf = arr.request();
            if (arr.ndim() != 2 && arr.ndim() != 3) {
              throw std::runtime_error("Array must have 2 or 3 dimensions");
            }
            if (arr.shape(0) <= 0 || arr.shape(1) <= 0 ||
                arr.shape(0) > INT_MAX || arr.shape(1) > INT_MAX)
              throw std::invalid_argument("Invalid image dimensions");
            int height = arr.shape(0);
            int width = arr.shape(1);
            int channels = arr.ndim() == 3 ? arr.shape(2) : 1;
            if (channels != 1 && channels != 3 && channels != 4)
              throw std::invalid_argument("Expected 1, 3, or 4 channels");
            uint8_t *data = static_cast<uint8_t *>(buf.ptr);
            auto results = self.process(data, width, height, channels);
            return results;
          },
          py::arg("image").noconvert(), py::return_value_policy::move,
          "Process an image array and return detection results.\n\n"
          ":param image: numpy.ndarray of uint8, shape (height, width) or "
          "(height, width, channels)\n"
          ":return: List[DetectionResult]");
}