cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(CudaTagsJavaJNI)

find_package(JNI REQUIRED)
find_package(Java REQUIRED)
include(UseJava)
find_package(OpenCV REQUIRED)
find_package(wpilib REQUIRED)

add_subdirectory(detection)
add_library(
    CudaTagsJavaJNI SHARED
    src/main/java/native/GpuDetectorJNI.cc
)

target_include_directories(
  CudaTagsJavaJNI PRIVATE
  ${JNI_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/detection
  /usr/local/cuda/include
)

target_link_libraries(
  CudaTagsJavaJNI PRIVATE
  971apriltag
  ${CMAKE_SOURCE_DIR}/detection/third_party/apriltag/build/libapriltag.a
  ${JNI_LIBRARIES}
  ${OpenCV_LIBS}
  wpiutil
)

##################################

project(cuda_tags_python)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(cuda_tags
    src/main/python/cpp/binding.cpp
    src/main/python/cpp/cuda_tags.cpp
)

target_include_directories(
  cuda_tags PRIVATE
  ${JNI_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/detection
  /usr/local/cuda/include
)

target_link_libraries(
  cuda_tags PRIVATE
  971apriltag
  ${CMAKE_SOURCE_DIR}/detection/third_party/apriltag/build/libapriltag.a
  ${JNI_LIBRARIES}
  ${OpenCV_LIBS}
  wpiutil
)
