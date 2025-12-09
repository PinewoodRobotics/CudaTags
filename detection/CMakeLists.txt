cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")
endif()

set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")

# set(CMAKE_BUILD_TYPE DEBUG)

project(971AprilTag LANGUAGES CXX CUDA)
find_package(OpenCV REQUIRED)

add_subdirectory(third_party/apriltag)
set(WITH_WPILIB OFF CACHE BOOL "Build hal, wpilibc/j, and developerRobot (needs OpenCV)" FORCE)
set(WITH_SIMULATION_MODULES OFF CACHE BOOL "Build simulation modules" FORCE)
set(WITH_CSCORE OFF CACHE BOOL "Build cscore (needs OpenCV)" FORCE)
set(WITH_NTCORE OFF CACHE BOOL "Build ntcore" FORCE)
set(WITH_WPIMATH OFF CACHE BOOL "Build wpimath" FORCE)
set(WITH_GUI OFF CACHE BOOL "Build GUI items" FORCE)
set(WITH_TESTS OFF CACHE BOOL "Build unit tests" FORCE)
set(WITH_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(WITH_BENCHMARK OFF CACHE BOOL "Build the benchmark project" FORCE)
set(WITH_PROTOBUF OFF CACHE BOOL "Build protobuf support" FORCE)
set(WITH_JAVA OFF CACHE BOOL "Include Java and JNI in the build" FORCE)
add_subdirectory(third_party/allwpilib)

add_library(apriltag::apriltag ALIAS apriltag)

add_compile_options(-Wno-psabi)

add_library(
  971apriltag SHARED
  frc971/orin/971apriltag.cu
  frc971/orin/apriltag_detect.cu
  frc971/orin/labeling_allegretti_2019_BKE.cu
  frc971/orin/line_fit_filter.cu
  frc971/orin/points.cu
  frc971/orin/threshold.cu
  frc971/orin/cuda.cc
)

target_include_directories(
  971apriltag PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/cuda/include
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/apriltag
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/
)

target_link_libraries(
  971apriltag PRIVATE
  apriltag::apriltag
  wpiutil
)

