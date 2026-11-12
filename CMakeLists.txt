cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Jetson Orin (JetPack 6) by default; override with -DCMAKE_CUDA_ARCHITECTURES.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()
# CMake < 3.25 does not know nvcc's C++20 flag.
if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")
endif()

project(CudaTags VERSION 0.2.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA is optional: without it the core, C ABI and bindings build with a
# detector that reports "built without CUDA", and the CPU tests still run.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(_cudatags_default_cuda ON)
else()
  set(_cudatags_default_cuda OFF)
endif()
option(CUDATAGS_WITH_CUDA "Build the GPU detector (needs the CUDA toolkit)" ${_cudatags_default_cuda})

set(_cudatags_default_python OFF)
if(SKBUILD)
  set(_cudatags_default_python ON)
endif()
option(CUDATAGS_BUILD_PYTHON "Build the Python module" ${_cudatags_default_python})
option(CUDATAGS_BUILD_JAVA "Build the Java JNI library" OFF)
option(CUDATAGS_BUILD_C_LIBRARY "Build libcuda_tags.so (the C ABI)" ON)
if(PROJECT_IS_TOP_LEVEL AND NOT SKBUILD)
  set(_cudatags_default_tests ON)
else()
  set(_cudatags_default_tests OFF)
endif()
option(CUDATAGS_BUILD_TESTS "Build the tests (ctest)" ${_cudatags_default_tests})

if(CUDATAGS_WITH_CUDA)
  enable_language(CUDA)
  message(STATUS "CudaTags: CUDA detector enabled (architectures ${CMAKE_CUDA_ARCHITECTURES})")
else()
  message(STATUS "CudaTags: building WITHOUT CUDA; Detector creation will fail at runtime")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
set(CUDATAGS_OPENCV_LIBS opencv_core opencv_imgproc opencv_calib3d)

# apriltag (always, the CPU tests use it) and 971apriltag (with CUDA).
add_subdirectory(detection)
add_subdirectory(src/main/cpp)

if(CUDATAGS_BUILD_PYTHON)
  # pybind11 finds Python itself (honouring PYTHON_EXECUTABLE, as
  # scikit-build passes it).
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(cuda_tags src/main/python/cpp/binding.cpp)
  target_link_libraries(cuda_tags PRIVATE cuda_tags_core)
  set_target_properties(cuda_tags PROPERTIES OUTPUT_NAME "cuda_tags")
  install(TARGETS cuda_tags
          LIBRARY DESTINATION .
          RUNTIME DESTINATION .)
endif()

if(CUDATAGS_BUILD_JAVA)
  find_package(JNI REQUIRED)
  find_package(Java REQUIRED)
  include(UseJava)
  add_library(cuda_tags_jni SHARED src/main/java/native/CudaTagsJNI.cc)
  target_include_directories(cuda_tags_jni PRIVATE ${JNI_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(cuda_tags_jni PRIVATE cuda_tags_core)
  set_target_properties(cuda_tags_jni PROPERTIES CXX_VISIBILITY_PRESET hidden)
  add_jar(cuda_tags_java
    SOURCES src/main/java/frc4765/cuda/tags/CudaTags.java
    OUTPUT_NAME cuda-tags)
endif()

if(CUDATAGS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
