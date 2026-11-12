# Builds the AprilTag C library (with the two functions frc971's detector
# needs split out of apriltag.c) and, when CUDA is enabled, Team 971's GPU
# detector (971apriltag). Both are static, position-independent libraries so
# they can be linked into executables, Python modules and shared libraries.
#
# Neither library needs OpenCV or WPILib.
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin
endif()
if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")
endif()

project(971AprilTag LANGUAGES C CXX)

if(NOT DEFINED CUDATAGS_WITH_CUDA)
  set(CUDATAGS_WITH_CUDA ON)
endif()

# --- AprilTag -------------------------------------------------------------
set(BUILD_SHARED_LIBS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_PYTHON_WRAPPER OFF)
add_subdirectory(third_party/apriltag EXCLUDE_FROM_ALL)

# frc971's GPU detector calls quad_decode_index() and reconcile_detections(),
# which upstream AprilTag keeps inside apriltag.c. Build the apriltag library
# from apriltag_971/apriltag.c instead: upstream's file with only those two
# split out and exported. It is a copy, so it must match the submodule.
set(APRILTAG_UPSTREAM_C ${CMAKE_CURRENT_SOURCE_DIR}/third_party/apriltag/apriltag.c)
set(APRILTAG_UPSTREAM_SHA256 2384af6e0b83383cb4a873dc6e928de5a4eb81e62867c3f598381afdd81e0432)
file(SHA256 ${APRILTAG_UPSTREAM_C} APRILTAG_ACTUAL_SHA256)
if(NOT APRILTAG_ACTUAL_SHA256 STREQUAL APRILTAG_UPSTREAM_SHA256)
  message(FATAL_ERROR
    "third_party/apriltag/apriltag.c changed (submodule updated?). Re-apply the "
    "quad_decode_index/reconcile_detections split to a fresh copy in "
    "apriltag_971/apriltag.c, then update APRILTAG_UPSTREAM_SHA256.")
endif()
get_target_property(APRILTAG_SOURCES apriltag SOURCES)
list(FIND APRILTAG_SOURCES apriltag.c APRILTAG_C_INDEX)
if(APRILTAG_C_INDEX EQUAL -1)
  message(FATAL_ERROR "apriltag target no longer lists apriltag.c; update the source swap above")
endif()
list(REMOVE_AT APRILTAG_SOURCES ${APRILTAG_C_INDEX})
list(APPEND APRILTAG_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/apriltag_971/apriltag.c)
set_property(TARGET apriltag PROPERTY SOURCES ${APRILTAG_SOURCES})
set_property(TARGET apriltag PROPERTY POSITION_INDEPENDENT_CODE ON)
# "apriltag/apriltag.h", as frc971's sources include it.
target_include_directories(apriltag PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>)
if(NOT TARGET apriltag::apriltag)
  add_library(apriltag::apriltag ALIAS apriltag)
endif()

# --- frc971 GPU detector ---------------------------------------------------
if(CUDATAGS_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(
    971apriltag STATIC
    frc971/orin/971apriltag.cu
    frc971/orin/apriltag_detect.cu
    frc971/orin/labeling_allegretti_2019_BKE.cu
    frc971/orin/line_fit_filter.cu
    frc971/orin/points.cu
    frc971/orin/threshold.cu
    frc971/orin/cuda.cc
  )
  set_target_properties(971apriltag PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)
  target_compile_options(971apriltag PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wno-psabi>)
  target_include_directories(971apriltag PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(971apriltag PUBLIC apriltag CUDA::cudart_static)
endif()
