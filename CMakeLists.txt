cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Global suppression of restrict warning for GCC 12
add_compile_options(-Wno-error=restrict)

project(CudaTags LANGUAGES CXX CUDA)

set(WITH_GUI OFF CACHE BOOL "" FORCE)
set(WITH_JAVA OFF CACHE BOOL "" FORCE)
set(WITH_SIMULATION_MODULES OFF CACHE BOOL "" FORCE)
set(WITH_TESTS OFF CACHE BOOL "" FORCE)

set(_default_build_java OFF)
if(SKBUILD)
  set(_default_build_java OFF)
endif()
option(CUDATAGS_BUILD_JAVA "Build the Java JNI bindings" ${_default_build_java})

set(_default_build_python OFF)
if(SKBUILD)
  set(_default_build_python ON)
endif()
option(CUDATAGS_BUILD_PYTHON "Build the Python bindings" ${_default_build_python})

find_package(OpenCV REQUIRED)

add_subdirectory(detection)

if(CUDATAGS_BUILD_JAVA)
  find_package(JNI REQUIRED)
  find_package(Java REQUIRED)
  include(UseJava)

  add_library(
      CudaTagsJavaJNI SHARED
      src/main/java/native/GpuDetectorJNI.cc
  )

  target_include_directories(
    CudaTagsJavaJNI PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/detection
    ${CMAKE_SOURCE_DIR}/detection/third_party
    /usr/local/cuda/include
  )

  target_link_libraries(
    CudaTagsJavaJNI PRIVATE
    971apriltag
    apriltag::apriltag
    ${JNI_LIBRARIES}
    ${OpenCV_LIBS}
    wpiutil
  )
endif()

if(CUDATAGS_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(cuda_tags
      src/main/python/cpp/binding.cpp
      src/main/python/cpp/cuda_tags.cpp
  )

  target_include_directories(
    cuda_tags PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/detection
    ${CMAKE_SOURCE_DIR}/detection/third_party
    /usr/local/cuda/include
  )

  target_link_libraries(
    cuda_tags PRIVATE
    971apriltag
    apriltag::apriltag
    ${OpenCV_LIBS}
    wpiutil
  )

  if(APPLE)
    set(_cudatags_rpath "@loader_path")
  else()
    set(_cudatags_rpath "\$ORIGIN")
  endif()

  set_target_properties(cuda_tags PROPERTIES
    OUTPUT_NAME "cuda_tags"
    BUILD_RPATH "${_cudatags_rpath}"
    INSTALL_RPATH "${_cudatags_rpath}"
  )

  install(TARGETS cuda_tags
          LIBRARY DESTINATION .
          RUNTIME DESTINATION .
          ARCHIVE DESTINATION .)

  set_target_properties(971apriltag PROPERTIES
    BUILD_RPATH "${_cudatags_rpath}"
    INSTALL_RPATH "${_cudatags_rpath}"
  )

  set_target_properties(apriltag PROPERTIES
    BUILD_RPATH "${_cudatags_rpath}"
    INSTALL_RPATH "${_cudatags_rpath}"
  )

  install(TARGETS 971apriltag
          LIBRARY DESTINATION .
          RUNTIME DESTINATION .
          ARCHIVE DESTINATION .)

  install(TARGETS apriltag
          LIBRARY DESTINATION .
          RUNTIME DESTINATION .
          ARCHIVE DESTINATION .)
endif()
