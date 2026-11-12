# ctest suite. Without CUDA: core/C ABI unit tests and the golden image
# through AprilTag's CPU detector. With CUDA (on a Jetson): also the golden
# image through the GPU detector.
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)

add_executable(core_cpu_test core_cpu_test.cpp)
target_include_directories(core_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(core_cpu_test PRIVATE cuda_tags_core opencv_core opencv_calib3d)
add_test(NAME core_cpu COMMAND core_cpu_test)

set(GOLDEN_IMAGE ${CMAKE_CURRENT_SOURCE_DIR}/golden/tags_640x480.pgm)
set(GOLDEN_SCENE ${CMAKE_CURRENT_SOURCE_DIR}/golden/tags_640x480.txt)

add_executable(golden_test golden_test.cpp)
target_include_directories(golden_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(golden_test PRIVATE cuda_tags_core apriltag opencv_core)
add_test(NAME golden_cpu COMMAND golden_test ${GOLDEN_IMAGE} ${GOLDEN_SCENE} cpu)
if(CUDATAGS_WITH_CUDA)
  add_test(NAME golden_gpu COMMAND golden_test ${GOLDEN_IMAGE} ${GOLDEN_SCENE} gpu)
  set_tests_properties(golden_gpu PROPERTIES LABELS gpu)
endif()

# Regenerates the golden image and scene (not run by ctest):
#   cmake --build build --target generate_golden
#   build/tests/generate_golden tests/golden/tags_640x480.pgm tests/golden/tags_640x480.txt
add_executable(generate_golden EXCLUDE_FROM_ALL golden/generate_golden.cpp)
target_include_directories(generate_golden PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(generate_golden PRIVATE apriltag opencv_core)
