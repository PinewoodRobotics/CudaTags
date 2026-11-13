# The C++ core (cuda_tags/detector.hpp) and C ABI (cuda_tags/cuda_tags.h)
# that every binding shares.
#
#   cuda_tags_core  static, position-independent: core + C ABI. Linked by the
#                   Rust crate, the Python module and the JNI library. Its
#                   dependencies (971apriltag, apriltag, OpenCV, cudart) must
#                   be linked after it.
#   cuda_tags_c     libcuda_tags.so: the C ABI as a shared library that
#                   exports only ct_* symbols (CUDATAGS_BUILD_C_LIBRARY).

set(CUDATAGS_CORE_SOURCES common.cpp c_api.cpp)
if(CUDATAGS_WITH_CUDA)
  list(APPEND CUDATAGS_CORE_SOURCES detector_cuda.cpp)
else()
  list(APPEND CUDATAGS_CORE_SOURCES detector_stub.cpp)
endif()

add_library(cuda_tags_core STATIC ${CUDATAGS_CORE_SOURCES})
set_target_properties(cuda_tags_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(cuda_tags_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(cuda_tags_core PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cuda_tags_core PRIVATE ${CUDATAGS_OPENCV_LIBS})
if(CUDATAGS_WITH_CUDA)
  target_link_libraries(cuda_tags_core PRIVATE 971apriltag apriltag)
endif()

if(CUDATAGS_BUILD_C_LIBRARY)
  # Same sources as the core, compiled again with hidden visibility so only
  # the CT_API functions are exported.
  add_library(cuda_tags_c SHARED ${CUDATAGS_CORE_SOURCES})
  target_include_directories(cuda_tags_c PRIVATE include ${OpenCV_INCLUDE_DIRS})
  set_target_properties(cuda_tags_c PROPERTIES
    OUTPUT_NAME cuda_tags
    SOVERSION 1
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN")
  target_link_libraries(cuda_tags_c PRIVATE ${CUDATAGS_OPENCV_LIBS})
  if(CUDATAGS_WITH_CUDA)
    target_link_libraries(cuda_tags_c PRIVATE 971apriltag apriltag)
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Keep apriltag, frc971 and CUDA runtime symbols private, and the C++
    # standard library's template instantiations (which hidden visibility
    # does not cover) too: export ct_* only.
    target_link_options(cuda_tags_c PRIVATE "LINKER:--exclude-libs,ALL"
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cuda_tags.map")
    set_property(TARGET cuda_tags_c APPEND PROPERTY
      LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_tags.map)
  endif()
  include(GNUInstallDirs)
  install(TARGETS cuda_tags_c
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES include/cuda_tags/cuda_tags.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuda_tags)
endif()
