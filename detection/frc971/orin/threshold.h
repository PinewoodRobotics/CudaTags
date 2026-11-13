#pragma once

#include <stdint.h>

#include "apriltag_971/ct_pixel_layout.h"
#include "frc971/orin/cuda.h"

namespace frc971::apriltag {

// Decimates, and thresholds an image on the provided
// stream.
void CudaDecimate(
    const uint8_t *gray_image, uint8_t *decimated_image,
    uint8_t *unfiltered_minmax_image, uint8_t *minmax_image,
    uint8_t *thresholded_image, size_t width, size_t height,
    size_t min_white_black_diff, CudaStream *stream);

// thresholds an image on the provided
// stream.
void CudaNoDecimate(
    const uint8_t *gray_image, uint8_t *decimated_image,
    uint8_t *unfiltered_minmax_image, uint8_t *minmax_image,
    uint8_t *thresholded_image, size_t width, size_t height,
    size_t min_white_black_diff, CudaStream *stream);

// Converts to grayscale, decimates, and thresholds an image on the provided
// stream.
void CudaToGreyscaleAndDecimate(
    const uint8_t *color_image, uint8_t *gray_image, uint8_t *decimated_image,
    uint8_t *unfiltered_minmax_image, uint8_t *minmax_image,
    uint8_t *thresholded_image, size_t width, size_t height,
    size_t min_white_black_diff, CudaStream *stream);

// CudaTags: converts a packed color frame (stride in bytes) to gray,
// writing gray and, when not null, gray_copy.
void CudaColorToGray(const uint8_t *color, size_t stride,
                     const ct_pixel_layout &layout, uint8_t *gray,
                     uint8_t *gray_copy, size_t width, size_t height,
                     CudaStream *stream);

// CudaTags: decimates (factor 1 or 2) and thresholds a packed color frame,
// converting only the pixels the decimation keeps.
void CudaColorDecimate(const uint8_t *color, size_t stride,
                       const ct_pixel_layout &layout, size_t factor,
                       uint8_t *decimated_image,
                       uint8_t *unfiltered_minmax_image, uint8_t *minmax_image,
                       uint8_t *thresholded_image, size_t width, size_t height,
                       size_t min_white_black_diff, CudaStream *stream);

}  // namespace frc971::apriltag


