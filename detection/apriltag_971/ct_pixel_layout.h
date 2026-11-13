// CudaTags: reads one gray value from a packed 8-bit gray or color pixel.
// Shared by the GPU kernels (frc971/orin/threshold.cu) and the CPU decode
// (apriltag.c, frc971/orin/apriltag_detect.cu) so that converting a frame
// before detection and converting pixels as the detector reads them give
// bit-identical gray values.
#pragma once

#include <stdint.h>

#ifdef __CUDACC__
#define CT_HOST_DEVICE __host__ __device__
#else
#define CT_HOST_DEVICE
#endif

#define CT_GRAY_SHIFT 15

// A color layout: bytes per pixel, byte offsets of R, G and B within a pixel,
// and their weights in 1/2^15 (OpenCV's BT.601 weights are 9798, 19235,
// 3735, bit-exact with its cvtColor; {0, 1 << 15, 0} reads the green byte
// alone). NULL means gray.
typedef struct {
  int channels;
  int offset[3];
  int weight[3];
} ct_pixel_layout;

// Gray value of pixel x of a row starting at `row`.
static inline CT_HOST_DEVICE int ct_pixel_gray(const uint8_t *row, int x,
                                               const ct_pixel_layout *layout) {
  if (layout == 0) return row[x];
  const uint8_t *pixel = row + x * layout->channels;
  return (pixel[layout->offset[0]] * layout->weight[0] +
          pixel[layout->offset[1]] * layout->weight[1] +
          pixel[layout->offset[2]] * layout->weight[2] +
          (1 << (CT_GRAY_SHIFT - 1))) >>
         CT_GRAY_SHIFT;
}
