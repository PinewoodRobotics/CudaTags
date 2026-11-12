#pragma once
/* CudaTags C ABI: Team 971's GPU AprilTag detector (tag36h11) behind a
 * stable C interface, for Rust and other FFI users.
 *
 * ABI version 1. ct_abi_version() returns CT_ABI_VERSION; a loader should
 * refuse a library whose version differs. Version 1 is the interface of the
 * drone project's cuda_tags_ffi shim, plus the functions marked "added in
 * version 1" below, which only add to it.
 *
 * Threading: a detector must only be used from the thread that created it,
 * and calls on one detector must not overlap. Separate detectors may run on
 * separate threads. Each owns its own CUDA stream and decode worker pool.
 *
 * Errors: no C++ exception crosses this interface. Failing calls return
 * NULL or -1 and record a message for ct_last_error(), which is
 * thread-local and lives until the next failing call on the same thread.
 *
 * Buffers passed in are borrowed for the duration of the call only. */
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define CT_API __declspec(dllexport)
#else
#define CT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CT_ABI_VERSION 1u

typedef struct ct_detector ct_detector;

/* Pinhole intrinsics (pixels) at the detector's image size, and OpenCV
 * (Brown-Conrady) distortion. All-zero distortion is an ideal pinhole. */
typedef struct {
  double fx, fy, cx, cy;
  double k1, k2, p1, p2, k3;
} ct_camera;

/* One tag. Corners are pixels in the input image (the detector uses the
 * distortion model to refine edges but reports corners where they appear in
 * the image) in AprilTag's order: bottom-left, bottom-right, top-right,
 * top-left of an upright tag. Like AprilTag, center and corners put the
 * center of pixel (x, y) at (x + 0.5, y + 0.5); OpenCV uses (x, y).
 * When pose_valid is nonzero, rotation (row-major 3x3) and translation
 * (meters) are the camera-from-tag pose from OpenCV's SOLVEPNP_IPPE_SQUARE
 * using the camera's intrinsics and distortion (corners shifted to OpenCV's
 * pixel convention first), with the tag in AprilTag's frame (x right, y
 * down, z into the tag). */
typedef struct {
  int32_t id;
  int32_t hamming;
  double decision_margin;
  double center[2];
  double corners[4][2];
  int32_t pose_valid;
  double rotation[9];
  double translation[3];
} ct_detection;

/* Detector tunables (added in version 1). Initialize with
 * ct_options_init(), then change fields. */
typedef struct {
  uint32_t width;  /* multiple of 8; width * height < 2^22 */
  uint32_t height; /* multiple of 8 */
  int32_t threads; /* CPU decode threads, 1..64 (default 1) */
  int32_t decimate; /* quad detection decimation, 1 or 2 (default 2) */
  int32_t min_white_black_diff; /* 0..255 (default 5) */
  int32_t max_hamming;          /* bit errors corrected, 0..2 (default 1) */
  int32_t refine_edges;         /* nonzero to refine edges (default 1) */
  double decode_sharpening;     /* >= 0 (default 0.25) */
  double tag_size_m; /* black border edge length; 0 disables pose (default) */
} ct_options;

CT_API uint32_t ct_abi_version(void);
/* Never NULL; empty if no call has failed on this thread. */
CT_API const char *ct_last_error(void);

/* Only tag36h11 is supported. tag_size_m is the black border's edge length
 * in meters; 0 disables the pose solve. Uses the default tunables.
 * Returns NULL on error (see ct_last_error). */
CT_API ct_detector *ct_create(const ct_camera *camera, uint32_t width,
                              uint32_t height, int32_t threads,
                              double tag_size_m);
CT_API void ct_destroy(ct_detector *detector);

/* Detects tags in a contiguous mono8 image of the detector's size
 * (length == width * height). A frame of another size is an error; the
 * detector never reallocates. Returns the number of detections, readable
 * with ct_detection_at until the next ct_detect, or -1 on error. */
CT_API int32_t ct_detect(ct_detector *detector, const uint8_t *pixels,
                         size_t length);
/* 0 on success, -1 for an index out of range. */
CT_API int32_t ct_detection_at(const ct_detector *detector, size_t index,
                               ct_detection *out);

/* --- Added in version 1 ------------------------------------------------ */

/* Fills options with the defaults for a width x height detector. */
CT_API void ct_options_init(ct_options *options, uint32_t width,
                            uint32_t height);
/* Like ct_create, with every tunable. NULL on error. */
CT_API ct_detector *ct_create_with_options(const ct_camera *camera,
                                           const ct_options *options);
/* Row-major homography from the ideal tag square [-1, 1]^2 to pixels.
 * 0 on success, -1 for an index out of range. */
CT_API int32_t ct_detection_homography(const ct_detector *detector,
                                       size_t index, double out[9]);
/* 1 if this library contains the CUDA detector, 0 if it is a build without
 * CUDA (where ct_create always fails). */
CT_API int32_t ct_cuda_available(void);

#ifdef __cplusplus
}
#endif
