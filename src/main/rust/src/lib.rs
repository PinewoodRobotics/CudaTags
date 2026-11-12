//! Safe Rust API for CudaTags: Team 971's GPU tag36h11 AprilTag detector
//! (Jetson Orin, CUDA) with an optional per-tag pose from OpenCV's
//! `SOLVEPNP_IPPE_SQUARE`.
//!
//! ```no_run
//! use cuda_tags::{CameraIntrinsics, Detector, Distortion, Options};
//!
//! let camera = CameraIntrinsics { fx: 600.0, fy: 600.0, cx: 320.0, cy: 240.0 };
//! let options = Options { threads: 2, tag_size_m: Some(0.1651), ..Options::default() };
//! let mut detector = Detector::with_options(camera, Distortion::default(), 640, 480, &options)?;
//! let frame = vec![0u8; 640 * 480]; // mono8, row-major, no padding
//! for tag in detector.detect(&frame)? {
//!     println!("tag {} at {:?}, pose {:?}", tag.id, tag.center, tag.pose);
//! }
//! # Ok::<(), cuda_tags::Error>(())
//! ```
//!
//! # Building
//!
//! The GPU detector is built only with the `cuda` feature, on Linux, with a
//! CUDA toolkit and OpenCV (core, imgproc, calib3d) installed. Everywhere
//! else the crate is a stub: it compiles and [`Detector::new`] returns
//! [`Error::Unavailable`]. [`cuda_available`] tells which one was built.
//!
//! # Threading
//!
//! A [`Detector`] owns a CUDA stream, GPU buffers and a decode thread pool.
//! It is neither `Send` nor `Sync`: build it on the thread that uses it.
//!
//! ```compile_fail
//! fn needs_send<T: Send>() {}
//! needs_send::<cuda_tags::Detector>();
//! ```
#![deny(unsafe_op_in_unsafe_fn)]

use std::fmt;
use std::marker::PhantomData;

#[cfg(cuda_tags_native)]
mod ffi;

/// Pinhole intrinsics in pixels, at the detector's image size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// OpenCV (Brown-Conrady) distortion coefficients. The default (all zero)
/// is an ideal pinhole camera.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Distortion {
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
}

/// Detector tunables. The defaults match AprilTag's, as frc971 uses them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// CPU threads that decode candidate quads, 1..=64. Default 1.
    pub threads: u32,
    /// Quad detection runs on the image decimated by this factor, 1 or 2.
    /// Default 2.
    pub decimate: u32,
    /// Minimum white/black contrast across a tag edge. Default 5.
    pub min_white_black_diff: u8,
    /// Bit errors corrected while decoding, 0..=2. Default 1.
    pub max_hamming: u8,
    /// Refine quad edges against the full-resolution image. Default true.
    pub refine_edges: bool,
    /// Sharpening of decoded bit samples. Default 0.25.
    pub decode_sharpening: f64,
    /// Black-border edge length of the tags in meters. When set, every
    /// detection carries a [`Pose`]. Default `None`.
    pub tag_size_m: Option<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            threads: 1,
            decimate: 2,
            min_white_black_diff: 5,
            max_hamming: 1,
            refine_edges: true,
            decode_sharpening: 0.25,
            tag_size_m: None,
        }
    }
}

/// Camera-from-tag transform: `p_camera = rotation * p_tag + translation`.
///
/// The tag frame is AprilTag's: origin at the tag center, x right, y down,
/// z into the tag (pointing away from a camera that sees its face).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    /// Rows of the rotation matrix.
    pub rotation: [[f64; 3]; 3],
    /// Meters.
    pub translation: [f64; 3],
}

/// One detected tag.
///
/// Pixel coordinates are in the input image (distorted, not undistorted)
/// and, like AprilTag's, put the center of pixel `(x, y)` at
/// `(x + 0.5, y + 0.5)`. Subtract 0.5 to compare with OpenCV.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub id: u32,
    /// Bit errors corrected while decoding.
    pub hamming: u32,
    pub decision_margin: f64,
    pub center: [f64; 2],
    /// AprilTag's order: bottom-left, bottom-right, top-right, top-left of
    /// an upright tag, i.e. tag-frame points (-s, s), (s, s), (s, -s),
    /// (-s, -s) for s = tag_size / 2.
    pub corners: [[f64; 2]; 4],
    /// Rows of the homography from the ideal tag square [-1, 1]^2 to pixels.
    pub homography: [[f64; 3]; 3],
    /// Set when [`Options::tag_size_m`] is, unless the pose solve failed.
    pub pose: Option<Pose>,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// This build has no GPU detector (see the crate documentation); the
    /// string says why.
    Unavailable(String),
    /// A frame's length is not the detector's `width * height`. The
    /// detector never reallocates for a new size.
    FrameSize { expected: usize, actual: usize },
    /// The configuration was rejected or the native detector failed.
    Native(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unavailable(reason) => {
                write!(f, "CudaTags GPU detection is unavailable: {reason}")
            }
            Error::FrameSize { expected, actual } => write!(
                f,
                "a {actual}-byte frame does not fit a detector built for {expected} pixels"
            ),
            Error::Native(message) => write!(f, "CudaTags: {message}"),
        }
    }
}

impl std::error::Error for Error {}

/// True when this build contains the CUDA detector.
pub fn cuda_available() -> bool {
    cfg!(cuda_tags_native)
}

/// Why this build is a stub, or `None` for a native build.
pub fn stub_reason() -> Option<&'static str> {
    #[cfg(cuda_tags_native)]
    return None;
    #[cfg(not(cuda_tags_native))]
    return Some(env!("CUDA_TAGS_STUB_REASON"));
}

/// A GPU tag36h11 detector for mono8 frames of one fixed size.
pub struct Detector {
    #[cfg(cuda_tags_native)]
    raw: std::ptr::NonNull<ffi::ct_detector>,
    width: u32,
    height: u32,
    /// Not Send or Sync: the GPU state is bound to the creating thread.
    _thread_bound: PhantomData<*mut ()>,
}

impl fmt::Debug for Detector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Detector")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl Detector {
    /// A detector with default [`Options`] and `threads` decode threads, and
    /// no pose output. `width` and `height` must be multiples of 8 with
    /// `width * height < 2^22`.
    pub fn new(
        camera: CameraIntrinsics,
        distortion: Distortion,
        width: u32,
        height: u32,
        threads: u32,
    ) -> Result<Detector, Error> {
        let options = Options {
            threads,
            ..Options::default()
        };
        Detector::with_options(camera, distortion, width, height, &options)
    }

    /// A detector with every tunable; see [`Options`].
    pub fn with_options(
        camera: CameraIntrinsics,
        distortion: Distortion,
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Detector, Error> {
        if let Some(size) = options.tag_size_m {
            if !(size.is_finite() && size > 0.0) {
                return Err(Error::Native(format!(
                    "tag_size_m must be finite and positive, got {size}"
                )));
            }
        }
        Detector::create(camera, distortion, width, height, options)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Detects tags in a contiguous mono8 frame of exactly
    /// `width * height` bytes.
    pub fn detect(&mut self, gray: &[u8]) -> Result<Vec<Detection>, Error> {
        let expected = self.width as usize * self.height as usize;
        if gray.len() != expected {
            return Err(Error::FrameSize {
                expected,
                actual: gray.len(),
            });
        }
        self.detect_native(gray)
    }
}

#[cfg(not(cuda_tags_native))]
impl Detector {
    fn create(
        _camera: CameraIntrinsics,
        _distortion: Distortion,
        _width: u32,
        _height: u32,
        _options: &Options,
    ) -> Result<Detector, Error> {
        Err(Error::Unavailable(
            stub_reason().unwrap_or("built without CUDA").to_string(),
        ))
    }

    fn detect_native(&mut self, _gray: &[u8]) -> Result<Vec<Detection>, Error> {
        // A stub Detector cannot be constructed.
        unreachable!("stub detectors are never created")
    }
}

#[cfg(cuda_tags_native)]
impl Detector {
    fn create(
        camera: CameraIntrinsics,
        distortion: Distortion,
        width: u32,
        height: u32,
        options: &Options,
    ) -> Result<Detector, Error> {
        // SAFETY: plain C calls; every pointer passed is valid for the call.
        unsafe {
            if ffi::ct_abi_version() != ffi::ABI_VERSION {
                return Err(Error::Native(format!(
                    "linked C ABI version {} but this crate expects {}",
                    ffi::ct_abi_version(),
                    ffi::ABI_VERSION
                )));
            }
            let raw_camera = ffi::ct_camera {
                fx: camera.fx,
                fy: camera.fy,
                cx: camera.cx,
                cy: camera.cy,
                k1: distortion.k1,
                k2: distortion.k2,
                p1: distortion.p1,
                p2: distortion.p2,
                k3: distortion.k3,
            };
            let mut raw_options = ffi::ct_options::default();
            ffi::ct_options_init(&mut raw_options, width, height);
            raw_options.threads = i32::try_from(options.threads).unwrap_or(i32::MAX);
            raw_options.decimate = i32::try_from(options.decimate).unwrap_or(i32::MAX);
            raw_options.min_white_black_diff = options.min_white_black_diff.into();
            raw_options.max_hamming = options.max_hamming.into();
            raw_options.refine_edges = options.refine_edges.into();
            raw_options.decode_sharpening = options.decode_sharpening;
            raw_options.tag_size_m = options.tag_size_m.unwrap_or(0.0);
            let raw = ffi::ct_create_with_options(&raw_camera, &raw_options);
            match std::ptr::NonNull::new(raw) {
                Some(raw) => Ok(Detector {
                    raw,
                    width,
                    height,
                    _thread_bound: PhantomData,
                }),
                None => Err(Error::Native(last_error())),
            }
        }
    }

    fn detect_native(&mut self, gray: &[u8]) -> Result<Vec<Detection>, Error> {
        // SAFETY: `raw` is a live detector owned by self (and &mut self
        // serializes calls); `gray` is valid for `gray.len()` bytes; the
        // detection buffers are valid until the next ct_detect, which cannot
        // happen while this borrow lasts.
        unsafe {
            let count = ffi::ct_detect(self.raw.as_ptr(), gray.as_ptr(), gray.len());
            if count < 0 {
                return Err(Error::Native(last_error()));
            }
            let mut detections = Vec::with_capacity(count as usize);
            for index in 0..count as usize {
                let mut raw = ffi::ct_detection::default();
                let mut homography = [0.0f64; 9];
                if ffi::ct_detection_at(self.raw.as_ptr(), index, &mut raw) != 0
                    || ffi::ct_detection_homography(
                        self.raw.as_ptr(),
                        index,
                        homography.as_mut_ptr(),
                    ) != 0
                {
                    return Err(Error::Native(last_error()));
                }
                detections.push(Detection {
                    id: raw.id as u32,
                    hamming: raw.hamming as u32,
                    decision_margin: raw.decision_margin,
                    center: raw.center,
                    corners: raw.corners,
                    homography: rows(&homography),
                    pose: (raw.pose_valid != 0).then(|| Pose {
                        rotation: rows(&raw.rotation),
                        translation: raw.translation,
                    }),
                });
            }
            Ok(detections)
        }
    }
}

#[cfg(cuda_tags_native)]
impl Drop for Detector {
    fn drop(&mut self) {
        // SAFETY: `raw` came from ct_create_with_options and is freed once.
        unsafe { ffi::ct_destroy(self.raw.as_ptr()) }
    }
}

#[cfg(cuda_tags_native)]
fn rows(values: &[f64; 9]) -> [[f64; 3]; 3] {
    [
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ]
}

#[cfg(cuda_tags_native)]
fn last_error() -> String {
    // SAFETY: ct_last_error returns a valid NUL-terminated thread-local
    // string, never NULL by contract; checked anyway.
    unsafe {
        let pointer = ffi::ct_last_error();
        if pointer.is_null() {
            return "unknown CudaTags error".into();
        }
        std::ffi::CStr::from_ptr(pointer)
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAMERA: CameraIntrinsics = CameraIntrinsics {
        fx: 600.0,
        fy: 600.0,
        cx: 320.0,
        cy: 240.0,
    };

    #[test]
    fn options_default_to_apriltag_values() {
        let options = Options::default();
        assert_eq!(options.threads, 1);
        assert_eq!(options.decimate, 2);
        assert_eq!(options.min_white_black_diff, 5);
        assert_eq!(options.max_hamming, 1);
        assert!(options.refine_edges);
        assert_eq!(options.decode_sharpening, 0.25);
        assert_eq!(options.tag_size_m, None);
    }

    #[test]
    fn rejects_invalid_tag_size_before_reaching_native_code() {
        for size in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let options = Options {
                tag_size_m: Some(size),
                ..Options::default()
            };
            let error = Detector::with_options(CAMERA, Distortion::default(), 640, 480, &options)
                .unwrap_err();
            assert!(matches!(error, Error::Native(_)), "{error:?}");
        }
    }

    #[test]
    fn availability_is_consistent() {
        assert_eq!(cuda_available(), stub_reason().is_none());
        #[cfg(not(cuda_tags_native))]
        assert!(!cuda_available());
    }

    #[cfg(not(cuda_tags_native))]
    #[test]
    fn stub_build_reports_unavailable() {
        let error = Detector::new(CAMERA, Distortion::default(), 640, 480, 1).unwrap_err();
        let Error::Unavailable(reason) = &error else {
            panic!("expected Unavailable, got {error:?}");
        };
        assert_eq!(Some(reason.as_str()), stub_reason());
        assert!(error.to_string().contains("unavailable"));
    }

    #[cfg(cuda_tags_native)]
    #[test]
    fn native_build_rejects_bad_configuration_without_panicking() {
        let error = Detector::new(CAMERA, Distortion::default(), 642, 480, 1).unwrap_err();
        assert!(
            matches!(&error, Error::Native(m) if m.contains("multiples of 8")),
            "{error:?}"
        );
        let error = Detector::new(CAMERA, Distortion::default(), 640, 480, 0).unwrap_err();
        assert!(matches!(error, Error::Native(_)));
    }

    #[test]
    fn error_messages() {
        let error = Error::FrameSize {
            expected: 307_200,
            actual: 12,
        };
        assert_eq!(
            error.to_string(),
            "a 12-byte frame does not fit a detector built for 307200 pixels"
        );
    }
}
