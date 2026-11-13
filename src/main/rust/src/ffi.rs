//! Declarations matching `src/main/cpp/include/cuda_tags/cuda_tags.h`
//! (ABI version 1). Only compiled into native builds.
#![allow(non_camel_case_types)]

use std::ffi::c_char;

pub const ABI_VERSION: u32 = 1;

pub const CT_COLOR_WHILE_DETECTING: i32 = 0;
pub const CT_COLOR_BEFORE_DETECTING: i32 = 1;

#[repr(C)]
pub struct ct_detector {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ct_camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
}

#[repr(C)]
#[derive(Default)]
pub struct ct_detection {
    pub id: i32,
    pub hamming: i32,
    pub decision_margin: f64,
    pub center: [f64; 2],
    pub corners: [[f64; 2]; 4],
    pub pose_valid: i32,
    pub rotation: [f64; 9],
    pub translation: [f64; 3],
}

#[repr(C)]
#[derive(Default)]
pub struct ct_options {
    pub width: u32,
    pub height: u32,
    pub threads: i32,
    pub decimate: i32,
    pub min_white_black_diff: i32,
    pub max_hamming: i32,
    pub refine_edges: i32,
    pub decode_sharpening: f64,
    pub tag_size_m: f64,
}

// Layouts must match the C header.
const _: () = assert!(std::mem::size_of::<ct_detection>() == 200);
const _: () = assert!(std::mem::size_of::<ct_options>() == 48);
const _: () = assert!(std::mem::size_of::<ct_camera>() == 72);

unsafe extern "C" {
    pub fn ct_abi_version() -> u32;
    pub fn ct_last_error() -> *const c_char;
    pub fn ct_options_init(options: *mut ct_options, width: u32, height: u32);
    pub fn ct_create_with_options(
        camera: *const ct_camera,
        options: *const ct_options,
    ) -> *mut ct_detector;
    pub fn ct_destroy(detector: *mut ct_detector);
    pub fn ct_detect(detector: *mut ct_detector, pixels: *const u8, length: usize) -> i32;
    pub fn ct_detection_at(
        detector: *const ct_detector,
        index: usize,
        out: *mut ct_detection,
    ) -> i32;
    pub fn ct_detection_homography(
        detector: *const ct_detector,
        index: usize,
        out: *mut f64,
    ) -> i32;
    pub fn ct_detect_image(
        detector: *mut ct_detector,
        pixels: *const u8,
        length: usize,
        stride: usize,
        format: i32,
    ) -> i32;
    pub fn ct_input_buffer(detector: *mut ct_detector, size: *mut usize) -> *mut u8;
    pub fn ct_set_color_conversion(detector: *mut ct_detector, conversion: i32) -> i32;
}
