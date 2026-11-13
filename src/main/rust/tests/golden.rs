//! The golden image (tests/golden at the repository root) through the
//! detector. Needs a CUDA build (`cargo test --features cuda` on a Jetson);
//! a stub build checks that construction fails cleanly instead.
use std::path::PathBuf;

use cuda_tags::{
    CameraIntrinsics, ColorConversion, Detector, Distortion, Error, Options, PixelFormat,
};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/golden")
        .join(name)
}

fn read_pgm(path: &PathBuf) -> (u32, u32, Vec<u8>) {
    let data = std::fs::read(path).unwrap();
    let header: Vec<&[u8]> = data.splitn(5, |b| b.is_ascii_whitespace()).collect();
    assert_eq!(header[0], b"P5");
    let parse = |field: &[u8]| std::str::from_utf8(field).unwrap().parse::<u32>().unwrap();
    let (width, height) = (parse(header[1]), parse(header[2]));
    let pixels = header[4].to_vec();
    assert_eq!(pixels.len(), (width * height) as usize);
    (width, height, pixels)
}

struct Expected {
    id: u32,
    corners: [[f64; 2]; 4],
    translation: [f64; 3],
}

fn read_scene() -> (CameraIntrinsics, f64, Vec<Expected>) {
    let text = std::fs::read_to_string(golden("tags_640x480.txt")).unwrap();
    let mut camera = None;
    let mut tag_size = 0.0;
    let mut tags = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let numbers = |range: std::ops::Range<usize>| -> Vec<f64> {
            fields[range].iter().map(|f| f.parse().unwrap()).collect()
        };
        match fields[0] {
            "camera" => {
                let v = numbers(1..5);
                camera = Some(CameraIntrinsics {
                    fx: v[0],
                    fy: v[1],
                    cx: v[2],
                    cy: v[3],
                });
            }
            "tag_size" => tag_size = fields[1].parse().unwrap(),
            "tag" => {
                let c = numbers(3..11);
                let t = numbers(fields.len() - 3..fields.len());
                tags.push(Expected {
                    id: fields[1].parse().unwrap(),
                    corners: [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]], [c[6], c[7]]],
                    translation: [t[0], t[1], t[2]],
                });
            }
            _ => {}
        }
    }
    (camera.unwrap(), tag_size, tags)
}

#[test]
fn golden_image() {
    let (camera, tag_size, expected) = read_scene();
    let (width, height, pixels) = read_pgm(&golden("tags_640x480.pgm"));
    let options = Options {
        threads: 2,
        tag_size_m: Some(tag_size),
        ..Options::default()
    };
    let mut detector =
        match Detector::with_options(camera, Distortion::default(), width, height, &options) {
            Ok(detector) => detector,
            Err(Error::Unavailable(reason)) => {
                assert!(!cuda_tags::cuda_available());
                eprintln!("skipping the GPU check: {reason}");
                return;
            }
            Err(error) => panic!("{error}"),
        };

    let detections = detector.detect(&pixels).unwrap();
    assert_eq!(detections.len(), expected.len());
    for tag in &expected {
        let found = detections
            .iter()
            .find(|d| d.id == tag.id)
            .expect("tag missing");
        assert_eq!(found.hamming, 0);
        for (actual, truth) in found.corners.iter().zip(&tag.corners) {
            // Detections put pixel centers at +0.5, the ground truth at 0.
            assert!(
                (actual[0] - 0.5 - truth[0]).abs() < 0.3,
                "{actual:?} vs {truth:?}"
            );
            assert!(
                (actual[1] - 0.5 - truth[1]).abs() < 0.3,
                "{actual:?} vs {truth:?}"
            );
        }
        let pose = found.pose.expect("pose");
        for (actual, truth) in pose.translation.iter().zip(&tag.translation) {
            assert!(
                (actual - truth).abs() < 0.03,
                "{:?} vs {:?}",
                pose.translation,
                tag.translation
            );
        }
    }

    // A frame of another size is an error, not a silent reallocation.
    assert_eq!(
        detector.detect(&pixels[..1000]),
        Err(Error::FrameSize {
            expected: pixels.len(),
            actual: 1000
        })
    );
    // The detector still works afterwards.
    assert_eq!(detector.detect(&pixels).unwrap().len(), expected.len());
}

/// OpenCV's 8-bit BT.601 conversion (cv::cvtColor), for building references.
fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((u32::from(r) * 9798 + u32::from(g) * 19235 + u32::from(b) * 3735 + (1 << 14)) >> 15) as u8
}

#[test]
fn color_frames_match_converting_first() {
    let (camera, tag_size, _) = read_scene();
    let (width, height, gray) = read_pgm(&golden("tags_640x480.pgm"));
    let (w, h) = (width as usize, height as usize);
    // A warm cast with uneven channels, so the weights matter.
    let tint = |v: u8, k: u32, add: u32| ((u32::from(v) * k / 100 + add).min(255)) as u8;
    let mut rgb = Vec::with_capacity(w * h * 3);
    let mut expected_gray = Vec::with_capacity(w * h);
    for (i, &v) in gray.iter().enumerate() {
        let (r, g, b) = (
            tint(v, 100, (i % 7) as u32),
            tint(v, 83, 0),
            tint(v, 61, (i % 5) as u32),
        );
        rgb.extend_from_slice(&[r, g, b]);
        expected_gray.push(luma(r, g, b));
    }
    let bgr: Vec<u8> = rgb.chunks(3).flat_map(|p| [p[2], p[1], p[0]]).collect();
    let rgba: Vec<u8> = rgb
        .chunks(3)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect();
    // RGB rows padded to a 2000-byte stride.
    let stride = 2000;
    let mut padded = vec![0u8; (h - 1) * stride + w * 3];
    for row in 0..h {
        padded[row * stride..row * stride + w * 3]
            .copy_from_slice(&rgb[row * w * 3..(row + 1) * w * 3]);
    }

    for conversion in [
        ColorConversion::WhileDetecting,
        ColorConversion::BeforeDetecting,
    ] {
        let options = Options {
            threads: 2,
            tag_size_m: Some(tag_size),
            color_conversion: conversion,
            ..Options::default()
        };
        let mut detector =
            match Detector::with_options(camera, Distortion::default(), width, height, &options) {
                Ok(detector) => detector,
                Err(Error::Unavailable(reason)) => {
                    eprintln!("skipping the GPU check: {reason}");
                    return;
                }
                Err(error) => panic!("{error}"),
            };
        let reference = detector.detect(&expected_gray).unwrap();
        assert_eq!(reference.len(), 3);
        let check = |label: &str, got: Vec<cuda_tags::Detection>| {
            assert_eq!(got, reference, "{label} with {conversion:?}");
        };
        check(
            "rgb slice",
            detector
                .detect_image(&rgb, w * 3, PixelFormat::Rgb)
                .unwrap(),
        );
        check(
            "bgr slice",
            detector
                .detect_image(&bgr, w * 3, PixelFormat::Bgr)
                .unwrap(),
        );
        check(
            "rgba slice",
            detector
                .detect_image(&rgba, w * 4, PixelFormat::Rgba)
                .unwrap(),
        );
        check(
            "padded rgb",
            detector
                .detect_image(&padded, stride, PixelFormat::Rgb)
                .unwrap(),
        );
        detector.input_buffer()[..rgb.len()].copy_from_slice(&rgb);
        check(
            "rgb in buffer",
            detector
                .detect_input_buffer(w * 3, PixelFormat::Rgb)
                .unwrap(),
        );
        detector.input_buffer()[..padded.len()].copy_from_slice(&padded);
        check(
            "padded rgb in buffer",
            detector
                .detect_input_buffer(stride, PixelFormat::Rgb)
                .unwrap(),
        );
        detector.input_buffer()[..bgr.len()].copy_from_slice(&bgr);
        check(
            "bgr in buffer",
            detector
                .detect_input_buffer(w * 3, PixelFormat::Bgr)
                .unwrap(),
        );
        detector.input_buffer()[..expected_gray.len()].copy_from_slice(&expected_gray);
        check(
            "gray in buffer",
            detector.detect_input_buffer(w, PixelFormat::Gray).unwrap(),
        );

        // Bad strides and short frames are errors, not reads past the end.
        assert_eq!(
            detector.detect_image(&rgb, w * 3 - 1, PixelFormat::Rgb),
            Err(Error::Stride {
                minimum: w * 3,
                actual: w * 3 - 1
            })
        );
        assert!(matches!(
            detector.detect_image(&rgb[..1000], w * 3, PixelFormat::Rgb),
            Err(Error::FrameSize { .. })
        ));
        assert!(matches!(
            detector.detect_input_buffer(w * 5, PixelFormat::Rgba),
            Err(Error::FrameSize { .. })
        ));
    }
}
