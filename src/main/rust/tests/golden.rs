//! The golden image (tests/golden at the repository root) through the
//! detector. Needs a CUDA build (`cargo test --features cuda` on a Jetson);
//! a stub build checks that construction fails cleanly instead.
use std::path::PathBuf;

use cuda_tags::{CameraIntrinsics, Detector, Distortion, Error, Options};

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
