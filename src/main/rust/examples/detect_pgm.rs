//! Detects tags in an 8-bit binary PGM image and prints them.
//!
//!   cargo run --release --features cuda --example detect_pgm -- \
//!       [image.pgm] [fx fy cx cy] [tag_size_m]
//!
//! Defaults to the repository's golden image and its camera
//! (fx = fy = 600, cx = 320, cy = 240, 0.1651 m tags).
use std::process::ExitCode;

use cuda_tags::{CameraIntrinsics, Detector, Distortion, Options};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().cloned().unwrap_or_else(|| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../tests/golden/tags_640x480.pgm"
        )
        .into()
    });
    let number = |index: usize, default: f64| {
        args.get(index)
            .map_or(default, |value| value.parse().expect("not a number"))
    };
    let camera = CameraIntrinsics {
        fx: number(1, 600.0),
        fy: number(2, 600.0),
        cx: number(3, 320.0),
        cy: number(4, 240.0),
    };
    let options = Options {
        threads: 2,
        tag_size_m: Some(number(5, 0.1651)),
        ..Options::default()
    };

    let data = match std::fs::read(&path) {
        Ok(data) => data,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let fields: Vec<&[u8]> = data.splitn(5, |b| b.is_ascii_whitespace()).collect();
    let parse = |field: &[u8]| std::str::from_utf8(field).ok()?.parse::<u32>().ok();
    let (Some(width), Some(height)) = (parse(fields[1]), parse(fields[2])) else {
        eprintln!("{path} is not an 8-bit binary PGM");
        return ExitCode::FAILURE;
    };
    if fields[0] != b"P5" || fields[3] != b"255" {
        eprintln!("{path} is not an 8-bit binary PGM");
        return ExitCode::FAILURE;
    }

    let mut detector =
        match Detector::with_options(camera, Distortion::default(), width, height, &options) {
            Ok(detector) => detector,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        };
    match detector.detect(fields[4]) {
        Ok(detections) => {
            for tag in detections {
                println!(
                    "tag {:>3}  hamming {}  margin {:6.1}  center ({:7.2}, {:7.2})  translation {:?}",
                    tag.id,
                    tag.hamming,
                    tag.decision_margin,
                    tag.center[0],
                    tag.center[1],
                    tag.pose.map(|pose| pose.translation)
                );
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
