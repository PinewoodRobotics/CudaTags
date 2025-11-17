use cuda_tags::bridge::ffi::{
    CameraMatrix, DistCoeffs, RustImage, TagType, make_cuda_tag_detector,
};
use opencv::{core, highgui, imgproc, prelude::*, videoio};

use std::time::Instant;

fn main() -> opencv::Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_V4L2)?;
    // Set capture format to MJPG
    cam.set(
        videoio::CAP_PROP_FOURCC,
        videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?.into(),
    )?;
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;
    cam.set(videoio::CAP_PROP_FPS, 30.0)?;

    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Camera not opened");
    }

    let mut gpu_detector = make_cuda_tag_detector(
        TagType::tag36h11,
        CameraMatrix {
            fx: 1.0,
            cx: 1.0,
            fy: 1.0,
            cy: 1.0,
        },
        DistCoeffs {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        },
        5,
        640,
        480,
    );

    let window = "cam";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();

    // FPS calculation variables
    let mut last_time = Instant::now();
    let mut frame_count = 0;
    let mut fps = 0.0;

    loop {
        cam.read(&mut frame)?;
        if frame.empty() {
            continue;
        }

        // Precompute image properties before taking a mutable borrow of the frame's data
        let width = frame.cols();
        let height = frame.rows();
        let stride = frame.step1(0)? as i32;
        let channels = frame.channels();

        // Get frame bytes safely
        let pixels = match frame.data_bytes_mut() {
            Ok(slice) => slice,
            Err(_) => {
                eprintln!("Could not get mutable access to frame bytes.");
                continue;
            }
        };

        let start = Instant::now();
        // Get detections instead of dropping them on the floor
        let detections = gpu_detector.pin_mut().process(RustImage {
            pixels,
            width,
            height,
            stride,
            channels,
        });
        let duration = start.elapsed();

        frame_count += 1;

        // Update FPS every 10 frames for stability
        if frame_count >= 30 {
            let now = Instant::now();
            let duration = now.duration_since(last_time).as_secs_f64();
            if duration > 0.0 {
                fps = frame_count as f64 / duration;
            }
            eprintln!(
                "[CudaTags-Rust] frame {}x{} channels={} stride={} detections={} time_ms={}",
                width,
                height,
                channels,
                stride,
                detections.len(),
                duration
            );
            last_time = now;
            frame_count = 0;
        }

        // Overlay FPS and duration on the frame, and plot detections
        let mut display_frame = frame.clone();

        // Draw detections
        for detection in &detections {
            // Draw the quadrilateral
            let corners = &detection.corners;
            for i in 0..4 {
                let j = (i + 1) % 4;
                let pt1 = core::Point::new(corners[i][0] as i32, corners[i][1] as i32);
                let pt2 = core::Point::new(corners[j][0] as i32, corners[j][1] as i32);
                imgproc::line(
                    &mut display_frame,
                    pt1,
                    pt2,
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_AA,
                    0,
                )?;
            }
            // Draw the center point
            let center_pt =
                core::Point::new(detection.center[0] as i32, detection.center[1] as i32);
            imgproc::circle(
                &mut display_frame,
                center_pt,
                5,
                core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                -1,
                imgproc::LINE_AA,
                0,
            )?;

            // Draw tag id near center
            imgproc::put_text(
                &mut display_frame,
                &format!("ID: {}", detection.id),
                core::Point::new(
                    (detection.center[0] as i32 + 10),
                    (detection.center[1] as i32 + 10),
                ),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                core::Scalar::new(255.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )?;
        }

        // Overlay FPS and process duration text
        let text = format!("FPS: {:.2}", fps);
        imgproc::put_text(
            &mut display_frame,
            &text,
            core::Point::new(10, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            1.0,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;
        imgproc::put_text(
            &mut display_frame,
            &format!("Duration: {:.2}ms", duration.as_millis() as f64),
            core::Point::new(10, 60),
            imgproc::FONT_HERSHEY_SIMPLEX,
            1.0,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;

        imgproc::put_text(
            &mut display_frame,
            &format!("Detections: {}", detections.len()),
            core::Point::new(10, 90),
            imgproc::FONT_HERSHEY_SIMPLEX,
            1.0,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;

        highgui::imshow(window, &display_frame)?;
        let key = highgui::wait_key(1)?;
        if key == 27 {
            break;
        }
    }

    Ok(())
}
