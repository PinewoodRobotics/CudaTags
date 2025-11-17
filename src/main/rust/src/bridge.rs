use cxx::UniquePtr;
use std::pin::Pin;

#[cxx::bridge]
pub mod ffi {

    unsafe extern "C++" {
        include!("cuda_tags_wrapper.h");
        type CudaTagsWrapper;

        fn make_cuda_tag_detector(
            tagType: TagType,
            camMatrix: CameraMatrix,
            camDistCoeffs: DistCoeffs,
            nthreads: i32,
            imgWidth: i32,
            imgHeight: i32,
        ) -> UniquePtr<CudaTagsWrapper>;

        fn process(self: Pin<&mut CudaTagsWrapper>, image: RustImage) -> Vec<DetectionResult>;
    }

    pub struct RustImage<'a> {
        pixels: &'a mut [u8],
        width: i32,
        height: i32,
        stride: i32,
        channels: i32,
    }

    #[derive(Debug, Clone, Copy)]
    pub enum TagType {
        tag36h11 = 1,
    }

    pub struct CameraMatrix {
        fx: f64,
        cx: f64,
        fy: f64,
        cy: f64,
    }

    pub struct DistCoeffs {
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    }

    pub struct DetectionResult {
        id: i32,
        hamming: i32,
        decision_margin: f32,
        homography: [[f64; 3]; 3],
        center: [f64; 2],
        corners: [[f64; 2]; 4],
    }
}
