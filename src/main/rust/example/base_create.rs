use cuda_tags::bridge::ffi::{
    CameraMatrix, DistCoeffs, RustImage, TagType, make_cuda_tag_detector,
};

fn main() {
    let mut _gpu_detector = make_cuda_tag_detector(
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

    println!("GPU detector created");
}
