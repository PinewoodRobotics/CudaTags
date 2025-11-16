use std::path::PathBuf;

fn main() {
    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    cxx_build::bridge("src/bridge.rs")
        .file("src/cpp/cuda_tags_wrapper.cpp")
        .include("src/cpp/")
        .include(".")
        .include("/usr/local/cuda/include/")
        .include("/usr/include/opencv4")
        .flag_if_supported("-std=c++20")
        .compile("cuda_tags");

    let base_search_dir = cargo_manifest_dir.join("lib").join("linux");
    let search_path = match std::env::consts::ARCH {
        "x86_64" => base_search_dir.join("x86_64"),
        "aarch64" => base_search_dir.join("aarch64"),
        _ => {
            panic!("Unsupported architecture: {}", std::env::consts::ARCH);
        }
    };

    println!("cargo:rustc-link-search=native={}", search_path.display());
    println!("cargo:rustc-link-lib=971apriltag");
}
