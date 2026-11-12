use std::path::PathBuf;

fn main() {
    if std::env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        return;
    }
    for variable in ["CUDA_HOME", "OPENCV_INCLUDE_DIR", "CUDATAGS_LIB_DIR"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    let cuda_home =
        PathBuf::from(std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into()));
    let cargo_manifest_dir = PathBuf::from(&std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root_project_dir = cargo_manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    println!("cargo:rerun-if-changed=src/cpp/cuda_tags_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/cpp/cuda_tags_wrapper.h");
    println!("cargo:rerun-if-changed=src/bridge.rs");

    cxx_build::bridge("src/bridge.rs")
        .file("src/cpp/cuda_tags_wrapper.cpp")
        .include(cargo_manifest_dir.join("src/cpp").display().to_string())
        .include(root_project_dir.join("detection").display().to_string())
        .include(cuda_home.join("include"))
        .include(
            std::env::var("OPENCV_INCLUDE_DIR").unwrap_or_else(|_| "/usr/include/opencv4".into()),
        )
        .include(
            root_project_dir
                .join("target/cxxbridge")
                .display()
                .to_string(),
        )
        .flag_if_supported("-std=c++20")
        .compile("cuda_tags");

    println!(
        "cargo:rustc-link-search=native={}",
        cuda_home.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=cudart");

    let base_search_dir = root_project_dir.join("lib").join("linux");
    let search_path = match std::env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "aarch64" => base_search_dir.join("aarch64"),
        "x86_64" => base_search_dir.join("x86_64"),
        _ => {
            panic!("Unsupported architecture: {}", std::env::consts::ARCH);
        }
    };

    let search_path = std::env::var_os("CUDATAGS_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or(search_path);
    println!("cargo:rustc-link-search=native={}", search_path.display());
    println!("cargo:rustc-link-lib=opencv_core");
    println!("cargo:rustc-link-lib=opencv_imgproc");
    println!("cargo:rustc-link-lib=971apriltag");
    println!("cargo:rustc-link-lib=static=apriltag");
}
