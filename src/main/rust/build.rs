use std::path::PathBuf;

fn main() {
    let workspace_root = PathBuf::from(&std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = workspace_root
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let cargo_manifest_dir = PathBuf::from(&std::env::var("CARGO_MANIFEST_DIR").unwrap());

    println!("cargo:rerun-if-changed=src/cpp/cuda_tags_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/cpp/cuda_tags_wrapper.h");
    println!("cargo:rerun-if-changed=src/bridge.rs");

    cxx_build::bridge("src/bridge.rs")
        .file("src/cpp/cuda_tags_wrapper.cpp")
        .include(cargo_manifest_dir.join("src/cpp").display().to_string())
        .include(workspace_root.join("detection").display().to_string())
        .include("/usr/local/cuda/include/")
        .include("/usr/include/opencv4")
        .include(
            workspace_root
                .join("target/cxxbridge")
                .display()
                .to_string(),
        )
        .flag("-w")
        .flag_if_supported("-std=c++20")
        .compile("cuda_tags");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    let base_search_dir = workspace_root.join("lib").join("linux");
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
