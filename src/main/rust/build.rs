//! Builds the CudaTags core (C ABI, frc971 GPU detector, AprilTag) with
//! CMake and links it statically, or configures a stub build.
//!
//! This script never panics on an unsupported platform: without the `cuda`
//! feature, off Linux, or without a CUDA toolkit it builds the stub (see
//! Cargo.toml). Only a failing native build on a machine that has CUDA stops
//! the build, with the CMake output.
//!
//! Environment:
//!   CUDA_HOME / CUDA_PATH / CUDAToolkit_ROOT  CUDA toolkit root
//!                                   (default: /usr/local/cuda, then nvcc on PATH)
//!   CMAKE_CUDA_ARCHITECTURES        GPU architectures (default 87, Jetson Orin)
//!   CUDA_TAGS_REQUIRE_CUDA=1        fail instead of building the stub
//!   CMAKE                           cmake executable (default: cmake)
//!   PKG_CONFIG_PATH                 where pkg-config finds opencv4.pc

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(cuda_tags_native)");
    for variable in [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDAToolkit_ROOT",
        "CMAKE_CUDA_ARCHITECTURES",
        "CUDA_TAGS_REQUIRE_CUDA",
        "CMAKE",
        "PKG_CONFIG_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    println!("cargo:rerun-if-changed=build.rs");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return stub("the `cuda` feature is off", false);
    }
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        return stub(
            &format!("CUDA builds are supported on Linux only (target OS: {target_os})"),
            true,
        );
    }
    let Some(cuda) = find_cuda() else {
        return stub(
            "no CUDA toolkit found (set CUDA_HOME to the directory containing bin/nvcc)",
            true,
        );
    };
    if let Err(error) = build_native(&cuda) {
        fail(&error);
    }
    println!("cargo:rustc-cfg=cuda_tags_native");
}

/// Configures the stub build. `warn` reports it as a cargo warning, for
/// builds that asked for CUDA.
fn stub(reason: &str, warn: bool) {
    if env::var("CUDA_TAGS_REQUIRE_CUDA").as_deref() == Ok("1") {
        fail(&format!("CUDA_TAGS_REQUIRE_CUDA=1 but {reason}"));
    }
    if warn {
        println!("cargo:warning=cuda_tags: building the stub detector: {reason}");
    }
    println!("cargo:rustc-env=CUDA_TAGS_STUB_REASON={reason}");
}

fn fail(message: &str) -> ! {
    eprintln!("error: cuda_tags: {message}");
    exit(1);
}

fn find_cuda() -> Option<PathBuf> {
    let has_nvcc = |root: &Path| root.join("bin").join("nvcc").is_file();
    for variable in ["CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT"] {
        if let Some(root) = env::var_os(variable).map(PathBuf::from) {
            if has_nvcc(&root) {
                return Some(root);
            }
            println!(
                "cargo:warning=cuda_tags: {variable}={} has no bin/nvcc",
                root.display()
            );
        }
    }
    let default = PathBuf::from("/usr/local/cuda");
    if has_nvcc(&default) {
        return Some(default);
    }
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|directory| directory.join("nvcc"))
        .find(|nvcc| nvcc.is_file())
        .and_then(|nvcc| nvcc.canonicalize().ok())
        .and_then(|nvcc| Some(nvcc.parent()?.parent()?.to_path_buf()))
}

fn build_native(cuda: &Path) -> Result<(), String> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").map_err(|e| e.to_string())?);
    let root = manifest
        .ancestors()
        .nth(3)
        .ok_or("cannot find the CudaTags repository root")?
        .to_path_buf();
    if !root
        .join("detection/third_party/apriltag/apriltag.c")
        .is_file()
    {
        return Err(format!(
            "the AprilTag submodule is missing; run `git submodule update --init \
             detection/third_party/apriltag` in {}",
            root.display()
        ));
    }
    for watched in [
        "CMakeLists.txt",
        "src/main/cpp",
        "detection/CMakeLists.txt",
        "detection/frc971",
        "detection/apriltag_971",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(watched).display());
    }

    let out = PathBuf::from(env::var("OUT_DIR").map_err(|e| e.to_string())?);
    let build = out.join("cmake-build");
    let cmake = env::var("CMAKE").unwrap_or_else(|_| "cmake".into());
    let architectures = env::var("CMAKE_CUDA_ARCHITECTURES").unwrap_or_else(|_| "87".into());
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "2".into());

    run(Command::new(&cmake)
        .arg("-S")
        .arg(&root)
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DCUDATAGS_WITH_CUDA=ON")
        .arg("-DCUDATAGS_BUILD_TESTS=OFF")
        .arg("-DCUDATAGS_BUILD_PYTHON=OFF")
        .arg("-DCUDATAGS_BUILD_JAVA=OFF")
        .arg("-DCUDATAGS_BUILD_C_LIBRARY=OFF")
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={architectures}"))
        .arg(format!("-DCUDAToolkit_ROOT={}", cuda.display()))
        .arg(format!(
            "-DCMAKE_CUDA_COMPILER={}",
            cuda.join("bin/nvcc").display()
        )))?;
    run(Command::new(&cmake).arg("--build").arg(&build).args([
        "--target",
        "cuda_tags_core",
        "--parallel",
        &jobs,
    ]))?;

    // Static libraries, dependents first.
    for (directory, library) in [
        ("src/main/cpp", "cuda_tags_core"),
        ("detection", "971apriltag"),
        ("detection/third_party/apriltag", "apriltag"),
    ] {
        println!(
            "cargo:rustc-link-search=native={}",
            build.join(directory).display()
        );
        println!("cargo:rustc-link-lib=static={library}");
    }

    link_opencv(&build);

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    for directory in [
        cuda.join("lib64"),
        cuda.join("lib"),
        cuda.join(format!("targets/{target_arch}-linux/lib")),
    ] {
        if directory.is_dir() {
            println!("cargo:rustc-link-search=native={}", directory.display());
        }
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    for library in ["stdc++", "dl", "rt", "pthread", "m"] {
        println!("cargo:rustc-link-lib={library}");
    }
    Ok(())
}

/// Links OpenCV's core, imgproc and calib3d, found with pkg-config, or next
/// to the OpenCV CMake found.
fn link_opencv(build: &Path) {
    match pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("opencv4")
    {
        Ok(library) => {
            for path in library.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
        }
        Err(error) => {
            let cache = std::fs::read_to_string(build.join("CMakeCache.txt")).unwrap_or_default();
            let opencv_dir = cache
                .lines()
                .find_map(|line| line.strip_prefix("OpenCV_DIR:PATH="))
                .map(PathBuf::from);
            // <libdir>/cmake/opencv4 -> <libdir>
            match opencv_dir.as_deref().and_then(|dir| dir.ancestors().nth(2)) {
                Some(libdir) => {
                    println!(
                        "cargo:warning=cuda_tags: pkg-config cannot find opencv4 ({error}); \
                         linking OpenCV from {}",
                        libdir.display()
                    );
                    println!("cargo:rustc-link-search=native={}", libdir.display());
                }
                None => println!(
                    "cargo:warning=cuda_tags: pkg-config cannot find opencv4 ({error}); \
                     relying on the default library path"
                ),
            }
        }
    }
    for library in ["opencv_calib3d", "opencv_imgproc", "opencv_core"] {
        println!("cargo:rustc-link-lib={library}");
    }
}

fn run(command: &mut Command) -> Result<(), String> {
    let status = command
        .status()
        .map_err(|error| format!("cannot run {command:?}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{command:?} failed with {status}; see the output above"
        ))
    }
}
