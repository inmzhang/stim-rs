use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn collect_vendor_sources(vendor_dir: &Path) -> Vec<PathBuf> {
    let file_list = vendor_dir.join("file_lists").join("source_files_no_main");
    println!("cargo:rerun-if-changed={}", file_list.display());

    fs::read_to_string(&file_list)
        .expect("failed to read Stim source file list")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let path = vendor_dir.join(line);
            println!("cargo:rerun-if-changed={}", path.display());
            path
        })
        .collect()
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=include/stim_rs_bridge.h");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/ffi/circuit.cc");
    println!("cargo:rerun-if-changed=src/ffi/smoke.cc");
    println!("cargo:rerun-if-env-changed=STIM_RS_VENDOR_DIR");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing CARGO_MANIFEST_DIR"));
    let packaged_vendor_dir = manifest_dir.join("vendor").join("stim");
    let (vendor_dir, vendor_mode) = match env::var("STIM_RS_VENDOR_DIR") {
        Ok(path) => (PathBuf::from(path), "override"),
        Err(_) => (packaged_vendor_dir, "packaged"),
    };
    if !vendor_dir.exists() {
        panic!(
            "Stim vendor directory is required at {}",
            vendor_dir.display()
        );
    }

    let vendor_src_dir = vendor_dir.join("src");
    let vendor_sources = collect_vendor_sources(&vendor_dir);

    println!("cargo:rustc-env=STIM_RS_CXX_STANDARD=c++20");
    println!("cargo:rustc-env=STIM_RS_PINNED_STIM_COMMIT=75d9cb12626a9dccca3e9917a5020aa1d3f3dc46");
    println!(
        "cargo:rustc-env=STIM_RS_TARGET={}",
        env::var("TARGET").expect("missing TARGET")
    );
    println!("cargo:rustc-env=STIM_RS_VENDOR_STIM_MODE={vendor_mode}");
    println!(
        "cargo:rustc-env=STIM_RS_VENDOR_STIM_DIR={}",
        vendor_dir.display()
    );

    let mut build = cxx_build::bridge("src/lib.rs");
    build
        .files(&vendor_sources)
        .file("src/ffi/circuit.cc")
        .file("src/ffi/smoke.cc")
        .std("c++20")
        .warnings(false)
        .include("include")
        .include(&vendor_dir)
        .include(&vendor_src_dir);
    build.define("STIM_RS_VENDOR_STIM_PRESENT", Some("1"));

    build.compile("stim-cxx-bridge");
}
