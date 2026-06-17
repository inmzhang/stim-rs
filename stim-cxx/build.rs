use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const PINNED_STIM_COMMIT: &str = "e9cc7dee0a67bb4dbea2733f0170d03dcecd3cd0";

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

fn env_is_true(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn prebuilt_names(target: &str) -> Option<(&'static str, &'static str, &'static str)> {
    if target.ends_with("windows-msvc") {
        Some(("libstim.lib", "libstim", "lib"))
    } else if target.ends_with("unknown-linux-gnu") || target == "aarch64-apple-darwin" {
        Some(("libstim.a", "stim", "a"))
    } else {
        None
    }
}

fn prebuilt_simd_width(target: &str) -> Option<u16> {
    if target.starts_with("x86_64-") {
        Some(128)
    } else if target.starts_with("aarch64-") {
        Some(64)
    } else {
        None
    }
}

fn release_asset_name(target: &str, extension: &str) -> String {
    format!("stim-cxx-libstim-{target}.{extension}")
}

fn release_asset_url(target: &str, extension: &str) -> String {
    let base = env::var("STIM_RS_PREBUILT_BASE_URL").unwrap_or_else(|_| {
        let repo = env::var("CARGO_PKG_REPOSITORY")
            .unwrap_or_else(|_| "https://github.com/inmzhang/stim-rs".to_owned())
            .trim_end_matches(".git")
            .to_owned();
        format!("{repo}/releases/download/v{}", env!("CARGO_PKG_VERSION"))
    });
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        release_asset_name(target, extension)
    )
}

fn expected_sha256(asset_name: &str) -> Option<String> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").ok()?);
    let manifest = fs::read_to_string(manifest_dir.join("prebuilt-sha256.txt")).ok()?;
    manifest
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            Some((parts.next()?, parts.next()?))
        })
        .find_map(|(hash, name)| {
            (name == asset_name && hash.len() == 64).then(|| hash.to_ascii_lowercase())
        })
}

fn computed_sha256(path: &Path) -> Option<String> {
    let output = if cfg!(windows) {
        Command::new("certutil")
            .arg("-hashfile")
            .arg(path)
            .arg("SHA256")
            .output()
            .ok()?
    } else if cfg!(target_os = "macos") {
        Command::new("shasum")
            .arg("-a")
            .arg("256")
            .arg(path)
            .output()
            .ok()?
    } else {
        Command::new("sha256sum").arg(path).output().ok()?
    };
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .find(|part| part.len() == 64 && part.chars().all(|c| c.is_ascii_hexdigit()))
        .map(|part| part.to_ascii_lowercase())
}

fn hash_matches(path: &Path, expected_hash: &str) -> bool {
    computed_sha256(path).as_deref() == Some(expected_hash)
}

fn copy_prebuilt_asset(source: &Path, out_dir: &Path, local_name: &str) -> Option<PathBuf> {
    let cache_dir = out_dir.join("stim-rs-prebuilt");
    fs::create_dir_all(&cache_dir).ok()?;
    fs::copy(source, cache_dir.join(local_name)).ok()?;
    Some(cache_dir)
}

fn find_prebuilt_dir(target: &str, out_dir: &Path) -> Option<(PathBuf, &'static str)> {
    if env_is_true("STIM_RS_BUILD_FROM_SOURCE") || env_is_true("CARGO_NET_OFFLINE") {
        return None;
    }

    let (local_name, _link_name, extension) = prebuilt_names(target)?;
    if let Ok(dir) = env::var("STIM_RS_PREBUILT_DIR") {
        let dir = PathBuf::from(dir);
        if dir.join(local_name).exists() {
            return Some((dir, "prebuilt-dir"));
        }
        let asset = dir.join(release_asset_name(target, extension));
        if asset.exists() {
            return copy_prebuilt_asset(&asset, out_dir, local_name)
                .map(|dir| (dir, "prebuilt-dir"));
        }
    }

    let cache_dir = out_dir.join("stim-rs-prebuilt");
    let lib_path = cache_dir.join(local_name);
    let asset_name = release_asset_name(target, extension);
    let expected_hash = expected_sha256(&asset_name)?;
    if lib_path.exists() && hash_matches(&lib_path, &expected_hash) {
        return Some((cache_dir, "prebuilt"));
    }

    fs::create_dir_all(&cache_dir).ok()?;
    let url = release_asset_url(target, extension);
    let status = Command::new("curl")
        .arg("-fsSL")
        .arg("--retry")
        .arg("3")
        .arg("--connect-timeout")
        .arg("10")
        .arg("--max-time")
        .arg("120")
        .arg("-o")
        .arg(&lib_path)
        .arg(&url)
        .status();

    match status {
        Ok(status)
            if status.success() && lib_path.exists() && hash_matches(&lib_path, &expected_hash) =>
        {
            Some((cache_dir, "prebuilt"))
        }
        _ => {
            let _ = fs::remove_file(&lib_path);
            println!(
                "cargo:warning=failed to download or verify {url}; building bundled Stim source"
            );
            None
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=include/stim_rs_bridge.h");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/ffi/circuit.cc");
    println!("cargo:rerun-if-changed=src/ffi/smoke.cc");
    println!("cargo:rerun-if-env-changed=STIM_RS_VENDOR_DIR");
    println!("cargo:rerun-if-env-changed=STIM_RS_BUILD_FROM_SOURCE");
    println!("cargo:rerun-if-env-changed=STIM_RS_PREBUILT_DIR");
    println!("cargo:rerun-if-env-changed=STIM_RS_PREBUILT_BASE_URL");
    println!("cargo:rerun-if-env-changed=CARGO_NET_OFFLINE");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("missing OUT_DIR"));
    let target = env::var("TARGET").expect("missing TARGET");
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
    let prebuilt = if vendor_mode == "packaged" {
        find_prebuilt_dir(&target, &out_dir)
    } else {
        None
    };

    println!("cargo:rustc-env=STIM_RS_CXX_STANDARD=c++20");
    println!("cargo:rustc-env=STIM_RS_PINNED_STIM_COMMIT={PINNED_STIM_COMMIT}");
    println!("cargo:rustc-env=STIM_RS_TARGET={target}");
    println!(
        "cargo:rustc-env=STIM_RS_VENDOR_STIM_MODE={}",
        prebuilt
            .as_ref()
            .map(|(_, mode)| *mode)
            .unwrap_or(vendor_mode)
    );
    println!(
        "cargo:rustc-env=STIM_RS_VENDOR_STIM_DIR={}",
        vendor_dir.display()
    );

    let mut build = cxx_build::bridge("src/lib.rs");
    if prebuilt.is_none() {
        build.files(collect_vendor_sources(&vendor_dir));
    }
    build
        .file("src/ffi/circuit.cc")
        .file("src/ffi/smoke.cc")
        .std("c++20")
        .warnings(false)
        .include("include")
        .include(&vendor_dir)
        .include(&vendor_src_dir);
    build.define("STIM_RS_VENDOR_STIM_PRESENT", Some("1"));
    if prebuilt.is_some() {
        match prebuilt_simd_width(&target) {
            Some(128) => {
                if target.ends_with("windows-msvc") {
                    build.flag_if_supported("/arch:SSE2");
                } else {
                    build.flag_if_supported("-mno-avx2");
                    build.flag_if_supported("-msse2");
                }
            }
            Some(64) if !target.ends_with("windows-msvc") => {
                build.flag_if_supported("-mno-avx2");
                build.flag_if_supported("-mno-sse2");
            }
            _ => {}
        }
    }

    if let Some((dir, _mode)) = &prebuilt {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    build.compile("stim-cxx-bridge");
    if prebuilt.is_some() {
        let (_local_name, link_name, _extension) =
            prebuilt_names(&target).expect("prebuilt target changed during build");
        println!("cargo:rustc-link-lib=static={link_name}");
    }
}
