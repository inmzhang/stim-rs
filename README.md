# stim-rs

[![crates.io](https://img.shields.io/crates/v/stim.svg)](https://crates.io/crates/stim)
[![docs.rs](https://docs.rs/stim/badge.svg)](https://docs.rs/stim)

`stim-rs` provides Rust bindings for [Stim](https://github.com/quantumlib/Stim), a high-performance stabilizer circuit simulator and analyzer widely used in quantum error-correction workflows.

It consists of two crates:

* `stim-cxx`: a low-level C++ bridge to `stim` built with `cxx`
* `stim`: a safe Rust API built on top of `stim-cxx`, intended to be fully feature-aligned with the upstream [Python API](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md)

The Rust API uses `ndarray` to provide an experience similar to working with NumPy in Python.

## Usage

In most cases, users should depend only on the `stim` crate:

```shell
cargo add stim
```

## Contributing

Issues and pull requests are very welcome, whether they relate to performance, ergonomics, or bugs.
