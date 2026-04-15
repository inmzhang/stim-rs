# stim-rs

[![crates.io](https://img.shields.io/crates/v/stim.svg)](https://crates.io/crates/stim)
[![docs.rs](https://docs.rs/stim/badge.svg)](https://docs.rs/stim)

`stim-rs` provides Rust bindings for [Stim](https://github.com/quantumlib/Stim), a high-performance stabilizer circuit simulator and analyzer used heavily in quantum error-correction workflows.

## Crates

- `stim`: safe, idiomatic Rust API
- `stim-cxx`: low-level C++ bridge crate used by `stim`

## Scope

`stim-rs` targets a safe, idiomatic Rust API for Stim while staying feature-aligned with the upstream Python API. The goal is practical parity with the Python surface for circuit construction, detector error models, simulation, analysis, and stabilizer/tableau utilities.

Where the Python API exposes NumPy-shaped data, the Rust API uses `ndarray` as the native matrix/vector representation instead of mirroring Python-specific NumPy types directly. This keeps the Rust surface idiomatic while preserving the same conceptual capabilities.
