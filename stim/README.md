# stim

Safe Rust bindings for Stim, a high-performance stabilizer circuit simulator and analyzer.

- crates.io: <https://crates.io/crates/stim>
- docs.rs: <https://docs.rs/stim>

This crate is the safe public Rust surface. The low-level native bridge lives in the companion `stim-cxx` crate.

## Rust-only utilities

Beyond the direct safe wrapper surface, this crate also ships a pure-Rust noise
model layer:

- `stim::UniformDepolarizing` and `stim::Si1000` provide ready-made presets for
  common stabilizer-noise workflows.
- `stim::Circuit::with_noise(...)` applies one of those pure-Rust noise models
  to an existing circuit.
