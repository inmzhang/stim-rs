# stim-rs

[![crates.io](https://img.shields.io/crates/v/stim.svg)](https://crates.io/crates/stim)
[![docs.rs](https://docs.rs/stim/badge.svg)](https://docs.rs/stim)
[![CI](https://github.com/inmzhang/stim-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/inmzhang/stim-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/inmzhang/stim-rs/graph/badge.svg)](https://codecov.io/gh/inmzhang/stim-rs)

`stim-rs` provides Rust bindings for [Stim](https://github.com/quantumlib/Stim), a high-performance stabilizer circuit simulator and analyzer widely used in quantum error-correction workflows.

It consists of two crates:

* `stim-cxx`: a low-level C++ bridge to `stim` built with `cxx`
* `stim`: a safe Rust API built on top of `stim-cxx`, intended to be fully feature-aligned with the upstream [Python API](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md)

The Rust API uses `ndarray` to provide an experience similar to working with NumPy in Python.

## Rust-only features

The `stim` crate also includes a small layer of Rust-native utilities that do not
come from upstream Stim directly:

* `stim::UniformDepolarizing` and `stim::Si1000`: ready-made pure-Rust noise models
* `stim::Circuit::with_noise(...)`: a convenience method for applying those models
  to an existing circuit

This keeps the low-level bridge minimal while still allowing Rust-specific
ergonomic features to grow independently when they can be implemented safely on
top of the existing binding surface.

## Usage

In most cases, users should depend only on the `stim` crate:

```shell
cargo add stim
```

## Contributing

Issues and pull requests are very welcome, whether they relate to performance, ergonomics, or bugs.

### Pre-commit

This repo ships a committed [pre-commit](https://pre-commit.com/) configuration.
Install it locally with:

```bash
pre-commit install
```

Run the full hook suite manually with:

```bash
pre-commit run --all-files
```

Common local workflows are also exposed through the repo `justfile`:

```bash
just verify
just pre-commit-run
```

The configured hooks run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --doc -p stim
env RUSTDOCFLAGS=-Dwarnings cargo doc -p stim --no-deps
```

## License

Licensed under Apache-2.0. The vendored upstream Stim sources under
`crates/stim-cxx/vendor/stim` are also distributed under Apache-2.0.

### Updating the vendored Stim C++ source

The upstream Stim C++ library is vendored as a git submodule at
`crates/stim-cxx/vendor/stim`. To update it:

```bash
git -C crates/stim-cxx/vendor/stim fetch origin
git -C crates/stim-cxx/vendor/stim checkout <commit-or-tag>
# Update STIM_RS_PINNED_STIM_COMMIT in crates/stim-cxx/build.rs to match.
cargo nextest run -p stim && cargo test --doc -p stim
git add crates/stim-cxx/vendor/stim crates/stim-cxx/build.rs
```
