set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all --check

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

test:
    cargo test --workspace

test-doc:
    cargo test --doc -p stim

doc-check:
    env RUSTDOCFLAGS=-Dwarnings cargo doc -p stim --no-deps

verify: fmt-check clippy test test-doc doc-check

pre-commit-install:
    pre-commit install

pre-commit-run:
    pre-commit run --all-files

package-check:
    cargo package -p stim-cxx --allow-dirty --locked
    cargo package -p stim --allow-dirty --locked --no-verify --list > /tmp/stim-package-list.txt
