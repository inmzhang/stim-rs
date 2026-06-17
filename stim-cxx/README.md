# stim-cxx

Low-level vendored C++ bridge crate for `stim`.

By default, release builds try to download a matching prebuilt static Stim library
from the `stim-rs` GitHub release, then fall back to the packaged C++ source.

Set `STIM_RS_BUILD_FROM_SOURCE=1` to skip prebuilt downloads. Set
`STIM_RS_PREBUILT_DIR=/path/to/libs` to use a local `libstim.a` or `libstim.lib`.
