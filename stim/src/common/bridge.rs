/// Returns the pinned upstream Stim commit recorded by the bridge layer.
pub fn upstream_commit() -> &'static str {
    stim_cxx::pinned_stim_commit()
}
