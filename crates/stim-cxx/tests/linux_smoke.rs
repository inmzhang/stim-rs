#![cfg(target_os = "linux")]

use stim_cxx::{SmokeProbe, build_metadata};

#[test]
fn linux_smoke_reports_bridge_state_and_handles_batched_input() {
    let metadata = build_metadata();
    assert_eq!(metadata.crate_name, "stim-cxx");
    assert_eq!(metadata.cxx_standard, "c++20");
    assert!(matches!(
        metadata.vendor_stim_mode,
        "override" | "packaged" | "workspace"
    ));

    let probe = SmokeProbe::new();
    let description = probe.describe();
    assert!(
        description.contains("stim-cxx bridge smoke"),
        "unexpected description: {description}"
    );
    assert!(
        description.contains("linux"),
        "unexpected description: {description}"
    );
    assert_eq!(probe.weighted_checksum(&[1, 2, 3, 4]), 30);
}
