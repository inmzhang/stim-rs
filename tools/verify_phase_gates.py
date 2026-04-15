from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any


PHASE_0A_REQUIRED_FILES = [
    "prd-stim-rs-rust-binding.md",
    "test-spec-stim-rs-rust-binding.md",
    "stim-rs-approval-ready-ralplan-draft.md",
    "stim-rs-api-inventory-parity-matrix.md",
    "stim-rs-representability-rubric.md",
    "stim-rs-benchmark-plan.md",
    "stim-rs-build-policy.md",
    "stim-rs-submodule-drift-policy.md",
    "stim-rs-adr-layered-workspace.md",
    "stim-rs-risk-register.md",
]

PHASE_0B_REQUIRED_PATHS = [
    "Cargo.toml",
    ".gitmodules",
    "crates/stim-cxx/Cargo.toml",
    "crates/stim-cxx/src/lib.rs",
    "crates/stim/Cargo.toml",
    "crates/stim/src/lib.rs",
    "vendor/stim/.git",
    "vendor/stim/doc/python_api_reference_vDev.md",
]

PHASE_0B_REQUIRED_WORKSPACE_MEMBERS = [
    "crates/stim-cxx",
    "crates/stim",
]

PHASE_0A_REQUIRED_ARTIFACTS = [
    "artifacts/stim-phase0a/api-inventory.json",
    "artifacts/stim-phase0a/api-inventory.md",
    "artifacts/stim-phase0a/doc-metadata.json",
    "artifacts/stim-phase0a/drift-report.json",
]

ALLOWED_TAXONOMY_CLASSES = {"D0", "D1", "D2", "P0", "I1", "I2"}
ALLOWED_BENCHMARK_TIERS = {"none", "pr-smoke", "scheduled-perf", "release-gate"}


def _existing_and_missing(base_dir: Path, required_paths: list[str]) -> tuple[list[str], list[str]]:
    existing: list[str] = []
    missing: list[str] = []

    for relative_path in required_paths:
        if (base_dir / relative_path).exists():
            existing.append(relative_path)
        else:
            missing.append(relative_path)

    return existing, missing


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_phase_0a_inventory_status(repo_root: Path) -> dict[str, Any]:
    existing, missing = _existing_and_missing(repo_root, PHASE_0A_REQUIRED_ARTIFACTS)
    status: dict[str, Any] = {
        "passed": not missing,
        "existing_artifacts": existing,
        "missing_artifacts": missing,
        "api_count": 0,
        "missing_behavioral_oracles": [],
        "missing_benchmark_tiers": [],
        "unknown_taxonomy_classes": [],
        "u0_entries": [],
        "metadata_api_count_matches": False,
    }

    if missing:
        return status

    entries = _load_json(repo_root / "artifacts/stim-phase0a/api-inventory.json")
    metadata = _load_json(repo_root / "artifacts/stim-phase0a/doc-metadata.json")

    missing_behavioral_oracles = [
        entry["api_name"] for entry in entries if not entry.get("behavioral_oracle")
    ]
    missing_benchmark_tiers = [
        entry["api_name"]
        for entry in entries
        if entry.get("benchmark_tier") not in ALLOWED_BENCHMARK_TIERS
    ]
    unknown_taxonomy_classes = [
        entry["api_name"]
        for entry in entries
        if entry.get("taxonomy_class") not in ALLOWED_TAXONOMY_CLASSES
    ]
    u0_entries = [entry["api_name"] for entry in entries if entry.get("taxonomy_class") == "U0"]
    metadata_api_count_matches = metadata.get("api_count") == len(entries)

    status.update(
        {
            "api_count": len(entries),
            "missing_behavioral_oracles": missing_behavioral_oracles,
            "missing_benchmark_tiers": missing_benchmark_tiers,
            "unknown_taxonomy_classes": unknown_taxonomy_classes,
            "u0_entries": u0_entries,
            "metadata_api_count_matches": metadata_api_count_matches,
            "passed": not (
                missing
                or missing_behavioral_oracles
                or missing_benchmark_tiers
                or unknown_taxonomy_classes
                or u0_entries
                or not metadata_api_count_matches
            ),
        }
    )
    return status


def collect_phase_0a_status(repo_root: Path) -> dict[str, Any]:
    plans_dir = repo_root / ".omx" / "plans"
    existing, missing = _existing_and_missing(plans_dir, PHASE_0A_REQUIRED_FILES)
    inventory = _collect_phase_0a_inventory_status(repo_root)
    return {
        "phase": "0A",
        "passed": not missing and inventory["passed"],
        "plans_dir": str(plans_dir),
        "existing_files": existing,
        "missing_files": missing,
        "inventory": inventory,
    }


def _collect_phase_0b_workspace_status(repo_root: Path) -> dict[str, Any]:
    manifest_path = repo_root / "Cargo.toml"
    if not manifest_path.exists():
        return {
            "passed": False,
            "manifest_path": str(manifest_path),
            "declared_members": [],
            "missing_members": PHASE_0B_REQUIRED_WORKSPACE_MEMBERS.copy(),
            "parse_error": "missing Cargo.toml",
        }

    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        return {
            "passed": False,
            "manifest_path": str(manifest_path),
            "declared_members": [],
            "missing_members": PHASE_0B_REQUIRED_WORKSPACE_MEMBERS.copy(),
            "parse_error": str(error),
        }

    workspace = manifest.get("workspace")
    declared_members = workspace.get("members", []) if isinstance(workspace, dict) else []
    missing_members = [
        member for member in PHASE_0B_REQUIRED_WORKSPACE_MEMBERS if member not in declared_members
    ]

    return {
        "passed": isinstance(workspace, dict) and not missing_members,
        "manifest_path": str(manifest_path),
        "declared_members": declared_members,
        "missing_members": missing_members,
        "parse_error": None,
    }


def collect_phase_0b_status(repo_root: Path) -> dict[str, Any]:
    existing, missing = _existing_and_missing(repo_root, PHASE_0B_REQUIRED_PATHS)
    workspace = _collect_phase_0b_workspace_status(repo_root)
    return {
        "phase": "0B",
        "passed": not missing and workspace["passed"],
        "repo_root": str(repo_root),
        "existing_paths": existing,
        "missing_paths": missing,
        "workspace": workspace,
    }


def build_report(repo_root: Path) -> dict[str, Any]:
    phase_0a = collect_phase_0a_status(repo_root)
    phase_0b = collect_phase_0b_status(repo_root)
    return {
        "repo_root": str(repo_root),
        "overall_passed": phase_0a["passed"] and phase_0b["passed"],
        "phases": [phase_0a, phase_0b],
    }


def _format_text(report: dict[str, Any]) -> str:
    lines = [f"repo: {report['repo_root']}", f"overall_passed: {report['overall_passed']}"]
    for phase in report["phases"]:
        lines.append(f"{phase['phase']}: {'PASS' if phase['passed'] else 'FAIL'}")
        if phase["phase"] == "0A":
            lines.append(f"  existing_files: {', '.join(phase['existing_files']) or '(none)'}")
            lines.append(f"  missing_files: {', '.join(phase['missing_files']) or '(none)'}")
            inventory = phase["inventory"]
            lines.append(
                f"  inventory_artifacts: {', '.join(inventory['existing_artifacts']) or '(none)'}"
            )
            if inventory["missing_artifacts"]:
                lines.append(
                    f"  missing_inventory_artifacts: {', '.join(inventory['missing_artifacts'])}"
                )
            lines.append(f"  api_count: {inventory['api_count']}")
            lines.append(
                "  metadata_api_count_matches: "
                + ("yes" if inventory["metadata_api_count_matches"] else "no")
            )
            if inventory["u0_entries"]:
                lines.append(f"  u0_entries: {', '.join(inventory['u0_entries'])}")
            if inventory["unknown_taxonomy_classes"]:
                lines.append(
                    "  unknown_taxonomy_classes: "
                    + ", ".join(inventory["unknown_taxonomy_classes"])
                )
            if inventory["missing_behavioral_oracles"]:
                lines.append(
                    "  missing_behavioral_oracles: "
                    + ", ".join(inventory["missing_behavioral_oracles"])
                )
            if inventory["missing_benchmark_tiers"]:
                lines.append(
                    "  missing_benchmark_tiers: "
                    + ", ".join(inventory["missing_benchmark_tiers"])
                )
        else:
            lines.append(f"  existing_paths: {', '.join(phase['existing_paths']) or '(none)'}")
            lines.append(f"  missing_paths: {', '.join(phase['missing_paths']) or '(none)'}")
            workspace = phase["workspace"]
            lines.append(
                "  workspace_members: "
                + (", ".join(workspace["declared_members"]) or "(none)")
            )
            if workspace["missing_members"]:
                lines.append(
                    "  missing_workspace_members: "
                    + ", ".join(workspace["missing_members"])
                )
            if workspace["parse_error"]:
                lines.append(f"  workspace_parse_error: {workspace['parse_error']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Stim Rust binding phase-gate artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root to inspect.")
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format.",
    )
    args = parser.parse_args()

    report = build_report(args.repo_root.resolve())

    if args.format == "text":
        print(_format_text(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
