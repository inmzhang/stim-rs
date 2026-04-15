from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PRD_PATH = Path(".omx/plans/prd-stim-rs-rust-binding.md")

REQUIRED_SECTION_HEADINGS = [
    "## Metadata",
    "## Problem",
    "## Goals",
    "## Non-goals",
    "## Key Evidence",
    "## Decision",
    "## Alternatives Considered",
    "## Architectural Principles",
    "## Required Planning Artifacts",
    "## Architectural Boundary Policy",
    "## Canonical Execution Sequence",
    "## Risks and Mitigations",
    "## Deliberate-Mode Pre-Mortem",
    "## ADR",
]

REQUIRED_PHASE_HEADINGS = [
    "### Phase 0A —",
    "### Phase 0B —",
    "### Phase 1 —",
    "### Phase 2 —",
    "### Phase 3 —",
    "### Phase 4 —",
]

REQUIRED_ADR_SUBSECTIONS = [
    "### Decision",
    "### Drivers",
    "### Why chosen",
    "### Consequences",
    "### Follow-ups",
]

REQUIRED_ARTIFACTS = [
    ".omx/plans/prd-stim-rs-rust-binding.md",
    ".omx/plans/test-spec-stim-rs-rust-binding.md",
    ".omx/plans/stim-rs-approval-ready-ralplan-draft.md",
    ".omx/plans/stim-rs-api-inventory-parity-matrix.md",
    ".omx/plans/stim-rs-representability-rubric.md",
    ".omx/plans/stim-rs-benchmark-plan.md",
    ".omx/plans/stim-rs-build-policy.md",
    ".omx/plans/stim-rs-submodule-drift-policy.md",
    ".omx/plans/stim-rs-adr-layered-workspace.md",
    ".omx/plans/stim-rs-risk-register.md",
]

REQUIRED_BOUNDARY_POLICY_ITEMS = {
    "opaque_boundary": "when a Stim type remains opaque across the public boundary,",
    "cxx_shim_preference": "when a C++ shim is preferred over exposing awkward raw bridge types,",
    "copy_materialization": "when copies/materialization are acceptable versus when borrowed or batched APIs are required,",
    "iterator_reshaping": "how iterator-like Python APIs are reshaped into Rust iterators/collections,",
    "exception_translation": "how upstream exceptions are translated into Rust errors,",
    "p0_escalation": "how performance-sensitive boundary cases are identified and escalated into `P0` benchmark gating.",
}

REQUIRED_GATE_CHECKS = {
    "phase_0a_all_api_rows": "every documented Python API has a matrix row,",
    "phase_0a_no_u0": "no row remains `U0`,",
    "phase_0a_oracles_and_tiers": "every row has a behavioral oracle and benchmark-tier assignment,",
    "phase_0a_non_d0_rationale": "every non-`D0` row has rationale,",
    "phase_0a_build_policy": "the build-policy artifact exists,",
    "phase_0a_drift_policy": "the submodule drift-policy artifact exists.",
    "phase_0b_workspace": "the minimal layered workspace exists,",
    "phase_0b_pinned_submodule": "the pinned submodule is checked in with recorded SHA/doc hash,",
    "phase_0b_tri_platform": "the minimal bridge plus risky exemplars pass on Linux/macOS/Windows,",
    "phase_0b_initial_benchmarks": "initial benchmark artifacts exist for bridge-only and representative end-to-end paths,",
    "phase_0b_build_policy_reference": "cross-platform build proof references the actual build-policy decisions.",
}


def _find_missing_headings(text: str, required_headings: list[str]) -> list[str]:
    lines = {line.strip() for line in text.splitlines()}
    missing: list[str] = []
    for heading in required_headings:
        if heading.endswith("—"):
            if not any(line.startswith(heading) for line in lines):
                missing.append(heading)
        elif heading not in lines:
            missing.append(heading)
    return missing


def collect_prd_status(prd_path: Path) -> dict[str, Any]:
    prd_path = prd_path.resolve()
    if not prd_path.exists():
        return {
            "passed": False,
            "prd_path": str(prd_path),
            "error": "missing prd file",
            "missing_sections": REQUIRED_SECTION_HEADINGS.copy(),
            "missing_phases": REQUIRED_PHASE_HEADINGS.copy(),
            "missing_artifacts": REQUIRED_ARTIFACTS.copy(),
            "missing_boundary_policy_items": list(REQUIRED_BOUNDARY_POLICY_ITEMS),
            "missing_adr_subsections": REQUIRED_ADR_SUBSECTIONS.copy(),
            "missing_gate_checks": list(REQUIRED_GATE_CHECKS),
        }

    text = prd_path.read_text(encoding="utf-8")

    missing_sections = _find_missing_headings(text, REQUIRED_SECTION_HEADINGS)
    missing_phases = _find_missing_headings(text, REQUIRED_PHASE_HEADINGS)
    missing_artifacts = [artifact for artifact in REQUIRED_ARTIFACTS if artifact not in text]
    missing_boundary_policy_items = [
        label for label, snippet in REQUIRED_BOUNDARY_POLICY_ITEMS.items() if snippet not in text
    ]
    missing_adr_subsections = _find_missing_headings(text, REQUIRED_ADR_SUBSECTIONS)
    missing_gate_checks = [
        label for label, snippet in REQUIRED_GATE_CHECKS.items() if snippet not in text
    ]

    return {
        "passed": not (
            missing_sections
            or missing_phases
            or missing_artifacts
            or missing_boundary_policy_items
            or missing_adr_subsections
            or missing_gate_checks
        ),
        "prd_path": str(prd_path),
        "error": None,
        "missing_sections": missing_sections,
        "missing_phases": missing_phases,
        "missing_artifacts": missing_artifacts,
        "missing_boundary_policy_items": missing_boundary_policy_items,
        "missing_adr_subsections": missing_adr_subsections,
        "missing_gate_checks": missing_gate_checks,
    }


def _format_text(status: dict[str, Any]) -> str:
    lines = [f"prd: {status['prd_path']}", f"passed: {status['passed']}"]
    if status["error"]:
        lines.append(f"error: {status['error']}")
    lines.append(f"missing_sections: {', '.join(status['missing_sections']) or '(none)'}")
    lines.append(f"missing_phases: {', '.join(status['missing_phases']) or '(none)'}")
    lines.append(f"missing_artifacts: {', '.join(status['missing_artifacts']) or '(none)'}")
    lines.append(
        "missing_boundary_policy_items: "
        + (", ".join(status["missing_boundary_policy_items"]) or "(none)")
    )
    lines.append(
        "missing_adr_subsections: "
        + (", ".join(status["missing_adr_subsections"]) or "(none)")
    )
    lines.append(
        "missing_gate_checks: "
        + (", ".join(status["missing_gate_checks"]) or "(none)")
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Stim Rust binding PRD structure and references.")
    parser.add_argument(
        "--prd-path",
        type=Path,
        default=DEFAULT_PRD_PATH,
        help="PRD markdown file to inspect.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format.",
    )
    args = parser.parse_args()

    status = collect_prd_status(args.prd_path)
    if args.format == "text":
        print(_format_text(status))
    else:
        print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
