from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import urllib.request
from collections import Counter
from typing import Iterable


DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/quantumlib/Stim/main/doc/python_api_reference_vDev.md"
)

INDEX_ENTRY_RE = re.compile(r"^\s*-\s+\[`([^`]+)`\]\((#[^)]+)\)\s*$")

P0_EXACT = {
    "stim.Circuit.compile_detector_sampler",
    "stim.Circuit.compile_m2d_converter",
    "stim.Circuit.compile_sampler",
    "stim.Circuit.reference_sample",
    "stim.Circuit.search_for_undetectable_logical_errors",
    "stim.Circuit.shortest_error_sat_problem",
    "stim.Circuit.shortest_graphlike_error",
    "stim.CompiledDemSampler.sample",
    "stim.CompiledDemSampler.sample_write",
    "stim.CompiledDetectorSampler.sample",
    "stim.CompiledDetectorSampler.sample_write",
    "stim.CompiledMeasurementSampler.sample",
    "stim.CompiledMeasurementSampler.sample_write",
    "stim.CompiledMeasurementsToDetectionEventsConverter.convert",
    "stim.CompiledMeasurementsToDetectionEventsConverter.convert_file",
    "stim.DetectorErrorModel.compile_sampler",
    "stim.DetectorErrorModel.shortest_graphlike_error",
    "stim.FlipSimulator.generate_bernoulli_samples",
    "stim.read_shot_data_file",
    "stim.write_shot_data_file",
}

D2_EXACT = {
    "stim.Circuit.detector_error_model",
    "stim.Circuit.explain_detector_error_model_errors",
    "stim.Circuit.flow_generators",
    "stim.Circuit.likeliest_error_sat_problem",
    "stim.Circuit.missing_detectors",
    "stim.Circuit.solve_flow_measurements",
    "stim.Circuit.to_tableau",
    "stim.Circuit.with_inlined_feedback",
    "stim.DetectorErrorModel.compile_sampler",
    "stim.GateData.tableau",
    "stim.PauliString.from_ndarray",
    "stim.PauliString.from_unitary_matrix",
    "stim.PauliString.to_ndarray",
    "stim.PauliString.to_tableau",
    "stim.PauliString.to_unitary_matrix",
    "stim.Tableau.from_ndarray",
    "stim.Tableau.from_state_vector",
    "stim.Tableau.from_unitary_matrix",
    "stim.Tableau.to_ndarray",
    "stim.Tableau.to_state_vector",
    "stim.Tableau.to_unitary_matrix",
    "stim.TableauSimulator.current_measurement_record",
    "stim.TableauSimulator.set_state_from_state_vector",
    "stim.TableauSimulator.state_vector",
}

D1_KEYWORDS = (
    "append",
    "clear",
    "copy",
    "from_file",
    "get_",
    "insert",
    "iter_",
    "pop",
    "random",
    "to_file",
)


def compute_family(api_name: str) -> str:
    if api_name.count(".") >= 2:
        return ".".join(api_name.split(".")[:2])
    return "stim" if api_name.startswith("stim.") and api_name.count(".") == 1 else api_name


def proposed_rust_surface(api_name: str) -> str:
    parts = api_name.split(".")
    if len(parts) == 1:
        return api_name.replace(".", "::")
    if len(parts) == 2:
        if parts[1] == "__init__":
            return f"stim::{parts[1].replace('__init__', 'new')}"
        return f"stim::{parts[1]}"
    receiver = parts[1]
    member = parts[2]
    if member == "__init__":
        member = "new"
    return f"stim::{receiver}::{member}"


def verification_artifact_path(api_name: str) -> str:
    slug = api_name.replace("stim.", "stim_").replace(".", "_")
    return f"tests/parity/{slug}.rs"


def classify_api(api_name: str) -> dict[str, str]:
    if api_name in P0_EXACT or api_name.endswith(".sample") or api_name.endswith(".sample_write"):
        return {
            "taxonomy_class": "P0",
            "behavioral_oracle": "composite: python-runtime + c++ upstream + benchmark harness",
            "benchmark_tier": "release-gate",
            "rationale": "Hot-path sampling, conversion, or decoding work where wrapper overhead can violate the Python parity target.",
        }

    if api_name in D2_EXACT or any(token in api_name for token in ("numpy", "state_vector", "unitary_matrix")):
        return {
            "taxonomy_class": "D2",
            "behavioral_oracle": "composite: python-runtime + golden fixture",
            "benchmark_tier": "scheduled-perf",
            "rationale": "Cross-language ownership, buffer, or unsafe bridge details must stay encapsulated behind a safe Rust API.",
        }

    if api_name.startswith("stim.TableauSimulator.") or api_name.startswith("stim.FlipSimulator."):
        return {
            "taxonomy_class": "P0",
            "behavioral_oracle": "python-runtime oracle",
            "benchmark_tier": "release-gate",
            "rationale": "Simulator methods are performance sensitive and require benchmark closure against Python.",
        }

    if api_name.count(".") >= 2:
        member = api_name.split(".")[2]
        if member.startswith("__") and member.endswith("__"):
            return {
                "taxonomy_class": "D1",
                "behavioral_oracle": "python-doc semantics",
                "benchmark_tier": "none",
                "rationale": "Python dunder behavior will likely map to Rust traits or inherent methods instead of identical syntax.",
            }

        if member in {"append", "append_from_stim_program_text", "insert", "pop"} or member.startswith(D1_KEYWORDS):
            return {
                "taxonomy_class": "D1",
                "behavioral_oracle": "python-runtime oracle",
                "benchmark_tier": "pr-smoke",
                "rationale": "Idiomatic Rust ownership and iterator shapes will differ from Python while preserving semantics.",
            }

    return {
        "taxonomy_class": "D0",
        "behavioral_oracle": "python-doc semantics",
        "benchmark_tier": "none",
        "rationale": "Directly representable through a safe Rust wrapper without semantic reshaping beyond normal signature translation.",
    }


def extract_api_entries(document: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    in_index = False

    for raw_line in document.splitlines():
        line = raw_line.rstrip()
        if line == "## Index":
            in_index = True
            continue
        if in_index and line.startswith("# "):
            break
        if not in_index:
            continue

        match = INDEX_ENTRY_RE.match(line)
        if not match:
            continue

        api_name, doc_anchor = match.groups()
        if api_name in seen:
            continue
        seen.add(api_name)
        entries.append(
            {
                "family": compute_family(api_name),
                "api_name": api_name,
                "doc_anchor": doc_anchor,
                "proposed_rust_surface": proposed_rust_surface(api_name),
                "verification_artifact_path": verification_artifact_path(api_name),
                "status": "planned",
            }
        )
    return entries


def annotate_entries(entries: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    annotated: list[dict[str, str]] = []
    for entry in entries:
        combined = dict(entry)
        combined.update(classify_api(entry["api_name"]))
        annotated.append(combined)
    return annotated


def render_metadata(
    *,
    entries: list[dict[str, str]],
    source_url: str,
    stim_commit: str,
    doc_sha256: str,
) -> dict[str, object]:
    taxonomy_counts = Counter(entry["taxonomy_class"] for entry in entries)
    benchmark_counts = Counter(entry["benchmark_tier"] for entry in entries)
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_url": source_url,
        "stim_commit": stim_commit,
        "doc_sha256": doc_sha256,
        "api_count": len(entries),
        "taxonomy_counts": dict(sorted(taxonomy_counts.items())),
        "benchmark_tier_counts": dict(sorted(benchmark_counts.items())),
    }


def build_drift_report(
    *,
    expected_stim_commit: str,
    current_stim_commit: str,
    expected_doc_sha256: str,
    current_doc_sha256: str,
) -> dict[str, object]:
    stim_commit_changed = expected_stim_commit != current_stim_commit
    doc_sha256_changed = expected_doc_sha256 != current_doc_sha256
    return {
        "status": "drifted" if (stim_commit_changed or doc_sha256_changed) else "in_sync",
        "expected_stim_commit": expected_stim_commit,
        "current_stim_commit": current_stim_commit,
        "stim_commit_changed": stim_commit_changed,
        "expected_doc_sha256": expected_doc_sha256,
        "current_doc_sha256": current_doc_sha256,
        "doc_sha256_changed": doc_sha256_changed,
    }


def render_markdown(entries: list[dict[str, str]], metadata: dict[str, object]) -> str:
    lines = [
        "# Stim API Inventory (Generated)",
        "",
        f"- Source URL: `{metadata['source_url']}`",
        f"- Stim commit: `{metadata['stim_commit']}`",
        f"- Doc SHA256: `{metadata['doc_sha256']}`",
        f"- API count: `{metadata['api_count']}`",
        "",
        "## Taxonomy Summary",
        "",
    ]
    for key, value in metadata["taxonomy_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Benchmark Tier Summary",
            "",
        ]
    )
    for key, value in metadata["benchmark_tier_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Inventory",
            "",
            "| Family | Python API | Doc Anchor | Rust Surface | Taxonomy | Oracle | Benchmark Tier | Verification Artifact | Status | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in entries:
        lines.append(
            "| {family} | {api_name} | {doc_anchor} | `{proposed_rust_surface}` | {taxonomy_class} | {behavioral_oracle} | {benchmark_tier} | `{verification_artifact_path}` | {status} | {rationale} |".format(
                **entry
            )
        )
    lines.append("")
    return "\n".join(lines)


def load_document(source_url: str, input_file: str | None) -> tuple[str, str]:
    if input_file:
        return pathlib.Path(input_file).read_text(encoding="utf-8"), f"file://{input_file}"
    with urllib.request.urlopen(source_url) as response:
        return response.read().decode("utf-8"), source_url


def write_json(path: str | None, payload: object) -> None:
    if not path:
        return
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: str | None, payload: str) -> None:
    if not path:
        return
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Stim API inventory/parity seed from the upstream Python API reference."
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--input-file")
    parser.add_argument("--stim-commit", default="unknown")
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    parser.add_argument("--metadata-out")
    parser.add_argument("--drift-out")
    parser.add_argument("--expected-stim-commit")
    parser.add_argument("--expected-doc-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document, resolved_source = load_document(args.source_url, args.input_file)
    doc_sha256 = hashlib.sha256(document.encode("utf-8")).hexdigest()
    entries = annotate_entries(extract_api_entries(document))
    metadata = render_metadata(
        entries=entries,
        source_url=resolved_source,
        stim_commit=args.stim_commit,
        doc_sha256=doc_sha256,
    )
    write_json(args.json_out, entries)
    write_json(args.metadata_out, metadata)
    write_text(args.markdown_out, render_markdown(entries, metadata))
    if args.expected_stim_commit or args.expected_doc_sha256:
        write_json(
            args.drift_out,
            build_drift_report(
                expected_stim_commit=args.expected_stim_commit or args.stim_commit,
                current_stim_commit=args.stim_commit,
                expected_doc_sha256=args.expected_doc_sha256 or doc_sha256,
                current_doc_sha256=doc_sha256,
            ),
        )
    print(
        json.dumps(
            {
                "api_count": metadata["api_count"],
                "stim_commit": metadata["stim_commit"],
                "doc_sha256": metadata["doc_sha256"],
                "taxonomy_counts": metadata["taxonomy_counts"],
                "benchmark_tier_counts": metadata["benchmark_tier_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
