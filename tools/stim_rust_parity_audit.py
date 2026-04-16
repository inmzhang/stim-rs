from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict
from typing import Iterable

TYPE_DEF_RE = re.compile(r"^pub\s+(?:struct|enum)\s+([A-Z][A-Za-z0-9_]*)", re.MULTILINE)
DERIVE_TYPE_RE = re.compile(
    r"#\[derive\(([^)]*)\)\](?:\s|///[^\n]*\n|//![^\n]*\n|#\[[^\]]+\]\s*)*pub\s+(?:struct|enum)\s+([A-Z][A-Za-z0-9_]*)",
    re.MULTILINE,
)
TYPE_ALIAS_RE = re.compile(r"^pub\s+type\s+([A-Z][A-Za-z0-9_]*)\s*=\s*([A-Z][A-Za-z0-9_]*)\s*;", re.MULTILINE)
IMPL_RE = re.compile(r"impl(?:<[^>]+>)?\s+([A-Z][A-Za-z0-9_]*)\s*\{", re.MULTILINE)
PUB_FN_RE = re.compile(r"^\s*pub\s+(?:const\s+)?fn\s+(?:r#)?([a-zA-Z_][A-Za-z0-9_]*)\s*(?:<[^>]*>)?\s*\(", re.MULTILINE)
TRAIT_IMPL_RE = re.compile(r"impl(?:<[^>]+>)?\s+([A-Za-z0-9_:<>]+)\s+for\s+([A-Z][A-Za-z0-9_]*)", re.MULTILINE)

DUnder_MAP = {
    "__init__": "new",
    "__getitem__": "get",
    "__setitem__": "set",
    "__iter__": "Iterator",
    "__next__": "Iterator",
    "__call__": "call",
    "__len__": "len",
    "__str__": "Display",
    "__repr__": "Debug",
    "__eq__": "PartialEq",
    "__ne__": "PartialEq",
    "__neg__": "Neg",
    "__add__": "Add",
    "__iadd__": "AddAssign",
    "__mul__": "Mul",
    "__imul__": "MulAssign",
    "__rmul__": "Rmul",
    "__pow__": "pow",
    "__ipow__": "ipow",
    "__pos__": "pos",
    "__truediv__": "div_complex_unit",
    "__itruediv__": "div_assign_complex_unit",
}


def split_impl_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for match in IMPL_RE.finditer(text):
        type_name = match.group(1)
        brace_start = text.find("{", match.start())
        if brace_start == -1:
            continue
        depth = 0
        for idx in range(brace_start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blocks.append((type_name, text[brace_start + 1 : idx]))
                    break
    return blocks


def impl_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in IMPL_RE.finditer(text):
        brace_start = text.find("{", match.start())
        if brace_start == -1:
            continue
        depth = 0
        for idx in range(brace_start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), idx + 1))
                    break
    return spans


def strip_ranges(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    parts = []
    cursor = 0
    for start, end in sorted(spans):
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def collect_surface_from_source(root: pathlib.Path) -> set[str]:
    surfaces: set[str] = set()
    trait_impls: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, str] = {}
    for path in root.rglob("*.rs"):
        text = path.read_text(encoding="utf-8")
        for type_name in TYPE_DEF_RE.findall(text):
            surfaces.add(f"stim::{type_name}")
        for derives, type_name in DERIVE_TYPE_RE.findall(text):
            for derive in [part.strip().split("::")[-1] for part in derives.split(",")]:
                if derive:
                    trait_impls[type_name].add(derive)
        for alias, target in TYPE_ALIAS_RE.findall(text):
            aliases[alias] = target
            surfaces.add(f"stim::{alias}")
        for trait_name, type_name in TRAIT_IMPL_RE.findall(text):
            trait_impls[type_name].add(trait_name.split("::")[-1].split("<")[0])
        for type_name, body in split_impl_blocks(text):
            for fn_name in PUB_FN_RE.findall(body):
                surfaces.add(f"stim::{type_name}::{fn_name}")
        free_text = strip_ranges(text, impl_spans(text))
        for fn_name in PUB_FN_RE.findall(free_text):
            surfaces.add(f"stim::{fn_name}")

    for type_name, traits in trait_impls.items():
        for trait in traits:
            surfaces.add(f"stim::{type_name}::[{trait}]")
        if "Mul" in traits:
            surfaces.add(f"stim::{type_name}::[Rmul]")

    alias_surfaces: set[str] = set()
    for alias, target in aliases.items():
        target_prefix = f"stim::{target}"
        alias_prefix = f"stim::{alias}"
        for surface in surfaces:
            if surface == target_prefix:
                continue
            if surface.startswith(target_prefix):
                alias_surfaces.add(alias_prefix + surface[len(target_prefix):])
    surfaces |= alias_surfaces
    return surfaces


def inventory_rows(path: pathlib.Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_surface(surface: str) -> set[str]:
    out = {surface}
    parts = surface.split("::")
    if len(parts) == 3 and parts[2] == "to_numpy":
        out.add(f"stim::{parts[1]}::to_ndarray")
    if len(parts) == 3 and parts[2] == "from_numpy":
        out.add(f"stim::{parts[1]}::from_ndarray")
    if len(parts) == 3 and parts[2] in DUnder_MAP:
        mapped = DUnder_MAP[parts[2]]
        if mapped in {"Display", "Debug", "PartialEq", "Neg", "Add", "AddAssign", "Mul", "MulAssign", "Rmul", "Iterator"}:
            out.add(f"stim::{parts[1]}::[{mapped}]")
        else:
            out.add(f"stim::{parts[1]}::{mapped}")
    if surface.endswith("::type"):
        out.add(surface.replace("::type", "::r#type"))
    return out


def audit_inventory(rows: Iterable[dict[str, str]], available: set[str]) -> dict[str, object]:
    missing = []
    covered = 0
    for row in rows:
        if row["api_name"] == "stim.main":
            continue
        candidates = normalize_surface(row["proposed_rust_surface"])
        if candidates & available:
            covered += 1
        else:
            missing.append(row)
    return {
        "total_rows": covered + len(missing),
        "covered_rows": covered,
        "missing_rows": len(missing),
        "missing": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic post-refactor Rust parity audit.")
    parser.add_argument("--src-root", default="crates/stim/src")
    parser.add_argument("--inventory-json", default="artifacts/stim-phase0a/api-inventory.json")
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def render_markdown(report: dict[str, object], limit: int) -> str:
    lines = [
        "# Stim Rust Parity Audit (Heuristic)",
        "",
        f"- Total rows: `{report['total_rows']}`",
        f"- Covered rows: `{report['covered_rows']}`",
        f"- Missing rows: `{report['missing_rows']}`",
        "",
        "This audit is heuristic and conservative. Trait-based and reshaped APIs are matched through a small normalization table.",
        "",
        "## Missing rows (first slice)",
        "",
        "| Python API | Rust Surface | Taxonomy | Status |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["missing"][:limit]:
        lines.append(
            f"| {row['api_name']} | `{row['proposed_rust_surface']}` | {row['taxonomy_class']} | {row['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_optional(path: str | None, text: str) -> None:
    if not path:
        return
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    available = collect_surface_from_source(pathlib.Path(args.src_root))
    report = audit_inventory(inventory_rows(pathlib.Path(args.inventory_json)), available)
    payload = json.dumps(report, indent=2, sort_keys=True)
    write_optional(args.json_out, payload + "\n")
    write_optional(args.markdown_out, render_markdown(report, args.limit))
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
