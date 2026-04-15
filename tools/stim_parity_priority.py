from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict

PREFERRED_FAMILY_ORDER = [
    'stim.Circuit',
    'stim.Flow',
    'stim.GateData',
    'stim.PauliString',
    'stim.Tableau',
    'stim.CliffordString',
    'stim.CompiledMeasurementsToDetectionEventsConverter',
    'stim.CompiledDetectorSampler',
    'stim.CompiledMeasurementSampler',
    'stim.FlipSimulator',
    'stim.TableauSimulator',
    'stim',
]


def load_report(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def family_rank(family: str) -> tuple[int, str]:
    try:
        return (PREFERRED_FAMILY_ORDER.index(family), family)
    except ValueError:
        return (len(PREFERRED_FAMILY_ORDER), family)


def summarize_missing(report: dict[str, object]) -> dict[str, object]:
    rows = report['missing']
    family_counts = Counter(row['family'] for row in rows)
    taxonomy_counts = defaultdict(Counter)
    examples = {}
    for row in rows:
        family = row['family']
        taxonomy_counts[family][row['taxonomy_class']] += 1
        examples.setdefault(family, []).append(row['api_name'])

    families = []
    for family, count in sorted(family_counts.items(), key=lambda item: (-item[1], family_rank(item[0]))):
        families.append(
            {
                'family': family,
                'missing_count': count,
                'taxonomy_counts': dict(sorted(taxonomy_counts[family].items())),
                'examples': examples[family][:8],
            }
        )
    return {
        'covered_rows': report['covered_rows'],
        'missing_rows': report['missing_rows'],
        'families': families,
    }


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        '# Stim Rust Parity Priority Backlog',
        '',
        f"- Covered rows (heuristic): `{summary['covered_rows']}`",
        f"- Missing rows (heuristic): `{summary['missing_rows']}`",
        '',
        '## Family priorities',
        '',
        '| Family | Missing | Taxonomy mix | Example APIs |',
        '| --- | ---: | --- | --- |',
    ]
    for family in summary['families']:
        taxonomy = ', '.join(f"{k}={v}" for k, v in family['taxonomy_counts'].items())
        examples = '<br>'.join(family['examples'])
        lines.append(f"| {family['family']} | {family['missing_count']} | {taxonomy} | {examples} |")
    lines.append('')
    return '\n'.join(lines)


def write_optional(path: str | None, text: str) -> None:
    if not path:
        return
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize parity-audit output into family priorities.')
    parser.add_argument('--audit-json', default='artifacts/stim-phase6/parity-audit.json')
    parser.add_argument('--json-out')
    parser.add_argument('--markdown-out')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_missing(load_report(pathlib.Path(args.audit_json)))
    payload = json.dumps(summary, indent=2, sort_keys=True)
    write_optional(args.json_out, payload + '\n')
    write_optional(args.markdown_out, render_markdown(summary))
    print(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
