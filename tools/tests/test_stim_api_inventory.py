import hashlib
import unittest

from tools.stim_api_inventory import (
    build_drift_report,
    classify_api,
    extract_api_entries,
    render_markdown,
    render_metadata,
)


SAMPLE_DOC = """# Stim (Development Version) API Reference

## Index
- [`stim.Circuit`](#stim.Circuit)
    - [`stim.Circuit.append`](#stim.Circuit.append)
    - [`stim.Circuit.compile_sampler`](#stim.Circuit.compile_sampler)
    - [`stim.Circuit.detector_error_model`](#stim.Circuit.detector_error_model)
- [`stim.PauliString`](#stim.PauliString)
    - [`stim.PauliString.to_ndarray`](#stim.PauliString.to_ndarray)
- [`stim.target_rec`](#stim.target_rec)
"""


class StimApiInventoryTests(unittest.TestCase):
    def test_extract_api_entries_preserves_family_and_order(self) -> None:
        entries = extract_api_entries(SAMPLE_DOC)

        self.assertEqual(
            [entry["api_name"] for entry in entries],
            [
                "stim.Circuit",
                "stim.Circuit.append",
                "stim.Circuit.compile_sampler",
                "stim.Circuit.detector_error_model",
                "stim.PauliString",
                "stim.PauliString.to_ndarray",
                "stim.target_rec",
            ],
        )
        self.assertEqual(entries[1]["family"], "stim.Circuit")
        self.assertEqual(entries[5]["family"], "stim.PauliString")
        self.assertEqual(entries[-1]["family"], "stim")

    def test_classify_api_distinguishes_direct_reshaped_and_perf_sensitive(self) -> None:
        self.assertEqual(classify_api("stim.Circuit.append")["taxonomy_class"], "D1")
        self.assertEqual(
            classify_api("stim.Circuit.compile_sampler")["taxonomy_class"], "P0"
        )
        self.assertEqual(
            classify_api("stim.Circuit.detector_error_model")["taxonomy_class"], "D2"
        )
        self.assertEqual(classify_api("stim.target_rec")["taxonomy_class"], "D0")

    def test_renderers_include_hash_and_required_columns(self) -> None:
        entries = []
        for raw in extract_api_entries(SAMPLE_DOC):
            annotated = dict(raw)
            annotated.update(classify_api(raw["api_name"]))
            entries.append(annotated)

        metadata = render_metadata(
            entries=entries,
            source_url="https://example.invalid/python_api_reference_vDev.md",
            stim_commit="deadbeef",
            doc_sha256=hashlib.sha256(SAMPLE_DOC.encode()).hexdigest(),
        )
        markdown = render_markdown(entries, metadata)

        self.assertEqual(metadata["api_count"], 7)
        self.assertIn("doc_sha256", metadata)
        self.assertIn("| Python API | Doc Anchor |", markdown)
        self.assertIn("stim.Circuit.compile_sampler", markdown)
        self.assertIn("release-gate", markdown)

    def test_build_drift_report_flags_hash_and_commit_changes(self) -> None:
        report = build_drift_report(
            expected_stim_commit="old-commit",
            current_stim_commit="new-commit",
            expected_doc_sha256="old-hash",
            current_doc_sha256="new-hash",
        )

        self.assertTrue(report["stim_commit_changed"])
        self.assertTrue(report["doc_sha256_changed"])
        self.assertEqual(report["status"], "drifted")


if __name__ == "__main__":
    unittest.main()
