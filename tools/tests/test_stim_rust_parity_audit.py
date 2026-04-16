import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.stim_rust_parity_audit import (
    audit_inventory,
    collect_surface_from_source,
    normalize_surface,
)


class StimRustParityAuditTests(unittest.TestCase):
    def test_repo_surface_does_not_expose_removed_duplicate_helpers(self) -> None:
        surface = collect_surface_from_source(Path("crates/stim/src"))
        removed = {
            "stim::Circuit::flattened_operations",
            "stim::Circuit::copy",
            "stim::DetectorErrorModel::copy",
            "stim::Tableau::copy",
            "stim::CliffordString::copy",
            "stim::PauliString::copy",
            "stim::TableauSimulator::copy",
            "stim::FlipSimulator::copy",
            "stim::CompiledMeasurementSampler",
            "stim::CompiledDetectorSampler",
            "stim::CompiledDemSampler",
            "stim::CompiledMeasurementsToDetectionEventsConverter",
            "stim::MeasurementSampler::new",
            "stim::DetectorSampler::new",
            "stim::MeasurementsToDetectionEventsConverter::new",
            "stim::gate_data",
            "stim::GateTarget::new",
            "stim::DemTarget::new",
            "stim::target_rec",
            "stim::target_inv",
            "stim::target_x",
            "stim::target_y",
            "stim::target_z",
            "stim::target_combiner",
            "stim::target_sweep_bit",
            "stim::target_pauli",
            "stim::target_relative_detector_id",
            "stim::target_logical_observable_id",
            "stim::target_separator",
            "stim::Tableau::from_circuit",
        }
        self.assertTrue(removed.isdisjoint(surface), removed & surface)

    def test_collect_surface_from_source_detects_structs_methods_and_traits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lib.rs").write_text(
                textwrap.dedent(
                    """
                    pub struct Circuit {}
                    impl Circuit {
                        pub fn new() -> Self { Self {} }
                        pub fn append(&mut self) {}
                    }
                    impl std::fmt::Display for Circuit {
                        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "") }
                    }
                    impl std::ops::Add for Circuit {
                        type Output = Self;
                        fn add(self, rhs: Self) -> Self { rhs }
                    }
                    pub fn gate_data() {}
                    """
                ),
                encoding="utf-8",
            )
            surface = collect_surface_from_source(root)
            self.assertIn("stim::Circuit", surface)
            self.assertIn("stim::Circuit::new", surface)
            self.assertIn("stim::Circuit::append", surface)
            self.assertIn("stim::Circuit::[Display]", surface)
            self.assertIn("stim::Circuit::[Add]", surface)
            self.assertIn("stim::gate_data", surface)
            self.assertNotIn("stim::new", surface)

    def test_collect_surface_from_source_detects_derived_traits_with_doc_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lib.rs").write_text(
                textwrap.dedent(
                    """
                    #[derive(Clone, PartialEq, Eq)]
                    /// Example docs between derive and type.
                    pub struct GateTarget {}
                    """
                ),
                encoding="utf-8",
            )
            surface = collect_surface_from_source(root)
            self.assertIn("stim::GateTarget::[PartialEq]", surface)
            self.assertIn("stim::GateTarget::[Eq]", surface)

    def test_normalize_surface_maps_common_python_dunders(self) -> None:
        self.assertIn("stim::Circuit::new", normalize_surface("stim::Circuit::__init__"))
        self.assertIn("stim::Circuit::[Add]", normalize_surface("stim::Circuit::__add__"))
        self.assertIn("stim::Circuit::[Display]", normalize_surface("stim::Circuit::__str__"))
        self.assertIn("stim::Circuit::get", normalize_surface("stim::Circuit::__getitem__"))

    def test_normalize_surface_maps_numpy_names_to_ndarray_names(self) -> None:
        self.assertIn(
            "stim::PauliString::to_ndarray",
            normalize_surface("stim::PauliString::to_numpy"),
        )
        self.assertIn(
            "stim::PauliString::from_ndarray",
            normalize_surface("stim::PauliString::from_numpy"),
        )

    def test_audit_inventory_counts_missing_rows(self) -> None:
        rows = [
            {"api_name": "stim.Circuit.__init__", "proposed_rust_surface": "stim::Circuit::__init__", "taxonomy_class": "D1", "status": "planned"},
            {"api_name": "stim.Circuit.append", "proposed_rust_surface": "stim::Circuit::append", "taxonomy_class": "D1", "status": "planned"},
            {"api_name": "stim.gate_data", "proposed_rust_surface": "stim::gate_data", "taxonomy_class": "D0", "status": "planned"},
            {"api_name": "stim.Circuit.missing", "proposed_rust_surface": "stim::Circuit::missing", "taxonomy_class": "D0", "status": "planned"},
            {"api_name": "stim.main", "proposed_rust_surface": "stim::main", "taxonomy_class": "D0", "status": "planned"},
        ]
        available = {"stim::Circuit::new", "stim::Circuit::append", "stim::gate_data", "stim::Circuit::[Add]"}
        report = audit_inventory(rows, available)
        self.assertEqual(report["covered_rows"], 3)
        self.assertEqual(report["missing_rows"], 1)
        self.assertEqual(report["missing"][0]["api_name"], "stim.Circuit.missing")


if __name__ == "__main__":
    unittest.main()
