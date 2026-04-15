//! Safe public Stim Rust bindings.

mod circuit;
mod common;
mod dem;
mod metadata;
mod simulators;
mod stabilizers;

pub use circuit::{
    Circuit, CircuitErrorLocation, CircuitErrorLocationStackFrame, CircuitInsertOperation,
    CircuitInstruction, CircuitItem, CircuitRepeatBlock, CircuitTargetsInsideInstruction,
    DetectingRegionFilter,
};
pub(crate) use common::bit_packing::{pack_bits, unpack_bits};
pub(crate) use common::slicing::{compute_slice_indices, normalize_index};
pub use common::upstream_commit;
pub use common::{Complex32, Result, StimError, read_shot_data_file, write_shot_data_file};
pub use ndarray::{Array1, Array2};

pub fn main(command_line_args: &[String]) -> i32 {
    stim_cxx::main(command_line_args.to_vec())
}

pub use dem::{
    DemAppendOperation, DemInstruction, DemInstructionTarget, DemItem, DemRepeatBlock, DemTarget,
    DemTargetWithCoords, DetectorErrorModel, target_logical_observable_id,
    target_relative_detector_id, target_separator,
};
pub use metadata::{
    ExplainedError, FlippedMeasurement, GateData, GateTarget, GateTargetWithCoords, all_gate_data,
    gate_data, target_combined_paulis, target_combiner, target_inv, target_pauli, target_rec,
    target_sweep_bit, target_x, target_y, target_z,
};
pub use simulators::{
    CompiledDemSampler, CompiledDetectorSampler, CompiledMeasurementSampler,
    CompiledMeasurementsToDetectionEventsConverter, ConvertedMeasurements, DemSampler,
    DetectorSampler, FlipSimulator, MeasurementSampler, MeasurementsToDetectionEventsConverter,
    TableauSimulator,
};
pub use stabilizers::{
    CliffordString, Flow, PauliPhase, PauliString, PauliStringConjugation, PauliStringIterator,
    PauliValue, Tableau, TableauIterator,
};

#[cfg(test)]
mod top_level_tests {
    use super::*;

    #[test]
    fn exposes_pinned_upstream_commit_metadata() {
        assert_eq!(upstream_commit().len(), 40);
    }

    #[test]
    fn stim_main_runs_help_command() {
        assert_eq!(crate::main(&["help".to_string()]), 0);
    }
}
