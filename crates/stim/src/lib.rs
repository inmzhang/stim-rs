//! Safe public Stim Rust bindings.
//!
//! [Stim](https://github.com/quantumlib/Stim) is a high-performance library
//! for simulating and analyzing quantum stabilizer circuits. This crate
//! provides safe Rust wrappers around the core C++ library, exposing types
//! for building circuits, sampling measurements, analyzing errors, and
//! working with stabilizer tableaus and Pauli strings.
//!
//! # Core types
//!
//! - [`Circuit`] — a mutable stabilizer circuit that describes a noisy quantum
//!   computation.
//! - [`DetectorErrorModel`] — an error model listing independent fault
//!   mechanisms and the detectors/observables they flip.
//! - [`PauliString`] — a signed tensor product of Pauli operators.
//! - [`Tableau`] — a stabilizer tableau representing a Clifford operation.
//!
//! # Simulators
//!
//! - [`TableauSimulator`] — an interactive stabilizer simulator backed by
//!   tableaus that supports gate-by-gate execution.
//! - [`FlipSimulator`] — a batched simulator that tracks Pauli flips and
//!   classical flip records, requiring only O(1) work per gate.
//! - [`MeasurementSampler`] — fast repeated measurement sampling from a
//!   compiled circuit.
//! - [`DetectorSampler`] — fast repeated detector-event sampling from a
//!   compiled circuit.
//! - [`DemSampler`] — fast repeated sampling from a compiled detector error
//!   model.
//!
//! # Quick start
//!
//! ```
//! // Build a simple Bell-state circuit with noise.
//! let circuit: stim::Circuit = "\
//!     H 0
//!     CNOT 0 1
//!     DEPOLARIZE2(0.01) 0 1
//!     M 0 1
//!     DETECTOR rec[-1] rec[-2]
//! ".parse().unwrap();
//!
//! // Sample detector events.
//! let mut sampler = circuit.compile_detector_sampler();
//! let events = sampler.sample(100);
//! assert_eq!(events.ncols(), 1);
//! ```

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
}
