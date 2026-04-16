#![allow(clippy::too_many_arguments)]

use std::fmt;
use std::path::Path;

use crate::common::bit_packing::{pack_rows_array, unpack_rows_array};
use crate::{Result, StimError};
use ndarray::{Array2, ArrayView2};

/// Fast repeated measurement sampling from a compiled Stim circuit.
///
/// An analyzed stabilizer circuit whose measurements can be sampled quickly.
/// The sampler uses a noiseless reference sample, collected from the circuit
/// using Stim's Tableau simulator during initialization, as a baseline for
/// deriving more samples using an error-propagation simulator.
///
/// This is the Rust equivalent of Python's `stim.CompiledMeasurementSampler`.
/// Obtain one via [`Circuit::compile_sampler`](crate::Circuit::compile_sampler)
/// or construct directly with [`MeasurementSampler::new`].
///
/// # Examples
///
/// ```
/// let circuit: stim::Circuit = "X 0\nM 0 1".parse().unwrap();
/// let mut sampler = circuit.compile_sampler(false);
/// let results = sampler.sample(4);
/// assert_eq!(results.ncols(), 2);
/// assert_eq!(results.nrows(), 4);
/// // Qubit 0 was flipped by X, qubit 1 was not.
/// assert!(results[[0, 0]]);
/// assert!(!results[[0, 1]]);
/// ```
pub struct MeasurementSampler {
    pub(crate) inner: stim_cxx::MeasurementSampler,
}

/// Fast repeated detector-event sampling from a compiled Stim circuit.
///
/// An analyzed stabilizer circuit whose detection events can be sampled quickly.
/// Detection events are defined by `DETECTOR` instructions in the circuit and
/// indicate parity checks that have been violated. Observable flips, defined by
/// `OBSERVABLE_INCLUDE` instructions, can also be sampled alongside or
/// separately from detectors.
///
/// This is the Rust equivalent of Python's `stim.CompiledDetectorSampler`.
/// Obtain one via
/// [`Circuit::compile_detector_sampler`](crate::Circuit::compile_detector_sampler)
/// or construct directly with [`DetectorSampler::new`].
///
/// # Examples
///
/// ```
/// let circuit: stim::Circuit = "\
///     H 0
///     CNOT 0 1
///     DEPOLARIZE2(0.01) 0 1
///     M 0 1
///     DETECTOR rec[-1] rec[-2]
/// ".parse().unwrap();
/// let mut sampler = circuit.compile_detector_sampler();
/// let events = sampler.sample(100);
/// assert_eq!(events.ncols(), 1);
/// ```
pub struct DetectorSampler {
    pub(crate) inner: stim_cxx::DetectorSampler,
}

/// Fast repeated sampling from a compiled detector error model.
///
/// A helper for efficiently sampling from a [`DetectorErrorModel`](crate::DetectorErrorModel).
/// Each sample produces three pieces of data: which detectors fired,
/// which observables were flipped, and (optionally) which error mechanisms
/// were activated. Previously recorded error data can also be replayed to
/// reproduce deterministic outcomes.
///
/// This is the Rust equivalent of Python's `stim.CompiledDemSampler`.
/// Obtain one via
/// [`DetectorErrorModel::compile_sampler`](crate::DetectorErrorModel::compile_sampler)
/// or
/// [`DetectorErrorModel::compile_sampler_with_seed`](crate::DetectorErrorModel::compile_sampler_with_seed).
///
/// # Examples
///
/// ```
/// let dem: stim::DetectorErrorModel =
///     "error(0) D0\nerror(1) D1 D2 L0".parse().unwrap();
/// let mut sampler = dem.compile_sampler_with_seed(7);
/// let (detectors, observables, errors) = sampler.sample(2);
/// assert_eq!(detectors.ncols(), 3);  // D0, D1, D2
/// assert_eq!(observables.ncols(), 1); // L0
/// assert_eq!(errors.ncols(), 2);      // two error mechanisms
/// ```
pub struct DemSampler {
    pub(crate) inner: stim_cxx::DemSampler,
}

/// Converts batched measurements into detector events using a compiled circuit.
///
/// A tool for quickly converting raw measurement results from an analyzed
/// stabilizer circuit into detection events (and optionally observable flips).
/// The converter uses a noiseless reference sample, collected from the circuit
/// during initialization, as the baseline for determining expected detector
/// values.
///
/// Supports both in-memory conversion (packed or unpacked boolean arrays)
/// and file-to-file conversion through [`convert_file`](Self::convert_file).
/// Sweep bits for `sweep[k]` controls in the circuit can also be provided.
///
/// This is the Rust equivalent of Python's
/// `stim.CompiledMeasurementsToDetectionEventsConverter`.
/// Obtain one via
/// [`Circuit::compile_m2d_converter`](crate::Circuit::compile_m2d_converter)
/// or construct directly with [`MeasurementsToDetectionEventsConverter::new`].
///
/// # Examples
///
/// ```
/// let circuit: stim::Circuit = "\
///     X 0
///     M 0 1
///     DETECTOR rec[-1]
///     DETECTOR rec[-2]
///     OBSERVABLE_INCLUDE(0) rec[-2]
/// ".parse().unwrap();
/// let mut converter = circuit.compile_m2d_converter(false);
///
/// let result = converter.convert(
///     ndarray::array![[true, false], [false, false]].view(),
///     None,
///     true,
///     false,
/// ).unwrap();
/// // Returns detection events and observable flips separately.
/// assert!(matches!(result,
///     stim::ConvertedMeasurements::DetectionEventsAndObservables(_, _)));
/// ```
pub struct MeasurementsToDetectionEventsConverter {
    pub(crate) inner: stim_cxx::MeasurementsToDetectionEventsConverter,
}

/// A stabilizer circuit simulator backed by an inverse stabilizer tableau.
///
/// An interactive Clifford-circuit simulator that tracks an inverse stabilizer
/// tableau. It supports gate-by-gate execution, mid-circuit measurements,
/// resets, and noise channels. Each operation updates the internal tableau
/// representation, giving exact stabilizer simulation with O(n^2) cost for
/// collapsing operations and O(n) for unitary gates, where n is the qubit
/// count.
///
/// This is the Rust equivalent of Python's `stim.TableauSimulator`.
///
/// # Interactive usage
///
/// Gates can be applied one at a time via dedicated methods (e.g. [`h`](Self::h),
/// [`cx`](Self::cx), [`s`](Self::s)), or an entire [`Circuit`](crate::Circuit)
/// can be fed in at once via [`do_circuit`](Self::do_circuit). Measurements
/// are destructive and append to an internal measurement record accessible
/// through [`current_measurement_record`](Self::current_measurement_record).
///
/// # Examples
///
/// ```
/// let mut sim = stim::TableauSimulator::new();
/// sim.h(&[0]).unwrap();
/// sim.cx(&[0, 1]).unwrap();
/// // The two qubits are now entangled in a Bell state.
/// let a = sim.measure(0);
/// let b = sim.measure(1);
/// assert_eq!(a, b);
/// ```
pub struct TableauSimulator {
    pub(crate) inner: stim_cxx::TableauSimulator,
}

/// A batched simulator that tracks Pauli flips and classical flip records.
///
/// Instead of tracking the actual quantum state, this simulator tracks
/// *whether things are flipped*, which is significantly cheaper: O(1) work
/// per gate, compared to O(n) for unitary operations and O(n^2) for
/// collapsing operations in the tableau simulator, where n is the qubit count.
///
/// The flip simulator processes many instances in parallel (the "batch").
/// For best performance, use a batch size that is a multiple of 256, because
/// internally the instance states are striped across SIMD words with one bit
/// per instance. Even a batch size of 1 does roughly the same work as 256.
///
/// This is the Rust equivalent of Python's `stim.FlipSimulator`.
///
/// # Examples
///
/// ```
/// let mut sim = stim::FlipSimulator::new(256, false, 3, 0);
/// let circuit: stim::Circuit = "\
///     X_ERROR(0.1) 0 1 2
///     M 0 1 2
///     DETECTOR rec[-1] rec[-2]
/// ".parse().unwrap();
/// sim.do_circuit(&circuit);
/// assert_eq!(sim.num_measurements(), 3);
/// assert_eq!(sim.num_detectors(), 1);
/// ```
pub struct FlipSimulator {
    pub(crate) inner: stim_cxx::FlipSimulator,
}

type UnpackedBitMatrix = Array2<bool>;
type PackedObservablePair = (Vec<u8>, Vec<u8>);
type UnpackedObservablePair = (UnpackedBitMatrix, UnpackedBitMatrix);
type PackedDemBatch = (Vec<u8>, Vec<u8>, Vec<u8>);
type UnpackedDemBatch = (UnpackedBitMatrix, UnpackedBitMatrix, UnpackedBitMatrix);

/// The result of converting measurements into detector data.
///
/// Returned by [`MeasurementsToDetectionEventsConverter::convert`]. The variant
/// depends on whether `separate_observables` was set to `true` during
/// conversion.
///
/// When observables are separated, the detection-event matrix and the
/// observable-flip matrix are returned as two distinct arrays. Otherwise
/// a single matrix is returned (which may or may not have observable bits
/// appended, depending on the `append_observables` flag).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertedMeasurements {
    /// Detector events only (or with observables appended if requested).
    ///
    /// The contained [`Array2<bool>`] has shape `(num_shots, num_detectors)` when
    /// `append_observables` is false, or
    /// `(num_shots, num_detectors + num_observables)` when it is true.
    DetectionEvents(Array2<bool>),
    /// Detector events plus a separate observable-flip matrix.
    ///
    /// The first [`Array2<bool>`] has shape `(num_shots, num_detectors)`.
    /// The second has shape `(num_shots, num_observables)`.
    DetectionEventsAndObservables(Array2<bool>, Array2<bool>),
}

/// A 2D boolean table, either unpacked or bit-packed by row.
///
/// Used by the [`FlipSimulator`] to return measurement, detector, and
/// observable flip records. The representation depends on the `bit_packed`
/// flag passed to the extraction method.
///
/// When unpacked ([`BoolMatrix`](Self::BoolMatrix)), each cell is a single
/// `bool`. When packed ([`PackedMatrix`](Self::PackedMatrix)), bits are packed
/// into `u8` bytes in little-endian order within each row, matching Stim's
/// `b8` data layout.
///
/// # Examples
///
/// ```
/// let mut sim = stim::FlipSimulator::new(1, true, 1, 0);
/// sim.broadcast_pauli_errors('X', ndarray::array![[true]].view(), 1.0)
///     .unwrap();
/// let arrays = sim.to_ndarray(false);
/// assert!(format!("{:?}", arrays.xs).contains("BoolMatrix"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitTable {
    /// Unpacked boolean matrix where each cell is one bit.
    BoolMatrix(Array2<bool>),
    /// Packed boolean matrix where bits are packed into bytes row by row.
    PackedMatrix(Array2<u8>),
}

/// A Rust-friendly export of the full [`FlipSimulator`] state.
///
/// Returned by [`FlipSimulator::to_ndarray`]. Contains five [`BitTable`]
/// matrices representing the X-basis flips, Z-basis flips, measurement flip
/// records, detector flip records, and observable flip records. Each matrix
/// has one row per batch instance.
///
/// # Examples
///
/// ```
/// let mut sim = stim::FlipSimulator::new(2, true, 1, 0);
/// sim.broadcast_pauli_errors('X', ndarray::array![[true, false]].view(), 1.0)
///     .unwrap();
/// let arrays = sim.to_ndarray(false);
/// assert!(format!("{arrays:?}").contains("measurement_flips"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlipSimulatorArrays {
    /// X-basis flip state per qubit per batch instance.
    pub xs: BitTable,
    /// Z-basis flip state per qubit per batch instance.
    pub zs: BitTable,
    /// Measurement flip records.
    pub measurement_flips: BitTable,
    /// Detector flip records.
    pub detector_flips: BitTable,
    /// Observable flip records.
    pub observable_flips: BitTable,
}

impl MeasurementSampler {
    /// Creates a measurement sampler for the given circuit.
    ///
    /// The sampler uses a noiseless reference sample, collected from the circuit
    /// using Stim's Tableau simulator during initialization, as a baseline for
    /// deriving more samples using an error-propagation simulator.
    ///
    /// # Arguments
    ///
    /// * `circuit` - The stabilizer circuit to sample measurements from.
    /// * `skip_reference_sample` - When `true`, the reference sample is set to
    ///   all zeros instead of being collected from the circuit. This means
    ///   returned results represent whether each measurement was *flipped*,
    ///   rather than actual measurement outcomes. Useful when you only care
    ///   about error propagation or when you know the all-zero result is a valid
    ///   noiseless outcome. Computing the reference sample is the most expensive
    ///   part of initialization, so skipping it is an effective optimization.
    /// * `seed` - Deterministically seeds the random number generator. Making
    ///   the exact same series of calls on the same machine with the same Stim
    ///   version will produce the same results. Results are *not* guaranteed to
    ///   be consistent across Stim versions, SIMD widths, or varying shot
    ///   counts.
    ///
    /// # Examples
    ///
    /// Returns the number of measurement bits produced per shot.
    ///
    /// This equals the total number of `M` (and similar measurement)
    /// operations in the compiled circuit.
    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        self.inner.num_measurements()
    }

    /// Samples measurement data in bit-packed form.
    ///
    /// Returns a flat `Vec<u8>` where each shot occupies
    /// `ceil(num_measurements / 8)` bytes. Within each byte, bits are packed
    /// in little-endian order, matching Stim's `b8` data layout: the bit for
    /// measurement `m` in shot `s` is at
    /// `result[s * row_bytes + m / 8] >> (m % 8) & 1`.
    #[must_use]
    pub fn sample_bit_packed(&mut self, shots: u64) -> Vec<u8> {
        self.inner.sample_bit_packed(shots)
    }

    /// Samples a batch of measurement results as an unpacked boolean matrix.
    ///
    /// Returns an `Array2<bool>` with shape `(shots, num_measurements)`.
    /// The bit for measurement `m` in shot `s` is at `result[[s, m]]`.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0".parse().unwrap();
    /// let mut sampler = circuit.compile_sampler(false);
    /// assert_eq!(sampler.sample(3), ndarray::array![[true], [true], [true]]);
    /// ```
    #[must_use]
    pub fn sample(&mut self, shots: u64) -> Array2<bool> {
        let packed = self.sample_bit_packed(shots);
        let num_measurements = self.num_measurements() as usize;
        unpack_rows_array(&packed, num_measurements)
    }

    /// Samples measurements from the circuit and writes them directly to a file.
    ///
    /// This is more memory-efficient than calling [`sample`](Self::sample) and
    /// then writing the result, because the data is streamed to disk without
    /// ever being fully materialized in memory.
    ///
    /// # Arguments
    ///
    /// * `shots` - The number of times to sample every measurement.
    /// * `filepath` - The file path to write results to.
    /// * `format_name` - The output format. Valid values are `"01"`, `"b8"`,
    ///   `"r8"`, `"ptb64"`, `"hits"`, and `"dets"`.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the file path is not valid
    /// UTF-8 or the file cannot be written.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0".parse().unwrap();
    /// let path = std::env::temp_dir().join("stim-rs-measurement-sampler-write.01");
    /// let mut sampler = circuit.compile_sampler(false);
    /// sampler.sample_write(2, &path, "01").unwrap();
    /// assert_eq!(std::fs::read_to_string(&path).unwrap(), "1\n1\n");
    /// std::fs::remove_file(path).unwrap();
    /// ```
    pub fn sample_write(
        &mut self,
        shots: u64,
        filepath: impl AsRef<Path>,
        format_name: &str,
    ) -> Result<()> {
        let path = filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("filepath must be valid UTF-8"))?;
        self.inner
            .sample_write(shots, path, format_name)
            .map_err(StimError::from)
    }
}

impl TableauSimulator {
    /// Creates a tableau simulator with a deterministic default seed of 0.
    ///
    /// The simulator starts with zero qubits tracked. Qubits are automatically
    /// allocated as gates reference them.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::TableauSimulator::new();
    /// sim.h(&[0]).unwrap();
    /// assert_eq!(sim.peek_bloch(0), stim::PauliString::from_text("+X").unwrap());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Creates a tableau simulator with an explicit RNG seed.
    ///
    /// Using the same seed on the same machine with the same Stim version
    /// produces identical simulation results. Results are *not* guaranteed
    /// to be consistent across Stim versions, SIMD widths, or different
    /// orderings of operations.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            inner: stim_cxx::TableauSimulator::new(0, seed),
        }
    }

    /// Returns the number of qubits currently tracked by the simulator.
    ///
    /// The simulator automatically grows to accommodate qubits referenced
    /// by gates. This count reflects the current high-water mark.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Resizes the simulator's qubit count.
    ///
    /// If the new count is larger, new qubits are initialized in the `|0>`
    /// state. If smaller, qubits beyond the new limit are discarded.
    pub fn set_num_qubits(&mut self, new_num_qubits: usize) {
        self.inner.set_num_qubits(new_num_qubits);
    }

    /// Returns a copy of the simulator's current inverse stabilizer tableau.
    ///
    /// The internal state of a `TableauSimulator` is an *inverse* tableau.
    /// To get the forward tableau (whose Z outputs are the stabilizers of
    /// the current state), call `.inverse(false)` on the result.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::TableauSimulator::new();
    /// sim.h(&[0]).unwrap();
    /// assert_eq!(
    ///     sim.current_inverse_tableau(),
    ///     stim::Tableau::from_conjugated_generators(
    ///         &[stim::PauliString::from_text("+Z").unwrap()],
    ///         &[stim::PauliString::from_text("+X").unwrap()],
    ///     )
    ///     .unwrap()
    /// );
    /// ```
    #[must_use]
    pub fn current_inverse_tableau(&self) -> crate::Tableau {
        crate::Tableau {
            inner: self.inner.current_inverse_tableau(),
        }
    }

    /// Replaces the simulator state with the given inverse tableau.
    ///
    /// This directly sets the internal representation. To set the state
    /// from stabilizers, use [`set_state_from_stabilizers`](Self::set_state_from_stabilizers)
    /// instead.
    pub fn set_inverse_tableau(&mut self, tableau: &crate::Tableau) {
        self.inner.set_inverse_tableau(&tableau.inner);
    }

    /// Returns a copy of the record of all measurements performed so far.
    ///
    /// Each entry corresponds to one measurement, in the order they were
    /// executed. The record grows each time [`measure`](Self::measure),
    /// [`measure_many`](Self::measure_many), or a circuit containing `M`
    /// instructions is applied.
    #[must_use]
    pub fn current_measurement_record(&self) -> Vec<bool> {
        self.inner.current_measurement_record()
    }

    /// Applies every instruction in a circuit to the simulator state.
    ///
    /// The circuit's gates, measurements, resets, noise channels, and
    /// annotations are all processed in order. Measurements append to the
    /// internal measurement record.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::TableauSimulator::new();
    /// let circuit: stim::Circuit = "X 0\nM 0".parse().unwrap();
    /// sim.do_circuit(&circuit);
    /// assert_eq!(sim.current_measurement_record(), vec![true]);
    /// ```
    pub fn do_circuit(&mut self, circuit: &crate::Circuit) {
        self.inner.do_circuit(&circuit.inner);
    }

    /// Applies a Pauli string as a Pauli product gate to the simulator.
    ///
    /// Each qubit in the string has the corresponding Pauli (X, Y, or Z)
    /// applied. Identity positions (`I` / `_`) are skipped. The sign of
    /// the Pauli string controls the global phase.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::TableauSimulator::new();
    /// sim.do_pauli_string(&stim::PauliString::from_text("IX").unwrap());
    /// assert_eq!(sim.measure_many(&[0, 1]), vec![false, true]);
    /// ```
    pub fn do_pauli_string(&mut self, pauli_string: &crate::PauliString) {
        self.inner.do_pauli_string(&pauli_string.inner);
    }

    /// Applies an arbitrary Clifford operation (given as a [`Tableau`](crate::Tableau))
    /// to the specified target qubits.
    ///
    /// The `targets` slice maps the tableau's qubit indices to simulator
    /// qubit indices. For example, with a 2-qubit tableau and
    /// `targets = &[5, 3]`, the tableau's qubit 0 acts on simulator
    /// qubit 5 and its qubit 1 acts on simulator qubit 3.
    pub fn do_tableau(&mut self, tableau: &crate::Tableau, targets: &[usize]) {
        self.inner.do_tableau(&tableau.inner, targets);
    }

    /// Applies a circuit-like operation to the simulator.
    ///
    /// Accepts anything that converts into a `TableauSimulatorOperation`:
    /// a [`Circuit`](crate::Circuit), a [`PauliString`](crate::PauliString),
    /// a [`CircuitInstruction`](crate::CircuitInstruction), or a
    /// [`CircuitRepeatBlock`](crate::CircuitRepeatBlock).
    ///
    /// This is the Rust equivalent of Python's `TableauSimulator.do()`.
    pub fn r#do<'a>(
        &mut self,
        operation: impl Into<TableauSimulatorOperation<'a>>,
    ) -> crate::Result<()> {
        match operation.into() {
            TableauSimulatorOperation::Circuit(circuit) => {
                self.do_circuit(circuit);
                Ok(())
            }
            TableauSimulatorOperation::PauliString(pauli_string) => {
                self.do_pauli_string(pauli_string);
                Ok(())
            }
            TableauSimulatorOperation::Instruction(instruction) => {
                let mut circuit = crate::Circuit::new();
                circuit.append_instruction(instruction)?;
                self.do_circuit(&circuit);
                Ok(())
            }
            TableauSimulatorOperation::RepeatBlock(block) => {
                let mut circuit = crate::Circuit::new();
                circuit.append_repeat_block(
                    block.repeat_count(),
                    &block.body_copy(),
                    block.tag(),
                )?;
                self.do_circuit(&circuit);
                Ok(())
            }
        }
    }

    /// Returns the single-qubit Bloch vector as a one-qubit [`PauliString`](crate::PauliString).
    ///
    /// If the qubit is in a computational basis eigenstate, the result is
    /// `+Z` or `-Z`. If it is in an X or Y eigenstate, the result is `+X`,
    /// `-X`, `+Y`, or `-Y` respectively. If the qubit is entangled with
    /// other qubits (not a single-qubit eigenstate), the result is `+_`
    /// (the identity Pauli string).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::TableauSimulator::new();
    /// assert_eq!(sim.peek_bloch(0), stim::PauliString::from_text("+Z").unwrap());
    /// sim.x(&[0]).unwrap();
    /// assert_eq!(sim.peek_bloch(0), stim::PauliString::from_text("-Z").unwrap());
    /// sim.h(&[0]).unwrap();
    /// assert_eq!(sim.peek_bloch(0), stim::PauliString::from_text("-X").unwrap());
    /// ```
    #[must_use]
    pub fn peek_bloch(&mut self, target: usize) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.peek_bloch(target),
            imag: false,
        }
    }

    /// Returns the X-axis expectation value of a qubit: `-1`, `0`, or `+1`.
    ///
    /// Returns `+1` if the qubit is in the `|+>` state, `-1` if in `|->`,
    /// and `0` if the X observable is not determined (e.g. in `|0>` or
    /// entangled).
    pub fn peek_x(&mut self, target: usize) -> i32 {
        self.inner.peek_x(target)
    }

    /// Returns the Y-axis expectation value of a qubit: `-1`, `0`, or `+1`.
    ///
    /// Returns `+1` if the qubit is in the `|+i>` state, `-1` if in `|-i>`,
    /// and `0` if the Y observable is not determined.
    pub fn peek_y(&mut self, target: usize) -> i32 {
        self.inner.peek_y(target)
    }

    /// Returns the Z-axis expectation value of a qubit: `-1`, `0`, or `+1`.
    ///
    /// Returns `+1` if the qubit is in the `|0>` state, `-1` if in `|1>`,
    /// and `0` if the Z observable is not determined.
    pub fn peek_z(&mut self, target: usize) -> i32 {
        self.inner.peek_z(target)
    }

    /// Destructively measures a single qubit in the Z basis and returns the result.
    ///
    /// The result is appended to the internal measurement record. If the
    /// qubit is in a definite Z eigenstate, the result is deterministic.
    /// If not, the outcome is random (controlled by the simulator's PRNG
    /// seed) and the qubit collapses to the measured eigenstate.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::TableauSimulator::new();
    /// assert!(!sim.measure(0));
    /// sim.x(&[0]).unwrap();
    /// assert!(sim.measure(0));
    /// assert_eq!(sim.current_measurement_record(), vec![false, true]);
    /// ```
    pub fn measure(&mut self, target: usize) -> bool {
        self.inner.measure(target)
    }

    /// Destructively measures several qubits in the Z basis and returns the results.
    ///
    /// Equivalent to calling [`measure`](Self::measure) on each target in
    /// order. All results are also appended to the measurement record.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::TableauSimulator::new();
    /// sim.do_pauli_string(&stim::PauliString::from_text("IXYZ").unwrap());
    /// assert_eq!(sim.measure_many(&[0, 1, 2, 3]), vec![false, true, true, false]);
    /// ```
    #[must_use]
    pub fn measure_many(&mut self, targets: &[usize]) -> Vec<bool> {
        self.inner.measure_many(targets)
    }

    /// Returns a standardized list of the simulator's current stabilizer generators.
    ///
    /// Two simulators have the same canonical stabilizers if and only if
    /// their current quantum states are equal (and they track the same
    /// number of qubits).
    ///
    /// The canonical form is computed by Gaussian elimination on the Z
    /// outputs of the forward tableau, considering generators in the order
    /// X0, Z0, X1, Z1, etc.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::TableauSimulator::new();
    /// sim.h(&[0]).unwrap();
    /// assert_eq!(
    ///     sim.canonical_stabilizers(),
    ///     vec![stim::PauliString::from_text("+X").unwrap()]
    /// );
    /// ```
    #[must_use]
    pub fn canonical_stabilizers(&self) -> Vec<crate::PauliString> {
        self.current_inverse_tableau()
            .inverse(false)
            .to_stabilizers(true)
            .expect("canonical stabilizers from the simulator tableau should be valid")
            .into_iter()
            .collect()
    }

    /// Returns the expectation value of a Pauli observable: `-1`, `0`, or `+1`.
    ///
    /// Returns `+1` if the observable is a stabilizer, `-1` if its negation
    /// is a stabilizer, and `0` if the observable anticommutes with at
    /// least one stabilizer (i.e., its expectation is not determined).
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the observable is invalid.
    pub fn peek_observable_expectation(
        &self,
        observable: &crate::PauliString,
    ) -> crate::Result<i32> {
        self.inner
            .peek_observable_expectation(&observable.inner)
            .map_err(StimError::from)
    }

    /// Measures a multi-qubit Pauli observable, optionally with classical flip noise.
    ///
    /// The measurement result is `true` if the observable's eigenvalue is
    /// `-1`, and `false` if it is `+1`. When `flip_probability > 0`, the
    /// result is randomly flipped with that probability (modeling readout
    /// noise). The result is also appended to the measurement record.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the observable is invalid.
    pub fn measure_observable(
        &mut self,
        observable: &crate::PauliString,
        flip_probability: f64,
    ) -> crate::Result<bool> {
        self.inner
            .measure_observable(&observable.inner, flip_probability)
            .map_err(StimError::from)
    }

    /// Postselects the simulator state on a multi-qubit Pauli observable outcome.
    ///
    /// Forces the observable to have the eigenvalue corresponding to
    /// `desired_value` (`false` for `+1`, `true` for `-1`). If the current
    /// state is incompatible with the desired outcome, the operation fails.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the postselection is
    /// impossible (the state has zero overlap with the desired eigenspace).
    pub fn postselect_observable(
        &mut self,
        observable: &crate::PauliString,
        desired_value: bool,
    ) -> crate::Result<()> {
        self.inner
            .postselect_observable(&observable.inner, desired_value)
            .map_err(StimError::from)
    }

    /// Postselects qubits in the X basis.
    ///
    /// Forces each target qubit into the `|+>` state (if `desired_value` is
    /// `false`) or the `|->` state (if `true`).
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the postselection is
    /// impossible for any target.
    pub fn postselect_x(&mut self, targets: &[usize], desired_value: bool) -> crate::Result<()> {
        self.inner
            .postselect_x(targets, desired_value)
            .map_err(StimError::from)
    }

    /// Postselects qubits in the Y basis.
    ///
    /// Forces each target qubit into the `|+i>` state (if `desired_value` is
    /// `false`) or the `|-i>` state (if `true`).
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the postselection is
    /// impossible for any target.
    pub fn postselect_y(&mut self, targets: &[usize], desired_value: bool) -> crate::Result<()> {
        self.inner
            .postselect_y(targets, desired_value)
            .map_err(StimError::from)
    }

    /// Postselects qubits in the Z basis.
    ///
    /// Forces each target qubit into the `|0>` state (if `desired_value` is
    /// `false`) or the `|1>` state (if `true`).
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the postselection is
    /// impossible for any target.
    pub fn postselect_z(&mut self, targets: &[usize], desired_value: bool) -> crate::Result<()> {
        self.inner
            .postselect_z(targets, desired_value)
            .map_err(StimError::from)
    }

    /// Measures a qubit in the Z basis and returns both the result and the
    /// kickback Pauli string (if any).
    ///
    /// The "kickback" is a Pauli string that, when applied, would flip the
    /// measurement result. It is `Some(...)` when the measurement outcome
    /// is non-deterministic (i.e. the qubit is entangled), and `None` when
    /// the outcome is deterministic. The measurement result is also appended
    /// to the measurement record.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::TableauSimulator::new();
    /// sim.h(&[0]).unwrap();
    /// assert_eq!(
    ///     sim.measure_kickback(0).1,
    ///     Some(stim::PauliString::from_text("+X").unwrap())
    /// );
    /// ```
    pub fn measure_kickback(&mut self, target: usize) -> (bool, Option<crate::PauliString>) {
        let (result, kickback) = self.inner.measure_kickback(target);
        (
            result,
            kickback.map(|inner| crate::PauliString { inner, imag: false }),
        )
    }

    /// Exports the stabilizer state as a dense state vector.
    ///
    /// # Arguments
    ///
    /// * `endian` - Either `"little"` or `"big"`, controlling the mapping
    ///   between qubit indices and state-vector indices.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the endian string is
    /// invalid.
    pub fn state_vector(&self, endian: &str) -> crate::Result<Vec<crate::Complex32>> {
        self.current_inverse_tableau()
            .inverse(false)
            .to_state_vector(endian)
    }

    /// Sets the simulator state from a list of stabilizer generators.
    ///
    /// The stabilizers must be independent (unless `allow_redundant` is
    /// `true`) and must fully constrain the state (unless
    /// `allow_underconstrained` is `true`, in which case unconstrained
    /// degrees of freedom default to `|0>`).
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the stabilizers are
    /// anticommuting, or if constraints on redundancy / underconstraint
    /// are violated.
    pub fn set_state_from_stabilizers(
        &mut self,
        stabilizers: &[crate::PauliString],
        allow_redundant: bool,
        allow_underconstrained: bool,
    ) -> crate::Result<()> {
        let tableau =
            crate::Tableau::from_stabilizers(stabilizers, allow_redundant, allow_underconstrained)?;
        self.set_inverse_tableau(&tableau.inverse(false));
        Ok(())
    }

    /// Sets the simulator state from a dense state vector.
    ///
    /// The state vector must represent a valid stabilizer state (i.e., it
    /// must be reachable from `|0...0>` by Clifford gates).
    ///
    /// # Arguments
    ///
    /// * `state_vector` - The amplitudes of the state.
    /// * `endian` - Either `"little"` or `"big"`.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the vector is not a
    /// valid stabilizer state or the endian string is invalid.
    pub fn set_state_from_state_vector(
        &mut self,
        state_vector: &[crate::Complex32],
        endian: &str,
    ) -> crate::Result<()> {
        let tableau = crate::Tableau::from_state_vector(state_vector, endian)?;
        self.set_inverse_tableau(&tableau.inverse(false));
        Ok(())
    }

    fn apply_gate(
        &mut self,
        gate_name: &str,
        targets: &[usize],
        args: &[f64],
    ) -> crate::Result<()> {
        let mut circuit = crate::Circuit::new();
        let raw_targets: Vec<u32> = targets
            .iter()
            .map(|&target| u32::try_from(target).expect("qubit index should fit into u32"))
            .collect();
        circuit.append(gate_name, &raw_targets, args)?;
        self.do_circuit(&circuit);
        Ok(())
    }

    /// Applies the Hadamard gate (`H`, also known as `H_XZ`) to the given targets.
    ///
    /// Swaps the X and Z axes of each target qubit: `|0> -> |+>`, `|1> -> |->`.
    pub fn h(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("H", targets, &[])
    }

    /// Applies the `H_XY` gate, which swaps the X and Y axes.
    pub fn h_xy(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("H_XY", targets, &[])
    }

    /// Applies the `H_XZ` gate (same as `H`) to the given targets.
    pub fn h_xz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("H", targets, &[])
    }

    /// Applies the `H_YZ` gate, which swaps the Y and Z axes.
    pub fn h_yz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("H_YZ", targets, &[])
    }

    /// Applies the `C_XYZ` gate (X -> Y -> Z -> X axis rotation) to the given targets.
    pub fn c_xyz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("C_XYZ", targets, &[])
    }

    /// Applies the `C_ZYX` gate (Z -> Y -> X -> Z axis rotation) to the given targets.
    pub fn c_zyx(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("C_ZYX", targets, &[])
    }

    /// Applies the Pauli X gate (bit flip) to the given targets.
    pub fn x(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("X", targets, &[])
    }

    /// Applies the Pauli Y gate to the given targets.
    pub fn y(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("Y", targets, &[])
    }

    /// Applies the Pauli Z gate (phase flip) to the given targets.
    pub fn z(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("Z", targets, &[])
    }

    /// Applies the S gate (sqrt(Z), quarter-turn around Z) to the given targets.
    pub fn s(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("S", targets, &[])
    }

    /// Applies the S^dag gate (inverse of S) to the given targets.
    pub fn s_dag(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("S_DAG", targets, &[])
    }

    /// Applies the sqrt(X) gate (quarter-turn around X) to the given targets.
    pub fn sqrt_x(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("SQRT_X", targets, &[])
    }

    /// Applies the inverse sqrt(X) gate to the given targets.
    pub fn sqrt_x_dag(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("SQRT_X_DAG", targets, &[])
    }

    /// Applies the sqrt(Y) gate (quarter-turn around Y) to the given targets.
    pub fn sqrt_y(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("SQRT_Y", targets, &[])
    }

    /// Applies the inverse sqrt(Y) gate to the given targets.
    pub fn sqrt_y_dag(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("SQRT_Y_DAG", targets, &[])
    }

    /// Resets the given qubits to the `|0>` state (Z-basis reset, Stim's `R` gate).
    pub fn reset(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("R", targets, &[])
    }

    /// Resets the given qubits to the `|+>` state (X-basis reset, Stim's `RX` gate).
    pub fn reset_x(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("RX", targets, &[])
    }

    /// Resets the given qubits to the `|+i>` state (Y-basis reset, Stim's `RY` gate).
    pub fn reset_y(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("RY", targets, &[])
    }

    /// Resets the given qubits to the `|0>` state (Z-basis reset, alias for [`reset`](Self::reset)).
    pub fn reset_z(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("R", targets, &[])
    }

    /// Applies the CX (CNOT) gate to pairs of targets.
    ///
    /// Targets are consumed in pairs: `targets[0]` controls `targets[1]`,
    /// `targets[2]` controls `targets[3]`, etc. The slice length must be even.
    pub fn cx(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("CX", targets, &[])
    }

    /// Applies the CNOT gate. This is an alias for [`cx`](Self::cx).
    pub fn cnot(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.cx(targets)
    }

    /// Applies the CY (controlled-Y) gate to pairs of targets.
    pub fn cy(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("CY", targets, &[])
    }

    /// Applies the CZ (controlled-Z) gate to pairs of targets.
    pub fn cz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("CZ", targets, &[])
    }

    /// Applies the SWAP gate to pairs of targets.
    pub fn swap(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("SWAP", targets, &[])
    }

    /// Applies the ISWAP gate to pairs of targets.
    pub fn iswap(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("ISWAP", targets, &[])
    }

    /// Applies the inverse ISWAP gate to pairs of targets.
    pub fn iswap_dag(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("ISWAP_DAG", targets, &[])
    }

    /// Applies the XCX (X-controlled-X) gate to pairs of targets.
    pub fn xcx(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("XCX", targets, &[])
    }

    /// Applies the XCY (X-controlled-Y) gate to pairs of targets.
    pub fn xcy(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("XCY", targets, &[])
    }

    /// Applies the XCZ (X-controlled-Z) gate to pairs of targets.
    pub fn xcz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("XCZ", targets, &[])
    }

    /// Applies the YCX (Y-controlled-X) gate to pairs of targets.
    pub fn ycx(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("YCX", targets, &[])
    }

    /// Applies the YCY (Y-controlled-Y) gate to pairs of targets.
    pub fn ycy(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("YCY", targets, &[])
    }

    /// Applies the YCZ (Y-controlled-Z) gate to pairs of targets.
    pub fn ycz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("YCZ", targets, &[])
    }

    /// Applies the ZCX (Z-controlled-X, equivalent to CX) gate to pairs of targets.
    pub fn zcx(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("ZCX", targets, &[])
    }

    /// Applies the ZCY (Z-controlled-Y) gate to pairs of targets.
    pub fn zcy(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("ZCY", targets, &[])
    }

    /// Applies the ZCZ (Z-controlled-Z, equivalent to CZ) gate to pairs of targets.
    pub fn zcz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("ZCZ", targets, &[])
    }

    /// Applies an X error channel with probability `p` to the given targets.
    ///
    /// Each target qubit independently has an X gate applied with probability `p`.
    pub fn x_error(&mut self, targets: &[usize], p: f64) -> crate::Result<()> {
        self.apply_gate("X_ERROR", targets, &[p])
    }

    /// Applies a Y error channel with probability `p` to the given targets.
    ///
    /// Each target qubit independently has a Y gate applied with probability `p`.
    pub fn y_error(&mut self, targets: &[usize], p: f64) -> crate::Result<()> {
        self.apply_gate("Y_ERROR", targets, &[p])
    }

    /// Applies a Z error channel with probability `p` to the given targets.
    ///
    /// Each target qubit independently has a Z gate applied with probability `p`.
    pub fn z_error(&mut self, targets: &[usize], p: f64) -> crate::Result<()> {
        self.apply_gate("Z_ERROR", targets, &[p])
    }

    /// Applies a single-qubit depolarizing channel with probability `p`.
    ///
    /// Each target qubit independently has one of {X, Y, Z} applied, each
    /// with probability `p/3`. The total error probability is `p`.
    pub fn depolarize1(&mut self, targets: &[usize], p: f64) -> crate::Result<()> {
        self.apply_gate("DEPOLARIZE1", targets, &[p])
    }

    /// Applies a two-qubit depolarizing channel with probability `p`.
    ///
    /// Each pair of target qubits independently has one of the 15
    /// non-identity two-qubit Pauli operators applied, each with
    /// probability `p/15`. Targets are consumed in pairs.
    pub fn depolarize2(&mut self, targets: &[usize], p: f64) -> crate::Result<()> {
        self.apply_gate("DEPOLARIZE2", targets, &[p])
    }
}

impl Clone for TableauSimulator {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl fmt::Debug for TableauSimulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableauSimulator")
            .field("num_qubits", &self.num_qubits())
            .field(
                "measurement_record_len",
                &self.current_measurement_record().len(),
            )
            .finish()
    }
}

impl Default for TableauSimulator {
    fn default() -> Self {
        Self::new()
    }
}

fn decode_bit_table(data: stim_cxx::BitTableData) -> BitTable {
    let rows = data.rows as usize;
    let cols = data.cols as usize;
    if data.bit_packed {
        let row_bytes = cols.div_ceil(8);
        BitTable::PackedMatrix(
            Array2::from_shape_vec((rows, row_bytes), data.data)
                .expect("packed bit tables from stim-cxx should be rectangular"),
        )
    } else {
        BitTable::BoolMatrix(
            Array2::from_shape_vec(
                (rows, cols),
                data.data.into_iter().map(|v| v != 0).collect(),
            )
            .expect("boolean bit tables from stim-cxx should be rectangular"),
        )
    }
}

impl FlipSimulator {
    /// Creates a new flip simulator.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The number of parallel instances to simulate. For best
    ///   performance, use a multiple of 256 (the SIMD word width). Even with
    ///   `batch_size = 1`, internally the same work is done as for 256 instances.
    /// * `disable_stabilizer_randomization` - When `false` (the default), a Z
    ///   error is added with 50% probability each time a stabilizer is introduced
    ///   (e.g., at resets and measurements). This enforces the uncertainty
    ///   principle and catches bugs where anticommuting stabilizers are measured.
    ///   Set to `true` only when you specifically want to trace error propagation
    ///   without the randomization noise.
    /// * `num_qubits` - Initial number of qubits. The simulator will still
    ///   auto-resize when gates reference qubits beyond this limit.
    /// * `seed` - Deterministic PRNG seed. Same caveats about cross-version and
    ///   cross-architecture consistency apply as for other Stim simulators.
    ///
    /// # Examples
    ///
    /// ```
    /// let sim = stim::FlipSimulator::new(2, true, 3, 0);
    /// assert_eq!(sim.batch_size(), 2);
    /// assert_eq!(sim.num_qubits(), 3);
    /// ```
    #[must_use]
    pub fn new(
        batch_size: usize,
        disable_stabilizer_randomization: bool,
        num_qubits: usize,
        seed: u64,
    ) -> Self {
        Self {
            inner: stim_cxx::FlipSimulator::new(
                batch_size,
                disable_stabilizer_randomization,
                num_qubits,
                seed,
            ),
        }
    }

    /// Returns the number of parallel instances (shots) tracked by the simulator.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    /// Returns the number of qubits currently tracked by the simulator.
    ///
    /// Auto-grows when gates reference higher-indexed qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of measurement-flip records accumulated so far.
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.inner.num_measurements()
    }

    /// Returns the number of detector-flip records accumulated so far.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    /// Returns the number of observable-flip records accumulated so far.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Clears all simulator state: qubit flips, and all recorded measurements,
    /// detectors, and observables.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Sets the Pauli flip on a specific `(qubit, batch-instance)` location.
    ///
    /// # Arguments
    ///
    /// * `pauli` - The Pauli to set: `'I'`/`'_'`/`0` for identity, `'X'`/`1`,
    ///   `'Y'`/`2`, `'Z'`/`3`.
    /// * `qubit_index` - The qubit to set the flip on.
    /// * `instance_index` - Which batch instance to modify.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the Pauli value is invalid
    /// or the indices are out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::FlipSimulator::new(2, true, 3, 0);
    /// sim.set_pauli_flip('X', 2, 1).unwrap();
    /// assert_eq!(
    ///     sim.peek_pauli_flip(1).unwrap(),
    ///     stim::PauliString::from_text("+__X").unwrap()
    /// );
    /// ```
    pub fn set_pauli_flip(
        &mut self,
        pauli: impl Into<crate::PauliValue>,
        qubit_index: isize,
        instance_index: isize,
    ) -> Result<()> {
        let code = match pauli.into() {
            crate::PauliValue::Code(value @ 0..=3) => value,
            crate::PauliValue::Code(_) => {
                return Err(StimError::new(
                    "Need pauli in ['I', 'X', 'Y', 'Z', 0, 1, 2, 3, '_'].",
                ));
            }
            crate::PauliValue::Symbol('_') | crate::PauliValue::Symbol('I') => 0,
            crate::PauliValue::Symbol('X') => 1,
            crate::PauliValue::Symbol('Y') => 2,
            crate::PauliValue::Symbol('Z') => 3,
            crate::PauliValue::Symbol(_) => {
                return Err(StimError::new(
                    "Need pauli in ['I', 'X', 'Y', 'Z', 0, 1, 2, 3, '_'].",
                ));
            }
        };
        self.inner
            .set_pauli_flip(code, qubit_index as i64, instance_index as i64)
            .map_err(StimError::from)
    }

    /// Returns the current Pauli flip state for every batch instance.
    ///
    /// The returned vector has one [`PauliString`](crate::PauliString) per
    /// batch instance, each with length equal to [`num_qubits`](Self::num_qubits).
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) on internal failures.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::FlipSimulator::new(2, true, 3, 0);
    /// sim.set_pauli_flip('X', 2, 1).unwrap();
    /// assert_eq!(
    ///     sim.peek_pauli_flips().unwrap(),
    ///     vec![
    ///         stim::PauliString::from_text("+___").unwrap(),
    ///         stim::PauliString::from_text("+__X").unwrap(),
    ///     ]
    /// );
    /// ```
    pub fn peek_pauli_flips(&self) -> Result<Vec<crate::PauliString>> {
        self.inner
            .peek_pauli_flips()
            .map(|values| {
                values
                    .into_iter()
                    .map(|inner| crate::PauliString { inner, imag: false })
                    .collect()
            })
            .map_err(StimError::from)
    }

    /// Returns the current Pauli flip state for a single batch instance.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the index is out of bounds.
    pub fn peek_pauli_flip(&self, instance_index: isize) -> Result<crate::PauliString> {
        self.inner
            .peek_pauli_flip(instance_index as i64)
            .map(|inner| crate::PauliString { inner, imag: false })
            .map_err(StimError::from)
    }

    /// Appends unpacked measurement-flip rows to the simulator's measurement record.
    ///
    /// The array must have shape `(num_new_measurements, batch_size)`,
    /// where each row represents one measurement across all batch instances.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::FlipSimulator::new(4, true, 0, 0);
    /// sim.append_measurement_flips(ndarray::array![[false, true, false, true]].view())
    ///     .unwrap();
    /// assert!(format!("{:?}", sim.get_measurement_flips(false)).contains("BoolMatrix"));
    /// ```
    pub fn append_measurement_flips(
        &mut self,
        measurement_flip_data: ArrayView2<'_, bool>,
    ) -> Result<()> {
        if measurement_flip_data.ncols() != self.batch_size() {
            return Err(StimError::new(
                "measurement flip rows must all have width batch_size",
            ));
        }
        let data: Vec<u8> = measurement_flip_data
            .iter()
            .map(|bit| u8::from(*bit))
            .collect();
        self.inner
            .append_measurement_flips(data, measurement_flip_data.nrows(), false)
            .map_err(StimError::from)
    }

    /// Appends bit-packed measurement-flip rows to the simulator's measurement record.
    ///
    /// The array must have shape `(num_new_measurements, ceil(batch_size / 8))`,
    /// with bits packed in little-endian order within each byte.
    pub fn append_measurement_flips_bit_packed(
        &mut self,
        measurement_flip_data: ArrayView2<'_, u8>,
    ) -> Result<()> {
        let row_bytes = self.batch_size().div_ceil(8);
        if measurement_flip_data.ncols() != row_bytes {
            return Err(StimError::new(
                "bit-packed measurement flip rows must all have width ceil(batch_size / 8)",
            ));
        }
        let data: Vec<u8> = measurement_flip_data.iter().copied().collect();
        self.inner
            .append_measurement_flips(data, measurement_flip_data.nrows(), true)
            .map_err(StimError::from)
    }

    /// Applies every instruction in a circuit to the flip simulator.
    ///
    /// This processes all gates, measurements, resets, noise channels,
    /// detectors, and observable annotations in order. Measurement, detector,
    /// and observable flip records are accumulated.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::FlipSimulator::new(2, true, 1, 0);
    /// let circuit: stim::Circuit =
    ///     "M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]".parse().unwrap();
    /// sim.do_circuit(&circuit);
    /// assert_eq!(sim.num_measurements(), 1);
    /// assert_eq!(sim.num_detectors(), 1);
    /// assert_eq!(sim.num_observables(), 1);
    /// ```
    pub fn do_circuit(&mut self, circuit: &crate::Circuit) {
        self.inner.do_circuit(&circuit.inner);
    }

    /// Applies a circuit-like operation to the flip simulator.
    ///
    /// Accepts anything that converts into a `FlipSimulatorOperation`:
    /// a [`Circuit`](crate::Circuit), a [`CircuitInstruction`](crate::CircuitInstruction),
    /// or a [`CircuitRepeatBlock`](crate::CircuitRepeatBlock).
    ///
    /// This is the Rust equivalent of Python's `FlipSimulator.do()`.
    pub fn r#do<'a>(&mut self, operation: impl Into<FlipSimulatorOperation<'a>>) -> Result<()> {
        match operation.into() {
            FlipSimulatorOperation::Circuit(circuit) => {
                self.do_circuit(circuit);
                Ok(())
            }
            FlipSimulatorOperation::Instruction(instruction) => {
                let mut circuit = crate::Circuit::new();
                circuit.append_instruction(instruction)?;
                self.do_circuit(&circuit);
                Ok(())
            }
            FlipSimulatorOperation::RepeatBlock(block) => {
                let mut circuit = crate::Circuit::new();
                circuit.append_repeat_block(
                    block.repeat_count(),
                    &block.body_copy(),
                    block.tag(),
                )?;
                self.do_circuit(&circuit);
                Ok(())
            }
        }
    }

    /// Broadcasts Pauli errors through a boolean mask.
    ///
    /// For each `(qubit, batch_instance)` where `mask` is `true`, the
    /// specified Pauli flip is applied with probability `p`. The mask must
    /// have shape `(num_qubits, batch_size)`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::FlipSimulator::new(2, true, 3, 0);
    /// sim.broadcast_pauli_errors(
    ///     'X',
    ///     ndarray::array![[true, false], [false, false], [true, true]].view(),
    ///     1.0,
    /// )
    /// .unwrap();
    /// assert_eq!(
    ///     sim.peek_pauli_flips().unwrap(),
    ///     vec![
    ///         stim::PauliString::from_text("+X_X").unwrap(),
    ///         stim::PauliString::from_text("+__X").unwrap(),
    ///     ]
    /// );
    /// ```
    pub fn broadcast_pauli_errors(
        &mut self,
        pauli: impl Into<crate::PauliValue>,
        mask: ArrayView2<'_, bool>,
        p: f32,
    ) -> Result<()> {
        if mask.ncols() != self.batch_size() {
            return Err(StimError::new("mask rows must all have width batch_size"));
        }
        let code = match pauli.into() {
            crate::PauliValue::Code(value @ 0..=3) => value,
            crate::PauliValue::Code(_) => {
                return Err(StimError::new(
                    "Need pauli in ['I', 'X', 'Y', 'Z', 0, 1, 2, 3, '_'].",
                ));
            }
            crate::PauliValue::Symbol('_') | crate::PauliValue::Symbol('I') => 0,
            crate::PauliValue::Symbol('X') => 1,
            crate::PauliValue::Symbol('Y') => 2,
            crate::PauliValue::Symbol('Z') => 3,
            crate::PauliValue::Symbol(_) => {
                return Err(StimError::new(
                    "Need pauli in ['I', 'X', 'Y', 'Z', 0, 1, 2, 3, '_'].",
                ));
            }
        };
        let flat: Vec<u8> = mask.iter().map(|b| u8::from(*b)).collect();
        self.inner
            .broadcast_pauli_errors(code, flat, mask.nrows(), p)
            .map_err(StimError::from)
    }

    /// Generates independent Bernoulli samples, optionally bit-packed.
    ///
    /// Returns `num_samples` random bits, each `true` with probability `p`.
    /// When `bit_packed` is true, the result has `ceil(num_samples / 8)` bytes.
    pub fn generate_bernoulli_samples(
        &mut self,
        num_samples: usize,
        p: f32,
        bit_packed: bool,
    ) -> Result<Vec<u8>> {
        self.inner
            .generate_bernoulli_samples(num_samples, p, bit_packed)
            .map_err(StimError::from)
    }

    /// Returns the accumulated measurement-flip records as a `BitTable`.
    ///
    /// When `bit_packed` is `false`, returns a `BoolMatrix` with shape
    /// `(num_measurements, batch_size)`. When `true`, returns a `PackedMatrix`
    /// with shape `(num_measurements, ceil(batch_size / 8))`.
    pub fn get_measurement_flips(&self, bit_packed: bool) -> BitTable {
        decode_bit_table(self.inner.get_measurement_flips(bit_packed))
    }

    /// Returns the accumulated detector-flip records as a `BitTable`.
    ///
    /// Shape conventions are the same as for
    /// [`get_measurement_flips`](Self::get_measurement_flips).
    pub fn get_detector_flips(&self, bit_packed: bool) -> BitTable {
        decode_bit_table(self.inner.get_detector_flips(bit_packed))
    }

    /// Returns the accumulated observable-flip records as a `BitTable`.
    ///
    /// Shape conventions are the same as for
    /// [`get_measurement_flips`](Self::get_measurement_flips).
    pub fn get_observable_flips(&self, bit_packed: bool) -> BitTable {
        decode_bit_table(self.inner.get_observable_flips(bit_packed))
    }

    /// Exports the full simulator state as a `FlipSimulatorArrays`.
    ///
    /// When `bit_packed` is `false`, all tables use `BoolMatrix` form.
    /// When `true`, all tables use `PackedMatrix` form.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sim = stim::FlipSimulator::new(2, true, 3, 0);
    /// sim.broadcast_pauli_errors(
    ///     'X',
    ///     ndarray::array![[true, false], [false, false], [true, true]].view(),
    ///     1.0,
    /// )
    /// .unwrap();
    /// let arrays = sim.to_ndarray(false);
    /// let debug = format!("{arrays:?}");
    /// assert!(debug.contains("BoolMatrix"));
    /// assert!(debug.contains("measurement_flips"));
    /// ```
    pub fn to_ndarray(&self, bit_packed: bool) -> FlipSimulatorArrays {
        let paulis = self
            .peek_pauli_flips()
            .expect("peek_pauli_flips should succeed for current simulator state");
        let (xs, zs) = if bit_packed {
            (
                BitTable::PackedMatrix(
                    Array2::from_shape_vec(
                        (paulis.len(), paulis[0].len().div_ceil(8)),
                        paulis
                            .iter()
                            .flat_map(|p| p.to_ndarray_bit_packed().0)
                            .collect(),
                    )
                    .expect("packed X tableau state should be rectangular"),
                ),
                BitTable::PackedMatrix(
                    Array2::from_shape_vec(
                        (paulis.len(), paulis[0].len().div_ceil(8)),
                        paulis
                            .iter()
                            .flat_map(|p| p.to_ndarray_bit_packed().1)
                            .collect(),
                    )
                    .expect("packed Z tableau state should be rectangular"),
                ),
            )
        } else {
            (
                BitTable::BoolMatrix(
                    Array2::from_shape_vec(
                        (paulis.len(), paulis[0].len()),
                        paulis.iter().flat_map(|p| p.to_ndarray().0).collect(),
                    )
                    .expect("boolean X tableau state should be rectangular"),
                ),
                BitTable::BoolMatrix(
                    Array2::from_shape_vec(
                        (paulis.len(), paulis[0].len()),
                        paulis.iter().flat_map(|p| p.to_ndarray().1).collect(),
                    )
                    .expect("boolean Z tableau state should be rectangular"),
                ),
            )
        };
        FlipSimulatorArrays {
            xs,
            zs,
            measurement_flips: self.get_measurement_flips(bit_packed),
            detector_flips: self.get_detector_flips(bit_packed),
            observable_flips: self.get_observable_flips(bit_packed),
        }
    }
}

impl Clone for FlipSimulator {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl fmt::Debug for FlipSimulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlipSimulator")
            .field("batch_size", &self.batch_size())
            .field("num_qubits", &self.num_qubits())
            .field("num_measurements", &self.num_measurements())
            .field("num_detectors", &self.num_detectors())
            .field("num_observables", &self.num_observables())
            .finish()
    }
}

/// Operations accepted by [`FlipSimulator::do`](FlipSimulator::do).
///
/// Implements `From` for [`Circuit`](crate::Circuit),
/// [`CircuitInstruction`](crate::CircuitInstruction), and
/// [`CircuitRepeatBlock`](crate::CircuitRepeatBlock), so those types can
/// be passed directly to `FlipSimulator::do()`.
pub enum FlipSimulatorOperation<'a> {
    /// Apply an entire circuit.
    Circuit(&'a crate::Circuit),
    /// Apply a single instruction.
    Instruction(&'a crate::CircuitInstruction),
    /// Apply a repeat block.
    RepeatBlock(&'a crate::CircuitRepeatBlock),
}

impl<'a> From<&'a crate::Circuit> for FlipSimulatorOperation<'a> {
    fn from(value: &'a crate::Circuit) -> Self {
        Self::Circuit(value)
    }
}

impl<'a> From<&'a crate::CircuitInstruction> for FlipSimulatorOperation<'a> {
    fn from(value: &'a crate::CircuitInstruction) -> Self {
        Self::Instruction(value)
    }
}

impl<'a> From<&'a crate::CircuitRepeatBlock> for FlipSimulatorOperation<'a> {
    fn from(value: &'a crate::CircuitRepeatBlock) -> Self {
        Self::RepeatBlock(value)
    }
}

/// Operations accepted by [`TableauSimulator::do`](TableauSimulator::do).
///
/// Implements `From` for [`Circuit`](crate::Circuit),
/// [`PauliString`](crate::PauliString),
/// [`CircuitInstruction`](crate::CircuitInstruction), and
/// [`CircuitRepeatBlock`](crate::CircuitRepeatBlock), so those types can
/// be passed directly to `TableauSimulator::do()`.
pub enum TableauSimulatorOperation<'a> {
    /// Apply an entire circuit.
    Circuit(&'a crate::Circuit),
    /// Apply a Pauli string as a gate.
    PauliString(&'a crate::PauliString),
    /// Apply a single instruction.
    Instruction(&'a crate::CircuitInstruction),
    /// Apply a repeat block.
    RepeatBlock(&'a crate::CircuitRepeatBlock),
}

impl<'a> From<&'a crate::Circuit> for TableauSimulatorOperation<'a> {
    fn from(value: &'a crate::Circuit) -> Self {
        Self::Circuit(value)
    }
}

impl<'a> From<&'a crate::PauliString> for TableauSimulatorOperation<'a> {
    fn from(value: &'a crate::PauliString) -> Self {
        Self::PauliString(value)
    }
}

impl<'a> From<&'a crate::CircuitInstruction> for TableauSimulatorOperation<'a> {
    fn from(value: &'a crate::CircuitInstruction) -> Self {
        Self::Instruction(value)
    }
}

impl<'a> From<&'a crate::CircuitRepeatBlock> for TableauSimulatorOperation<'a> {
    fn from(value: &'a crate::CircuitRepeatBlock) -> Self {
        Self::RepeatBlock(value)
    }
}

impl fmt::Debug for MeasurementSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MeasurementSampler")
            .field("num_measurements", &self.num_measurements())
            .finish()
    }
}

impl DetectorSampler {
    /// Creates a detector sampler for the given circuit with an explicit PRNG seed.
    ///
    /// # Arguments
    ///
    /// * `circuit` - The stabilizer circuit to sample detector events from.
    ///   Must contain `DETECTOR` instructions.
    /// * `seed` - Deterministic PRNG seed. Same caveats about cross-version
    ///   and cross-architecture consistency apply as for other Stim simulators.
    ///
    /// # Examples
    ///
    /// Returns the number of detector bits produced per shot.
    ///
    /// Equals the number of `DETECTOR` instructions in the circuit.
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        self.inner.num_detectors()
    }

    /// Returns the number of observable bits produced per shot.
    ///
    /// Equals the number of distinct `OBSERVABLE_INCLUDE` indices in the circuit.
    #[must_use]
    pub fn num_observables(&self) -> u64 {
        self.inner.num_observables()
    }

    /// Samples detector-event data in bit-packed form.
    ///
    /// Returns a flat `Vec<u8>` where each shot occupies
    /// `ceil(num_detectors / 8)` bytes with little-endian bit packing.
    #[must_use]
    pub fn sample_bit_packed(&mut self, shots: u64) -> Vec<u8> {
        self.inner.sample_bit_packed(shots)
    }

    /// Samples observable-flip data in bit-packed form.
    ///
    /// Returns a flat `Vec<u8>` where each shot occupies
    /// `ceil(num_observables / 8)` bytes with little-endian bit packing.
    #[must_use]
    pub fn sample_observables_bit_packed(&mut self, shots: u64) -> Vec<u8> {
        self.inner.sample_observables_bit_packed(shots)
    }

    /// Samples a batch of detector events as an unpacked boolean matrix.
    ///
    /// Returns an `Array2<bool>` with shape `(shots, num_detectors)`.
    /// The bit for detector `d` in shot `s` is at `result[[s, d]]`.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0\nDETECTOR rec[-1]".parse().unwrap();
    /// let mut sampler = circuit.compile_detector_sampler();
    /// assert_eq!(sampler.sample(2), ndarray::array![[false], [false]]);
    /// ```
    #[must_use]
    pub fn sample(&mut self, shots: u64) -> Array2<bool> {
        let packed = self.sample_bit_packed(shots);
        let num_detectors = self.num_detectors() as usize;
        unpack_rows_array(&packed, num_detectors)
    }

    /// Writes detector-event samples directly to a file.
    ///
    /// # Arguments
    ///
    /// * `shots` - Number of times to sample.
    /// * `filepath` - File path to write results to.
    /// * `format_name` - Output format (`"01"`, `"b8"`, `"r8"`, `"ptb64"`,
    ///   `"hits"`, `"dets"`).
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) on I/O or encoding failure.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0\nDETECTOR rec[-1]".parse().unwrap();
    /// let path = std::env::temp_dir().join("stim-rs-detector-sampler-write.01");
    /// let mut sampler = circuit.compile_detector_sampler_with_seed(11);
    /// sampler.sample_write(2, &path, "01").unwrap();
    /// assert_eq!(std::fs::read_to_string(&path).unwrap(), "0\n0\n");
    /// std::fs::remove_file(path).unwrap();
    /// ```
    pub fn sample_write(
        &mut self,
        shots: u64,
        filepath: impl AsRef<Path>,
        format_name: &str,
    ) -> Result<()> {
        let path = filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("filepath must be valid UTF-8"))?;
        self.inner
            .sample_write(shots, path, format_name)
            .map_err(StimError::from)
    }

    /// Writes detector events and observable flips to two separate files.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) on I/O or encoding failure.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit =
    ///     "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]".parse().unwrap();
    /// let dets = std::env::temp_dir().join("stim-rs-detector-sampler-dets.01");
    /// let obs = std::env::temp_dir().join("stim-rs-detector-sampler-obs.01");
    /// let mut sampler = circuit.compile_detector_sampler_with_seed(11);
    /// sampler
    ///     .sample_write_separate_observables(2, &dets, "01", &obs, "01")
    ///     .unwrap();
    /// assert_eq!(std::fs::read_to_string(&dets).unwrap(), "0\n0\n");
    /// assert_eq!(std::fs::read_to_string(&obs).unwrap(), "0\n0\n");
    /// std::fs::remove_file(dets).unwrap();
    /// std::fs::remove_file(obs).unwrap();
    /// ```
    pub fn sample_write_separate_observables(
        &mut self,
        shots: u64,
        dets_filepath: impl AsRef<Path>,
        dets_format_name: &str,
        obs_filepath: impl AsRef<Path>,
        obs_format_name: &str,
    ) -> Result<()> {
        let dets_path = dets_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("detector filepath must be valid UTF-8"))?;
        let obs_path = obs_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("observable filepath must be valid UTF-8"))?;
        self.inner
            .sample_write_separate_observables(
                shots,
                dets_path,
                dets_format_name,
                obs_path,
                obs_format_name,
            )
            .map_err(StimError::from)
    }

    /// Samples detector and observable data separately in bit-packed form.
    ///
    /// Returns `(detector_bytes, observable_bytes)` where each follows
    /// the same little-endian packing convention.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit =
    ///     "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]".parse().unwrap();
    /// let mut sampler = circuit.compile_detector_sampler_with_seed(11);
    /// let (dets, obs) = sampler.sample_bit_packed_separate_observables(2);
    /// assert_eq!(dets, vec![0b0, 0b0]);
    /// assert_eq!(obs, vec![0b0, 0b0]);
    /// ```
    #[must_use]
    pub fn sample_bit_packed_separate_observables(&mut self, shots: u64) -> (Vec<u8>, Vec<u8>) {
        self.inner.sample_bit_packed_separate_observables(shots)
    }

    /// Samples detector and observable data separately as unpacked boolean matrices.
    ///
    /// Returns `(detectors, observables)` where `detectors` has shape
    /// `(shots, num_detectors)` and `observables` has shape
    /// `(shots, num_observables)`.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit =
    ///     "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]".parse().unwrap();
    /// let mut sampler = circuit.compile_detector_sampler_with_seed(11);
    /// let (dets, obs) = sampler.sample_separate_observables(2);
    /// assert_eq!(dets, ndarray::array![[false], [false]]);
    /// assert_eq!(obs, ndarray::array![[false], [false]]);
    /// ```
    #[must_use]
    pub fn sample_separate_observables(&mut self, shots: u64) -> (Array2<bool>, Array2<bool>) {
        let (dets, obs) = self.sample_bit_packed_separate_observables(shots);
        let num_detectors = self.num_detectors() as usize;
        let num_observables = self.num_observables() as usize;
        (
            unpack_rows_array(&dets, num_detectors),
            unpack_rows_array(&obs, num_observables),
        )
    }

    /// Samples detector events with observable flips prepended.
    ///
    /// Returns an `Array2<bool>` with shape
    /// `(shots, num_observables + num_detectors)` where observable columns
    /// come first, followed by detector columns.
    #[must_use]
    pub fn sample_prepend_observables(&mut self, shots: u64) -> Array2<bool> {
        let (dets, obs) = self.sample_separate_observables(shots);
        ndarray::concatenate(ndarray::Axis(1), &[obs.view(), dets.view()])
            .expect("observable/detector arrays should share shot dimension")
    }

    /// Samples detector events with observable flips appended.
    ///
    /// Returns an `Array2<bool>` with shape
    /// `(shots, num_detectors + num_observables)` where detector columns
    /// come first, followed by observable columns.
    #[must_use]
    pub fn sample_append_observables(&mut self, shots: u64) -> Array2<bool> {
        let (dets, obs) = self.sample_separate_observables(shots);
        ndarray::concatenate(ndarray::Axis(1), &[dets.view(), obs.view()])
            .expect("detector/observable arrays should share shot dimension")
    }
}

impl fmt::Debug for DetectorSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DetectorSampler")
            .field("num_detectors", &self.num_detectors())
            .field("num_observables", &self.num_observables())
            .finish()
    }
}

impl DemSampler {
    /// Returns the number of detector bits produced per DEM sample.
    ///
    /// Equals the number of distinct detector indices referenced by the model.
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        self.inner.num_detectors()
    }

    /// Returns the number of observable bits produced per DEM sample.
    ///
    /// Equals the number of distinct observable indices referenced by the model.
    #[must_use]
    pub fn num_observables(&self) -> u64 {
        self.inner.num_observables()
    }

    /// Returns the number of error mechanism bits produced per DEM sample.
    ///
    /// Equals the number of `error(...)` instructions in the model.
    #[must_use]
    pub fn num_errors(&self) -> u64 {
        self.inner.num_errors()
    }

    /// Samples the DEM's error mechanisms and returns bit-packed results.
    ///
    /// Returns `(detector_bytes, observable_bytes, error_bytes)` in
    /// little-endian packed form, where each shot occupies
    /// `ceil(num_X / 8)` bytes for the corresponding quantity.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "error(0) D0\nerror(1) D1 D2 L0".parse().unwrap();
    /// let mut sampler = dem.compile_sampler_with_seed(7);
    /// let (detectors, observables, errors) = sampler.sample_bit_packed(2);
    /// assert_eq!(detectors, vec![0b110, 0b110]);
    /// assert_eq!(observables, vec![0b1, 0b1]);
    /// assert_eq!(errors, vec![0b10, 0b10]);
    /// ```
    #[must_use]
    pub fn sample_bit_packed(&mut self, shots: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let batch = self.inner.sample_bit_packed(shots);
        (batch.detectors, batch.observables, batch.errors)
    }

    /// Samples the DEM's error mechanisms and returns unpacked boolean matrices.
    ///
    /// Returns `(detectors, observables, errors)` where each is an
    /// `Array2<bool>` with shape `(shots, num_X)` for the corresponding
    /// quantity.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "error(0) D0\nerror(1) D1 D2 L0".parse().unwrap();
    /// let mut sampler = dem.compile_sampler_with_seed(7);
    /// let (detectors, observables, errors) = sampler.sample(2);
    /// assert_eq!(detectors, ndarray::array![[false, true, true], [false, true, true]]);
    /// assert_eq!(observables, ndarray::array![[true], [true]]);
    /// assert_eq!(errors, ndarray::array![[false, true], [false, true]]);
    /// ```
    #[must_use]
    pub fn sample(&mut self, shots: u64) -> UnpackedDemBatch {
        let (detectors, observables, errors) = self.sample_bit_packed(shots);
        (
            unpack_rows_array(&detectors, self.num_detectors() as usize),
            unpack_rows_array(&observables, self.num_observables() as usize),
            unpack_rows_array(&errors, self.num_errors() as usize),
        )
    }

    /// Replays previously recorded bit-packed errors and returns bit-packed results.
    ///
    /// Instead of sampling errors randomly, the provided `recorded_errors`
    /// data (e.g. from a previous call to [`sample_bit_packed`](Self::sample_bit_packed))
    /// is replayed deterministically. This is useful for debugging or
    /// analyzing specific error configurations.
    pub fn sample_bit_packed_replay(
        &mut self,
        recorded_errors: &[u8],
        shots: u64,
    ) -> PackedDemBatch {
        let batch = self.inner.sample_bit_packed_replay(recorded_errors, shots);
        (batch.detectors, batch.observables, batch.errors)
    }

    /// Replays previously recorded errors and returns unpacked boolean results.
    ///
    /// This is the unpacked equivalent of [`sample_bit_packed_replay`](Self::sample_bit_packed_replay).
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "error(0.125) D0\nerror(0.25) D1".parse().unwrap();
    /// let mut noisy = dem.compile_sampler_with_seed(9);
    /// let (det_data, obs_data, err_data) = noisy.sample_bit_packed(16);
    ///
    /// let mut replay = dem.compile_sampler_with_seed(999);
    /// let (det_again, obs_again, err_again) = replay.sample_bit_packed_replay(&err_data, 16);
    ///
    /// assert_eq!(det_data, det_again);
    /// assert_eq!(obs_data, obs_again);
    /// assert_eq!(err_data, err_again);
    /// ```
    #[must_use]
    pub fn sample_replay(&mut self, recorded_errors: &[u8], shots: u64) -> UnpackedDemBatch {
        let (detectors, observables, errors) =
            self.sample_bit_packed_replay(recorded_errors, shots);
        (
            unpack_rows_array(&detectors, self.num_detectors() as usize),
            unpack_rows_array(&observables, self.num_observables() as usize),
            unpack_rows_array(&errors, self.num_errors() as usize),
        )
    }

    /// Writes sampled detector and observable data to separate files.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) on I/O or encoding failure.
    pub fn sample_write(
        &mut self,
        shots: u64,
        dets_filepath: impl AsRef<Path>,
        dets_format_name: &str,
        obs_filepath: impl AsRef<Path>,
        obs_format_name: &str,
    ) -> Result<()> {
        let dets_path = dets_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("detector filepath must be valid UTF-8"))?;
        let obs_path = obs_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("observable filepath must be valid UTF-8"))?;
        self.inner
            .sample_write(
                shots,
                dets_path,
                dets_format_name,
                obs_path,
                obs_format_name,
                "",
                "01",
                false,
            )
            .map_err(StimError::from)
    }

    /// Writes sampled detector, observable, and error data to separate files.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) on I/O or encoding failure.
    pub fn sample_write_with_errors(
        &mut self,
        shots: u64,
        dets_filepath: impl AsRef<Path>,
        dets_format_name: &str,
        obs_filepath: impl AsRef<Path>,
        obs_format_name: &str,
        err_filepath: impl AsRef<Path>,
        err_format_name: &str,
    ) -> Result<()> {
        let dets_path = dets_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("detector filepath must be valid UTF-8"))?;
        let obs_path = obs_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("observable filepath must be valid UTF-8"))?;
        let err_path = err_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("error filepath must be valid UTF-8"))?;
        self.inner
            .sample_write(
                shots,
                dets_path,
                dets_format_name,
                obs_path,
                obs_format_name,
                err_path,
                err_format_name,
                true,
            )
            .map_err(StimError::from)
    }

    /// Replays recorded errors from a file and writes detector/observable outputs.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) on I/O or encoding failure.
    pub fn sample_write_replay(
        &mut self,
        shots: u64,
        dets_filepath: impl AsRef<Path>,
        dets_format_name: &str,
        obs_filepath: impl AsRef<Path>,
        obs_format_name: &str,
        replay_err_filepath: impl AsRef<Path>,
        replay_err_format_name: &str,
    ) -> Result<()> {
        let dets_path = dets_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("detector filepath must be valid UTF-8"))?;
        let obs_path = obs_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("observable filepath must be valid UTF-8"))?;
        let replay_path = replay_err_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("replay error filepath must be valid UTF-8"))?;
        self.inner
            .sample_write_replay(
                shots,
                dets_path,
                dets_format_name,
                obs_path,
                obs_format_name,
                "",
                "01",
                false,
                replay_path,
                replay_err_format_name,
            )
            .map_err(StimError::from)
    }

    /// Replays recorded errors from a file and writes all three outputs
    /// (detectors, observables, and sampled errors) to separate files.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) on I/O or encoding failure.
    pub fn sample_write_replay_with_errors(
        &mut self,
        shots: u64,
        dets_filepath: impl AsRef<Path>,
        dets_format_name: &str,
        obs_filepath: impl AsRef<Path>,
        obs_format_name: &str,
        err_filepath: impl AsRef<Path>,
        err_format_name: &str,
        replay_err_filepath: impl AsRef<Path>,
        replay_err_format_name: &str,
    ) -> Result<()> {
        let dets_path = dets_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("detector filepath must be valid UTF-8"))?;
        let obs_path = obs_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("observable filepath must be valid UTF-8"))?;
        let err_path = err_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("error filepath must be valid UTF-8"))?;
        let replay_path = replay_err_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("replay error filepath must be valid UTF-8"))?;
        self.inner
            .sample_write_replay(
                shots,
                dets_path,
                dets_format_name,
                obs_path,
                obs_format_name,
                err_path,
                err_format_name,
                true,
                replay_path,
                replay_err_format_name,
            )
            .map_err(StimError::from)
    }
}

impl MeasurementsToDetectionEventsConverter {
    /// Creates a compiled measurements-to-detection-events converter.
    ///
    /// The converter uses a noiseless reference sample, collected from the
    /// circuit during initialization, as the baseline for determining
    /// expected detector values.
    ///
    /// # Arguments
    ///
    /// * `circuit` - The circuit whose detector definitions drive the conversion.
    /// Converts measurement data into detection events (and optionally observable flips).
    ///
    /// This is the primary conversion entry point, combining the behavior of
    /// several lower-level methods into a single call. The result variant
    /// depends on `separate_observables`:
    ///
    /// * `false` -> [`ConvertedMeasurements::DetectionEvents`] (observables may be
    ///   appended if `append_observables` is `true`).
    /// * `true` -> [`ConvertedMeasurements::DetectionEventsAndObservables`] with
    ///   detectors and observables as separate matrices.
    ///
    /// # Arguments
    ///
    /// * `measurements` - Boolean matrix of shape `(num_shots, num_measurements)`.
    /// * `sweep_bits` - Optional boolean matrix of shape `(num_shots, num_sweep_bits)`.
    ///   Used for `sweep[k]` controls in the circuit.
    /// * `separate_observables` - When `true`, return observables as a separate array.
    /// * `append_observables` - When `true` and `separate_observables` is `false`,
    ///   observable bits are appended after detector bits in the output.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit =
    ///     "X 0\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]"
    ///         .parse()
    ///         .unwrap();
    /// let mut converter = circuit.compile_m2d_converter(false);
    /// let converted = converter
    ///     .convert(
    ///         ndarray::array![[true, false], [false, false]].view(),
    ///         None,
    ///         true,
    ///         false,
    ///     )
    ///     .unwrap();
    /// assert_eq!(
    ///     converted,
    ///     stim::ConvertedMeasurements::DetectionEventsAndObservables(
    ///         ndarray::array![[false, false], [false, true]],
    ///         ndarray::array![[false], [true]],
    ///     )
    /// );
    /// ```
    pub fn convert(
        &mut self,
        measurements: ArrayView2<'_, bool>,
        sweep_bits: Option<ArrayView2<'_, bool>>,
        separate_observables: bool,
        append_observables: bool,
    ) -> Result<ConvertedMeasurements> {
        match (sweep_bits, separate_observables) {
            (Some(sweep_bits), true) => self
                .convert_measurements_and_sweep_bits_separate_observables(measurements, sweep_bits)
                .map(|(dets, obs)| ConvertedMeasurements::DetectionEventsAndObservables(dets, obs)),
            (Some(sweep_bits), false) => self
                .convert_measurements_and_sweep_bits(measurements, sweep_bits, append_observables)
                .map(ConvertedMeasurements::DetectionEvents),
            (None, true) => self
                .convert_measurements_separate_observables(measurements)
                .map(|(dets, obs)| ConvertedMeasurements::DetectionEventsAndObservables(dets, obs)),
            (None, false) => self
                .convert_measurements(measurements, append_observables)
                .map(ConvertedMeasurements::DetectionEvents),
        }
    }

    /// Converts bit-packed measurement data into bit-packed detector-event data.
    ///
    /// # Arguments
    ///
    /// * `measurements` - Bit-packed measurement bytes (little-endian, one row
    ///   of `ceil(num_measurements / 8)` bytes per shot, concatenated).
    /// * `shots` - Number of shots in the packed data.
    /// * `append_observables` - When `true`, observable bits are appended after
    ///   detector bits in the output.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit =
    ///     "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]".parse().unwrap();
    /// let mut converter = circuit.compile_m2d_converter(false);
    /// assert_eq!(
    ///     converter.convert_measurements_bit_packed(&[0b0, 0b1], 2, false),
    ///     vec![0b1, 0b0]
    /// );
    /// assert_eq!(
    ///     converter.convert_measurements_bit_packed(&[0b0, 0b1], 2, true),
    ///     vec![0b11, 0b00]
    /// );
    /// ```
    pub fn convert_measurements_bit_packed(
        &mut self,
        measurements: &[u8],
        shots: u64,
        append_observables: bool,
    ) -> Vec<u8> {
        self.inner
            .convert_measurements_bit_packed(measurements, shots, append_observables)
    }

    /// Converts bit-packed measurements plus bit-packed sweep bits into
    /// bit-packed detector-event data.
    pub fn convert_measurements_and_sweep_bits_bit_packed(
        &mut self,
        measurements: &[u8],
        sweep_bits: &[u8],
        shots: u64,
        append_observables: bool,
    ) -> Vec<u8> {
        self.inner.convert_measurements_and_sweep_bits_bit_packed(
            measurements,
            sweep_bits,
            shots,
            append_observables,
        )
    }

    /// Converts bit-packed measurements plus sweep bits, returning bit-packed
    /// detectors and observables as a separate `(detectors, observables)` pair.
    pub fn convert_measurements_and_sweep_bits_bit_packed_separate_observables(
        &mut self,
        measurements: &[u8],
        sweep_bits: &[u8],
        shots: u64,
    ) -> PackedObservablePair {
        (
            self.convert_measurements_and_sweep_bits_bit_packed(
                measurements,
                sweep_bits,
                shots,
                false,
            ),
            self.inner.convert_observables_with_sweep_bits_bit_packed(
                measurements,
                sweep_bits,
                shots,
            ),
        )
    }

    /// Converts unpacked measurements plus unpacked sweep bits into
    /// unpacked detector-event data.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the shot counts of
    /// `measurements` and `sweep_bits` differ, or if the column widths
    /// do not match expectations.
    pub fn convert_measurements_and_sweep_bits(
        &mut self,
        measurements: ArrayView2<'_, bool>,
        sweep_bits: ArrayView2<'_, bool>,
        append_observables: bool,
    ) -> Result<Array2<bool>> {
        if measurements.nrows() != sweep_bits.nrows() {
            return Err(StimError::new(format!(
                "expected equal shot counts for measurements and sweep bits, got {} and {}",
                measurements.nrows(),
                sweep_bits.nrows()
            )));
        }
        let packed_measurements = pack_rows_array(measurements, self.num_measurements() as usize)?;
        let packed_sweep_bits = pack_rows_array(sweep_bits, self.num_sweep_bits() as usize)?;
        let converted = self.convert_measurements_and_sweep_bits_bit_packed(
            &packed_measurements,
            &packed_sweep_bits,
            measurements.nrows() as u64,
            append_observables,
        );
        let bit_len = self.num_detectors() as usize
            + self.num_observables() as usize * usize::from(append_observables);
        Ok(unpack_rows_array(&converted, bit_len))
    }

    /// Converts unpacked measurements plus sweep bits, returning unpacked
    /// detectors and observables separately.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the shot counts of
    /// `measurements` and `sweep_bits` differ.
    pub fn convert_measurements_and_sweep_bits_separate_observables(
        &mut self,
        measurements: ArrayView2<'_, bool>,
        sweep_bits: ArrayView2<'_, bool>,
    ) -> Result<UnpackedObservablePair> {
        if measurements.nrows() != sweep_bits.nrows() {
            return Err(StimError::new(format!(
                "expected equal shot counts for measurements and sweep bits, got {} and {}",
                measurements.nrows(),
                sweep_bits.nrows()
            )));
        }
        let packed_measurements = pack_rows_array(measurements, self.num_measurements() as usize)?;
        let packed_sweep_bits = pack_rows_array(sweep_bits, self.num_sweep_bits() as usize)?;
        let (dets, obs) = self.convert_measurements_and_sweep_bits_bit_packed_separate_observables(
            &packed_measurements,
            &packed_sweep_bits,
            measurements.nrows() as u64,
        );
        Ok((
            unpack_rows_array(&dets, self.num_detectors() as usize),
            unpack_rows_array(&obs, self.num_observables() as usize),
        ))
    }

    /// Converts bit-packed measurements, returning bit-packed detectors and
    /// observables as a separate `(detectors, observables)` pair.
    pub fn convert_measurements_bit_packed_separate_observables(
        &mut self,
        measurements: &[u8],
        shots: u64,
    ) -> PackedObservablePair {
        (
            self.convert_measurements_bit_packed(measurements, shots, false),
            self.inner
                .convert_observables_bit_packed(measurements, shots),
        )
    }

    /// Converts unpacked measurements into unpacked detector-event data.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the column width of
    /// `measurements` does not match [`num_measurements`](Self::num_measurements).
    pub fn convert_measurements(
        &mut self,
        measurements: ArrayView2<'_, bool>,
        append_observables: bool,
    ) -> Result<Array2<bool>> {
        let packed = pack_rows_array(measurements, self.num_measurements() as usize)?;
        let converted = self.convert_measurements_bit_packed(
            &packed,
            measurements.nrows() as u64,
            append_observables,
        );
        let bit_len = self.num_detectors() as usize
            + self.num_observables() as usize * usize::from(append_observables);
        Ok(unpack_rows_array(&converted, bit_len))
    }

    /// Converts unpacked measurements, returning unpacked detectors and
    /// observables separately.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if the column width of
    /// `measurements` does not match [`num_measurements`](Self::num_measurements).
    pub fn convert_measurements_separate_observables(
        &mut self,
        measurements: ArrayView2<'_, bool>,
    ) -> Result<UnpackedObservablePair> {
        let packed = pack_rows_array(measurements, self.num_measurements() as usize)?;
        let (dets, obs) = self.convert_measurements_bit_packed_separate_observables(
            &packed,
            measurements.nrows() as u64,
        );
        Ok((
            unpack_rows_array(&dets, self.num_detectors() as usize),
            unpack_rows_array(&obs, self.num_observables() as usize),
        ))
    }

    /// Returns the number of measurement bits consumed per shot during conversion.
    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        self.inner.num_measurements()
    }

    /// Returns the number of detector bits produced per shot during conversion.
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        self.inner.num_detectors()
    }

    /// Returns the number of observable bits produced per shot during conversion.
    #[must_use]
    pub fn num_observables(&self) -> u64 {
        self.inner.num_observables()
    }

    /// Returns the number of sweep bits consumed per shot during conversion.
    ///
    /// Sweep bits correspond to `sweep[k]` controls in the circuit and can
    /// vary from shot to shot.
    #[must_use]
    pub fn num_sweep_bits(&self) -> u64 {
        self.inner.num_sweep_bits()
    }

    /// Reads measurement data from a file and writes detection events to another file.
    ///
    /// This is the file-to-file equivalent of the in-memory
    /// [`convert`](Self::convert) method. Supports optional sweep-bit input
    /// and separate observable output.
    ///
    /// # Arguments
    ///
    /// * `measurements_filepath` - Path to the file containing measurement data.
    /// * `measurements_format` - Format of the measurement file (`"01"`, `"b8"`, etc.).
    /// * `sweep_bits_filepath` - Optional path to sweep-bit data.
    /// * `sweep_bits_format` - Format of the sweep-bit file.
    /// * `detection_events_filepath` - Where to write detection-event output.
    /// * `detection_events_format` - Format for the detection-event output.
    /// * `append_observables` - When `true`, observables are appended as
    ///   additional detectors in the output.
    /// * `obs_out_filepath` - Optional path for separate observable output.
    /// * `obs_out_format` - Format for the observable output.
    ///
    /// # Errors
    ///
    /// Returns a [`StimError`](crate::StimError) if any file path is not valid
    /// UTF-8 or an I/O error occurs.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit =
    ///     "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]".parse().unwrap();
    /// let mut converter = circuit.compile_m2d_converter(false);
    ///
    /// let measurements_path = std::env::temp_dir().join("stim-rs-m2d-measurements.01");
    /// let detections_path = std::env::temp_dir().join("stim-rs-m2d-detections.01");
    /// std::fs::write(&measurements_path, "0\n1\n").unwrap();
    ///
    /// converter
    ///     .convert_file(
    ///         &measurements_path,
    ///         "01",
    ///         None::<&std::path::Path>,
    ///         "01",
    ///         &detections_path,
    ///         "01",
    ///         false,
    ///         None::<&std::path::Path>,
    ///         "01",
    ///     )
    ///     .unwrap();
    ///
    /// assert_eq!(std::fs::read_to_string(&detections_path).unwrap(), "1\n0\n");
    /// std::fs::remove_file(measurements_path).unwrap();
    /// std::fs::remove_file(detections_path).unwrap();
    /// ```
    pub fn convert_file<SweepPath, ObsPath>(
        &mut self,
        measurements_filepath: impl AsRef<Path>,
        measurements_format: &str,
        sweep_bits_filepath: Option<SweepPath>,
        sweep_bits_format: &str,
        detection_events_filepath: impl AsRef<Path>,
        detection_events_format: &str,
        append_observables: bool,
        obs_out_filepath: Option<ObsPath>,
        obs_out_format: &str,
    ) -> Result<()>
    where
        SweepPath: AsRef<Path>,
        ObsPath: AsRef<Path>,
    {
        let measurements_path = measurements_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("measurement filepath must be valid UTF-8"))?;
        let sweep_bits_path = sweep_bits_filepath
            .as_ref()
            .map(|path| {
                path.as_ref()
                    .to_str()
                    .ok_or_else(|| StimError::new("sweep-bit filepath must be valid UTF-8"))
            })
            .transpose()?
            .unwrap_or("");
        let detection_events_path = detection_events_filepath
            .as_ref()
            .to_str()
            .ok_or_else(|| StimError::new("detection-event filepath must be valid UTF-8"))?;
        let obs_out_path = obs_out_filepath
            .as_ref()
            .map(|path| {
                path.as_ref()
                    .to_str()
                    .ok_or_else(|| StimError::new("observable filepath must be valid UTF-8"))
            })
            .transpose()?
            .unwrap_or("");
        self.inner
            .convert_file(
                measurements_path,
                measurements_format,
                sweep_bits_path,
                sweep_bits_format,
                detection_events_path,
                detection_events_format,
                append_observables,
                obs_out_path,
                obs_out_format,
            )
            .map_err(StimError::from)
    }
}

impl fmt::Debug for MeasurementsToDetectionEventsConverter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MeasurementsToDetectionEventsConverter")
            .field("num_measurements", &self.num_measurements())
            .field("num_detectors", &self.num_detectors())
            .field("num_observables", &self.num_observables())
            .field("num_sweep_bits", &self.num_sweep_bits())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::{fs, path::PathBuf, time::SystemTime};

    use ndarray::Array2;

    use super::{BitTable, FlipSimulator, FlipSimulatorArrays, TableauSimulator};
    use crate::{
        Circuit, CircuitInstruction, CircuitRepeatBlock, ConvertedMeasurements, DetectorErrorModel,
        MeasurementSampler, MeasurementsToDetectionEventsConverter, PauliString, Tableau,
    };

    fn bool_matrix(rows: Vec<Vec<bool>>) -> Array2<bool> {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, Vec::len);
        Array2::from_shape_vec((nrows, ncols), rows.into_iter().flatten().collect())
            .expect("rows should be rectangular")
    }

    fn unique_temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        std::env::temp_dir().join(format!("stim-rs-{name}-{nanos}.01"))
    }

    fn unpack_rows(packed: &[u8], bits_per_row: usize) -> Vec<Vec<bool>> {
        packed
            .iter()
            .map(|byte| {
                (0..bits_per_row)
                    .map(|bit| ((byte >> bit) & 1) == 1)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn render_01(rows: &[Vec<bool>]) -> String {
        let mut text = String::new();
        for row in rows {
            for bit in row {
                text.push(if *bit { '1' } else { '0' });
            }
            text.push('\n');
        }
        text
    }

    fn render_01_array(rows: &Array2<bool>) -> String {
        let mut text = String::new();
        for row in rows.rows() {
            for bit in row {
                text.push(if *bit { '1' } else { '0' });
            }
            text.push('\n');
        }
        text
    }

    #[test]
    fn compiled_sampler_debugs_include_basic_shape_information() {
        let circuit =
            Circuit::from_str("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]").unwrap();

        let sampler = circuit.compile_sampler(false);
        let detector_sampler = circuit.compile_detector_sampler();
        let converter = circuit.compile_m2d_converter(false);

        assert!(format!("{sampler:?}").contains("MeasurementSampler"));
        assert!(format!("{detector_sampler:?}").contains("DetectorSampler"));
        assert!(format!("{converter:?}").contains("MeasurementsToDetectionEventsConverter"));
    }

    #[test]
    fn measurement_and_detector_sampler_direct_constructors_cover_remaining_methods() {
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]").unwrap();

        let mut measurement_sampler = circuit.compile_sampler_with_seed(false, 0);
        assert_eq!(measurement_sampler.num_measurements(), 1);
        assert_eq!(measurement_sampler.sample_bit_packed(2), vec![1, 1]);
        assert_eq!(
            measurement_sampler.sample(2),
            bool_matrix(vec![vec![true]; 2])
        );
        assert!(format!("{measurement_sampler:?}").contains("MeasurementSampler"));

        let mut detector_sampler = circuit.compile_detector_sampler_with_seed(0);
        assert_eq!(detector_sampler.num_detectors(), 1);
        assert_eq!(detector_sampler.num_observables(), 1);
        assert_eq!(detector_sampler.sample_bit_packed(2), vec![0, 0]);
        assert_eq!(
            detector_sampler.sample_observables_bit_packed(2),
            vec![0, 0]
        );
        assert_eq!(
            detector_sampler.sample(2),
            bool_matrix(vec![vec![false]; 2])
        );
        assert!(format!("{detector_sampler:?}").contains("DetectorSampler"));
    }

    #[test]
    fn tableau_simulator_core_state_methods_match_documented_examples() {
        let mut s = TableauSimulator::new();
        s.h(&[0]).unwrap();
        assert_eq!(
            s.current_inverse_tableau(),
            Tableau::from_conjugated_generators(
                &[PauliString::from_text("+Z").unwrap()],
                &[PauliString::from_text("+X").unwrap()],
            )
            .unwrap()
        );

        assert_eq!(s.current_measurement_record(), Vec::<bool>::new());
        assert!(!s.measure(0));
        s.x(&[0]).unwrap();
        assert!(s.measure(0));
        assert_eq!(s.current_measurement_record(), vec![false, true]);

        let mut s2 = s.clone();
        assert_eq!(s2.current_inverse_tableau(), s.current_inverse_tableau());
        s2.set_num_qubits(3);
        assert_eq!(s2.num_qubits(), 3);
    }

    #[test]
    fn tableau_simulator_do_and_gate_wrappers_match_documented_examples() {
        let mut s = TableauSimulator::new();
        s.r#do(&Circuit::from_str("X 0\nM 0").unwrap()).unwrap();
        assert_eq!(s.current_measurement_record(), vec![true]);

        let mut s = TableauSimulator::new();
        s.do_pauli_string(&PauliString::from_text("IXYZ").unwrap());
        assert_eq!(
            s.measure_many(&[0, 1, 2, 3]),
            vec![false, true, true, false]
        );

        let mut sim = TableauSimulator::new();
        sim.h(&[1]).unwrap();
        sim.h_yz(&[2]).unwrap();
        assert_eq!(
            (0..4)
                .map(|k| sim.peek_bloch(k).to_string())
                .collect::<Vec<_>>(),
            vec!["+Z", "+X", "+Y", "+Z"]
        );
        let rot3 = Tableau::from_conjugated_generators(
            &[
                PauliString::from_text("_X_").unwrap(),
                PauliString::from_text("__X").unwrap(),
                PauliString::from_text("X__").unwrap(),
            ],
            &[
                PauliString::from_text("_Z_").unwrap(),
                PauliString::from_text("__Z").unwrap(),
                PauliString::from_text("Z__").unwrap(),
            ],
        )
        .unwrap();
        sim.do_tableau(&rot3, &[1, 2, 3]);
        assert_eq!(
            (0..4)
                .map(|k| sim.peek_bloch(k).to_string())
                .collect::<Vec<_>>(),
            vec!["+Z", "+Z", "+X", "+Y"]
        );
    }

    #[test]
    fn tableau_simulator_convenience_paths_cover_remaining_methods() {
        let default_sim = TableauSimulator::default();
        assert_eq!(default_sim.num_qubits(), 0);
        assert!(format!("{default_sim:?}").contains("measurement_record_len"));

        let mut pauli_sim = TableauSimulator::with_seed(5);
        pauli_sim
            .r#do(&PauliString::from_text("X").unwrap())
            .unwrap();
        assert_eq!(pauli_sim.measure_many(&[0]), vec![true]);

        let mut instruction_sim = TableauSimulator::new();
        let instruction = CircuitInstruction::from_stim_program_text("H 0").unwrap();
        instruction_sim.r#do(&instruction).unwrap();
        assert_eq!(
            instruction_sim.peek_bloch(0),
            PauliString::from_text("+X").unwrap()
        );

        let mut repeat_sim = TableauSimulator::new();
        let repeat_body: Circuit = "X 0".parse().unwrap();
        let repeat_block = CircuitRepeatBlock::new(1, &repeat_body, "tag").unwrap();
        repeat_sim.r#do(&repeat_block).unwrap();
        assert_eq!(repeat_sim.measure_many(&[0]), vec![true]);

        let mut sim = TableauSimulator::with_seed(7);
        sim.h_xy(&[0]).unwrap();
        sim.h_xz(&[1]).unwrap();
        sim.c_xyz(&[2]).unwrap();
        sim.c_zyx(&[3]).unwrap();
        sim.y(&[4]).unwrap();
        sim.s(&[5]).unwrap();
        sim.s_dag(&[5]).unwrap();
        sim.sqrt_x(&[6]).unwrap();
        sim.sqrt_x_dag(&[6]).unwrap();
        sim.sqrt_y(&[7]).unwrap();
        sim.sqrt_y_dag(&[7]).unwrap();
        sim.reset(&[8]).unwrap();
        sim.reset_z(&[9]).unwrap();
        sim.cx(&[0, 1]).unwrap();
        sim.cnot(&[2, 3]).unwrap();
        sim.cy(&[4, 5]).unwrap();
        sim.swap(&[6, 7]).unwrap();
        sim.iswap(&[8, 9]).unwrap();
        sim.iswap_dag(&[8, 9]).unwrap();
        sim.xcx(&[0, 2]).unwrap();
        sim.xcy(&[1, 3]).unwrap();
        sim.xcz(&[4, 6]).unwrap();
        sim.ycx(&[5, 7]).unwrap();
        sim.ycy(&[0, 8]).unwrap();
        sim.zcx(&[1, 9]).unwrap();
        sim.zcy(&[2, 4]).unwrap();
        sim.zcz(&[3, 5]).unwrap();
        sim.x_error(&[0], 0.25).unwrap();
        sim.y_error(&[1], 0.25).unwrap();
        sim.z_error(&[2], 0.25).unwrap();

        let copy = sim.clone();
        assert_eq!(copy.num_qubits(), sim.num_qubits());
        assert!(format!("{copy:?}").contains("measurement_record_len"));
    }

    #[test]
    fn tableau_simulator_peek_methods_match_documented_examples() {
        let mut s = TableauSimulator::new();
        assert_eq!(s.peek_bloch(0), PauliString::from_text("+Z").unwrap());
        s.x(&[0]).unwrap();
        assert_eq!(s.peek_bloch(0), PauliString::from_text("-Z").unwrap());
        s.h(&[0]).unwrap();
        assert_eq!(s.peek_bloch(0), PauliString::from_text("-X").unwrap());

        let mut s = TableauSimulator::new();
        s.reset_z(&[0]).unwrap();
        assert_eq!(s.peek_x(0), 0);
        s.reset_x(&[0]).unwrap();
        assert_eq!(s.peek_x(0), 1);
        s.z(&[0]).unwrap();
        assert_eq!(s.peek_x(0), -1);
    }

    #[test]
    fn tableau_simulator_extended_gate_wrappers_match_documented_examples() {
        let mut s = TableauSimulator::new();
        s.reset_x(&[0, 3]).unwrap();
        s.reset_y(&[1]).unwrap();
        assert_eq!(
            (0..4)
                .map(|k| s.peek_bloch(k).to_string())
                .collect::<Vec<_>>(),
            vec!["+X", "+Y", "+Z", "+X"]
        );
        s.cz(&[0, 1, 2, 3]).unwrap();
        assert_eq!(
            (0..4)
                .map(|k| s.peek_bloch(k).to_string())
                .collect::<Vec<_>>(),
            vec!["+_", "+_", "+Z", "+X"]
        );

        let mut s = TableauSimulator::new();
        s.reset_x(&[0, 3]).unwrap();
        s.reset_y(&[1]).unwrap();
        s.ycz(&[0, 1, 2, 3]).unwrap();
        assert_eq!(
            (0..4)
                .map(|k| s.peek_bloch(k).to_string())
                .collect::<Vec<_>>(),
            vec!["+_", "+_", "+_", "+_"]
        );
    }

    #[test]
    fn tableau_simulator_noise_wrappers_accept_documented_shapes() {
        let mut s = TableauSimulator::with_seed(0);
        s.depolarize1(&[0, 1, 2], 0.01).unwrap();
        s.depolarize2(&[0, 1, 4, 5], 0.01).unwrap();
        s.x_error(&[0], 0.5).unwrap();
        s.y_error(&[1], 0.5).unwrap();
        s.z_error(&[2], 0.5).unwrap();
        assert_eq!(s.num_qubits(), 6);
    }

    #[test]
    fn tableau_simulator_observable_and_postselect_methods_match_documented_examples() {
        let mut s = TableauSimulator::new();
        assert_eq!(
            s.peek_observable_expectation(&PauliString::from_text("+Z").unwrap())
                .unwrap(),
            1
        );
        assert_eq!(
            s.peek_observable_expectation(&PauliString::from_text("+X").unwrap())
                .unwrap(),
            0
        );
        assert_eq!(
            s.peek_observable_expectation(&PauliString::from_text("-Z").unwrap())
                .unwrap(),
            -1
        );

        s.r#do(&Circuit::from_str("H 0\nCNOT 0 1").unwrap())
            .unwrap();
        let xx = s
            .measure_observable(&PauliString::from_text("XX").unwrap(), 0.5)
            .unwrap();
        let yy = s
            .measure_observable(&PauliString::from_text("YY").unwrap(), 0.5)
            .unwrap();
        let zz = s
            .measure_observable(&PauliString::from_text("-ZZ").unwrap(), 0.5)
            .unwrap();
        assert!(matches!(xx, true | false));
        assert!(matches!(yy, true | false));
        assert!(matches!(zz, true | false));

        let mut s = TableauSimulator::new();
        s.postselect_x(&[0], false).unwrap();
        assert_eq!(s.peek_x(0), 1);
        s.h(&[0]).unwrap();
        let mut s2 = TableauSimulator::new();
        s2.h(&[0]).unwrap();
        s2.postselect_z(&[0], true).unwrap();
        assert_eq!(s2.peek_z(0), -1);
        s.postselect_y(&[1], false).unwrap();
        assert_eq!(s.peek_y(1), 1);

        let mut s = TableauSimulator::new();
        s.postselect_observable(&PauliString::from_text("+XX").unwrap(), false)
            .unwrap();
        s.postselect_observable(&PauliString::from_text("+ZZ").unwrap(), false)
            .unwrap();
        assert_eq!(
            s.peek_observable_expectation(&PauliString::from_text("+YY").unwrap())
                .unwrap(),
            -1
        );
    }

    #[test]
    fn tableau_simulator_state_and_kickback_methods_match_documented_examples() {
        let mut s = TableauSimulator::new();
        assert_eq!(s.measure_kickback(0), (false, None));
        s.h(&[0]).unwrap();
        assert_eq!(
            s.measure_kickback(0).1,
            Some(PauliString::from_text("+X").unwrap())
        );

        let mut tab_sim = TableauSimulator::new();
        tab_sim
            .set_state_from_stabilizers(
                &[
                    PauliString::from_text("XX").unwrap(),
                    PauliString::from_text("ZZ").unwrap(),
                ],
                false,
                false,
            )
            .unwrap();
        assert_eq!(
            tab_sim.current_inverse_tableau().inverse(false),
            Tableau::from_conjugated_generators(
                &[
                    PauliString::from_text("+Z_").unwrap(),
                    PauliString::from_text("+_X").unwrap(),
                ],
                &[
                    PauliString::from_text("+XX").unwrap(),
                    PauliString::from_text("+ZZ").unwrap(),
                ],
            )
            .unwrap()
        );

        let mut tab_sim = TableauSimulator::new();
        tab_sim
            .set_state_from_state_vector(
                &[
                    crate::Complex32::new(0.5_f32.sqrt(), 0.0),
                    crate::Complex32::new(0.0, 0.5_f32.sqrt()),
                ],
                "little",
            )
            .unwrap();
        assert_eq!(
            tab_sim.current_inverse_tableau().inverse(false),
            Tableau::from_conjugated_generators(
                &[PauliString::from_text("+Z").unwrap()],
                &[PauliString::from_text("+Y").unwrap()],
            )
            .unwrap()
        );

        let mut s = TableauSimulator::new();
        s.h(&[0]).unwrap();
        s.h(&[1]).unwrap();
        let state = s.state_vector("little").unwrap();
        assert_eq!(state.len(), 4);
        assert_eq!(s.canonical_stabilizers().len(), 2);
    }

    #[test]
    fn measurement_sampler_keeps_packed_unpacked_and_written_outputs_in_sync() {
        let circuit = Circuit::from_str("X 0\nM 0 1").expect("circuit should parse");
        let mut packed_sampler = circuit.compile_sampler(false);
        let mut unpacked_sampler = circuit.compile_sampler(false);
        let mut file_sampler = circuit.compile_sampler(false);
        let path = unique_temp_path("measurement-sampler");

        assert_eq!(packed_sampler.num_measurements(), 2);
        assert_eq!(unpacked_sampler.num_measurements(), 2);
        assert_eq!(file_sampler.num_measurements(), 2);

        let packed = packed_sampler.sample_bit_packed(3);
        let unpacked = unpacked_sampler.sample(3);

        assert_eq!(packed, vec![0b0000_0001; 3]);
        assert_eq!(bool_matrix(unpack_rows(&packed, 2)), unpacked);
        assert_eq!(unpacked, bool_matrix(vec![vec![true, false]; 3]));

        file_sampler
            .sample_write(3, &path, "01")
            .expect("sample_write should succeed");

        assert_eq!(
            fs::read_to_string(&path).expect("sample file should read"),
            render_01_array(&unpacked)
        );

        fs::remove_file(path).expect("sample file should delete");
    }

    #[test]
    fn measurement_sampler_with_seed_is_repeatable_across_output_forms() {
        let circuit = Circuit::from_str("X_ERROR(0.5) 0\nM 0 1").expect("circuit should parse");
        let mut packed_sampler = circuit.compile_sampler_with_seed(false, 5);
        let mut repeat_sampler = circuit.compile_sampler_with_seed(false, 5);
        let mut unpacked_sampler = circuit.compile_sampler_with_seed(false, 5);
        let mut file_sampler = circuit.compile_sampler_with_seed(false, 5);
        let path = unique_temp_path("measurement-sampler-seeded");

        assert_eq!(packed_sampler.num_measurements(), 2);

        let packed = packed_sampler.sample_bit_packed(8);
        let repeat_packed = repeat_sampler.sample_bit_packed(8);
        let unpacked = unpacked_sampler.sample(8);

        assert_eq!(packed, repeat_packed);
        assert_eq!(bool_matrix(unpack_rows(&packed, 2)), unpacked);

        file_sampler
            .sample_write(8, &path, "01")
            .expect("sample_write should succeed");

        assert_eq!(
            fs::read_to_string(&path).expect("sample file should read"),
            render_01_array(&unpacked)
        );

        fs::remove_file(path).expect("sample file should delete");
    }

    #[test]
    fn detector_sampler_with_seed_repeats_noisy_parity_samples() {
        let circuit = Circuit::from_str(
                "X_ERROR(0.5) 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]",
            )
            .expect("circuit should parse");

        let mut packed_a = circuit.compile_detector_sampler_with_seed(7);
        let mut packed_b = circuit.compile_detector_sampler_with_seed(7);

        assert_eq!(packed_a.num_detectors(), 2);
        assert_eq!(packed_a.num_observables(), 1);
        assert_eq!(packed_a.sample_bit_packed(8), packed_b.sample_bit_packed(8));

        let mut separate_a = circuit.compile_detector_sampler_with_seed(7);
        let mut separate_b = circuit.compile_detector_sampler_with_seed(7);

        assert_eq!(
            separate_a.sample_bit_packed_separate_observables(8),
            separate_b.sample_bit_packed_separate_observables(8)
        );

        let mut unpacked_a = circuit.compile_detector_sampler_with_seed(7);
        let mut unpacked_b = circuit.compile_detector_sampler_with_seed(7);

        assert_eq!(
            unpacked_a.sample_separate_observables(4),
            unpacked_b.sample_separate_observables(4)
        );
    }

    #[test]
    fn detector_sampler_helpers_and_writes_stay_consistent() {
        let circuit = Circuit::from_str(
                "X_ERROR(0.5) 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]",
            )
            .expect("circuit should parse");
        let shots = 6;

        let mut packed_sampler = circuit.compile_detector_sampler_with_seed(11);
        let packed_detectors = packed_sampler.sample_bit_packed(shots);

        let mut unpacked_sampler = circuit.compile_detector_sampler_with_seed(11);
        let expected_detectors = unpacked_sampler.sample(shots);
        assert_eq!(
            expected_detectors,
            bool_matrix(unpack_rows(&packed_detectors, 2))
        );

        let mut separate_packed = circuit.compile_detector_sampler_with_seed(11);
        let (packed_separate_detectors, packed_observables) =
            separate_packed.sample_bit_packed_separate_observables(shots);

        let mut separate_unpacked = circuit.compile_detector_sampler_with_seed(11);
        let (expected_detectors, expected_observables) =
            separate_unpacked.sample_separate_observables(shots);
        assert_eq!(
            expected_detectors,
            bool_matrix(unpack_rows(&packed_separate_detectors, 2))
        );
        assert_eq!(
            expected_observables,
            bool_matrix(unpack_rows(&packed_observables, 1))
        );

        let expected_prepend = ndarray::concatenate(
            ndarray::Axis(1),
            &[expected_observables.view(), expected_detectors.view()],
        )
        .unwrap();
        let expected_append = ndarray::concatenate(
            ndarray::Axis(1),
            &[expected_detectors.view(), expected_observables.view()],
        )
        .unwrap();

        let mut prepend = circuit.compile_detector_sampler_with_seed(11);
        assert_eq!(prepend.sample_prepend_observables(shots), expected_prepend);

        let mut append = circuit.compile_detector_sampler_with_seed(11);
        assert_eq!(append.sample_append_observables(shots), expected_append);

        let file_circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
                .expect("file-write circuit should parse");
        let file_rows = vec![vec![false]; shots as usize];

        let shots_path = unique_temp_path("parity-detector-sampler-shots");
        let dets_path = unique_temp_path("parity-detector-sampler-dets");
        let obs_path = unique_temp_path("parity-detector-sampler-obs");

        let mut writer = file_circuit.compile_detector_sampler_with_seed(11);
        writer
            .sample_write(shots, &shots_path, "01")
            .expect("sample_write should succeed");
        assert_eq!(
            fs::read_to_string(&shots_path).expect("shot file should read"),
            render_01(&file_rows)
        );

        let mut separate_writer = file_circuit.compile_detector_sampler_with_seed(11);
        separate_writer
            .sample_write_separate_observables(shots, &dets_path, "01", &obs_path, "01")
            .expect("sample_write_separate_observables should succeed");
        assert_eq!(
            fs::read_to_string(&dets_path).expect("detector file should read"),
            render_01(&file_rows)
        );
        assert_eq!(
            fs::read_to_string(&obs_path).expect("observable file should read"),
            render_01(&file_rows)
        );

        for path in [shots_path, dets_path, obs_path] {
            fs::remove_file(path).expect("temp file should delete");
        }
    }

    #[test]
    fn detector_sampler_separate_observables_stay_shot_paired() {
        let circuit = Circuit::from_str(
            "X_ERROR(0.5) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        )
        .expect("circuit should parse");
        let shots = 16;

        let mut separate = circuit.compile_detector_sampler_with_seed(7);
        let (detectors, observables) = separate.sample_separate_observables(shots);

        assert_eq!(detectors, observables);
    }

    fn m2d_circuit() -> Circuit {
        Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
            .expect("circuit should parse")
    }

    fn sweep_m2d_circuit() -> Circuit {
        Circuit::from_str(
            "X 0\nCNOT sweep[0] 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        )
        .expect("circuit should parse")
    }

    #[test]
    fn m2d_converter_converts_packed_and_unpacked_measurements() {
        let mut converter = m2d_circuit().compile_m2d_converter(false);

        assert_eq!(converter.num_measurements(), 1);
        assert_eq!(converter.num_detectors(), 1);
        assert_eq!(converter.num_observables(), 1);
        assert_eq!(converter.num_sweep_bits(), 0);

        assert_eq!(
            converter.convert_measurements_bit_packed(&[0b0, 0b1], 2, false),
            vec![0b1, 0b0]
        );
        assert_eq!(
            converter.convert_measurements_bit_packed(&[0b0, 0b1], 2, true),
            vec![0b11, 0b00]
        );

        let (packed_dets, packed_obs) =
            converter.convert_measurements_bit_packed_separate_observables(&[0b0, 0b1], 2);
        assert_eq!(packed_dets, vec![0b1, 0b0]);
        assert_eq!(packed_obs, vec![0b1, 0b0]);

        assert_eq!(
            converter
                .convert_measurements(bool_matrix(vec![vec![false], vec![true]]).view(), false)
                .expect("conversion should succeed"),
            bool_matrix(vec![vec![true], vec![false]])
        );
        assert_eq!(
            converter
                .convert_measurements(bool_matrix(vec![vec![false], vec![true]]).view(), true)
                .expect("conversion should succeed"),
            bool_matrix(vec![vec![true, true], vec![false, false]])
        );

        let (dets, obs) = converter
            .convert_measurements_separate_observables(
                bool_matrix(vec![vec![false], vec![true]]).view(),
            )
            .expect("conversion should succeed");
        assert_eq!(dets, bool_matrix(vec![vec![true], vec![false]]));
        assert_eq!(obs, bool_matrix(vec![vec![true], vec![false]]));
    }

    #[test]
    fn m2d_converter_supports_sweep_bits_in_packed_and_unpacked_forms() {
        let mut converter = sweep_m2d_circuit().compile_m2d_converter(false);

        assert_eq!(converter.num_measurements(), 1);
        assert_eq!(converter.num_detectors(), 1);
        assert_eq!(converter.num_observables(), 1);
        assert_eq!(converter.num_sweep_bits(), 1);

        let measurements = [0b0, 0b1, 0b0, 0b1];
        let sweep_bits = [0b0, 0b0, 0b1, 0b1];

        assert_eq!(
            converter.convert_measurements_and_sweep_bits_bit_packed(
                &measurements,
                &sweep_bits,
                4,
                false,
            ),
            vec![0b1, 0b0, 0b0, 0b1]
        );
        assert_eq!(
            converter.convert_measurements_and_sweep_bits_bit_packed(
                &measurements,
                &sweep_bits,
                4,
                true,
            ),
            vec![0b11, 0b00, 0b00, 0b11]
        );

        let (packed_dets, packed_obs) = converter
            .convert_measurements_and_sweep_bits_bit_packed_separate_observables(
                &measurements,
                &sweep_bits,
                4,
            );
        assert_eq!(packed_dets, vec![0b1, 0b0, 0b0, 0b1]);
        assert_eq!(packed_obs, vec![0b1, 0b0, 0b0, 0b1]);

        let unpacked_measurements = vec![vec![false], vec![true], vec![false], vec![true]];
        let unpacked_sweep_bits = vec![vec![false], vec![false], vec![true], vec![true]];

        assert_eq!(
            converter
                .convert_measurements_and_sweep_bits(
                    bool_matrix(unpacked_measurements.clone()).view(),
                    bool_matrix(unpacked_sweep_bits.clone()).view(),
                    true,
                )
                .expect("conversion should succeed"),
            bool_matrix(vec![
                vec![true, true],
                vec![false, false],
                vec![false, false],
                vec![true, true],
            ])
        );

        let (dets, obs) = converter
            .convert_measurements_and_sweep_bits_separate_observables(
                bool_matrix(unpacked_measurements).view(),
                bool_matrix(unpacked_sweep_bits).view(),
            )
            .expect("conversion should succeed");
        assert_eq!(
            dets,
            bool_matrix(vec![vec![true], vec![false], vec![false], vec![true]])
        );
        assert_eq!(
            obs,
            bool_matrix(vec![vec![true], vec![false], vec![false], vec![true]])
        );
    }

    #[test]
    fn m2d_converter_can_convert_files_with_appended_and_separate_observables() {
        let measurements_path = unique_temp_path("m2d-measurements");
        let sweep_bits_path = unique_temp_path("m2d-sweep-bits");
        let detections_path = unique_temp_path("m2d-detections");
        let appended_path = unique_temp_path("m2d-appended");
        let obs_path = unique_temp_path("m2d-obs");
        let mut converter = sweep_m2d_circuit().compile_m2d_converter(false);

        fs::write(&measurements_path, "0\n1\n0\n1\n").expect("measurement file should write");
        fs::write(&sweep_bits_path, "0\n0\n1\n1\n").expect("sweep-bit file should write");

        converter
            .convert_file(
                &measurements_path,
                "01",
                Some(&sweep_bits_path),
                "01",
                &detections_path,
                "01",
                false,
                Some(&obs_path),
                "01",
            )
            .expect("convert_file should succeed");
        assert_eq!(
            fs::read_to_string(&detections_path).expect("detection file should read"),
            "1\n0\n0\n1\n"
        );
        assert_eq!(
            fs::read_to_string(&obs_path).expect("observable file should read"),
            "1\n0\n0\n1\n"
        );

        converter
            .convert_file(
                &measurements_path,
                "01",
                Some(&sweep_bits_path),
                "01",
                &appended_path,
                "01",
                true,
                None::<&std::path::Path>,
                "01",
            )
            .expect("convert_file should append observables");
        assert_eq!(
            fs::read_to_string(&appended_path).expect("appended file should read"),
            "11\n00\n00\n11\n"
        );

        for path in [
            measurements_path,
            sweep_bits_path,
            detections_path,
            appended_path,
            obs_path,
        ] {
            fs::remove_file(path).expect("temp file should delete");
        }
    }

    #[test]
    fn m2d_converter_convert_matches_existing_unpacked_helpers() {
        let mut converter = m2d_circuit().compile_m2d_converter(false);

        assert_eq!(
            converter
                .convert(
                    bool_matrix(vec![vec![false], vec![true]]).view(),
                    None,
                    false,
                    false,
                )
                .unwrap(),
            ConvertedMeasurements::DetectionEvents(bool_matrix(vec![vec![true], vec![false]]))
        );

        assert_eq!(
            converter
                .convert(
                    bool_matrix(vec![vec![false], vec![true]]).view(),
                    None,
                    true,
                    false,
                )
                .unwrap(),
            ConvertedMeasurements::DetectionEventsAndObservables(
                bool_matrix(vec![vec![true], vec![false]]),
                bool_matrix(vec![vec![true], vec![false]]),
            )
        );

        let mut converter = sweep_m2d_circuit().compile_m2d_converter(false);
        assert_eq!(
            converter
                .convert(
                    bool_matrix(vec![vec![false], vec![true], vec![false], vec![true]]).view(),
                    Some(
                        bool_matrix(vec![vec![false], vec![false], vec![true], vec![true]]).view()
                    ),
                    true,
                    false,
                )
                .unwrap(),
            ConvertedMeasurements::DetectionEventsAndObservables(
                bool_matrix(vec![vec![true], vec![false], vec![false], vec![true]]),
                bool_matrix(vec![vec![true], vec![false], vec![false], vec![true]]),
            )
        );
    }

    #[test]
    fn flip_simulator_core_shape_and_pauli_flips_match_documented_examples() {
        let mut sim = FlipSimulator::new(2, true, 3, 0);
        assert_eq!(sim.batch_size(), 2);
        assert_eq!(sim.num_qubits(), 3);
        assert_eq!(sim.num_measurements(), 0);
        assert_eq!(sim.num_detectors(), 0);
        assert_eq!(sim.num_observables(), 0);

        sim.set_pauli_flip('X', 2, 1).unwrap();
        assert_eq!(
            sim.peek_pauli_flips().unwrap(),
            vec![
                PauliString::from_text("+___").unwrap(),
                PauliString::from_text("+__X").unwrap()
            ]
        );
        assert_eq!(
            sim.peek_pauli_flip(1).unwrap(),
            PauliString::from_text("+__X").unwrap()
        );
    }

    #[test]
    fn flip_simulator_measurement_append_and_record_access_match_examples() {
        let mut sim = FlipSimulator::new(9, true, 0, 0);
        sim.append_measurement_flips(
            bool_matrix(vec![vec![
                false, true, false, false, true, false, false, true, true,
            ]])
            .view(),
        )
        .unwrap();
        assert_eq!(
            sim.get_measurement_flips(false),
            BitTable::BoolMatrix(ndarray::array![[
                false, true, false, false, true, false, false, true, true
            ]])
        );

        sim.append_measurement_flips_bit_packed(ndarray::array![[0b11001001, 0]].view())
            .unwrap();
        assert_eq!(
            sim.get_measurement_flips(false),
            BitTable::BoolMatrix(ndarray::array![
                [false, true, false, false, true, false, false, true, true],
                [true, false, false, true, false, false, true, true, false],
            ])
        );
    }

    #[test]
    fn flip_simulator_do_broadcast_and_to_ndarray_match_documented_examples() {
        let mut sim = FlipSimulator::new(2, true, 3, 0);
        sim.broadcast_pauli_errors(
            'X',
            bool_matrix(vec![
                vec![true, false],
                vec![false, false],
                vec![true, true],
            ])
            .view(),
            1.0,
        )
        .unwrap();
        assert_eq!(
            sim.peek_pauli_flips().unwrap(),
            vec![
                PauliString::from_text("+X_X").unwrap(),
                PauliString::from_text("+__X").unwrap()
            ]
        );

        sim.broadcast_pauli_errors(
            'Z',
            bool_matrix(vec![
                vec![false, true],
                vec![false, false],
                vec![true, true],
            ])
            .view(),
            1.0,
        )
        .unwrap();
        assert_eq!(
            sim.peek_pauli_flips().unwrap(),
            vec![
                PauliString::from_text("+X_Y").unwrap(),
                PauliString::from_text("+Z_Y").unwrap()
            ]
        );

        match sim.to_ndarray(false) {
            FlipSimulatorArrays {
                xs: BitTable::BoolMatrix(xs),
                zs: BitTable::BoolMatrix(zs),
                ..
            } => {
                assert_eq!(xs.nrows(), 2);
                assert_eq!(zs.nrows(), 2);
            }
            other => panic!("unexpected to_ndarray shape: {other:?}"),
        }
    }

    #[test]
    fn flip_simulator_error_and_operation_variants_cover_remaining_surface() {
        let mut sim = FlipSimulator::new(2, true, 2, 0);

        let invalid_code = sim.set_pauli_flip(9u8, 0, 0).unwrap_err();
        assert!(invalid_code.to_string().contains("Need pauli"));
        let invalid_symbol = sim.set_pauli_flip('Q', 0, 0).unwrap_err();
        assert!(invalid_symbol.to_string().contains("Need pauli"));

        let append_err = sim
            .append_measurement_flips(bool_matrix(vec![vec![true]]).view())
            .unwrap_err();
        assert!(
            append_err
                .to_string()
                .contains("measurement flip rows must all have width batch_size")
        );
        let packed_err = sim
            .append_measurement_flips_bit_packed(ndarray::array![[1, 2]].view())
            .unwrap_err();
        assert!(
            packed_err
                .to_string()
                .contains("bit-packed measurement flip rows must all have width")
        );

        let instruction = CircuitInstruction::from_stim_program_text("M 0").unwrap();
        sim.r#do(&instruction).unwrap();
        let repeat_body: Circuit = "M 1".parse().unwrap();
        let repeat_block = CircuitRepeatBlock::new(2, &repeat_body, "").unwrap();
        sim.r#do(&repeat_block).unwrap();
        assert_eq!(sim.num_measurements(), 3);

        let width_err = sim
            .broadcast_pauli_errors('X', bool_matrix(vec![vec![true]]).view(), 1.0)
            .unwrap_err();
        assert!(
            width_err
                .to_string()
                .contains("mask rows must all have width batch_size")
        );
        let invalid_broadcast = sim
            .broadcast_pauli_errors('Q', bool_matrix(vec![vec![true, false]]).view(), 1.0)
            .unwrap_err();
        assert!(invalid_broadcast.to_string().contains("Need pauli"));

        sim.broadcast_pauli_errors(
            'Z',
            bool_matrix(vec![vec![true, false], vec![false, true]]).view(),
            1.0,
        )
        .unwrap();
        match sim.to_ndarray(true) {
            FlipSimulatorArrays {
                xs: BitTable::PackedMatrix(xs),
                zs: BitTable::PackedMatrix(zs),
                ..
            } => {
                assert_eq!(xs.nrows(), 2);
                assert_eq!(zs.nrows(), 2);
            }
            other => panic!("unexpected packed arrays: {other:?}"),
        }
        assert!(format!("{sim:?}").contains("batch_size"));
    }

    #[test]
    fn flip_simulator_records_and_sampling_helpers_work() {
        let mut sim = FlipSimulator::new(9, true, 0, 0);
        sim.r#do(
            &Circuit::from_str(
                "
                M 0 0 0
                DETECTOR rec[-2] rec[-3]
                DETECTOR rec[-1] rec[-2]
                OBSERVABLE_INCLUDE(0) rec[-2]
                OBSERVABLE_INCLUDE(1) rec[-1]
                ",
            )
            .unwrap(),
        )
        .unwrap();

        match sim.get_detector_flips(true) {
            BitTable::PackedMatrix(rows) => assert_eq!(rows.nrows(), 2),
            other => panic!("unexpected detector flips shape: {other:?}"),
        }
        match sim.get_observable_flips(false) {
            BitTable::BoolMatrix(rows) => assert_eq!(rows.nrows(), 2),
            other => panic!("unexpected observable flips shape: {other:?}"),
        }

        let bools = sim.generate_bernoulli_samples(17, 0.25, false).unwrap();
        assert_eq!(bools.len(), 17);
        let packed = sim.generate_bernoulli_samples(53, 0.1, true).unwrap();
        assert_eq!(packed.len(), 7);
        assert_eq!(packed[6] & 0b1110_0000, 0);

        let copy = sim.clone();
        assert_eq!(copy.batch_size(), sim.batch_size());

        sim.clear();
        assert_eq!(sim.num_measurements(), 0);
    }

    #[test]
    fn dem_sampler_and_converter_cover_replay_and_error_branches() {
        let dem: DetectorErrorModel = "error(0) D0\nerror(1) D1 D2 L0".parse().unwrap();
        let mut direct_sampler = dem.compile_sampler();
        let (packed_detectors, packed_observables, packed_errors) =
            direct_sampler.sample_bit_packed(2);
        assert_eq!(
            unpack_rows(&packed_detectors, 3),
            vec![vec![false, true, true]; 2]
        );
        assert_eq!(unpack_rows(&packed_observables, 1), vec![vec![true]; 2]);
        assert_eq!(unpack_rows(&packed_errors, 2), vec![vec![false, true]; 2]);

        let mut replay_sampler = dem.compile_sampler_with_seed(999);
        let replay = replay_sampler.sample_replay(&packed_errors, 2);
        assert_eq!(replay.0, bool_matrix(vec![vec![false, true, true]; 2]));
        assert_eq!(replay.1, bool_matrix(vec![vec![true]; 2]));
        assert_eq!(replay.2, bool_matrix(vec![vec![false, true]; 2]));

        let dets_path = unique_temp_path("dem-dets-direct");
        let obs_path = unique_temp_path("dem-obs-direct");
        let err_path = unique_temp_path("dem-err-direct");
        let replay_dets_path = unique_temp_path("dem-dets-replay");
        let replay_obs_path = unique_temp_path("dem-obs-replay");
        let replay_all_dets_path = unique_temp_path("dem-dets-replay-all");
        let replay_all_obs_path = unique_temp_path("dem-obs-replay-all");
        let replay_all_err_path = unique_temp_path("dem-err-replay-all");

        let mut writer = dem.compile_sampler();
        writer
            .sample_write(2, &dets_path, "01", &obs_path, "01")
            .unwrap();
        assert_eq!(fs::read_to_string(&dets_path).unwrap(), "011\n011\n");
        assert_eq!(fs::read_to_string(&obs_path).unwrap(), "1\n1\n");

        let mut writer = dem.compile_sampler();
        writer
            .sample_write_with_errors(2, &dets_path, "01", &obs_path, "01", &err_path, "01")
            .unwrap();
        assert_eq!(fs::read_to_string(&err_path).unwrap(), "01\n01\n");

        let mut replay_writer = dem.compile_sampler_with_seed(123);
        replay_writer
            .sample_write_replay(
                2,
                &replay_dets_path,
                "01",
                &replay_obs_path,
                "01",
                &err_path,
                "01",
            )
            .unwrap();
        assert_eq!(
            fs::read_to_string(&replay_dets_path).unwrap(),
            fs::read_to_string(&dets_path).unwrap()
        );
        assert_eq!(
            fs::read_to_string(&replay_obs_path).unwrap(),
            fs::read_to_string(&obs_path).unwrap()
        );

        let mut replay_writer = dem.compile_sampler_with_seed(321);
        replay_writer
            .sample_write_replay_with_errors(
                2,
                &replay_all_dets_path,
                "01",
                &replay_all_obs_path,
                "01",
                &replay_all_err_path,
                "01",
                &err_path,
                "01",
            )
            .unwrap();
        assert_eq!(
            fs::read_to_string(&replay_all_dets_path).unwrap(),
            fs::read_to_string(&dets_path).unwrap()
        );
        assert_eq!(
            fs::read_to_string(&replay_all_obs_path).unwrap(),
            fs::read_to_string(&obs_path).unwrap()
        );
        assert_eq!(
            fs::read_to_string(&replay_all_err_path).unwrap(),
            fs::read_to_string(&err_path).unwrap()
        );

        for path in [
            dets_path,
            obs_path,
            err_path,
            replay_dets_path,
            replay_obs_path,
            replay_all_dets_path,
            replay_all_obs_path,
            replay_all_err_path,
        ] {
            fs::remove_file(path).unwrap();
        }

        let sweep_circuit = sweep_m2d_circuit();
        let mut converter = sweep_circuit.compile_m2d_converter(false);
        assert_eq!(
            converter
                .convert(
                    bool_matrix(vec![vec![false], vec![true]]).view(),
                    Some(bool_matrix(vec![vec![false], vec![true]]).view()),
                    false,
                    true,
                )
                .unwrap(),
            ConvertedMeasurements::DetectionEvents(bool_matrix(vec![
                vec![true, true],
                vec![true, true]
            ]))
        );

        let mismatch = converter
            .convert_measurements_and_sweep_bits(
                bool_matrix(vec![vec![false], vec![true]]).view(),
                bool_matrix(vec![vec![false]]).view(),
                false,
            )
            .unwrap_err();
        assert!(
            mismatch
                .to_string()
                .contains("expected equal shot counts for measurements and sweep bits")
        );
        let mismatch = converter
            .convert_measurements_and_sweep_bits_separate_observables(
                bool_matrix(vec![vec![false], vec![true]]).view(),
                bool_matrix(vec![vec![false]]).view(),
            )
            .unwrap_err();
        assert!(
            mismatch
                .to_string()
                .contains("expected equal shot counts for measurements and sweep bits")
        );

        let measurements_path = unique_temp_path("m2d-no-sweep-measurements");
        let detections_path = unique_temp_path("m2d-no-sweep-detections");
        fs::write(&measurements_path, "0\n1\n").unwrap();
        let mut converter = m2d_circuit().compile_m2d_converter(false);
        converter
            .convert_file(
                &measurements_path,
                "01",
                None::<&std::path::Path>,
                "01",
                &detections_path,
                "01",
                false,
                None::<&std::path::Path>,
                "01",
            )
            .unwrap();
        assert_eq!(fs::read_to_string(&detections_path).unwrap(), "1\n0\n");
        fs::remove_file(measurements_path).unwrap();
        fs::remove_file(detections_path).unwrap();
    }
}
