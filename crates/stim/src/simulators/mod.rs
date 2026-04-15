#![allow(clippy::too_many_arguments)]

use std::fmt;
use std::path::Path;

use crate::common::bit_packing::{pack_rows_array, unpack_rows_array};
use crate::{Result, StimError};
use ndarray::{Array2, ArrayView2};

/// Fast repeated measurement sampling from a compiled Stim circuit.
pub struct MeasurementSampler {
    pub(crate) inner: stim_cxx::MeasurementSampler,
}

/// Fast repeated detector-event sampling from a compiled Stim circuit.
pub struct DetectorSampler {
    pub(crate) inner: stim_cxx::DetectorSampler,
}

/// Fast repeated sampling from a compiled detector error model.
pub struct DemSampler {
    pub(crate) inner: stim_cxx::DemSampler,
}

/// Converts batched measurements into detector events using a compiled circuit.
pub struct MeasurementsToDetectionEventsConverter {
    pub(crate) inner: stim_cxx::MeasurementsToDetectionEventsConverter,
}

/// A stabilizer simulator backed by tableaus.
pub struct TableauSimulator {
    pub(crate) inner: stim_cxx::TableauSimulator,
}

/// A batched simulator that tracks Pauli flips and classical flip records.
pub struct FlipSimulator {
    pub(crate) inner: stim_cxx::FlipSimulator,
}

pub type CompiledMeasurementSampler = MeasurementSampler;
pub type CompiledDetectorSampler = DetectorSampler;
pub type CompiledDemSampler = DemSampler;
pub type CompiledMeasurementsToDetectionEventsConverter = MeasurementsToDetectionEventsConverter;
type UnpackedBitMatrix = Array2<bool>;
type PackedObservablePair = (Vec<u8>, Vec<u8>);
type UnpackedObservablePair = (UnpackedBitMatrix, UnpackedBitMatrix);
type PackedDemBatch = (Vec<u8>, Vec<u8>, Vec<u8>);
type UnpackedDemBatch = (UnpackedBitMatrix, UnpackedBitMatrix, UnpackedBitMatrix);

/// The result of converting measurements into detector data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertedMeasurements {
    /// Detector events only.
    DetectionEvents(Array2<bool>),
    /// Detector events plus a separate observable-flip matrix.
    DetectionEventsAndObservables(Array2<bool>, Array2<bool>),
}

/// A 2D boolean table, either unpacked or bit-packed by row.
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
    BoolMatrix(Array2<bool>),
    PackedMatrix(Array2<u8>),
}

/// A Rust-friendly export of the full [`FlipSimulator`] state.
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
    pub xs: BitTable,
    pub zs: BitTable,
    pub measurement_flips: BitTable,
    pub detector_flips: BitTable,
    pub observable_flips: BitTable,
}

impl MeasurementSampler {
    /// Creates a seeded measurement sampler from a circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0".parse().unwrap();
    /// let mut sampler = stim::MeasurementSampler::new(&circuit, false, 0);
    /// assert_eq!(sampler.sample(2), ndarray::array![[true], [true]]);
    /// ```
    #[must_use]
    pub fn new(circuit: &crate::Circuit, skip_reference_sample: bool, seed: u64) -> Self {
        circuit.compile_sampler_with_seed(skip_reference_sample, seed)
    }

    /// Returns the number of measurement bits produced by the sampler.
    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        self.inner.num_measurements()
    }

    /// Samples packed measurement data.
    ///
    /// The returned bytes are little-endian within each byte, matching Stim's
    /// `b8` data layout.
    #[must_use]
    pub fn sample_bit_packed(&mut self, shots: u64) -> Vec<u8> {
        self.inner.sample_bit_packed(shots)
    }

    /// Samples unpacked measurement data.
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

    /// Writes sampled measurement data to a file.
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
    /// Creates a tableau simulator with a deterministic default seed.
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
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            inner: stim_cxx::TableauSimulator::new(0, seed),
        }
    }

    /// Returns an owned copy of the simulator.
    #[must_use]
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Returns the number of qubits currently tracked by the simulator.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Resizes the simulator's qubit count.
    pub fn set_num_qubits(&mut self, new_num_qubits: usize) {
        self.inner.set_num_qubits(new_num_qubits);
    }

    /// Returns the simulator's current inverse tableau.
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

    /// Replaces the simulator state with an inverse tableau.
    pub fn set_inverse_tableau(&mut self, tableau: &crate::Tableau) {
        self.inner.set_inverse_tableau(&tableau.inner);
    }

    /// Returns the current measurement record.
    #[must_use]
    pub fn current_measurement_record(&self) -> Vec<bool> {
        self.inner.current_measurement_record()
    }

    /// Applies a circuit to the simulator state.
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

    /// Applies a Pauli string operation to the simulator.
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

    /// Applies a tableau on the specified targets.
    pub fn do_tableau(&mut self, tableau: &crate::Tableau, targets: &[usize]) {
        self.inner.do_tableau(&tableau.inner, targets);
    }

    /// Applies a circuit-like operation accepted by [`TableauSimulatorOperation`].
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

    /// Returns the Bloch-axis eigenstate currently prepared on a qubit.
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

    /// Returns the X-axis Bloch sign of a qubit (`-1`, `0`, or `+1`).
    pub fn peek_x(&mut self, target: usize) -> i32 {
        self.inner.peek_x(target)
    }

    /// Returns the Y-axis Bloch sign of a qubit (`-1`, `0`, or `+1`).
    pub fn peek_y(&mut self, target: usize) -> i32 {
        self.inner.peek_y(target)
    }

    /// Returns the Z-axis Bloch sign of a qubit (`-1`, `0`, or `+1`).
    pub fn peek_z(&mut self, target: usize) -> i32 {
        self.inner.peek_z(target)
    }

    /// Measures a qubit in the Z basis and returns the result.
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

    /// Measures several qubits in the Z basis.
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

    /// Returns the canonical stabilizers of the current state.
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

    /// Returns the expectation of an observable (`-1`, `0`, or `+1`).
    pub fn peek_observable_expectation(
        &self,
        observable: &crate::PauliString,
    ) -> crate::Result<i32> {
        self.inner
            .peek_observable_expectation(&observable.inner)
            .map_err(StimError::from)
    }

    /// Measures an observable, optionally with classical flip noise.
    pub fn measure_observable(
        &mut self,
        observable: &crate::PauliString,
        flip_probability: f64,
    ) -> crate::Result<bool> {
        self.inner
            .measure_observable(&observable.inner, flip_probability)
            .map_err(StimError::from)
    }

    /// Postselects the simulator state on an observable outcome.
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
    pub fn postselect_x(&mut self, targets: &[usize], desired_value: bool) -> crate::Result<()> {
        self.inner
            .postselect_x(targets, desired_value)
            .map_err(StimError::from)
    }

    /// Postselects qubits in the Y basis.
    pub fn postselect_y(&mut self, targets: &[usize], desired_value: bool) -> crate::Result<()> {
        self.inner
            .postselect_y(targets, desired_value)
            .map_err(StimError::from)
    }

    /// Postselects qubits in the Z basis.
    pub fn postselect_z(&mut self, targets: &[usize], desired_value: bool) -> crate::Result<()> {
        self.inner
            .postselect_z(targets, desired_value)
            .map_err(StimError::from)
    }

    /// Measures a qubit and returns the optional kickback Pauli.
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

    /// Exports the stabilizer state as a state vector.
    pub fn state_vector(&self, endian: &str) -> crate::Result<Vec<crate::Complex32>> {
        self.current_inverse_tableau()
            .inverse(false)
            .to_state_vector(endian)
    }

    /// Sets the state from stabilizer generators.
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

    /// Sets the state from a stabilizer state vector.
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

    /// Applies `H` to the given targets.
    pub fn h(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("H", targets, &[])
    }

    /// Applies `H_XY` to the given targets.
    pub fn h_xy(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("H_XY", targets, &[])
    }

    /// Applies `H_XZ` to the given targets.
    pub fn h_xz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("H", targets, &[])
    }

    /// Applies `H_YZ` to the given targets.
    pub fn h_yz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("H_YZ", targets, &[])
    }

    /// Applies `C_XYZ` to the given targets.
    pub fn c_xyz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("C_XYZ", targets, &[])
    }

    /// Applies `C_ZYX` to the given targets.
    pub fn c_zyx(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("C_ZYX", targets, &[])
    }

    /// Applies `X` to the given targets.
    pub fn x(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("X", targets, &[])
    }

    /// Applies `Y` to the given targets.
    pub fn y(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("Y", targets, &[])
    }

    /// Applies `Z` to the given targets.
    pub fn z(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("Z", targets, &[])
    }

    /// Applies `S` to the given targets.
    pub fn s(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("S", targets, &[])
    }

    /// Applies `S_DAG` to the given targets.
    pub fn s_dag(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("S_DAG", targets, &[])
    }

    /// Applies `SQRT_X` to the given targets.
    pub fn sqrt_x(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("SQRT_X", targets, &[])
    }

    /// Applies `SQRT_X_DAG` to the given targets.
    pub fn sqrt_x_dag(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("SQRT_X_DAG", targets, &[])
    }

    /// Applies `SQRT_Y` to the given targets.
    pub fn sqrt_y(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("SQRT_Y", targets, &[])
    }

    /// Applies `SQRT_Y_DAG` to the given targets.
    pub fn sqrt_y_dag(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("SQRT_Y_DAG", targets, &[])
    }

    /// Applies `R` to the given targets.
    pub fn reset(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("R", targets, &[])
    }

    /// Applies `RX` to the given targets.
    pub fn reset_x(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("RX", targets, &[])
    }

    /// Applies `RY` to the given targets.
    pub fn reset_y(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("RY", targets, &[])
    }

    /// Applies `RZ`/`R` to the given targets.
    pub fn reset_z(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("R", targets, &[])
    }

    /// Applies `CX` to the given targets.
    pub fn cx(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("CX", targets, &[])
    }

    /// Applies `CX` to the given targets.
    pub fn cnot(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.cx(targets)
    }

    /// Applies `CY` to the given targets.
    pub fn cy(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("CY", targets, &[])
    }

    /// Applies `CZ` to the given targets.
    pub fn cz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("CZ", targets, &[])
    }

    /// Applies `SWAP` to the given targets.
    pub fn swap(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("SWAP", targets, &[])
    }

    /// Applies `ISWAP` to the given targets.
    pub fn iswap(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("ISWAP", targets, &[])
    }

    /// Applies `ISWAP_DAG` to the given targets.
    pub fn iswap_dag(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("ISWAP_DAG", targets, &[])
    }

    /// Applies `XCX` to the given targets.
    pub fn xcx(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("XCX", targets, &[])
    }

    /// Applies `XCY` to the given targets.
    pub fn xcy(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("XCY", targets, &[])
    }

    /// Applies `XCZ` to the given targets.
    pub fn xcz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("XCZ", targets, &[])
    }

    /// Applies `YCX` to the given targets.
    pub fn ycx(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("YCX", targets, &[])
    }

    /// Applies `YCY` to the given targets.
    pub fn ycy(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("YCY", targets, &[])
    }

    /// Applies `YCZ` to the given targets.
    pub fn ycz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("YCZ", targets, &[])
    }

    /// Applies `ZCX` to the given targets.
    pub fn zcx(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("ZCX", targets, &[])
    }

    /// Applies `ZCY` to the given targets.
    pub fn zcy(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("ZCY", targets, &[])
    }

    /// Applies `ZCZ` to the given targets.
    pub fn zcz(&mut self, targets: &[usize]) -> crate::Result<()> {
        self.apply_gate("ZCZ", targets, &[])
    }

    /// Applies `X_ERROR(p)` to the given targets.
    pub fn x_error(&mut self, targets: &[usize], p: f64) -> crate::Result<()> {
        self.apply_gate("X_ERROR", targets, &[p])
    }

    /// Applies `Y_ERROR(p)` to the given targets.
    pub fn y_error(&mut self, targets: &[usize], p: f64) -> crate::Result<()> {
        self.apply_gate("Y_ERROR", targets, &[p])
    }

    /// Applies `Z_ERROR(p)` to the given targets.
    pub fn z_error(&mut self, targets: &[usize], p: f64) -> crate::Result<()> {
        self.apply_gate("Z_ERROR", targets, &[p])
    }

    /// Applies `DEPOLARIZE1(p)` to the given targets.
    pub fn depolarize1(&mut self, targets: &[usize], p: f64) -> crate::Result<()> {
        self.apply_gate("DEPOLARIZE1", targets, &[p])
    }

    /// Applies `DEPOLARIZE2(p)` to the given targets.
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
    /// Creates a flip simulator with the given batch size, qubit count, and seed.
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

    /// Returns an owned copy of the flip simulator.
    #[must_use]
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Returns the number of parallel instances tracked by the simulator.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    /// Returns the number of qubits tracked by the simulator.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of recorded measurements.
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.inner.num_measurements()
    }

    /// Returns the number of recorded detectors.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    /// Returns the number of recorded observables.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Clears the simulator state and recorded outputs.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Sets a Pauli flip for one `(qubit, batch-instance)` location.
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

    /// Returns the current Pauli flips for every batch instance.
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

    /// Returns the current Pauli flip for a single batch instance.
    pub fn peek_pauli_flip(&self, instance_index: isize) -> Result<crate::PauliString> {
        self.inner
            .peek_pauli_flip(instance_index as i64)
            .map(|inner| crate::PauliString { inner, imag: false })
            .map_err(StimError::from)
    }

    /// Appends unpacked measurement-flip rows.
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

    /// Appends bit-packed measurement-flip rows.
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

    /// Applies a circuit to the flip simulator.
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

    /// Applies a circuit-like operation accepted by [`FlipSimulatorOperation`].
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

    /// Generates Bernoulli samples, optionally bit-packed.
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

    /// Returns recorded measurement flips.
    pub fn get_measurement_flips(&self, bit_packed: bool) -> BitTable {
        decode_bit_table(self.inner.get_measurement_flips(bit_packed))
    }

    /// Returns recorded detector flips.
    pub fn get_detector_flips(&self, bit_packed: bool) -> BitTable {
        decode_bit_table(self.inner.get_detector_flips(bit_packed))
    }

    /// Returns recorded observable flips.
    pub fn get_observable_flips(&self, bit_packed: bool) -> BitTable {
        decode_bit_table(self.inner.get_observable_flips(bit_packed))
    }

    /// Exports the simulator state into Rust-friendly matrix views.
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

/// Operations accepted by [`FlipSimulator::do`].
pub enum FlipSimulatorOperation<'a> {
    Circuit(&'a crate::Circuit),
    Instruction(&'a crate::CircuitInstruction),
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

/// Operations accepted by [`TableauSimulator::do`].
pub enum TableauSimulatorOperation<'a> {
    Circuit(&'a crate::Circuit),
    PauliString(&'a crate::PauliString),
    Instruction(&'a crate::CircuitInstruction),
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
    /// Creates a seeded detector sampler from a circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0\nDETECTOR rec[-1]".parse().unwrap();
    /// let mut sampler = stim::DetectorSampler::new(&circuit, 0);
    /// assert_eq!(sampler.sample(2), ndarray::array![[false], [false]]);
    /// ```
    #[must_use]
    pub fn new(circuit: &crate::Circuit, seed: u64) -> Self {
        circuit.compile_detector_sampler_with_seed(seed)
    }

    /// Returns the number of detector bits produced per shot.
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        self.inner.num_detectors()
    }

    /// Returns the number of observable bits produced per shot.
    #[must_use]
    pub fn num_observables(&self) -> u64 {
        self.inner.num_observables()
    }

    /// Samples packed detector-event data.
    #[must_use]
    pub fn sample_bit_packed(&mut self, shots: u64) -> Vec<u8> {
        self.inner.sample_bit_packed(shots)
    }

    /// Samples packed observable-flip data.
    #[must_use]
    pub fn sample_observables_bit_packed(&mut self, shots: u64) -> Vec<u8> {
        self.inner.sample_observables_bit_packed(shots)
    }

    /// Samples unpacked detector-event data.
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

    /// Writes detector-event samples to a file.
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

    /// Writes detector and observable samples into separate files.
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

    /// Samples packed detector and observable data separately.
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
        (
            self.sample_bit_packed(shots),
            self.sample_observables_bit_packed(shots),
        )
    }

    /// Samples unpacked detector and observable data separately.
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

    /// Samples with observables prepended before detector bits.
    #[must_use]
    pub fn sample_prepend_observables(&mut self, shots: u64) -> Array2<bool> {
        let (dets, obs) = self.sample_separate_observables(shots);
        ndarray::concatenate(ndarray::Axis(1), &[obs.view(), dets.view()])
            .expect("observable/detector arrays should share shot dimension")
    }

    /// Samples with observables appended after detector bits.
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
    /// Returns the number of detectors produced per DEM sample.
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        self.inner.num_detectors()
    }

    /// Returns the number of observables produced per DEM sample.
    #[must_use]
    pub fn num_observables(&self) -> u64 {
        self.inner.num_observables()
    }

    /// Returns the number of error bits produced per DEM sample.
    #[must_use]
    pub fn num_errors(&self) -> u64 {
        self.inner.num_errors()
    }

    /// Samples packed detector, observable, and error data.
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

    /// Samples unpacked detector, observable, and error data.
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

    #[must_use]
    /// Replays previously recorded packed errors.
    pub fn sample_bit_packed_replay(
        &mut self,
        recorded_errors: &[u8],
        shots: u64,
    ) -> PackedDemBatch {
        let batch = self.inner.sample_bit_packed_replay(recorded_errors, shots);
        (batch.detectors, batch.observables, batch.errors)
    }

    /// Replays previously recorded errors and returns unpacked results.
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

    /// Writes detector and observable data to files.
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

    /// Writes detector, observable, and error data to files.
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

    /// Replays recorded errors from a file and writes outputs.
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

    /// Replays recorded errors and also writes sampled error data.
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
    /// Creates a compiled converter from a circuit.
    #[must_use]
    pub fn new(circuit: &crate::Circuit, skip_reference_sample: bool) -> Self {
        circuit.compile_m2d_converter(skip_reference_sample)
    }

    /// Converts measurement batches into detection-event batches.
    ///
    /// When `separate_observables` is `true`, observable flips are returned as a
    /// second matrix instead of being appended onto the detector-event rows.
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

    /// Converts packed measurement data into packed detector-event data.
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

    /// Converts packed measurements plus packed sweep bits.
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

    /// Converts packed measurements plus sweep bits, returning detectors and observables separately.
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

    /// Converts unpacked measurements plus unpacked sweep bits.
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

    /// Converts unpacked measurements plus sweep bits, returning detectors and observables separately.
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

    /// Converts packed measurements, returning packed detectors and observables separately.
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

    /// Converts unpacked measurements into unpacked detector data.
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

    /// Converts unpacked measurements, returning detectors and observables separately.
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

    /// Returns the number of measurement bits consumed by the converter.
    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        self.inner.num_measurements()
    }

    /// Returns the number of detector bits produced by the converter.
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        self.inner.num_detectors()
    }

    /// Returns the number of observable bits produced by the converter.
    #[must_use]
    pub fn num_observables(&self) -> u64 {
        self.inner.num_observables()
    }

    /// Returns the number of sweep bits consumed by the converter.
    #[must_use]
    pub fn num_sweep_bits(&self) -> u64 {
        self.inner.num_sweep_bits()
    }

    /// Converts measurement files into detection-event files.
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
    use crate::{Circuit, ConvertedMeasurements, PauliString, Tableau};

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

        let mut s2 = s.copy();
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

        let copy = sim.copy();
        assert_eq!(copy.batch_size(), sim.batch_size());

        sim.clear();
        assert_eq!(sim.num_measurements(), 0);
    }
}
