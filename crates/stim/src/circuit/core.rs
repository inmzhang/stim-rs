use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::ops::{Add, AddAssign, Index, Mul, MulAssign};
use std::path::Path;
use std::str::FromStr;

use super::{
    CircuitInsertOperation, CircuitInstruction, CircuitItem, CircuitRepeatBlock,
    DetectingRegionFilter,
    support::{convert_explained_error, detecting_region_entries_to_map},
};
use crate::common::bit_packing::unpack_bits;
use crate::common::parse::{coordinate_entries_to_map, decode_measurement_solution};
use crate::common::slicing::{compute_slice_indices, normalize_index};
use crate::{
    DemTarget, DetectorErrorModel, DetectorSampler, ExplainedError, Flow, Gate, GateTarget,
    MeasurementSampler, MeasurementsToDetectionEventsConverter, NoiseModel, OpenQasmVersion,
    PauliString, Result, SatProblemFormat, StimError, Tableau,
};

/// Diagram style accepted by [`Circuit::diagram`] and related helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CircuitDiagramType {
    TimelineText,
    TimelineSvg,
    TimelineSvgHtml,
    Timeline3d,
    Timeline3dHtml,
    DetSliceText,
    DetSliceSvg,
    TimeSliceSvg,
    DetSliceWithOpsSvg,
    MatchGraphSvg,
    MatchGraphSvgHtml,
    MatchGraph3d,
    MatchGraph3dHtml,
    Interactive,
    InteractiveHtml,
}

impl CircuitDiagramType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TimelineText => "timeline-text",
            Self::TimelineSvg => "timeline-svg",
            Self::TimelineSvgHtml => "timeline-svg-html",
            Self::Timeline3d => "timeline-3d",
            Self::Timeline3dHtml => "timeline-3d-html",
            Self::DetSliceText => "detslice-text",
            Self::DetSliceSvg => "detslice-svg",
            Self::TimeSliceSvg => "timeslice-svg",
            Self::DetSliceWithOpsSvg => "detslice-with-ops-svg",
            Self::MatchGraphSvg => "matchgraph-svg",
            Self::MatchGraphSvgHtml => "matchgraph-svg-html",
            Self::MatchGraph3d => "matchgraph-3d",
            Self::MatchGraph3dHtml => "matchgraph-3d-html",
            Self::Interactive => "interactive",
            Self::InteractiveHtml => "interactive-html",
        }
    }
}

impl Display for CircuitDiagramType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for CircuitDiagramType {
    type Err = StimError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "timeline-text" => Ok(Self::TimelineText),
            "timeline-svg" | "timeline" => Ok(Self::TimelineSvg),
            "timeline-svg-html" | "timeline-html" => Ok(Self::TimelineSvgHtml),
            "timeline-3d" => Ok(Self::Timeline3d),
            "timeline-3d-html" => Ok(Self::Timeline3dHtml),
            "detslice-text" | "detector-slice-text" => Ok(Self::DetSliceText),
            "detslice-svg" | "detslice" | "detslice-html" | "detslice-svg-html"
            | "detector-slice-svg" | "detector-slice" => Ok(Self::DetSliceSvg),
            "timeslice-svg"
            | "time-slice-svg"
            | "timeslice"
            | "time-slice"
            | "timeslice-html"
            | "time-slice-html"
            | "timeslice-svg-html"
            | "time-slice-svg-html" => Ok(Self::TimeSliceSvg),
            "detslice-with-ops-svg"
            | "detslice-with-ops"
            | "detslice-with-ops-html"
            | "detslice-with-ops-svg-html"
            | "time+detector-slice-svg" => Ok(Self::DetSliceWithOpsSvg),
            "matchgraph-svg" | "match-graph-svg" => Ok(Self::MatchGraphSvg),
            "matchgraph-svg-html" | "match-graph-svg-html" => Ok(Self::MatchGraphSvgHtml),
            "matchgraph-3d" | "match-graph-3d" => Ok(Self::MatchGraph3d),
            "matchgraph-3d-html" | "match-graph-3d-html" => Ok(Self::MatchGraph3dHtml),
            "interactive" => Ok(Self::Interactive),
            "interactive-html" => Ok(Self::InteractiveHtml),
            _ => Err(StimError::new(format!("unknown circuit diagram type: {s}"))),
        }
    }
}

/// A mutable stabilizer circuit.
///
/// The `Circuit` struct is arguably the most important type in the entire Stim
/// library. It is the interface through which you describe a noisy quantum
/// computation to Stim, in order to perform fast bulk sampling or fast error
/// analysis.
///
/// A circuit is a sequence of operations (gates, noise channels, measurements,
/// resets, annotations) applied to qubits. Circuits can contain `REPEAT` blocks
/// to compactly represent repetitive structure, such as the rounds of a quantum
/// error correction code. They support standard arithmetic: two circuits can be
/// concatenated with `+`, and a circuit can be repeated with `*`.
///
/// # Typical workflow
///
/// Suppose you want to use a matching-based decoder on a new quantum error
/// correction construction. Stim can help, but the very first step is to create
/// a `Circuit` implementing the construction. Once you have the circuit you can
/// then use:
///
/// - [`Circuit::detector_error_model`] to create an object that can be used to
///   configure the decoder,
/// - [`Circuit::compile_detector_sampler`] to produce detection-event samples
///   for the decoder to solve, or
/// - [`Circuit::shortest_graphlike_error`] to check for mistakes in the
///   implementation of the code.
///
/// # Parsing and display
///
/// Circuits are parsed from and displayed as Stim program text via the
/// [`FromStr`] and [`Display`] trait implementations. The file format is
/// documented in the [Stim repository](https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md).
///
/// # Examples
///
/// ```
/// use std::str::FromStr;
///
/// // Build a circuit imperatively.
/// let mut circuit = stim::Circuit::new();
/// circuit.append(stim::Gate::X, &[0], &[]).unwrap();
/// circuit.append(stim::Gate::M, &[0], &[]).unwrap();
/// let mut sampler = circuit.compile_sampler(false);
/// assert_eq!(sampler.sample(1), ndarray::array![[true]]);
///
/// // Or parse one from Stim program text.
/// let circuit = stim::Circuit::from_str(
///     "H 0\nCNOT 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]"
/// ).unwrap();
/// let mut sampler = circuit.compile_detector_sampler();
/// assert_eq!(sampler.sample(1), ndarray::array![[false]]);
/// ```
pub struct Circuit {
    pub(crate) inner: stim_cxx::Circuit,
    item_cache: OnceCell<Vec<CircuitItem>>,
}

impl Circuit {
    pub(crate) fn from_inner(inner: stim_cxx::Circuit) -> Self {
        Self {
            inner,
            item_cache: OnceCell::new(),
        }
    }

    pub(crate) fn invalidate_item_cache(&mut self) {
        let _ = self.item_cache.take();
    }

    /// Creates a new, empty circuit containing no operations and referencing
    /// zero qubits.
    ///
    /// This is the starting point for building a circuit imperatively. You can
    /// add operations using [`Circuit::append`],
    /// [`Circuit::append_from_stim_program_text`], or the `+=` operator.
    /// Alternatively, you can parse a circuit from a string with
    /// `"H 0\nM 0".parse::<stim::Circuit>()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit = stim::Circuit::new();
    /// assert!(circuit.is_empty());
    /// assert_eq!(circuit.len(), 0);
    /// assert_eq!(circuit.num_qubits(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::from_inner(stim_cxx::Circuit::new())
    }

    /// Returns the number of qubits used when simulating the circuit.
    ///
    /// This is always one more than the largest qubit index referenced by any
    /// operation in the circuit. For example, a circuit that only applies gates
    /// to qubits 0 and 100 will report `num_qubits() == 101`, because qubit
    /// indices 0 through 100 must all be allocated during simulation even
    /// though most of them are idle.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0 1".parse().unwrap();
    /// assert_eq!(circuit.num_qubits(), 2);
    ///
    /// let circuit: stim::Circuit = "H 100".parse().unwrap();
    /// assert_eq!(circuit.num_qubits(), 101);
    /// ```
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Derives the detector error model of the circuit using default options.
    ///
    /// A detector error model (DEM) describes the error processes in the circuit
    /// from the perspective of detectors and logical observables. It lists error
    /// mechanisms with their probabilities and the detection events they produce.
    /// This is the primary object consumed by matching-based decoders such as
    /// PyMatching.
    ///
    /// This convenience method calls [`Circuit::detector_error_model_with_options`]
    /// with all options set to their defaults (no decomposition, no loop
    /// flattening, no gauge detectors, no disjoint error approximation).
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit contains features that prevent a valid
    /// detector error model from being derived, such as non-deterministic
    /// detectors (gauge detectors) or disjoint error mechanisms that cannot be
    /// represented as independent errors.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X_ERROR(0.125) 0\nM 0\nDETECTOR rec[-1]".parse().unwrap();
    /// let dem = circuit.detector_error_model().unwrap();
    /// assert_eq!(dem.to_string(), "error(0.125) D0");
    /// ```
    pub fn detector_error_model(&self) -> Result<DetectorErrorModel> {
        self.detector_error_model_with_options(false, false, false, 0.0, false, false)
    }

    /// Derives the detector error model using explicit options.
    ///
    /// This is the fully-configurable variant of [`Circuit::detector_error_model`].
    /// It provides fine-grained control over how composite errors are decomposed,
    /// whether loops are flattened, and how gauge detectors and disjoint errors
    /// are handled.
    ///
    /// # Arguments
    ///
    /// * `decompose_errors` - When `true`, composite error mechanisms (such as
    ///   depolarization) are decomposed into simpler graphlike errors separated by
    ///   `stim::DemTarget::separator()`. Decomposition fails if large errors cannot be
    ///   split into components affecting at most two detectors.
    /// * `flatten_loops` - When `true`, the output will not contain any `repeat`
    ///   blocks. When `false`, the analysis watches for periodic steady states in
    ///   loops and emits compact `repeat` blocks, which is much more efficient for
    ///   circuits with many rounds.
    /// * `allow_gauge_detectors` - When `true`, non-deterministic detectors are
    ///   treated as gauge degrees of freedom and removed via Gaussian elimination
    ///   rather than causing an error.
    /// * `approximate_disjoint_errors` - When greater than `0.0`, disjoint error
    ///   components with probability at or below this threshold are approximated
    ///   as independent errors. Set to `0.0` (the default) to reject any
    ///   circuits that produce disjoint errors.
    /// * `ignore_decomposition_failures` - When `true`, errors that fail to
    ///   decompose into graphlike parts are silently inserted undecomposed rather
    ///   than aborting the conversion. Only relevant when `decompose_errors` is
    ///   `true`.
    /// * `block_decomposition_from_introducing_remnant_edges` - When `true`,
    ///   decomposing `A B C D` into `A B ^ C D` requires both `A B` and `C D` to
    ///   already appear elsewhere in the model. Remnant edges can reduce the
    ///   effective code distance. Only relevant when `decompose_errors` is `true`.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit contains gauge detectors (and
    /// `allow_gauge_detectors` is `false`), if decomposition is requested but
    /// fails, or if disjoint errors exceed the approximation threshold.
    pub fn detector_error_model_with_options(
        &self,
        decompose_errors: bool,
        flatten_loops: bool,
        allow_gauge_detectors: bool,
        approximate_disjoint_errors: f64,
        ignore_decomposition_failures: bool,
        block_decomposition_from_introducing_remnant_edges: bool,
    ) -> Result<DetectorErrorModel> {
        self.inner
            .detector_error_model(
                decompose_errors,
                flatten_loops,
                allow_gauge_detectors,
                approximate_disjoint_errors,
                ignore_decomposition_failures,
                block_decomposition_from_introducing_remnant_edges,
            )
            .map(DetectorErrorModel::from_inner)
            .map_err(StimError::from)
    }

    /// Returns a circuit containing detector declarations for deterministic
    /// measurement degrees of freedom that are not already covered by the
    /// circuit's existing `DETECTOR` and `OBSERVABLE_INCLUDE` annotations.
    ///
    /// This method is primarily useful for debugging missing detectors. It
    /// identifies generators for the uncovered degrees of freedom by simulating
    /// the circuit with a stabilizer tableau and checking which measurement
    /// parities are determined but not yet declared as detectors or observables.
    ///
    /// The returned circuit can be appended to the original circuit to obtain a
    /// circuit with no missing detectors.
    ///
    /// **Caveat:** the returned detectors are not guaranteed to be stable across
    /// Stim versions, nor are they optimized to form a low-weight or matchable
    /// basis. It is not recommended to use this method to automatically generate
    /// the detector annotations for a circuit; it is better used as a diagnostic
    /// tool to discover what you forgot to annotate.
    ///
    /// # Arguments
    ///
    /// * `unknown_input` - When `false` (the default), qubits are assumed to
    ///   start in the |0⟩ state, so initial Z-basis measurements are determined.
    ///   When `true`, qubits start in an unknown random state, meaning initial
    ///   measurements are random unless correlated with prior resets.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "R 0\nM 0".parse().unwrap();
    /// let missing = circuit.missing_detectors(false);
    /// assert_eq!(missing.to_string(), "DETECTOR rec[-1]");
    /// ```
    #[must_use]
    pub fn missing_detectors(&self, unknown_input: bool) -> Self {
        Self::from_inner(self.inner.missing_detectors(unknown_input))
    }

    /// Converts the circuit into an equivalent stabilizer tableau.
    ///
    /// A [`Tableau`] represents the Clifford unitary implemented by the circuit
    /// as a mapping from input Pauli operators to output Pauli operators. This
    /// conversion is only valid for circuits that consist entirely of Clifford
    /// gates. Noise, measurements, and resets will cause the conversion to fail
    /// unless the corresponding `ignore_*` flags are set.
    ///
    /// # Arguments
    ///
    /// * `ignore_noise` - When `true`, noise operations (e.g. `DEPOLARIZE1`,
    ///   `X_ERROR`) are skipped as if they were not present.
    /// * `ignore_measurement` - When `true`, measurement operations are skipped.
    /// * `ignore_reset` - When `true`, reset operations are skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit contains noise operations (and
    /// `ignore_noise` is `false`), measurement operations (and
    /// `ignore_measurement` is `false`), or reset operations (and
    /// `ignore_reset` is `false`).
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0\nS 0".parse().unwrap();
    /// let tableau = circuit.to_tableau(false, false, false).unwrap();
    /// assert_eq!(tableau, circuit.to_tableau(false, false, false).unwrap());
    /// ```
    pub fn to_tableau(
        &self,
        ignore_noise: bool,
        ignore_measurement: bool,
        ignore_reset: bool,
    ) -> Result<Tableau> {
        self.inner
            .to_tableau(ignore_noise, ignore_measurement, ignore_reset)
            .map(|inner| Tableau { inner })
            .map_err(StimError::from)
    }

    /// Determines whether the circuit implements the given stabilizer flow.
    ///
    /// A circuit has a stabilizer flow `P -> Q` if it maps the instantaneous
    /// stabilizer `P` at the start of the circuit to the instantaneous stabilizer
    /// `Q` at the end of the circuit. The flow may be mediated by certain
    /// measurements. For example, a lattice surgery CNOT involves `MXX` and `MZZ`
    /// measurements, and the CNOT flows implemented by the circuit reference
    /// these measurements.
    ///
    /// Interpretation of flow notation:
    /// - `P -> Q` means the circuit transforms `P` into `Q`.
    /// - `1 -> P` means the circuit prepares `P`.
    /// - `P -> 1` means the circuit measures `P`.
    /// - `1 -> 1` means the circuit contains a deterministic check (like a
    ///   `DETECTOR`).
    ///
    /// This method ignores any noise in the circuit.
    ///
    /// When `unsigned` is `false`, the signs of the Pauli strings must match
    /// exactly. When `true`, only the Pauli terms are compared and signs are
    /// permitted to be inverted (i.e., the circuit need only be correct up to
    /// Pauli gates).
    ///
    /// # Errors
    ///
    /// Returns an error if the flow string cannot be parsed or if there is
    /// an internal simulation failure.
    ///
    /// # Caveats
    ///
    /// The `unsigned=false` check is implemented using 256 randomized tests.
    /// Each test has a 50% chance of a false positive and 0% chance of a false
    /// negative, giving a total false positive probability of 2^-256.
    pub fn has_flow(&self, flow: &Flow, unsigned: bool) -> Result<bool> {
        self.inner
            .has_flow(&flow.to_string(), unsigned)
            .map_err(StimError::from)
    }

    /// Determines whether the circuit implements *all* of the given stabilizer
    /// flows.
    ///
    /// This is semantically equivalent to
    /// `flows.iter().all(|f| circuit.has_flow(f, unsigned))` but significantly
    /// faster, because the circuit only needs to be iterated once internally
    /// rather than once per flow.
    ///
    /// This method ignores any noise in the circuit.
    ///
    /// See [`Circuit::has_flow`] for a detailed description of stabilizer flows,
    /// the `unsigned` flag, and caveats around false positive rates.
    pub fn has_all_flows(&self, flows: &[Flow], unsigned: bool) -> Result<bool> {
        self.inner
            .has_all_flows(flows.iter().map(ToString::to_string).collect(), unsigned)
            .map_err(StimError::from)
    }

    /// Computes a set of stabilizer flow generators for the circuit.
    ///
    /// Every stabilizer flow that the circuit implements is a product of some
    /// subset of the returned generators. Conversely, every returned flow is
    /// guaranteed to be a flow of the circuit. This is useful for understanding
    /// the complete set of stabilizer transformations a circuit performs.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal flow computation or parsing fails.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0".parse().unwrap();
    /// let generators = circuit.flow_generators().unwrap();
    /// // H maps X -> Z and Z -> X.
    /// assert_eq!(generators.len(), 2);
    /// ```
    pub fn flow_generators(&self) -> Result<Vec<Flow>> {
        Ok(self
            .inner
            .flow_generators()
            .into_iter()
            .map(Flow::from_canonical_text)
            .collect())
    }

    /// Finds measurement sets that explain the starts and ends of the given
    /// flows, ignoring sign.
    ///
    /// For each flow in the input, this method finds a set of measurement
    /// indices such that the circuit has the flow
    /// `input -> output xor rec[solution_indices]` (unsigned). If no such
    /// solution exists for a given flow, the corresponding entry in the result
    /// is `None`.
    ///
    /// **Caution:** the solutions returned by this method are not guaranteed to
    /// be minimal. The method applies simple heuristics that attempt to reduce
    /// the size, but these heuristics are imperfect. The recommended use case
    /// is on small circuit fragments (e.g. a single error correction round)
    /// where there is ideally exactly one solution per flow.
    ///
    /// # Errors
    ///
    /// Returns an error if a flow has an empty input and empty output (the
    /// vacuous case), as a safety check for calling code. Also returns an
    /// error on internal simulation failures.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "M 2".parse().unwrap();
    /// let solutions = circuit.solve_flow_measurements(
    ///     &["Z2 -> 1".parse().unwrap()],
    /// ).unwrap();
    /// assert_eq!(solutions, vec![Some(vec![0])]);
    /// ```
    pub fn solve_flow_measurements(&self, flows: &[Flow]) -> Result<Vec<Option<Vec<i32>>>> {
        self.inner
            .solve_flow_measurements(flows.iter().map(ToString::to_string).collect())
            .map_err(StimError::from)?
            .into_iter()
            .map(decode_measurement_solution)
            .collect()
    }

    /// Time-reverses the circuit while preserving error correction structure.
    ///
    /// Returns a new circuit that has the same internal detecting regions as the
    /// original, as well as the same internal-to-external flows given in the
    /// `flows` argument, except they are all time-reversed. For example, passing
    /// a fault-tolerant preparation circuit (`1 -> Z`) produces a fault-tolerant
    /// *measurement* circuit (`Z -> 1`). Passing a fault-tolerant C_XYZ circuit
    /// (`X->Y`, `Y->Z`, `Z->X`) produces a C_ZYX circuit (`X->Z`, `Y->X`,
    /// `Z->Y`).
    ///
    /// The method turns time-reversed resets into measurements, and attempts to
    /// turn time-reversed measurements into resets. A measurement time-reverses
    /// into a reset when annotated detectors/observables or given flows have
    /// detecting regions with sensitivity just before the measurement but none
    /// with sensitivity after it.
    ///
    /// **Note:** the sign of detecting regions and stabilizer flows is *not*
    /// guaranteed to be preserved.
    ///
    /// # Arguments
    ///
    /// * `flows` - The external flows you care about. An error is raised if the
    ///   circuit does not have all of these flows (unsigned).
    /// * `dont_turn_measurements_into_resets` - When `true`, measurements
    ///   time-reverse into measurements even when nothing is sensitive to the
    ///   measured qubit afterward. This preserves all flows (up to sign/feedback).
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit does not implement all of the given flows,
    /// or if parsing/simulation fails internally.
    pub fn time_reversed_for_flows(
        &self,
        flows: &[Flow],
        dont_turn_measurements_into_resets: bool,
    ) -> Result<(Self, Vec<Flow>)> {
        let (circuit_text, flow_texts) = self
            .inner
            .time_reversed_for_flows(
                flows.iter().map(ToString::to_string).collect(),
                dont_turn_measurements_into_resets,
            )
            .map_err(StimError::from)?;
        let circuit = Self::from_str(&circuit_text)?;
        let flows = flow_texts
            .into_iter()
            .map(Flow::from_canonical_text)
            .collect::<Vec<_>>();
        Ok((circuit, flows))
    }

    /// Returns the number of top-level operations in the circuit.
    ///
    /// Top-level operations include individual gate instructions and `REPEAT`
    /// blocks. Instructions nested inside a `REPEAT` block count as a single
    /// top-level item (the block itself).
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the circuit has no top-level operations.
    ///
    /// An empty circuit is equivalent to the identity operation on zero qubits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the total number of measurement results produced when the
    /// circuit is executed, accounting for measurements inside `REPEAT` blocks
    /// (each iteration contributes its measurements).
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "M 0\nREPEAT 100 {\n    M 0 1\n}".parse().unwrap();
    /// assert_eq!(circuit.num_measurements(), 201);
    /// ```
    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        self.inner.num_measurements()
    }

    /// Counts the number of predictable (determined) measurements in the circuit.
    ///
    /// This method ignores any noise in the circuit and works by performing a
    /// stabilizer tableau simulation. Before each measurement, it checks whether
    /// the measurement's expectation is non-zero. A measurement is "determined"
    /// if its result can be predicted using other measurements that have already
    /// been performed, assuming noiseless execution.
    ///
    /// When multiple measurements occur at the same time, reordering their
    /// resolution can change *which* specific measurements are determined but
    /// will not change *how many* are determined in total.
    ///
    /// This quantity is useful because it relates to how many detectors and
    /// observables a circuit should declare. If
    /// `num_detectors + num_observables < count_determined_measurements()`,
    /// this is a warning sign that some detector declarations are missing.
    ///
    /// # Arguments
    ///
    /// * `unknown_input` - When `false`, qubits start in the |0⟩ state, so
    ///   initial Z-basis measurements are determined. When `true`, qubits start
    ///   in unknown random states.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "R 0\nM 0".parse().unwrap();
    /// assert_eq!(circuit.count_determined_measurements(false), 1);
    ///
    /// let circuit: stim::Circuit = "R 0\nH 0\nM 0".parse().unwrap();
    /// assert_eq!(circuit.count_determined_measurements(false), 0);
    /// ```
    #[must_use]
    pub fn count_determined_measurements(&self, unknown_input: bool) -> u64 {
        self.inner.count_determined_measurements(unknown_input)
    }

    /// Returns the total number of detector bits produced when sampling the
    /// circuit's detectors, accounting for detectors inside `REPEAT` blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit =
    ///     "M 0\nDETECTOR rec[-1]\nREPEAT 100 {\n    M 0 1 2\n    DETECTOR rec[-1]\n    DETECTOR rec[-2]\n}".parse().unwrap();
    /// assert_eq!(circuit.num_detectors(), 201);
    /// ```
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        self.inner.num_detectors()
    }

    /// Returns the number of logical observables defined by the circuit.
    ///
    /// This is one more than the largest index that appears as an argument to
    /// an `OBSERVABLE_INCLUDE` instruction. If no `OBSERVABLE_INCLUDE`
    /// instructions are present, returns 0.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit =
    ///     "M 0\nOBSERVABLE_INCLUDE(2) rec[-1]\nOBSERVABLE_INCLUDE(5) rec[-1]".parse().unwrap();
    /// assert_eq!(circuit.num_observables(), 6);
    /// ```
    #[must_use]
    pub fn num_observables(&self) -> u64 {
        self.inner.num_observables()
    }

    /// Returns the number of `TICK` instructions executed when running the
    /// circuit. `TICK`s inside `REPEAT` blocks are counted once per iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit =
    ///     "H 0\nTICK\nREPEAT 100 {\n    CX 0 1\n    TICK\n}".parse().unwrap();
    /// assert_eq!(circuit.num_ticks(), 101);
    /// ```
    #[must_use]
    pub fn num_ticks(&self) -> u64 {
        self.inner.num_ticks()
    }

    /// Returns the number of sweep bits needed to completely configure the
    /// circuit. This is one more than the largest sweep bit index referenced
    /// by a `sweep` target in the circuit (e.g. `CX sweep[5] 0` makes this 6).
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "CZ sweep[5] 0\nCX sweep[2] 0".parse().unwrap();
    /// assert_eq!(circuit.num_sweep_bits(), 6);
    /// ```
    #[must_use]
    pub fn num_sweep_bits(&self) -> usize {
        self.inner.num_sweep_bits()
    }

    /// Removes all operations from the circuit, resetting it to an empty state
    /// equivalent to [`Circuit::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit: stim::Circuit = "X 0\nM 0".parse().unwrap();
    /// circuit.clear();
    /// assert_eq!(circuit, stim::Circuit::new());
    /// ```
    pub fn clear(&mut self) {
        self.inner.clear();
        self.invalidate_item_cache();
    }

    /// Appends operations described by Stim program text onto the end of the
    /// circuit.
    ///
    /// The text is parsed as a complete Stim program fragment, meaning it may
    /// contain multiple instructions, comments, and `REPEAT` blocks. This is a
    /// convenient way to add several operations at once without calling
    /// [`Circuit::append`] for each one.
    ///
    /// # Errors
    ///
    /// Returns an error if the text contains syntax errors or references
    /// unknown gate names.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit = stim::Circuit::new();
    /// circuit.append_from_stim_program_text("H 0\nM 0").unwrap();
    /// assert_eq!(circuit.to_string(), "H 0\nM 0");
    /// ```
    pub fn append_from_stim_program_text(&mut self, text: &str) -> Result<()> {
        self.inner
            .append_from_stim_program_text(text)
            .map_err(StimError::from)?;
        self.invalidate_item_cache();
        Ok(())
    }

    /// Appends an operation into the circuit, specified by gate name, qubit
    /// targets (as raw indices), and gate arguments (the parenthesized
    /// parameters like probabilities or observable indices).
    ///
    /// Each target is a plain qubit index (`u32`). For richer targets such as
    /// measurement record targets, Pauli targets, or inverted targets, use
    /// [`Circuit::append_gate_targets`] instead.
    ///
    /// # Errors
    ///
    /// Returns an error if the gate name is unrecognized, or if the number of
    /// targets or arguments is invalid for the gate.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit = stim::Circuit::new();
    /// circuit.append(stim::Gate::X, &[0, 2], &[]).unwrap();
    /// assert_eq!(circuit.to_string(), "X 0 2");
    ///
    /// circuit.append(stim::Gate::X_ERROR, &[0], &[0.125]).unwrap();
    /// assert_eq!(circuit.to_string(), "X 0 2\nX_ERROR(0.125) 0");
    /// ```
    pub fn append(&mut self, gate: Gate, targets: &[u32], args: &[f64]) -> Result<()> {
        self.inner
            .append_with_tag(gate.name(), targets, args, "")
            .map_err(StimError::from)?;
        self.invalidate_item_cache();
        Ok(())
    }

    /// Appends an operation into the circuit using rich [`GateTarget`] values.
    ///
    /// Unlike [`Circuit::append`] which only accepts plain qubit indices, this
    /// method accepts the full range of target types supported by Stim,
    /// including Pauli targets (`target_x`, `target_y`, `target_z`),
    /// measurement record targets, sweep bit targets, and inverted targets.
    /// This is necessary for gates like `CORRELATED_ERROR`, `DETECTOR`, and
    /// `MPP` which require these specialized target types.
    ///
    /// # Errors
    ///
    /// Returns an error if the gate name is unrecognized, or if the targets
    /// or arguments are invalid for the gate.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit = stim::Circuit::new();
    /// circuit
    ///     .append_gate_targets(
    ///         stim::Gate::E,
    ///         &[
    ///             stim::GateTarget::x(0_u32, false).unwrap(),
    ///             stim::GateTarget::y(1_u32, false).unwrap(),
    ///             stim::GateTarget::z(2_u32, false).unwrap(),
    ///         ],
    ///         &[0.125],
    ///     )
    ///     .unwrap();
    /// assert_eq!(circuit.to_string(), "E(0.125) X0 Y1 Z2");
    /// ```
    pub fn append_gate_targets(
        &mut self,
        gate: Gate,
        targets: &[GateTarget],
        args: &[f64],
    ) -> Result<()> {
        let raw_targets: Vec<u32> = targets.iter().map(|target| target.raw_data()).collect();
        self.inner
            .append_with_tag(gate.name(), &raw_targets, args, "")
            .map_err(StimError::from)?;
        self.invalidate_item_cache();
        Ok(())
    }

    /// Checks whether two circuits are approximately equal, allowing slight
    /// perturbations of instruction arguments such as probabilities.
    ///
    /// Two circuits are approximately equal if they are equal up to an absolute
    /// tolerance `atol` on each numeric argument. For example,
    /// `X_ERROR(0.100) 0` is approximately equal to `X_ERROR(0.099) 0` within
    /// an absolute tolerance of `0.002`. All other details of the circuits
    /// (the ordering of instructions, the gate names, and the targets) must be
    /// exactly the same.
    ///
    /// This is useful for comparing circuits that were generated by slightly
    /// different noise models or floating-point rounding paths.
    ///
    /// # Examples
    ///
    /// ```
    /// let a: stim::Circuit = "X_ERROR(0.099) 0\nM 0".parse().unwrap();
    /// let b: stim::Circuit = "X_ERROR(0.101) 0\nM 0".parse().unwrap();
    /// assert!(a.approx_equals(&b, 0.01));
    /// assert!(!a.approx_equals(&b, 0.0001));
    /// ```
    #[must_use]
    pub fn approx_equals(&self, other: &Self, atol: f64) -> bool {
        self.inner.approx_equals(&other.inner, atol)
    }

    /// Returns a copy of the circuit with all noise processes removed.
    ///
    /// Pure noise instructions such as `X_ERROR`, `DEPOLARIZE1`, `DEPOLARIZE2`,
    /// and `CORRELATED_ERROR` are dropped entirely. Noisy measurement
    /// instructions like `M(0.001)` have their noise parameter removed,
    /// becoming noiseless `M` instructions.
    ///
    /// This is useful for obtaining a noiseless reference version of a noisy
    /// circuit, e.g. to compute a reference sample or to verify that the ideal
    /// circuit implements the correct logical operation.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X_ERROR(0.25) 0\nCX 0 1\nM(0.125) 0".parse().unwrap();
    /// assert_eq!(circuit.without_noise().to_string(), "CX 0 1\nM 0");
    /// ```
    #[must_use]
    pub fn without_noise(&self) -> Self {
        Self::from_inner(self.inner.without_noise())
    }

    /// Returns a noisy copy of the circuit using a Rust-side [`NoiseModel`].
    ///
    /// This is a convenience wrapper around [`NoiseModel::noisy_circuit`].
    ///
    /// # Errors
    ///
    /// Returns an error if the supplied noise model rejects this circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0\nTICK\nM 0".parse().unwrap();
    /// let noisy = circuit
    ///     .with_noise(stim::UniformDepolarizing::new(0.001).unwrap())
    ///     .unwrap();
    /// assert_eq!(
    ///     noisy.to_string(),
    ///     "H 0\nDEPOLARIZE1(0.001) 0\nTICK\nM(0.001) 0\nDEPOLARIZE1(0.001) 0"
    /// );
    /// ```
    pub fn with_noise(&self, noise_model: impl NoiseModel) -> Result<Self> {
        noise_model.noisy_circuit(self)
    }

    /// Returns a copy of the circuit with feedback operations removed and
    /// detector/observable annotations rewritten to preserve their meaning.
    ///
    /// When a feedback operation (e.g. `CX rec[-1] 0`) affects the expected
    /// parity of a detector or observable, the measurement controlling that
    /// feedback is implicitly part of the measurement set defining the detector
    /// or observable. This method removes all feedback but avoids changing the
    /// meaning of detectors/observables by turning those implicit measurement
    /// dependencies into explicit ones.
    ///
    /// This guarantees that the detector error model derived from the original
    /// circuit and the transformed circuit will be equivalent (modulo
    /// floating-point rounding). Specifically:
    ///
    /// ```text
    /// dem1 = circuit.flattened().detector_error_model()
    /// dem2 = circuit.with_inlined_feedback().flattened().detector_error_model()
    /// assert dem1 ≈ dem2
    /// ```
    #[must_use]
    pub fn with_inlined_feedback(&self) -> Self {
        Self::from_inner(self.inner.with_inlined_feedback())
    }

    /// Returns a copy of the circuit with all instruction tags removed.
    ///
    /// Tags are optional strings attached to instructions (e.g.
    /// `X[my-tag] 0`). This method strips them, leaving the instructions
    /// otherwise unchanged (including any noise parameters).
    #[must_use]
    pub fn without_tags(&self) -> Self {
        Self::from_inner(self.inner.without_tags())
    }

    /// Returns an equivalent circuit with all `REPEAT` blocks unrolled and
    /// all `SHIFT_COORDS` instructions inlined.
    ///
    /// The result contains the same instructions in the same order, but with
    /// loops flattened into repeated instructions and all coordinate shifts
    /// applied directly to `DETECTOR` and `QUBIT_COORDS` annotations.
    ///
    /// **Warning:** for circuits with many rounds, flattening can produce an
    /// extremely large circuit. Use this only when you need a flat instruction
    /// sequence (e.g. for analysis tools that don't support loops).
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "REPEAT 2 {\n    H 0\n    TICK\n}".parse().unwrap();
    /// assert_eq!(circuit.flattened().to_string(), "H 0\nTICK\nH 0\nTICK");
    /// ```
    #[must_use]
    pub fn flattened(&self) -> Self {
        Self::from_inner(self.inner.flattened())
    }

    /// Returns an equivalent circuit with gates decomposed into (mostly) the
    /// {H, S, CX, M, R} gate set.
    ///
    /// The intent of this method is to simplify the circuit to use fewer gate
    /// types so it is easier for other tools to consume. Currently the method
    /// performs the following simplifications:
    ///
    /// - Single-qubit Cliffords are decomposed into {H, S}.
    /// - Multi-qubit Cliffords are decomposed into {H, S, CX}.
    /// - Single-qubit dissipative gates are decomposed into {H, S, M, R}.
    /// - Multi-qubit dissipative gates are decomposed into {H, S, CX, M, R}.
    ///
    /// The following types of gate are currently **not** simplified (but may be
    /// in the future):
    ///
    /// - Noise instructions (like `X_ERROR`, `DEPOLARIZE2`, `E`).
    /// - Annotations (like `TICK`, `DETECTOR`, `SHIFT_COORDS`).
    /// - The `MPAD` instruction.
    /// - `REPEAT` blocks are not flattened.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "SWAP 0 1".parse().unwrap();
    /// let decomposed = circuit.decomposed();
    /// assert_eq!(decomposed.to_string(), "CX 0 1 1 0 0 1");
    /// ```
    #[must_use]
    pub fn decomposed(&self) -> Self {
        Self::from_inner(self.inner.decomposed())
    }

    /// Returns the inverse of the circuit, when one exists.
    ///
    /// The inverse circuit undoes the unitary implemented by the original. This
    /// is only possible when the circuit consists entirely of reversible gates.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit contains irreversible operations such as
    /// measurements, resets, or noise channels.
    pub fn inverse(&self) -> Result<Self> {
        self.inner
            .inverse()
            .map(Self::from_inner)
            .map_err(StimError::from)
    }

    /// Converts the circuit into OpenQASM text.
    ///
    /// The `open_qasm_version` should be `2` or `3`. Version 3 supports
    /// operations on classical bits, feedback, subroutines, and detector/
    /// observable annotations. Version 2 requires inlining non-standard
    /// dissipative gates and drops detectors/observables.
    ///
    /// When `skip_dets_and_obs` is `true`, the output omits detector and
    /// observable registers, avoiding the need for an internal circuit
    /// simulation. When `false`, those registers are computed and included.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit uses features that cannot be represented
    /// in the requested OpenQASM version.
    pub fn to_qasm(
        &self,
        open_qasm_version: OpenQasmVersion,
        skip_dets_and_obs: bool,
    ) -> Result<String> {
        self.inner
            .to_qasm(open_qasm_version.as_i32(), skip_dets_and_obs)
            .map_err(StimError::from)
    }

    /// Converts the circuit into a Quirk URL.
    ///
    /// [Quirk](https://algassert.com/quirk) is an open-source drag-and-drop
    /// circuit editor supporting up to 16 qubits. It does not support noise,
    /// feedback, or detectors, so those operations are silently dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails internally.
    pub fn to_quirk_url(&self) -> Result<String> {
        self.inner.to_quirk_url().map_err(StimError::from)
    }

    /// Converts the circuit into a [Crumble](https://algassert.com/crumble) URL.
    ///
    /// Crumble is a tool for editing stabilizer circuits and visualizing their
    /// stabilizer flows. Opening the returned URL in a web browser will load
    /// the circuit into Crumble's interactive editor.
    ///
    /// When `skip_detectors` is `true`, detector annotations are omitted from
    /// the URL, reducing visual clutter and improving Crumble's performance.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails internally.
    pub fn to_crumble_url(&self, skip_detectors: bool) -> Result<String> {
        self.inner
            .to_crumble_url(skip_detectors)
            .map_err(StimError::from)
    }

    /// Renders a circuit diagram of the requested type.
    ///
    /// Many diagram types are supported:
    ///
    /// - `"timeline-text"` (default): An ASCII diagram of operations over time,
    ///   including measurement record indices and detector annotations.
    /// - `"timeline-svg"`: An SVG image of operations over time.
    /// - `"timeline-3d"`: A 3D GLTF model of operations over time.
    /// - `"detslice-text"`: An ASCII diagram of detector stabilizers at a given
    ///   tick.
    /// - `"detslice-svg"`: An SVG image of detector stabilizers using the Pauli
    ///   color convention XYZ = RGB.
    /// - `"matchgraph-svg"`: An SVG image of the matching graph extracted from
    ///   the circuit's detector error model.
    /// - `"matchgraph-3d"`: A 3D GLTF model of the matching graph.
    /// - `"timeslice-svg"`: An SVG image of operations between two ticks, laid
    ///   out in 2D.
    /// - `"detslice-with-ops-svg"`: A combination of `timeslice-svg` and
    ///   `detslice-svg` overlaid.
    /// - `"interactive"` / `"interactive-html"`: An HTML page containing
    ///   Crumble initialized with the circuit.
    ///
    /// Variants ending in `"-html"` wrap the SVG/3D content in a resizable
    /// HTML iframe.
    ///
    /// # Errors
    ///
    /// Returns an error if the diagram type is unrecognized or the circuit
    /// contains features incompatible with the requested diagram.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0\nCNOT 0 1".parse().unwrap();
    /// let timeline = circuit.diagram(stim::CircuitDiagramType::TimelineText).unwrap();
    /// assert!(timeline.contains("q0:"));
    /// assert!(timeline.contains("q1:"));
    /// ```
    pub fn diagram(&self, type_name: CircuitDiagramType) -> Result<String> {
        self.inner
            .diagram(type_name.as_str())
            .map_err(StimError::from)
    }

    /// Renders a diagram focused on a single tick.
    pub fn diagram_with_tick(&self, type_name: CircuitDiagramType, tick: u64) -> Result<String> {
        self.inner
            .diagram_with_options(type_name.as_str(), Some((tick, 1)), None)
            .map_err(StimError::from)
    }

    /// Renders a diagram over a tick range.
    pub fn diagram_with_tick_range(
        &self,
        type_name: CircuitDiagramType,
        tick_start: u64,
        tick_count: u64,
        rows: Option<usize>,
    ) -> Result<String> {
        self.inner
            .diagram_with_options(type_name.as_str(), Some((tick_start, tick_count)), rows)
            .map_err(StimError::from)
    }

    /// Renders a diagram filtered to selected detecting regions.
    pub fn diagram_with_filters(
        &self,
        type_name: CircuitDiagramType,
        tick_range: Option<(u64, u64)>,
        rows: Option<usize>,
        filters: &[DetectingRegionFilter],
    ) -> Result<String> {
        self.inner
            .diagram_with_options_and_filters(
                type_name.as_str(),
                tick_range,
                rows,
                self.encode_diagram_filters(filters)?,
            )
            .map_err(StimError::from)
    }

    fn encode_diagram_filters(&self, filters: &[DetectingRegionFilter]) -> Result<Vec<String>> {
        let mut result = Vec::new();
        for filter in filters {
            match filter {
                DetectingRegionFilter::AllDetectors => {
                    for index in 0..self.num_detectors() {
                        result.push(format!("D{index}"));
                    }
                }
                DetectingRegionFilter::AllObservables => {
                    for index in 0..self.num_observables() {
                        result.push(format!("L{index}"));
                    }
                }
                DetectingRegionFilter::Target(target) => result.push(target.to_string()),
                DetectingRegionFilter::DetectorCoordinatePrefix(prefix) => {
                    result.push(
                        prefix
                            .iter()
                            .map(|value| value.to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                    );
                }
            }
        }
        Ok(result)
    }

    /// Returns a SAT problem encoding the shortest logical error search.
    pub fn shortest_error_sat_problem(&self) -> Result<String> {
        self.shortest_error_sat_problem_with_format(SatProblemFormat::Wdimacs)
    }

    /// Returns a SAT problem encoding in the requested format.
    pub fn shortest_error_sat_problem_with_format(
        &self,
        format_name: SatProblemFormat,
    ) -> Result<String> {
        self.inner
            .shortest_error_sat_problem(format_name.as_str())
            .map_err(StimError::from)
    }

    /// Returns a SAT problem encoding the likeliest logical error search.
    pub fn likeliest_error_sat_problem(&self) -> Result<String> {
        self.likeliest_error_sat_problem_with_options(100, SatProblemFormat::Wdimacs)
    }

    /// Returns a likeliest-error SAT encoding with explicit options.
    pub fn likeliest_error_sat_problem_with_options(
        &self,
        quantization: i32,
        format_name: SatProblemFormat,
    ) -> Result<String> {
        self.inner
            .likeliest_error_sat_problem(quantization, format_name.as_str())
            .map_err(StimError::from)
    }

    /// Returns detecting regions for all detectors and observables.
    pub fn detecting_regions(&self) -> Result<BTreeMap<DemTarget, BTreeMap<u64, PauliString>>> {
        self.inner
            .detecting_regions()
            .map_err(StimError::from)
            .and_then(detecting_region_entries_to_map)
    }

    /// Returns detecting regions with explicit target and tick filters.
    pub fn detecting_regions_with_options(
        &self,
        targets: Option<&[DemTarget]>,
        ticks: Option<&[u64]>,
        ignore_anticommutation_errors: bool,
    ) -> Result<BTreeMap<DemTarget, BTreeMap<u64, PauliString>>> {
        self.inner
            .detecting_regions_with_options(
                targets
                    .map(|targets| targets.iter().map(ToString::to_string).collect())
                    .unwrap_or_default(),
                ticks
                    .map(std::borrow::ToOwned::to_owned)
                    .unwrap_or_default(),
                ignore_anticommutation_errors,
            )
            .map_err(StimError::from)
            .and_then(detecting_region_entries_to_map)
    }

    /// Returns detecting regions using higher-level filter helpers.
    pub fn detecting_regions_with_filters(
        &self,
        filters: &[DetectingRegionFilter],
        ticks: Option<&[u64]>,
        ignore_anticommutation_errors: bool,
    ) -> Result<BTreeMap<DemTarget, BTreeMap<u64, PauliString>>> {
        if filters.is_empty() {
            return self.detecting_regions_with_options(None, ticks, ignore_anticommutation_errors);
        }

        let mut detector_coordinates: Option<BTreeMap<u64, Vec<f64>>> = None;
        let mut targets = BTreeSet::new();
        for filter in filters {
            for target in
                filter.matching_targets(self.num_detectors(), self.num_observables(), || {
                    if let Some(coords) = &detector_coordinates {
                        return Ok(coords.clone());
                    }
                    let coords = self.get_detector_coordinates(None)?;
                    detector_coordinates = Some(coords.clone());
                    Ok(coords)
                })?
            {
                targets.insert(target);
            }
        }

        let targets: Vec<DemTarget> = targets.into_iter().collect();
        self.detecting_regions_with_options(Some(&targets), ticks, ignore_anticommutation_errors)
    }

    /// Returns detector coordinates, optionally filtered to selected ids.
    pub fn get_detector_coordinates(
        &self,
        only: Option<&[u64]>,
    ) -> Result<BTreeMap<u64, Vec<f64>>> {
        let included = only
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| (0..self.num_detectors()).collect());
        self.inner
            .get_detector_coordinates(&included)
            .map(coordinate_entries_to_map)
            .map_err(StimError::from)
    }

    /// Explains which circuit locations produce detector-error-model terms.
    pub fn explain_detector_error_model_errors(
        &self,
        dem_filter: Option<&DetectorErrorModel>,
        reduce_to_one_representative_error: bool,
    ) -> Result<Vec<ExplainedError>> {
        self.inner
            .explain_detector_error_model_errors(
                dem_filter
                    .map(ToString::to_string)
                    .unwrap_or_default()
                    .as_str(),
                dem_filter.is_some(),
                reduce_to_one_representative_error,
            )
            .map_err(StimError::from)?
            .into_iter()
            .map(convert_explained_error)
            .collect()
    }

    /// Returns a shortest graphlike logical error.
    pub fn shortest_graphlike_error(&self) -> Result<Vec<ExplainedError>> {
        self.shortest_graphlike_error_with_options(true, false)
    }

    /// Returns shortest graphlike logical errors with explicit options.
    pub fn shortest_graphlike_error_with_options(
        &self,
        ignore_ungraphlike_errors: bool,
        canonicalize_circuit_errors: bool,
    ) -> Result<Vec<ExplainedError>> {
        self.inner
            .shortest_graphlike_error(ignore_ungraphlike_errors, canonicalize_circuit_errors)
            .map_err(StimError::from)?
            .into_iter()
            .map(convert_explained_error)
            .collect()
    }

    /// Searches for undetectable logical errors using bounded graph exploration.
    pub fn search_for_undetectable_logical_errors(
        &self,
        dont_explore_detection_event_sets_with_size_above: u64,
        dont_explore_edges_with_degree_above: u64,
        dont_explore_edges_increasing_symptom_degree: bool,
        canonicalize_circuit_errors: bool,
    ) -> Result<Vec<ExplainedError>> {
        self.inner
            .search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above,
                dont_explore_edges_with_degree_above,
                dont_explore_edges_increasing_symptom_degree,
                canonicalize_circuit_errors,
            )
            .map_err(StimError::from)?
            .into_iter()
            .map(convert_explained_error)
            .collect()
    }

    /// Returns final qubit coordinates after all coordinate-shift instructions.
    pub fn get_final_qubit_coordinates(&self) -> Result<BTreeMap<u64, Vec<f64>>> {
        self.inner
            .get_final_qubit_coordinates()
            .map(coordinate_entries_to_map)
            .map_err(StimError::from)
    }

    /// Returns a top-level item by index.
    ///
    /// Negative indices count from the end.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0\nREPEAT 2 {\n    X 1\n}\nM 0".parse().unwrap();
    /// assert_eq!(
    ///     circuit.get(0).unwrap(),
    ///     stim::CircuitItem::Instruction("H 0".parse().unwrap())
    /// );
    /// assert_eq!(
    ///     circuit.get(-1).unwrap(),
    ///     stim::CircuitItem::Instruction("M 0".parse().unwrap())
    /// );
    /// ```
    pub fn get(&self, index: isize) -> Result<CircuitItem> {
        let normalized = normalize_index(index, self.len())
            .ok_or_else(|| StimError::new(format!("index {index} out of range")))?;
        Ok(self.cached_top_level_items()[normalized].clone())
    }

    /// Returns a sliced circuit over top-level items.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0\nX 1\nY 2\nM 0".parse().unwrap();
    /// assert_eq!(circuit.slice(Some(1), None, 2).unwrap().to_string(), "X 1\nM 0");
    /// assert_eq!(circuit.slice(None, None, -1).unwrap().to_string(), "M 0\nY 2\nX 1\nH 0");
    /// ```
    pub fn slice(&self, start: Option<isize>, stop: Option<isize>, step: isize) -> Result<Self> {
        if step == 0 {
            return Err(StimError::new("slice step cannot be zero"));
        }
        let len = self.len() as isize;
        let indices = compute_slice_indices(len, start, stop, step);
        if indices.is_empty() {
            return Ok(Self::new());
        }
        let slice_start = indices[0];
        let slice_step = if indices.len() >= 2 {
            indices[1] - indices[0]
        } else {
            step
        };
        Ok(Self::from_inner(self.inner.get_slice(
            slice_start as i64,
            slice_step as i64,
            indices.len() as i64,
        )))
    }

    /// Removes and returns a top-level item by index.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit: stim::Circuit = "H 0\nX 1\nM 0".parse().unwrap();
    /// let popped = circuit.pop(1).unwrap();
    /// assert_eq!(popped, stim::CircuitItem::Instruction("X 1".parse().unwrap()));
    /// assert_eq!(circuit.to_string(), "H 0\nM 0");
    /// ```
    pub fn pop(&mut self, index: isize) -> Result<CircuitItem> {
        let normalized = normalize_index(index, self.len())
            .ok_or_else(|| StimError::new(format!("index {index} out of range")))?;
        let popped = self.cached_top_level_items()[normalized].clone();
        self.inner.remove_top_level(normalized);
        self.invalidate_item_cache();
        Ok(popped)
    }

    /// Inserts an operation before the given top-level index.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit: stim::Circuit = "H 0\nM 0".parse().unwrap();
    /// circuit.insert(1, stim::CircuitInstruction::parse("X 1").unwrap()).unwrap();
    /// assert_eq!(circuit.to_string(), "H 0\nX 1\nM 0");
    /// ```
    pub fn insert(
        &mut self,
        index: isize,
        operation: impl Into<CircuitInsertOperation>,
    ) -> Result<()> {
        let len = self.len() as isize;
        let normalized = if index < 0 { len + index } else { index };
        if !(0..len).contains(&normalized) {
            return Err(StimError::new(format!("index {index} out of range")));
        }

        match operation.into() {
            CircuitInsertOperation::Instruction(instruction) => {
                let raw_targets = instruction
                    .targets()
                    .iter()
                    .copied()
                    .map(|target| target.raw_data())
                    .collect::<Vec<_>>();
                self.inner
                    .insert_with_tag(
                        normalized as usize,
                        instruction.name(),
                        &raw_targets,
                        instruction.gate_args(),
                        instruction.tag(),
                    )
                    .map_err(StimError::from)?;
            }
            CircuitInsertOperation::Circuit(circuit) => {
                if !circuit.is_empty() {
                    self.inner
                        .insert_circuit(normalized as usize, &circuit.inner);
                }
            }
            CircuitInsertOperation::RepeatBlock(repeat_block) => {
                self.inner
                    .insert_repeat_block(
                        normalized as usize,
                        repeat_block.repeat_count(),
                        &repeat_block.body_copy().inner,
                        repeat_block.tag(),
                    )
                    .map_err(StimError::from)?;
            }
        }
        self.invalidate_item_cache();
        Ok(())
    }

    /// Appends an owned operation of any supported type.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit = stim::Circuit::new();
    /// circuit
    ///     .append_operation(stim::CircuitInstruction::parse("H 0").unwrap())
    ///     .unwrap();
    /// assert_eq!(circuit.to_string(), "H 0");
    /// ```
    pub fn append_operation(&mut self, operation: impl Into<CircuitInsertOperation>) -> Result<()> {
        match operation.into() {
            CircuitInsertOperation::Instruction(instruction) => {
                self.append_instruction(&instruction)
            }
            CircuitInsertOperation::Circuit(circuit) => {
                if circuit.is_empty() {
                    Ok(())
                } else {
                    self.inner.add_assign(&circuit.inner);
                    self.invalidate_item_cache();
                    Ok(())
                }
            }
            CircuitInsertOperation::RepeatBlock(repeat_block) => {
                self.inner
                    .append_repeat_block(
                        repeat_block.repeat_count(),
                        &repeat_block.body_copy().inner,
                        repeat_block.tag(),
                    )
                    .map_err(StimError::from)?;
                self.invalidate_item_cache();
                Ok(())
            }
        }
    }

    fn top_level_items(&self) -> Result<Vec<CircuitItem>> {
        Ok(self.cached_top_level_items().clone())
    }

    fn top_level_item(&self, index: usize) -> CircuitItem {
        Self::build_top_level_item(&self.inner, index)
    }

    fn build_top_level_item(inner: &stim_cxx::Circuit, index: usize) -> CircuitItem {
        let item = inner.top_level_item(index);
        if item.is_repeat_block {
            CircuitItem::RepeatBlock(
                CircuitRepeatBlock::new(
                    item.repeat_count,
                    &Self::from_inner(inner.top_level_repeat_block_body(index)),
                    item.tag,
                )
                .expect("stim-cxx repeat block data should be valid"),
            )
        } else {
            CircuitItem::Instruction(
                CircuitInstruction::new(
                    item.name,
                    item.targets.into_iter().map(GateTarget::from_raw_data),
                    item.gate_args,
                    item.tag,
                )
                .expect("stim-cxx instruction data should be valid"),
            )
        }
    }

    fn cached_top_level_items(&self) -> &Vec<CircuitItem> {
        self.item_cache.get_or_init(|| {
            (0..self.len())
                .map(|index| self.top_level_item(index))
                .collect()
        })
    }

    /// Returns the circuit's reference sample in bit-packed form.
    #[must_use]
    pub fn reference_sample_bit_packed(&self) -> Vec<u8> {
        self.inner.reference_sample_bit_packed()
    }

    /// Returns the circuit's reference sample as unpacked booleans.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0 1".parse().unwrap();
    /// assert_eq!(circuit.reference_sample(), vec![true, false]);
    /// ```
    #[must_use]
    pub fn reference_sample(&self) -> Vec<bool> {
        unpack_bits(
            &self.reference_sample_bit_packed(),
            self.num_measurements() as usize,
        )
    }

    /// Returns packed detector signs for the reference sample.
    #[must_use]
    pub fn reference_detector_signs_bit_packed(&self) -> Vec<u8> {
        self.inner.reference_detector_signs_bit_packed()
    }

    /// Returns packed detector and observable signs for the reference sample.
    #[must_use]
    pub fn reference_detector_and_observable_signs_bit_packed(&self) -> (Vec<u8>, Vec<u8>) {
        (
            self.reference_detector_signs_bit_packed(),
            self.reference_observable_signs_bit_packed(),
        )
    }

    /// Returns unpacked detector signs for the reference sample.
    #[must_use]
    pub fn reference_detector_signs(&self) -> Vec<bool> {
        unpack_bits(
            &self.reference_detector_signs_bit_packed(),
            self.num_detectors() as usize,
        )
    }

    /// Returns packed observable signs for the reference sample.
    #[must_use]
    pub fn reference_observable_signs_bit_packed(&self) -> Vec<u8> {
        self.inner.reference_observable_signs_bit_packed()
    }

    /// Returns unpacked observable signs for the reference sample.
    #[must_use]
    pub fn reference_observable_signs(&self) -> Vec<bool> {
        unpack_bits(
            &self.reference_observable_signs_bit_packed(),
            self.num_observables() as usize,
        )
    }

    /// Returns unpacked detector and observable signs for the reference sample.
    #[must_use]
    pub fn reference_detector_and_observable_signs(&self) -> (Vec<bool>, Vec<bool>) {
        (
            self.reference_detector_signs(),
            self.reference_observable_signs(),
        )
    }

    /// Reads a circuit from a file containing Stim program text.
    ///
    /// # Examples
    ///
    /// ```
    /// let path = std::env::temp_dir().join("stim-rs-circuit-from-file.stim");
    /// std::fs::write(&path, "H 0\nM 0\n").unwrap();
    /// let circuit = stim::Circuit::from_file(&path).unwrap();
    /// assert_eq!(circuit.to_string(), "H 0\nM 0");
    /// std::fs::remove_file(path).unwrap();
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(|error| StimError::new(error.to_string()))?;
        Self::from_str(&text)
    }

    /// Writes the circuit to a file, always ending with a trailing newline.
    ///
    /// # Examples
    ///
    /// ```
    /// let path = std::env::temp_dir().join("stim-rs-circuit-to-file.stim");
    /// let circuit: stim::Circuit = "H 0\nM 0".parse().unwrap();
    /// circuit.to_file(&path).unwrap();
    /// assert_eq!(std::fs::read_to_string(&path).unwrap(), "H 0\nM 0\n");
    /// std::fs::remove_file(path).unwrap();
    /// ```
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, format!("{self}\n")).map_err(|error| StimError::new(error.to_string()))
    }

    /// Generates a named example circuit using Stim's built-in generators.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit = stim::Circuit::generated("repetition_code:memory", 3, 2).unwrap();
    /// assert!(circuit.num_qubits() >= 3);
    /// assert!(circuit.num_detectors() > 0);
    /// ```
    pub fn generated(code_task: &str, distance: usize, rounds: usize) -> Result<Self> {
        Self::generated_with_noise(code_task, distance, rounds, 0.0, 0.0, 0.0, 0.0)
    }

    /// Generates a named example circuit with explicit noise parameters.
    pub fn generated_with_noise(
        code_task: &str,
        distance: usize,
        rounds: usize,
        after_clifford_depolarization: f64,
        before_round_data_depolarization: f64,
        before_measure_flip_probability: f64,
        after_reset_flip_probability: f64,
    ) -> Result<Self> {
        stim_cxx::Circuit::generated(
            code_task,
            distance,
            rounds,
            after_clifford_depolarization,
            before_round_data_depolarization,
            before_measure_flip_probability,
            after_reset_flip_probability,
        )
        .map(Self::from_inner)
        .map_err(StimError::from)
    }

    /// Compiles a measurement sampler for the circuit.
    ///
    /// When `skip_reference_sample` is `false`, the sampler first computes a
    /// noiseless reference sample and xors sampled flips against it before
    /// returning measurement results. Set it to `true` when you intentionally
    /// want raw flip information instead of physical measurement values.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0".parse().unwrap();
    /// let mut sampler = circuit.compile_sampler(false);
    /// assert_eq!(sampler.sample(2), ndarray::array![[true], [true]]);
    /// ```
    #[must_use]
    pub fn compile_sampler(&self, skip_reference_sample: bool) -> MeasurementSampler {
        self.compile_sampler_with_seed(skip_reference_sample, 0)
    }

    /// Compiles a seeded measurement sampler for the circuit.
    ///
    /// This is the explicit-seed variant of [`Self::compile_sampler`].
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X_ERROR(0.5) 0\nM 0".parse().unwrap();
    /// let mut sampler_a = circuit.compile_sampler_with_seed(false, 5);
    /// let mut sampler_b = circuit.compile_sampler_with_seed(false, 5);
    /// assert_eq!(sampler_a.sample(4), sampler_b.sample(4));
    /// ```
    #[must_use]
    pub fn compile_sampler_with_seed(
        &self,
        skip_reference_sample: bool,
        seed: u64,
    ) -> MeasurementSampler {
        MeasurementSampler {
            inner: self
                .inner
                .compile_sampler_with_seed(skip_reference_sample, seed),
        }
    }

    /// Compiles a detector-event sampler for the circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0\nDETECTOR rec[-1]".parse().unwrap();
    /// let mut sampler = circuit.compile_detector_sampler();
    /// assert_eq!(sampler.sample(2), ndarray::array![[false], [false]]);
    /// ```
    #[must_use]
    pub fn compile_detector_sampler(&self) -> DetectorSampler {
        self.compile_detector_sampler_with_seed(0)
    }

    /// Compiles a seeded detector-event sampler for the circuit.
    #[must_use]
    pub fn compile_detector_sampler_with_seed(&self, seed: u64) -> DetectorSampler {
        DetectorSampler {
            inner: self.inner.compile_detector_sampler_with_seed(seed),
        }
    }

    /// Compiles a measurements-to-detection-events converter for the circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X 0\nM 0\nDETECTOR rec[-1]".parse().unwrap();
    /// let mut converter = circuit.compile_m2d_converter(false);
    /// let converted = converter
    ///     .convert(ndarray::array![[false], [true]].view(), None, false, false)
    ///     .unwrap();
    /// assert_eq!(
    ///     converted,
    ///     stim::ConvertedMeasurements::DetectionEvents(ndarray::array![[true], [false]])
    /// );
    /// ```
    pub fn compile_m2d_converter(
        &self,
        skip_reference_sample: bool,
    ) -> MeasurementsToDetectionEventsConverter {
        MeasurementsToDetectionEventsConverter {
            inner: self.inner.compile_m2d_converter(skip_reference_sample),
        }
    }
}

impl IntoIterator for Circuit {
    type Item = CircuitItem;
    type IntoIter = std::vec::IntoIter<CircuitItem>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.len();
        let Self { inner, item_cache } = self;
        item_cache
            .into_inner()
            .unwrap_or_else(|| {
                (0..len)
                    .map(|index| Self::build_top_level_item(&inner, index))
                    .collect()
            })
            .into_iter()
    }
}

impl IntoIterator for &Circuit {
    type Item = CircuitItem;
    type IntoIter = std::vec::IntoIter<CircuitItem>;

    fn into_iter(self) -> Self::IntoIter {
        self.top_level_items()
            .expect("valid Circuit values must iterate as valid top-level items")
            .into_iter()
    }
}

impl IntoIterator for &mut Circuit {
    type Item = CircuitItem;
    type IntoIter = std::vec::IntoIter<CircuitItem>;

    fn into_iter(self) -> Self::IntoIter {
        self.top_level_items()
            .expect("valid Circuit values must iterate as valid top-level items")
            .into_iter()
    }
}

impl Clone for Circuit {
    fn clone(&self) -> Self {
        Self::from_inner(self.inner.clone())
    }
}

impl Default for Circuit {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Circuit {
    fn eq(&self, other: &Self) -> bool {
        self.inner.equals(&other.inner)
    }
}

impl Eq for Circuit {}

impl Add for Circuit {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_inner(self.inner.add(&rhs.inner))
    }
}

impl AddAssign for Circuit {
    fn add_assign(&mut self, rhs: Self) {
        self.inner.add_assign(&rhs.inner);
        self.invalidate_item_cache();
    }
}

impl Mul<u64> for Circuit {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self::Output {
        Self::from_inner(self.inner.repeat(rhs))
    }
}

impl Mul<Circuit> for u64 {
    type Output = Circuit;

    fn mul(self, rhs: Circuit) -> Self::Output {
        rhs * self
    }
}

impl MulAssign<u64> for Circuit {
    fn mul_assign(&mut self, rhs: u64) {
        self.inner.repeat_assign(rhs);
        self.invalidate_item_cache();
    }
}

impl Index<usize> for Circuit {
    type Output = CircuitItem;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cached_top_level_items()[index]
    }
}

impl Display for Circuit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.to_stim_program_text())
    }
}

impl fmt::Debug for Circuit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self == &Self::new() {
            return f.write_str("stim::Circuit()");
        }

        write!(f, "stim::Circuit(\"\"\"\n{}\n\"\"\")", self)
    }
}

impl FromStr for Circuit {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self> {
        stim_cxx::Circuit::from_stim_program_text(s)
            .map(Self::from_inner)
            .map_err(StimError::from)
    }
}

#[cfg(test)]
mod api_tests {
    use std::str::FromStr;
    use std::{fs, path::PathBuf, time::SystemTime};

    use ndarray::Array2;

    use crate::{Circuit, Gate, all_gates};

    fn unique_temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        std::env::temp_dir().join(format!("stim-rs-{name}-{nanos}.stim"))
    }

    fn bool_matrix(rows: Vec<Vec<bool>>) -> Array2<bool> {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, Vec::len);
        Array2::from_shape_vec((nrows, ncols), rows.into_iter().flatten().collect())
            .expect("rows should be rectangular")
    }

    fn gate(name: &str) -> Gate {
        name.parse().expect("gate should parse")
    }

    #[test]
    fn empty_circuit_is_zeroed() {
        let circuit = Circuit::new();

        assert_eq!(circuit.to_string(), "");
        assert!(circuit.is_empty());
        assert_eq!(circuit.len(), 0);
        assert_eq!(circuit.num_qubits(), 0);
        assert_eq!(circuit.num_measurements(), 0);
        assert_eq!(circuit.num_detectors(), 0);
        assert_eq!(circuit.num_observables(), 0);
        assert_eq!(circuit.num_ticks(), 0);
        assert_eq!(circuit.num_sweep_bits(), 0);
    }

    #[test]
    fn parses_and_renders_stim_program_text() {
        let circuit = Circuit::from_str(
            "\
    H 0
    TICK
    M 0",
        )
        .expect("circuit should parse");

        assert!(!circuit.is_empty());
        assert_eq!(circuit.len(), 3);
        assert_eq!(circuit.to_string(), "H 0\nTICK\nM 0");
        assert_eq!(circuit.num_qubits(), 1);
        assert_eq!(circuit.num_measurements(), 1);
        assert_eq!(circuit.num_detectors(), 0);
        assert_eq!(circuit.num_observables(), 0);
        assert_eq!(circuit.num_ticks(), 1);
        assert_eq!(circuit.num_sweep_bits(), 0);
    }

    #[test]
    fn clear_resets_a_circuit() {
        let mut circuit = Circuit::from_str("X 0\nM 0").expect("circuit should parse");

        circuit.clear();

        assert_eq!(circuit, Circuit::new());
    }

    #[test]
    fn without_noise_removes_noise_processes() {
        let circuit =
            Circuit::from_str("X_ERROR(0.25) 0\nCX 0 1\nM(0.125) 0").expect("circuit should parse");

        let noiseless = circuit.without_noise();

        assert_eq!(noiseless.to_string(), "CX 0 1\nM 0");
    }

    #[test]
    fn with_noise_applies_rust_side_noise_models() {
        let circuit = Circuit::from_str("H 0\nTICK\nM 0").expect("circuit should parse");

        let noisy = circuit
            .with_noise(crate::UniformDepolarizing::new(0.001).unwrap())
            .unwrap();

        assert_eq!(
            noisy.to_string(),
            "H 0\nDEPOLARIZE1(0.001) 0\nTICK\nM(0.001) 0\nDEPOLARIZE1(0.001) 0"
        );
    }

    #[test]
    fn without_tags_removes_instruction_tags() {
        let circuit = Circuit::from_str("X[test-tag] 0\nM[test-tag-2](0.125) 0")
            .expect("circuit should parse");

        let untagged = circuit.without_tags();

        assert_eq!(untagged.to_string(), "X 0\nM(0.125) 0");
    }

    #[test]
    fn flattened_unrolls_repeat_blocks() {
        let circuit = Circuit::from_str(
            "\
    REPEAT 2 {
        H 0
        TICK
    }",
        )
        .expect("circuit should parse");

        let flattened = circuit.flattened();

        assert_eq!(flattened.to_string(), "H 0\nTICK\nH 0\nTICK");
    }

    #[test]
    fn inverse_reverses_operations() {
        let circuit = Circuit::from_str("H 0\nS 0").expect("circuit should parse");

        let inverse = circuit.inverse().expect("circuit should invert");

        assert_eq!(inverse.to_string(), "S_DAG 0\nH 0");
    }

    #[test]
    fn approx_equals_uses_absolute_tolerance() {
        let base = Circuit::from_str("X_ERROR(0.099) 0\nM 0").expect("circuit should parse");
        let other = Circuit::from_str("X_ERROR(0.101) 0\nM 0").expect("circuit should parse");

        assert!(!base.approx_equals(&other, 0.001));
        assert!(base.approx_equals(&other, 0.01));
    }

    #[test]
    fn append_from_stim_program_text_extends_circuit() {
        let mut circuit = Circuit::from_str("H 0").expect("circuit should parse");

        circuit
            .append_from_stim_program_text("M 0\nCX rec[-1] 1")
            .expect("circuit should append");

        assert_eq!(circuit.to_string(), "H 0\nM 0\nCX rec[-1] 1");
        assert_eq!(circuit.len(), 3);
    }

    #[test]
    fn from_file_reads_stim_program_text() {
        let path = unique_temp_path("from-file");
        fs::write(&path, "H 5\nX 0\n").expect("temp file should write");

        let circuit = Circuit::from_file(&path).expect("circuit should read");

        assert_eq!(circuit.to_string(), "H 5\nX 0");

        fs::remove_file(path).expect("temp file should delete");
    }

    #[test]
    fn to_file_writes_stim_program_text_with_trailing_newline() {
        let path = unique_temp_path("to-file");
        let circuit = Circuit::from_str("H 5\nX 0").expect("circuit should parse");

        circuit.to_file(&path).expect("circuit should write");

        assert_eq!(
            fs::read_to_string(&path).expect("temp file should read"),
            "H 5\nX 0\n"
        );

        fs::remove_file(path).expect("temp file should delete");
    }

    #[test]
    fn add_combines_two_circuits() {
        let left = Circuit::from_str("X 0\nY 1 2").expect("left circuit should parse");
        let right = Circuit::from_str("M 0 1 2").expect("right circuit should parse");

        let combined = left.clone() + right.clone();

        assert_eq!(combined.to_string(), "X 0\nY 1 2\nM 0 1 2");
        assert_eq!(left.to_string(), "X 0\nY 1 2");
        assert_eq!(right.to_string(), "M 0 1 2");
    }

    #[test]
    fn add_assign_appends_in_place() {
        let mut circuit = Circuit::from_str("X 0\nY 1 2").expect("circuit should parse");
        let suffix = Circuit::from_str("M 0 1 2").expect("suffix should parse");

        circuit += suffix;

        assert_eq!(circuit.to_string(), "X 0\nY 1 2\nM 0 1 2");
    }

    #[test]
    fn multiply_wraps_in_repeat_block() {
        let circuit = Circuit::from_str("X 0\nY 1 2").expect("circuit should parse");

        let repeated = circuit.clone() * 3;

        assert_eq!(repeated.to_string(), "REPEAT 3 {\n    X 0\n    Y 1 2\n}");
        let zero = u64::default();
        let one = 1_u64;
        assert_eq!((circuit.clone() * zero), Circuit::new());
        assert_eq!((circuit.clone() * one), circuit);
    }

    #[test]
    fn multiply_assign_wraps_in_repeat_block() {
        let mut circuit = Circuit::from_str("X 0\nY 1 2").expect("circuit should parse");

        circuit *= 3;

        assert_eq!(circuit.to_string(), "REPEAT 3 {\n    X 0\n    Y 1 2\n}");
    }

    #[test]
    fn copy_returns_an_independent_clone() {
        let original = Circuit::from_str("H 0").expect("circuit should parse");

        let copy = original.clone();

        assert_eq!(copy, original);
        assert_ne!((&copy as *const Circuit), (&original as *const Circuit));
    }

    #[test]
    fn generated_repetition_code_circuit_is_available() {
        let circuit = Circuit::generated("repetition_code:memory", 3, 2)
            .expect("generated circuit should succeed");

        assert!(!circuit.is_empty());
        assert!(circuit.to_string().contains("DETECTOR"));
        assert!(circuit.to_string().contains("OBSERVABLE_INCLUDE"));
    }

    #[test]
    fn append_adds_gate_operations() {
        let mut circuit = Circuit::new();

        circuit
            .append(gate("X"), &[0], &[])
            .expect("append should succeed");
        circuit
            .append(gate("M"), &[0, 1], &[])
            .expect("append should succeed");

        assert_eq!(circuit.to_string(), "X 0\nM 0 1");
    }

    #[test]
    fn append_supports_gate_arguments() {
        let mut circuit = Circuit::new();

        circuit
            .append(gate("X_ERROR"), &[0], &[0.125])
            .expect("append should succeed");

        assert_eq!(circuit.to_string(), "X_ERROR(0.125) 0");
    }

    #[test]
    fn append_reports_stim_parse_errors() {
        let error = Gate::new("NOT_A_GATE").expect_err("invalid gate should fail");

        assert!(error.message().contains("NOT_A_GATE"));
    }

    #[test]
    fn count_determined_measurements_matches_stim_semantics() {
        let circuit = Circuit::from_str("R 0\nM 0").expect("circuit should parse");

        assert_eq!(circuit.count_determined_measurements(false), 1);
    }

    #[test]
    fn count_determined_measurements_supports_unknown_input() {
        let circuit = Circuit::from_str("M 0").expect("circuit should parse");

        assert_eq!(circuit.count_determined_measurements(false), 1);
        assert_eq!(circuit.count_determined_measurements(true), 0);
    }

    #[test]
    fn decomposed_rewrites_to_simpler_gate_set() {
        let circuit = Circuit::from_str("SWAP 0 1").expect("circuit should parse");

        let decomposed = circuit.decomposed();

        assert_eq!(decomposed.to_string(), "CX 0 1 1 0 0 1");
    }

    #[test]
    fn to_quirk_url_exports_supported_circuits() {
        let circuit = Circuit::from_str("H 0\nCX 0 1\nS 1").expect("circuit should parse");

        let url = circuit.to_quirk_url().expect("quirk export should succeed");

        assert!(url.starts_with("https://algassert.com/quirk#circuit="));
    }

    #[test]
    fn to_qasm_exports_openqasm() {
        let circuit = Circuit::from_str("H 0\nM 0").expect("circuit should parse");

        let qasm = circuit
            .to_qasm(crate::OpenQasmVersion::V3, false)
            .expect("qasm export should succeed");

        assert!(qasm.contains("OPENQASM 3.0;"));
        assert!(qasm.contains("h q[0];"));
    }

    #[test]
    fn to_crumble_url_exports_supported_circuits() {
        let circuit = Circuit::from_str("H 0\nCX 0 1\nS 1").expect("circuit should parse");

        let url = circuit
            .to_crumble_url(false)
            .expect("crumble export should succeed");

        assert!(url.starts_with("https://algassert.com/crumble#circuit="));
    }

    #[test]
    fn reference_sample_bit_packed_matches_expected_measurements() {
        let circuit = Circuit::from_str("X 1\nM 0 1").expect("circuit should parse");

        let sample = circuit.reference_sample_bit_packed();

        assert_eq!(sample, vec![0b0000_0010]);
    }

    #[test]
    fn reference_detector_and_observable_signs_are_bit_packed() {
        let circuit = Circuit::from_str(
            "X 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(3) rec[-1] rec[-2]",
        )
        .expect("circuit should parse");

        assert_eq!(
            circuit.reference_detector_signs_bit_packed(),
            vec![0b0000_0001]
        );
        assert_eq!(
            circuit.reference_observable_signs_bit_packed(),
            vec![0b0000_1000]
        );
    }

    #[test]
    fn reference_detector_and_observable_signs_bit_packed_returns_both_vectors() {
        let circuit = Circuit::from_str(
            "X 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(3) rec[-1] rec[-2]",
        )
        .expect("circuit should parse");

        let (detectors, observables) = circuit.reference_detector_and_observable_signs_bit_packed();

        assert_eq!(detectors, vec![0b0000_0001]);
        assert_eq!(observables, vec![0b0000_1000]);
    }

    #[test]
    fn reference_sample_unpacked_matches_expected_measurements() {
        let circuit = Circuit::from_str("X 1\nM 0 1").expect("circuit should parse");

        assert_eq!(circuit.reference_sample(), vec![false, true]);
    }

    #[test]
    fn reference_detector_and_observable_signs_unpack_to_bool_vectors() {
        let circuit = Circuit::from_str(
            "X 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(3) rec[-1] rec[-2]",
        )
        .expect("circuit should parse");

        assert_eq!(circuit.reference_detector_signs(), vec![true, false]);
        assert_eq!(
            circuit.reference_observable_signs(),
            vec![false, false, false, true]
        );
    }

    #[test]
    fn reference_detector_and_observable_signs_returns_both_bool_vectors() {
        let circuit = Circuit::from_str(
            "X 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(3) rec[-1] rec[-2]",
        )
        .expect("circuit should parse");

        let (detectors, observables) = circuit.reference_detector_and_observable_signs();

        assert_eq!(detectors, vec![true, false]);
        assert_eq!(observables, vec![false, false, false, true]);
    }

    #[test]
    fn compile_sampler_samples_bit_packed_shots() {
        let circuit = Circuit::from_str("X 0\nM 0 1").expect("circuit should parse");
        let mut sampler = circuit.compile_sampler(false);

        assert_eq!(sampler.num_measurements(), 2);
        assert_eq!(sampler.sample_bit_packed(3), vec![0b0000_0001; 3]);
    }

    #[test]
    fn compile_sampler_samples_unpacked_shots() {
        let circuit = Circuit::from_str("X 0\nM 0 1").expect("circuit should parse");
        let mut sampler = circuit.compile_sampler(false);

        assert_eq!(
            sampler.sample(2),
            bool_matrix(vec![vec![true, false], vec![true, false]])
        );
    }

    #[test]
    fn compile_sampler_writes_shots_to_file() {
        let path = unique_temp_path("sampler-write");
        let circuit = Circuit::from_str("X 0\nM 0 1").expect("circuit should parse");
        let mut sampler = circuit.compile_sampler(false);

        sampler
            .sample_write(2, &path, "01")
            .expect("sample_write should succeed");

        assert_eq!(
            fs::read_to_string(&path).expect("temp file should read"),
            "10\n10\n"
        );

        fs::remove_file(path).expect("temp file should delete");
    }

    #[test]
    fn compile_sampler_with_seed_is_repeatable() {
        let circuit = Circuit::from_str("X_ERROR(0.5) 0\nM 0").expect("circuit should parse");
        let mut sampler_a = circuit.compile_sampler_with_seed(false, 5);
        let mut sampler_b = circuit.compile_sampler_with_seed(false, 5);

        assert_eq!(
            sampler_a.sample_bit_packed(8),
            sampler_b.sample_bit_packed(8)
        );
    }

    #[test]
    fn compile_detector_sampler_samples_bit_packed_detectors() {
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]").expect("circuit should parse");
        let mut sampler = circuit.compile_detector_sampler_with_seed(11);

        assert_eq!(sampler.num_detectors(), 1);
        assert_eq!(sampler.sample_bit_packed(3), vec![0b0000_0000; 3]);
    }

    #[test]
    fn compile_detector_sampler_samples_unpacked_detectors() {
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]").expect("circuit should parse");
        let mut sampler = circuit.compile_detector_sampler_with_seed(11);

        assert_eq!(
            sampler.sample(2),
            bool_matrix(vec![vec![false], vec![false]])
        );
    }

    #[test]
    fn compile_detector_sampler_writes_shots_to_file() {
        let path = unique_temp_path("detector-sampler-write");
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]").expect("circuit should parse");
        let mut sampler = circuit.compile_detector_sampler_with_seed(11);

        sampler
            .sample_write(2, &path, "01")
            .expect("sample_write should succeed");

        assert_eq!(
            fs::read_to_string(&path).expect("temp file should read"),
            "0\n0\n"
        );

        fs::remove_file(path).expect("temp file should delete");
    }

    #[test]
    fn compile_detector_sampler_can_write_observables_to_separate_file() {
        let det_path = unique_temp_path("detector-sampler-dets");
        let obs_path = unique_temp_path("detector-sampler-obs");
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
                .expect("circuit should parse");
        let mut sampler = circuit.compile_detector_sampler_with_seed(11);

        sampler
            .sample_write_separate_observables(2, &det_path, "01", &obs_path, "01")
            .expect("sample_write should succeed");

        assert_eq!(
            fs::read_to_string(&det_path).expect("det file should read"),
            "0\n0\n"
        );
        assert_eq!(
            fs::read_to_string(&obs_path).expect("obs file should read"),
            "0\n0\n"
        );

        fs::remove_file(det_path).expect("det file should delete");
        fs::remove_file(obs_path).expect("obs file should delete");
    }

    #[test]
    fn compile_detector_sampler_can_separate_observables() {
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
                .expect("circuit should parse");
        let mut sampler = circuit.compile_detector_sampler_with_seed(11);

        let (dets, obs) = sampler.sample_bit_packed_separate_observables(2);

        assert_eq!(dets, vec![0b0000_0000; 2]);
        assert_eq!(obs, vec![0b0000_0000; 2]);
    }

    #[test]
    fn compile_detector_sampler_can_unpack_separate_observables() {
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
                .expect("circuit should parse");
        let mut sampler = circuit.compile_detector_sampler_with_seed(11);

        let (dets, obs) = sampler.sample_separate_observables(2);

        assert_eq!(dets, bool_matrix(vec![vec![false], vec![false]]));
        assert_eq!(obs, bool_matrix(vec![vec![false], vec![false]]));
    }

    #[test]
    fn compile_detector_sampler_can_prepend_and_append_observables() {
        let circuit = Circuit::from_str(
            "X_ERROR(0.5) 0\nM 0\nDETECTOR rec[-1] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        )
        .expect("circuit should parse");
        let mut prepend_sampler = circuit.compile_detector_sampler_with_seed(11);
        let mut append_sampler = circuit.compile_detector_sampler_with_seed(11);
        let mut separate_sampler = circuit.compile_detector_sampler_with_seed(11);

        let prepend = prepend_sampler.sample_prepend_observables(4);
        let append = append_sampler.sample_append_observables(4);
        let (dets, obs) = separate_sampler.sample_separate_observables(4);

        let expected_prepend =
            ndarray::concatenate(ndarray::Axis(1), &[obs.view(), dets.view()]).unwrap();
        let expected_append =
            ndarray::concatenate(ndarray::Axis(1), &[dets.view(), obs.view()]).unwrap();

        assert_eq!(prepend, expected_prepend);
        assert_eq!(append, expected_append);
    }

    #[test]
    fn compile_m2d_converter_converts_bit_packed_measurements() {
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]").expect("circuit should parse");
        let converter = circuit.compile_m2d_converter(false);

        let mut converter = converter;
        let dets = converter.convert_measurements_bit_packed(&[0b0, 0b1], 2, false);

        assert_eq!(dets, vec![0b1, 0b0]);
    }

    #[test]
    fn compile_m2d_converter_can_separate_observables() {
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
                .expect("circuit should parse");
        let converter = circuit.compile_m2d_converter(false);

        let mut converter = converter;
        let (dets, obs) =
            converter.convert_measurements_bit_packed_separate_observables(&[0b0, 0b1], 2);

        assert_eq!(dets, vec![0b1, 0b0]);
        assert_eq!(obs, vec![0b1, 0b0]);
    }

    #[test]
    fn compile_m2d_converter_supports_packed_sweep_bits() {
        let circuit = Circuit::from_str(
            "X 0\nCNOT sweep[0] 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        )
        .expect("circuit should parse");
        let converter = circuit.compile_m2d_converter(false);

        let mut converter = converter;
        let dets = converter.convert_measurements_and_sweep_bits_bit_packed(
            &[0b0, 0b1, 0b0, 0b1],
            &[0b0, 0b0, 0b1, 0b1],
            4,
            true,
        );

        assert_eq!(dets, vec![0b11, 0b00, 0b00, 0b11]);
    }

    #[test]
    fn compile_m2d_converter_supports_unpacked_sweep_bits() {
        let circuit = Circuit::from_str(
            "X 0\nCNOT sweep[0] 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        )
        .expect("circuit should parse");
        let converter = circuit.compile_m2d_converter(false);

        let mut converter = converter;
        let dets = converter
            .convert_measurements_and_sweep_bits(
                bool_matrix(vec![vec![false], vec![true], vec![false], vec![true]]).view(),
                bool_matrix(vec![vec![false], vec![false], vec![true], vec![true]]).view(),
                true,
            )
            .expect("conversion should succeed");

        assert_eq!(
            dets,
            bool_matrix(vec![
                vec![true, true],
                vec![false, false],
                vec![false, false],
                vec![true, true]
            ])
        );
    }

    #[test]
    fn compile_m2d_converter_supports_sweep_bits_with_separate_observables() {
        let circuit = Circuit::from_str(
            "X 0\nCNOT sweep[0] 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        )
        .expect("circuit should parse");
        let converter = circuit.compile_m2d_converter(false);

        let mut converter = converter;
        let (dets, obs) = converter
            .convert_measurements_and_sweep_bits_separate_observables(
                bool_matrix(vec![vec![false], vec![true], vec![false], vec![true]]).view(),
                bool_matrix(vec![vec![false], vec![false], vec![true], vec![true]]).view(),
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
    fn compile_m2d_converter_converts_unpacked_measurements() {
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]").expect("circuit should parse");
        let converter = circuit.compile_m2d_converter(false);

        let mut converter = converter;
        let dets = converter
            .convert_measurements(bool_matrix(vec![vec![false], vec![true]]).view(), false)
            .expect("conversion should succeed");

        assert_eq!(dets, bool_matrix(vec![vec![true], vec![false]]));
    }

    #[test]
    fn compile_m2d_converter_can_unpack_separate_observables() {
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
                .expect("circuit should parse");
        let converter = circuit.compile_m2d_converter(false);

        let mut converter = converter;
        let (dets, obs) = converter
            .convert_measurements_separate_observables(
                bool_matrix(vec![vec![false], vec![true]]).view(),
            )
            .expect("conversion should succeed");

        assert_eq!(dets, bool_matrix(vec![vec![true], vec![false]]));
        assert_eq!(obs, bool_matrix(vec![vec![true], vec![false]]));
    }

    #[test]
    fn compile_m2d_converter_can_convert_measurements_from_files() {
        let measurements_path = unique_temp_path("m2d-measurements");
        let detections_path = unique_temp_path("m2d-detections");
        let appended_path = unique_temp_path("m2d-appended");
        let obs_path = unique_temp_path("m2d-obs");
        let circuit =
            Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
                .expect("circuit should parse");
        let mut converter = circuit.compile_m2d_converter(false);

        fs::write(&measurements_path, "0\n1\n").expect("measurement file should write");

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
            .expect("convert_file should succeed");
        assert_eq!(
            fs::read_to_string(&detections_path).expect("detection file should read"),
            "1\n0\n"
        );

        converter
            .convert_file(
                &measurements_path,
                "01",
                None::<&std::path::Path>,
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
            "11\n00\n"
        );

        converter
            .convert_file(
                &measurements_path,
                "01",
                None::<&std::path::Path>,
                "01",
                &detections_path,
                "01",
                false,
                Some(obs_path.as_path()),
                "01",
            )
            .expect("convert_file should separate observables");
        assert_eq!(
            fs::read_to_string(&detections_path).expect("detection file should read"),
            "1\n0\n"
        );
        assert_eq!(
            fs::read_to_string(&obs_path).expect("observable file should read"),
            "1\n0\n"
        );

        fs::remove_file(measurements_path).expect("measurement file should delete");
        fs::remove_file(detections_path).expect("detection file should delete");
        fs::remove_file(appended_path).expect("appended file should delete");
        fs::remove_file(obs_path).expect("observable file should delete");
    }

    #[test]
    fn gate_data_lookup_exposes_basic_metadata() {
        let gate = Gate::new("cnot").expect("gate should exist");

        assert_eq!(gate.name(), "CX");
        assert!(gate.aliases().contains(&"CNOT".to_string()));
        assert!(gate.aliases().contains(&"CX".to_string()));
        assert!(gate.is_two_qubit_gate());
        assert!(gate.is_unitary());
        assert!(!gate.is_reset());
    }

    #[test]
    fn gate_data_exposes_more_metadata_flags() {
        let h = Gate::H;
        let x_error = Gate::X_ERROR;
        let m = Gate::M;
        let r = Gate::R;
        let detector = Gate::DETECTOR;

        assert!(h.is_single_qubit_gate());
        assert!(!h.is_noisy_gate());
        assert_eq!(h.num_parens_arguments_range(), vec![0]);

        assert!(x_error.is_noisy_gate());
        assert_eq!(x_error.num_parens_arguments_range(), vec![1]);

        assert_eq!(m.num_parens_arguments_range(), vec![0, 1]);
        assert!(r.is_reset());
        assert_eq!(r.num_parens_arguments_range(), vec![0]);

        assert!(m.produces_measurements());
        assert!(!detector.produces_measurements());
        assert!(detector.takes_measurement_record_targets());
        assert!(!h.takes_measurement_record_targets());
        assert!(!detector.takes_pauli_targets());
    }

    #[test]
    fn gate_data_exposes_inverse_relationships() {
        let h = Gate::H;
        let s = Gate::S;
        let x_error = Gate::X_ERROR;
        let r = Gate::R;

        assert_eq!(h.inverse().expect("H has inverse").name(), "H");
        assert_eq!(s.inverse().expect("S has inverse").name(), "S_DAG");
        assert!(x_error.inverse().is_none());

        assert_eq!(x_error.generalized_inverse().name(), "X_ERROR");
        assert_eq!(r.generalized_inverse().name(), "M");
    }

    #[test]
    fn gate_data_exposes_hadamard_conjugation() {
        let x = Gate::X;
        let cx = Gate::CX;
        let ry = Gate::RY;

        assert_eq!(
            x.hadamard_conjugated(false)
                .expect("X has H conjugate")
                .name(),
            "Z"
        );
        assert_eq!(
            cx.hadamard_conjugated(false)
                .expect("CX has H conjugate")
                .name(),
            "XCZ"
        );
        assert!(ry.hadamard_conjugated(false).is_none());
        assert_eq!(
            ry.hadamard_conjugated(true)
                .expect("unsigned H conjugate exists")
                .name(),
            "RY"
        );
    }

    #[test]
    fn gate_data_exposes_symmetric_gate_property() {
        let cx = Gate::CX;
        let cz = Gate::CZ;
        let h = Gate::H;
        let detector = Gate::DETECTOR;

        assert!(!cx.is_symmetric_gate());
        assert!(cz.is_symmetric_gate());
        assert!(h.is_symmetric_gate());
        assert!(!detector.is_symmetric_gate());
    }

    #[test]
    fn gate_data_supports_identity_traits() {
        let canonical = Gate::CX;
        let alias = Gate::new("cnot").expect("alias should resolve");
        let cloned = alias;
        let other = Gate::H;

        assert_eq!(canonical, alias);
        assert_eq!(cloned, canonical);
        assert_ne!(canonical, other);
    }

    #[test]
    fn gate_data_display_and_debug_use_canonical_names() {
        let gate = Gate::new("mpp").expect("gate should exist");
        let alias = Gate::new("cnot").expect("alias should resolve");

        assert_eq!(gate.to_string(), "MPP");
        assert_eq!(format!("{gate:?}"), "stim::Gate::MPP");
        assert_eq!(alias.to_string(), "CX");
        assert_eq!(format!("{alias:?}"), "stim::Gate::CX");
    }

    #[test]
    fn gate_data_lookup_reports_unknown_gates() {
        let error = Gate::new("definitely_not_a_gate").expect_err("unknown gate should fail");

        assert!(error.message().contains("definitely_not_a_gate"));
    }

    #[test]
    fn all_gate_data_returns_canonical_gate_inventory() {
        let inventory = all_gates();

        assert!(inventory.contains(&Gate::CX));
        assert!(inventory.contains(&Gate::DETECTOR));
        assert!(inventory.contains(&Gate::H));
        assert!(inventory.contains(&Gate::MPP));

        let cx = inventory
            .iter()
            .find(|gate| **gate == Gate::CX)
            .expect("CX should be present");
        assert_eq!(*cx, Gate::new("cnot").expect("lookup should resolve alias"));
        assert_eq!(cx.name(), "CX");
        assert!(cx.aliases().contains(&"CNOT".to_string()));

        for gate in inventory {
            let name = gate.name();
            assert_eq!(
                *gate,
                Gate::new(name).expect("inventory key should roundtrip")
            );
            assert_eq!(gate.to_string(), name);
        }
    }
}

#[cfg(test)]
mod residual_api_tests {
    use std::collections::BTreeMap;
    use std::str::FromStr;

    use super::Circuit;
    use crate::DetectingRegionFilter;
    use crate::{
        CircuitInstruction, CircuitItem, CircuitRepeatBlock, DemTarget, DetectorErrorModel, Flow,
        GateTarget, PauliString,
    };
    #[test]
    fn circuit_get_detector_coordinates_matches_documented_examples() {
        let circuit = Circuit::from_str(
            "\
    M 0
    DETECTOR(1, 2, 3) rec[-1]
    SHIFT_COORDS(5)
    M 1
    DETECTOR(1, 2) rec[-1]",
        )
        .unwrap();

        assert_eq!(
            circuit.get_detector_coordinates(None).unwrap(),
            BTreeMap::from([(0, vec![1.0, 2.0, 3.0]), (1, vec![6.0, 2.0])])
        );
        assert_eq!(
            circuit.get_detector_coordinates(Some(&[1])).unwrap(),
            BTreeMap::from([(1, vec![6.0, 2.0])])
        );
    }

    #[test]
    fn circuit_get_detector_coordinates_reports_empty_coords_and_range_errors() {
        let circuit = Circuit::from_str(
            "\
    M 0
    DETECTOR rec[-1]
    M 1
    DETECTOR(4) rec[-1]",
        )
        .unwrap();

        assert_eq!(
            circuit.get_detector_coordinates(None).unwrap(),
            BTreeMap::from([(0, vec![]), (1, vec![4.0])])
        );
        assert!(circuit.get_detector_coordinates(Some(&[2])).is_err());
    }

    #[test]
    fn circuit_get_final_qubit_coordinates_matches_documented_examples() {
        let circuit = Circuit::from_str("QUBIT_COORDS(1, 2, 3) 1").unwrap();

        assert_eq!(
            circuit.get_final_qubit_coordinates().unwrap(),
            BTreeMap::from([(1, vec![1.0, 2.0, 3.0])])
        );
    }

    #[test]
    fn circuit_get_final_qubit_coordinates_tracks_last_assignment() {
        let circuit = Circuit::from_str(
            "\
    QUBIT_COORDS(1, 2) 0
    QUBIT_COORDS(3) 1
    QUBIT_COORDS(9, 8) 0",
        )
        .unwrap();

        assert_eq!(
            circuit.get_final_qubit_coordinates().unwrap(),
            BTreeMap::from([(0, vec![1.0, 2.0, 9.0, 8.0]), (1, vec![3.0])])
        );
    }

    #[test]
    fn circuit_missing_detectors_matches_documented_examples() {
        assert_eq!(
            Circuit::from_str("R 0\nM 0")
                .unwrap()
                .missing_detectors(false),
            Circuit::from_str("DETECTOR rec[-1]").unwrap()
        );

        assert_eq!(
            Circuit::from_str(
                "\
    MZZ 0 1
    MYY 0 1
    MXX 0 1
    DEPOLARIZE1(0.1) 0 1
    MZZ 0 1
    MYY 0 1
    MXX 0 1
    DETECTOR rec[-1] rec[-4]
    DETECTOR rec[-2] rec[-5]
    DETECTOR rec[-3] rec[-6]"
            )
            .unwrap()
            .missing_detectors(true),
            Circuit::from_str("DETECTOR rec[-3] rec[-2] rec[-1]").unwrap()
        );
    }

    #[test]
    fn circuit_missing_detectors_returns_empty_circuit_when_none_are_missing() {
        let circuit = Circuit::from_str(
            "\
    R 0
    M 0
    DETECTOR rec[-1]",
        )
        .unwrap();

        assert_eq!(circuit.missing_detectors(false), Circuit::new());
    }

    #[test]
    fn circuit_with_inlined_feedback_matches_documented_example() {
        let circuit = Circuit::from_str(
            r"
            CX 0 1
            M 1
            CX rec[-1] 1
            CX 0 1
            M 1
            DETECTOR rec[-1] rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-1]
        ",
        )
        .unwrap();

        let expected = Circuit::from_str(
            r"
            CX 0 1
            M 1
            OBSERVABLE_INCLUDE(0) rec[-1]
            CX 0 1
            M 1
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        ",
        )
        .unwrap();

        assert_eq!(circuit.with_inlined_feedback(), expected);
    }

    #[test]
    fn left_repeat_matches_documented_behavior() {
        let circuit = Circuit::from_str("X 0\nY 1 2").unwrap();

        let zero = u64::default();
        let one = 1_u64;
        assert_eq!((zero * circuit.clone()).to_string(), "");
        assert_eq!((one * circuit.clone()).to_string(), circuit.to_string());
        assert_eq!(
            (3_u64 * circuit).to_string(),
            "REPEAT 3 {\n    X 0\n    Y 1 2\n}"
        );

        let model = DetectorErrorModel::from_str("error(0.25) D0\nshift_detectors 1").unwrap();

        assert_eq!((zero * model.clone()).to_string(), "");
        assert_eq!((one * model.clone()).to_string(), model.to_string());
        assert_eq!(
            (3_u64 * model).to_string(),
            "repeat 3 {\n    error(0.25) D0\n    shift_detectors 1\n}"
        );
    }

    fn parse_circuit(text: &str) -> Circuit {
        Circuit::from_str(text).expect("circuit should parse")
    }

    fn gate(name: &str) -> crate::Gate {
        name.parse().expect("gate should parse")
    }

    #[test]
    fn circuit_append_gate_targets_supports_qubit_targets() {
        let mut circuit = Circuit::new();

        circuit
            .append_gate_targets(
                gate("CX"),
                &[
                    GateTarget::from(0_u32),
                    GateTarget::from(1_u32),
                    GateTarget::from(2_u32),
                    GateTarget::from(3_u32),
                ],
                &[],
            )
            .expect("qubit targets should append");
        circuit
            .append_gate_targets(
                gate("H"),
                &[GateTarget::from(4_u32), GateTarget::from(5_u32)],
                &[],
            )
            .expect("single-qubit targets should append");

        assert_eq!(
            circuit,
            parse_circuit(
                "CX 0 1 2 3
    H 4 5"
            )
        );
    }

    #[test]
    fn circuit_append_gate_targets_supports_pauli_targets() {
        let mut circuit = Circuit::new();

        circuit
            .append_gate_targets(
                gate("CORRELATED_ERROR"),
                &[
                    crate::GateTarget::x(0_u32, false).expect("X target should construct"),
                    crate::GateTarget::y(1_u32, false).expect("Y target should construct"),
                    crate::GateTarget::pauli(2, 'Z', false).expect("Z target should construct"),
                ],
                &[0.125],
            )
            .expect("pauli targets should append");
        assert_eq!(circuit, parse_circuit("E(0.125) X0 Y1 Z2"));
    }

    #[test]
    fn circuit_append_gate_targets_supports_measurement_record_targets() {
        let mut circuit = Circuit::new();

        circuit
            .append_gate_targets(
                gate("M"),
                &[GateTarget::from(0_u32), GateTarget::from(1_u32)],
                &[],
            )
            .expect("measurements should append");
        circuit
            .append_gate_targets(
                gate("CX"),
                &[
                    crate::GateTarget::rec(-1).expect("record target should construct"),
                    GateTarget::from(5_u32),
                ],
                &[],
            )
            .expect("record-controlled CX should append");
        circuit
            .append_gate_targets(
                gate("DETECTOR"),
                &[
                    crate::GateTarget::rec(-1).expect("record target should construct"),
                    crate::GateTarget::rec(-2).expect("record target should construct"),
                ],
                &[],
            )
            .expect("detector targets should append");
        circuit
            .append_gate_targets(
                gate("OBSERVABLE_INCLUDE"),
                &[
                    crate::GateTarget::rec(-1).expect("record target should construct"),
                    crate::GateTarget::rec(-2).expect("record target should construct"),
                ],
                &[3.0],
            )
            .expect("observable targets should append");

        assert_eq!(
            circuit,
            parse_circuit(
                "M 0 1
    CX rec[-1] 5
    DETECTOR rec[-1] rec[-2]
    OBSERVABLE_INCLUDE(3) rec[-1] rec[-2]"
            )
        );
    }

    #[test]
    fn circuit_append_gate_targets_supports_combined_pauli_products_for_mpp() {
        let mut circuit = Circuit::new();

        let mut targets = crate::target_combined_paulis(
            &[
                crate::GateTarget::x(1_u32, false).expect("X target should construct"),
                crate::GateTarget::y(2_u32, false).expect("Y target should construct"),
                crate::GateTarget::z(3_u32, false).expect("Z target should construct"),
            ],
            false,
        )
        .expect("combined pauli product should construct");
        targets.push(crate::GateTarget::y(4_u32, false).expect("single Y target should construct"));
        targets.push(crate::GateTarget::z(5_u32, false).expect("single Z target should construct"));
        targets.extend(
            crate::target_combined_paulis(
                &[
                    crate::GateTarget::x(1_u32, false).expect("X target should construct"),
                    crate::GateTarget::x(2_u32, false).expect("X target should construct"),
                ],
                true,
            )
            .expect("inverted combined pauli product should construct"),
        );

        circuit
            .append_gate_targets(gate("MPP"), &targets, &[])
            .expect("MPP targets should append");

        assert_eq!(circuit, parse_circuit("MPP X1*Y2*Z3 Y4 Z5 !X1*X2"));
    }

    #[test]
    fn circuit_append_operation_accepts_instruction_values() {
        let mut circuit = Circuit::new();

        circuit
            .append_operation(CircuitInstruction::new("X", [0u32], [], "").unwrap())
            .unwrap();
        circuit
            .append_operation(CircuitInstruction::new("H", [0u32, 1u32], [], "").unwrap())
            .unwrap();

        assert_eq!(circuit, Circuit::from_str("X 0\nH 0 1").unwrap());
    }

    #[test]
    fn circuit_append_operation_accepts_repeat_blocks() {
        let mut circuit = Circuit::from_str("H 0").unwrap();
        let repeat =
            CircuitRepeatBlock::new(2, &Circuit::from_str("X 1").unwrap(), "loop").unwrap();

        circuit.append_operation(repeat).unwrap();

        assert_eq!(
            circuit,
            Circuit::from_str("H 0\nREPEAT[loop] 2 {\n    X 1\n}").unwrap()
        );
    }

    #[test]
    fn circuit_append_operation_accepts_circuits_and_ignores_empty_input() {
        let mut circuit = Circuit::from_str("H 0").unwrap();

        circuit
            .append_operation(Circuit::from_str("S 999\nCX 0 1\nCZ 2 3").unwrap())
            .unwrap();
        circuit.append_operation(Circuit::new()).unwrap();

        assert_eq!(
            circuit,
            Circuit::from_str("H 0\nS 999\nCX 0 1\nCZ 2 3").unwrap()
        );
    }

    #[test]
    fn circuit_append_operation_copies_inputs_instead_of_aliasing() {
        let mut circuit = Circuit::new();
        let instruction = CircuitInstruction::new("Y", [3u32, 4u32, 5u32], [], "").unwrap();
        let inserted = Circuit::from_str("S 1\nX 2").unwrap();

        circuit.append_operation(&instruction).unwrap();
        circuit.append_operation(&inserted).unwrap();

        assert_eq!(
            instruction,
            CircuitInstruction::new("Y", [3u32, 4u32, 5u32], [], "").unwrap()
        );
        assert_eq!(inserted, Circuit::from_str("S 1\nX 2").unwrap());
        assert_eq!(circuit, Circuit::from_str("Y 3 4 5\nS 1\nX 2").unwrap());
    }

    #[test]
    fn circuit_get_returns_top_level_instruction_and_repeat_block_copies() {
        let circuit = Circuit::from_str(
            "\
    X 0
    X_ERROR(0.5) 2
    REPEAT 100 {
        X 0
        Y 1 2
    }
    TICK
    M 0
    DETECTOR rec[-1]",
        )
        .unwrap();

        assert_eq!(
            circuit.get(1).unwrap(),
            CircuitItem::Instruction(
                CircuitInstruction::new("X_ERROR", [2u32], [0.5], "").unwrap()
            )
        );
        assert_eq!(
            circuit.get(2).unwrap(),
            CircuitItem::RepeatBlock(
                CircuitRepeatBlock::new(100, &Circuit::from_str("X 0\nY 1 2").unwrap(), "")
                    .unwrap()
            )
        );
        assert_eq!(
            circuit.get(-1).unwrap(),
            CircuitItem::Instruction(
                CircuitInstruction::new(
                    "DETECTOR",
                    [GateTarget::from_target_str("rec[-1]").unwrap()],
                    [],
                    ""
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn circuit_slice_matches_documented_and_reverse_examples() {
        let circuit = Circuit::from_str(
            "\
    X 0
    X_ERROR(0.5) 2
    REPEAT 100 {
        X 0
        Y 1 2
    }
    TICK
    M 0
    DETECTOR rec[-1]",
        )
        .unwrap();

        assert_eq!(
            circuit.slice(Some(1), None, 2).unwrap(),
            Circuit::from_str(
                "\
    X_ERROR(0.5) 2
    TICK
    DETECTOR rec[-1]"
            )
            .unwrap()
        );
        assert_eq!(
            circuit.slice(None, None, -1).unwrap(),
            Circuit::from_str(
                "\
    DETECTOR rec[-1]
    M 0
    TICK
    REPEAT 100 {
        X 0
        Y 1 2
    }
    X_ERROR(0.5) 2
    X 0"
            )
            .unwrap()
        );
    }

    #[test]
    fn circuit_pop_removes_and_returns_requested_item() {
        let mut circuit = Circuit::from_str(
            "\
    H 0
    S 1
    X 2
    Y 3",
        )
        .unwrap();

        assert_eq!(
            circuit.pop(-1).unwrap(),
            CircuitItem::Instruction(CircuitInstruction::new("Y", [3u32], [], "").unwrap())
        );
        assert_eq!(
            circuit.pop(1).unwrap(),
            CircuitItem::Instruction(CircuitInstruction::new("S", [1u32], [], "").unwrap())
        );
        assert_eq!(circuit, Circuit::from_str("H 0\nX 2").unwrap());
    }

    #[test]
    fn circuit_get_slice_and_pop_validate_bounds() {
        let mut circuit = Circuit::from_str("X 0").unwrap();

        assert!(circuit.get(2).is_err());
        assert!(circuit.slice(None, None, 0).is_err());
        assert!(circuit.pop(2).is_err());
    }

    #[test]
    fn circuit_popped_and_sliced_circuits_are_independent() {
        let mut circuit = Circuit::from_str("X 0\nM 0\nDETECTOR rec[-1]").unwrap();
        let popped = circuit.pop(1).unwrap();
        let mut sliced = circuit.slice(None, None, 1).unwrap();
        sliced.clear();

        assert_eq!(
            popped,
            CircuitItem::Instruction(CircuitInstruction::new("M", [0u32], [], "").unwrap())
        );
        assert_eq!(circuit, Circuit::from_str("X 0\nDETECTOR rec[-1]").unwrap());
        assert_eq!(sliced, Circuit::new());
    }

    #[test]
    fn circuit_supports_borrowed_and_owned_iteration() {
        let circuit = Circuit::from_str(
            "\
    H 0
    REPEAT 2 {
        X 1
    }
    M 0",
        )
        .unwrap();

        let expected = vec![
            CircuitItem::Instruction(CircuitInstruction::new("H", [0u32], [], "").unwrap()),
            CircuitItem::RepeatBlock(
                CircuitRepeatBlock::new(2, &Circuit::from_str("X 1").unwrap(), "").unwrap(),
            ),
            CircuitItem::Instruction(CircuitInstruction::new("M", [0u32], [], "").unwrap()),
        ];
        let mut mutable = circuit.clone();

        assert_eq!((&circuit).into_iter().collect::<Vec<_>>(), expected);
        assert_eq!((&mut mutable).into_iter().collect::<Vec<_>>(), expected);
        assert_eq!(circuit.clone().into_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn circuit_supports_indexing_top_level_items() {
        let circuit = Circuit::from_str(
            "\
    H 0
    REPEAT 2 {
        X 1
    }
    M 0",
        )
        .unwrap();

        assert_eq!(
            circuit[0],
            CircuitItem::Instruction(CircuitInstruction::new("H", [0u32], [], "").unwrap())
        );
        assert_eq!(
            circuit[1],
            CircuitItem::RepeatBlock(
                CircuitRepeatBlock::new(2, &Circuit::from_str("X 1").unwrap(), "").unwrap()
            )
        );
    }

    #[test]
    fn circuit_index_cache_invalidates_after_mutation() {
        let mut circuit = Circuit::from_str("H 0").unwrap();
        assert_eq!(
            circuit[0],
            CircuitItem::Instruction(CircuitInstruction::new("H", [0u32], [], "").unwrap())
        );

        circuit.append_from_stim_program_text("M 0").unwrap();

        assert_eq!(circuit.len(), 2);
        assert_eq!(
            circuit[1],
            CircuitItem::Instruction(CircuitInstruction::new("M", [0u32], [], "").unwrap())
        );
    }

    #[test]
    fn circuit_parser_preserves_repeat_tags() {
        let circuit = Circuit::from_str(
            "\
    REPEAT[look] 2 {
        X 0
    }",
        )
        .unwrap();

        assert_eq!(
            circuit.get(0).unwrap(),
            CircuitItem::RepeatBlock(
                CircuitRepeatBlock::new(2, &Circuit::from_str("X 0").unwrap(), "look").unwrap()
            )
        );
    }

    #[test]
    fn circuit_insert_accepts_single_instruction() {
        let mut circuit = Circuit::from_str(
            "\
    H 0
    S 1
    X 2",
        )
        .unwrap();

        circuit
            .insert(
                1,
                CircuitInstruction::new("Y", [3u32, 4u32, 5u32], [], "").unwrap(),
            )
            .unwrap();

        assert_eq!(
            circuit,
            Circuit::from_str("H 0\nY 3 4 5\nS 1\nX 2").unwrap()
        );
    }

    #[test]
    fn circuit_insert_accepts_circuit_and_preserves_normalized_fusion() {
        let mut circuit = Circuit::from_str(
            "\
    H 0
    Y 3 4 5
    S 1
    X 2",
        )
        .unwrap();

        circuit
            .insert(-1, Circuit::from_str("S 999\nCX 0 1\nCZ 2 3").unwrap())
            .unwrap();

        assert_eq!(
            circuit,
            Circuit::from_str("H 0\nY 3 4 5\nS 1 999\nCX 0 1\nCZ 2 3\nX 2").unwrap()
        );
    }

    #[test]
    fn circuit_insert_can_place_repeat_blocks() {
        let mut circuit = Circuit::from_str("H 0\nM 0").unwrap();
        let repeat =
            CircuitRepeatBlock::new(2, &Circuit::from_str("X 1").unwrap(), "loop").unwrap();

        circuit.insert(1, repeat).unwrap();

        assert_eq!(
            circuit,
            Circuit::from_str("H 0\nREPEAT[loop] 2 {\n    X 1\n}\nM 0").unwrap()
        );
    }

    #[test]
    fn circuit_insert_validates_bounds_and_empty_circuit_case() {
        let mut circuit = Circuit::new();
        assert!(
            circuit
                .insert(0, CircuitInstruction::new("X", [0u32], [], "").unwrap())
                .is_err()
        );

        let mut non_empty = Circuit::from_str("X 0\nY 1").unwrap();
        assert!(
            non_empty
                .insert(2, CircuitInstruction::new("Z", [2u32], [], "").unwrap())
                .is_err()
        );
        assert!(
            non_empty
                .insert(-3, CircuitInstruction::new("Z", [2u32], [], "").unwrap())
                .is_err()
        );
    }

    #[test]
    fn circuit_insert_copies_inputs_instead_of_aliasing_them() {
        let mut circuit = Circuit::from_str("H 0\nM 0").unwrap();
        let op = CircuitInstruction::new("X", [1u32], [], "").unwrap();
        let inserted = Circuit::from_str("S 2").unwrap();

        circuit.insert(1, &op).unwrap();
        circuit.insert(0, &inserted).unwrap();

        assert_eq!(op, CircuitInstruction::new("X", [1u32], [], "").unwrap());
        assert_eq!(inserted, Circuit::from_str("S 2").unwrap());
        assert_eq!(circuit, Circuit::from_str("S 2\nH 0\nX 1\nM 0").unwrap());
    }

    #[test]
    fn circuit_flow_generators_match_documented_examples() {
        assert_eq!(
            Circuit::from_str("H 0").unwrap().flow_generators().unwrap(),
            vec![Flow::new("X -> Z").unwrap(), Flow::new("Z -> X").unwrap(),]
        );

        assert_eq!(
            Circuit::from_str("M 0").unwrap().flow_generators().unwrap(),
            vec![
                Flow::new("1 -> Z xor rec[0]").unwrap(),
                Flow::new("Z -> rec[0]").unwrap(),
            ]
        );

        assert_eq!(
            Circuit::from_str("RX 0")
                .unwrap()
                .flow_generators()
                .unwrap(),
            vec![Flow::new("1 -> X").unwrap()]
        );

        let flows: Vec<String> = Circuit::from_str("MXX 0 1")
            .unwrap()
            .flow_generators()
            .unwrap()
            .into_iter()
            .map(|flow| flow.to_string())
            .collect();
        assert_eq!(
            flows,
            vec![
                "1 -> XX xor rec[0]",
                "_X -> _X",
                "X_ -> _X xor rec[0]",
                "ZZ -> ZZ"
            ]
        );
    }

    #[test]
    fn circuit_has_all_flows_matches_documented_examples() {
        let h = Circuit::from_str("H 0").unwrap();

        assert!(
            !h.has_all_flows(
                &[
                    Flow::new("X -> Z").unwrap(),
                    Flow::new("Y -> Y").unwrap(),
                    Flow::new("Z -> X").unwrap(),
                ],
                false,
            )
            .unwrap()
        );

        assert!(
            h.has_all_flows(
                &[
                    Flow::new("X -> Z").unwrap(),
                    Flow::new("Y -> -Y").unwrap(),
                    Flow::new("Z -> X").unwrap(),
                ],
                false,
            )
            .unwrap()
        );

        assert!(
            h.has_all_flows(
                &[
                    Flow::new("X -> Z").unwrap(),
                    Flow::new("Y -> Y").unwrap(),
                    Flow::new("Z -> X").unwrap(),
                ],
                true,
            )
            .unwrap()
        );
    }

    #[test]
    fn circuit_has_flow_matches_documented_examples() {
        let m = Circuit::from_str("M 0").unwrap();
        assert!(m.has_flow(&Flow::new("Z -> Z").unwrap(), false).unwrap());
        assert!(!m.has_flow(&Flow::new("X -> X").unwrap(), false).unwrap());
        assert!(!m.has_flow(&Flow::new("Z -> I").unwrap(), false).unwrap());
        assert!(
            m.has_flow(&Flow::new("Z -> I xor rec[-1]").unwrap(), false)
                .unwrap()
        );
        assert!(
            m.has_flow(&Flow::new("Z -> rec[-1]").unwrap(), false)
                .unwrap()
        );

        let cx58 = Circuit::from_str("CX 5 8").unwrap();
        assert!(
            cx58.has_flow(&Flow::new("X5 -> X5*X8").unwrap(), false)
                .unwrap()
        );
        assert!(
            !cx58
                .has_flow(&Flow::new("X_ -> XX").unwrap(), false)
                .unwrap()
        );
        assert!(
            cx58.has_flow(&Flow::new("_____X___ -> _____X__X").unwrap(), false)
                .unwrap()
        );

        let ry = Circuit::from_str("RY 0").unwrap();
        assert!(ry.has_flow(&Flow::new("1 -> Y").unwrap(), false).unwrap());
        assert!(!ry.has_flow(&Flow::new("1 -> X").unwrap(), false).unwrap());

        let cx01 = Circuit::from_str("CX 0 1").unwrap();
        let flow = Flow::new("+X_ -> +XX").unwrap();
        assert!(cx01.has_flow(&flow, false).unwrap());

        let h0 = Circuit::from_str("H 0").unwrap();
        let y_flow = Flow::new("Y -> Y").unwrap();
        assert!(h0.has_flow(&y_flow, true).unwrap());
        assert!(!h0.has_flow(&y_flow, false).unwrap());
    }

    #[test]
    fn circuit_solve_flow_measurements_matches_documented_examples() {
        assert_eq!(
            Circuit::from_str("M 2")
                .unwrap()
                .solve_flow_measurements(&[Flow::new("Z2 -> 1").unwrap()])
                .unwrap(),
            vec![Some(vec![0])]
        );

        assert_eq!(
            Circuit::from_str("M 2")
                .unwrap()
                .solve_flow_measurements(&[Flow::new("X2 -> X2").unwrap()])
                .unwrap(),
            vec![None]
        );

        assert_eq!(
            Circuit::from_str("MXX 0 1")
                .unwrap()
                .solve_flow_measurements(&[Flow::new("YY -> ZZ").unwrap()])
                .unwrap(),
            vec![Some(vec![0])]
        );

        assert_eq!(
            Circuit::from_str(
                r"
                R 1 3
                CX 0 1 2 3
                CX 4 3 2 1
                M 1 3
            ",
            )
            .unwrap()
            .solve_flow_measurements(&[
                Flow::new("1 -> Z0*Z4").unwrap(),
                Flow::new("Z0 -> Z2").unwrap(),
                Flow::new("X0*X2*X4 -> X0*X2*X4").unwrap(),
                Flow::new("Y0 -> Y0").unwrap(),
            ])
            .unwrap(),
            vec![Some(vec![0, 1]), Some(vec![0]), Some(vec![]), None]
        );
    }

    #[test]
    fn circuit_time_reversed_for_flows_matches_documented_examples() {
        assert_eq!(
            Circuit::new().time_reversed_for_flows(&[], false).unwrap(),
            (Circuit::new(), vec![])
        );

        let (inv_circuit, inv_flows) = Circuit::from_str("M 0")
            .unwrap()
            .time_reversed_for_flows(&[Flow::new("Z -> rec[-1]").unwrap()], false)
            .unwrap();
        assert_eq!(inv_circuit, Circuit::from_str("R 0").unwrap());
        assert_eq!(inv_flows, vec![Flow::new("1 -> Z").unwrap()]);
        assert!(inv_circuit.has_all_flows(&inv_flows, true).unwrap());

        let (inv_circuit, inv_flows) = Circuit::from_str("R 0")
            .unwrap()
            .time_reversed_for_flows(&[Flow::new("1 -> Z").unwrap()], false)
            .unwrap();
        assert_eq!(inv_circuit, Circuit::from_str("M 0").unwrap());
        assert_eq!(inv_flows, vec![Flow::new("Z -> rec[-1]").unwrap()]);

        let (inv_circuit, inv_flows) = Circuit::from_str("M 0")
            .unwrap()
            .time_reversed_for_flows(&[Flow::new("Z -> rec[-1]").unwrap()], true)
            .unwrap();
        assert_eq!(inv_circuit, Circuit::from_str("M 0").unwrap());
        assert_eq!(inv_flows, vec![Flow::new("1 -> Z xor rec[-1]").unwrap()]);
    }

    #[test]
    fn circuit_detecting_regions_matches_documented_example() {
        let actual = Circuit::from_str(
            r"
            R 0
            TICK
            H 0
            TICK
            CX 0 1
            TICK
            MX 0 1
            DETECTOR rec[-1] rec[-2]
        ",
        )
        .unwrap()
        .detecting_regions()
        .unwrap();

        let mut ticks = BTreeMap::new();
        ticks.insert(0, PauliString::from_text("+Z_").unwrap());
        ticks.insert(1, PauliString::from_text("+X_").unwrap());
        ticks.insert(2, PauliString::from_text("+XX").unwrap());

        let mut expected = BTreeMap::new();
        expected.insert(DemTarget::relative_detector_id(0).unwrap(), ticks);

        assert_eq!(actual, expected);
    }

    #[test]
    fn circuit_detecting_regions_with_options_filters_targets_and_ticks() {
        let circuit = Circuit::from_str(
            r"
            R 0
            TICK
            H 0
            TICK
            MX 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        ",
        )
        .unwrap();

        let filtered = circuit
            .detecting_regions_with_options(
                Some(&[DemTarget::relative_detector_id(0).unwrap()]),
                Some(&[1]),
                false,
            )
            .unwrap();

        let mut ticks = BTreeMap::new();
        ticks.insert(1, PauliString::from_text("+X").unwrap());
        let mut expected = BTreeMap::new();
        expected.insert(DemTarget::relative_detector_id(0).unwrap(), ticks);
        assert_eq!(filtered, expected);

        let observable_only = circuit
            .detecting_regions_with_options(
                Some(&[DemTarget::logical_observable_id(0).unwrap()]),
                None,
                false,
            )
            .unwrap();
        assert_eq!(observable_only.len(), 1);
        assert!(observable_only.contains_key(&DemTarget::logical_observable_id(0).unwrap()));
    }

    #[test]
    fn circuit_detecting_regions_with_filters_supports_typed_filter_variants() {
        let circuit = Circuit::from_str(
            r"
            R 0
            TICK
            H 0
            TICK
            MX 0
            DETECTOR(2, 4) rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        ",
        )
        .unwrap();

        let all_detectors = circuit
            .detecting_regions_with_filters(&[DetectingRegionFilter::AllDetectors], None, false)
            .unwrap();
        assert_eq!(all_detectors.len(), circuit.num_detectors() as usize);

        let all_observables = circuit
            .detecting_regions_with_filters(&[DetectingRegionFilter::AllObservables], None, false)
            .unwrap();
        assert_eq!(all_observables.len(), circuit.num_observables() as usize);

        let by_target = circuit
            .detecting_regions_with_filters(
                &[DetectingRegionFilter::Target(
                    DemTarget::relative_detector_id(0).unwrap(),
                )],
                Some(&[1]),
                false,
            )
            .unwrap();
        assert_eq!(by_target.len(), 1);
        assert!(by_target.contains_key(&DemTarget::relative_detector_id(0).unwrap()));

        let by_coords = circuit
            .detecting_regions_with_filters(
                &[DetectingRegionFilter::DetectorCoordinatePrefix(vec![2.0])],
                None,
                false,
            )
            .unwrap();
        assert_eq!(by_coords.len(), 1);
        assert!(by_coords.contains_key(&DemTarget::relative_detector_id(0).unwrap()));
    }

    #[test]
    fn circuit_detector_error_model_matches_documented_default_example() {
        let circuit = Circuit::from_str(
            "\
    X_ERROR(0.125) 0
    X_ERROR(0.25) 1
    CORRELATED_ERROR(0.375) X0 X1
    M 0 1
    DETECTOR rec[-2]
    DETECTOR rec[-1]",
        )
        .unwrap();

        assert_eq!(
            circuit.detector_error_model().unwrap(),
            DetectorErrorModel::from_str(
                "\
    error(0.125) D0
    error(0.375) D0 D1
    error(0.25) D1"
            )
            .unwrap()
        );
    }

    #[test]
    fn circuit_detector_error_model_options_support_decomposition_controls() {
        let circuit = Circuit::from_str(
            "\
    X_ERROR(0.125) 0
    CORRELATED_ERROR(0.25) X0 X1
    M 0 1
    DETECTOR rec[-1]
    DETECTOR rec[-1]
    DETECTOR rec[-2]
    DETECTOR rec[-2]",
        )
        .unwrap();

        assert_eq!(
            circuit
                .detector_error_model_with_options(true, false, false, 0.0, false, false)
                .unwrap(),
            DetectorErrorModel::from_str(
                "\
    error(0.125) D2 D3
    error(0.25) D2 D3 ^ D0 D1"
            )
            .unwrap()
        );

        assert!(
            circuit
                .detector_error_model_with_options(true, false, false, 0.0, false, true)
                .is_err()
        );

        assert_eq!(
            circuit
                .detector_error_model_with_options(true, false, false, 0.0, true, true)
                .unwrap(),
            DetectorErrorModel::from_str(
                "\
    error(0.25) D0 D1 D2 D3
    error(0.125) D2 D3"
            )
            .unwrap()
        );
    }

    #[test]
    fn circuit_diagram_supports_default_timeline_and_matchgraph_variants() {
        let circuit = Circuit::from_str(
            "\
    H 0
    CNOT 0 1",
        )
        .unwrap();

        let timeline = circuit
            .diagram(crate::CircuitDiagramType::TimelineText)
            .unwrap();
        assert!(timeline.contains("q0:"));
        assert!(timeline.contains("q1:"));

        let svg = circuit
            .diagram(crate::CircuitDiagramType::TimelineSvg)
            .unwrap();
        let svg_html = circuit
            .diagram(crate::CircuitDiagramType::TimelineSvgHtml)
            .unwrap();
        let gltf = circuit
            .diagram(crate::CircuitDiagramType::Timeline3d)
            .unwrap();
        let gltf_html = circuit
            .diagram(crate::CircuitDiagramType::Timeline3dHtml)
            .unwrap();
        let interactive = circuit
            .diagram(crate::CircuitDiagramType::Interactive)
            .unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg_html.contains("<svg"));
        assert!(gltf.contains("\"nodes\"") || gltf.contains("\"scenes\""));
        assert!(gltf_html.contains("<html") || gltf_html.contains("iframe"));
        assert!(interactive.contains("<html") || interactive.contains("Crumble"));

        let detector_circuit = Circuit::from_str(
            "\
    X_ERROR(0.125) 0
    M 0
    DETECTOR rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-1]",
        )
        .unwrap();
        let match_svg = detector_circuit
            .diagram(crate::CircuitDiagramType::MatchGraphSvg)
            .unwrap();
        let match_svg_alias = detector_circuit
            .diagram("match-graph-svg".parse().unwrap())
            .unwrap();
        let match_svg_html = detector_circuit
            .diagram(crate::CircuitDiagramType::MatchGraphSvgHtml)
            .unwrap();
        let match_gltf = detector_circuit
            .diagram(crate::CircuitDiagramType::MatchGraph3d)
            .unwrap();
        let match_gltf_html = detector_circuit
            .diagram(crate::CircuitDiagramType::MatchGraph3dHtml)
            .unwrap();
        assert!(match_svg.contains("<svg"));
        assert_eq!(match_svg, match_svg_alias);
        assert!(match_svg_html.contains("iframe"));
        assert!(match_gltf.contains("\"nodes\"") || match_gltf.contains("\"scenes\""));
        assert!(match_gltf_html.contains("<html") || match_gltf_html.contains("iframe"));
    }

    #[test]
    fn circuit_diagram_type_parses_upstream_aliases() {
        assert_eq!(
            "timeline".parse::<crate::CircuitDiagramType>().unwrap(),
            crate::CircuitDiagramType::TimelineSvg
        );
        assert_eq!(
            "timeline-html"
                .parse::<crate::CircuitDiagramType>()
                .unwrap(),
            crate::CircuitDiagramType::TimelineSvgHtml
        );
        assert_eq!(
            "detector-slice-text"
                .parse::<crate::CircuitDiagramType>()
                .unwrap(),
            crate::CircuitDiagramType::DetSliceText
        );
        assert_eq!(
            "detslice-svg-html"
                .parse::<crate::CircuitDiagramType>()
                .unwrap(),
            crate::CircuitDiagramType::DetSliceSvg
        );
        assert_eq!(
            "time-slice-svg"
                .parse::<crate::CircuitDiagramType>()
                .unwrap(),
            crate::CircuitDiagramType::TimeSliceSvg
        );
        assert_eq!(
            "timeslice".parse::<crate::CircuitDiagramType>().unwrap(),
            crate::CircuitDiagramType::TimeSliceSvg
        );
        assert_eq!(
            "detslice-with-ops-svg-html"
                .parse::<crate::CircuitDiagramType>()
                .unwrap(),
            crate::CircuitDiagramType::DetSliceWithOpsSvg
        );
        assert_eq!(
            "time+detector-slice-svg"
                .parse::<crate::CircuitDiagramType>()
                .unwrap(),
            crate::CircuitDiagramType::DetSliceWithOpsSvg
        );
    }

    #[test]
    fn circuit_diagram_with_tick_supports_documented_detector_slice_example() {
        let circuit = Circuit::from_str(
            "\
    H 0
    CNOT 0 1
    TICK
    M 0 1
    DETECTOR rec[-1] rec[-2]",
        )
        .unwrap();

        assert_eq!(
            circuit
                .diagram_with_tick(crate::CircuitDiagramType::DetSliceText, 1)
                .unwrap()
                .trim(),
            "q0: -Z:D0-\n     |\nq1: -Z:D0-"
        );

        let detslice_svg = circuit
            .diagram_with_tick(crate::CircuitDiagramType::DetSliceSvg, 1)
            .unwrap();
        let timeslice_svg = circuit
            .diagram_with_tick_range(crate::CircuitDiagramType::TimeSliceSvg, 1, 1, Some(1))
            .unwrap();
        let with_ops_svg = circuit
            .diagram_with_tick_range(crate::CircuitDiagramType::DetSliceWithOpsSvg, 1, 1, Some(1))
            .unwrap();
        assert!(detslice_svg.contains("<svg"));
        assert!(timeslice_svg.contains("<svg"));
        assert!(with_ops_svg.contains("<svg"));
    }

    #[test]
    fn circuit_diagram_with_filters_supports_detector_and_observable_filters() {
        let circuit = Circuit::from_str(
            "\
    R 0
    TICK
    H 0
    TICK
    MX 0
    DETECTOR(2, 4) rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-1]",
        )
        .unwrap();

        let detector_text = circuit
            .diagram_with_filters(
                crate::CircuitDiagramType::DetSliceText,
                Some((1, 1)),
                None,
                &[DetectingRegionFilter::Target(
                    DemTarget::relative_detector_id(0).unwrap(),
                )],
            )
            .unwrap();
        assert!(detector_text.contains("D0"));

        let observable_svg = circuit
            .diagram_with_filters(
                crate::CircuitDiagramType::DetSliceSvg,
                Some((1, 1)),
                Some(1),
                &[DetectingRegionFilter::Target(
                    DemTarget::logical_observable_id(0).unwrap(),
                )],
            )
            .unwrap();
        assert!(observable_svg.contains("<svg"));

        let coords_svg = circuit
            .diagram_with_filters(
                crate::CircuitDiagramType::DetSliceWithOpsSvg,
                Some((1, 1)),
                Some(1),
                &[DetectingRegionFilter::DetectorCoordinatePrefix(vec![2.0])],
            )
            .unwrap();
        assert!(coords_svg.contains("<svg"));
    }

    #[test]
    fn circuit_explain_detector_error_model_errors_matches_documented_example() {
        let circuit = Circuit::from_str(
            r"
            H 0
            CNOT 0 1
            DEPOLARIZE1(0.01) 0
            CNOT 0 1
            H 0
            M 0 1
            DETECTOR rec[-1]
            DETECTOR rec[-2]
        ",
        )
        .unwrap();

        let explained = circuit
            .explain_detector_error_model_errors(None, false)
            .unwrap();
        assert_eq!(explained.len(), 3);
        assert_eq!(
            explained[0].to_string(),
            "ExplainedError {\n    dem_error_terms: D0\n    CircuitErrorLocation {\n        flipped_pauli_product: X0\n        Circuit location stack trace:\n            (after 0 TICKs)\n            at instruction #3 (DEPOLARIZE1) in the circuit\n            at target #1 of the instruction\n            resolving to DEPOLARIZE1(0.01) 0\n    }\n}"
        );

        let filtered = circuit
            .explain_detector_error_model_errors(
                Some(&DetectorErrorModel::from_str("error(1) D0 D1").unwrap()),
                true,
            )
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(
            filtered[0].to_string(),
            "ExplainedError {\n    dem_error_terms: D0 D1\n    CircuitErrorLocation {\n        flipped_pauli_product: Y0\n        Circuit location stack trace:\n            (after 0 TICKs)\n            at instruction #3 (DEPOLARIZE1) in the circuit\n            at target #1 of the instruction\n            resolving to DEPOLARIZE1(0.01) 0\n    }\n}"
        );
    }

    #[test]
    fn circuit_sat_problem_helpers_match_documented_examples() {
        let circuit = Circuit::from_str(
            r"
            X_ERROR(0.1) 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
            X_ERROR(0.4) 0
            M 0
            DETECTOR rec[-1] rec[-2]
        ",
        )
        .unwrap();

        assert_eq!(
            circuit.shortest_error_sat_problem().unwrap(),
            "p wcnf 2 4 5\n1 -1 0\n1 -2 0\n5 -1 0\n5 2 0\n"
        );
        assert_eq!(
            circuit
                .likeliest_error_sat_problem_with_options(100, crate::SatProblemFormat::Wdimacs)
                .unwrap(),
            "p wcnf 2 4 401\n18 -1 0\n100 -2 0\n401 -1 0\n401 2 0\n"
        );
    }

    #[test]
    fn circuit_shortest_graphlike_error_matches_documented_examples() {
        let circuit = Circuit::from_str(
            r"
            TICK
            X_ERROR(0.125) 0
            Y_ERROR(0.125) 0
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
        ",
        )
        .unwrap();

        let actual = circuit.shortest_graphlike_error().unwrap();
        assert_eq!(actual.len(), 1);
        assert_eq!(
            actual[0].to_string(),
            "ExplainedError {\n    dem_error_terms: L0\n    CircuitErrorLocation {\n        flipped_pauli_product: Y0\n        Circuit location stack trace:\n            (after 1 TICKs)\n            at instruction #3 (Y_ERROR) in the circuit\n            at target #1 of the instruction\n            resolving to Y_ERROR(0.125) 0\n    }\n    CircuitErrorLocation {\n        flipped_pauli_product: X0\n        Circuit location stack trace:\n            (after 1 TICKs)\n            at instruction #2 (X_ERROR) in the circuit\n            at target #1 of the instruction\n            resolving to X_ERROR(0.125) 0\n    }\n}"
        );

        let canonical = circuit
            .shortest_graphlike_error_with_options(true, true)
            .unwrap();
        assert_eq!(canonical.len(), 1);
        assert_eq!(
            canonical[0].to_string(),
            "ExplainedError {\n    dem_error_terms: L0\n    CircuitErrorLocation {\n        flipped_pauli_product: X0\n        Circuit location stack trace:\n            (after 1 TICKs)\n            at instruction #2 (X_ERROR) in the circuit\n            at target #1 of the instruction\n            resolving to X_ERROR(0.125) 0\n    }\n}"
        );
    }

    #[test]
    fn circuit_shortest_graphlike_error_reports_documented_error_shapes() {
        let no_observable = Circuit::from_str(
            r"
            X_ERROR(0.1) 0
            M 0
            DETECTOR rec[-1]
        ",
        )
        .unwrap()
        .shortest_graphlike_error()
        .unwrap_err();
        assert!(no_observable.message().contains("NO OBSERVABLES"));

        let no_graphlike = Circuit::from_str(
            r"
            M(0.1) 0
            DETECTOR rec[-1]
            DETECTOR rec[-1]
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        ",
        )
        .unwrap()
        .shortest_graphlike_error()
        .unwrap_err();
        assert!(no_graphlike.message().contains("NO GRAPHLIKE ERRORS"));
    }

    #[test]
    fn circuit_search_for_undetectable_logical_errors_matches_documented_example() {
        let circuit = Circuit::generated_with_noise(
            "surface_code:rotated_memory_x",
            5,
            5,
            0.001,
            0.0,
            0.0,
            0.0,
        )
        .unwrap();

        let errors = circuit
            .search_for_undetectable_logical_errors(4, 4, true, false)
            .unwrap();

        assert_eq!(errors.len(), 5);
    }

    #[test]
    fn circuit_search_for_undetectable_logical_errors_reports_documented_error_shapes() {
        let no_observable = Circuit::from_str(
            r"
            X_ERROR(0.1) 0
            M 0
            DETECTOR rec[-1]
        ",
        )
        .unwrap()
        .search_for_undetectable_logical_errors(4, 4, true, false)
        .unwrap_err();
        assert!(no_observable.message().contains("NO OBSERVABLES"));

        let no_errors = Circuit::from_str(
            r"
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        ",
        )
        .unwrap()
        .search_for_undetectable_logical_errors(4, 4, true, false)
        .unwrap_err();
        assert!(no_errors.message().contains("NO ERRORS"));
    }

    #[test]
    fn residual_small_helper_paths_are_covered() {
        let circuit: Circuit = "M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
            .parse()
            .unwrap();
        assert_eq!(format!("{:?}", Circuit::new()), "stim::Circuit()");
        let encoded = circuit
            .encode_diagram_filters(&[
                DetectingRegionFilter::AllDetectors,
                DetectingRegionFilter::AllObservables,
            ])
            .unwrap();
        assert_eq!(encoded, vec!["D0".to_string(), "L0".to_string()]);
        assert!(circuit.likeliest_error_sat_problem().unwrap().contains('p'));
        assert_eq!(
            circuit
                .detecting_regions_with_filters(&[], None, false)
                .unwrap(),
            circuit
                .detecting_regions_with_options(None, None, false)
                .unwrap()
        );
        assert!(circuit.slice(Some(10), Some(11), 1).unwrap().is_empty());
        let zero_step = circuit.slice(None, None, 0).unwrap_err();
        assert!(zero_step.message().contains("slice step cannot be zero"));
        assert_eq!(
            circuit.slice(Some(0), Some(1), 9).unwrap().to_string(),
            "M 0"
        );
    }
}
