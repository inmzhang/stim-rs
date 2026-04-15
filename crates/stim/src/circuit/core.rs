use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::path::Path;
use std::str::FromStr;

use super::{
    CircuitInsertOperation, CircuitInstruction, CircuitItem, DetectingRegionFilter,
    support::{
        convert_explained_error, parse_circuit_item, parse_detecting_regions_text,
        split_top_level_circuit_items,
    },
};
use crate::common::bit_packing::unpack_bits;
use crate::common::parse::{decode_measurement_solution, parse_detector_coordinate_map};
use crate::common::slicing::{compute_slice_indices, normalize_index};
use crate::{
    DemTarget, DetectorErrorModel, DetectorSampler, ExplainedError, Flow, GateTarget,
    MeasurementSampler, MeasurementsToDetectionEventsConverter, PauliString, Result, StimError,
    Tableau,
};

/// A Stim circuit.
pub struct Circuit {
    pub(crate) inner: stim_cxx::Circuit,
}

impl Circuit {
    /// Creates an empty circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit = stim::Circuit::new();
    /// assert!(circuit.is_empty());
    /// assert_eq!(circuit.len(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: stim_cxx::Circuit::new(),
        }
    }

    /// Returns the number of qubits referenced by the circuit.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Derives the detector error model of the circuit using default options.
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
            .map(|inner| DetectorErrorModel { inner })
            .map_err(StimError::from)
    }

    /// Returns a circuit containing the detectors that are missing from the original.
    #[must_use]
    pub fn missing_detectors(&self, unknown_input: bool) -> Self {
        Self {
            inner: self.inner.missing_detectors(unknown_input),
        }
    }

    /// Converts the circuit into a tableau when the requested instruction classes are allowed.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0\nS 0".parse().unwrap();
    /// let tableau = circuit.to_tableau(false, false, false).unwrap();
    /// assert_eq!(tableau, stim::Tableau::from_circuit(&circuit, false, false, false).unwrap());
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

    /// Returns whether the circuit has the given flow.
    pub fn has_flow(&self, flow: &Flow, unsigned: bool) -> Result<bool> {
        self.inner
            .has_flow(&flow.to_string(), unsigned)
            .map_err(StimError::from)
    }

    /// Returns whether the circuit has all of the given flows.
    pub fn has_all_flows(&self, flows: &[Flow], unsigned: bool) -> Result<bool> {
        self.inner
            .has_all_flows(flows.iter().map(ToString::to_string).collect(), unsigned)
            .map_err(StimError::from)
    }

    /// Returns the circuit's generated flows.
    pub fn flow_generators(&self) -> Result<Vec<Flow>> {
        self.inner
            .flow_generators()
            .into_iter()
            .map(|text| Flow::from_text(&text))
            .collect()
    }

    /// Solves which measurements satisfy the requested flows.
    pub fn solve_flow_measurements(&self, flows: &[Flow]) -> Result<Vec<Option<Vec<i32>>>> {
        self.inner
            .solve_flow_measurements(flows.iter().map(ToString::to_string).collect())
            .map_err(StimError::from)?
            .into_iter()
            .map(decode_measurement_solution)
            .collect()
    }

    /// Returns the flow-reversed circuit and reversed flows.
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
            .map(|text| Flow::from_text(&text))
            .collect::<Result<Vec<_>>>()?;
        Ok((circuit, flows))
    }

    /// Returns the number of top-level operations in the circuit.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the circuit has no top-level operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of measurement results produced by the circuit.
    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        self.inner.num_measurements()
    }

    /// Counts determined measurements in the circuit.
    #[must_use]
    pub fn count_determined_measurements(&self, unknown_input: bool) -> u64 {
        self.inner.count_determined_measurements(unknown_input)
    }

    /// Returns the number of detectors in the circuit.
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        self.inner.num_detectors()
    }

    /// Returns the number of observables in the circuit.
    #[must_use]
    pub fn num_observables(&self) -> u64 {
        self.inner.num_observables()
    }

    /// Returns the number of ticks in the circuit.
    #[must_use]
    pub fn num_ticks(&self) -> u64 {
        self.inner.num_ticks()
    }

    /// Returns the number of sweep bits referenced by the circuit.
    #[must_use]
    pub fn num_sweep_bits(&self) -> usize {
        self.inner.num_sweep_bits()
    }

    /// Clears all operations from the circuit.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Appends Stim program text onto the circuit.
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
            .map_err(StimError::from)
    }

    /// Appends a gate applied to qubit targets.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit = stim::Circuit::new();
    /// circuit.append("X", &[0, 2], &[]).unwrap();
    /// assert_eq!(circuit.to_string(), "X 0 2");
    /// ```
    pub fn append(&mut self, gate_name: &str, targets: &[u32], args: &[f64]) -> Result<()> {
        self.inner
            .append(gate_name, targets, args)
            .map_err(StimError::from)
    }

    /// Appends a gate using rich [`GateTarget`] values.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit = stim::Circuit::new();
    /// circuit
    ///     .append_gate_targets(
    ///         "CORRELATED_ERROR",
    ///         &[
    ///             stim::target_x(0_u32, false).unwrap(),
    ///             stim::target_y(1_u32, false).unwrap(),
    ///             stim::target_z(2_u32, false).unwrap(),
    ///         ],
    ///         &[0.125],
    ///     )
    ///     .unwrap();
    /// assert_eq!(circuit.to_string(), "E(0.125) X0 Y1 Z2");
    /// ```
    pub fn append_gate_targets(
        &mut self,
        gate_name: &str,
        targets: &[GateTarget],
        args: &[f64],
    ) -> Result<()> {
        let raw_targets: Vec<u32> = targets.iter().map(|target| target.raw_data()).collect();
        self.inner
            .append(gate_name, &raw_targets, args)
            .map_err(StimError::from)
    }

    /// Compares two circuits using an absolute tolerance on numeric arguments.
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

    /// Returns the circuit with noise processes removed.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "X_ERROR(0.25) 0\nCX 0 1\nM(0.125) 0".parse().unwrap();
    /// assert_eq!(circuit.without_noise().to_string(), "CX 0 1\nM 0");
    /// ```
    #[must_use]
    pub fn without_noise(&self) -> Self {
        Self {
            inner: self.inner.without_noise(),
        }
    }

    /// Returns the circuit with feedback inlined.
    #[must_use]
    pub fn with_inlined_feedback(&self) -> Self {
        Self {
            inner: self.inner.with_inlined_feedback(),
        }
    }

    /// Returns the circuit with instruction tags removed.
    #[must_use]
    pub fn without_tags(&self) -> Self {
        Self {
            inner: self.inner.without_tags(),
        }
    }

    /// Returns the circuit with repeat blocks inlined.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "REPEAT 2 {\n    H 0\n    TICK\n}".parse().unwrap();
    /// assert_eq!(circuit.flattened().to_string(), "H 0\nTICK\nH 0\nTICK");
    /// ```
    #[must_use]
    pub fn flattened(&self) -> Self {
        Self {
            inner: self.inner.flattened(),
        }
    }

    /// Returns the flattened operations as top-level instructions.
    pub fn flattened_operations(&self) -> Result<Vec<CircuitInstruction>> {
        self.inner
            .flattened_operation_texts()
            .into_iter()
            .map(|text| CircuitInstruction::from_stim_program_text(&text))
            .collect()
    }

    /// Returns the circuit decomposed into simpler operations.
    #[must_use]
    pub fn decomposed(&self) -> Self {
        Self {
            inner: self.inner.decomposed(),
        }
    }

    /// Returns the inverse circuit when one exists.
    pub fn inverse(&self) -> Result<Self> {
        self.inner
            .inverse()
            .map(|inner| Self { inner })
            .map_err(StimError::from)
    }

    /// Converts the circuit into OpenQASM text.
    pub fn to_qasm(&self, open_qasm_version: i32, skip_dets_and_obs: bool) -> Result<String> {
        self.inner
            .to_qasm(open_qasm_version, skip_dets_and_obs)
            .map_err(StimError::from)
    }

    /// Converts the circuit into a Quirk URL.
    pub fn to_quirk_url(&self) -> Result<String> {
        self.inner.to_quirk_url().map_err(StimError::from)
    }

    /// Converts the circuit into a Crumble URL.
    pub fn to_crumble_url(&self, skip_detectors: bool) -> Result<String> {
        self.inner
            .to_crumble_url(skip_detectors)
            .map_err(StimError::from)
    }

    /// Renders a circuit diagram of the requested type.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0\nCNOT 0 1".parse().unwrap();
    /// let timeline = circuit.diagram("timeline-text").unwrap();
    /// assert!(timeline.contains("q0:"));
    /// assert!(timeline.contains("q1:"));
    /// ```
    pub fn diagram(&self, type_name: &str) -> Result<String> {
        self.inner.diagram(type_name).map_err(StimError::from)
    }

    /// Renders a diagram focused on a single tick.
    pub fn diagram_with_tick(&self, type_name: &str, tick: u64) -> Result<String> {
        self.inner
            .diagram_with_options(type_name, Some((tick, 1)), None)
            .map_err(StimError::from)
    }

    /// Renders a diagram over a tick range.
    pub fn diagram_with_tick_range(
        &self,
        type_name: &str,
        tick_start: u64,
        tick_count: u64,
        rows: Option<usize>,
    ) -> Result<String> {
        self.inner
            .diagram_with_options(type_name, Some((tick_start, tick_count)), rows)
            .map_err(StimError::from)
    }

    /// Renders a diagram filtered to selected detecting regions.
    pub fn diagram_with_filters(
        &self,
        type_name: &str,
        tick_range: Option<(u64, u64)>,
        rows: Option<usize>,
        filters: &[DetectingRegionFilter],
    ) -> Result<String> {
        self.inner
            .diagram_with_options_and_filters(
                type_name,
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
        self.shortest_error_sat_problem_with_format("WDIMACS")
    }

    /// Returns a SAT problem encoding in the requested format.
    pub fn shortest_error_sat_problem_with_format(&self, format_name: &str) -> Result<String> {
        self.inner
            .shortest_error_sat_problem(format_name)
            .map_err(StimError::from)
    }

    /// Returns a SAT problem encoding the likeliest logical error search.
    pub fn likeliest_error_sat_problem(&self) -> Result<String> {
        self.likeliest_error_sat_problem_with_options(100, "WDIMACS")
    }

    /// Returns a likeliest-error SAT encoding with explicit options.
    pub fn likeliest_error_sat_problem_with_options(
        &self,
        quantization: i32,
        format_name: &str,
    ) -> Result<String> {
        self.inner
            .likeliest_error_sat_problem(quantization, format_name)
            .map_err(StimError::from)
    }

    /// Returns detecting regions for all detectors and observables.
    pub fn detecting_regions(&self) -> Result<BTreeMap<DemTarget, BTreeMap<u64, PauliString>>> {
        let text = self
            .inner
            .detecting_regions_text()
            .map_err(StimError::from)?;
        parse_detecting_regions_text(&text)
    }

    /// Returns detecting regions with explicit target and tick filters.
    pub fn detecting_regions_with_options(
        &self,
        targets: Option<&[DemTarget]>,
        ticks: Option<&[u64]>,
        ignore_anticommutation_errors: bool,
    ) -> Result<BTreeMap<DemTarget, BTreeMap<u64, PauliString>>> {
        let text = self
            .inner
            .detecting_regions_text_with_options(
                targets
                    .map(|targets| targets.iter().map(ToString::to_string).collect())
                    .unwrap_or_default(),
                ticks
                    .map(std::borrow::ToOwned::to_owned)
                    .unwrap_or_default(),
                ignore_anticommutation_errors,
            )
            .map_err(StimError::from)?;
        parse_detecting_regions_text(&text)
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
        let serialized = self
            .inner
            .get_detector_coordinates_text(&included)
            .map_err(StimError::from)?;
        parse_detector_coordinate_map(&serialized)
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
        let serialized = self
            .inner
            .get_final_qubit_coordinates_text()
            .map_err(StimError::from)?;
        parse_detector_coordinate_map(&serialized)
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
        let items = self.top_level_item_texts()?;
        let normalized = normalize_index(index, items.len())
            .ok_or_else(|| StimError::new(format!("index {index} out of range")))?;
        parse_circuit_item(&items[normalized])
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
        let items = self.top_level_item_texts()?;
        let len = items.len() as isize;
        let indices = compute_slice_indices(len, start, stop, step);
        if indices.is_empty() {
            return Ok(Self::new());
        }
        let text = indices
            .into_iter()
            .map(|index| items[index as usize].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        Self::from_str(&text)
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
        let mut items = self.top_level_item_texts()?;
        let normalized = normalize_index(index, items.len())
            .ok_or_else(|| StimError::new(format!("index {index} out of range")))?;
        let popped = parse_circuit_item(&items.remove(normalized))?;
        *self = if items.is_empty() {
            Self::new()
        } else {
            Self::from_str(&items.join("\n"))?
        };
        Ok(popped)
    }

    /// Inserts an operation before the given top-level index.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit: stim::Circuit = "H 0\nM 0".parse().unwrap();
    /// circuit.insert(1, stim::CircuitInstruction::from_stim_program_text("X 1").unwrap()).unwrap();
    /// assert_eq!(circuit.to_string(), "H 0\nX 1\nM 0");
    /// ```
    pub fn insert(
        &mut self,
        index: isize,
        operation: impl Into<CircuitInsertOperation>,
    ) -> Result<()> {
        let mut items = self.top_level_item_texts()?;
        let len = items.len() as isize;
        let normalized = if index < 0 { len + index } else { index };
        if !(0..len).contains(&normalized) {
            return Err(StimError::new(format!("index {index} out of range")));
        }

        let new_items = match operation.into() {
            CircuitInsertOperation::Instruction(instruction) => vec![instruction.to_string()],
            CircuitInsertOperation::Circuit(circuit) => circuit.top_level_item_texts()?,
            CircuitInsertOperation::RepeatBlock(repeat_block) => vec![repeat_block.to_string()],
        };
        items.splice(normalized as usize..normalized as usize, new_items);
        *self = Self::from_str(&items.join("\n"))?;
        Ok(())
    }

    /// Appends an owned operation of any supported type.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit = stim::Circuit::new();
    /// circuit
    ///     .append_operation(stim::CircuitInstruction::from_stim_program_text("H 0").unwrap())
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
                    self.append_from_stim_program_text(&circuit.to_string())
                }
            }
            CircuitInsertOperation::RepeatBlock(repeat_block) => {
                self.append_from_stim_program_text(&repeat_block.to_string())
            }
        }
    }

    fn top_level_item_texts(&self) -> Result<Vec<String>> {
        split_top_level_circuit_items(&self.to_string())
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

    /// Returns an owned copy of the circuit.
    #[must_use]
    pub fn copy(&self) -> Self {
        self.clone()
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
        .map(|inner| Self { inner })
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

impl Clone for Circuit {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
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
        Self {
            inner: self.inner.add(&rhs.inner),
        }
    }
}

impl AddAssign for Circuit {
    fn add_assign(&mut self, rhs: Self) {
        self.inner.add_assign(&rhs.inner);
    }
}

impl Mul<u64> for Circuit {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self::Output {
        Self {
            inner: self.inner.repeat(rhs),
        }
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
            .map(|inner| Self { inner })
            .map_err(StimError::from)
    }
}

#[cfg(test)]
mod api_tests {
    use std::str::FromStr;
    use std::{fs, path::PathBuf, time::SystemTime};

    use ndarray::Array2;

    use crate::{Circuit, all_gate_data, gate_data};

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

        let copy = original.copy();

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
            .append("X", &[0], &[])
            .expect("append should succeed");
        circuit
            .append("M", &[0, 1], &[])
            .expect("append should succeed");

        assert_eq!(circuit.to_string(), "X 0\nM 0 1");
    }

    #[test]
    fn append_supports_gate_arguments() {
        let mut circuit = Circuit::new();

        circuit
            .append("X_ERROR", &[0], &[0.125])
            .expect("append should succeed");

        assert_eq!(circuit.to_string(), "X_ERROR(0.125) 0");
    }

    #[test]
    fn append_reports_stim_parse_errors() {
        let mut circuit = Circuit::new();
        let error = circuit
            .append("NOT_A_GATE", &[0], &[])
            .expect_err("invalid gate should fail");

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
            .to_qasm(3, false)
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
        let gate = gate_data("cnot").expect("gate should exist");

        assert_eq!(gate.name(), "CX");
        assert!(gate.aliases().contains(&"CNOT".to_string()));
        assert!(gate.aliases().contains(&"CX".to_string()));
        assert!(gate.is_two_qubit_gate());
        assert!(gate.is_unitary());
        assert!(!gate.is_reset());
    }

    #[test]
    fn gate_data_exposes_more_metadata_flags() {
        let h = gate_data("H").expect("gate should exist");
        let x_error = gate_data("X_ERROR").expect("gate should exist");
        let m = gate_data("M").expect("gate should exist");
        let r = gate_data("R").expect("gate should exist");
        let detector = gate_data("DETECTOR").expect("gate should exist");

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
        let h = gate_data("H").expect("gate should exist");
        let s = gate_data("S").expect("gate should exist");
        let x_error = gate_data("X_ERROR").expect("gate should exist");
        let r = gate_data("R").expect("gate should exist");

        assert_eq!(h.inverse().expect("H has inverse").name(), "H");
        assert_eq!(s.inverse().expect("S has inverse").name(), "S_DAG");
        assert!(x_error.inverse().is_none());

        assert_eq!(x_error.generalized_inverse().name(), "X_ERROR");
        assert_eq!(r.generalized_inverse().name(), "M");
    }

    #[test]
    fn gate_data_exposes_hadamard_conjugation() {
        let x = gate_data("X").expect("gate should exist");
        let cx = gate_data("CX").expect("gate should exist");
        let ry = gate_data("RY").expect("gate should exist");

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
        let cx = gate_data("CX").expect("gate should exist");
        let cz = gate_data("CZ").expect("gate should exist");
        let h = gate_data("H").expect("gate should exist");
        let detector = gate_data("DETECTOR").expect("gate should exist");

        assert!(!cx.is_symmetric_gate());
        assert!(cz.is_symmetric_gate());
        assert!(h.is_symmetric_gate());
        assert!(!detector.is_symmetric_gate());
    }

    #[test]
    fn gate_data_supports_identity_traits() {
        let canonical = gate_data("CX").expect("gate should exist");
        let alias = gate_data("cnot").expect("alias should resolve");
        let cloned = alias.clone();
        let other = gate_data("H").expect("other gate should exist");

        assert_eq!(canonical, alias);
        assert_eq!(cloned, canonical);
        assert_ne!(canonical, other);
    }

    #[test]
    fn gate_data_display_and_debug_use_canonical_names() {
        let gate = gate_data("mpp").expect("gate should exist");
        let alias = gate_data("cnot").expect("alias should resolve");

        assert_eq!(gate.to_string(), "MPP");
        assert_eq!(format!("{gate:?}"), "stim::gate_data(\"MPP\")");
        assert_eq!(alias.to_string(), "CX");
        assert_eq!(format!("{alias:?}"), "stim::gate_data(\"CX\")");
    }

    #[test]
    fn gate_data_lookup_reports_unknown_gates() {
        let error = gate_data("definitely_not_a_gate").expect_err("unknown gate should fail");

        assert!(error.message().contains("definitely_not_a_gate"));
    }

    #[test]
    fn all_gate_data_returns_canonical_gate_inventory() {
        let inventory = all_gate_data();

        assert!(inventory.contains_key("CX"));
        assert!(inventory.contains_key("DETECTOR"));
        assert!(inventory.contains_key("H"));
        assert!(inventory.contains_key("MPP"));
        assert!(!inventory.contains_key("CNOT"));
        assert!(!inventory.contains_key("NOT_A_GATE"));

        let cx = inventory.get("CX").expect("CX should be present");
        assert_eq!(cx, &gate_data("cnot").expect("lookup should resolve alias"));
        assert_eq!(cx.name(), "CX");
        assert!(cx.aliases().contains(&"CNOT".to_string()));

        for (name, gate) in &inventory {
            assert_eq!(gate.name(), *name);
            assert_eq!(
                gate,
                &gate_data(name).expect("inventory key should roundtrip")
            );
            assert_eq!(gate.to_string(), *name);
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
        CircuitInstruction, CircuitItem, CircuitRepeatBlock, DetectorErrorModel, Flow, GateTarget,
        PauliString, target_logical_observable_id, target_relative_detector_id,
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
    fn circuit_flattened_operations_match_documented_examples() {
        let actual = Circuit::from_str(
            r"
            H 0
            REPEAT 3 {
                X_ERROR(0.125) 1
            }
            CORRELATED_ERROR(0.25) X3 Y4 Z5
            M 0 !1
            DETECTOR rec[-1]
        ",
        )
        .unwrap()
        .flattened_operations()
        .unwrap()
        .into_iter()
        .map(|instruction| instruction.to_string())
        .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                "H 0",
                "X_ERROR(0.125) 1",
                "X_ERROR(0.125) 1",
                "X_ERROR(0.125) 1",
                "E(0.25) X3 Y4 Z5",
                "M 0 !1",
                "DETECTOR rec[-1]",
            ]
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

    #[test]
    fn circuit_append_gate_targets_supports_qubit_targets() {
        let mut circuit = Circuit::new();

        circuit
            .append_gate_targets(
                "CX",
                &[
                    GateTarget::new(0_u32),
                    GateTarget::new(1_u32),
                    GateTarget::new(2_u32),
                    GateTarget::new(3_u32),
                ],
                &[],
            )
            .expect("qubit targets should append");
        circuit
            .append_gate_targets("H", &[GateTarget::new(4_u32), GateTarget::new(5_u32)], &[])
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
                "CORRELATED_ERROR",
                &[
                    crate::target_x(0_u32, false).expect("X target should construct"),
                    crate::target_y(1_u32, false).expect("Y target should construct"),
                    crate::target_pauli(2, 'Z', false).expect("Z target should construct"),
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
            .append_gate_targets("M", &[GateTarget::new(0_u32), GateTarget::new(1_u32)], &[])
            .expect("measurements should append");
        circuit
            .append_gate_targets(
                "CX",
                &[
                    crate::target_rec(-1).expect("record target should construct"),
                    GateTarget::new(5_u32),
                ],
                &[],
            )
            .expect("record-controlled CX should append");
        circuit
            .append_gate_targets(
                "DETECTOR",
                &[
                    crate::target_rec(-1).expect("record target should construct"),
                    crate::target_rec(-2).expect("record target should construct"),
                ],
                &[],
            )
            .expect("detector targets should append");
        circuit
            .append_gate_targets(
                "OBSERVABLE_INCLUDE",
                &[
                    crate::target_rec(-1).expect("record target should construct"),
                    crate::target_rec(-2).expect("record target should construct"),
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
                crate::target_x(1_u32, false).expect("X target should construct"),
                crate::target_y(2_u32, false).expect("Y target should construct"),
                crate::target_z(3_u32, false).expect("Z target should construct"),
            ],
            false,
        )
        .expect("combined pauli product should construct");
        targets.push(crate::target_y(4_u32, false).expect("single Y target should construct"));
        targets.push(crate::target_z(5_u32, false).expect("single Z target should construct"));
        targets.extend(
            crate::target_combined_paulis(
                &[
                    crate::target_x(1_u32, false).expect("X target should construct"),
                    crate::target_x(2_u32, false).expect("X target should construct"),
                ],
                true,
            )
            .expect("inverted combined pauli product should construct"),
        );

        circuit
            .append_gate_targets("MPP", &targets, &[])
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
            CircuitItem::instruction(
                CircuitInstruction::new("X_ERROR", [2u32], [0.5], "").unwrap()
            )
        );
        assert_eq!(
            circuit.get(2).unwrap(),
            CircuitItem::repeat_block(
                CircuitRepeatBlock::new(100, &Circuit::from_str("X 0\nY 1 2").unwrap(), "")
                    .unwrap()
            )
        );
        assert_eq!(
            circuit.get(-1).unwrap(),
            CircuitItem::instruction(
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
            CircuitItem::instruction(CircuitInstruction::new("Y", [3u32], [], "").unwrap())
        );
        assert_eq!(
            circuit.pop(1).unwrap(),
            CircuitItem::instruction(CircuitInstruction::new("S", [1u32], [], "").unwrap())
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
            CircuitItem::instruction(CircuitInstruction::new("M", [0u32], [], "").unwrap())
        );
        assert_eq!(circuit, Circuit::from_str("X 0\nDETECTOR rec[-1]").unwrap());
        assert_eq!(sliced, Circuit::new());
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
            CircuitItem::repeat_block(
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
            vec![
                Flow::from_text("X -> Z").unwrap(),
                Flow::from_text("Z -> X").unwrap(),
            ]
        );

        assert_eq!(
            Circuit::from_str("M 0").unwrap().flow_generators().unwrap(),
            vec![
                Flow::from_text("1 -> Z xor rec[0]").unwrap(),
                Flow::from_text("Z -> rec[0]").unwrap(),
            ]
        );

        assert_eq!(
            Circuit::from_str("RX 0")
                .unwrap()
                .flow_generators()
                .unwrap(),
            vec![Flow::from_text("1 -> X").unwrap()]
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
                    Flow::from_text("X -> Z").unwrap(),
                    Flow::from_text("Y -> Y").unwrap(),
                    Flow::from_text("Z -> X").unwrap(),
                ],
                false,
            )
            .unwrap()
        );

        assert!(
            h.has_all_flows(
                &[
                    Flow::from_text("X -> Z").unwrap(),
                    Flow::from_text("Y -> -Y").unwrap(),
                    Flow::from_text("Z -> X").unwrap(),
                ],
                false,
            )
            .unwrap()
        );

        assert!(
            h.has_all_flows(
                &[
                    Flow::from_text("X -> Z").unwrap(),
                    Flow::from_text("Y -> Y").unwrap(),
                    Flow::from_text("Z -> X").unwrap(),
                ],
                true,
            )
            .unwrap()
        );
    }

    #[test]
    fn circuit_has_flow_matches_documented_examples() {
        let m = Circuit::from_str("M 0").unwrap();
        assert!(
            m.has_flow(&Flow::from_text("Z -> Z").unwrap(), false)
                .unwrap()
        );
        assert!(
            !m.has_flow(&Flow::from_text("X -> X").unwrap(), false)
                .unwrap()
        );
        assert!(
            !m.has_flow(&Flow::from_text("Z -> I").unwrap(), false)
                .unwrap()
        );
        assert!(
            m.has_flow(&Flow::from_text("Z -> I xor rec[-1]").unwrap(), false)
                .unwrap()
        );
        assert!(
            m.has_flow(&Flow::from_text("Z -> rec[-1]").unwrap(), false)
                .unwrap()
        );

        let cx58 = Circuit::from_str("CX 5 8").unwrap();
        assert!(
            cx58.has_flow(&Flow::from_text("X5 -> X5*X8").unwrap(), false)
                .unwrap()
        );
        assert!(
            !cx58
                .has_flow(&Flow::from_text("X_ -> XX").unwrap(), false)
                .unwrap()
        );
        assert!(
            cx58.has_flow(&Flow::from_text("_____X___ -> _____X__X").unwrap(), false)
                .unwrap()
        );

        let ry = Circuit::from_str("RY 0").unwrap();
        assert!(
            ry.has_flow(&Flow::from_text("1 -> Y").unwrap(), false)
                .unwrap()
        );
        assert!(
            !ry.has_flow(&Flow::from_text("1 -> X").unwrap(), false)
                .unwrap()
        );

        let cx01 = Circuit::from_str("CX 0 1").unwrap();
        let flow = Flow::from_text("+X_ -> +XX").unwrap();
        assert!(cx01.has_flow(&flow, false).unwrap());

        let h0 = Circuit::from_str("H 0").unwrap();
        let y_flow = Flow::from_text("Y -> Y").unwrap();
        assert!(h0.has_flow(&y_flow, true).unwrap());
        assert!(!h0.has_flow(&y_flow, false).unwrap());
    }

    #[test]
    fn circuit_solve_flow_measurements_matches_documented_examples() {
        assert_eq!(
            Circuit::from_str("M 2")
                .unwrap()
                .solve_flow_measurements(&[Flow::from_text("Z2 -> 1").unwrap()])
                .unwrap(),
            vec![Some(vec![0])]
        );

        assert_eq!(
            Circuit::from_str("M 2")
                .unwrap()
                .solve_flow_measurements(&[Flow::from_text("X2 -> X2").unwrap()])
                .unwrap(),
            vec![None]
        );

        assert_eq!(
            Circuit::from_str("MXX 0 1")
                .unwrap()
                .solve_flow_measurements(&[Flow::from_text("YY -> ZZ").unwrap()])
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
                Flow::from_text("1 -> Z0*Z4").unwrap(),
                Flow::from_text("Z0 -> Z2").unwrap(),
                Flow::from_text("X0*X2*X4 -> X0*X2*X4").unwrap(),
                Flow::from_text("Y0 -> Y0").unwrap(),
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
            .time_reversed_for_flows(&[Flow::from_text("Z -> rec[-1]").unwrap()], false)
            .unwrap();
        assert_eq!(inv_circuit, Circuit::from_str("R 0").unwrap());
        assert_eq!(inv_flows, vec![Flow::from_text("1 -> Z").unwrap()]);
        assert!(inv_circuit.has_all_flows(&inv_flows, true).unwrap());

        let (inv_circuit, inv_flows) = Circuit::from_str("R 0")
            .unwrap()
            .time_reversed_for_flows(&[Flow::from_text("1 -> Z").unwrap()], false)
            .unwrap();
        assert_eq!(inv_circuit, Circuit::from_str("M 0").unwrap());
        assert_eq!(inv_flows, vec![Flow::from_text("Z -> rec[-1]").unwrap()]);

        let (inv_circuit, inv_flows) = Circuit::from_str("M 0")
            .unwrap()
            .time_reversed_for_flows(&[Flow::from_text("Z -> rec[-1]").unwrap()], true)
            .unwrap();
        assert_eq!(inv_circuit, Circuit::from_str("M 0").unwrap());
        assert_eq!(
            inv_flows,
            vec![Flow::from_text("1 -> Z xor rec[-1]").unwrap()]
        );
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
        expected.insert(target_relative_detector_id(0).unwrap(), ticks);

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
                Some(&[target_relative_detector_id(0).unwrap()]),
                Some(&[1]),
                false,
            )
            .unwrap();

        let mut ticks = BTreeMap::new();
        ticks.insert(1, PauliString::from_text("+X").unwrap());
        let mut expected = BTreeMap::new();
        expected.insert(target_relative_detector_id(0).unwrap(), ticks);
        assert_eq!(filtered, expected);

        let observable_only = circuit
            .detecting_regions_with_options(
                Some(&[target_logical_observable_id(0).unwrap()]),
                None,
                false,
            )
            .unwrap();
        assert_eq!(observable_only.len(), 1);
        assert!(observable_only.contains_key(&target_logical_observable_id(0).unwrap()));
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
                    target_relative_detector_id(0).unwrap(),
                )],
                Some(&[1]),
                false,
            )
            .unwrap();
        assert_eq!(by_target.len(), 1);
        assert!(by_target.contains_key(&target_relative_detector_id(0).unwrap()));

        let by_coords = circuit
            .detecting_regions_with_filters(
                &[DetectingRegionFilter::DetectorCoordinatePrefix(vec![2.0])],
                None,
                false,
            )
            .unwrap();
        assert_eq!(by_coords.len(), 1);
        assert!(by_coords.contains_key(&target_relative_detector_id(0).unwrap()));
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

        let timeline = circuit.diagram("timeline-text").unwrap();
        assert!(timeline.contains("q0:"));
        assert!(timeline.contains("q1:"));

        let svg = circuit.diagram("timeline-svg").unwrap();
        let svg_html = circuit.diagram("timeline-svg-html").unwrap();
        let gltf = circuit.diagram("timeline-3d").unwrap();
        let gltf_html = circuit.diagram("timeline-3d-html").unwrap();
        let interactive = circuit.diagram("interactive").unwrap();
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
        let match_svg = detector_circuit.diagram("matchgraph-svg").unwrap();
        let match_svg_alias = detector_circuit.diagram("match-graph-svg").unwrap();
        let match_svg_html = detector_circuit.diagram("matchgraph-svg-html").unwrap();
        let match_gltf = detector_circuit.diagram("matchgraph-3d").unwrap();
        let match_gltf_html = detector_circuit.diagram("matchgraph-3d-html").unwrap();
        assert!(match_svg.contains("<svg"));
        assert_eq!(match_svg, match_svg_alias);
        assert!(match_svg_html.contains("iframe"));
        assert!(match_gltf.contains("\"nodes\"") || match_gltf.contains("\"scenes\""));
        assert!(match_gltf_html.contains("<html") || match_gltf_html.contains("iframe"));
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
                .diagram_with_tick("detslice-text", 1)
                .unwrap()
                .trim(),
            "q0: -Z:D0-\n     |\nq1: -Z:D0-"
        );

        let detslice_svg = circuit.diagram_with_tick("detslice-svg", 1).unwrap();
        let timeslice_svg = circuit
            .diagram_with_tick_range("timeslice-svg", 1, 1, Some(1))
            .unwrap();
        let with_ops_svg = circuit
            .diagram_with_tick_range("detslice-with-ops-svg", 1, 1, Some(1))
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
                "detslice-text",
                Some((1, 1)),
                None,
                &[DetectingRegionFilter::Target(
                    target_relative_detector_id(0).unwrap(),
                )],
            )
            .unwrap();
        assert!(detector_text.contains("D0"));

        let observable_svg = circuit
            .diagram_with_filters(
                "detslice-svg",
                Some((1, 1)),
                Some(1),
                &[DetectingRegionFilter::Target(
                    target_logical_observable_id(0).unwrap(),
                )],
            )
            .unwrap();
        assert!(observable_svg.contains("<svg"));

        let coords_svg = circuit
            .diagram_with_filters(
                "detslice-with-ops-svg",
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
                .likeliest_error_sat_problem_with_options(100, "WDIMACS")
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
}
