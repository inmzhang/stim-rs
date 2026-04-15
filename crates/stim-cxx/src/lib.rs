#![allow(clippy::too_many_arguments)]

#[allow(unused_imports)]
use core::pin::Pin;
use cxx::UniquePtr;

pub use ffi::{
    BitTableData, CircuitErrorLocationData, CircuitErrorLocationStackFrameData,
    CircuitTargetsInsideInstructionData, DemSampleBatch, DemTargetWithCoordsData,
    ExplainedErrorData, FlippedMeasurementData, GateTargetWithCoordsData,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildMetadata {
    pub crate_name: &'static str,
    pub cxx_standard: &'static str,
    pub target: &'static str,
    pub vendor_stim_mode: &'static str,
    pub vendor_stim_dir: &'static str,
}

pub fn build_metadata() -> BuildMetadata {
    BuildMetadata {
        crate_name: env!("CARGO_PKG_NAME"),
        cxx_standard: env!("STIM_RS_CXX_STANDARD"),
        target: env!("STIM_RS_TARGET"),
        vendor_stim_mode: env!("STIM_RS_VENDOR_STIM_MODE"),
        vendor_stim_dir: env!("STIM_RS_VENDOR_STIM_DIR"),
    }
}

pub fn pinned_stim_commit() -> &'static str {
    env!("STIM_RS_PINNED_STIM_COMMIT")
}

pub fn read_shot_data_file_bit_packed(
    filepath: &str,
    format_name: &str,
    num_measurements: u64,
    num_detectors: u64,
    num_observables: u64,
) -> Result<Vec<u8>, cxx::Exception> {
    ffi::read_shot_data_file_bit_packed(
        filepath,
        format_name,
        num_measurements,
        num_detectors,
        num_observables,
    )
}

pub fn write_shot_data_file_bit_packed(
    data: &[u8],
    shots: u64,
    filepath: &str,
    format_name: &str,
    num_measurements: u64,
    num_detectors: u64,
    num_observables: u64,
) -> Result<(), cxx::Exception> {
    ffi::write_shot_data_file_bit_packed(
        data,
        shots,
        filepath,
        format_name,
        num_measurements,
        num_detectors,
        num_observables,
    )
}

pub fn all_gate_names() -> Vec<String> {
    ffi::all_gate_names()
}

pub fn main(command_line_args: Vec<String>) -> i32 {
    ffi::stim_main(command_line_args)
}

pub fn canonicalize_flow_text(text: &str) -> Result<String, cxx::Exception> {
    ffi::canonicalize_flow_text(text)
}

pub fn multiply_flow_texts(left: &str, right: &str) -> Result<String, cxx::Exception> {
    ffi::multiply_flow_texts(left, right)
}

pub struct Circuit {
    inner: UniquePtr<ffi::CircuitHandle>,
}

pub struct DetectorErrorModel {
    inner: UniquePtr<ffi::DetectorErrorModelHandle>,
}

pub struct Tableau {
    inner: UniquePtr<ffi::TableauHandle>,
}

pub struct TableauSimulator {
    inner: UniquePtr<ffi::TableauSimulatorHandle>,
}

pub struct FlipSimulator {
    inner: UniquePtr<ffi::FrameSimulatorHandle>,
}

pub struct PauliString {
    inner: UniquePtr<ffi::PauliStringHandle>,
}

pub struct CliffordString {
    inner: UniquePtr<ffi::CliffordStringHandle>,
}

pub struct PauliStringIterator {
    inner: UniquePtr<ffi::PauliStringIteratorHandle>,
}

pub struct TableauIterator {
    inner: UniquePtr<ffi::TableauIteratorHandle>,
}

pub struct MeasurementSampler {
    inner: UniquePtr<ffi::MeasurementSamplerHandle>,
}

pub struct DetectorSampler {
    inner: UniquePtr<ffi::DetectorSamplerHandle>,
}

pub struct DemSampler {
    inner: UniquePtr<ffi::DemSamplerHandle>,
}

pub struct MeasurementsToDetectionEventsConverter {
    inner: UniquePtr<ffi::MeasurementsToDetectionEventsConverterHandle>,
}

pub struct GateData {
    inner: UniquePtr<ffi::GateDataHandle>,
}

impl MeasurementSampler {
    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        ffi::measurement_sampler_num_measurements(self.inner())
    }

    #[must_use]
    pub fn sample_bit_packed(&mut self, shots: u64) -> Vec<u8> {
        ffi::measurement_sampler_sample_bit_packed(self.inner.pin_mut(), shots)
    }

    pub fn sample_write(
        &mut self,
        shots: u64,
        filepath: &str,
        format_name: &str,
    ) -> Result<(), cxx::Exception> {
        ffi::measurement_sampler_sample_write(self.inner.pin_mut(), shots, filepath, format_name)
    }

    fn inner(&self) -> &ffi::MeasurementSamplerHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx measurement sampler construction returned null")
    }
}

impl DetectorSampler {
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        ffi::detector_sampler_num_detectors(self.inner())
    }

    #[must_use]
    pub fn num_observables(&self) -> u64 {
        ffi::detector_sampler_num_observables(self.inner())
    }

    #[must_use]
    pub fn sample_bit_packed(&mut self, shots: u64) -> Vec<u8> {
        ffi::detector_sampler_sample_bit_packed(self.inner.pin_mut(), shots)
    }

    #[must_use]
    pub fn sample_observables_bit_packed(&mut self, shots: u64) -> Vec<u8> {
        ffi::detector_sampler_sample_observables_bit_packed(self.inner.pin_mut(), shots)
    }

    pub fn sample_write(
        &mut self,
        shots: u64,
        filepath: &str,
        format_name: &str,
    ) -> Result<(), cxx::Exception> {
        ffi::detector_sampler_sample_write(self.inner.pin_mut(), shots, filepath, format_name)
    }

    pub fn sample_write_separate_observables(
        &mut self,
        shots: u64,
        dets_filepath: &str,
        dets_format_name: &str,
        obs_filepath: &str,
        obs_format_name: &str,
    ) -> Result<(), cxx::Exception> {
        ffi::detector_sampler_sample_write_separate_observables(
            self.inner.pin_mut(),
            shots,
            dets_filepath,
            dets_format_name,
            obs_filepath,
            obs_format_name,
        )
    }

    fn inner(&self) -> &ffi::DetectorSamplerHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx detector sampler construction returned null")
    }
}

impl DemSampler {
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        ffi::dem_sampler_num_detectors(self.inner())
    }

    #[must_use]
    pub fn num_observables(&self) -> u64 {
        ffi::dem_sampler_num_observables(self.inner())
    }

    #[must_use]
    pub fn num_errors(&self) -> u64 {
        ffi::dem_sampler_num_errors(self.inner())
    }

    #[must_use]
    pub fn sample_bit_packed(&mut self, shots: u64) -> ffi::DemSampleBatch {
        ffi::dem_sampler_sample_bit_packed(self.inner.pin_mut(), shots)
    }

    pub fn sample_bit_packed_replay(
        &mut self,
        recorded_errors: &[u8],
        shots: u64,
    ) -> ffi::DemSampleBatch {
        ffi::dem_sampler_sample_bit_packed_replay(self.inner.pin_mut(), recorded_errors, shots)
    }

    pub fn sample_write(
        &mut self,
        shots: u64,
        dets_filepath: &str,
        dets_format_name: &str,
        obs_filepath: &str,
        obs_format_name: &str,
        err_filepath: &str,
        err_format_name: &str,
        write_errors: bool,
    ) -> Result<(), cxx::Exception> {
        ffi::dem_sampler_sample_write(
            self.inner.pin_mut(),
            shots,
            dets_filepath,
            dets_format_name,
            obs_filepath,
            obs_format_name,
            err_filepath,
            err_format_name,
            write_errors,
            "",
            "01",
            false,
        )
    }

    pub fn sample_write_replay(
        &mut self,
        shots: u64,
        dets_filepath: &str,
        dets_format_name: &str,
        obs_filepath: &str,
        obs_format_name: &str,
        err_filepath: &str,
        err_format_name: &str,
        write_errors: bool,
        replay_err_filepath: &str,
        replay_err_format_name: &str,
    ) -> Result<(), cxx::Exception> {
        ffi::dem_sampler_sample_write(
            self.inner.pin_mut(),
            shots,
            dets_filepath,
            dets_format_name,
            obs_filepath,
            obs_format_name,
            err_filepath,
            err_format_name,
            write_errors,
            replay_err_filepath,
            replay_err_format_name,
            true,
        )
    }

    fn inner(&self) -> &ffi::DemSamplerHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx DEM sampler construction returned null")
    }
}

impl MeasurementsToDetectionEventsConverter {
    fn inner(&self) -> &ffi::MeasurementsToDetectionEventsConverterHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx M2D converter construction returned null")
    }

    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        ffi::m2d_converter_num_measurements(self.inner())
    }

    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        ffi::m2d_converter_num_detectors(self.inner())
    }

    #[must_use]
    pub fn num_observables(&self) -> u64 {
        ffi::m2d_converter_num_observables(self.inner())
    }

    #[must_use]
    pub fn num_sweep_bits(&self) -> u64 {
        ffi::m2d_converter_num_sweep_bits(self.inner())
    }

    pub fn convert_measurements_bit_packed(
        &mut self,
        measurements: &[u8],
        shots: u64,
        append_observables: bool,
    ) -> Vec<u8> {
        ffi::m2d_converter_convert_measurements_bit_packed(
            self.inner.pin_mut(),
            measurements,
            shots,
            append_observables,
        )
    }

    pub fn convert_measurements_and_sweep_bits_bit_packed(
        &mut self,
        measurements: &[u8],
        sweep_bits: &[u8],
        shots: u64,
        append_observables: bool,
    ) -> Vec<u8> {
        ffi::m2d_converter_convert_measurements_and_sweep_bits_bit_packed(
            self.inner.pin_mut(),
            measurements,
            sweep_bits,
            shots,
            append_observables,
        )
    }

    pub fn convert_observables_with_sweep_bits_bit_packed(
        &mut self,
        measurements: &[u8],
        sweep_bits: &[u8],
        shots: u64,
    ) -> Vec<u8> {
        ffi::m2d_converter_convert_observables_with_sweep_bits_bit_packed(
            self.inner.pin_mut(),
            measurements,
            sweep_bits,
            shots,
        )
    }

    pub fn convert_observables_bit_packed(&mut self, measurements: &[u8], shots: u64) -> Vec<u8> {
        ffi::m2d_converter_convert_observables_bit_packed(self.inner.pin_mut(), measurements, shots)
    }

    pub fn convert_file(
        &mut self,
        measurements_filepath: &str,
        measurements_format: &str,
        sweep_bits_filepath: &str,
        sweep_bits_format: &str,
        detection_events_filepath: &str,
        detection_events_format: &str,
        append_observables: bool,
        obs_out_filepath: &str,
        obs_out_format: &str,
    ) -> Result<(), cxx::Exception> {
        ffi::m2d_converter_convert_file(
            self.inner.pin_mut(),
            measurements_filepath,
            measurements_format,
            sweep_bits_filepath,
            sweep_bits_format,
            detection_events_filepath,
            detection_events_format,
            append_observables,
            obs_out_filepath,
            obs_out_format,
        )
    }
}

impl GateData {
    #[must_use]
    pub fn name(&self) -> String {
        self.inner().name()
    }

    #[must_use]
    pub fn aliases(&self) -> Vec<String> {
        self.inner().aliases()
    }

    #[must_use]
    pub fn num_parens_arguments_range(&self) -> Vec<u8> {
        self.inner().num_parens_arguments_range()
    }

    #[must_use]
    pub fn is_noisy_gate(&self) -> bool {
        self.inner().is_noisy_gate()
    }

    #[must_use]
    pub fn is_reset(&self) -> bool {
        self.inner().is_reset()
    }

    #[must_use]
    pub fn is_single_qubit_gate(&self) -> bool {
        self.inner().is_single_qubit_gate()
    }

    #[must_use]
    pub fn is_symmetric_gate(&self) -> bool {
        self.inner().is_symmetric_gate()
    }

    #[must_use]
    pub fn is_two_qubit_gate(&self) -> bool {
        self.inner().is_two_qubit_gate()
    }

    #[must_use]
    pub fn is_unitary(&self) -> bool {
        self.inner().is_unitary()
    }

    #[must_use]
    pub fn produces_measurements(&self) -> bool {
        self.inner().produces_measurements()
    }

    #[must_use]
    pub fn takes_measurement_record_targets(&self) -> bool {
        self.inner().takes_measurement_record_targets()
    }

    #[must_use]
    pub fn takes_pauli_targets(&self) -> bool {
        self.inner().takes_pauli_targets()
    }

    pub fn flows(&self) -> Vec<String> {
        self.inner().flows()
    }

    #[must_use]
    pub fn tableau(&self) -> Option<Tableau> {
        let inner = self.inner().tableau();
        if inner.is_null() {
            None
        } else {
            Some(Tableau { inner })
        }
    }

    #[must_use]
    pub fn clone_handle(&self) -> Self {
        Self {
            inner: self.inner().clone_handle(),
        }
    }

    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let inner = self.inner().inverse();
        if inner.is_null() {
            None
        } else {
            Some(Self { inner })
        }
    }

    #[must_use]
    pub fn generalized_inverse(&self) -> Self {
        Self {
            inner: self.inner().generalized_inverse(),
        }
    }

    #[must_use]
    pub fn hadamard_conjugated(&self, unsigned_only: bool) -> Option<Self> {
        let inner = self.inner().hadamard_conjugated(unsigned_only);
        if inner.is_null() {
            None
        } else {
            Some(Self { inner })
        }
    }

    fn inner(&self) -> &ffi::GateDataHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx gate data construction returned null")
    }
}

pub fn gate_data(name: &str) -> Result<GateData, cxx::Exception> {
    ffi::gate_data_by_name(name).map(|inner| GateData { inner })
}

impl Circuit {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: ffi::new_circuit(),
        }
    }

    pub fn from_stim_program_text(text: &str) -> Result<Self, cxx::Exception> {
        ffi::circuit_from_stim_program_text(text).map(|inner| Self { inner })
    }

    pub fn generated(
        code_task: &str,
        distance: usize,
        rounds: usize,
        after_clifford_depolarization: f64,
        before_round_data_depolarization: f64,
        before_measure_flip_probability: f64,
        after_reset_flip_probability: f64,
    ) -> Result<Self, cxx::Exception> {
        ffi::circuit_generated(
            code_task,
            distance,
            rounds,
            after_clifford_depolarization,
            before_round_data_depolarization,
            before_measure_flip_probability,
            after_reset_flip_probability,
        )
        .map(|inner| Self { inner })
    }

    pub fn has_flow(&self, flow: &str, unsigned: bool) -> Result<bool, cxx::Exception> {
        ffi::circuit_has_flow_text(self.inner(), flow, unsigned)
    }

    pub fn has_all_flows(
        &self,
        flows: Vec<String>,
        unsigned: bool,
    ) -> Result<bool, cxx::Exception> {
        ffi::circuit_has_all_flows_text(self.inner(), flows, unsigned)
    }

    pub fn flow_generators(&self) -> Vec<String> {
        ffi::circuit_flow_generators_texts(self.inner())
    }

    pub fn solve_flow_measurements(
        &self,
        flows: Vec<String>,
    ) -> Result<Vec<String>, cxx::Exception> {
        ffi::circuit_solve_flow_measurements_text(self.inner(), flows)
    }

    pub fn time_reversed_for_flows(
        &self,
        flows: Vec<String>,
        dont_turn_measurements_into_resets: bool,
    ) -> Result<(String, Vec<String>), cxx::Exception> {
        let mut out_flow_texts = Vec::new();
        let circuit_text = ffi::circuit_time_reversed_for_flows_text(
            &mut out_flow_texts,
            self.inner(),
            flows,
            dont_turn_measurements_into_resets,
        )?;
        Ok((circuit_text, out_flow_texts))
    }

    pub fn detector_error_model(
        &self,
        decompose_errors: bool,
        flatten_loops: bool,
        allow_gauge_detectors: bool,
        approximate_disjoint_errors: f64,
        ignore_decomposition_failures: bool,
        block_decomposition_from_introducing_remnant_edges: bool,
    ) -> Result<DetectorErrorModel, cxx::Exception> {
        ffi::circuit_detector_error_model(
            self.inner(),
            decompose_errors,
            flatten_loops,
            allow_gauge_detectors,
            approximate_disjoint_errors,
            ignore_decomposition_failures,
            block_decomposition_from_introducing_remnant_edges,
        )
        .map(|inner| DetectorErrorModel { inner })
    }

    #[must_use]
    pub fn missing_detectors(&self, unknown_input: bool) -> Self {
        Self {
            inner: ffi::circuit_missing_detectors(self.inner(), unknown_input),
        }
    }

    pub fn to_tableau(
        &self,
        ignore_noise: bool,
        ignore_measurement: bool,
        ignore_reset: bool,
    ) -> Result<Tableau, cxx::Exception> {
        ffi::circuit_to_tableau(self.inner(), ignore_noise, ignore_measurement, ignore_reset)
            .map(|inner| Tableau { inner })
    }

    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            inner: ffi::circuit_add(self.inner(), other.inner()),
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        ffi::circuit_add_assign(self.inner.pin_mut(), other.inner());
    }

    #[must_use]
    pub fn repeat(&self, repetitions: u64) -> Self {
        Self {
            inner: ffi::circuit_mul(self.inner(), repetitions),
        }
    }

    pub fn repeat_assign(&mut self, repetitions: u64) {
        ffi::circuit_mul_assign(self.inner.pin_mut(), repetitions);
    }

    #[must_use]
    pub fn to_stim_program_text(&self) -> String {
        ffi::circuit_to_stim_program_text(self.inner())
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        ffi::circuit_num_qubits(self.inner())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        ffi::circuit_len(self.inner())
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        ffi::circuit_num_measurements(self.inner())
    }

    #[must_use]
    pub fn count_determined_measurements(&self, unknown_input: bool) -> u64 {
        ffi::circuit_count_determined_measurements(self.inner(), unknown_input)
    }

    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        ffi::circuit_num_detectors(self.inner())
    }

    #[must_use]
    pub fn num_observables(&self) -> u64 {
        ffi::circuit_num_observables(self.inner())
    }

    #[must_use]
    pub fn num_ticks(&self) -> u64 {
        ffi::circuit_num_ticks(self.inner())
    }

    #[must_use]
    pub fn num_sweep_bits(&self) -> usize {
        ffi::circuit_num_sweep_bits(self.inner())
    }

    pub fn clear(&mut self) {
        ffi::circuit_clear(self.inner.pin_mut());
    }

    pub fn append_from_stim_program_text(&mut self, text: &str) -> Result<(), cxx::Exception> {
        ffi::circuit_append_from_stim_program_text(self.inner.pin_mut(), text)
    }

    pub fn append(
        &mut self,
        gate_name: &str,
        targets: &[u32],
        args: &[f64],
    ) -> Result<(), cxx::Exception> {
        ffi::circuit_append_gate(self.inner.pin_mut(), gate_name, targets, args)
    }

    pub fn append_repeat_block(
        &mut self,
        repeat_count: u64,
        body: &Self,
        tag: &str,
    ) -> Result<(), cxx::Exception> {
        ffi::circuit_append_repeat_block(self.inner.pin_mut(), repeat_count, body.inner(), tag)
    }

    #[must_use]
    pub fn approx_equals(&self, other: &Self, atol: f64) -> bool {
        ffi::circuit_approx_equals(self.inner(), other.inner(), atol)
    }

    #[must_use]
    pub fn equals(&self, other: &Self) -> bool {
        ffi::circuit_equals(self.inner(), other.inner())
    }

    #[must_use]
    pub fn without_noise(&self) -> Self {
        Self {
            inner: ffi::circuit_without_noise(self.inner()),
        }
    }

    #[must_use]
    pub fn with_inlined_feedback(&self) -> Self {
        Self {
            inner: ffi::circuit_with_inlined_feedback(self.inner()),
        }
    }

    #[must_use]
    pub fn without_tags(&self) -> Self {
        Self {
            inner: ffi::circuit_without_tags(self.inner()),
        }
    }

    #[must_use]
    pub fn flattened(&self) -> Self {
        Self {
            inner: ffi::circuit_flattened(self.inner()),
        }
    }

    #[must_use]
    pub fn decomposed(&self) -> Self {
        Self {
            inner: ffi::circuit_decomposed(self.inner()),
        }
    }

    pub fn inverse(&self) -> Result<Self, cxx::Exception> {
        ffi::circuit_inverse(self.inner()).map(|inner| Self { inner })
    }

    pub fn to_qasm(
        &self,
        open_qasm_version: i32,
        skip_dets_and_obs: bool,
    ) -> Result<String, cxx::Exception> {
        ffi::circuit_to_qasm(self.inner(), open_qasm_version, skip_dets_and_obs)
    }

    pub fn to_quirk_url(&self) -> Result<String, cxx::Exception> {
        ffi::circuit_to_quirk_url(self.inner())
    }

    pub fn to_crumble_url(&self, skip_detectors: bool) -> Result<String, cxx::Exception> {
        ffi::circuit_to_crumble_url(self.inner(), skip_detectors)
    }

    pub fn shortest_error_sat_problem(&self, format_name: &str) -> Result<String, cxx::Exception> {
        ffi::circuit_shortest_error_sat_problem(self.inner(), format_name)
    }

    pub fn likeliest_error_sat_problem(
        &self,
        quantization: i32,
        format_name: &str,
    ) -> Result<String, cxx::Exception> {
        ffi::circuit_likeliest_error_sat_problem(self.inner(), quantization, format_name)
    }

    pub fn detecting_regions_text(&self) -> Result<String, cxx::Exception> {
        ffi::circuit_detecting_regions_text(self.inner())
    }

    pub fn detecting_regions_text_with_options(
        &self,
        target_texts: Vec<String>,
        ticks: Vec<u64>,
        ignore_anticommutation_errors: bool,
    ) -> Result<String, cxx::Exception> {
        ffi::circuit_detecting_regions_text_with_options(
            self.inner(),
            target_texts,
            ticks,
            ignore_anticommutation_errors,
        )
    }

    pub fn explain_detector_error_model_errors(
        &self,
        dem_filter_text: &str,
        has_dem_filter: bool,
        reduce_to_one_representative_error: bool,
    ) -> Result<Vec<ffi::ExplainedErrorData>, cxx::Exception> {
        ffi::circuit_explain_detector_error_model_errors(
            self.inner(),
            dem_filter_text,
            has_dem_filter,
            reduce_to_one_representative_error,
        )
    }

    pub fn flattened_operation_texts(&self) -> Vec<String> {
        ffi::circuit_flattened_operation_texts(self.inner())
    }

    pub fn shortest_graphlike_error(
        &self,
        ignore_ungraphlike_errors: bool,
        canonicalize_circuit_errors: bool,
    ) -> Result<Vec<ffi::ExplainedErrorData>, cxx::Exception> {
        ffi::circuit_shortest_graphlike_error(
            self.inner(),
            ignore_ungraphlike_errors,
            canonicalize_circuit_errors,
        )
    }

    pub fn search_for_undetectable_logical_errors(
        &self,
        dont_explore_detection_event_sets_with_size_above: u64,
        dont_explore_edges_with_degree_above: u64,
        dont_explore_edges_increasing_symptom_degree: bool,
        canonicalize_circuit_errors: bool,
    ) -> Result<Vec<ffi::ExplainedErrorData>, cxx::Exception> {
        ffi::circuit_search_for_undetectable_logical_errors(
            self.inner(),
            dont_explore_detection_event_sets_with_size_above,
            dont_explore_edges_with_degree_above,
            dont_explore_edges_increasing_symptom_degree,
            canonicalize_circuit_errors,
        )
    }

    pub fn get_detector_coordinates_text(
        &self,
        included_detector_indices: &[u64],
    ) -> Result<String, cxx::Exception> {
        ffi::circuit_get_detector_coordinates_text(self.inner(), included_detector_indices)
    }

    pub fn get_final_qubit_coordinates_text(&self) -> Result<String, cxx::Exception> {
        ffi::circuit_get_final_qubit_coordinates_text(self.inner())
    }

    pub fn diagram(&self, type_name: &str) -> Result<String, cxx::Exception> {
        ffi::circuit_diagram(self.inner(), type_name)
    }

    pub fn diagram_with_options(
        &self,
        type_name: &str,
        tick_range: Option<(u64, u64)>,
        rows: Option<usize>,
    ) -> Result<String, cxx::Exception> {
        ffi::circuit_diagram_with_options(
            self.inner(),
            type_name,
            tick_range.is_some(),
            tick_range.map(|(start, _)| start).unwrap_or(0),
            tick_range.map(|(_, count)| count).unwrap_or(0),
            rows.is_some(),
            rows.unwrap_or(0),
        )
    }

    pub fn diagram_with_options_and_filters(
        &self,
        type_name: &str,
        tick_range: Option<(u64, u64)>,
        rows: Option<usize>,
        filter_coords: Vec<String>,
    ) -> Result<String, cxx::Exception> {
        ffi::circuit_diagram_with_options_and_filters(
            self.inner(),
            type_name,
            tick_range.is_some(),
            tick_range.map(|(start, _)| start).unwrap_or(0),
            tick_range.map(|(_, count)| count).unwrap_or(0),
            rows.is_some(),
            rows.unwrap_or(0),
            filter_coords,
        )
    }

    #[must_use]
    pub fn reference_sample_bit_packed(&self) -> Vec<u8> {
        ffi::circuit_reference_sample_bit_packed(self.inner())
    }

    #[must_use]
    pub fn reference_detector_signs_bit_packed(&self) -> Vec<u8> {
        ffi::circuit_reference_detector_signs_bit_packed(self.inner())
    }

    #[must_use]
    pub fn reference_observable_signs_bit_packed(&self) -> Vec<u8> {
        ffi::circuit_reference_observable_signs_bit_packed(self.inner())
    }

    #[must_use]
    pub fn compile_sampler(&self, skip_reference_sample: bool) -> MeasurementSampler {
        self.compile_sampler_with_seed(skip_reference_sample, 0)
    }

    #[must_use]
    pub fn compile_sampler_with_seed(
        &self,
        skip_reference_sample: bool,
        seed: u64,
    ) -> MeasurementSampler {
        MeasurementSampler {
            inner: ffi::circuit_compile_sampler(self.inner(), skip_reference_sample, seed),
        }
    }

    #[must_use]
    pub fn compile_detector_sampler_with_seed(&self, seed: u64) -> DetectorSampler {
        DetectorSampler {
            inner: ffi::circuit_compile_detector_sampler(self.inner(), seed),
        }
    }

    pub fn compile_m2d_converter(
        &self,
        skip_reference_sample: bool,
    ) -> MeasurementsToDetectionEventsConverter {
        MeasurementsToDetectionEventsConverter {
            inner: ffi::circuit_compile_m2d_converter(self.inner(), skip_reference_sample),
        }
    }

    fn inner(&self) -> &ffi::CircuitHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx circuit construction returned null")
    }
}

impl DetectorErrorModel {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: ffi::new_detector_error_model(),
        }
    }

    pub fn from_dem_text(text: &str) -> Result<Self, cxx::Exception> {
        ffi::detector_error_model_from_dem_text(text).map(|inner| Self { inner })
    }

    pub fn diagram(&self, type_name: &str) -> Result<String, cxx::Exception> {
        ffi::detector_error_model_diagram(self.inner(), type_name)
    }

    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            inner: ffi::detector_error_model_add(self.inner(), other.inner()),
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        ffi::detector_error_model_add_assign(self.inner.pin_mut(), other.inner());
    }

    #[must_use]
    pub fn repeat(&self, repetitions: u64) -> Self {
        Self {
            inner: ffi::detector_error_model_mul(self.inner(), repetitions),
        }
    }

    pub fn repeat_assign(&mut self, repetitions: u64) {
        ffi::detector_error_model_mul_assign(self.inner.pin_mut(), repetitions);
    }

    #[must_use]
    pub fn to_dem_text(&self) -> String {
        ffi::detector_error_model_to_dem_text(self.inner())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        ffi::detector_error_model_len(self.inner())
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        ffi::detector_error_model_num_detectors(self.inner())
    }

    #[must_use]
    pub fn num_errors(&self) -> u64 {
        ffi::detector_error_model_num_errors(self.inner())
    }

    #[must_use]
    pub fn num_observables(&self) -> u64 {
        ffi::detector_error_model_num_observables(self.inner())
    }

    pub fn get_detector_coordinates_text(
        &self,
        included_detector_indices: &[u64],
    ) -> Result<String, cxx::Exception> {
        ffi::detector_error_model_get_detector_coordinates_text(
            self.inner(),
            included_detector_indices,
        )
    }

    pub fn clear(&mut self) {
        ffi::detector_error_model_clear(self.inner.pin_mut());
    }

    #[must_use]
    pub fn approx_equals(&self, other: &Self, atol: f64) -> bool {
        ffi::detector_error_model_approx_equals(self.inner(), other.inner(), atol)
    }

    #[must_use]
    pub fn equals(&self, other: &Self) -> bool {
        ffi::detector_error_model_equals(self.inner(), other.inner())
    }

    #[must_use]
    pub fn without_tags(&self) -> Self {
        Self {
            inner: ffi::detector_error_model_without_tags(self.inner()),
        }
    }

    #[must_use]
    pub fn flattened(&self) -> Self {
        Self {
            inner: ffi::detector_error_model_flattened(self.inner()),
        }
    }

    #[must_use]
    pub fn rounded(&self, digits: u8) -> Self {
        Self {
            inner: ffi::detector_error_model_rounded(self.inner(), digits),
        }
    }

    #[must_use]
    pub fn compile_sampler(&self) -> DemSampler {
        self.compile_sampler_with_seed(0)
    }

    #[must_use]
    pub fn compile_sampler_with_seed(&self, seed: u64) -> DemSampler {
        DemSampler {
            inner: ffi::detector_error_model_compile_sampler(self.inner(), seed),
        }
    }

    pub fn shortest_graphlike_error(
        &self,
        ignore_ungraphlike_errors: bool,
    ) -> Result<Self, cxx::Exception> {
        ffi::detector_error_model_shortest_graphlike_error(self.inner(), ignore_ungraphlike_errors)
            .map(|inner| Self { inner })
    }

    pub fn shortest_error_sat_problem(&self, format_name: &str) -> Result<String, cxx::Exception> {
        ffi::detector_error_model_shortest_error_sat_problem(self.inner(), format_name)
    }

    pub fn likeliest_error_sat_problem(
        &self,
        quantization: i32,
        format_name: &str,
    ) -> Result<String, cxx::Exception> {
        ffi::detector_error_model_likeliest_error_sat_problem(
            self.inner(),
            quantization,
            format_name,
        )
    }

    fn inner(&self) -> &ffi::DetectorErrorModelHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx detector error model construction returned null")
    }
}

impl Tableau {
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            inner: ffi::new_tableau(num_qubits),
        }
    }

    #[must_use]
    pub fn random(num_qubits: usize) -> Self {
        Self {
            inner: ffi::tableau_random(num_qubits),
        }
    }

    #[must_use]
    pub fn iter_all(num_qubits: usize, unsigned: bool) -> TableauIterator {
        TableauIterator {
            inner: ffi::tableau_iter_all(num_qubits, unsigned),
        }
    }

    pub fn from_named_gate(name: &str) -> Result<Self, cxx::Exception> {
        ffi::tableau_from_named_gate(name).map(|inner| Self { inner })
    }

    pub fn from_state_vector_data(
        state_vector: Vec<f32>,
        endian: &str,
    ) -> Result<Self, cxx::Exception> {
        ffi::tableau_from_state_vector_data(state_vector, endian).map(|inner| Self { inner })
    }

    pub fn from_unitary_matrix_data(
        matrix: Vec<f32>,
        endian: &str,
    ) -> Result<Self, cxx::Exception> {
        ffi::tableau_from_unitary_matrix_data(matrix, endian).map(|inner| Self { inner })
    }

    pub fn from_conjugated_generator_texts(
        xs: Vec<String>,
        zs: Vec<String>,
    ) -> Result<Self, cxx::Exception> {
        ffi::tableau_from_conjugated_generator_texts(xs, zs).map(|inner| Self { inner })
    }

    pub fn from_stabilizer_texts(
        stabilizers: Vec<String>,
        allow_redundant: bool,
        allow_underconstrained: bool,
    ) -> Result<Self, cxx::Exception> {
        ffi::tableau_from_stabilizer_texts(stabilizers, allow_redundant, allow_underconstrained)
            .map(|inner| Self { inner })
    }

    pub fn then(&self, second: &Self) -> Result<Self, cxx::Exception> {
        ffi::tableau_then(self.inner(), second.inner()).map(|inner| Self { inner })
    }

    fn inner(&self) -> &ffi::TableauHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx tableau construction returned null")
    }

    fn inner_mut(self: Pin<&mut Self>) -> Pin<&mut ffi::TableauHandle> {
        self.get_mut().inner.pin_mut()
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        ffi::tableau_num_qubits(self.inner())
    }

    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            inner: ffi::tableau_add(self.inner(), other.inner()),
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        ffi::tableau_add_assign(self.inner.pin_mut(), other.inner());
    }

    pub fn append(
        self: Pin<&mut Self>,
        gate: &Self,
        targets: &[usize],
    ) -> Result<(), cxx::Exception> {
        ffi::tableau_append(self.inner_mut(), gate.inner(), targets)
    }

    pub fn prepend(
        self: Pin<&mut Self>,
        gate: &Self,
        targets: &[usize],
    ) -> Result<(), cxx::Exception> {
        ffi::tableau_prepend(self.inner_mut(), gate.inner(), targets)
    }

    #[must_use]
    pub fn inverse(&self, unsigned_only: bool) -> Self {
        Self {
            inner: ffi::tableau_inverse(self.inner(), unsigned_only),
        }
    }

    #[must_use]
    pub fn raised_to(&self, exponent: i64) -> Self {
        Self {
            inner: ffi::tableau_raised_to(self.inner(), exponent),
        }
    }

    pub fn x_sign(&self, target: usize) -> Result<i32, cxx::Exception> {
        ffi::tableau_x_sign(self.inner(), target)
    }

    pub fn y_sign(&self, target: usize) -> Result<i32, cxx::Exception> {
        ffi::tableau_y_sign(self.inner(), target)
    }

    pub fn z_sign(&self, target: usize) -> Result<i32, cxx::Exception> {
        ffi::tableau_z_sign(self.inner(), target)
    }

    pub fn x_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> Result<u8, cxx::Exception> {
        ffi::tableau_x_output_pauli(self.inner(), input_index, output_index)
    }

    pub fn y_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> Result<u8, cxx::Exception> {
        ffi::tableau_y_output_pauli(self.inner(), input_index, output_index)
    }

    pub fn z_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> Result<u8, cxx::Exception> {
        ffi::tableau_z_output_pauli(self.inner(), input_index, output_index)
    }

    pub fn x_output(&self, target: usize) -> PauliString {
        PauliString {
            inner: ffi::tableau_x_output(self.inner(), target),
        }
    }

    pub fn y_output(&self, target: usize) -> PauliString {
        PauliString {
            inner: ffi::tableau_y_output(self.inner(), target),
        }
    }

    pub fn z_output(&self, target: usize) -> PauliString {
        PauliString {
            inner: ffi::tableau_z_output(self.inner(), target),
        }
    }

    pub fn inverse_x_output(&self, target: usize, unsigned_only: bool) -> PauliString {
        PauliString {
            inner: ffi::tableau_inverse_x_output(self.inner(), target, unsigned_only),
        }
    }

    pub fn inverse_y_output(&self, target: usize, unsigned_only: bool) -> PauliString {
        PauliString {
            inner: ffi::tableau_inverse_y_output(self.inner(), target, unsigned_only),
        }
    }

    pub fn inverse_z_output(&self, target: usize, unsigned_only: bool) -> PauliString {
        PauliString {
            inner: ffi::tableau_inverse_z_output(self.inner(), target, unsigned_only),
        }
    }

    pub fn conjugate_pauli_string(&self, pauli_string: &PauliString) -> PauliString {
        PauliString {
            inner: ffi::tableau_conjugate_pauli_string(self.inner(), pauli_string.inner()),
        }
    }

    pub fn conjugate_pauli_string_within(
        &self,
        pauli_string: &PauliString,
        targets: &[usize],
        inverse: bool,
    ) -> Result<PauliString, cxx::Exception> {
        ffi::tableau_conjugate_pauli_string_within(
            self.inner(),
            pauli_string.inner(),
            targets,
            inverse,
        )
        .map(|inner| PauliString { inner })
    }

    pub fn to_stabilizer_texts(&self, canonicalize: bool) -> Vec<String> {
        ffi::tableau_to_stabilizer_texts(self.inner(), canonicalize)
    }

    pub fn to_circuit(&self, method: &str) -> Result<Circuit, cxx::Exception> {
        ffi::tableau_to_circuit(self.inner(), method).map(|inner| Circuit { inner })
    }

    pub fn to_pauli_string(&self) -> Result<PauliString, cxx::Exception> {
        ffi::tableau_to_pauli_string(self.inner()).map(|inner| PauliString { inner })
    }

    pub fn inverse_x_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> Result<u8, cxx::Exception> {
        ffi::tableau_inverse_x_output_pauli(self.inner(), input_index, output_index)
    }

    pub fn inverse_y_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> Result<u8, cxx::Exception> {
        ffi::tableau_inverse_y_output_pauli(self.inner(), input_index, output_index)
    }

    pub fn inverse_z_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> Result<u8, cxx::Exception> {
        ffi::tableau_inverse_z_output_pauli(self.inner(), input_index, output_index)
    }

    #[must_use]
    pub fn to_text(&self) -> String {
        ffi::tableau_to_string(self.inner())
    }

    #[must_use]
    pub fn to_repr_text(&self) -> String {
        ffi::tableau_to_repr(self.inner())
    }

    pub fn to_unitary_matrix_data(&self, endian: &str) -> Result<Vec<f32>, cxx::Exception> {
        ffi::tableau_to_unitary_matrix_data(self.inner(), endian)
    }

    pub fn to_state_vector_data(&self, endian: &str) -> Result<Vec<f32>, cxx::Exception> {
        ffi::tableau_to_state_vector_data(self.inner(), endian)
    }
}

impl TableauSimulator {
    #[must_use]
    pub fn new(num_qubits: usize, seed: u64) -> Self {
        Self {
            inner: ffi::new_tableau_simulator(seed, num_qubits),
        }
    }

    fn inner(&self) -> &ffi::TableauSimulatorHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx tableau simulator construction returned null")
    }

    fn inner_mut(&mut self) -> Pin<&mut ffi::TableauSimulatorHandle> {
        self.inner.pin_mut()
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        ffi::tableau_simulator_num_qubits(self.inner())
    }

    pub fn set_num_qubits(&mut self, new_num_qubits: usize) {
        ffi::tableau_simulator_set_num_qubits(self.inner_mut(), new_num_qubits);
    }

    #[must_use]
    pub fn current_inverse_tableau(&self) -> Tableau {
        Tableau {
            inner: ffi::tableau_simulator_current_inverse_tableau(self.inner()),
        }
    }

    pub fn set_inverse_tableau(&mut self, tableau: &Tableau) {
        ffi::tableau_simulator_set_inverse_tableau(self.inner_mut(), tableau.inner());
    }

    #[must_use]
    pub fn current_measurement_record(&self) -> Vec<bool> {
        ffi::tableau_simulator_current_measurement_record(self.inner())
            .into_iter()
            .map(|bit| bit != 0)
            .collect()
    }

    pub fn do_circuit(&mut self, circuit: &Circuit) {
        ffi::tableau_simulator_do_circuit(self.inner_mut(), circuit.inner());
    }

    pub fn do_pauli_string(&mut self, pauli_string: &PauliString) {
        ffi::tableau_simulator_do_pauli_string(self.inner_mut(), pauli_string.inner());
    }

    pub fn do_tableau(&mut self, tableau: &Tableau, targets: &[usize]) {
        ffi::tableau_simulator_do_tableau(self.inner_mut(), tableau.inner(), targets);
    }

    #[must_use]
    pub fn peek_bloch(&mut self, target: usize) -> PauliString {
        PauliString {
            inner: ffi::tableau_simulator_peek_bloch(self.inner_mut(), target),
        }
    }

    pub fn peek_x(&mut self, target: usize) -> i32 {
        ffi::tableau_simulator_peek_x(self.inner_mut(), target)
    }

    pub fn peek_y(&mut self, target: usize) -> i32 {
        ffi::tableau_simulator_peek_y(self.inner_mut(), target)
    }

    pub fn peek_z(&mut self, target: usize) -> i32 {
        ffi::tableau_simulator_peek_z(self.inner_mut(), target)
    }

    pub fn measure(&mut self, target: usize) -> bool {
        ffi::tableau_simulator_measure(self.inner_mut(), target)
    }

    #[must_use]
    pub fn measure_many(&mut self, targets: &[usize]) -> Vec<bool> {
        ffi::tableau_simulator_measure_many(self.inner_mut(), targets)
            .into_iter()
            .map(|bit| bit != 0)
            .collect()
    }

    pub fn peek_observable_expectation(
        &self,
        observable: &PauliString,
    ) -> Result<i32, cxx::Exception> {
        ffi::tableau_simulator_peek_observable_expectation(self.inner(), observable.inner())
    }

    pub fn measure_observable(
        &mut self,
        observable: &PauliString,
        flip_probability: f64,
    ) -> Result<bool, cxx::Exception> {
        ffi::tableau_simulator_measure_observable(
            self.inner_mut(),
            observable.inner(),
            flip_probability,
        )
    }

    pub fn postselect_observable(
        &mut self,
        observable: &PauliString,
        desired_value: bool,
    ) -> Result<(), cxx::Exception> {
        ffi::tableau_simulator_postselect_observable(
            self.inner_mut(),
            observable.inner(),
            desired_value,
        )
    }

    pub fn postselect_x(
        &mut self,
        targets: &[usize],
        desired_value: bool,
    ) -> Result<(), cxx::Exception> {
        ffi::tableau_simulator_postselect_x(self.inner_mut(), targets, desired_value)
    }

    pub fn postselect_y(
        &mut self,
        targets: &[usize],
        desired_value: bool,
    ) -> Result<(), cxx::Exception> {
        ffi::tableau_simulator_postselect_y(self.inner_mut(), targets, desired_value)
    }

    pub fn postselect_z(
        &mut self,
        targets: &[usize],
        desired_value: bool,
    ) -> Result<(), cxx::Exception> {
        ffi::tableau_simulator_postselect_z(self.inner_mut(), targets, desired_value)
    }

    pub fn measure_kickback(&mut self, target: usize) -> (bool, Option<PauliString>) {
        let data = ffi::tableau_simulator_measure_kickback(self.inner_mut(), target);
        let kickback = if data.has_kickback {
            Some(
                PauliString::from_text(&data.kickback_text)
                    .expect("kickback text from stim-cxx should parse as a PauliString"),
            )
        } else {
            None
        };
        (data.result, kickback)
    }
}

impl FlipSimulator {
    #[must_use]
    pub fn new(
        batch_size: usize,
        disable_stabilizer_randomization: bool,
        num_qubits: usize,
        seed: u64,
    ) -> Self {
        Self {
            inner: ffi::new_frame_simulator(
                batch_size,
                disable_stabilizer_randomization,
                num_qubits,
                seed,
            ),
        }
    }

    fn inner(&self) -> &ffi::FrameSimulatorHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx frame simulator construction returned null")
    }

    fn inner_mut(&mut self) -> Pin<&mut ffi::FrameSimulatorHandle> {
        self.inner.pin_mut()
    }

    #[must_use]
    pub fn batch_size(&self) -> usize {
        ffi::frame_simulator_batch_size(self.inner())
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        ffi::frame_simulator_num_qubits(self.inner())
    }

    #[must_use]
    pub fn num_measurements(&self) -> usize {
        ffi::frame_simulator_num_measurements(self.inner())
    }

    #[must_use]
    pub fn num_detectors(&self) -> usize {
        ffi::frame_simulator_num_detectors(self.inner())
    }

    #[must_use]
    pub fn num_observables(&self) -> usize {
        ffi::frame_simulator_num_observables(self.inner())
    }

    pub fn clear(&mut self) {
        ffi::frame_simulator_clear(self.inner_mut());
    }

    pub fn do_circuit(&mut self, circuit: &Circuit) {
        ffi::frame_simulator_do_circuit(self.inner_mut(), circuit.inner());
    }

    pub fn set_pauli_flip(
        &mut self,
        pauli: u8,
        qubit_index: i64,
        instance_index: i64,
    ) -> Result<(), cxx::Exception> {
        ffi::frame_simulator_set_pauli_flip(self.inner_mut(), pauli, qubit_index, instance_index)
    }

    pub fn peek_pauli_flips(&self) -> Result<Vec<PauliString>, cxx::Exception> {
        ffi::frame_simulator_peek_pauli_flips(self.inner())
            .into_iter()
            .map(|text| PauliString::from_text(&text))
            .collect()
    }

    pub fn peek_pauli_flip(&self, instance_index: i64) -> Result<PauliString, cxx::Exception> {
        ffi::frame_simulator_peek_pauli_flip(self.inner(), instance_index)
            .and_then(|text| PauliString::from_text(&text))
    }

    pub fn broadcast_pauli_errors(
        &mut self,
        pauli: u8,
        mask: Vec<u8>,
        mask_num_qubits: usize,
        p: f32,
    ) -> Result<(), cxx::Exception> {
        ffi::frame_simulator_broadcast_pauli_errors(
            self.inner_mut(),
            pauli,
            mask,
            mask_num_qubits,
            p,
        )
    }

    pub fn generate_bernoulli_samples(
        &mut self,
        num_samples: usize,
        p: f32,
        bit_packed: bool,
    ) -> Result<Vec<u8>, cxx::Exception> {
        ffi::frame_simulator_generate_bernoulli_samples(
            self.inner_mut(),
            num_samples,
            p,
            bit_packed,
        )
    }

    pub fn append_measurement_flips(
        &mut self,
        data: Vec<u8>,
        num_measurements: usize,
        bit_packed: bool,
    ) -> Result<(), cxx::Exception> {
        ffi::frame_simulator_append_measurement_flips(
            self.inner_mut(),
            data,
            num_measurements,
            bit_packed,
        )
    }

    pub fn get_measurement_flips(&self, bit_packed: bool) -> ffi::BitTableData {
        ffi::frame_simulator_get_measurement_flips(self.inner(), bit_packed)
    }

    pub fn get_detector_flips(&self, bit_packed: bool) -> ffi::BitTableData {
        ffi::frame_simulator_get_detector_flips(self.inner(), bit_packed)
    }

    pub fn get_observable_flips(&self, bit_packed: bool) -> ffi::BitTableData {
        ffi::frame_simulator_get_observable_flips(self.inner(), bit_packed)
    }
}

impl PauliString {
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            inner: ffi::new_pauli_string(num_qubits),
        }
    }

    pub fn from_text(text: &str) -> Result<Self, cxx::Exception> {
        ffi::pauli_string_from_text(text).map(|inner| Self { inner })
    }

    #[must_use]
    pub fn random(num_qubits: usize) -> Self {
        Self {
            inner: ffi::pauli_string_random(num_qubits),
        }
    }

    #[must_use]
    pub fn iter_all(
        num_qubits: usize,
        min_weight: usize,
        max_weight: usize,
        allow_x: bool,
        allow_y: bool,
        allow_z: bool,
    ) -> PauliStringIterator {
        PauliStringIterator {
            inner: ffi::pauli_string_iter_all(
                num_qubits, min_weight, max_weight, allow_x, allow_y, allow_z,
            ),
        }
    }

    fn inner(&self) -> &ffi::PauliStringHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx pauli string construction returned null")
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        ffi::pauli_string_num_qubits(self.inner())
    }

    #[must_use]
    pub fn weight(&self) -> usize {
        ffi::pauli_string_weight(self.inner())
    }

    pub fn get_item(&self, index: i64) -> Result<u8, cxx::Exception> {
        ffi::pauli_string_get_item(self.inner(), index)
    }

    pub fn set_item(&mut self, index: i64, new_pauli: u8) -> Result<(), cxx::Exception> {
        ffi::pauli_string_set_item(self.inner.pin_mut(), index, new_pauli)
    }

    pub fn get_slice(&self, start: i64, step: i64, slice_length: i64) -> PauliString {
        PauliString {
            inner: ffi::pauli_string_get_slice(self.inner(), start, step, slice_length),
        }
    }

    #[must_use]
    pub fn commutes(&self, other: &Self) -> bool {
        ffi::pauli_string_commutes(self.inner(), other.inner())
    }

    pub fn pauli_indices(&self, included_paulis: &str) -> Result<Vec<u64>, cxx::Exception> {
        ffi::pauli_string_pauli_indices(self.inner(), included_paulis)
    }

    pub fn sign_code(&self) -> i32 {
        ffi::pauli_string_sign_code(self.inner())
    }

    pub fn to_tableau(&self) -> Tableau {
        Tableau {
            inner: ffi::pauli_string_to_tableau(self.inner()),
        }
    }

    #[must_use]
    pub fn to_text(&self) -> String {
        ffi::pauli_string_to_string(self.inner())
    }

    #[must_use]
    pub fn to_repr_text(&self) -> String {
        ffi::pauli_string_to_repr(self.inner())
    }
}

impl CliffordString {
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            inner: ffi::new_clifford_string(num_qubits),
        }
    }

    pub fn from_text(text: &str) -> Result<Self, cxx::Exception> {
        ffi::clifford_string_from_text(text).map(|inner| Self { inner })
    }

    #[must_use]
    pub fn from_pauli_string(pauli_string: &PauliString) -> Self {
        Self {
            inner: ffi::clifford_string_from_pauli_string(pauli_string.inner()),
        }
    }

    pub fn from_circuit(circuit: &Circuit) -> Result<Self, cxx::Exception> {
        ffi::clifford_string_from_circuit(circuit.inner()).map(|inner| Self { inner })
    }

    #[must_use]
    pub fn random(num_qubits: usize) -> Self {
        Self {
            inner: ffi::clifford_string_random(num_qubits),
        }
    }

    #[must_use]
    pub fn all_cliffords_string() -> Self {
        Self {
            inner: ffi::clifford_string_all_cliffords_string(),
        }
    }

    fn inner(&self) -> &ffi::CliffordStringHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx clifford string construction returned null")
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        ffi::clifford_string_num_qubits(self.inner())
    }

    pub fn get_item_name(&self, index: i64) -> Result<String, cxx::Exception> {
        ffi::clifford_string_get_item_name(self.inner(), index)
    }

    #[must_use]
    pub fn get_slice(&self, start: i64, step: i64, slice_length: i64) -> Self {
        Self {
            inner: ffi::clifford_string_get_slice(self.inner(), start, step, slice_length),
        }
    }

    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            inner: ffi::clifford_string_add(self.inner(), other.inner()),
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        ffi::clifford_string_add_assign(self.inner.pin_mut(), other.inner());
    }

    #[must_use]
    pub fn mul_clifford(&self, other: &Self) -> Self {
        Self {
            inner: ffi::clifford_string_mul(self.inner(), other.inner()),
        }
    }

    pub fn mul_assign_clifford(&mut self, other: &Self) {
        ffi::clifford_string_mul_assign(self.inner.pin_mut(), other.inner());
    }

    pub fn repeat(&self, repetitions: u64) -> Result<Self, cxx::Exception> {
        ffi::clifford_string_repeat(self.inner(), repetitions).map(|inner| Self { inner })
    }

    pub fn repeat_assign(&mut self, repetitions: u64) -> Result<(), cxx::Exception> {
        ffi::clifford_string_repeat_assign(self.inner.pin_mut(), repetitions)
    }

    #[must_use]
    pub fn pow(&self, exponent: i64) -> Self {
        Self {
            inner: ffi::clifford_string_pow(self.inner(), exponent),
        }
    }

    pub fn ipow(&mut self, exponent: i64) {
        ffi::clifford_string_ipow(self.inner.pin_mut(), exponent);
    }

    #[must_use]
    pub fn x_outputs(&self) -> PauliString {
        PauliString {
            inner: ffi::clifford_string_x_outputs(self.inner()),
        }
    }

    #[must_use]
    pub fn x_signs_bit_packed(&self) -> Vec<u8> {
        ffi::clifford_string_x_signs_bit_packed(self.inner())
    }

    #[must_use]
    pub fn y_outputs(&self) -> PauliString {
        PauliString {
            inner: ffi::clifford_string_y_outputs(self.inner()),
        }
    }

    #[must_use]
    pub fn y_signs_bit_packed(&self) -> Vec<u8> {
        ffi::clifford_string_y_signs_bit_packed(self.inner())
    }

    #[must_use]
    pub fn z_outputs(&self) -> PauliString {
        PauliString {
            inner: ffi::clifford_string_z_outputs(self.inner()),
        }
    }

    #[must_use]
    pub fn z_signs_bit_packed(&self) -> Vec<u8> {
        ffi::clifford_string_z_signs_bit_packed(self.inner())
    }

    #[must_use]
    pub fn to_text(&self) -> String {
        ffi::clifford_string_to_string(self.inner())
    }

    #[must_use]
    pub fn to_repr_text(&self) -> String {
        ffi::clifford_string_to_repr(self.inner())
    }
}

impl Clone for Circuit {
    fn clone(&self) -> Self {
        Self {
            inner: ffi::circuit_clone(self.inner()),
        }
    }
}

impl Clone for DetectorErrorModel {
    fn clone(&self) -> Self {
        Self {
            inner: ffi::detector_error_model_clone(self.inner()),
        }
    }
}

impl Clone for Tableau {
    fn clone(&self) -> Self {
        Self {
            inner: ffi::tableau_clone(self.inner()),
        }
    }
}

impl Clone for TableauSimulator {
    fn clone(&self) -> Self {
        Self {
            inner: ffi::tableau_simulator_clone(self.inner()),
        }
    }
}

impl Clone for FlipSimulator {
    fn clone(&self) -> Self {
        Self {
            inner: ffi::frame_simulator_clone(self.inner()),
        }
    }
}

impl TableauIterator {
    fn inner(&self) -> &ffi::TableauIteratorHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx tableau iterator construction returned null")
    }

    fn inner_mut(&mut self) -> Pin<&mut ffi::TableauIteratorHandle> {
        self.inner.pin_mut()
    }
}

impl Clone for TableauIterator {
    fn clone(&self) -> Self {
        Self {
            inner: ffi::tableau_iterator_clone(self.inner()),
        }
    }
}

impl Iterator for TableauIterator {
    type Item = Tableau;

    fn next(&mut self) -> Option<Self::Item> {
        let inner = ffi::tableau_iterator_next(self.inner_mut());
        if inner.is_null() {
            None
        } else {
            Some(Tableau { inner })
        }
    }
}

impl PartialEq for Tableau {
    fn eq(&self, other: &Self) -> bool {
        ffi::tableau_equals(self.inner(), other.inner())
    }
}

impl Eq for Tableau {}

impl Clone for PauliString {
    fn clone(&self) -> Self {
        Self {
            inner: ffi::pauli_string_clone(self.inner()),
        }
    }
}

impl PauliStringIterator {
    fn inner(&self) -> &ffi::PauliStringIteratorHandle {
        self.inner
            .as_ref()
            .expect("stim-cxx pauli string iterator construction returned null")
    }

    fn inner_mut(&mut self) -> Pin<&mut ffi::PauliStringIteratorHandle> {
        self.inner.pin_mut()
    }
}

impl Clone for PauliStringIterator {
    fn clone(&self) -> Self {
        Self {
            inner: ffi::pauli_string_iterator_clone(self.inner()),
        }
    }
}

impl Iterator for PauliStringIterator {
    type Item = PauliString;

    fn next(&mut self) -> Option<Self::Item> {
        let inner = ffi::pauli_string_iterator_next(self.inner_mut());
        if inner.is_null() {
            None
        } else {
            Some(PauliString { inner })
        }
    }
}

impl PartialEq for PauliString {
    fn eq(&self, other: &Self) -> bool {
        ffi::pauli_string_equals(self.inner(), other.inner())
    }
}

impl Eq for PauliString {}

impl Clone for CliffordString {
    fn clone(&self) -> Self {
        Self {
            inner: ffi::clifford_string_clone(self.inner()),
        }
    }
}

impl PartialEq for CliffordString {
    fn eq(&self, other: &Self) -> bool {
        ffi::clifford_string_equals(self.inner(), other.inner())
    }
}

impl Eq for CliffordString {}

impl Clone for GateData {
    fn clone(&self) -> Self {
        self.clone_handle()
    }
}

impl Default for Circuit {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DetectorErrorModel {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SmokeProbe {
    inner: UniquePtr<ffi::SmokeProbe>,
}

impl SmokeProbe {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: ffi::new_smoke_probe(),
        }
    }

    #[must_use]
    pub fn describe(&self) -> String {
        self.inner().describe().to_string()
    }

    #[must_use]
    pub fn weighted_checksum(&self, values: &[u64]) -> u64 {
        self.inner().weighted_checksum(values)
    }

    fn inner(&self) -> &ffi::SmokeProbe {
        self.inner
            .as_ref()
            .expect("stim-cxx smoke probe construction returned null")
    }
}

impl Default for SmokeProbe {
    fn default() -> Self {
        Self::new()
    }
}

#[cxx::bridge(namespace = "stimrs::bridge")]
mod ffi {
    struct DemSampleBatch {
        detectors: Vec<u8>,
        observables: Vec<u8>,
        errors: Vec<u8>,
    }

    struct TableauMeasureKickbackData {
        result: bool,
        has_kickback: bool,
        kickback_text: String,
    }

    struct BitTableData {
        data: Vec<u8>,
        rows: u64,
        cols: u64,
        bit_packed: bool,
    }

    struct GateTargetWithCoordsData {
        raw_target: u32,
        coords: Vec<f64>,
    }

    struct DemTargetWithCoordsData {
        dem_target: String,
        coords: Vec<f64>,
    }

    struct CircuitErrorLocationStackFrameData {
        instruction_offset: u64,
        iteration_index: u64,
        instruction_repetitions_arg: u64,
    }

    struct CircuitTargetsInsideInstructionData {
        gate: String,
        tag: String,
        args: Vec<f64>,
        target_range_start: u64,
        target_range_end: u64,
        targets_in_range: Vec<GateTargetWithCoordsData>,
    }

    struct FlippedMeasurementData {
        record_index: u64,
        observable: Vec<GateTargetWithCoordsData>,
    }

    struct CircuitErrorLocationData {
        tick_offset: u64,
        flipped_pauli_product: Vec<GateTargetWithCoordsData>,
        flipped_measurement: FlippedMeasurementData,
        instruction_targets: CircuitTargetsInsideInstructionData,
        stack_frames: Vec<CircuitErrorLocationStackFrameData>,
        noise_tag: String,
    }

    struct ExplainedErrorData {
        dem_error_terms: Vec<DemTargetWithCoordsData>,
        circuit_error_locations: Vec<CircuitErrorLocationData>,
    }

    unsafe extern "C++" {
        include!("stim-cxx/include/stim_rs_bridge.h");

        fn read_shot_data_file_bit_packed(
            filepath: &str,
            format_name: &str,
            num_measurements: u64,
            num_detectors: u64,
            num_observables: u64,
        ) -> Result<Vec<u8>>;
        fn write_shot_data_file_bit_packed(
            data: &[u8],
            shots: u64,
            filepath: &str,
            format_name: &str,
            num_measurements: u64,
            num_detectors: u64,
            num_observables: u64,
        ) -> Result<()>;
        fn all_gate_names() -> Vec<String>;
        fn stim_main(command_line_args: Vec<String>) -> i32;
        fn canonicalize_flow_text(text: &str) -> Result<String>;
        fn multiply_flow_texts(left: &str, right: &str) -> Result<String>;

        type CircuitHandle;
        type DetectorErrorModelHandle;
        type TableauHandle;
        type TableauSimulatorHandle;
        type FrameSimulatorHandle;
        type TableauIteratorHandle;
        type PauliStringHandle;
        type CliffordStringHandle;
        type PauliStringIteratorHandle;
        type DetectorSamplerHandle;
        type GateDataHandle;
        type MeasurementSamplerHandle;
        type MeasurementsToDetectionEventsConverterHandle;
        type DemSamplerHandle;
        type SmokeProbe;

        fn gate_data_by_name(name: &str) -> Result<UniquePtr<GateDataHandle>>;
        fn name(self: &GateDataHandle) -> String;
        fn aliases(self: &GateDataHandle) -> Vec<String>;
        fn num_parens_arguments_range(self: &GateDataHandle) -> Vec<u8>;
        fn is_noisy_gate(self: &GateDataHandle) -> bool;
        fn is_reset(self: &GateDataHandle) -> bool;
        fn is_single_qubit_gate(self: &GateDataHandle) -> bool;
        fn is_symmetric_gate(self: &GateDataHandle) -> bool;
        fn is_two_qubit_gate(self: &GateDataHandle) -> bool;
        fn is_unitary(self: &GateDataHandle) -> bool;
        fn produces_measurements(self: &GateDataHandle) -> bool;
        fn takes_measurement_record_targets(self: &GateDataHandle) -> bool;
        fn takes_pauli_targets(self: &GateDataHandle) -> bool;
        fn flows(self: &GateDataHandle) -> Vec<String>;
        fn tableau(self: &GateDataHandle) -> UniquePtr<TableauHandle>;
        fn clone_handle(self: &GateDataHandle) -> UniquePtr<GateDataHandle>;
        fn inverse(self: &GateDataHandle) -> UniquePtr<GateDataHandle>;
        fn generalized_inverse(self: &GateDataHandle) -> UniquePtr<GateDataHandle>;
        fn hadamard_conjugated(
            self: &GateDataHandle,
            unsigned_only: bool,
        ) -> UniquePtr<GateDataHandle>;

        fn new_circuit() -> UniquePtr<CircuitHandle>;
        fn circuit_from_stim_program_text(text: &str) -> Result<UniquePtr<CircuitHandle>>;
        fn circuit_clone(handle: &CircuitHandle) -> UniquePtr<CircuitHandle>;
        fn circuit_add(left: &CircuitHandle, right: &CircuitHandle) -> UniquePtr<CircuitHandle>;
        fn circuit_add_assign(left: Pin<&mut CircuitHandle>, right: &CircuitHandle);
        fn circuit_mul(handle: &CircuitHandle, repetitions: u64) -> UniquePtr<CircuitHandle>;
        fn circuit_mul_assign(handle: Pin<&mut CircuitHandle>, repetitions: u64);
        fn circuit_generated(
            code_task: &str,
            distance: usize,
            rounds: usize,
            after_clifford_depolarization: f64,
            before_round_data_depolarization: f64,
            before_measure_flip_probability: f64,
            after_reset_flip_probability: f64,
        ) -> Result<UniquePtr<CircuitHandle>>;
        fn circuit_has_flow_text(
            handle: &CircuitHandle,
            flow_text: &str,
            unsigned_only: bool,
        ) -> Result<bool>;
        fn circuit_has_all_flows_text(
            handle: &CircuitHandle,
            flow_texts: Vec<String>,
            unsigned_only: bool,
        ) -> Result<bool>;
        fn circuit_flow_generators_texts(handle: &CircuitHandle) -> Vec<String>;
        fn circuit_solve_flow_measurements_text(
            handle: &CircuitHandle,
            flow_texts: Vec<String>,
        ) -> Result<Vec<String>>;
        fn circuit_time_reversed_for_flows_text(
            out_flow_texts: &mut Vec<String>,
            handle: &CircuitHandle,
            flow_texts: Vec<String>,
            dont_turn_measurements_into_resets: bool,
        ) -> Result<String>;
        fn circuit_diagram_with_options(
            handle: &CircuitHandle,
            type_name: &str,
            has_tick_range: bool,
            tick_start: u64,
            tick_count: u64,
            has_rows: bool,
            rows: usize,
        ) -> Result<String>;
        fn circuit_diagram_with_options_and_filters(
            handle: &CircuitHandle,
            type_name: &str,
            has_tick_range: bool,
            tick_start: u64,
            tick_count: u64,
            has_rows: bool,
            rows: usize,
            filter_coords: Vec<String>,
        ) -> Result<String>;
        fn circuit_to_stim_program_text(handle: &CircuitHandle) -> String;
        fn circuit_num_qubits(handle: &CircuitHandle) -> usize;
        fn circuit_len(handle: &CircuitHandle) -> usize;
        fn circuit_num_measurements(handle: &CircuitHandle) -> u64;
        fn circuit_count_determined_measurements(
            handle: &CircuitHandle,
            unknown_input: bool,
        ) -> u64;
        fn circuit_num_detectors(handle: &CircuitHandle) -> u64;
        fn circuit_num_observables(handle: &CircuitHandle) -> u64;
        fn circuit_num_ticks(handle: &CircuitHandle) -> u64;
        fn circuit_num_sweep_bits(handle: &CircuitHandle) -> usize;
        fn circuit_append_from_stim_program_text(
            handle: Pin<&mut CircuitHandle>,
            text: &str,
        ) -> Result<()>;
        fn circuit_append_gate(
            handle: Pin<&mut CircuitHandle>,
            gate_name: &str,
            targets: &[u32],
            args: &[f64],
        ) -> Result<()>;
        fn circuit_append_repeat_block(
            handle: Pin<&mut CircuitHandle>,
            repeat_count: u64,
            body: &CircuitHandle,
            tag: &str,
        ) -> Result<()>;
        fn circuit_clear(handle: Pin<&mut CircuitHandle>);
        fn circuit_equals(left: &CircuitHandle, right: &CircuitHandle) -> bool;
        fn circuit_approx_equals(left: &CircuitHandle, right: &CircuitHandle, atol: f64) -> bool;
        fn circuit_without_noise(handle: &CircuitHandle) -> UniquePtr<CircuitHandle>;
        fn circuit_with_inlined_feedback(handle: &CircuitHandle) -> UniquePtr<CircuitHandle>;
        fn circuit_without_tags(handle: &CircuitHandle) -> UniquePtr<CircuitHandle>;
        fn circuit_flattened(handle: &CircuitHandle) -> UniquePtr<CircuitHandle>;
        fn circuit_decomposed(handle: &CircuitHandle) -> UniquePtr<CircuitHandle>;
        fn circuit_inverse(handle: &CircuitHandle) -> Result<UniquePtr<CircuitHandle>>;
        fn circuit_to_qasm(
            handle: &CircuitHandle,
            open_qasm_version: i32,
            skip_dets_and_obs: bool,
        ) -> Result<String>;
        fn circuit_to_quirk_url(handle: &CircuitHandle) -> Result<String>;
        fn circuit_to_crumble_url(handle: &CircuitHandle, skip_detectors: bool) -> Result<String>;
        fn circuit_shortest_error_sat_problem(
            handle: &CircuitHandle,
            format_name: &str,
        ) -> Result<String>;
        fn circuit_likeliest_error_sat_problem(
            handle: &CircuitHandle,
            quantization: i32,
            format_name: &str,
        ) -> Result<String>;
        fn circuit_detecting_regions_text(handle: &CircuitHandle) -> Result<String>;
        fn circuit_detecting_regions_text_with_options(
            handle: &CircuitHandle,
            target_texts: Vec<String>,
            ticks: Vec<u64>,
            ignore_anticommutation_errors: bool,
        ) -> Result<String>;
        fn circuit_explain_detector_error_model_errors(
            handle: &CircuitHandle,
            dem_filter_text: &str,
            has_dem_filter: bool,
            reduce_to_one_representative_error: bool,
        ) -> Result<Vec<ExplainedErrorData>>;
        fn circuit_shortest_graphlike_error(
            handle: &CircuitHandle,
            ignore_ungraphlike_errors: bool,
            canonicalize_circuit_errors: bool,
        ) -> Result<Vec<ExplainedErrorData>>;
        fn circuit_flattened_operation_texts(handle: &CircuitHandle) -> Vec<String>;
        fn circuit_search_for_undetectable_logical_errors(
            handle: &CircuitHandle,
            dont_explore_detection_event_sets_with_size_above: u64,
            dont_explore_edges_with_degree_above: u64,
            dont_explore_edges_increasing_symptom_degree: bool,
            canonicalize_circuit_errors: bool,
        ) -> Result<Vec<ExplainedErrorData>>;
        fn circuit_get_detector_coordinates_text(
            handle: &CircuitHandle,
            included_detector_indices: &[u64],
        ) -> Result<String>;
        fn circuit_get_final_qubit_coordinates_text(handle: &CircuitHandle) -> Result<String>;
        fn circuit_reference_sample_bit_packed(handle: &CircuitHandle) -> Vec<u8>;
        fn circuit_reference_detector_signs_bit_packed(handle: &CircuitHandle) -> Vec<u8>;
        fn circuit_reference_observable_signs_bit_packed(handle: &CircuitHandle) -> Vec<u8>;
        fn new_detector_error_model() -> UniquePtr<DetectorErrorModelHandle>;
        fn circuit_detector_error_model(
            handle: &CircuitHandle,
            decompose_errors: bool,
            flatten_loops: bool,
            allow_gauge_detectors: bool,
            approximate_disjoint_errors: f64,
            ignore_decomposition_failures: bool,
            block_decomposition_from_introducing_remnant_edges: bool,
        ) -> Result<UniquePtr<DetectorErrorModelHandle>>;
        fn circuit_missing_detectors(
            handle: &CircuitHandle,
            unknown_input: bool,
        ) -> UniquePtr<CircuitHandle>;
        fn circuit_to_tableau(
            handle: &CircuitHandle,
            ignore_noise: bool,
            ignore_measurement: bool,
            ignore_reset: bool,
        ) -> Result<UniquePtr<TableauHandle>>;
        fn new_tableau(num_qubits: usize) -> UniquePtr<TableauHandle>;
        fn tableau_random(num_qubits: usize) -> UniquePtr<TableauHandle>;
        fn tableau_iter_all(
            num_qubits: usize,
            unsigned_only: bool,
        ) -> UniquePtr<TableauIteratorHandle>;
        fn tableau_iterator_clone(
            handle: &TableauIteratorHandle,
        ) -> UniquePtr<TableauIteratorHandle>;
        fn tableau_iterator_next(
            handle: Pin<&mut TableauIteratorHandle>,
        ) -> UniquePtr<TableauHandle>;
        fn tableau_from_named_gate(name: &str) -> Result<UniquePtr<TableauHandle>>;
        fn tableau_from_state_vector_data(
            state_vector: Vec<f32>,
            endian: &str,
        ) -> Result<UniquePtr<TableauHandle>>;
        fn tableau_from_unitary_matrix_data(
            matrix: Vec<f32>,
            endian: &str,
        ) -> Result<UniquePtr<TableauHandle>>;
        fn tableau_from_conjugated_generator_texts(
            xs: Vec<String>,
            zs: Vec<String>,
        ) -> Result<UniquePtr<TableauHandle>>;
        fn tableau_from_stabilizer_texts(
            stabilizers: Vec<String>,
            allow_redundant: bool,
            allow_underconstrained: bool,
        ) -> Result<UniquePtr<TableauHandle>>;
        fn tableau_then(
            handle: &TableauHandle,
            second: &TableauHandle,
        ) -> Result<UniquePtr<TableauHandle>>;
        fn new_tableau_simulator(seed: u64, num_qubits: usize)
        -> UniquePtr<TableauSimulatorHandle>;
        fn new_clifford_string(num_qubits: usize) -> UniquePtr<CliffordStringHandle>;
        fn clifford_string_from_text(text: &str) -> Result<UniquePtr<CliffordStringHandle>>;
        fn clifford_string_from_pauli_string(
            handle: &PauliStringHandle,
        ) -> UniquePtr<CliffordStringHandle>;
        fn clifford_string_from_circuit(
            handle: &CircuitHandle,
        ) -> Result<UniquePtr<CliffordStringHandle>>;
        fn clifford_string_random(num_qubits: usize) -> UniquePtr<CliffordStringHandle>;
        fn clifford_string_all_cliffords_string() -> UniquePtr<CliffordStringHandle>;
        fn new_pauli_string(num_qubits: usize) -> UniquePtr<PauliStringHandle>;
        fn pauli_string_from_text(text: &str) -> Result<UniquePtr<PauliStringHandle>>;
        fn detector_error_model_from_dem_text(
            text: &str,
        ) -> Result<UniquePtr<DetectorErrorModelHandle>>;
        fn circuit_diagram(handle: &CircuitHandle, type_name: &str) -> Result<String>;
        fn detector_error_model_diagram(
            handle: &DetectorErrorModelHandle,
            type_name: &str,
        ) -> Result<String>;
        fn detector_error_model_clone(
            handle: &DetectorErrorModelHandle,
        ) -> UniquePtr<DetectorErrorModelHandle>;
        fn detector_error_model_add(
            left: &DetectorErrorModelHandle,
            right: &DetectorErrorModelHandle,
        ) -> UniquePtr<DetectorErrorModelHandle>;
        fn detector_error_model_add_assign(
            left: Pin<&mut DetectorErrorModelHandle>,
            right: &DetectorErrorModelHandle,
        );
        fn detector_error_model_mul(
            handle: &DetectorErrorModelHandle,
            repetitions: u64,
        ) -> UniquePtr<DetectorErrorModelHandle>;
        fn detector_error_model_mul_assign(
            handle: Pin<&mut DetectorErrorModelHandle>,
            repetitions: u64,
        );
        fn detector_error_model_to_dem_text(handle: &DetectorErrorModelHandle) -> String;
        fn detector_error_model_len(handle: &DetectorErrorModelHandle) -> usize;
        fn detector_error_model_num_detectors(handle: &DetectorErrorModelHandle) -> u64;
        fn detector_error_model_num_errors(handle: &DetectorErrorModelHandle) -> u64;
        fn detector_error_model_num_observables(handle: &DetectorErrorModelHandle) -> u64;
        fn detector_error_model_get_detector_coordinates_text(
            handle: &DetectorErrorModelHandle,
            included_detector_indices: &[u64],
        ) -> Result<String>;
        fn detector_error_model_clear(handle: Pin<&mut DetectorErrorModelHandle>);
        fn detector_error_model_equals(
            left: &DetectorErrorModelHandle,
            right: &DetectorErrorModelHandle,
        ) -> bool;
        fn detector_error_model_approx_equals(
            left: &DetectorErrorModelHandle,
            right: &DetectorErrorModelHandle,
            atol: f64,
        ) -> bool;
        fn detector_error_model_without_tags(
            handle: &DetectorErrorModelHandle,
        ) -> UniquePtr<DetectorErrorModelHandle>;
        fn detector_error_model_flattened(
            handle: &DetectorErrorModelHandle,
        ) -> UniquePtr<DetectorErrorModelHandle>;
        fn detector_error_model_rounded(
            handle: &DetectorErrorModelHandle,
            digits: u8,
        ) -> UniquePtr<DetectorErrorModelHandle>;
        fn detector_error_model_shortest_graphlike_error(
            handle: &DetectorErrorModelHandle,
            ignore_ungraphlike_errors: bool,
        ) -> Result<UniquePtr<DetectorErrorModelHandle>>;
        fn detector_error_model_shortest_error_sat_problem(
            handle: &DetectorErrorModelHandle,
            format_name: &str,
        ) -> Result<String>;
        fn detector_error_model_likeliest_error_sat_problem(
            handle: &DetectorErrorModelHandle,
            quantization: i32,
            format_name: &str,
        ) -> Result<String>;
        fn tableau_clone(handle: &TableauHandle) -> UniquePtr<TableauHandle>;
        fn tableau_equals(left: &TableauHandle, right: &TableauHandle) -> bool;
        fn tableau_add(left: &TableauHandle, right: &TableauHandle) -> UniquePtr<TableauHandle>;
        fn tableau_add_assign(left: Pin<&mut TableauHandle>, right: &TableauHandle);
        fn tableau_append(
            handle: Pin<&mut TableauHandle>,
            gate: &TableauHandle,
            targets: &[usize],
        ) -> Result<()>;
        fn tableau_prepend(
            handle: Pin<&mut TableauHandle>,
            gate: &TableauHandle,
            targets: &[usize],
        ) -> Result<()>;
        fn tableau_inverse(handle: &TableauHandle, unsigned_only: bool)
        -> UniquePtr<TableauHandle>;
        fn tableau_raised_to(handle: &TableauHandle, exponent: i64) -> UniquePtr<TableauHandle>;
        fn tableau_x_sign(handle: &TableauHandle, target: usize) -> Result<i32>;
        fn tableau_y_sign(handle: &TableauHandle, target: usize) -> Result<i32>;
        fn tableau_z_sign(handle: &TableauHandle, target: usize) -> Result<i32>;
        fn tableau_x_output_pauli(
            handle: &TableauHandle,
            input_index: usize,
            output_index: usize,
        ) -> Result<u8>;
        fn tableau_y_output_pauli(
            handle: &TableauHandle,
            input_index: usize,
            output_index: usize,
        ) -> Result<u8>;
        fn tableau_z_output_pauli(
            handle: &TableauHandle,
            input_index: usize,
            output_index: usize,
        ) -> Result<u8>;
        fn tableau_inverse_x_output_pauli(
            handle: &TableauHandle,
            input_index: usize,
            output_index: usize,
        ) -> Result<u8>;
        fn tableau_inverse_y_output_pauli(
            handle: &TableauHandle,
            input_index: usize,
            output_index: usize,
        ) -> Result<u8>;
        fn tableau_inverse_z_output_pauli(
            handle: &TableauHandle,
            input_index: usize,
            output_index: usize,
        ) -> Result<u8>;
        fn tableau_x_output(handle: &TableauHandle, target: usize) -> UniquePtr<PauliStringHandle>;
        fn tableau_y_output(handle: &TableauHandle, target: usize) -> UniquePtr<PauliStringHandle>;
        fn tableau_z_output(handle: &TableauHandle, target: usize) -> UniquePtr<PauliStringHandle>;
        fn tableau_inverse_x_output(
            handle: &TableauHandle,
            target: usize,
            unsigned_only: bool,
        ) -> UniquePtr<PauliStringHandle>;
        fn tableau_inverse_y_output(
            handle: &TableauHandle,
            target: usize,
            unsigned_only: bool,
        ) -> UniquePtr<PauliStringHandle>;
        fn tableau_inverse_z_output(
            handle: &TableauHandle,
            target: usize,
            unsigned_only: bool,
        ) -> UniquePtr<PauliStringHandle>;
        fn tableau_conjugate_pauli_string(
            handle: &TableauHandle,
            pauli_string: &PauliStringHandle,
        ) -> UniquePtr<PauliStringHandle>;
        fn tableau_conjugate_pauli_string_within(
            handle: &TableauHandle,
            pauli_string: &PauliStringHandle,
            targets: &[usize],
            inverse: bool,
        ) -> Result<UniquePtr<PauliStringHandle>>;
        fn tableau_to_stabilizer_texts(handle: &TableauHandle, canonicalize: bool) -> Vec<String>;
        fn tableau_to_circuit(
            handle: &TableauHandle,
            method: &str,
        ) -> Result<UniquePtr<CircuitHandle>>;
        fn tableau_to_pauli_string(handle: &TableauHandle) -> Result<UniquePtr<PauliStringHandle>>;
        fn tableau_simulator_clone(
            handle: &TableauSimulatorHandle,
        ) -> UniquePtr<TableauSimulatorHandle>;
        fn tableau_simulator_num_qubits(handle: &TableauSimulatorHandle) -> usize;
        fn tableau_simulator_set_num_qubits(
            handle: Pin<&mut TableauSimulatorHandle>,
            new_num_qubits: usize,
        );
        fn tableau_simulator_current_inverse_tableau(
            handle: &TableauSimulatorHandle,
        ) -> UniquePtr<TableauHandle>;
        fn tableau_simulator_set_inverse_tableau(
            handle: Pin<&mut TableauSimulatorHandle>,
            tableau: &TableauHandle,
        );
        fn tableau_simulator_current_measurement_record(handle: &TableauSimulatorHandle)
        -> Vec<u8>;
        fn tableau_simulator_do_circuit(
            handle: Pin<&mut TableauSimulatorHandle>,
            circuit: &CircuitHandle,
        );
        fn tableau_simulator_do_pauli_string(
            handle: Pin<&mut TableauSimulatorHandle>,
            pauli_string: &PauliStringHandle,
        );
        fn tableau_simulator_do_tableau(
            handle: Pin<&mut TableauSimulatorHandle>,
            tableau: &TableauHandle,
            targets: &[usize],
        );
        fn tableau_simulator_peek_bloch(
            handle: Pin<&mut TableauSimulatorHandle>,
            target: usize,
        ) -> UniquePtr<PauliStringHandle>;
        fn tableau_simulator_peek_x(handle: Pin<&mut TableauSimulatorHandle>, target: usize)
        -> i32;
        fn tableau_simulator_peek_y(handle: Pin<&mut TableauSimulatorHandle>, target: usize)
        -> i32;
        fn tableau_simulator_peek_z(handle: Pin<&mut TableauSimulatorHandle>, target: usize)
        -> i32;
        fn tableau_simulator_measure(
            handle: Pin<&mut TableauSimulatorHandle>,
            target: usize,
        ) -> bool;
        fn tableau_simulator_measure_many(
            handle: Pin<&mut TableauSimulatorHandle>,
            targets: &[usize],
        ) -> Vec<u8>;
        fn tableau_simulator_peek_observable_expectation(
            handle: &TableauSimulatorHandle,
            observable: &PauliStringHandle,
        ) -> Result<i32>;
        fn tableau_simulator_measure_observable(
            handle: Pin<&mut TableauSimulatorHandle>,
            observable: &PauliStringHandle,
            flip_probability: f64,
        ) -> Result<bool>;
        fn tableau_simulator_postselect_observable(
            handle: Pin<&mut TableauSimulatorHandle>,
            observable: &PauliStringHandle,
            desired_value: bool,
        ) -> Result<()>;
        fn tableau_simulator_postselect_x(
            handle: Pin<&mut TableauSimulatorHandle>,
            targets: &[usize],
            desired_value: bool,
        ) -> Result<()>;
        fn tableau_simulator_postselect_y(
            handle: Pin<&mut TableauSimulatorHandle>,
            targets: &[usize],
            desired_value: bool,
        ) -> Result<()>;
        fn tableau_simulator_postselect_z(
            handle: Pin<&mut TableauSimulatorHandle>,
            targets: &[usize],
            desired_value: bool,
        ) -> Result<()>;
        fn tableau_simulator_measure_kickback(
            handle: Pin<&mut TableauSimulatorHandle>,
            target: usize,
        ) -> TableauMeasureKickbackData;
        fn new_frame_simulator(
            batch_size: usize,
            disable_stabilizer_randomization: bool,
            num_qubits: usize,
            seed: u64,
        ) -> UniquePtr<FrameSimulatorHandle>;
        fn frame_simulator_clone(handle: &FrameSimulatorHandle) -> UniquePtr<FrameSimulatorHandle>;
        fn frame_simulator_batch_size(handle: &FrameSimulatorHandle) -> usize;
        fn frame_simulator_num_qubits(handle: &FrameSimulatorHandle) -> usize;
        fn frame_simulator_num_measurements(handle: &FrameSimulatorHandle) -> usize;
        fn frame_simulator_num_detectors(handle: &FrameSimulatorHandle) -> usize;
        fn frame_simulator_num_observables(handle: &FrameSimulatorHandle) -> usize;
        fn frame_simulator_clear(handle: Pin<&mut FrameSimulatorHandle>);
        fn frame_simulator_do_circuit(
            handle: Pin<&mut FrameSimulatorHandle>,
            circuit: &CircuitHandle,
        );
        fn frame_simulator_set_pauli_flip(
            handle: Pin<&mut FrameSimulatorHandle>,
            pauli: u8,
            qubit_index: i64,
            instance_index: i64,
        ) -> Result<()>;
        fn frame_simulator_peek_pauli_flips(handle: &FrameSimulatorHandle) -> Vec<String>;
        fn frame_simulator_peek_pauli_flip(
            handle: &FrameSimulatorHandle,
            instance_index: i64,
        ) -> Result<String>;
        fn frame_simulator_broadcast_pauli_errors(
            handle: Pin<&mut FrameSimulatorHandle>,
            pauli: u8,
            mask: Vec<u8>,
            mask_num_qubits: usize,
            p: f32,
        ) -> Result<()>;
        fn frame_simulator_generate_bernoulli_samples(
            handle: Pin<&mut FrameSimulatorHandle>,
            num_samples: usize,
            p: f32,
            bit_packed: bool,
        ) -> Result<Vec<u8>>;
        fn frame_simulator_append_measurement_flips(
            handle: Pin<&mut FrameSimulatorHandle>,
            data: Vec<u8>,
            num_measurements: usize,
            bit_packed: bool,
        ) -> Result<()>;
        fn frame_simulator_get_measurement_flips(
            handle: &FrameSimulatorHandle,
            bit_packed: bool,
        ) -> BitTableData;
        fn frame_simulator_get_detector_flips(
            handle: &FrameSimulatorHandle,
            bit_packed: bool,
        ) -> BitTableData;
        fn frame_simulator_get_observable_flips(
            handle: &FrameSimulatorHandle,
            bit_packed: bool,
        ) -> BitTableData;
        fn clifford_string_clone(handle: &CliffordStringHandle) -> UniquePtr<CliffordStringHandle>;
        fn clifford_string_equals(
            left: &CliffordStringHandle,
            right: &CliffordStringHandle,
        ) -> bool;
        fn clifford_string_num_qubits(handle: &CliffordStringHandle) -> usize;
        fn clifford_string_get_item_name(
            handle: &CliffordStringHandle,
            index: i64,
        ) -> Result<String>;
        fn clifford_string_get_slice(
            handle: &CliffordStringHandle,
            start: i64,
            step: i64,
            slice_length: i64,
        ) -> UniquePtr<CliffordStringHandle>;
        fn clifford_string_add(
            left: &CliffordStringHandle,
            right: &CliffordStringHandle,
        ) -> UniquePtr<CliffordStringHandle>;
        fn clifford_string_add_assign(
            left: Pin<&mut CliffordStringHandle>,
            right: &CliffordStringHandle,
        );
        fn clifford_string_mul(
            left: &CliffordStringHandle,
            right: &CliffordStringHandle,
        ) -> UniquePtr<CliffordStringHandle>;
        fn clifford_string_mul_assign(
            left: Pin<&mut CliffordStringHandle>,
            right: &CliffordStringHandle,
        );
        fn clifford_string_repeat(
            handle: &CliffordStringHandle,
            repetitions: u64,
        ) -> Result<UniquePtr<CliffordStringHandle>>;
        fn clifford_string_repeat_assign(
            handle: Pin<&mut CliffordStringHandle>,
            repetitions: u64,
        ) -> Result<()>;
        fn clifford_string_pow(
            handle: &CliffordStringHandle,
            exponent: i64,
        ) -> UniquePtr<CliffordStringHandle>;
        fn clifford_string_ipow(handle: Pin<&mut CliffordStringHandle>, exponent: i64);
        fn clifford_string_x_outputs(handle: &CliffordStringHandle)
        -> UniquePtr<PauliStringHandle>;
        fn clifford_string_x_signs_bit_packed(handle: &CliffordStringHandle) -> Vec<u8>;
        fn clifford_string_y_outputs(handle: &CliffordStringHandle)
        -> UniquePtr<PauliStringHandle>;
        fn clifford_string_y_signs_bit_packed(handle: &CliffordStringHandle) -> Vec<u8>;
        fn clifford_string_z_outputs(handle: &CliffordStringHandle)
        -> UniquePtr<PauliStringHandle>;
        fn clifford_string_z_signs_bit_packed(handle: &CliffordStringHandle) -> Vec<u8>;
        fn clifford_string_to_string(handle: &CliffordStringHandle) -> String;
        fn clifford_string_to_repr(handle: &CliffordStringHandle) -> String;
        fn pauli_string_clone(handle: &PauliStringHandle) -> UniquePtr<PauliStringHandle>;
        fn pauli_string_equals(left: &PauliStringHandle, right: &PauliStringHandle) -> bool;
        fn pauli_string_num_qubits(handle: &PauliStringHandle) -> usize;
        fn pauli_string_weight(handle: &PauliStringHandle) -> usize;
        fn pauli_string_get_item(handle: &PauliStringHandle, index: i64) -> Result<u8>;
        fn pauli_string_set_item(
            handle: Pin<&mut PauliStringHandle>,
            index: i64,
            new_pauli: u8,
        ) -> Result<()>;
        fn pauli_string_get_slice(
            handle: &PauliStringHandle,
            start: i64,
            step: i64,
            slice_length: i64,
        ) -> UniquePtr<PauliStringHandle>;
        fn pauli_string_random(num_qubits: usize) -> UniquePtr<PauliStringHandle>;
        fn pauli_string_iter_all(
            num_qubits: usize,
            min_weight: usize,
            max_weight: usize,
            allow_x: bool,
            allow_y: bool,
            allow_z: bool,
        ) -> UniquePtr<PauliStringIteratorHandle>;
        fn pauli_string_iterator_clone(
            handle: &PauliStringIteratorHandle,
        ) -> UniquePtr<PauliStringIteratorHandle>;
        fn pauli_string_iterator_next(
            handle: Pin<&mut PauliStringIteratorHandle>,
        ) -> UniquePtr<PauliStringHandle>;
        fn pauli_string_commutes(handle: &PauliStringHandle, other: &PauliStringHandle) -> bool;
        fn pauli_string_pauli_indices(
            handle: &PauliStringHandle,
            included_paulis: &str,
        ) -> Result<Vec<u64>>;
        fn pauli_string_sign_code(handle: &PauliStringHandle) -> i32;
        fn pauli_string_to_tableau(handle: &PauliStringHandle) -> UniquePtr<TableauHandle>;
        fn pauli_string_to_string(handle: &PauliStringHandle) -> String;
        fn pauli_string_to_repr(handle: &PauliStringHandle) -> String;
        fn tableau_num_qubits(handle: &TableauHandle) -> usize;
        fn tableau_to_string(handle: &TableauHandle) -> String;
        fn tableau_to_repr(handle: &TableauHandle) -> String;
        fn tableau_to_unitary_matrix_data(handle: &TableauHandle, endian: &str)
        -> Result<Vec<f32>>;
        fn tableau_to_state_vector_data(handle: &TableauHandle, endian: &str) -> Result<Vec<f32>>;
        fn circuit_compile_sampler(
            handle: &CircuitHandle,
            skip_reference_sample: bool,
            seed: u64,
        ) -> UniquePtr<MeasurementSamplerHandle>;
        fn measurement_sampler_num_measurements(handle: &MeasurementSamplerHandle) -> u64;
        fn measurement_sampler_sample_bit_packed(
            handle: Pin<&mut MeasurementSamplerHandle>,
            shots: u64,
        ) -> Vec<u8>;
        fn measurement_sampler_sample_write(
            handle: Pin<&mut MeasurementSamplerHandle>,
            shots: u64,
            filepath: &str,
            format_name: &str,
        ) -> Result<()>;
        fn circuit_compile_detector_sampler(
            handle: &CircuitHandle,
            seed: u64,
        ) -> UniquePtr<DetectorSamplerHandle>;
        fn detector_error_model_compile_sampler(
            handle: &DetectorErrorModelHandle,
            seed: u64,
        ) -> UniquePtr<DemSamplerHandle>;
        fn detector_sampler_num_detectors(handle: &DetectorSamplerHandle) -> u64;
        fn detector_sampler_num_observables(handle: &DetectorSamplerHandle) -> u64;
        fn dem_sampler_num_detectors(handle: &DemSamplerHandle) -> u64;
        fn dem_sampler_num_observables(handle: &DemSamplerHandle) -> u64;
        fn dem_sampler_num_errors(handle: &DemSamplerHandle) -> u64;
        fn detector_sampler_sample_bit_packed(
            handle: Pin<&mut DetectorSamplerHandle>,
            shots: u64,
        ) -> Vec<u8>;
        fn detector_sampler_sample_observables_bit_packed(
            handle: Pin<&mut DetectorSamplerHandle>,
            shots: u64,
        ) -> Vec<u8>;
        fn dem_sampler_sample_bit_packed(
            handle: Pin<&mut DemSamplerHandle>,
            shots: u64,
        ) -> DemSampleBatch;
        fn dem_sampler_sample_bit_packed_replay(
            handle: Pin<&mut DemSamplerHandle>,
            recorded_errors: &[u8],
            shots: u64,
        ) -> DemSampleBatch;
        fn dem_sampler_sample_write(
            handle: Pin<&mut DemSamplerHandle>,
            shots: u64,
            dets_filepath: &str,
            dets_format_name: &str,
            obs_filepath: &str,
            obs_format_name: &str,
            err_filepath: &str,
            err_format_name: &str,
            write_errors: bool,
            replay_err_filepath: &str,
            replay_err_format_name: &str,
            replay_errors: bool,
        ) -> Result<()>;
        fn detector_sampler_sample_write(
            handle: Pin<&mut DetectorSamplerHandle>,
            shots: u64,
            filepath: &str,
            format_name: &str,
        ) -> Result<()>;
        fn detector_sampler_sample_write_separate_observables(
            handle: Pin<&mut DetectorSamplerHandle>,
            shots: u64,
            dets_filepath: &str,
            dets_format_name: &str,
            obs_filepath: &str,
            obs_format_name: &str,
        ) -> Result<()>;
        fn circuit_compile_m2d_converter(
            handle: &CircuitHandle,
            skip_reference_sample: bool,
        ) -> UniquePtr<MeasurementsToDetectionEventsConverterHandle>;
        fn m2d_converter_num_measurements(
            handle: &MeasurementsToDetectionEventsConverterHandle,
        ) -> u64;
        fn m2d_converter_num_detectors(
            handle: &MeasurementsToDetectionEventsConverterHandle,
        ) -> u64;
        fn m2d_converter_num_observables(
            handle: &MeasurementsToDetectionEventsConverterHandle,
        ) -> u64;
        fn m2d_converter_num_sweep_bits(
            handle: &MeasurementsToDetectionEventsConverterHandle,
        ) -> u64;
        fn m2d_converter_convert_measurements_bit_packed(
            handle: Pin<&mut MeasurementsToDetectionEventsConverterHandle>,
            measurements: &[u8],
            shots: u64,
            append_observables: bool,
        ) -> Vec<u8>;
        fn m2d_converter_convert_measurements_and_sweep_bits_bit_packed(
            handle: Pin<&mut MeasurementsToDetectionEventsConverterHandle>,
            measurements: &[u8],
            sweep_bits: &[u8],
            shots: u64,
            append_observables: bool,
        ) -> Vec<u8>;
        fn m2d_converter_convert_observables_with_sweep_bits_bit_packed(
            handle: Pin<&mut MeasurementsToDetectionEventsConverterHandle>,
            measurements: &[u8],
            sweep_bits: &[u8],
            shots: u64,
        ) -> Vec<u8>;
        fn m2d_converter_convert_observables_bit_packed(
            handle: Pin<&mut MeasurementsToDetectionEventsConverterHandle>,
            measurements: &[u8],
            shots: u64,
        ) -> Vec<u8>;
        fn m2d_converter_convert_file(
            handle: Pin<&mut MeasurementsToDetectionEventsConverterHandle>,
            measurements_filepath: &str,
            measurements_format: &str,
            sweep_bits_filepath: &str,
            sweep_bits_format: &str,
            detection_events_filepath: &str,
            detection_events_format: &str,
            append_observables: bool,
            obs_out_filepath: &str,
            obs_out_format: &str,
        ) -> Result<()>;

        fn new_smoke_probe() -> UniquePtr<SmokeProbe>;
        fn describe(self: &SmokeProbe) -> String;
        fn weighted_checksum(self: &SmokeProbe, values: &[u64]) -> u64;
    }
}
