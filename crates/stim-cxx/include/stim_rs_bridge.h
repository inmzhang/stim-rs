#pragma once

#include "rust/cxx.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

#include "stim/circuit/circuit.h"
#include "stim/dem/detector_error_model.h"
#include "stim/gates/gates.h"
#include "stim/mem/simd_bits.h"
#include "stim/simulators/dem_sampler.h"
#include "stim/simulators/frame_simulator.h"
#include "stim/simulators/tableau_simulator.h"
#include "stim/stabilizers/clifford_string.h"
#include "stim/stabilizers/flow.h"
#include "stim/stabilizers/pauli_string.h"
#include "stim/stabilizers/pauli_string_iter.h"
#include "stim/stabilizers/tableau.h"
#include "stim/stabilizers/tableau_iter.h"
#include "stim/util_top/circuit_flow_generators.h"
#include "stim/util_top/has_flow.h"

namespace stimrs::bridge {

struct GateTargetWithCoordsData;
struct DemTargetWithCoordsData;
struct CircuitErrorLocationStackFrameData;
struct CircuitTargetsInsideInstructionData;
struct FlippedMeasurementData;
struct CircuitErrorLocationData;
struct ExplainedErrorData;
struct DemSampleBatch;
struct DetectorSampleBatch;
struct CircuitTopLevelItemData;
struct DemTopLevelItemData;
struct TableauMeasureKickbackData;
struct BitTableData;

class CircuitHandle final {
 public:
  CircuitHandle();
  explicit CircuitHandle(stim::Circuit circuit);

  stim::Circuit &get();
  const stim::Circuit &get() const;

 private:
  stim::Circuit circuit_;
};

class DetectorErrorModelHandle final {
 public:
  DetectorErrorModelHandle();
  explicit DetectorErrorModelHandle(stim::DetectorErrorModel detector_error_model);

  stim::DetectorErrorModel &get();
  const stim::DetectorErrorModel &get() const;

 private:
  stim::DetectorErrorModel detector_error_model_;
};

class TableauHandle final {
 public:
  TableauHandle();
  explicit TableauHandle(stim::Tableau<stim::MAX_BITWORD_WIDTH> tableau);

  stim::Tableau<stim::MAX_BITWORD_WIDTH> &get();
  const stim::Tableau<stim::MAX_BITWORD_WIDTH> &get() const;

 private:
  stim::Tableau<stim::MAX_BITWORD_WIDTH> tableau_;
};

class PauliStringHandle final {
 public:
  PauliStringHandle();
  explicit PauliStringHandle(stim::PauliString<stim::MAX_BITWORD_WIDTH> pauli_string);

  stim::PauliString<stim::MAX_BITWORD_WIDTH> &get();
  const stim::PauliString<stim::MAX_BITWORD_WIDTH> &get() const;

 private:
  stim::PauliString<stim::MAX_BITWORD_WIDTH> pauli_string_;
};

class CliffordStringHandle final {
 public:
  CliffordStringHandle();
  explicit CliffordStringHandle(stim::CliffordString<stim::MAX_BITWORD_WIDTH> clifford_string);

  stim::CliffordString<stim::MAX_BITWORD_WIDTH> &get();
  const stim::CliffordString<stim::MAX_BITWORD_WIDTH> &get() const;

 private:
  stim::CliffordString<stim::MAX_BITWORD_WIDTH> clifford_string_;
};

class PauliStringIteratorHandle final {
 public:
  PauliStringIteratorHandle(
      std::size_t num_qubits,
      std::size_t min_weight,
      std::size_t max_weight,
      bool allow_x,
      bool allow_y,
      bool allow_z);
  explicit PauliStringIteratorHandle(stim::PauliStringIterator<stim::MAX_BITWORD_WIDTH> iterator);

  stim::PauliStringIterator<stim::MAX_BITWORD_WIDTH> &get();
  const stim::PauliStringIterator<stim::MAX_BITWORD_WIDTH> &get() const;

 private:
  stim::PauliStringIterator<stim::MAX_BITWORD_WIDTH> iterator_;
};

class TableauIteratorHandle final {
 public:
  TableauIteratorHandle(std::size_t num_qubits, bool unsigned_only);
  explicit TableauIteratorHandle(stim::TableauIterator<stim::MAX_BITWORD_WIDTH> iterator);

  stim::TableauIterator<stim::MAX_BITWORD_WIDTH> &get();
  const stim::TableauIterator<stim::MAX_BITWORD_WIDTH> &get() const;

 private:
  stim::TableauIterator<stim::MAX_BITWORD_WIDTH> iterator_;
};

class TableauSimulatorHandle final {
 public:
  TableauSimulatorHandle(std::uint64_t seed, std::size_t num_qubits);
  explicit TableauSimulatorHandle(stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> simulator);

  stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> &get();
  const stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> &get() const;

 private:
  stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> simulator_;
};

class FrameSimulatorHandle final {
 public:
  FrameSimulatorHandle(
      std::size_t batch_size,
      bool disable_stabilizer_randomization,
      std::size_t num_qubits,
      std::uint64_t seed);
  explicit FrameSimulatorHandle(stim::FrameSimulator<stim::MAX_BITWORD_WIDTH> simulator);

  stim::FrameSimulator<stim::MAX_BITWORD_WIDTH> &get();
  const stim::FrameSimulator<stim::MAX_BITWORD_WIDTH> &get() const;

 private:
  stim::FrameSimulator<stim::MAX_BITWORD_WIDTH> simulator_;
};

class MeasurementSamplerHandle final {
 public:
  MeasurementSamplerHandle(
      stim::simd_bits<stim::MAX_BITWORD_WIDTH> reference_sample,
      stim::Circuit circuit,
      bool skip_reference_sample,
      std::uint64_t seed);

  std::uint64_t num_measurements() const;
  rust::Vec<std::uint8_t> sample_bit_packed(std::uint64_t shots);
  void sample_write(
      std::uint64_t shots,
      rust::Str filepath,
      rust::Str format_name);

 private:
  stim::simd_bits<stim::MAX_BITWORD_WIDTH> reference_sample_;
  stim::Circuit circuit_;
  bool skip_reference_sample_;
  std::mt19937_64 rng_;
};

class DetectorSamplerHandle final {
 public:
  DetectorSamplerHandle(stim::Circuit circuit, std::uint64_t seed);

  std::uint64_t num_detectors() const;
  std::uint64_t num_observables() const;
  rust::Vec<std::uint8_t> sample_bit_packed(std::uint64_t shots);
  rust::Vec<std::uint8_t> sample_observables_bit_packed(std::uint64_t shots);
  DetectorSampleBatch sample_bit_packed_separate_observables(std::uint64_t shots);
  void sample_write(
      std::uint64_t shots,
      rust::Str filepath,
      rust::Str format_name);
  void sample_write_separate_observables(
      std::uint64_t shots,
      rust::Str dets_filepath,
      rust::Str dets_format_name,
      rust::Str obs_filepath,
      rust::Str obs_format_name);

 private:
  stim::Circuit circuit_;
  std::mt19937_64 rng_;
};

class MeasurementsToDetectionEventsConverterHandle final {
 public:
  MeasurementsToDetectionEventsConverterHandle(
      stim::simd_bits<stim::MAX_BITWORD_WIDTH> reference_sample,
      stim::Circuit circuit,
      bool skip_reference_sample);

  std::uint64_t num_measurements() const;
  std::uint64_t num_detectors() const;
  std::uint64_t num_observables() const;
  std::uint64_t num_sweep_bits() const;
  rust::Vec<std::uint8_t> convert_measurements_bit_packed(
      rust::Slice<const std::uint8_t> measurements,
      std::uint64_t shots,
      bool append_observables);
  rust::Vec<std::uint8_t> convert_measurements_and_sweep_bits_bit_packed(
      rust::Slice<const std::uint8_t> measurements,
      rust::Slice<const std::uint8_t> sweep_bits,
      std::uint64_t shots,
      bool append_observables);
  rust::Vec<std::uint8_t> convert_observables_with_sweep_bits_bit_packed(
      rust::Slice<const std::uint8_t> measurements,
      rust::Slice<const std::uint8_t> sweep_bits,
      std::uint64_t shots);
  rust::Vec<std::uint8_t> convert_observables_bit_packed(
      rust::Slice<const std::uint8_t> measurements,
      std::uint64_t shots);
  void convert_file(
      rust::Str measurements_filepath,
      rust::Str measurements_format,
      rust::Str sweep_bits_filepath,
      rust::Str sweep_bits_format,
      rust::Str detection_events_filepath,
      rust::Str detection_events_format,
      bool append_observables,
      rust::Str obs_out_filepath,
      rust::Str obs_out_format);

 private:
  bool skip_reference_sample_;
  stim::simd_bits<stim::MAX_BITWORD_WIDTH> reference_sample_;
  stim::CircuitStats circuit_stats_;
  stim::Circuit circuit_;
};

class DemSamplerHandle final {
 public:
  DemSamplerHandle(stim::DetectorErrorModel detector_error_model, std::uint64_t seed);

  std::uint64_t num_detectors() const;
  std::uint64_t num_observables() const;
  std::uint64_t num_errors() const;
  DemSampleBatch sample_bit_packed(std::uint64_t shots);
  DemSampleBatch sample_bit_packed_replay(
      rust::Slice<const std::uint8_t> recorded_errors,
      std::uint64_t shots);
  void sample_write(
      std::uint64_t shots,
      FILE *det_out,
      stim::SampleFormat det_out_format,
      FILE *obs_out,
      stim::SampleFormat obs_out_format,
      FILE *err_out,
      stim::SampleFormat err_out_format,
      FILE *replay_err_in,
      stim::SampleFormat replay_err_in_format);

 private:
  stim::DemSampler<stim::MAX_BITWORD_WIDTH> sampler_;
};

class GateDataHandle final {
 public:
  explicit GateDataHandle(stim::GateType type);

  rust::String name() const;
  rust::Vec<rust::String> aliases() const;
  rust::Vec<std::uint8_t> num_parens_arguments_range() const;
  bool is_noisy_gate() const;
  bool is_reset() const;
  bool is_single_qubit_gate() const;
  bool is_symmetric_gate() const;
  bool is_two_qubit_gate() const;
  bool is_unitary() const;
  bool produces_measurements() const;
  bool takes_measurement_record_targets() const;
  bool takes_pauli_targets() const;
  rust::Vec<rust::String> flows() const;
  std::unique_ptr<TableauHandle> tableau() const;
  std::unique_ptr<GateDataHandle> clone_handle() const;
  std::unique_ptr<GateDataHandle> inverse() const;
  std::unique_ptr<GateDataHandle> generalized_inverse() const;
  std::unique_ptr<GateDataHandle> hadamard_conjugated(bool unsigned_only) const;

 private:
  stim::GateType type_;
};

rust::Vec<std::uint8_t> read_shot_data_file_bit_packed(
    rust::Str filepath,
    rust::Str format_name,
    std::uint64_t num_measurements,
    std::uint64_t num_detectors,
    std::uint64_t num_observables);
void write_shot_data_file_bit_packed(
    rust::Slice<const std::uint8_t> data,
    std::uint64_t shots,
    rust::Str filepath,
    rust::Str format_name,
    std::uint64_t num_measurements,
    std::uint64_t num_detectors,
    std::uint64_t num_observables);
rust::Vec<rust::String> all_gate_names();
std::unique_ptr<GateDataHandle> gate_data_by_name(rust::Str name);
rust::String canonicalize_flow_text(rust::Str text);
rust::String multiply_flow_texts(rust::Str left, rust::Str right);

std::unique_ptr<CircuitHandle> new_circuit();
std::unique_ptr<CircuitHandle> circuit_from_stim_program_text(rust::Str text);
std::unique_ptr<CircuitHandle> circuit_clone(const CircuitHandle &handle);
std::unique_ptr<CircuitHandle> circuit_add(
    const CircuitHandle &left,
    const CircuitHandle &right);
void circuit_add_assign(CircuitHandle &left, const CircuitHandle &right);
std::unique_ptr<CircuitHandle> circuit_mul(
    const CircuitHandle &handle,
    std::uint64_t repetitions);
void circuit_mul_assign(CircuitHandle &handle, std::uint64_t repetitions);
CircuitTopLevelItemData circuit_get_top_level_item(const CircuitHandle &handle, std::size_t index);
std::unique_ptr<CircuitHandle> circuit_get_top_level_repeat_block_body(
    const CircuitHandle &handle,
    std::size_t index);
std::unique_ptr<CircuitHandle> circuit_get_slice(
    const CircuitHandle &handle,
    std::int64_t start,
    std::int64_t step,
    std::int64_t slice_length);
void circuit_remove_top_level(CircuitHandle &handle, std::size_t index);
std::unique_ptr<CircuitHandle> circuit_generated(
    rust::Str code_task,
    std::size_t distance,
    std::size_t rounds,
    double after_clifford_depolarization,
    double before_round_data_depolarization,
    double before_measure_flip_probability,
    double after_reset_flip_probability);
bool circuit_has_flow_text(
    const CircuitHandle &handle,
    rust::Str flow_text,
    bool unsigned_only);
bool circuit_has_all_flows_text(
    const CircuitHandle &handle,
    rust::Vec<rust::String> flow_texts,
    bool unsigned_only);
rust::Vec<rust::String> circuit_flow_generators_texts(const CircuitHandle &handle);
rust::Vec<rust::String> circuit_solve_flow_measurements_text(
    const CircuitHandle &handle,
    rust::Vec<rust::String> flow_texts);
rust::String circuit_time_reversed_for_flows_text(
    rust::Vec<rust::String> &out_flow_texts,
    const CircuitHandle &handle,
    rust::Vec<rust::String> flow_texts,
    bool dont_turn_measurements_into_resets);
rust::String circuit_diagram_with_options(
    const CircuitHandle &handle,
    rust::Str type_name,
    bool has_tick_range,
    std::uint64_t tick_start,
    std::uint64_t tick_count,
    bool has_rows,
    std::size_t rows);
rust::String circuit_diagram_with_options_and_filters(
    const CircuitHandle &handle,
    rust::Str type_name,
    bool has_tick_range,
    std::uint64_t tick_start,
    std::uint64_t tick_count,
    bool has_rows,
    std::size_t rows,
    rust::Vec<rust::String> filter_coords);
rust::String circuit_to_stim_program_text(const CircuitHandle &handle);
rust::Vec<rust::String> circuit_flattened_operation_texts(const CircuitHandle &handle);
std::size_t circuit_num_qubits(const CircuitHandle &handle);
std::size_t circuit_len(const CircuitHandle &handle);
std::uint64_t circuit_num_measurements(const CircuitHandle &handle);
std::uint64_t circuit_count_determined_measurements(
    const CircuitHandle &handle,
    bool unknown_input);
std::uint64_t circuit_num_detectors(const CircuitHandle &handle);
std::uint64_t circuit_num_observables(const CircuitHandle &handle);
std::uint64_t circuit_num_ticks(const CircuitHandle &handle);
std::size_t circuit_num_sweep_bits(const CircuitHandle &handle);
void circuit_append_from_stim_program_text(CircuitHandle &handle, rust::Str text);
void circuit_append_gate(
    CircuitHandle &handle,
    rust::Str gate_name,
    rust::Slice<const std::uint32_t> targets,
    rust::Slice<const double> args,
    rust::Str tag);
void circuit_append_repeat_block(
    CircuitHandle &handle,
    std::uint64_t repeat_count,
    const CircuitHandle &body,
    rust::Str tag);
void circuit_insert_gate(
    CircuitHandle &handle,
    std::size_t index,
    rust::Str gate_name,
    rust::Slice<const std::uint32_t> targets,
    rust::Slice<const double> args,
    rust::Str tag);
void circuit_insert_repeat_block(
    CircuitHandle &handle,
    std::size_t index,
    std::uint64_t repeat_count,
    const CircuitHandle &body,
    rust::Str tag);
void circuit_insert_circuit(
    CircuitHandle &handle,
    std::size_t index,
    const CircuitHandle &circuit);
void circuit_clear(CircuitHandle &handle);
bool circuit_equals(const CircuitHandle &left, const CircuitHandle &right);
bool circuit_approx_equals(
    const CircuitHandle &left,
    const CircuitHandle &right,
    double atol);
std::unique_ptr<CircuitHandle> circuit_without_noise(const CircuitHandle &handle);
std::unique_ptr<CircuitHandle> circuit_with_inlined_feedback(const CircuitHandle &handle);
std::unique_ptr<CircuitHandle> circuit_without_tags(const CircuitHandle &handle);
std::unique_ptr<CircuitHandle> circuit_flattened(const CircuitHandle &handle);
std::unique_ptr<CircuitHandle> circuit_decomposed(const CircuitHandle &handle);
std::unique_ptr<CircuitHandle> circuit_inverse(const CircuitHandle &handle);
rust::String circuit_to_qasm(
    const CircuitHandle &handle,
    int open_qasm_version,
    bool skip_dets_and_obs);
rust::String circuit_to_quirk_url(const CircuitHandle &handle);
rust::String circuit_to_crumble_url(
    const CircuitHandle &handle,
    bool skip_detectors);
rust::String circuit_shortest_error_sat_problem(
    const CircuitHandle &handle,
    rust::Str format_name);
rust::String circuit_likeliest_error_sat_problem(
    const CircuitHandle &handle,
    std::int32_t quantization,
    rust::Str format_name);
rust::String circuit_detecting_regions_text(const CircuitHandle &handle);
rust::String circuit_detecting_regions_text_with_options(
    const CircuitHandle &handle,
    rust::Vec<rust::String> target_texts,
    rust::Vec<std::uint64_t> ticks,
    bool ignore_anticommutation_errors);
rust::Vec<ExplainedErrorData> circuit_explain_detector_error_model_errors(
    const CircuitHandle &handle,
    rust::Str dem_filter_text,
    bool has_dem_filter,
    bool reduce_to_one_representative_error);
rust::Vec<ExplainedErrorData> circuit_shortest_graphlike_error(
    const CircuitHandle &handle,
    bool ignore_ungraphlike_errors,
    bool canonicalize_circuit_errors);
rust::Vec<ExplainedErrorData> circuit_search_for_undetectable_logical_errors(
    const CircuitHandle &handle,
    std::uint64_t dont_explore_detection_event_sets_with_size_above,
    std::uint64_t dont_explore_edges_with_degree_above,
    bool dont_explore_edges_increasing_symptom_degree,
    bool canonicalize_circuit_errors);
rust::String circuit_get_detector_coordinates_text(
    const CircuitHandle &handle,
    rust::Slice<const std::uint64_t> included_detector_indices);
rust::String circuit_get_final_qubit_coordinates_text(const CircuitHandle &handle);
rust::Vec<std::uint8_t> circuit_reference_sample_bit_packed(const CircuitHandle &handle);
rust::Vec<std::uint8_t> circuit_reference_detector_signs_bit_packed(const CircuitHandle &handle);
rust::Vec<std::uint8_t> circuit_reference_observable_signs_bit_packed(const CircuitHandle &handle);
std::unique_ptr<DetectorErrorModelHandle> new_detector_error_model();
std::unique_ptr<DetectorErrorModelHandle> circuit_detector_error_model(
    const CircuitHandle &handle,
    bool decompose_errors,
    bool flatten_loops,
    bool allow_gauge_detectors,
    double approximate_disjoint_errors,
    bool ignore_decomposition_failures,
    bool block_decomposition_from_introducing_remnant_edges);
std::unique_ptr<CircuitHandle> circuit_missing_detectors(
    const CircuitHandle &handle,
    bool unknown_input);
std::unique_ptr<TableauHandle> circuit_to_tableau(
    const CircuitHandle &handle,
    bool ignore_noise,
    bool ignore_measurement,
    bool ignore_reset);
std::unique_ptr<TableauHandle> new_tableau(std::size_t num_qubits);
std::unique_ptr<TableauHandle> tableau_random(std::size_t num_qubits);
std::unique_ptr<TableauIteratorHandle> tableau_iter_all(
    std::size_t num_qubits,
    bool unsigned_only);
std::unique_ptr<TableauIteratorHandle> tableau_iterator_clone(const TableauIteratorHandle &handle);
std::unique_ptr<TableauHandle> tableau_iterator_next(TableauIteratorHandle &handle);
std::unique_ptr<TableauHandle> tableau_from_named_gate(rust::Str name);
std::unique_ptr<TableauHandle> tableau_from_state_vector_data(
    rust::Vec<float> state_vector,
    rust::Str endian);
std::unique_ptr<TableauHandle> tableau_from_unitary_matrix_data(
    rust::Vec<float> matrix,
    rust::Str endian);
std::unique_ptr<TableauHandle> tableau_from_conjugated_generator_texts(
    rust::Vec<rust::String> xs,
    rust::Vec<rust::String> zs);
std::unique_ptr<TableauHandle> tableau_from_stabilizer_texts(
    rust::Vec<rust::String> stabilizers,
    bool allow_redundant,
    bool allow_underconstrained);
std::unique_ptr<TableauHandle> tableau_then(
    const TableauHandle &handle,
    const TableauHandle &second);
std::unique_ptr<TableauSimulatorHandle> new_tableau_simulator(
    std::uint64_t seed,
    std::size_t num_qubits);
std::unique_ptr<CliffordStringHandle> new_clifford_string(std::size_t num_qubits);
std::unique_ptr<CliffordStringHandle> clifford_string_from_text(rust::Str text);
std::unique_ptr<CliffordStringHandle> clifford_string_from_pauli_string(
    const PauliStringHandle &handle);
std::unique_ptr<CliffordStringHandle> clifford_string_from_circuit(const CircuitHandle &handle);
std::unique_ptr<CliffordStringHandle> clifford_string_random(std::size_t num_qubits);
std::unique_ptr<CliffordStringHandle> clifford_string_all_cliffords_string();
std::unique_ptr<PauliStringHandle> pauli_string_from_text(rust::Str text);
std::unique_ptr<PauliStringHandle> new_pauli_string(std::size_t num_qubits);
std::unique_ptr<DetectorErrorModelHandle> detector_error_model_from_dem_text(rust::Str text);
rust::String circuit_diagram(
    const CircuitHandle &handle,
    rust::Str type_name);
rust::String detector_error_model_diagram(
    const DetectorErrorModelHandle &handle,
    rust::Str type_name);
std::unique_ptr<DetectorErrorModelHandle> detector_error_model_clone(
    const DetectorErrorModelHandle &handle);
std::unique_ptr<DetectorErrorModelHandle> detector_error_model_add(
    const DetectorErrorModelHandle &left,
    const DetectorErrorModelHandle &right);
void detector_error_model_add_assign(
    DetectorErrorModelHandle &left,
    const DetectorErrorModelHandle &right);
std::unique_ptr<DetectorErrorModelHandle> detector_error_model_mul(
    const DetectorErrorModelHandle &handle,
    std::uint64_t repetitions);
void detector_error_model_mul_assign(
    DetectorErrorModelHandle &handle,
    std::uint64_t repetitions);
DemTopLevelItemData detector_error_model_get_top_level_item(
    const DetectorErrorModelHandle &handle,
    std::size_t index);
std::unique_ptr<DetectorErrorModelHandle> detector_error_model_get_top_level_repeat_block_body(
    const DetectorErrorModelHandle &handle,
    std::size_t index);
std::unique_ptr<DetectorErrorModelHandle> detector_error_model_get_slice(
    const DetectorErrorModelHandle &handle,
    std::int64_t start,
    std::int64_t step,
    std::int64_t slice_length);
rust::String detector_error_model_to_dem_text(const DetectorErrorModelHandle &handle);
std::size_t detector_error_model_len(const DetectorErrorModelHandle &handle);
std::uint64_t detector_error_model_num_detectors(const DetectorErrorModelHandle &handle);
std::uint64_t detector_error_model_num_errors(const DetectorErrorModelHandle &handle);
std::uint64_t detector_error_model_num_observables(const DetectorErrorModelHandle &handle);
rust::String detector_error_model_get_detector_coordinates_text(
    const DetectorErrorModelHandle &handle,
    rust::Slice<const std::uint64_t> included_detector_indices);
void detector_error_model_clear(DetectorErrorModelHandle &handle);
bool detector_error_model_equals(
    const DetectorErrorModelHandle &left,
    const DetectorErrorModelHandle &right);
bool detector_error_model_approx_equals(
    const DetectorErrorModelHandle &left,
    const DetectorErrorModelHandle &right,
    double atol);
std::unique_ptr<DetectorErrorModelHandle> detector_error_model_without_tags(
    const DetectorErrorModelHandle &handle);
std::unique_ptr<DetectorErrorModelHandle> detector_error_model_flattened(
    const DetectorErrorModelHandle &handle);
std::unique_ptr<DetectorErrorModelHandle> detector_error_model_rounded(
    const DetectorErrorModelHandle &handle,
    std::uint8_t digits);
void detector_error_model_append_instruction(
    DetectorErrorModelHandle &handle,
    rust::Str instruction_type,
    rust::Slice<const double> args,
    rust::Slice<const std::uint64_t> targets,
    rust::Str tag);
void detector_error_model_append_repeat_block(
    DetectorErrorModelHandle &handle,
    std::uint64_t repeat_count,
    const DetectorErrorModelHandle &body);
std::unique_ptr<DetectorErrorModelHandle> detector_error_model_shortest_graphlike_error(
    const DetectorErrorModelHandle &handle,
    bool ignore_ungraphlike_errors);
rust::String detector_error_model_shortest_error_sat_problem(
    const DetectorErrorModelHandle &handle,
    rust::Str format_name);
rust::String detector_error_model_likeliest_error_sat_problem(
    const DetectorErrorModelHandle &handle,
    std::int32_t quantization,
    rust::Str format_name);
std::unique_ptr<TableauHandle> tableau_clone(const TableauHandle &handle);
bool tableau_equals(const TableauHandle &left, const TableauHandle &right);
std::unique_ptr<TableauHandle> tableau_add(
    const TableauHandle &left,
    const TableauHandle &right);
void tableau_add_assign(TableauHandle &left, const TableauHandle &right);
void tableau_append(
    TableauHandle &handle,
    const TableauHandle &gate,
    rust::Slice<const std::size_t> targets);
void tableau_prepend(
    TableauHandle &handle,
    const TableauHandle &gate,
    rust::Slice<const std::size_t> targets);
std::unique_ptr<TableauHandle> tableau_inverse(const TableauHandle &handle, bool unsigned_only);
std::unique_ptr<TableauHandle> tableau_raised_to(const TableauHandle &handle, std::int64_t exponent);
int tableau_x_sign(const TableauHandle &handle, std::size_t target);
int tableau_y_sign(const TableauHandle &handle, std::size_t target);
int tableau_z_sign(const TableauHandle &handle, std::size_t target);
std::uint8_t tableau_x_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index);
std::uint8_t tableau_y_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index);
std::uint8_t tableau_z_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index);
std::uint8_t tableau_inverse_x_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index);
std::uint8_t tableau_inverse_y_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index);
std::uint8_t tableau_inverse_z_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index);
std::unique_ptr<PauliStringHandle> tableau_x_output(
    const TableauHandle &handle,
    std::size_t target);
std::unique_ptr<PauliStringHandle> tableau_y_output(
    const TableauHandle &handle,
    std::size_t target);
std::unique_ptr<PauliStringHandle> tableau_z_output(
    const TableauHandle &handle,
    std::size_t target);
std::unique_ptr<PauliStringHandle> tableau_inverse_x_output(
    const TableauHandle &handle,
    std::size_t target,
    bool unsigned_only);
std::unique_ptr<PauliStringHandle> tableau_inverse_y_output(
    const TableauHandle &handle,
    std::size_t target,
    bool unsigned_only);
std::unique_ptr<PauliStringHandle> tableau_inverse_z_output(
    const TableauHandle &handle,
    std::size_t target,
    bool unsigned_only);
std::unique_ptr<PauliStringHandle> tableau_conjugate_pauli_string(
    const TableauHandle &handle,
    const PauliStringHandle &pauli_string);
std::unique_ptr<PauliStringHandle> tableau_conjugate_pauli_string_within(
    const TableauHandle &handle,
    const PauliStringHandle &pauli_string,
    rust::Slice<const std::size_t> targets,
    bool inverse);
rust::Vec<rust::String> tableau_to_stabilizer_texts(
    const TableauHandle &handle,
    bool canonicalize);
std::unique_ptr<CircuitHandle> tableau_to_circuit(
    const TableauHandle &handle,
    rust::Str method);
std::unique_ptr<PauliStringHandle> tableau_to_pauli_string(const TableauHandle &handle);
std::unique_ptr<TableauSimulatorHandle> tableau_simulator_clone(const TableauSimulatorHandle &handle);
std::size_t tableau_simulator_num_qubits(const TableauSimulatorHandle &handle);
void tableau_simulator_set_num_qubits(TableauSimulatorHandle &handle, std::size_t new_num_qubits);
std::unique_ptr<TableauHandle> tableau_simulator_current_inverse_tableau(
    const TableauSimulatorHandle &handle);
void tableau_simulator_set_inverse_tableau(
    TableauSimulatorHandle &handle,
    const TableauHandle &tableau);
rust::Vec<std::uint8_t> tableau_simulator_current_measurement_record(
    const TableauSimulatorHandle &handle);
void tableau_simulator_do_circuit(TableauSimulatorHandle &handle, const CircuitHandle &circuit);
void tableau_simulator_do_pauli_string(
    TableauSimulatorHandle &handle,
    const PauliStringHandle &pauli_string);
void tableau_simulator_do_tableau(
    TableauSimulatorHandle &handle,
    const TableauHandle &tableau,
    rust::Slice<const std::size_t> targets);
std::unique_ptr<PauliStringHandle> tableau_simulator_peek_bloch(
    TableauSimulatorHandle &handle,
    std::size_t target);
int tableau_simulator_peek_x(TableauSimulatorHandle &handle, std::size_t target);
int tableau_simulator_peek_y(TableauSimulatorHandle &handle, std::size_t target);
int tableau_simulator_peek_z(TableauSimulatorHandle &handle, std::size_t target);
bool tableau_simulator_measure(TableauSimulatorHandle &handle, std::size_t target);
rust::Vec<std::uint8_t> tableau_simulator_measure_many(
    TableauSimulatorHandle &handle,
    rust::Slice<const std::size_t> targets);
int tableau_simulator_peek_observable_expectation(
    const TableauSimulatorHandle &handle,
    const PauliStringHandle &observable);
bool tableau_simulator_measure_observable(
    TableauSimulatorHandle &handle,
    const PauliStringHandle &observable,
    double flip_probability);
void tableau_simulator_postselect_observable(
    TableauSimulatorHandle &handle,
    const PauliStringHandle &observable,
    bool desired_value);
void tableau_simulator_postselect_x(
    TableauSimulatorHandle &handle,
    rust::Slice<const std::size_t> targets,
    bool desired_value);
void tableau_simulator_postselect_y(
    TableauSimulatorHandle &handle,
    rust::Slice<const std::size_t> targets,
    bool desired_value);
void tableau_simulator_postselect_z(
    TableauSimulatorHandle &handle,
    rust::Slice<const std::size_t> targets,
    bool desired_value);
TableauMeasureKickbackData tableau_simulator_measure_kickback(
    TableauSimulatorHandle &handle,
    std::size_t target);
std::unique_ptr<FrameSimulatorHandle> new_frame_simulator(
    std::size_t batch_size,
    bool disable_stabilizer_randomization,
    std::size_t num_qubits,
    std::uint64_t seed);
std::unique_ptr<FrameSimulatorHandle> frame_simulator_clone(const FrameSimulatorHandle &handle);
std::size_t frame_simulator_batch_size(const FrameSimulatorHandle &handle);
std::size_t frame_simulator_num_qubits(const FrameSimulatorHandle &handle);
std::size_t frame_simulator_num_measurements(const FrameSimulatorHandle &handle);
std::size_t frame_simulator_num_detectors(const FrameSimulatorHandle &handle);
std::size_t frame_simulator_num_observables(const FrameSimulatorHandle &handle);
void frame_simulator_clear(FrameSimulatorHandle &handle);
void frame_simulator_do_circuit(FrameSimulatorHandle &handle, const CircuitHandle &circuit);
void frame_simulator_set_pauli_flip(
    FrameSimulatorHandle &handle,
    std::uint8_t pauli,
    std::int64_t qubit_index,
    std::int64_t instance_index);
rust::Vec<rust::String> frame_simulator_peek_pauli_flips(const FrameSimulatorHandle &handle);
rust::String frame_simulator_peek_pauli_flip(
    const FrameSimulatorHandle &handle,
    std::int64_t instance_index);
void frame_simulator_broadcast_pauli_errors(
    FrameSimulatorHandle &handle,
    std::uint8_t pauli,
    rust::Vec<std::uint8_t> mask,
    std::size_t mask_num_qubits,
    float p);
rust::Vec<std::uint8_t> frame_simulator_generate_bernoulli_samples(
    FrameSimulatorHandle &handle,
    std::size_t num_samples,
    float p,
    bool bit_packed);
void frame_simulator_append_measurement_flips(
    FrameSimulatorHandle &handle,
    rust::Vec<std::uint8_t> data,
    std::size_t num_measurements,
    bool bit_packed);
BitTableData frame_simulator_get_measurement_flips(
    const FrameSimulatorHandle &handle,
    bool bit_packed);
BitTableData frame_simulator_get_detector_flips(
    const FrameSimulatorHandle &handle,
    bool bit_packed);
BitTableData frame_simulator_get_observable_flips(
    const FrameSimulatorHandle &handle,
    bool bit_packed);
std::unique_ptr<CliffordStringHandle> clifford_string_clone(const CliffordStringHandle &handle);
bool clifford_string_equals(const CliffordStringHandle &left, const CliffordStringHandle &right);
std::size_t clifford_string_num_qubits(const CliffordStringHandle &handle);
rust::String clifford_string_get_item_name(const CliffordStringHandle &handle, std::int64_t index);
std::unique_ptr<CliffordStringHandle> clifford_string_get_slice(
    const CliffordStringHandle &handle,
    std::int64_t start,
    std::int64_t step,
    std::int64_t slice_length);
std::unique_ptr<CliffordStringHandle> clifford_string_add(
    const CliffordStringHandle &left,
    const CliffordStringHandle &right);
void clifford_string_add_assign(CliffordStringHandle &left, const CliffordStringHandle &right);
std::unique_ptr<CliffordStringHandle> clifford_string_mul(
    const CliffordStringHandle &left,
    const CliffordStringHandle &right);
void clifford_string_mul_assign(CliffordStringHandle &left, const CliffordStringHandle &right);
std::unique_ptr<CliffordStringHandle> clifford_string_repeat(
    const CliffordStringHandle &handle,
    std::uint64_t repetitions);
void clifford_string_repeat_assign(CliffordStringHandle &handle, std::uint64_t repetitions);
std::unique_ptr<CliffordStringHandle> clifford_string_pow(
    const CliffordStringHandle &handle,
    std::int64_t exponent);
void clifford_string_ipow(CliffordStringHandle &handle, std::int64_t exponent);
std::unique_ptr<PauliStringHandle> clifford_string_x_outputs(const CliffordStringHandle &handle);
rust::Vec<std::uint8_t> clifford_string_x_signs_bit_packed(const CliffordStringHandle &handle);
std::unique_ptr<PauliStringHandle> clifford_string_y_outputs(const CliffordStringHandle &handle);
rust::Vec<std::uint8_t> clifford_string_y_signs_bit_packed(const CliffordStringHandle &handle);
std::unique_ptr<PauliStringHandle> clifford_string_z_outputs(const CliffordStringHandle &handle);
rust::Vec<std::uint8_t> clifford_string_z_signs_bit_packed(const CliffordStringHandle &handle);
rust::String clifford_string_to_string(const CliffordStringHandle &handle);
rust::String clifford_string_to_repr(const CliffordStringHandle &handle);
std::unique_ptr<PauliStringHandle> pauli_string_clone(const PauliStringHandle &handle);
bool pauli_string_equals(const PauliStringHandle &left, const PauliStringHandle &right);
std::size_t pauli_string_num_qubits(const PauliStringHandle &handle);
std::size_t pauli_string_weight(const PauliStringHandle &handle);
std::uint8_t pauli_string_get_item(const PauliStringHandle &handle, std::int64_t index);
void pauli_string_set_item(PauliStringHandle &handle, std::int64_t index, std::uint8_t new_pauli);
std::unique_ptr<PauliStringHandle> pauli_string_get_slice(
    const PauliStringHandle &handle,
    std::int64_t start,
    std::int64_t step,
    std::int64_t slice_length);
std::unique_ptr<PauliStringHandle> pauli_string_random(std::size_t num_qubits);
std::unique_ptr<PauliStringIteratorHandle> pauli_string_iter_all(
    std::size_t num_qubits,
    std::size_t min_weight,
    std::size_t max_weight,
    bool allow_x,
    bool allow_y,
    bool allow_z);
std::unique_ptr<PauliStringIteratorHandle> pauli_string_iterator_clone(const PauliStringIteratorHandle &handle);
std::unique_ptr<PauliStringHandle> pauli_string_iterator_next(PauliStringIteratorHandle &handle);
bool pauli_string_commutes(
    const PauliStringHandle &handle,
    const PauliStringHandle &other);
rust::Vec<std::uint64_t> pauli_string_pauli_indices(
    const PauliStringHandle &handle,
    rust::Str included_paulis);
int pauli_string_sign_code(const PauliStringHandle &handle);
std::unique_ptr<TableauHandle> pauli_string_to_tableau(const PauliStringHandle &handle);
rust::String pauli_string_to_string(const PauliStringHandle &handle);
rust::String pauli_string_to_repr(const PauliStringHandle &handle);
std::size_t tableau_num_qubits(const TableauHandle &handle);
rust::String tableau_to_string(const TableauHandle &handle);
rust::String tableau_to_repr(const TableauHandle &handle);
rust::Vec<float> tableau_to_unitary_matrix_data(const TableauHandle &handle, rust::Str endian);
rust::Vec<float> tableau_to_state_vector_data(const TableauHandle &handle, rust::Str endian);
std::unique_ptr<MeasurementSamplerHandle> circuit_compile_sampler(
    const CircuitHandle &handle,
    bool skip_reference_sample,
    std::uint64_t seed);
std::uint64_t measurement_sampler_num_measurements(const MeasurementSamplerHandle &handle);
rust::Vec<std::uint8_t> measurement_sampler_sample_bit_packed(
    MeasurementSamplerHandle &handle,
    std::uint64_t shots);
void measurement_sampler_sample_write(
    MeasurementSamplerHandle &handle,
    std::uint64_t shots,
    rust::Str filepath,
    rust::Str format_name);
std::unique_ptr<DetectorSamplerHandle> circuit_compile_detector_sampler(
    const CircuitHandle &handle,
    std::uint64_t seed);
std::unique_ptr<DemSamplerHandle> detector_error_model_compile_sampler(
    const DetectorErrorModelHandle &handle,
    std::uint64_t seed);
std::uint64_t detector_sampler_num_detectors(const DetectorSamplerHandle &handle);
std::uint64_t detector_sampler_num_observables(const DetectorSamplerHandle &handle);
std::uint64_t dem_sampler_num_detectors(const DemSamplerHandle &handle);
std::uint64_t dem_sampler_num_observables(const DemSamplerHandle &handle);
std::uint64_t dem_sampler_num_errors(const DemSamplerHandle &handle);
rust::Vec<std::uint8_t> detector_sampler_sample_bit_packed(
    DetectorSamplerHandle &handle,
    std::uint64_t shots);
rust::Vec<std::uint8_t> detector_sampler_sample_observables_bit_packed(
    DetectorSamplerHandle &handle,
    std::uint64_t shots);
DetectorSampleBatch detector_sampler_sample_bit_packed_separate_observables(
    DetectorSamplerHandle &handle,
    std::uint64_t shots);
DemSampleBatch dem_sampler_sample_bit_packed(
    DemSamplerHandle &handle,
    std::uint64_t shots);
DemSampleBatch dem_sampler_sample_bit_packed_replay(
    DemSamplerHandle &handle,
    rust::Slice<const std::uint8_t> recorded_errors,
    std::uint64_t shots);
void dem_sampler_sample_write(
    DemSamplerHandle &handle,
    std::uint64_t shots,
    rust::Str dets_filepath,
    rust::Str dets_format_name,
    rust::Str obs_filepath,
    rust::Str obs_format_name,
    rust::Str err_filepath,
    rust::Str err_format_name,
    bool write_errors,
    rust::Str replay_err_filepath,
    rust::Str replay_err_format_name,
    bool replay_errors);
void detector_sampler_sample_write(
    DetectorSamplerHandle &handle,
    std::uint64_t shots,
    rust::Str filepath,
    rust::Str format_name);
void detector_sampler_sample_write_separate_observables(
    DetectorSamplerHandle &handle,
    std::uint64_t shots,
    rust::Str dets_filepath,
    rust::Str dets_format_name,
    rust::Str obs_filepath,
    rust::Str obs_format_name);
std::unique_ptr<MeasurementsToDetectionEventsConverterHandle> circuit_compile_m2d_converter(
    const CircuitHandle &handle,
    bool skip_reference_sample);
std::uint64_t m2d_converter_num_measurements(const MeasurementsToDetectionEventsConverterHandle &handle);
std::uint64_t m2d_converter_num_detectors(const MeasurementsToDetectionEventsConverterHandle &handle);
std::uint64_t m2d_converter_num_observables(const MeasurementsToDetectionEventsConverterHandle &handle);
std::uint64_t m2d_converter_num_sweep_bits(const MeasurementsToDetectionEventsConverterHandle &handle);
rust::Vec<std::uint8_t> m2d_converter_convert_measurements_bit_packed(
    MeasurementsToDetectionEventsConverterHandle &handle,
    rust::Slice<const std::uint8_t> measurements,
    std::uint64_t shots,
    bool append_observables);
rust::Vec<std::uint8_t> m2d_converter_convert_measurements_and_sweep_bits_bit_packed(
    MeasurementsToDetectionEventsConverterHandle &handle,
    rust::Slice<const std::uint8_t> measurements,
    rust::Slice<const std::uint8_t> sweep_bits,
    std::uint64_t shots,
    bool append_observables);
rust::Vec<std::uint8_t> m2d_converter_convert_observables_with_sweep_bits_bit_packed(
    MeasurementsToDetectionEventsConverterHandle &handle,
    rust::Slice<const std::uint8_t> measurements,
    rust::Slice<const std::uint8_t> sweep_bits,
    std::uint64_t shots);
rust::Vec<std::uint8_t> m2d_converter_convert_observables_bit_packed(
    MeasurementsToDetectionEventsConverterHandle &handle,
    rust::Slice<const std::uint8_t> measurements,
    std::uint64_t shots);
void m2d_converter_convert_file(
    MeasurementsToDetectionEventsConverterHandle &handle,
    rust::Str measurements_filepath,
    rust::Str measurements_format,
    rust::Str sweep_bits_filepath,
    rust::Str sweep_bits_format,
    rust::Str detection_events_filepath,
    rust::Str detection_events_format,
    bool append_observables,
    rust::Str obs_out_filepath,
    rust::Str obs_out_format);

class SmokeProbe final {
 public:
  SmokeProbe() = default;

  rust::String describe() const;
  std::uint64_t weighted_checksum(rust::Slice<const std::uint64_t> values) const;
};

std::unique_ptr<SmokeProbe> new_smoke_probe();

}  // namespace stimrs::bridge
