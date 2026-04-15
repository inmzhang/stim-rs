#include "stim-cxx/include/stim_rs_bridge.h"
#include "stim-cxx/src/lib.rs.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <sstream>
#include <utility>
#include <vector>

#include "stim/main_namespaced.h"
#include "stim/diagram/base64.h"
#include "stim/diagram/crumble.h"
#include "stim/diagram/detector_slice/detector_slice_set.h"
#include "stim/diagram/gltf.h"
#include "stim/diagram/graph/match_graph_3d_drawer.h"
#include "stim/diagram/graph/match_graph_svg_drawer.h"
#include "stim/diagram/timeline/timeline_3d_drawer.h"
#include "stim/diagram/timeline/timeline_ascii_drawer.h"
#include "stim/diagram/timeline/timeline_svg_drawer.h"
#include "stim/gen/circuit_gen_params.h"
#include "stim/gen/gen_color_code.h"
#include "stim/gen/gen_rep_code.h"
#include "stim/gen/gen_surface_code.h"
#include "stim/io/measure_record_reader.h"
#include "stim/io/measure_record_writer.h"
#include "stim/io/raii_file.h"
#include "stim/util_top/count_determined_measurements.h"
#include "stim/util_top/export_crumble_url.h"
#include "stim/util_top/export_qasm.h"
#include "stim/util_top/export_quirk_url.h"
#include "stim/util_top/simplified_circuit.h"
#include "stim/io/raii_file.h"
#include "stim/simulators/measurements_to_detection_events.h"
#include "stim/simulators/tableau_simulator.h"
#include "stim/simulators/frame_simulator_util.h"
#include "stim/io/stim_data_formats.h"
#include "stim/io/measure_record_reader.h"
#include "stim/io/measure_record_writer.h"
#include "stim/io/raii_file.h"
#include "stim/search/search.h"
#include "stim/stabilizers/flow.h"
#include "stim/simulators/error_matcher.h"
#include "stim/simulators/error_analyzer.h"
#include "stim/util_top/circuit_vs_tableau.h"
#include "stim/util_top/circuit_inverse_qec.h"
#include "stim/util_top/circuit_to_detecting_regions.h"
#include "stim/util_top/missing_detectors.h"
#include "stim/util_top/stabilizers_vs_amplitudes.h"

namespace {

std::string_view trim_ascii(std::string_view text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
    text.remove_prefix(1);
  }
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.remove_suffix(1);
  }
  return text;
}

size_t normalize_index_or_throw(std::int64_t index, size_t len) {
  if (index < 0) {
    index += static_cast<std::int64_t>(len);
  }
  if (index < 0 || static_cast<size_t>(index) >= len) {
    throw std::invalid_argument("index out of range");
  }
  return static_cast<size_t>(index);
}

rust::Vec<std::uint8_t> packed_bits_to_rust_vec(
    const stim::simd_bits<stim::MAX_BITWORD_WIDTH> &bits,
    size_t num_bits) {
  size_t num_bytes = (num_bits + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(num_bytes);
  for (size_t i = 0; i < num_bytes; i++) {
    result.push_back(bits.u8[i]);
  }
  return result;
}

template <size_t W>
stimrs::bridge::BitTableData bit_table_to_data(
    const stim::simd_bit_table<W> &table,
    size_t rows,
    size_t cols,
    bool bit_packed) {
  stimrs::bridge::BitTableData out{
      .data = {},
      .rows = static_cast<std::uint64_t>(rows),
      .cols = static_cast<std::uint64_t>(cols),
      .bit_packed = bit_packed,
  };
  if (bit_packed) {
    size_t row_bytes = (cols + 7) / 8;
    out.data.reserve(rows * row_bytes);
    for (size_t r = 0; r < rows; r++) {
      auto row = table[r];
      for (size_t b = 0; b < row_bytes; b++) {
        out.data.push_back(row.u8[b]);
      }
    }
  } else {
    out.data.reserve(rows * cols);
    for (size_t r = 0; r < rows; r++) {
      auto row = table[r];
      for (size_t c = 0; c < cols; c++) {
        out.data.push_back(row[c] ? 1 : 0);
      }
    }
  }
  return out;
}

std::string escape_html_for_srcdoc(std::string_view src) {
  std::stringstream out;
  for (char c : src) {
    switch (c) {
      case '&':
        out << "&amp;";
        break;
      case '<':
        out << "&lt;";
        break;
      case '>':
        out << "&gt;";
        break;
      case '"':
        out << "&quot;";
        break;
      default:
        out << c;
        break;
    }
  }
  return out.str();
}

}  // namespace

#include "stim/util_top/stabilizers_to_tableau.h"
#include "stim/util_top/transform_without_feedback.h"

namespace stimrs::bridge {

GateTargetWithCoordsData convert_gate_target_with_coords(const stim::GateTargetWithCoords &value) {
  GateTargetWithCoordsData result{
      .raw_target = value.gate_target.data,
      .coords = {},
  };
  result.coords.reserve(value.coords.size());
  for (double coord : value.coords) {
    result.coords.push_back(coord);
  }
  return result;
}

DemTargetWithCoordsData convert_dem_target_with_coords(const stim::DemTargetWithCoords &value) {
  DemTargetWithCoordsData result{
      .dem_target = rust::String(value.dem_target.str()),
      .coords = {},
  };
  result.coords.reserve(value.coords.size());
  for (double coord : value.coords) {
    result.coords.push_back(coord);
  }
  return result;
}

CircuitErrorLocationStackFrameData convert_circuit_error_location_stack_frame(
    const stim::CircuitErrorLocationStackFrame &value) {
  return CircuitErrorLocationStackFrameData{
      .instruction_offset = value.instruction_offset,
      .iteration_index = value.iteration_index,
      .instruction_repetitions_arg = value.instruction_repetitions_arg,
  };
}

CircuitTargetsInsideInstructionData convert_circuit_targets_inside_instruction(
    const stim::CircuitTargetsInsideInstruction &value) {
  CircuitTargetsInsideInstructionData result{
      .gate = rust::String(std::string(
          value.gate_type == stim::GateType::NOT_A_GATE ? "NULL" : stim::GATE_DATA[value.gate_type].name)),
      .tag = rust::String(value.gate_tag),
      .args = {},
      .target_range_start = value.target_range_start,
      .target_range_end = value.target_range_end,
      .targets_in_range = {},
  };
  result.args.reserve(value.args.size());
  for (double arg : value.args) {
    result.args.push_back(arg);
  }
  result.targets_in_range.reserve(value.targets_in_range.size());
  for (const auto &target : value.targets_in_range) {
    result.targets_in_range.push_back(convert_gate_target_with_coords(target));
  }
  return result;
}

FlippedMeasurementData convert_flipped_measurement(const stim::FlippedMeasurement &value) {
  FlippedMeasurementData result{
      .record_index = value.measurement_record_index,
      .observable = {},
  };
  result.observable.reserve(value.measured_observable.size());
  for (const auto &target : value.measured_observable) {
    result.observable.push_back(convert_gate_target_with_coords(target));
  }
  return result;
}

CircuitErrorLocationData convert_circuit_error_location(const stim::CircuitErrorLocation &value) {
  CircuitErrorLocationData result{
      .tick_offset = value.tick_offset,
      .flipped_pauli_product = {},
      .flipped_measurement = convert_flipped_measurement(value.flipped_measurement),
      .instruction_targets = convert_circuit_targets_inside_instruction(value.instruction_targets),
      .stack_frames = {},
      .noise_tag = rust::String(value.noise_tag),
  };
  result.flipped_pauli_product.reserve(value.flipped_pauli_product.size());
  for (const auto &target : value.flipped_pauli_product) {
    result.flipped_pauli_product.push_back(convert_gate_target_with_coords(target));
  }
  result.stack_frames.reserve(value.stack_frames.size());
  for (const auto &frame : value.stack_frames) {
    result.stack_frames.push_back(convert_circuit_error_location_stack_frame(frame));
  }
  return result;
}

ExplainedErrorData convert_explained_error(const stim::ExplainedError &value) {
  ExplainedErrorData result{
      .dem_error_terms = {},
      .circuit_error_locations = {},
  };
  result.dem_error_terms.reserve(value.dem_error_terms.size());
  for (const auto &term : value.dem_error_terms) {
    result.dem_error_terms.push_back(convert_dem_target_with_coords(term));
  }
  result.circuit_error_locations.reserve(value.circuit_error_locations.size());
  for (const auto &location : value.circuit_error_locations) {
    result.circuit_error_locations.push_back(convert_circuit_error_location(location));
  }
  return result;
}

std::string escape_html_for_srcdoc(std::string_view src) {
  std::stringstream dst;
  for (char ch : src) {
    switch (ch) {
      case '&':
        dst << "&amp;";
        break;
      case '\'':
        dst << "&apos;";
        break;
      case '"':
        dst << "&quot;";
        break;
      case '<':
        dst << "&lt;";
        break;
      case '>':
        dst << "&gt;";
        break;
      default:
        dst << ch;
        break;
    }
  }
  return dst.str();
}

TableauHandle::TableauHandle() : tableau_(0) {}

TableauHandle::TableauHandle(stim::Tableau<stim::MAX_BITWORD_WIDTH> tableau)
    : tableau_(std::move(tableau)) {}

stim::Tableau<stim::MAX_BITWORD_WIDTH> &TableauHandle::get() {
  return tableau_;
}

const stim::Tableau<stim::MAX_BITWORD_WIDTH> &TableauHandle::get() const {
  return tableau_;
}

PauliStringHandle::PauliStringHandle() : pauli_string_(0) {}

PauliStringHandle::PauliStringHandle(stim::PauliString<stim::MAX_BITWORD_WIDTH> pauli_string)
    : pauli_string_(std::move(pauli_string)) {}

stim::PauliString<stim::MAX_BITWORD_WIDTH> &PauliStringHandle::get() {
  return pauli_string_;
}

const stim::PauliString<stim::MAX_BITWORD_WIDTH> &PauliStringHandle::get() const {
  return pauli_string_;
}

CliffordStringHandle::CliffordStringHandle() : clifford_string_(0) {}

CliffordStringHandle::CliffordStringHandle(
    stim::CliffordString<stim::MAX_BITWORD_WIDTH> clifford_string)
    : clifford_string_(std::move(clifford_string)) {}

stim::CliffordString<stim::MAX_BITWORD_WIDTH> &CliffordStringHandle::get() {
  return clifford_string_;
}

const stim::CliffordString<stim::MAX_BITWORD_WIDTH> &CliffordStringHandle::get() const {
  return clifford_string_;
}

PauliStringIteratorHandle::PauliStringIteratorHandle(
    std::size_t num_qubits,
    std::size_t min_weight,
    std::size_t max_weight,
    bool allow_x,
    bool allow_y,
    bool allow_z)
    : iterator_(num_qubits, min_weight, max_weight, allow_x, allow_y, allow_z) {}

PauliStringIteratorHandle::PauliStringIteratorHandle(
    stim::PauliStringIterator<stim::MAX_BITWORD_WIDTH> iterator)
    : iterator_(std::move(iterator)) {}

stim::PauliStringIterator<stim::MAX_BITWORD_WIDTH> &PauliStringIteratorHandle::get() {
  return iterator_;
}

const stim::PauliStringIterator<stim::MAX_BITWORD_WIDTH> &PauliStringIteratorHandle::get() const {
  return iterator_;
}

TableauIteratorHandle::TableauIteratorHandle(std::size_t num_qubits, bool unsigned_only)
    : iterator_(num_qubits, !unsigned_only) {}

TableauIteratorHandle::TableauIteratorHandle(
    stim::TableauIterator<stim::MAX_BITWORD_WIDTH> iterator)
    : iterator_(std::move(iterator)) {}

stim::TableauIterator<stim::MAX_BITWORD_WIDTH> &TableauIteratorHandle::get() {
  return iterator_;
}

const stim::TableauIterator<stim::MAX_BITWORD_WIDTH> &TableauIteratorHandle::get() const {
  return iterator_;
}

TableauSimulatorHandle::TableauSimulatorHandle(std::uint64_t seed, std::size_t num_qubits)
    : simulator_(std::mt19937_64{seed}, num_qubits) {}

TableauSimulatorHandle::TableauSimulatorHandle(
    stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> simulator)
    : simulator_(std::move(simulator)) {}

stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> &TableauSimulatorHandle::get() {
  return simulator_;
}

const stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> &TableauSimulatorHandle::get() const {
  return simulator_;
}

FrameSimulatorHandle::FrameSimulatorHandle(
    std::size_t batch_size,
    bool disable_stabilizer_randomization,
    std::size_t num_qubits,
    std::uint64_t seed)
    : simulator_(
          stim::CircuitStats{
              0,
              0,
              0,
              static_cast<uint32_t>(num_qubits),
              0,
              static_cast<uint32_t>(1 << 24),
              0,
          },
          stim::FrameSimulatorMode::STORE_EVERYTHING_TO_MEMORY,
          batch_size,
          std::mt19937_64{seed}) {
  simulator_.guarantee_anticommutation_via_frame_randomization = !disable_stabilizer_randomization;
  simulator_.reset_all();
}

FrameSimulatorHandle::FrameSimulatorHandle(stim::FrameSimulator<stim::MAX_BITWORD_WIDTH> simulator)
    : simulator_(std::move(simulator)) {}

stim::FrameSimulator<stim::MAX_BITWORD_WIDTH> &FrameSimulatorHandle::get() {
  return simulator_;
}

const stim::FrameSimulator<stim::MAX_BITWORD_WIDTH> &FrameSimulatorHandle::get() const {
  return simulator_;
}

GateDataHandle::GateDataHandle(stim::GateType type) : type_(type) {}

rust::String GateDataHandle::name() const {
  return rust::String(std::string(stim::GATE_DATA.at(type_).name));
}

rust::Vec<rust::String> GateDataHandle::aliases() const {
  rust::Vec<rust::String> result;
  std::vector<std::string_view> aliases;
  for (const auto &entry : stim::GATE_DATA.hashed_name_to_gate_type_table) {
    if (entry.id == type_) {
      aliases.push_back(entry.expected_name);
    }
  }
  std::sort(aliases.begin(), aliases.end());
  for (auto alias : aliases) {
    result.push_back(rust::String(std::string(alias)));
  }
  return result;
}

rust::Vec<std::uint8_t> GateDataHandle::num_parens_arguments_range() const {
  rust::Vec<std::uint8_t> result;
  const auto &gate = stim::GATE_DATA.at(type_);
  if (gate.arg_count == stim::ARG_COUNT_SYGIL_ZERO_OR_ONE) {
    result.push_back(0);
    result.push_back(1);
    return result;
  }
  if (gate.arg_count == stim::ARG_COUNT_SYGIL_ANY) {
    return result;
  }
  result.push_back(gate.arg_count);
  return result;
}

bool GateDataHandle::is_noisy_gate() const {
  return stim::GATE_DATA.at(type_).flags & stim::GateFlags::GATE_IS_NOISY;
}

bool GateDataHandle::is_reset() const {
  return stim::GATE_DATA.at(type_).flags & stim::GateFlags::GATE_IS_RESET;
}

bool GateDataHandle::is_single_qubit_gate() const {
  return stim::GATE_DATA.at(type_).flags & stim::GateFlags::GATE_IS_SINGLE_QUBIT_GATE;
}

bool GateDataHandle::is_symmetric_gate() const {
  return stim::GATE_DATA.at(type_).is_symmetric();
}

bool GateDataHandle::is_two_qubit_gate() const {
  return stim::GATE_DATA.at(type_).flags & stim::GateFlags::GATE_TARGETS_PAIRS;
}

bool GateDataHandle::is_unitary() const {
  return stim::GATE_DATA.at(type_).flags & stim::GateFlags::GATE_IS_UNITARY;
}

bool GateDataHandle::produces_measurements() const {
  return stim::GATE_DATA.at(type_).flags & stim::GateFlags::GATE_PRODUCES_RESULTS;
}

bool GateDataHandle::takes_measurement_record_targets() const {
  return stim::GATE_DATA.at(type_).flags &
         (stim::GateFlags::GATE_CAN_TARGET_BITS | stim::GateFlags::GATE_ONLY_TARGETS_MEASUREMENT_RECORD);
}

bool GateDataHandle::takes_pauli_targets() const {
  return stim::GATE_DATA.at(type_).flags & stim::GateFlags::GATE_TARGETS_PAULI_STRING;
}

rust::Vec<rust::String> GateDataHandle::flows() const {
  rust::Vec<rust::String> result;
  for (const auto &flow : stim::GATE_DATA.at(type_).flows<stim::MAX_BITWORD_WIDTH>()) {
    result.push_back(rust::String(flow.str()));
  }
  return result;
}

std::unique_ptr<TableauHandle> GateDataHandle::tableau() const {
  const auto &gate = stim::GATE_DATA.at(type_);
  if (!(gate.flags & stim::GateFlags::GATE_IS_UNITARY)) {
    return nullptr;
  }
  return std::make_unique<TableauHandle>(gate.tableau<stim::MAX_BITWORD_WIDTH>());
}

std::unique_ptr<GateDataHandle> GateDataHandle::clone_handle() const {
  return std::make_unique<GateDataHandle>(type_);
}

std::unique_ptr<GateDataHandle> GateDataHandle::inverse() const {
  const auto &gate = stim::GATE_DATA.at(type_);
  if (gate.flags & stim::GateFlags::GATE_IS_UNITARY) {
    return std::make_unique<GateDataHandle>(gate.best_candidate_inverse_id);
  }
  return nullptr;
}

std::unique_ptr<GateDataHandle> GateDataHandle::generalized_inverse() const {
  return std::make_unique<GateDataHandle>(stim::GATE_DATA.at(type_).best_candidate_inverse_id);
}

std::unique_ptr<GateDataHandle> GateDataHandle::hadamard_conjugated(bool unsigned_only) const {
  auto gate = stim::GATE_DATA.at(type_).hadamard_conjugated(unsigned_only);
  if (gate == stim::GateType::NOT_A_GATE) {
    return nullptr;
  }
  return std::make_unique<GateDataHandle>(gate);
}

rust::String canonicalize_flow_text(rust::Str text) {
  return rust::String(
      stim::Flow<stim::MAX_BITWORD_WIDTH>::from_str(std::string_view(text.data(), text.size())).str());
}

rust::String multiply_flow_texts(rust::Str left, rust::Str right) {
  auto lhs = stim::Flow<stim::MAX_BITWORD_WIDTH>::from_str(std::string_view(left.data(), left.size()));
  auto rhs = stim::Flow<stim::MAX_BITWORD_WIDTH>::from_str(std::string_view(right.data(), right.size()));
  return rust::String((lhs * rhs).str());
}

namespace {
stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> packed_rows_to_transposed_table(
    rust::Slice<const std::uint8_t> packed_rows,
    std::uint64_t shots,
    size_t bits_per_shot) {
  size_t row_bytes = (bits_per_shot + 7) / 8;
  if (packed_rows.size() != shots * row_bytes) {
    throw std::invalid_argument("Unexpected packed byte count.");
  }
  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> shot_major(shots, bits_per_shot);
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = shot_major[shot];
    for (size_t i = 0; i < row_bytes; i++) {
      row.u8[i] = packed_rows[shot * row_bytes + i];
    }
  }
  return shot_major.transposed();
}
}  // namespace

rust::Vec<std::uint8_t> read_shot_data_file_bit_packed(
    rust::Str filepath,
    rust::Str format_name,
    std::uint64_t num_measurements,
    std::uint64_t num_detectors,
    std::uint64_t num_observables) {
  auto parsed_format =
      stim::format_name_to_enum_map().at(std::string_view(format_name.data(), format_name.size())).id;
  size_t num_bits_per_shot = num_measurements + num_detectors + num_observables;
  size_t num_bytes_per_shot = (num_bits_per_shot + 7) / 8;
  rust::Vec<std::uint8_t> full_buffer;
  {
    stim::RaiiFile file(std::string(filepath).c_str(), "rb");
    auto reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
        file.f, parsed_format, num_measurements, num_detectors, num_observables);
    stim::simd_bits<stim::MAX_BITWORD_WIDTH> buffer(num_bits_per_shot);
    while (reader->start_and_read_entire_record(buffer)) {
      for (size_t i = 0; i < num_bytes_per_shot; i++) {
        full_buffer.push_back(buffer.u8[i]);
      }
    }
  }
  return full_buffer;
}

void write_shot_data_file_bit_packed(
    rust::Slice<const std::uint8_t> data,
    std::uint64_t shots,
    rust::Str filepath,
    rust::Str format_name,
    std::uint64_t num_measurements,
    std::uint64_t num_detectors,
    std::uint64_t num_observables) {
  auto parsed_format =
      stim::format_name_to_enum_map().at(std::string_view(format_name.data(), format_name.size())).id;
  size_t num_bits_per_shot = num_measurements + num_detectors + num_observables;
  size_t row_bytes = (num_bits_per_shot + 7) / 8;
  if (data.size() != shots * row_bytes) {
    throw std::invalid_argument("Unexpected packed shot byte count.");
  }
  auto table = packed_rows_to_transposed_table(data, shots, num_bits_per_shot);
  stim::RaiiFile file(std::string(filepath).c_str(), "wb");
  stim::simd_bits<stim::MAX_BITWORD_WIDTH> unused(0);
  stim::write_table_data(
      file.f,
      shots,
      num_bits_per_shot,
      unused,
      table,
      parsed_format,
      num_measurements == 0 ? 'D' : 'M',
      num_measurements == 0 ? 'L' : 'M',
      num_measurements + num_detectors);
}

rust::Vec<rust::String> all_gate_names() {
  rust::Vec<rust::String> result;
  for (const auto &gate : stim::GATE_DATA.items) {
    if (gate.id != stim::GateType::NOT_A_GATE) {
      result.push_back(rust::String(std::string(gate.name)));
    }
  }
  return result;
}

std::unique_ptr<GateDataHandle> gate_data_by_name(rust::Str name) {
  return std::make_unique<GateDataHandle>(
      stim::GATE_DATA.at(std::string_view(name.data(), name.size())).id);
}

namespace {

stim::SampleFormat sample_format_from_name(rust::Str format_name, const char *error_message) {
  auto format_it = stim::format_name_to_enum_map().find(
      std::string_view(format_name.data(), format_name.size()));
  if (format_it == stim::format_name_to_enum_map().end()) {
    throw std::invalid_argument(error_message);
  }
  return format_it->second.id;
}

}  // namespace

MeasurementSamplerHandle::MeasurementSamplerHandle(
    stim::simd_bits<stim::MAX_BITWORD_WIDTH> reference_sample,
    stim::Circuit circuit,
    bool skip_reference_sample,
    std::uint64_t seed)
    : reference_sample_(std::move(reference_sample)),
      circuit_(std::move(circuit)),
      skip_reference_sample_(skip_reference_sample),
      rng_(seed) {}

DetectorSamplerHandle::DetectorSamplerHandle(stim::Circuit circuit, std::uint64_t seed)
    : circuit_(std::move(circuit)), rng_(seed) {}

MeasurementsToDetectionEventsConverterHandle::MeasurementsToDetectionEventsConverterHandle(
    stim::simd_bits<stim::MAX_BITWORD_WIDTH> reference_sample,
    stim::Circuit circuit,
    bool skip_reference_sample)
    : skip_reference_sample_(skip_reference_sample),
      reference_sample_(std::move(reference_sample)),
      circuit_stats_(circuit.compute_stats()),
      circuit_(std::move(circuit)) {}

DemSamplerHandle::DemSamplerHandle(stim::DetectorErrorModel detector_error_model, std::uint64_t seed)
    : sampler_(std::move(detector_error_model), std::mt19937_64(seed), 1024) {}

namespace {

template <typename TABLE>
rust::Vec<std::uint8_t> sample_table_to_packed_bytes(TABLE &table, std::uint64_t bit_count, std::uint64_t shots) {
  auto shot_major = table.transposed();
  size_t row_bytes = (bit_count + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(shots * row_bytes);
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = shot_major[shot];
    for (size_t i = 0; i < row_bytes; i++) {
      result.push_back(row.u8[i]);
    }
  }
  return result;
}

}  // namespace

std::uint64_t MeasurementsToDetectionEventsConverterHandle::num_measurements() const {
  return circuit_.count_measurements();
}

std::uint64_t MeasurementsToDetectionEventsConverterHandle::num_detectors() const {
  return circuit_.count_detectors();
}

std::uint64_t MeasurementsToDetectionEventsConverterHandle::num_observables() const {
  return circuit_.count_observables();
}

std::uint64_t MeasurementsToDetectionEventsConverterHandle::num_sweep_bits() const {
  return circuit_stats_.num_sweep_bits;
}

std::uint64_t MeasurementSamplerHandle::num_measurements() const {
  return circuit_.count_measurements();
}

rust::Vec<std::uint8_t> MeasurementSamplerHandle::sample_bit_packed(std::uint64_t shots) {
  auto table = stim::sample_batch_measurements<stim::MAX_BITWORD_WIDTH>(
      circuit_,
      reference_sample_,
      shots,
      rng_,
      true);
  size_t row_bytes = (circuit_.count_measurements() + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(shots * row_bytes);
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = table[shot];
    for (size_t i = 0; i < row_bytes; i++) {
      result.push_back(row.u8[i]);
    }
  }
  return result;
}

void MeasurementSamplerHandle::sample_write(
    std::uint64_t shots,
    rust::Str filepath,
    rust::Str format_name) {
  auto format_it = stim::format_name_to_enum_map().find(
      std::string_view(format_name.data(), format_name.size()));
  if (format_it == stim::format_name_to_enum_map().end()) {
    throw std::invalid_argument("Unrecognized sample format.");
  }
  FILE *out = fopen(std::string(filepath).c_str(), "wb");
  if (out == nullptr) {
    throw std::invalid_argument("Failed to open output file.");
  }
  stim::sample_batch_measurements_writing_results_to_disk<stim::MAX_BITWORD_WIDTH>(
      circuit_,
      reference_sample_,
      shots,
      out,
      format_it->second.id,
      rng_);
  fclose(out);
}

std::uint64_t DetectorSamplerHandle::num_detectors() const {
  return circuit_.count_detectors();
}

std::uint64_t DetectorSamplerHandle::num_observables() const {
  return circuit_.count_observables();
}

std::uint64_t DemSamplerHandle::num_detectors() const {
  return sampler_.num_detectors;
}

std::uint64_t DemSamplerHandle::num_observables() const {
  return sampler_.num_observables;
}

std::uint64_t DemSamplerHandle::num_errors() const {
  return sampler_.num_errors;
}

rust::Vec<std::uint8_t> DetectorSamplerHandle::sample_bit_packed(std::uint64_t shots) {
  auto result_pair = stim::sample_batch_detection_events<stim::MAX_BITWORD_WIDTH>(circuit_, shots, rng_);
  auto shot_major = result_pair.first.transposed();
  size_t row_bytes = (circuit_.count_detectors() + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(shots * row_bytes);
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = shot_major[shot];
    for (size_t i = 0; i < row_bytes; i++) {
      result.push_back(row.u8[i]);
    }
  }
  return result;
}

DemSampleBatch DemSamplerHandle::sample_bit_packed(std::uint64_t shots) {
  sampler_.set_min_stripes(shots);
  sampler_.resample(false);
  return DemSampleBatch{
      .detectors = sample_table_to_packed_bytes(sampler_.det_buffer, sampler_.num_detectors, shots),
      .observables = sample_table_to_packed_bytes(sampler_.obs_buffer, sampler_.num_observables, shots),
      .errors = sample_table_to_packed_bytes(sampler_.err_buffer, sampler_.num_errors, shots),
  };
}

DemSampleBatch DemSamplerHandle::sample_bit_packed_replay(
    rust::Slice<const std::uint8_t> recorded_errors,
    std::uint64_t shots) {
  sampler_.set_min_stripes(shots);
  size_t row_bytes = (sampler_.num_errors + 7) / 8;
  if (recorded_errors.size() != shots * row_bytes) {
    throw std::invalid_argument("recorded_errors length does not match shots * ceil(num_errors / 8)");
  }

  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> shot_major(shots, sampler_.num_errors);
  shot_major.clear();
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = shot_major[shot];
    memcpy(row.u8, recorded_errors.data() + shot * row_bytes, row_bytes);
  }
  sampler_.err_buffer = shot_major.transposed();
  sampler_.resample(true);
  return DemSampleBatch{
      .detectors = sample_table_to_packed_bytes(sampler_.det_buffer, sampler_.num_detectors, shots),
      .observables = sample_table_to_packed_bytes(sampler_.obs_buffer, sampler_.num_observables, shots),
      .errors = sample_table_to_packed_bytes(sampler_.err_buffer, sampler_.num_errors, shots),
  };
}

void DemSamplerHandle::sample_write(
    std::uint64_t shots,
    FILE *det_out,
    stim::SampleFormat det_out_format,
    FILE *obs_out,
    stim::SampleFormat obs_out_format,
    FILE *err_out,
    stim::SampleFormat err_out_format,
    FILE *replay_err_in,
    stim::SampleFormat replay_err_in_format) {
  sampler_.sample_write(
      shots,
      det_out,
      det_out_format,
      obs_out,
      obs_out_format,
      err_out,
      err_out_format,
      replay_err_in,
      replay_err_in_format);
}

rust::Vec<std::uint8_t> DetectorSamplerHandle::sample_observables_bit_packed(std::uint64_t shots) {
  auto result_pair = stim::sample_batch_detection_events<stim::MAX_BITWORD_WIDTH>(circuit_, shots, rng_);
  auto shot_major = result_pair.second.transposed();
  size_t row_bytes = (circuit_.count_observables() + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(shots * row_bytes);
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = shot_major[shot];
    for (size_t i = 0; i < row_bytes; i++) {
      result.push_back(row.u8[i]);
    }
  }
  return result;
}

void DetectorSamplerHandle::sample_write(
    std::uint64_t shots,
    rust::Str filepath,
    rust::Str format_name) {
  auto format_it = stim::format_name_to_enum_map().find(
      std::string_view(format_name.data(), format_name.size()));
  if (format_it == stim::format_name_to_enum_map().end()) {
    throw std::invalid_argument("Unrecognized detector sample format.");
  }
  FILE *out = fopen(std::string(filepath).c_str(), "wb");
  if (out == nullptr) {
    throw std::invalid_argument("Failed to open output file.");
  }
  stim::sample_batch_detection_events_writing_results_to_disk<stim::MAX_BITWORD_WIDTH>(
      circuit_,
      shots,
      false,
      false,
      out,
      format_it->second.id,
      rng_,
      nullptr,
      stim::SampleFormat::SAMPLE_FORMAT_01);
  fclose(out);
}

void DetectorSamplerHandle::sample_write_separate_observables(
    std::uint64_t shots,
    rust::Str dets_filepath,
    rust::Str dets_format_name,
    rust::Str obs_filepath,
    rust::Str obs_format_name) {
  auto dets_format_it = stim::format_name_to_enum_map().find(
      std::string_view(dets_format_name.data(), dets_format_name.size()));
  auto obs_format_it = stim::format_name_to_enum_map().find(
      std::string_view(obs_format_name.data(), obs_format_name.size()));
  if (dets_format_it == stim::format_name_to_enum_map().end() ||
      obs_format_it == stim::format_name_to_enum_map().end()) {
    throw std::invalid_argument("Unrecognized detector/observable sample format.");
  }
  FILE *dets_out = fopen(std::string(dets_filepath).c_str(), "wb");
  if (dets_out == nullptr) {
    throw std::invalid_argument("Failed to open detector output file.");
  }
  FILE *obs_out = fopen(std::string(obs_filepath).c_str(), "wb");
  if (obs_out == nullptr) {
    fclose(dets_out);
    throw std::invalid_argument("Failed to open observable output file.");
  }
  stim::sample_batch_detection_events_writing_results_to_disk<stim::MAX_BITWORD_WIDTH>(
      circuit_,
      shots,
      false,
      false,
      dets_out,
      dets_format_it->second.id,
      rng_,
      obs_out,
      obs_format_it->second.id);
  fclose(dets_out);
  fclose(obs_out);
}

rust::Vec<std::uint8_t>
MeasurementsToDetectionEventsConverterHandle::convert_measurements_bit_packed(
    rust::Slice<const std::uint8_t> measurements,
    std::uint64_t shots,
    bool append_observables) {
  auto measurements_minor_shot_index =
      packed_rows_to_transposed_table(measurements, shots, circuit_stats_.num_measurements);
  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> sweep_bits_minor_shot_index(0, shots);
  size_t num_output_bits =
      circuit_stats_.num_detectors + circuit_stats_.num_observables * append_observables;
  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> out_detection_results_minor_shot_index(
      num_output_bits, shots);
  stim::measurements_to_detection_events_helper(
      measurements_minor_shot_index,
      sweep_bits_minor_shot_index,
      out_detection_results_minor_shot_index,
      circuit_.aliased_noiseless_circuit(),
      circuit_stats_,
      reference_sample_,
      append_observables);

  auto shot_major = out_detection_results_minor_shot_index.transposed();
  size_t row_bytes = (num_output_bits + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(shots * row_bytes);
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = shot_major[shot];
    for (size_t i = 0; i < row_bytes; i++) {
      result.push_back(row.u8[i]);
    }
  }
  return result;
}

rust::Vec<std::uint8_t>
MeasurementsToDetectionEventsConverterHandle::convert_measurements_and_sweep_bits_bit_packed(
    rust::Slice<const std::uint8_t> measurements,
    rust::Slice<const std::uint8_t> sweep_bits,
    std::uint64_t shots,
    bool append_observables) {
  auto measurements_minor_shot_index =
      packed_rows_to_transposed_table(measurements, shots, circuit_stats_.num_measurements);
  auto sweep_bits_minor_shot_index =
      packed_rows_to_transposed_table(sweep_bits, shots, num_sweep_bits());
  size_t num_output_bits =
      circuit_stats_.num_detectors + circuit_stats_.num_observables * append_observables;
  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> out_detection_results_minor_shot_index(
      num_output_bits, shots);
  stim::measurements_to_detection_events_helper(
      measurements_minor_shot_index,
      sweep_bits_minor_shot_index,
      out_detection_results_minor_shot_index,
      circuit_.aliased_noiseless_circuit(),
      circuit_stats_,
      reference_sample_,
      append_observables);
  auto shot_major = out_detection_results_minor_shot_index.transposed();
  size_t row_bytes = (num_output_bits + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(shots * row_bytes);
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = shot_major[shot];
    for (size_t i = 0; i < row_bytes; i++) {
      result.push_back(row.u8[i]);
    }
  }
  return result;
}

rust::Vec<std::uint8_t>
MeasurementsToDetectionEventsConverterHandle::convert_observables_bit_packed(
    rust::Slice<const std::uint8_t> measurements,
    std::uint64_t shots) {
  size_t packed_cols = (circuit_stats_.num_measurements + 7) / 8;
  if (measurements.size() != shots * packed_cols) {
    throw std::invalid_argument("Unexpected packed measurement byte count.");
  }

  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> measurements_shot_major(
      shots, circuit_stats_.num_measurements);
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = measurements_shot_major[shot];
    for (size_t i = 0; i < packed_cols; i++) {
      row.u8[i] = measurements[shot * packed_cols + i];
    }
  }
  auto measurements_minor_shot_index = measurements_shot_major.transposed();

  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> sweep_bits_minor_shot_index(0, shots);
  size_t num_intermediate_bits = circuit_stats_.num_detectors + circuit_stats_.num_observables;
  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> out_detection_results_minor_shot_index(
      num_intermediate_bits, shots);
  stim::measurements_to_detection_events_helper(
      measurements_minor_shot_index,
      sweep_bits_minor_shot_index,
      out_detection_results_minor_shot_index,
      circuit_.aliased_noiseless_circuit(),
      circuit_stats_,
      reference_sample_,
      true);

  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> obs_minor_shot_index(
      circuit_stats_.num_observables, shots);
  for (size_t obs = 0; obs < circuit_stats_.num_observables; obs++) {
    obs_minor_shot_index[obs] =
        out_detection_results_minor_shot_index[circuit_stats_.num_detectors + obs];
  }
  auto shot_major = obs_minor_shot_index.transposed();
  size_t row_bytes = (circuit_stats_.num_observables + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(shots * row_bytes);
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = shot_major[shot];
    for (size_t i = 0; i < row_bytes; i++) {
      result.push_back(row.u8[i]);
    }
  }
  return result;
}

rust::Vec<std::uint8_t>
MeasurementsToDetectionEventsConverterHandle::convert_observables_with_sweep_bits_bit_packed(
    rust::Slice<const std::uint8_t> measurements,
    rust::Slice<const std::uint8_t> sweep_bits,
    std::uint64_t shots) {
  auto measurements_minor_shot_index =
      packed_rows_to_transposed_table(measurements, shots, circuit_stats_.num_measurements);
  auto sweep_bits_minor_shot_index =
      packed_rows_to_transposed_table(sweep_bits, shots, num_sweep_bits());

  size_t num_intermediate_bits = circuit_stats_.num_detectors + circuit_stats_.num_observables;
  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> out_detection_results_minor_shot_index(
      num_intermediate_bits, shots);
  stim::measurements_to_detection_events_helper(
      measurements_minor_shot_index,
      sweep_bits_minor_shot_index,
      out_detection_results_minor_shot_index,
      circuit_.aliased_noiseless_circuit(),
      circuit_stats_,
      reference_sample_,
      true);

  stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> obs_minor_shot_index(
      circuit_stats_.num_observables, shots);
  for (size_t obs = 0; obs < circuit_stats_.num_observables; obs++) {
    obs_minor_shot_index[obs] =
        out_detection_results_minor_shot_index[circuit_stats_.num_detectors + obs];
  }
  auto shot_major = obs_minor_shot_index.transposed();
  size_t row_bytes = (circuit_stats_.num_observables + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(shots * row_bytes);
  for (size_t shot = 0; shot < shots; shot++) {
    auto row = shot_major[shot];
    for (size_t i = 0; i < row_bytes; i++) {
      result.push_back(row.u8[i]);
    }
  }
  return result;
}

void MeasurementsToDetectionEventsConverterHandle::convert_file(
    rust::Str measurements_filepath,
    rust::Str measurements_format,
    rust::Str sweep_bits_filepath,
    rust::Str sweep_bits_format,
    rust::Str detection_events_filepath,
    rust::Str detection_events_format,
    bool append_observables,
    rust::Str obs_out_filepath,
    rust::Str obs_out_format) {
  auto format_in =
      sample_format_from_name(measurements_format, "Unrecognized measurement sample format.");
  auto format_sweep_bits =
      sample_format_from_name(sweep_bits_format, "Unrecognized sweep-bit sample format.");
  auto format_out = sample_format_from_name(
      detection_events_format, "Unrecognized detection-event sample format.");
  auto parsed_obs_out_format =
      sample_format_from_name(obs_out_format, "Unrecognized observable sample format.");

  stim::RaiiFile file_in(
      std::string_view(measurements_filepath.data(), measurements_filepath.size()), "rb");
  stim::RaiiFile sweep_bits_in(
      std::string_view(sweep_bits_filepath.data(), sweep_bits_filepath.size()), "rb");
  stim::RaiiFile detections_out(
      std::string_view(detection_events_filepath.data(), detection_events_filepath.size()), "wb");
  stim::RaiiFile obs_out(
      std::string_view(obs_out_filepath.data(), obs_out_filepath.size()), "wb");

  stim::stream_measurements_to_detection_events_helper<stim::MAX_BITWORD_WIDTH>(
      file_in.f,
      format_in,
      sweep_bits_in.f,
      format_sweep_bits,
      detections_out.f,
      format_out,
      circuit_.aliased_noiseless_circuit(),
      circuit_stats_,
      append_observables,
      reference_sample_,
      obs_out.f,
      parsed_obs_out_format);
}

CircuitHandle::CircuitHandle() = default;

CircuitHandle::CircuitHandle(stim::Circuit circuit) : circuit_(std::move(circuit)) {}

stim::Circuit &CircuitHandle::get() {
  return circuit_;
}

const stim::Circuit &CircuitHandle::get() const {
  return circuit_;
}

DetectorErrorModelHandle::DetectorErrorModelHandle() = default;

DetectorErrorModelHandle::DetectorErrorModelHandle(
    stim::DetectorErrorModel detector_error_model)
    : detector_error_model_(std::move(detector_error_model)) {}

stim::DetectorErrorModel &DetectorErrorModelHandle::get() {
  return detector_error_model_;
}

const stim::DetectorErrorModel &DetectorErrorModelHandle::get() const {
  return detector_error_model_;
}

std::unique_ptr<CircuitHandle> new_circuit() {
  return std::make_unique<CircuitHandle>();
}

std::unique_ptr<CircuitHandle> circuit_from_stim_program_text(rust::Str text) {
  return std::make_unique<CircuitHandle>(
      stim::Circuit(std::string_view(text.data(), text.size())));
}

std::unique_ptr<CircuitHandle> circuit_clone(const CircuitHandle &handle) {
  return std::make_unique<CircuitHandle>(handle.get());
}

std::unique_ptr<CircuitHandle> circuit_add(
    const CircuitHandle &left,
    const CircuitHandle &right) {
  return std::make_unique<CircuitHandle>(left.get() + right.get());
}

void circuit_add_assign(CircuitHandle &left, const CircuitHandle &right) {
  left.get() += right.get();
}

std::unique_ptr<CircuitHandle> circuit_mul(
    const CircuitHandle &handle,
    std::uint64_t repetitions) {
  return std::make_unique<CircuitHandle>(handle.get() * repetitions);
}

void circuit_mul_assign(CircuitHandle &handle, std::uint64_t repetitions) {
  handle.get() *= repetitions;
}

std::unique_ptr<CircuitHandle> circuit_generated(
    rust::Str code_task,
    std::size_t distance,
    std::size_t rounds,
    double after_clifford_depolarization,
    double before_round_data_depolarization,
    double before_measure_flip_probability,
    double after_reset_flip_probability) {
  std::string_view type(code_task.data(), code_task.size());
  auto split = type.find(':');
  std::string_view code = split == std::string_view::npos ? "" : type.substr(0, split);
  std::string_view task = split == std::string_view::npos ? type : type.substr(split + 1);

  stim::CircuitGenParameters params(rounds, distance, std::string(task));
  params.after_clifford_depolarization = after_clifford_depolarization;
  params.before_round_data_depolarization = before_round_data_depolarization;
  params.before_measure_flip_probability = before_measure_flip_probability;
  params.after_reset_flip_probability = after_reset_flip_probability;
  params.validate_params();

  if (code == "surface_code") {
    return std::make_unique<CircuitHandle>(
        stim::generate_surface_code_circuit(params).circuit);
  }
  if (code == "repetition_code") {
    return std::make_unique<CircuitHandle>(stim::generate_rep_code_circuit(params).circuit);
  }
  if (code == "color_code") {
    return std::make_unique<CircuitHandle>(
        stim::generate_color_code_circuit(params).circuit);
  }

  throw std::invalid_argument(
      "Unrecognized circuit type. Expected type to start with "
      "'surface_code:', 'repetition_code:', or 'color_code:'.");
}

bool circuit_has_flow_text(
    const CircuitHandle &handle,
    rust::Str flow_text,
    bool unsigned_only) {
  auto flow =
      stim::Flow<stim::MAX_BITWORD_WIDTH>::from_str(std::string_view(flow_text.data(), flow_text.size()));
  auto flows = std::span<const stim::Flow<stim::MAX_BITWORD_WIDTH>>(&flow, 1);
  if (unsigned_only) {
    return stim::check_if_circuit_has_unsigned_stabilizer_flows<stim::MAX_BITWORD_WIDTH>(
               handle.get(), flows)[0];
  }
  std::random_device rd;
  std::mt19937_64 rng(rd());
  return stim::sample_if_circuit_has_stabilizer_flows<stim::MAX_BITWORD_WIDTH>(
             256, rng, handle.get(), flows)[0];
}

bool circuit_has_all_flows_text(
    const CircuitHandle &handle,
    rust::Vec<rust::String> flow_texts,
    bool unsigned_only) {
  std::vector<stim::Flow<stim::MAX_BITWORD_WIDTH>> flows;
  flows.reserve(flow_texts.size());
  for (const auto &text : flow_texts) {
    flows.push_back(
        stim::Flow<stim::MAX_BITWORD_WIDTH>::from_str(std::string_view(text.data(), text.size())));
  }
  if (unsigned_only) {
    auto results = stim::check_if_circuit_has_unsigned_stabilizer_flows<stim::MAX_BITWORD_WIDTH>(
        handle.get(), std::span<const stim::Flow<stim::MAX_BITWORD_WIDTH>>(flows));
    return std::all_of(results.begin(), results.end(), [](bool b) { return b; });
  }
  std::random_device rd;
  std::mt19937_64 rng(rd());
  auto results = stim::sample_if_circuit_has_stabilizer_flows<stim::MAX_BITWORD_WIDTH>(
      256, rng, handle.get(), std::span<const stim::Flow<stim::MAX_BITWORD_WIDTH>>(flows));
  return std::all_of(results.begin(), results.end(), [](bool b) { return b; });
}

rust::Vec<rust::String> circuit_flow_generators_texts(const CircuitHandle &handle) {
  rust::Vec<rust::String> result;
  for (const auto &flow : stim::circuit_flow_generators<stim::MAX_BITWORD_WIDTH>(handle.get())) {
    result.push_back(rust::String(flow.str()));
  }
  return result;
}

rust::Vec<rust::String> circuit_solve_flow_measurements_text(
    const CircuitHandle &handle,
    rust::Vec<rust::String> flow_texts) {
  std::vector<stim::Flow<stim::MAX_BITWORD_WIDTH>> flows;
  flows.reserve(flow_texts.size());
  for (const auto &text : flow_texts) {
    flows.push_back(
        stim::Flow<stim::MAX_BITWORD_WIDTH>::from_str(std::string_view(text.data(), text.size())));
  }

  rust::Vec<rust::String> result;
  for (const auto &solution :
       stim::solve_for_flow_measurements<stim::MAX_BITWORD_WIDTH>(handle.get(), flows)) {
    if (!solution.has_value()) {
      result.push_back(rust::String("!"));
      continue;
    }

    std::stringstream ss;
    bool first = true;
    for (int32_t measurement : *solution) {
      if (!first) {
        ss << ',';
      }
      first = false;
      ss << measurement;
    }
    result.push_back(rust::String(ss.str()));
  }
  return result;
}

rust::String circuit_time_reversed_for_flows_text(
    rust::Vec<rust::String> &out_flow_texts,
    const CircuitHandle &handle,
    rust::Vec<rust::String> flow_texts,
    bool dont_turn_measurements_into_resets) {
  std::vector<stim::Flow<stim::MAX_BITWORD_WIDTH>> flows;
  flows.reserve(flow_texts.size());
  for (const auto &text : flow_texts) {
    flows.push_back(
        stim::Flow<stim::MAX_BITWORD_WIDTH>::from_str(std::string_view(text.data(), text.size())));
  }

  auto [inverted_circuit, inverted_flows] = stim::circuit_inverse_qec<stim::MAX_BITWORD_WIDTH>(
      handle.get(), flows, dont_turn_measurements_into_resets);

  for (const auto &flow : inverted_flows) {
    out_flow_texts.push_back(rust::String(flow.str()));
  }
  return rust::String(inverted_circuit.str());
}

rust::String circuit_to_stim_program_text(const CircuitHandle &handle) {
  return rust::String(handle.get().str());
}

rust::Vec<rust::String> circuit_flattened_operation_texts(const CircuitHandle &handle) {
  rust::Vec<rust::String> result;
  handle.get().for_each_operation([&](const stim::CircuitInstruction &inst) {
    result.push_back(rust::String(inst.str()));
  });
  return result;
}

std::size_t circuit_num_qubits(const CircuitHandle &handle) {
  return handle.get().count_qubits();
}

std::size_t circuit_len(const CircuitHandle &handle) {
  return handle.get().operations.size();
}

std::uint64_t circuit_num_measurements(const CircuitHandle &handle) {
  return handle.get().count_measurements();
}

std::uint64_t circuit_count_determined_measurements(
    const CircuitHandle &handle,
    bool unknown_input) {
  return stim::count_determined_measurements<stim::MAX_BITWORD_WIDTH>(
      handle.get(),
      unknown_input);
}

std::uint64_t circuit_num_detectors(const CircuitHandle &handle) {
  return handle.get().count_detectors();
}

std::uint64_t circuit_num_observables(const CircuitHandle &handle) {
  return handle.get().count_observables();
}

std::uint64_t circuit_num_ticks(const CircuitHandle &handle) {
  return handle.get().count_ticks();
}

std::size_t circuit_num_sweep_bits(const CircuitHandle &handle) {
  return handle.get().count_sweep_bits();
}

void circuit_append_from_stim_program_text(CircuitHandle &handle, rust::Str text) {
  handle.get().append_from_text(std::string_view(text.data(), text.size()));
}

void circuit_append_gate(
    CircuitHandle &handle,
    rust::Str gate_name,
    rust::Slice<const std::uint32_t> targets,
    rust::Slice<const double> args) {
  std::vector<std::uint32_t> owned_targets(targets.begin(), targets.end());
  std::vector<double> owned_args(args.begin(), args.end());
  handle.get().safe_append_u(
      std::string_view(gate_name.data(), gate_name.size()),
      owned_targets,
      owned_args);
}

void circuit_append_repeat_block(
    CircuitHandle &handle,
    std::uint64_t repeat_count,
    const CircuitHandle &body,
    rust::Str tag) {
  handle.get().append_repeat_block(
      repeat_count,
      body.get(),
      std::string_view(tag.data(), tag.size()));
}

void circuit_clear(CircuitHandle &handle) {
  handle.get().clear();
}

bool circuit_equals(const CircuitHandle &left, const CircuitHandle &right) {
  return left.get() == right.get();
}

bool circuit_approx_equals(
    const CircuitHandle &left,
    const CircuitHandle &right,
    double atol) {
  return left.get().approx_equals(right.get(), atol);
}

std::unique_ptr<CircuitHandle> circuit_without_noise(const CircuitHandle &handle) {
  return std::make_unique<CircuitHandle>(handle.get().without_noise());
}

std::unique_ptr<CircuitHandle> circuit_with_inlined_feedback(const CircuitHandle &handle) {
  return std::make_unique<CircuitHandle>(stim::circuit_with_inlined_feedback(handle.get()));
}

std::unique_ptr<CircuitHandle> circuit_without_tags(const CircuitHandle &handle) {
  return std::make_unique<CircuitHandle>(handle.get().without_tags());
}

std::unique_ptr<CircuitHandle> circuit_flattened(const CircuitHandle &handle) {
  return std::make_unique<CircuitHandle>(handle.get().flattened());
}

std::unique_ptr<CircuitHandle> circuit_decomposed(const CircuitHandle &handle) {
  return std::make_unique<CircuitHandle>(stim::simplified_circuit(handle.get()));
}

std::unique_ptr<CircuitHandle> circuit_inverse(const CircuitHandle &handle) {
  return std::make_unique<CircuitHandle>(handle.get().inverse());
}

rust::String circuit_to_qasm(
    const CircuitHandle &handle,
    int open_qasm_version,
    bool skip_dets_and_obs) {
  std::stringstream out;
  stim::export_open_qasm(handle.get(), out, open_qasm_version, skip_dets_and_obs);
  return rust::String(out.str());
}

rust::String circuit_to_quirk_url(const CircuitHandle &handle) {
  return rust::String(stim::export_quirk_url(handle.get()));
}

rust::String circuit_to_crumble_url(
    const CircuitHandle &handle,
    bool skip_detectors) {
  return rust::String(stim::export_crumble_url(handle.get(), skip_detectors));
}

rust::String circuit_shortest_error_sat_problem(
    const CircuitHandle &handle,
    rust::Str format_name) {
  auto dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
      handle.get(), false, true, false, 1, false, false);
  return rust::String(stim::shortest_error_sat_problem(
      dem, std::string_view(format_name.data(), format_name.size())));
}

rust::String circuit_likeliest_error_sat_problem(
    const CircuitHandle &handle,
    std::int32_t quantization,
    rust::Str format_name) {
  auto dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
      handle.get(), false, true, false, 1, false, false);
  return rust::String(stim::likeliest_error_sat_problem(
      dem, quantization, std::string_view(format_name.data(), format_name.size())));
}

rust::String circuit_detecting_regions_text(const CircuitHandle &handle) {
  auto stats = handle.get().compute_stats();
  std::set<stim::DemTarget> targets;
  std::set<uint64_t> ticks;
  for (uint64_t k = 0; k < stats.num_detectors; k++) {
    targets.insert(stim::DemTarget::relative_detector_id(k));
  }
  for (uint64_t k = 0; k < stats.num_observables; k++) {
    targets.insert(stim::DemTarget::observable_id(k));
  }
  for (uint64_t k = 0; k < stats.num_ticks; k++) {
    ticks.insert(k);
  }

  auto regions = stim::circuit_to_detecting_regions(handle.get(), targets, ticks, false);
  std::stringstream ss;
  ss.precision(17);
  for (const auto &[target, tick_map] : regions) {
    for (const auto &[tick, pauli] : tick_map) {
      ss << target.str() << '\t' << tick << '\t' << pauli.str() << '\n';
    }
  }
  return rust::String(ss.str());
}

rust::String circuit_detecting_regions_text_with_options(
    const CircuitHandle &handle,
    rust::Vec<rust::String> target_texts,
    rust::Vec<std::uint64_t> ticks,
    bool ignore_anticommutation_errors) {
  auto stats = handle.get().compute_stats();

  std::set<stim::DemTarget> targets;
  if (target_texts.empty()) {
    for (uint64_t k = 0; k < stats.num_detectors; k++) {
      targets.insert(stim::DemTarget::relative_detector_id(k));
    }
    for (uint64_t k = 0; k < stats.num_observables; k++) {
      targets.insert(stim::DemTarget::observable_id(k));
    }
  } else {
    for (const auto &text : target_texts) {
      targets.insert(
          stim::DemTarget::from_text(std::string_view(text.data(), text.size())));
    }
  }

  std::set<uint64_t> included_ticks;
  if (ticks.empty()) {
    for (uint64_t k = 0; k < stats.num_ticks; k++) {
      included_ticks.insert(k);
    }
  } else {
    included_ticks.insert(ticks.begin(), ticks.end());
  }

  auto regions = stim::circuit_to_detecting_regions(
      handle.get(), targets, included_ticks, ignore_anticommutation_errors);
  std::stringstream ss;
  ss.precision(17);
  for (const auto &[target, tick_map] : regions) {
    for (const auto &[tick, pauli] : tick_map) {
      ss << target.str() << '\t' << tick << '\t' << pauli.str() << '\n';
    }
  }
  return rust::String(ss.str());
}

rust::Vec<ExplainedErrorData> circuit_explain_detector_error_model_errors(
    const CircuitHandle &handle,
    rust::Str dem_filter_text,
    bool has_dem_filter,
    bool reduce_to_one_representative_error) {
  rust::Vec<ExplainedErrorData> result;
  std::vector<stim::ExplainedError> explained;
  if (has_dem_filter) {
    auto model = stim::DetectorErrorModel(std::string_view(dem_filter_text.data(), dem_filter_text.size()));
    explained = stim::ErrorMatcher::explain_errors_from_circuit(
        handle.get(), &model, reduce_to_one_representative_error);
  } else {
    explained = stim::ErrorMatcher::explain_errors_from_circuit(
        handle.get(), nullptr, reduce_to_one_representative_error);
  }
  result.reserve(explained.size());
  for (const auto &item : explained) {
    result.push_back(convert_explained_error(item));
  }
  return result;
}

rust::Vec<ExplainedErrorData> circuit_shortest_graphlike_error(
    const CircuitHandle &handle,
    bool ignore_ungraphlike_errors,
    bool canonicalize_circuit_errors) {
  auto dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
      handle.get(),
      !ignore_ungraphlike_errors,
      true,
      false,
      1,
      false,
      false);
  auto filter =
      stim::shortest_graphlike_undetectable_logical_error(dem, ignore_ungraphlike_errors);
  auto explained = stim::ErrorMatcher::explain_errors_from_circuit(
      handle.get(), &filter, canonicalize_circuit_errors);
  rust::Vec<ExplainedErrorData> result;
  result.reserve(explained.size());
  for (const auto &item : explained) {
    result.push_back(convert_explained_error(item));
  }
  return result;
}

rust::Vec<ExplainedErrorData> circuit_search_for_undetectable_logical_errors(
    const CircuitHandle &handle,
    std::uint64_t dont_explore_detection_event_sets_with_size_above,
    std::uint64_t dont_explore_edges_with_degree_above,
    bool dont_explore_edges_increasing_symptom_degree,
    bool canonicalize_circuit_errors) {
  auto dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
      handle.get(), false, true, false, 1, false, false);
  auto filter = stim::find_undetectable_logical_error(
      dem,
      dont_explore_detection_event_sets_with_size_above,
      dont_explore_edges_with_degree_above,
      dont_explore_edges_increasing_symptom_degree);
  auto explained = stim::ErrorMatcher::explain_errors_from_circuit(
      handle.get(), &filter, canonicalize_circuit_errors);
  rust::Vec<ExplainedErrorData> result;
  result.reserve(explained.size());
  for (const auto &item : explained) {
    result.push_back(convert_explained_error(item));
  }
  return result;
}

rust::String circuit_get_detector_coordinates_text(
    const CircuitHandle &handle,
    rust::Slice<const std::uint64_t> included_detector_indices) {
  std::set<std::uint64_t> included(
      included_detector_indices.begin(),
      included_detector_indices.end());
  auto coords = handle.get().get_detector_coordinates(included);
  std::stringstream ss;
  ss.precision(17);
  for (const auto &entry : coords) {
    ss << entry.first;
    for (double coord : entry.second) {
      ss << '\t' << coord;
    }
    ss << '\n';
  }
  return rust::String(ss.str());
}

rust::String circuit_get_final_qubit_coordinates_text(const CircuitHandle &handle) {
  auto coords = handle.get().get_final_qubit_coords();
  std::stringstream ss;
  ss.precision(17);
  for (const auto &entry : coords) {
    ss << entry.first;
    for (double coord : entry.second) {
      ss << '\t' << coord;
    }
    ss << '\n';
  }
  return rust::String(ss.str());
}

rust::Vec<std::uint8_t> circuit_reference_sample_bit_packed(const CircuitHandle &handle) {
  auto ref = stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(handle.get());
  size_t num_measurements = handle.get().count_measurements();
  size_t num_bytes = (num_measurements + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(num_bytes);
  for (size_t i = 0; i < num_bytes; i++) {
    result.push_back(ref.u8[i]);
  }
  return result;
}

rust::Vec<std::uint8_t> circuit_reference_detector_signs_bit_packed(const CircuitHandle &handle) {
  auto ref = stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(handle.get());
  size_t num_det = handle.get().count_detectors();
  stim::simd_bits<stim::MAX_BITWORD_WIDTH> dets(num_det);
  size_t offset = 0;
  size_t k_det = 0;
  handle.get().for_each_operation([&](stim::CircuitInstruction inst) {
    if (inst.gate_type == stim::GateType::DETECTOR) {
      stim::bit_ref d = dets[k_det++];
      for (const auto &t : inst.targets) {
        if (t.is_measurement_record_target()) {
          d ^= ref[offset + t.value()];
        }
      }
    } else {
      offset += inst.count_measurement_results();
    }
  });
  size_t num_bytes = (num_det + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(num_bytes);
  for (size_t i = 0; i < num_bytes; i++) {
    result.push_back(dets.u8[i]);
  }
  return result;
}

rust::Vec<std::uint8_t> circuit_reference_observable_signs_bit_packed(const CircuitHandle &handle) {
  auto ref = stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(handle.get());
  size_t num_obs = handle.get().count_observables();
  stim::simd_bits<stim::MAX_BITWORD_WIDTH> obs(num_obs);
  size_t offset = 0;
  handle.get().for_each_operation([&](stim::CircuitInstruction inst) {
    if (inst.gate_type == stim::GateType::OBSERVABLE_INCLUDE) {
      stim::bit_ref d = obs[(int)inst.args[0]];
      for (const auto &t : inst.targets) {
        if (t.is_measurement_record_target()) {
          d ^= ref[offset + t.value()];
        }
      }
    } else {
      offset += inst.count_measurement_results();
    }
  });
  size_t num_bytes = (num_obs + 7) / 8;
  rust::Vec<std::uint8_t> result;
  result.reserve(num_bytes);
  for (size_t i = 0; i < num_bytes; i++) {
    result.push_back(obs.u8[i]);
  }
  return result;
}

std::unique_ptr<DetectorErrorModelHandle> new_detector_error_model() {
  return std::make_unique<DetectorErrorModelHandle>();
}

std::unique_ptr<DetectorErrorModelHandle> circuit_detector_error_model(
    const CircuitHandle &handle,
    bool decompose_errors,
    bool flatten_loops,
    bool allow_gauge_detectors,
    double approximate_disjoint_errors,
    bool ignore_decomposition_failures,
    bool block_decomposition_from_introducing_remnant_edges) {
  return std::make_unique<DetectorErrorModelHandle>(
      stim::ErrorAnalyzer::circuit_to_detector_error_model(
          handle.get(),
          decompose_errors,
          !flatten_loops,
          allow_gauge_detectors,
          approximate_disjoint_errors,
          ignore_decomposition_failures,
          block_decomposition_from_introducing_remnant_edges));
}

std::unique_ptr<CircuitHandle> circuit_missing_detectors(
    const CircuitHandle &handle,
    bool unknown_input) {
  return std::make_unique<CircuitHandle>(
      stim::missing_detectors(handle.get(), unknown_input));
}

std::unique_ptr<TableauHandle> circuit_to_tableau(
    const CircuitHandle &handle,
    bool ignore_noise,
    bool ignore_measurement,
    bool ignore_reset) {
  return std::make_unique<TableauHandle>(
      stim::circuit_to_tableau<stim::MAX_BITWORD_WIDTH>(
          handle.get(), ignore_noise, ignore_measurement, ignore_reset));
}

std::unique_ptr<TableauHandle> new_tableau(std::size_t num_qubits) {
  return std::make_unique<TableauHandle>(stim::Tableau<stim::MAX_BITWORD_WIDTH>(num_qubits));
}

std::unique_ptr<TableauHandle> tableau_random(std::size_t num_qubits) {
  std::random_device rd;
  std::mt19937_64 rng(rd());
  return std::make_unique<TableauHandle>(
      stim::Tableau<stim::MAX_BITWORD_WIDTH>::random(num_qubits, rng));
}

std::unique_ptr<TableauIteratorHandle> tableau_iter_all(
    std::size_t num_qubits,
    bool unsigned_only) {
  return std::make_unique<TableauIteratorHandle>(num_qubits, unsigned_only);
}

std::unique_ptr<TableauIteratorHandle> tableau_iterator_clone(const TableauIteratorHandle &handle) {
  return std::make_unique<TableauIteratorHandle>(handle.get());
}

std::unique_ptr<TableauHandle> tableau_iterator_next(TableauIteratorHandle &handle) {
  if (!handle.get().iter_next()) {
    return nullptr;
  }
  return std::make_unique<TableauHandle>(handle.get().result);
}

std::unique_ptr<TableauHandle> tableau_from_named_gate(rust::Str name) {
  const auto &gate = stim::GATE_DATA.at(std::string_view(name.data(), name.size()));
  if (!(gate.flags & stim::GATE_IS_UNITARY)) {
    throw std::out_of_range("Recognized name, but not unitary: " + std::string(name));
  }
  return std::make_unique<TableauHandle>(gate.tableau<stim::MAX_BITWORD_WIDTH>());
}

std::unique_ptr<TableauHandle> tableau_from_state_vector_data(
    rust::Vec<float> state_vector,
    rust::Str endian) {
  auto endian_view = std::string_view(endian.data(), endian.size());
  bool little_endian;
  if (endian_view == "little") {
    little_endian = true;
  } else if (endian_view == "big") {
    little_endian = false;
  } else {
    throw std::invalid_argument("endian not in ['little', 'big']");
  }
  if (state_vector.size() % 2 != 0) {
    throw std::invalid_argument("state_vector must contain an even number of floats");
  }
  std::vector<std::complex<float>> amplitudes;
  amplitudes.reserve(state_vector.size() / 2);
  for (size_t k = 0; k < state_vector.size(); k += 2) {
    amplitudes.emplace_back(state_vector[k], state_vector[k + 1]);
  }
  auto circuit = stim::stabilizer_state_vector_to_circuit(amplitudes, little_endian);
  return std::make_unique<TableauHandle>(
      stim::circuit_to_tableau<stim::MAX_BITWORD_WIDTH>(circuit, false, false, false));
}

std::unique_ptr<TableauHandle> tableau_from_unitary_matrix_data(
    rust::Vec<float> matrix,
    rust::Str endian) {
  auto endian_view = std::string_view(endian.data(), endian.size());
  bool little_endian;
  if (endian_view == "little") {
    little_endian = true;
  } else if (endian_view == "big") {
    little_endian = false;
  } else {
    throw std::invalid_argument("endian not in ['little', 'big']");
  }
  if (matrix.size() % 2 != 0) {
    throw std::invalid_argument("matrix must contain an even number of floats");
  }
  size_t num_complex = matrix.size() / 2;
  size_t n = (size_t)std::sqrt((double)num_complex);
  if (n * n != num_complex) {
    throw std::invalid_argument("matrix must be square");
  }
  std::vector<std::vector<std::complex<float>>> converted(n, std::vector<std::complex<float>>(n));
  for (size_t row = 0; row < n; row++) {
    for (size_t col = 0; col < n; col++) {
      size_t k = (row * n + col) * 2;
      converted[row][col] = std::complex<float>(matrix[k], matrix[k + 1]);
    }
  }
  return std::make_unique<TableauHandle>(
      stim::unitary_to_tableau<stim::MAX_BITWORD_WIDTH>(converted, little_endian));
}

std::unique_ptr<TableauHandle> tableau_from_conjugated_generator_texts(
    rust::Vec<rust::String> xs,
    rust::Vec<rust::String> zs) {
  size_t n = xs.size();
  if (n != zs.size()) {
    throw std::invalid_argument("len(xs) != len(zs)");
  }

  std::vector<stim::PauliString<stim::MAX_BITWORD_WIDTH>> parsed_xs;
  std::vector<stim::PauliString<stim::MAX_BITWORD_WIDTH>> parsed_zs;
  parsed_xs.reserve(n);
  parsed_zs.reserve(n);

  for (const auto &text : xs) {
    auto p = stim::PauliString<stim::MAX_BITWORD_WIDTH>(std::string(text));
    if (p.num_qubits != n) {
      throw std::invalid_argument("not all(len(p) == len(xs) for p in xs)");
    }
    parsed_xs.push_back(std::move(p));
  }
  for (const auto &text : zs) {
    auto p = stim::PauliString<stim::MAX_BITWORD_WIDTH>(std::string(text));
    if (p.num_qubits != n) {
      throw std::invalid_argument("not all(len(p) == len(zs) for p in zs)");
    }
    parsed_zs.push_back(std::move(p));
  }

  stim::Tableau<stim::MAX_BITWORD_WIDTH> result(n);
  for (size_t q = 0; q < n; q++) {
    result.xs[q] = parsed_xs[q];
    result.zs[q] = parsed_zs[q];
  }
  if (!result.satisfies_invariants()) {
    throw std::invalid_argument(
        "The given generator outputs don't describe a valid Clifford operation.\n"
        "They don't preserve commutativity.\n"
        "Everything must commute, except for X_k anticommuting with Z_k for each k.");
  }
  return std::make_unique<TableauHandle>(std::move(result));
}

std::unique_ptr<TableauHandle> tableau_from_stabilizer_texts(
    rust::Vec<rust::String> stabilizers,
    bool allow_redundant,
    bool allow_underconstrained) {
  std::vector<stim::PauliString<stim::MAX_BITWORD_WIDTH>> parsed;
  parsed.reserve(stabilizers.size());
  for (const auto &text : stabilizers) {
    parsed.emplace_back(std::string(text));
  }
  return std::make_unique<TableauHandle>(stim::stabilizers_to_tableau<stim::MAX_BITWORD_WIDTH>(
      parsed, allow_redundant, allow_underconstrained, false));
}

std::unique_ptr<TableauHandle> tableau_then(
    const TableauHandle &handle,
    const TableauHandle &second) {
  if (handle.get().num_qubits != second.get().num_qubits) {
    throw std::invalid_argument("len(self) != len(second)");
  }
  return std::make_unique<TableauHandle>(handle.get().then(second.get()));
}

std::unique_ptr<TableauSimulatorHandle> new_tableau_simulator(
    std::uint64_t seed,
    std::size_t num_qubits) {
  return std::make_unique<TableauSimulatorHandle>(seed, num_qubits);
}

std::unique_ptr<CliffordStringHandle> new_clifford_string(std::size_t num_qubits) {
  return std::make_unique<CliffordStringHandle>(
      stim::CliffordString<stim::MAX_BITWORD_WIDTH>(num_qubits));
}

std::unique_ptr<CliffordStringHandle> clifford_string_from_text(rust::Str text) {
  std::string_view view(text.data(), text.size());
  view = trim_ascii(view);
  if (view.empty()) {
    return std::make_unique<CliffordStringHandle>(
        stim::CliffordString<stim::MAX_BITWORD_WIDTH>(0));
  }
  if (view.ends_with(',')) {
    view.remove_suffix(1);
  }

  size_t n = 1;
  for (char ch : view) {
    n += ch == ',';
  }
  stim::CliffordString<stim::MAX_BITWORD_WIDTH> result(n);
  size_t start = 0;
  size_t out_index = 0;
  for (size_t end = 0; end <= view.size(); end++) {
    if (end == view.size() || view[end] == ',') {
      std::string_view segment = trim_ascii(view.substr(start, end - start));
      result.set_gate_at(out_index, stim::GATE_DATA.at(segment).id);
      start = end + 1;
      out_index += 1;
    }
  }
  return std::make_unique<CliffordStringHandle>(std::move(result));
}

std::unique_ptr<CliffordStringHandle> clifford_string_from_pauli_string(
    const PauliStringHandle &handle) {
  stim::CliffordString<stim::MAX_BITWORD_WIDTH> result(handle.get().num_qubits);
  result.z_signs = handle.get().xs;
  result.x_signs = handle.get().zs;
  return std::make_unique<CliffordStringHandle>(std::move(result));
}

std::unique_ptr<CliffordStringHandle> clifford_string_from_circuit(const CircuitHandle &handle) {
  return std::make_unique<CliffordStringHandle>(
      stim::CliffordString<stim::MAX_BITWORD_WIDTH>::from_circuit(handle.get()));
}

std::unique_ptr<CliffordStringHandle> clifford_string_random(std::size_t num_qubits) {
  std::mt19937_64 rng{0};
  return std::make_unique<CliffordStringHandle>(
      stim::CliffordString<stim::MAX_BITWORD_WIDTH>::random(num_qubits, rng));
}

std::unique_ptr<CliffordStringHandle> clifford_string_all_cliffords_string() {
  stim::CliffordString<stim::MAX_BITWORD_WIDTH> result(24);
  result.set_gate_at(0, stim::GateType::I);
  result.set_gate_at(4, stim::GateType::H_XY);
  result.set_gate_at(8, stim::GateType::H);
  result.set_gate_at(12, stim::GateType::H_YZ);
  result.set_gate_at(16, stim::GateType::C_XYZ);
  result.set_gate_at(20, stim::GateType::C_ZYX);
  for (size_t q = 0; q < 24; q++) {
    if (q % 4) {
      result.set_gate_at(q, result.gate_at(q - 1));
    }
  }

  stim::CliffordString<stim::MAX_BITWORD_WIDTH> ixyz(24);
  for (size_t q = 0; q < 24; q += 4) {
    ixyz.set_gate_at(q + 0, stim::GateType::I);
    ixyz.set_gate_at(q + 1, stim::GateType::X);
    ixyz.set_gate_at(q + 2, stim::GateType::Y);
    ixyz.set_gate_at(q + 3, stim::GateType::Z);
  }
  result *= ixyz;
  return std::make_unique<CliffordStringHandle>(std::move(result));
}

std::unique_ptr<PauliStringHandle> pauli_string_from_text(rust::Str text) {
  return std::make_unique<PauliStringHandle>(
      stim::PauliString<stim::MAX_BITWORD_WIDTH>(std::string_view(text.data(), text.size())));
}

std::unique_ptr<PauliStringHandle> new_pauli_string(std::size_t num_qubits) {
  return std::make_unique<PauliStringHandle>(stim::PauliString<stim::MAX_BITWORD_WIDTH>(num_qubits));
}

std::unique_ptr<DetectorErrorModelHandle> detector_error_model_from_dem_text(rust::Str text) {
  return std::make_unique<DetectorErrorModelHandle>(
      stim::DetectorErrorModel(std::string_view(text.data(), text.size())));
}

rust::String detector_error_model_diagram(
    const DetectorErrorModelHandle &handle,
    rust::Str type_name) {
  auto type = std::string_view(type_name.data(), type_name.size());
  if (type == "matchgraph-svg" || type == "match-graph-svg") {
    std::stringstream out;
    stim_draw_internal::dem_match_graph_to_svg_diagram_write_to(handle.get(), out);
    return rust::String(out.str());
  }
  if (type == "matchgraph-svg-html" || type == "match-graph-svg-html") {
    std::stringstream svg;
    stim_draw_internal::dem_match_graph_to_svg_diagram_write_to(handle.get(), svg);
    std::stringstream img;
    img << R"HTML(<img style="max-width: 100%; max-height: 100%" src="data:image/svg+xml;base64,)HTML";
    stim_draw_internal::write_data_as_base64_to(svg.str(), img);
    img << R"HTML("/>)HTML";
    std::stringstream framed;
    framed
        << R"HTML(<iframe style="width: 100%; height: 300px; overflow: hidden; resize: both; border: 1px dashed gray;" frameBorder="0" srcdoc=")HTML"
        << escape_html_for_srcdoc(img.str()) << R"HTML("></iframe>)HTML";
    return rust::String(framed.str());
  }
  if (type == "matchgraph-3d" || type == "match-graph-3d") {
    std::stringstream out;
    stim_draw_internal::dem_match_graph_to_basic_3d_diagram(handle.get()).to_gltf_scene().to_json().write(out);
    return rust::String(out.str());
  }
  if (type == "matchgraph-3d-html" || type == "match-graph-3d-html") {
    std::stringstream out;
    stim_draw_internal::dem_match_graph_to_basic_3d_diagram(handle.get()).to_gltf_scene().to_json().write(out);
    std::stringstream html;
    stim_draw_internal::write_html_viewer_for_gltf_data(out.str(), html);
    return rust::String(html.str());
  }
  throw std::invalid_argument("Unrecognized diagram type: " + std::string(type));
}

rust::String circuit_diagram(
    const CircuitHandle &handle,
    rust::Str type_name) {
  auto type = std::string_view(type_name.data(), type_name.size());
  std::vector<stim_draw_internal::CoordFilter> empty_filter{stim_draw_internal::CoordFilter{}};
  if (type == "timeline-text") {
    std::stringstream out;
    out << stim_draw_internal::DiagramTimelineAsciiDrawer::make_diagram(handle.get());
    return rust::String(out.str());
  }
  if (type == "timeline-svg" || type == "timeline" || type == "timeline-svg-html" || type == "timeline-html") {
    std::stringstream out;
    stim_draw_internal::DiagramTimelineSvgDrawer::make_diagram_write_to(
        handle.get(),
        out,
        0,
        UINT64_MAX,
        stim_draw_internal::DiagramTimelineSvgDrawerMode::SVG_MODE_TIMELINE,
        empty_filter);
    return rust::String(out.str());
  }
  if (type == "timeline-3d") {
    std::stringstream out;
    stim_draw_internal::DiagramTimeline3DDrawer::circuit_to_basic_3d_diagram(handle.get()).to_gltf_scene().to_json().write(out);
    return rust::String(out.str());
  }
  if (type == "timeline-3d-html") {
    std::stringstream out;
    stim_draw_internal::DiagramTimeline3DDrawer::circuit_to_basic_3d_diagram(handle.get()).to_gltf_scene().to_json().write(out);
    std::stringstream html;
    stim_draw_internal::write_html_viewer_for_gltf_data(out.str(), html);
    return rust::String(html.str());
  }
  if (type == "interactive" || type == "interactive-html") {
    std::stringstream out;
    stim_draw_internal::write_crumble_html_with_preloaded_circuit(handle.get(), out);
    return rust::String(out.str());
  }
  if (
      type == "match-graph-svg" || type == "matchgraph-svg" || type == "matchgraph-svg-html" ||
      type == "matchgraph-html" || type == "match-graph-svg-html" || type == "match-graph-html" ||
      type == "match-graph-3d" || type == "matchgraph-3d" || type == "match-graph-3d-html" ||
      type == "matchgraph-3d-html") {
    stim::DetectorErrorModel dem;
    try {
      dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(handle.get(), true, true, false, 1, false, false);
    } catch (const std::invalid_argument &) {
      dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(handle.get(), false, true, false, 1, false, false);
    }
    return detector_error_model_diagram(DetectorErrorModelHandle(dem), rust::Str(type_name));
  }
  throw std::invalid_argument("Unrecognized diagram type: " + std::string(type));
}

rust::String circuit_diagram_with_options(
    const CircuitHandle &handle,
    rust::Str type_name,
    bool has_tick_range,
    std::uint64_t tick_start,
    std::uint64_t tick_count,
    bool has_rows,
    std::size_t rows) {
  auto type = std::string_view(type_name.data(), type_name.size());
  std::vector<stim_draw_internal::CoordFilter> empty_filter{stim_draw_internal::CoordFilter{}};
  uint64_t tick_min = has_tick_range ? tick_start : 0;
  uint64_t num_ticks = has_tick_range ? tick_count : UINT64_MAX;
  size_t num_rows = has_rows ? rows : 0;

  if (type == "detslice-text" || type == "detector-slice-text") {
    std::stringstream out;
    stim_draw_internal::DetectorSliceSet::from_circuit_ticks(handle.get(), tick_min, num_ticks, empty_filter)
        .write_text_diagram_to(out);
    return rust::String(out.str());
  }
  if (
      type == "detslice-svg" || type == "detslice" || type == "detslice-html" || type == "detslice-svg-html" ||
      type == "detector-slice-svg" || type == "detector-slice") {
    std::stringstream out;
    stim_draw_internal::DetectorSliceSet::from_circuit_ticks(handle.get(), tick_min, num_ticks, empty_filter)
        .write_svg_diagram_to(out, num_rows);
    return rust::String(out.str());
  }
  if (
      type == "time-slice-svg" || type == "timeslice-svg" || type == "timeslice-html" ||
      type == "timeslice-svg-html" || type == "time-slice-html" || type == "time-slice-svg-html" ||
      type == "timeslice" || type == "time-slice") {
    std::stringstream out;
    stim_draw_internal::DiagramTimelineSvgDrawer::make_diagram_write_to(
        handle.get(),
        out,
        tick_min,
        num_ticks,
        stim_draw_internal::DiagramTimelineSvgDrawerMode::SVG_MODE_TIME_SLICE,
        empty_filter,
        num_rows);
    return rust::String(out.str());
  }
  if (
      type == "detslice-with-ops" || type == "detslice-with-ops-svg" || type == "detslice-with-ops-html" ||
      type == "detslice-with-ops-svg-html" || type == "time+detector-slice-svg") {
    std::stringstream out;
    stim_draw_internal::DiagramTimelineSvgDrawer::make_diagram_write_to(
        handle.get(),
        out,
        tick_min,
        num_ticks,
        stim_draw_internal::DiagramTimelineSvgDrawerMode::SVG_MODE_TIME_DETECTOR_SLICE,
        empty_filter,
        num_rows);
    return rust::String(out.str());
  }
  throw std::invalid_argument("Unrecognized diagram type: " + std::string(type));
}

rust::String circuit_diagram_with_options_and_filters(
    const CircuitHandle &handle,
    rust::Str type_name,
    bool has_tick_range,
    std::uint64_t tick_start,
    std::uint64_t tick_count,
    bool has_rows,
    std::size_t rows,
    rust::Vec<rust::String> filter_coords) {
  auto type = std::string_view(type_name.data(), type_name.size());
  std::vector<stim_draw_internal::CoordFilter> coord_filters;
  if (filter_coords.empty()) {
    coord_filters.push_back(stim_draw_internal::CoordFilter{});
  } else {
    coord_filters.reserve(filter_coords.size());
    for (const auto &entry : filter_coords) {
      coord_filters.push_back(stim_draw_internal::CoordFilter::parse_from(
          std::string_view(entry.data(), entry.size())));
    }
  }
  uint64_t tick_min = has_tick_range ? tick_start : 0;
  uint64_t num_ticks = has_tick_range ? tick_count : UINT64_MAX;
  size_t num_rows = has_rows ? rows : 0;

  if (type == "detslice-text" || type == "detector-slice-text") {
    std::stringstream out;
    stim_draw_internal::DetectorSliceSet::from_circuit_ticks(handle.get(), tick_min, num_ticks, coord_filters)
        .write_text_diagram_to(out);
    return rust::String(out.str());
  }
  if (
      type == "detslice-svg" || type == "detslice" || type == "detslice-html" || type == "detslice-svg-html" ||
      type == "detector-slice-svg" || type == "detector-slice") {
    std::stringstream out;
    stim_draw_internal::DetectorSliceSet::from_circuit_ticks(handle.get(), tick_min, num_ticks, coord_filters)
        .write_svg_diagram_to(out, num_rows);
    return rust::String(out.str());
  }
  if (
      type == "time-slice-svg" || type == "timeslice-svg" || type == "timeslice-html" ||
      type == "timeslice-svg-html" || type == "time-slice-html" || type == "time-slice-svg-html" ||
      type == "timeslice" || type == "time-slice") {
    std::stringstream out;
    stim_draw_internal::DiagramTimelineSvgDrawer::make_diagram_write_to(
        handle.get(),
        out,
        tick_min,
        num_ticks,
        stim_draw_internal::DiagramTimelineSvgDrawerMode::SVG_MODE_TIME_SLICE,
        coord_filters,
        num_rows);
    return rust::String(out.str());
  }
  if (
      type == "detslice-with-ops" || type == "detslice-with-ops-svg" || type == "detslice-with-ops-html" ||
      type == "detslice-with-ops-svg-html" || type == "time+detector-slice-svg") {
    std::stringstream out;
    stim_draw_internal::DiagramTimelineSvgDrawer::make_diagram_write_to(
        handle.get(),
        out,
        tick_min,
        num_ticks,
        stim_draw_internal::DiagramTimelineSvgDrawerMode::SVG_MODE_TIME_DETECTOR_SLICE,
        coord_filters,
        num_rows);
    return rust::String(out.str());
  }
  throw std::invalid_argument("Unrecognized diagram type: " + std::string(type));
}

std::unique_ptr<DetectorErrorModelHandle> detector_error_model_clone(
    const DetectorErrorModelHandle &handle) {
  return std::make_unique<DetectorErrorModelHandle>(handle.get());
}

std::unique_ptr<DetectorErrorModelHandle> detector_error_model_add(
    const DetectorErrorModelHandle &left,
    const DetectorErrorModelHandle &right) {
  return std::make_unique<DetectorErrorModelHandle>(left.get() + right.get());
}

void detector_error_model_add_assign(
    DetectorErrorModelHandle &left,
    const DetectorErrorModelHandle &right) {
  left.get() += right.get();
}

std::unique_ptr<DetectorErrorModelHandle> detector_error_model_mul(
    const DetectorErrorModelHandle &handle,
    std::uint64_t repetitions) {
  return std::make_unique<DetectorErrorModelHandle>(handle.get() * repetitions);
}

void detector_error_model_mul_assign(
    DetectorErrorModelHandle &handle,
    std::uint64_t repetitions) {
  handle.get() *= repetitions;
}

rust::String detector_error_model_to_dem_text(const DetectorErrorModelHandle &handle) {
  return rust::String(handle.get().str());
}

std::size_t detector_error_model_len(const DetectorErrorModelHandle &handle) {
  return handle.get().instructions.size();
}

std::uint64_t detector_error_model_num_detectors(const DetectorErrorModelHandle &handle) {
  return handle.get().count_detectors();
}

std::uint64_t detector_error_model_num_errors(const DetectorErrorModelHandle &handle) {
  return handle.get().count_errors();
}

std::uint64_t detector_error_model_num_observables(const DetectorErrorModelHandle &handle) {
  return handle.get().count_observables();
}

rust::String detector_error_model_get_detector_coordinates_text(
    const DetectorErrorModelHandle &handle,
    rust::Slice<const std::uint64_t> included_detector_indices) {
  std::set<std::uint64_t> included(
      included_detector_indices.begin(),
      included_detector_indices.end());
  auto coords = handle.get().get_detector_coordinates(included);
  std::stringstream ss;
  ss.precision(17);
  for (const auto &entry : coords) {
    ss << entry.first;
    for (double coord : entry.second) {
      ss << '\t' << coord;
    }
    ss << '\n';
  }
  return rust::String(ss.str());
}

void detector_error_model_clear(DetectorErrorModelHandle &handle) {
  handle.get().clear();
}

bool detector_error_model_equals(
    const DetectorErrorModelHandle &left,
    const DetectorErrorModelHandle &right) {
  return left.get() == right.get();
}

bool detector_error_model_approx_equals(
    const DetectorErrorModelHandle &left,
    const DetectorErrorModelHandle &right,
    double atol) {
  return left.get().approx_equals(right.get(), atol);
}

std::unique_ptr<DetectorErrorModelHandle> detector_error_model_without_tags(
    const DetectorErrorModelHandle &handle) {
  return std::make_unique<DetectorErrorModelHandle>(handle.get().without_tags());
}

std::unique_ptr<DetectorErrorModelHandle> detector_error_model_flattened(
    const DetectorErrorModelHandle &handle) {
  return std::make_unique<DetectorErrorModelHandle>(handle.get().flattened());
}

std::unique_ptr<DetectorErrorModelHandle> detector_error_model_rounded(
    const DetectorErrorModelHandle &handle,
    std::uint8_t digits) {
  return std::make_unique<DetectorErrorModelHandle>(handle.get().rounded(digits));
}

std::unique_ptr<DetectorErrorModelHandle> detector_error_model_shortest_graphlike_error(
    const DetectorErrorModelHandle &handle,
    bool ignore_ungraphlike_errors) {
  return std::make_unique<DetectorErrorModelHandle>(
      stim::shortest_graphlike_undetectable_logical_error(handle.get(), ignore_ungraphlike_errors));
}

rust::String detector_error_model_shortest_error_sat_problem(
    const DetectorErrorModelHandle &handle,
    rust::Str format_name) {
  return rust::String(stim::shortest_error_sat_problem(
      handle.get(), std::string_view(format_name.data(), format_name.size())));
}

rust::String detector_error_model_likeliest_error_sat_problem(
    const DetectorErrorModelHandle &handle,
    std::int32_t quantization,
    rust::Str format_name) {
  return rust::String(stim::likeliest_error_sat_problem(
      handle.get(), quantization, std::string_view(format_name.data(), format_name.size())));
}

std::unique_ptr<TableauHandle> tableau_clone(const TableauHandle &handle) {
  return std::make_unique<TableauHandle>(handle.get());
}

bool tableau_equals(const TableauHandle &left, const TableauHandle &right) {
  return left.get() == right.get();
}

std::unique_ptr<TableauHandle> tableau_add(
    const TableauHandle &left,
    const TableauHandle &right) {
  return std::make_unique<TableauHandle>(left.get() + right.get());
}

void tableau_add_assign(TableauHandle &left, const TableauHandle &right) {
  left.get() += right.get();
}

namespace {
std::vector<size_t> validated_tableau_targets(
    const stim::Tableau<stim::MAX_BITWORD_WIDTH> &tableau,
    const stim::Tableau<stim::MAX_BITWORD_WIDTH> &gate,
    rust::Slice<const std::size_t> targets) {
  std::vector<size_t> out;
  out.reserve(targets.size());
  std::vector<bool> use(tableau.num_qubits, false);
  if (targets.size() != gate.num_qubits) {
    throw std::invalid_argument("len(targets) != len(gate)");
  }
  for (size_t k : targets) {
    if (k >= tableau.num_qubits) {
      throw std::invalid_argument("target >= len(tableau)");
    }
    if (use[k]) {
      throw std::invalid_argument("target collision on qubit " + std::to_string(k));
    }
    use[k] = true;
    out.push_back(k);
  }
  return out;
}
}  // namespace

void tableau_append(
    TableauHandle &handle,
    const TableauHandle &gate,
    rust::Slice<const std::size_t> targets) {
  handle.get().inplace_scatter_append(
      gate.get(), validated_tableau_targets(handle.get(), gate.get(), targets));
}

void tableau_prepend(
    TableauHandle &handle,
    const TableauHandle &gate,
    rust::Slice<const std::size_t> targets) {
  handle.get().inplace_scatter_prepend(
      gate.get(), validated_tableau_targets(handle.get(), gate.get(), targets));
}

std::unique_ptr<TableauHandle> tableau_inverse(const TableauHandle &handle, bool unsigned_only) {
  return std::make_unique<TableauHandle>(handle.get().inverse(unsigned_only));
}

std::unique_ptr<TableauHandle> tableau_raised_to(const TableauHandle &handle, std::int64_t exponent) {
  return std::make_unique<TableauHandle>(handle.get().raised_to(exponent));
}

int tableau_x_sign(const TableauHandle &handle, std::size_t target) {
  if (target >= handle.get().num_qubits) {
    throw std::invalid_argument("not 0 <= target < len(tableau)");
  }
  return handle.get().xs.signs[target] ? -1 : +1;
}

int tableau_y_sign(const TableauHandle &handle, std::size_t target) {
  if (target >= handle.get().num_qubits) {
    throw std::invalid_argument("not 0 <= target < len(tableau)");
  }
  return handle.get().eval_y_obs(target).sign ? -1 : +1;
}

int tableau_z_sign(const TableauHandle &handle, std::size_t target) {
  if (target >= handle.get().num_qubits) {
    throw std::invalid_argument("not 0 <= target < len(tableau)");
  }
  return handle.get().zs.signs[target] ? -1 : +1;
}

std::uint8_t tableau_x_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index) {
  return handle.get().x_output_pauli_xyz(input_index, output_index);
}

std::uint8_t tableau_y_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index) {
  auto x = handle.get().x_output_pauli_xyz(input_index, output_index);
  auto z = handle.get().z_output_pauli_xyz(input_index, output_index);
  bool x_bit = (x == 1 || x == 2) ^ (z == 1 || z == 2);
  bool z_bit = (x == 2 || x == 3) ^ (z == 2 || z == 3);
  return static_cast<std::uint8_t>((x_bit ^ z_bit) | (static_cast<std::uint8_t>(z_bit) << 1));
}

std::uint8_t tableau_z_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index) {
  return handle.get().z_output_pauli_xyz(input_index, output_index);
}

std::uint8_t tableau_inverse_x_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index) {
  return handle.get().inverse_x_output_pauli_xyz(input_index, output_index);
}

std::uint8_t tableau_inverse_y_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index) {
  auto x = handle.get().inverse_x_output_pauli_xyz(input_index, output_index);
  auto z = handle.get().inverse_z_output_pauli_xyz(input_index, output_index);
  bool x_bit = (x == 1 || x == 2) ^ (z == 1 || z == 2);
  bool z_bit = (x == 2 || x == 3) ^ (z == 2 || z == 3);
  return static_cast<std::uint8_t>((x_bit ^ z_bit) | (static_cast<std::uint8_t>(z_bit) << 1));
}

std::uint8_t tableau_inverse_z_output_pauli(
    const TableauHandle &handle,
    std::size_t input_index,
    std::size_t output_index) {
  return handle.get().inverse_z_output_pauli_xyz(input_index, output_index);
}

std::unique_ptr<PauliStringHandle> tableau_x_output(
    const TableauHandle &handle,
    std::size_t target) {
  return std::make_unique<PauliStringHandle>(handle.get().xs[target]);
}

std::unique_ptr<PauliStringHandle> tableau_y_output(
    const TableauHandle &handle,
    std::size_t target) {
  return std::make_unique<PauliStringHandle>(handle.get().eval_y_obs(target));
}

std::unique_ptr<PauliStringHandle> tableau_z_output(
    const TableauHandle &handle,
    std::size_t target) {
  return std::make_unique<PauliStringHandle>(handle.get().zs[target]);
}

std::unique_ptr<PauliStringHandle> tableau_inverse_x_output(
    const TableauHandle &handle,
    std::size_t target,
    bool unsigned_only) {
  return std::make_unique<PauliStringHandle>(handle.get().inverse_x_output(target, unsigned_only));
}

std::unique_ptr<PauliStringHandle> tableau_inverse_y_output(
    const TableauHandle &handle,
    std::size_t target,
    bool unsigned_only) {
  return std::make_unique<PauliStringHandle>(handle.get().inverse_y_output(target, unsigned_only));
}

std::unique_ptr<PauliStringHandle> tableau_inverse_z_output(
    const TableauHandle &handle,
    std::size_t target,
    bool unsigned_only) {
  return std::make_unique<PauliStringHandle>(handle.get().inverse_z_output(target, unsigned_only));
}

std::unique_ptr<PauliStringHandle> tableau_conjugate_pauli_string(
    const TableauHandle &handle,
    const PauliStringHandle &pauli_string) {
  return std::make_unique<PauliStringHandle>(handle.get()(pauli_string.get().ref()));
}

std::unique_ptr<PauliStringHandle> tableau_conjugate_pauli_string_within(
    const TableauHandle &handle,
    const PauliStringHandle &pauli_string,
    rust::Slice<const std::size_t> targets,
    bool inverse) {
  auto validated = validated_tableau_targets(handle.get(), handle.get(), targets);
  stim::PauliString<stim::MAX_BITWORD_WIDTH> result = pauli_string.get();
  if (!validated.empty()) {
    auto max_target = *std::max_element(validated.begin(), validated.end());
    result.ensure_num_qubits(max_target + 1, 1.0);
  }
  auto result_ref = result.ref();
  if (inverse) {
    auto inv = handle.get().inverse(false);
    inv.apply_within(result_ref, validated);
  } else {
    handle.get().apply_within(result_ref, validated);
  }
  return std::make_unique<PauliStringHandle>(std::move(result));
}

rust::Vec<rust::String> tableau_to_stabilizer_texts(
    const TableauHandle &handle,
    bool canonicalize) {
  rust::Vec<rust::String> result;
  auto stabilizers = handle.get().stabilizers(canonicalize);
  result.reserve(stabilizers.size());
  for (const auto &stabilizer : stabilizers) {
    result.push_back(rust::String(stabilizer.str()));
  }
  return result;
}

std::unique_ptr<CircuitHandle> tableau_to_circuit(
    const TableauHandle &handle,
    rust::Str method) {
  return std::make_unique<CircuitHandle>(
      stim::tableau_to_circuit(handle.get(), std::string_view(method.data(), method.size())));
}

std::unique_ptr<PauliStringHandle> tableau_to_pauli_string(const TableauHandle &handle) {
  return std::make_unique<PauliStringHandle>(handle.get().to_pauli_string());
}

std::unique_ptr<TableauSimulatorHandle> tableau_simulator_clone(const TableauSimulatorHandle &handle) {
  return std::make_unique<TableauSimulatorHandle>(handle.get());
}

std::size_t tableau_simulator_num_qubits(const TableauSimulatorHandle &handle) {
  return handle.get().inv_state.num_qubits;
}

void tableau_simulator_set_num_qubits(TableauSimulatorHandle &handle, std::size_t new_num_qubits) {
  handle.get().set_num_qubits(new_num_qubits);
}

std::unique_ptr<TableauHandle> tableau_simulator_current_inverse_tableau(
    const TableauSimulatorHandle &handle) {
  return std::make_unique<TableauHandle>(handle.get().inv_state);
}

void tableau_simulator_set_inverse_tableau(
    TableauSimulatorHandle &handle,
    const TableauHandle &tableau) {
  handle.get().inv_state = tableau.get();
}

rust::Vec<std::uint8_t> tableau_simulator_current_measurement_record(
    const TableauSimulatorHandle &handle) {
  rust::Vec<std::uint8_t> result;
  result.reserve(handle.get().measurement_record.storage.size());
  for (bool bit : handle.get().measurement_record.storage) {
    result.push_back(bit ? 1 : 0);
  }
  return result;
}

void tableau_simulator_do_circuit(TableauSimulatorHandle &handle, const CircuitHandle &circuit) {
  handle.get().safe_do_circuit(circuit.get());
}

void tableau_simulator_do_pauli_string(
    TableauSimulatorHandle &handle,
    const PauliStringHandle &pauli_string) {
  handle.get().ensure_large_enough_for_qubits(pauli_string.get().num_qubits);
  handle.get().paulis(pauli_string.get());
}

void tableau_simulator_do_tableau(
    TableauSimulatorHandle &handle,
    const TableauHandle &tableau,
    rust::Slice<const std::size_t> targets) {
  std::vector<size_t> target_vec(targets.begin(), targets.end());
  size_t max_target = target_vec.empty() ? 0 : *std::max_element(target_vec.begin(), target_vec.end());
  if (!target_vec.empty()) {
    handle.get().ensure_large_enough_for_qubits(max_target + 1);
  }
  handle.get().apply_tableau(tableau.get(), target_vec);
}

std::unique_ptr<PauliStringHandle> tableau_simulator_peek_bloch(
    TableauSimulatorHandle &handle,
    std::size_t target) {
  handle.get().ensure_large_enough_for_qubits(target + 1);
  return std::make_unique<PauliStringHandle>(handle.get().peek_bloch(target));
}

int tableau_simulator_peek_x(TableauSimulatorHandle &handle, std::size_t target) {
  handle.get().ensure_large_enough_for_qubits(target + 1);
  return handle.get().peek_x(target);
}

int tableau_simulator_peek_y(TableauSimulatorHandle &handle, std::size_t target) {
  handle.get().ensure_large_enough_for_qubits(target + 1);
  return handle.get().peek_y(target);
}

int tableau_simulator_peek_z(TableauSimulatorHandle &handle, std::size_t target) {
  handle.get().ensure_large_enough_for_qubits(target + 1);
  return handle.get().peek_z(target);
}

bool tableau_simulator_measure(TableauSimulatorHandle &handle, std::size_t target) {
  handle.get().ensure_large_enough_for_qubits(target + 1);
  stim::GateTarget gate_target{static_cast<uint32_t>(target)};
  handle.get().do_MZ(stim::CircuitInstruction{stim::GateType::M, {}, &gate_target, ""});
  return handle.get().measurement_record.storage.back();
}

rust::Vec<std::uint8_t> tableau_simulator_measure_many(
    TableauSimulatorHandle &handle,
    rust::Slice<const std::size_t> targets) {
  std::vector<stim::GateTarget> gate_targets;
  gate_targets.reserve(targets.size());
  size_t max_target = 0;
  for (size_t target : targets) {
    max_target = std::max(max_target, target);
    gate_targets.push_back(stim::GateTarget{static_cast<uint32_t>(target)});
  }
  if (!gate_targets.empty()) {
    handle.get().ensure_large_enough_for_qubits(max_target + 1);
    handle.get().do_MZ(stim::CircuitInstruction{stim::GateType::M, {}, gate_targets, ""});
  }
  rust::Vec<std::uint8_t> result;
  result.reserve(gate_targets.size());
  auto end = handle.get().measurement_record.storage.end();
  auto begin = end - static_cast<std::ptrdiff_t>(gate_targets.size());
  for (auto it = begin; it != end; ++it) {
    result.push_back(*it ? 1 : 0);
  }
  return result;
}

int tableau_simulator_peek_observable_expectation(
    const TableauSimulatorHandle &handle,
    const PauliStringHandle &observable) {
  return handle.get().peek_observable_expectation(observable.get());
}

bool tableau_simulator_measure_observable(
    TableauSimulatorHandle &handle,
    const PauliStringHandle &observable,
    double flip_probability) {
  double p = flip_probability;
  double eps = std::numeric_limits<double>::epsilon();
  if (p == 0.0) {
    p = eps;
  } else if (p == 1.0) {
    p = 1.0 - eps;
  }
  return handle.get().measure_pauli_string(observable.get(), p);
}

void tableau_simulator_postselect_observable(
    TableauSimulatorHandle &handle,
    const PauliStringHandle &observable,
    bool desired_value) {
  handle.get().postselect_observable(observable.get().ref(), desired_value);
}

void tableau_simulator_postselect_x(
    TableauSimulatorHandle &handle,
    rust::Slice<const std::size_t> targets,
    bool desired_value) {
  std::vector<stim::GateTarget> gate_targets;
  gate_targets.reserve(targets.size());
  size_t max_target = 0;
  for (size_t target : targets) {
    max_target = std::max(max_target, target);
    gate_targets.push_back(stim::GateTarget{static_cast<uint32_t>(target)});
  }
  if (!gate_targets.empty()) {
    handle.get().ensure_large_enough_for_qubits(max_target + 1);
    handle.get().postselect_x(gate_targets, desired_value);
  }
}

void tableau_simulator_postselect_y(
    TableauSimulatorHandle &handle,
    rust::Slice<const std::size_t> targets,
    bool desired_value) {
  std::vector<stim::GateTarget> gate_targets;
  gate_targets.reserve(targets.size());
  size_t max_target = 0;
  for (size_t target : targets) {
    max_target = std::max(max_target, target);
    gate_targets.push_back(stim::GateTarget{static_cast<uint32_t>(target)});
  }
  if (!gate_targets.empty()) {
    handle.get().ensure_large_enough_for_qubits(max_target + 1);
    handle.get().postselect_y(gate_targets, desired_value);
  }
}

void tableau_simulator_postselect_z(
    TableauSimulatorHandle &handle,
    rust::Slice<const std::size_t> targets,
    bool desired_value) {
  std::vector<stim::GateTarget> gate_targets;
  gate_targets.reserve(targets.size());
  size_t max_target = 0;
  for (size_t target : targets) {
    max_target = std::max(max_target, target);
    gate_targets.push_back(stim::GateTarget{static_cast<uint32_t>(target)});
  }
  if (!gate_targets.empty()) {
    handle.get().ensure_large_enough_for_qubits(max_target + 1);
    handle.get().postselect_z(gate_targets, desired_value);
  }
}

TableauMeasureKickbackData tableau_simulator_measure_kickback(
    TableauSimulatorHandle &handle,
    std::size_t target) {
  handle.get().ensure_large_enough_for_qubits(target + 1);
  auto result = handle.get().measure_kickback_z({static_cast<uint32_t>(target)});
  TableauMeasureKickbackData out{
      .result = result.first,
      .has_kickback = result.second.num_qubits != 0,
      .kickback_text = "",
  };
  if (out.has_kickback) {
    out.kickback_text = rust::String(result.second.str());
  }
  return out;
}

std::unique_ptr<FrameSimulatorHandle> new_frame_simulator(
    std::size_t batch_size,
    bool disable_stabilizer_randomization,
    std::size_t num_qubits,
    std::uint64_t seed) {
  return std::make_unique<FrameSimulatorHandle>(
      batch_size, disable_stabilizer_randomization, num_qubits, seed);
}

std::unique_ptr<FrameSimulatorHandle> frame_simulator_clone(const FrameSimulatorHandle &handle) {
  return std::make_unique<FrameSimulatorHandle>(handle.get());
}

std::size_t frame_simulator_batch_size(const FrameSimulatorHandle &handle) {
  return handle.get().batch_size;
}

std::size_t frame_simulator_num_qubits(const FrameSimulatorHandle &handle) {
  return handle.get().num_qubits;
}

std::size_t frame_simulator_num_measurements(const FrameSimulatorHandle &handle) {
  return handle.get().m_record.stored;
}

std::size_t frame_simulator_num_detectors(const FrameSimulatorHandle &handle) {
  return handle.get().det_record.stored;
}

std::size_t frame_simulator_num_observables(const FrameSimulatorHandle &handle) {
  return handle.get().num_observables;
}

void frame_simulator_clear(FrameSimulatorHandle &handle) {
  handle.get().reset_all();
}

void frame_simulator_do_circuit(FrameSimulatorHandle &handle, const CircuitHandle &circuit) {
  handle.get().safe_do_circuit(circuit.get());
}

void frame_simulator_set_pauli_flip(
    FrameSimulatorHandle &handle,
    std::uint8_t pauli,
    std::int64_t qubit_index,
    std::int64_t instance_index) {
  if (qubit_index < 0) {
    throw std::out_of_range("qubit_index");
  }
  if (instance_index < 0) {
    instance_index += static_cast<std::int64_t>(handle.get().batch_size);
  }
  if (instance_index < 0 || static_cast<std::uint64_t>(instance_index) >= handle.get().batch_size) {
    throw std::out_of_range("instance_index");
  }
  if (pauli > 3) {
    throw std::invalid_argument("Need pauli in ['I', 'X', 'Y', 'Z', 0, 1, 2, 3, '_'].");
  }
  if (static_cast<std::uint64_t>(qubit_index) >= handle.get().num_qubits) {
    stim::CircuitStats stats;
    stats.num_qubits = static_cast<uint64_t>(qubit_index) + 1;
    handle.get().ensure_safe_to_do_circuit_with_stats(stats);
  }
  pauli ^= pauli >> 1;
  handle.get().x_table[qubit_index][instance_index] = (pauli & 1) != 0;
  handle.get().z_table[qubit_index][instance_index] = (pauli & 2) != 0;
}

rust::Vec<rust::String> frame_simulator_peek_pauli_flips(const FrameSimulatorHandle &handle) {
  rust::Vec<rust::String> out;
  out.reserve(handle.get().batch_size);
  for (size_t k = 0; k < handle.get().batch_size; k++) {
    out.push_back(rust::String(handle.get().get_frame(k).str()));
  }
  return out;
}

rust::String frame_simulator_peek_pauli_flip(
    const FrameSimulatorHandle &handle,
    std::int64_t instance_index) {
  size_t normalized = normalize_index_or_throw(instance_index, handle.get().batch_size);
  return rust::String(handle.get().get_frame(normalized).str());
}

void frame_simulator_broadcast_pauli_errors(
    FrameSimulatorHandle &handle,
    std::uint8_t pauli,
    rust::Vec<std::uint8_t> mask,
    std::size_t mask_num_qubits,
    float p) {
  if (!(0 <= p && p <= 1)) {
    throw std::invalid_argument("Need 0 <= p <= 1");
  }
  if (mask.size() != mask_num_qubits * handle.get().batch_size) {
    throw std::invalid_argument("mask must have length mask_num_qubits * batch_size");
  }
  if (pauli > 3) {
    throw std::invalid_argument("Need pauli in ['I', 'X', 'Y', 'Z', 0, 1, 2, 3, '_'].");
  }
  stim::CircuitStats stats;
  stats.num_qubits = mask_num_qubits;
  handle.get().ensure_safe_to_do_circuit_with_stats(stats);

  bool p_x = (0b0110 >> pauli) & 1;
  bool p_z = pauli & 2;
  size_t batch_size = handle.get().batch_size;
  if (p != 1 && p != 0) {
    for (size_t i = 0; i < mask_num_qubits; i++) {
      stim::biased_randomize_bits(
          p,
          handle.get().rng_buffer.u64,
          handle.get().rng_buffer.u64 + (batch_size / 64),
          handle.get().rng);
      for (size_t j = 0; j < batch_size; j++) {
        bool b = mask[i * batch_size + j] != 0;
        bool r = handle.get().rng_buffer[j];
        handle.get().x_table[i][j] ^= b & p_x & r;
        handle.get().z_table[i][j] ^= b & p_z & r;
      }
    }
  } else {
    for (size_t i = 0; i < mask_num_qubits; i++) {
      for (size_t j = 0; j < batch_size; j++) {
        bool b = mask[i * batch_size + j] != 0;
        handle.get().x_table[i][j] ^= b & p_x;
        handle.get().z_table[i][j] ^= b & p_z;
      }
    }
  }
}

rust::Vec<std::uint8_t> frame_simulator_generate_bernoulli_samples(
    FrameSimulatorHandle &handle,
    std::size_t num_samples,
    float p,
    bool bit_packed) {
  if (!(0 <= p && p <= 1)) {
    throw std::invalid_argument("Need 0 <= p <= 1");
  }
  rust::Vec<std::uint8_t> out;
  if (bit_packed) {
    size_t num_bytes = (num_samples + 7) / 8;
    out.reserve(num_bytes);
    std::vector<uint8_t> buffer(num_bytes);
    for (size_t k1 = 0; k1 < num_bytes; k1 += 64 * 8) {
      size_t n2 = std::min(num_bytes - k1, static_cast<size_t>(64 * 8));
      uint64_t stack[64];
      stim::biased_randomize_bits(p, &stack[0], &stack[0] + (n2 + 7) / 8, handle.get().rng);
      uint8_t *stack_data = reinterpret_cast<uint8_t *>(&stack[0]);
      for (size_t k2 = 0; k2 < n2; k2++) {
        buffer[k1 + k2] = stack_data[k2];
      }
    }
    if (num_samples & 0b111) {
      uint8_t mask_bits = (1 << (num_samples & 0b111)) - 1;
      buffer[num_bytes - 1] &= mask_bits;
    }
    for (uint8_t v : buffer) out.push_back(v);
  } else {
    out.reserve(num_samples);
    uint64_t stack[64];
    for (size_t k1 = 0; k1 < num_samples; k1 += 64 * 64) {
      size_t n2 = std::min(num_samples - k1, static_cast<size_t>(64 * 64));
      stim::biased_randomize_bits(p, &stack[0], &stack[0] + (n2 + 63) / 64, handle.get().rng);
      for (size_t k2 = 0; k2 < n2; k2++) {
        bool bit = (stack[k2 / 64] >> (k2 & 63)) & 1;
        out.push_back(bit ? 1 : 0);
      }
    }
  }
  return out;
}

void frame_simulator_append_measurement_flips(
    FrameSimulatorHandle &handle,
    rust::Vec<std::uint8_t> data,
    std::size_t num_measurements,
    bool bit_packed) {
  size_t batch_size = handle.get().batch_size;
  size_t row_width = bit_packed ? (batch_size + 7) / 8 : batch_size;
  if (data.size() != num_measurements * row_width) {
    throw std::invalid_argument("measurement flip data length does not match num_measurements and batch_size");
  }
  for (size_t k = 0; k < num_measurements; k++) {
    stim::simd_bits_range_ref<stim::MAX_BITWORD_WIDTH> r = handle.get().m_record.record_zero_result_to_edit();
    size_t offset = k * row_width;
    if (bit_packed) {
      std::memcpy(r.u8, data.data() + offset, row_width);
    } else {
      for (size_t j = 0; j < batch_size; j++) {
        r[j] = data[offset + j] != 0;
      }
    }
  }
}

BitTableData frame_simulator_get_measurement_flips(
    const FrameSimulatorHandle &handle,
    bool bit_packed) {
  return bit_table_to_data(
      handle.get().m_record.storage,
      handle.get().m_record.stored,
      handle.get().batch_size,
      bit_packed);
}

BitTableData frame_simulator_get_detector_flips(
    const FrameSimulatorHandle &handle,
    bool bit_packed) {
  return bit_table_to_data(
      handle.get().det_record.storage,
      handle.get().det_record.stored,
      handle.get().batch_size,
      bit_packed);
}

BitTableData frame_simulator_get_observable_flips(
    const FrameSimulatorHandle &handle,
    bool bit_packed) {
  return bit_table_to_data(
      handle.get().obs_record,
      handle.get().num_observables,
      handle.get().batch_size,
      bit_packed);
}

std::unique_ptr<CliffordStringHandle> clifford_string_clone(const CliffordStringHandle &handle) {
  return std::make_unique<CliffordStringHandle>(handle.get());
}

bool clifford_string_equals(const CliffordStringHandle &left, const CliffordStringHandle &right) {
  return left.get() == right.get();
}

std::size_t clifford_string_num_qubits(const CliffordStringHandle &handle) {
  return handle.get().num_qubits;
}

rust::String clifford_string_get_item_name(const CliffordStringHandle &handle, std::int64_t index) {
  auto normalized = normalize_index_or_throw(index, handle.get().num_qubits);
  auto name = stim::GATE_DATA[handle.get().gate_at(normalized)].name;
  return rust::String(name.data(), name.size());
}

std::unique_ptr<CliffordStringHandle> clifford_string_get_slice(
    const CliffordStringHandle &handle,
    std::int64_t start,
    std::int64_t step,
    std::int64_t slice_length) {
  return std::make_unique<CliffordStringHandle>(
      handle.get().py_get_slice(start, step, slice_length));
}

std::unique_ptr<CliffordStringHandle> clifford_string_add(
    const CliffordStringHandle &left,
    const CliffordStringHandle &right) {
  return std::make_unique<CliffordStringHandle>(left.get() + right.get());
}

void clifford_string_add_assign(CliffordStringHandle &left, const CliffordStringHandle &right) {
  left.get() += right.get();
}

std::unique_ptr<CliffordStringHandle> clifford_string_mul(
    const CliffordStringHandle &left,
    const CliffordStringHandle &right) {
  return std::make_unique<CliffordStringHandle>(left.get() * right.get());
}

void clifford_string_mul_assign(CliffordStringHandle &left, const CliffordStringHandle &right) {
  left.get() *= right.get();
}

std::unique_ptr<CliffordStringHandle> clifford_string_repeat(
    const CliffordStringHandle &handle,
    std::uint64_t repetitions) {
  size_t reps = static_cast<size_t>(repetitions);
  if (static_cast<std::uint64_t>(reps) != repetitions) {
    throw std::invalid_argument("repetitions overflowed size_t");
  }
  return std::make_unique<CliffordStringHandle>(handle.get() * reps);
}

void clifford_string_repeat_assign(CliffordStringHandle &handle, std::uint64_t repetitions) {
  size_t reps = static_cast<size_t>(repetitions);
  if (static_cast<std::uint64_t>(reps) != repetitions) {
    throw std::invalid_argument("repetitions overflowed size_t");
  }
  handle.get() *= reps;
}

std::unique_ptr<CliffordStringHandle> clifford_string_pow(
    const CliffordStringHandle &handle,
    std::int64_t exponent) {
  auto copy = handle.get();
  copy.ipow(exponent);
  return std::make_unique<CliffordStringHandle>(std::move(copy));
}

void clifford_string_ipow(CliffordStringHandle &handle, std::int64_t exponent) {
  handle.get().ipow(exponent);
}

std::unique_ptr<PauliStringHandle> clifford_string_x_outputs(const CliffordStringHandle &handle) {
  return std::make_unique<PauliStringHandle>(handle.get().x_outputs());
}

rust::Vec<std::uint8_t> clifford_string_x_signs_bit_packed(const CliffordStringHandle &handle) {
  return packed_bits_to_rust_vec(handle.get().x_signs, handle.get().num_qubits);
}

std::unique_ptr<PauliStringHandle> clifford_string_y_outputs(const CliffordStringHandle &handle) {
  stim::simd_bits<stim::MAX_BITWORD_WIDTH> signs(handle.get().num_qubits);
  return std::make_unique<PauliStringHandle>(handle.get().y_outputs_and_signs(signs));
}

rust::Vec<std::uint8_t> clifford_string_y_signs_bit_packed(const CliffordStringHandle &handle) {
  stim::simd_bits<stim::MAX_BITWORD_WIDTH> signs(handle.get().num_qubits);
  handle.get().y_outputs_and_signs(signs);
  return packed_bits_to_rust_vec(signs, handle.get().num_qubits);
}

std::unique_ptr<PauliStringHandle> clifford_string_z_outputs(const CliffordStringHandle &handle) {
  return std::make_unique<PauliStringHandle>(handle.get().z_outputs());
}

rust::Vec<std::uint8_t> clifford_string_z_signs_bit_packed(const CliffordStringHandle &handle) {
  return packed_bits_to_rust_vec(handle.get().z_signs, handle.get().num_qubits);
}

rust::String clifford_string_to_string(const CliffordStringHandle &handle) {
  return rust::String(handle.get().py_str());
}

rust::String clifford_string_to_repr(const CliffordStringHandle &handle) {
  return rust::String(handle.get().py_repr());
}

std::unique_ptr<PauliStringHandle> pauli_string_clone(const PauliStringHandle &handle) {
  return std::make_unique<PauliStringHandle>(handle.get());
}

bool pauli_string_equals(const PauliStringHandle &left, const PauliStringHandle &right) {
  return left.get() == right.get();
}

std::size_t pauli_string_num_qubits(const PauliStringHandle &handle) {
  return handle.get().num_qubits;
}

std::size_t pauli_string_weight(const PauliStringHandle &handle) {
  return handle.get().ref().weight();
}

std::uint8_t pauli_string_get_item(const PauliStringHandle &handle, std::int64_t index) {
  return handle.get().py_get_item(index);
}

void pauli_string_set_item(PauliStringHandle &handle, std::int64_t index, std::uint8_t new_pauli) {
  auto normalized = normalize_index_or_throw(index, handle.get().num_qubits);
  if (new_pauli > 3) {
    throw std::out_of_range("Expected new_pauli in [0, 1, 2, 3]");
  }
  int z = (new_pauli >> 1) & 1;
  int x = (new_pauli & 1) ^ z;
  handle.get().xs[normalized] = x;
  handle.get().zs[normalized] = z;
}

std::unique_ptr<PauliStringHandle> pauli_string_get_slice(
    const PauliStringHandle &handle,
    std::int64_t start,
    std::int64_t step,
    std::int64_t slice_length) {
  return std::make_unique<PauliStringHandle>(handle.get().py_get_slice(start, step, slice_length));
}

std::unique_ptr<PauliStringHandle> pauli_string_random(std::size_t num_qubits) {
  std::random_device rd;
  std::mt19937_64 rng(rd());
  return std::make_unique<PauliStringHandle>(
      stim::PauliString<stim::MAX_BITWORD_WIDTH>::random(num_qubits, rng));
}

std::unique_ptr<PauliStringIteratorHandle> pauli_string_iter_all(
    std::size_t num_qubits,
    std::size_t min_weight,
    std::size_t max_weight,
    bool allow_x,
    bool allow_y,
    bool allow_z) {
  return std::make_unique<PauliStringIteratorHandle>(
      num_qubits, min_weight, max_weight, allow_x, allow_y, allow_z);
}

std::unique_ptr<PauliStringIteratorHandle> pauli_string_iterator_clone(const PauliStringIteratorHandle &handle) {
  return std::make_unique<PauliStringIteratorHandle>(handle.get());
}

std::unique_ptr<PauliStringHandle> pauli_string_iterator_next(PauliStringIteratorHandle &handle) {
  if (!handle.get().iter_next()) {
    return nullptr;
  }
  return std::make_unique<PauliStringHandle>(handle.get().result);
}

bool pauli_string_commutes(const PauliStringHandle &handle, const PauliStringHandle &other) {
  return handle.get().ref().commutes(other.get().ref());
}

rust::Vec<std::uint64_t> pauli_string_pauli_indices(
    const PauliStringHandle &handle,
    rust::Str included_paulis) {
  bool keep_i = false;
  bool keep_x = false;
  bool keep_y = false;
  bool keep_z = false;
  for (char c : std::string_view(included_paulis.data(), included_paulis.size())) {
    switch (c) {
      case 'i':
      case 'I':
      case '_':
        keep_i = true;
        break;
      case 'x':
      case 'X':
        keep_x = true;
        break;
      case 'y':
      case 'Y':
        keep_y = true;
        break;
      case 'z':
      case 'Z':
        keep_z = true;
        break;
      default:
        throw std::invalid_argument("Invalid character in include string: " + std::string(1, c));
    }
  }

  rust::Vec<std::uint64_t> result;
  size_t n64 = handle.get().xs.num_u64_padded();
  for (size_t k = 0; k < n64; k++) {
    uint64_t x = handle.get().xs.u64[k];
    uint64_t z = handle.get().zs.u64[k];
    uint64_t u = 0;
    if (keep_i) {
      u |= ~x & ~z;
    }
    if (keep_x) {
      u |= x & ~z;
    }
    if (keep_y) {
      u |= x & z;
    }
    if (keep_z) {
      u |= ~x & z;
    }
    while (u) {
      uint8_t v = std::countr_zero(u);
      uint64_t q = k * 64 + v;
      if (q >= handle.get().num_qubits) {
        return result;
      }
      result.push_back(q);
      u &= u - 1;
    }
  }
  return result;
}

int pauli_string_sign_code(const PauliStringHandle &handle) {
  return handle.get().sign ? -1 : 1;
}

std::unique_ptr<TableauHandle> pauli_string_to_tableau(const PauliStringHandle &handle) {
  return std::make_unique<TableauHandle>(
      stim::Tableau<stim::MAX_BITWORD_WIDTH>::from_pauli_string(handle.get()));
}

rust::String pauli_string_to_string(const PauliStringHandle &handle) {
  return rust::String(handle.get().str());
}

rust::String pauli_string_to_repr(const PauliStringHandle &handle) {
  return rust::String("stim.PauliString(\"" + handle.get().str() + "\")");
}

std::size_t tableau_num_qubits(const TableauHandle &handle) {
  return handle.get().num_qubits;
}

rust::String tableau_to_string(const TableauHandle &handle) {
  return rust::String(handle.get().str());
}

rust::String tableau_to_repr(const TableauHandle &handle) {
  std::stringstream result;
  result << "stim.Tableau.from_conjugated_generators(\n    xs=[\n";
  for (size_t q = 0; q < handle.get().num_qubits; q++) {
    result << "        stim.PauliString(\"" << handle.get().xs[q].str() << "\"),\n";
  }
  result << "    ],\n    zs=[\n";
  for (size_t q = 0; q < handle.get().num_qubits; q++) {
    result << "        stim.PauliString(\"" << handle.get().zs[q].str() << "\"),\n";
  }
  result << "    ],\n)";
  return rust::String(result.str());
}

rust::Vec<float> tableau_to_unitary_matrix_data(const TableauHandle &handle, rust::Str endian) {
  auto endian_view = std::string_view(endian.data(), endian.size());
  bool little_endian;
  if (endian_view == "little") {
    little_endian = true;
  } else if (endian_view == "big") {
    little_endian = false;
  } else {
    throw std::invalid_argument("endian not in ['little', 'big']");
  }

  auto data = handle.get().to_flat_unitary_matrix(little_endian);
  rust::Vec<float> result;
  result.reserve(data.size() * 2);
  for (const auto &value : data) {
    result.push_back(value.real());
    result.push_back(value.imag());
  }
  return result;
}

rust::Vec<float> tableau_to_state_vector_data(const TableauHandle &handle, rust::Str endian) {
  auto endian_view = std::string_view(endian.data(), endian.size());
  bool little_endian;
  if (endian_view == "little") {
    little_endian = true;
  } else if (endian_view == "big") {
    little_endian = false;
  } else {
    throw std::invalid_argument("endian not in ['little', 'big']");
  }

  stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> sim(std::mt19937_64{0}, handle.get().num_qubits);
  sim.inv_state = handle.get().inverse(false);
  auto data = sim.to_state_vector(little_endian);

  rust::Vec<float> result;
  result.reserve(data.size() * 2);
  for (const auto &value : data) {
    result.push_back(value.real());
    result.push_back(value.imag());
  }
  return result;
}

std::unique_ptr<MeasurementSamplerHandle> circuit_compile_sampler(
    const CircuitHandle &handle,
    bool skip_reference_sample,
    std::uint64_t seed) {
  auto ref_sample = skip_reference_sample
      ? stim::simd_bits<stim::MAX_BITWORD_WIDTH>(handle.get().count_measurements())
      : stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(handle.get());
  return std::make_unique<MeasurementSamplerHandle>(
      std::move(ref_sample),
      handle.get(),
      skip_reference_sample,
      seed);
}

std::uint64_t measurement_sampler_num_measurements(const MeasurementSamplerHandle &handle) {
  return handle.num_measurements();
}

rust::Vec<std::uint8_t> measurement_sampler_sample_bit_packed(
    MeasurementSamplerHandle &handle,
    std::uint64_t shots) {
  return handle.sample_bit_packed(shots);
}

void measurement_sampler_sample_write(
    MeasurementSamplerHandle &handle,
    std::uint64_t shots,
    rust::Str filepath,
    rust::Str format_name) {
  handle.sample_write(shots, filepath, format_name);
}

std::unique_ptr<DetectorSamplerHandle> circuit_compile_detector_sampler(
    const CircuitHandle &handle,
    std::uint64_t seed) {
  return std::make_unique<DetectorSamplerHandle>(handle.get(), seed);
}

std::unique_ptr<DemSamplerHandle> detector_error_model_compile_sampler(
    const DetectorErrorModelHandle &handle,
    std::uint64_t seed) {
  return std::make_unique<DemSamplerHandle>(handle.get(), seed);
}

std::uint64_t detector_sampler_num_detectors(const DetectorSamplerHandle &handle) {
  return handle.num_detectors();
}

std::uint64_t detector_sampler_num_observables(const DetectorSamplerHandle &handle) {
  return handle.num_observables();
}

std::uint64_t dem_sampler_num_detectors(const DemSamplerHandle &handle) {
  return handle.num_detectors();
}

std::uint64_t dem_sampler_num_observables(const DemSamplerHandle &handle) {
  return handle.num_observables();
}

std::uint64_t dem_sampler_num_errors(const DemSamplerHandle &handle) {
  return handle.num_errors();
}

rust::Vec<std::uint8_t> detector_sampler_sample_bit_packed(
    DetectorSamplerHandle &handle,
    std::uint64_t shots) {
  return handle.sample_bit_packed(shots);
}

rust::Vec<std::uint8_t> detector_sampler_sample_observables_bit_packed(
    DetectorSamplerHandle &handle,
    std::uint64_t shots) {
  return handle.sample_observables_bit_packed(shots);
}

DemSampleBatch dem_sampler_sample_bit_packed(
    DemSamplerHandle &handle,
    std::uint64_t shots) {
  return handle.sample_bit_packed(shots);
}

DemSampleBatch dem_sampler_sample_bit_packed_replay(
    DemSamplerHandle &handle,
    rust::Slice<const std::uint8_t> recorded_errors,
    std::uint64_t shots) {
  return handle.sample_bit_packed_replay(recorded_errors, shots);
}

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
    bool replay_errors) {
  auto dets_format = sample_format_from_name(dets_format_name, "Unrecognized detector sample format.");
  auto obs_format = sample_format_from_name(obs_format_name, "Unrecognized observable sample format.");
  auto err_format = sample_format_from_name(err_format_name, "Unrecognized error sample format.");
  auto replay_err_format =
      sample_format_from_name(replay_err_format_name, "Unrecognized replay error sample format.");

  FILE *dets_out = fopen(std::string(dets_filepath).c_str(), "wb");
  if (dets_out == nullptr) {
    throw std::invalid_argument("Failed to open detector output file.");
  }
  FILE *obs_out = fopen(std::string(obs_filepath).c_str(), "wb");
  if (obs_out == nullptr) {
    fclose(dets_out);
    throw std::invalid_argument("Failed to open observable output file.");
  }
  FILE *err_out = nullptr;
  if (write_errors) {
    err_out = fopen(std::string(err_filepath).c_str(), "wb");
    if (err_out == nullptr) {
      fclose(dets_out);
      fclose(obs_out);
      throw std::invalid_argument("Failed to open error output file.");
    }
  }
  FILE *replay_err_in = nullptr;
  if (replay_errors) {
    replay_err_in = fopen(std::string(replay_err_filepath).c_str(), "rb");
    if (replay_err_in == nullptr) {
      fclose(dets_out);
      fclose(obs_out);
      if (err_out != nullptr) {
        fclose(err_out);
      }
      throw std::invalid_argument("Failed to open replay error input file.");
    }
  }

  handle.sample_write(
      shots,
      dets_out,
      dets_format,
      obs_out,
      obs_format,
      err_out,
      err_format,
      replay_err_in,
      replay_err_format);
  fclose(dets_out);
  fclose(obs_out);
  if (err_out != nullptr) {
    fclose(err_out);
  }
  if (replay_err_in != nullptr) {
    fclose(replay_err_in);
  }
}

void detector_sampler_sample_write(
    DetectorSamplerHandle &handle,
    std::uint64_t shots,
    rust::Str filepath,
    rust::Str format_name) {
  handle.sample_write(shots, filepath, format_name);
}

void detector_sampler_sample_write_separate_observables(
    DetectorSamplerHandle &handle,
    std::uint64_t shots,
    rust::Str dets_filepath,
    rust::Str dets_format_name,
    rust::Str obs_filepath,
    rust::Str obs_format_name) {
  handle.sample_write_separate_observables(
      shots, dets_filepath, dets_format_name, obs_filepath, obs_format_name);
}

std::unique_ptr<MeasurementsToDetectionEventsConverterHandle> circuit_compile_m2d_converter(
    const CircuitHandle &handle,
    bool skip_reference_sample) {
  auto ref_sample = skip_reference_sample
      ? stim::simd_bits<stim::MAX_BITWORD_WIDTH>(handle.get().count_measurements())
      : stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(handle.get());
  return std::make_unique<MeasurementsToDetectionEventsConverterHandle>(
      std::move(ref_sample), handle.get(), skip_reference_sample);
}

std::uint64_t m2d_converter_num_measurements(const MeasurementsToDetectionEventsConverterHandle &handle) {
  return handle.num_measurements();
}

std::uint64_t m2d_converter_num_detectors(const MeasurementsToDetectionEventsConverterHandle &handle) {
  return handle.num_detectors();
}

std::uint64_t m2d_converter_num_observables(const MeasurementsToDetectionEventsConverterHandle &handle) {
  return handle.num_observables();
}

std::uint64_t m2d_converter_num_sweep_bits(const MeasurementsToDetectionEventsConverterHandle &handle) {
  return handle.num_sweep_bits();
}

rust::Vec<std::uint8_t> m2d_converter_convert_measurements_bit_packed(
    MeasurementsToDetectionEventsConverterHandle &handle,
    rust::Slice<const std::uint8_t> measurements,
    std::uint64_t shots,
    bool append_observables) {
  return handle.convert_measurements_bit_packed(measurements, shots, append_observables);
}

rust::Vec<std::uint8_t> m2d_converter_convert_measurements_and_sweep_bits_bit_packed(
    MeasurementsToDetectionEventsConverterHandle &handle,
    rust::Slice<const std::uint8_t> measurements,
    rust::Slice<const std::uint8_t> sweep_bits,
    std::uint64_t shots,
    bool append_observables) {
  return handle.convert_measurements_and_sweep_bits_bit_packed(
      measurements, sweep_bits, shots, append_observables);
}

rust::Vec<std::uint8_t> m2d_converter_convert_observables_with_sweep_bits_bit_packed(
    MeasurementsToDetectionEventsConverterHandle &handle,
    rust::Slice<const std::uint8_t> measurements,
    rust::Slice<const std::uint8_t> sweep_bits,
    std::uint64_t shots) {
  return handle.convert_observables_with_sweep_bits_bit_packed(measurements, sweep_bits, shots);
}

rust::Vec<std::uint8_t> m2d_converter_convert_observables_bit_packed(
    MeasurementsToDetectionEventsConverterHandle &handle,
    rust::Slice<const std::uint8_t> measurements,
    std::uint64_t shots) {
  return handle.convert_observables_bit_packed(measurements, shots);
}

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
    rust::Str obs_out_format) {
  handle.convert_file(
      measurements_filepath,
      measurements_format,
      sweep_bits_filepath,
      sweep_bits_format,
      detection_events_filepath,
      detection_events_format,
      append_observables,
      obs_out_filepath,
      obs_out_format);
}

}  // namespace stimrs::bridge
