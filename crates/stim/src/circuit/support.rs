use std::collections::BTreeMap;

use crate::{
    CircuitErrorLocation, CircuitErrorLocationStackFrame, CircuitTargetsInsideInstruction,
    DemTarget, DemTargetWithCoords, ExplainedError, FlippedMeasurement, GateTarget,
    GateTargetWithCoords, PauliString, Result, StimError,
};

pub fn detecting_region_entries_to_map(
    entries: Vec<stim_cxx::DetectingRegionEntryData>,
) -> Result<BTreeMap<DemTarget, BTreeMap<u64, PauliString>>> {
    let mut result: BTreeMap<DemTarget, BTreeMap<u64, PauliString>> = BTreeMap::new();
    for entry in entries {
        let target = if entry.target_is_observable {
            DemTarget::logical_observable_id(entry.target_index)?
        } else {
            DemTarget::relative_detector_id(entry.target_index)?
        };
        result
            .entry(target)
            .or_default()
            .insert(entry.tick, PauliString::from_text(&entry.pauli)?);
    }
    Ok(result)
}

fn convert_gate_target_with_coords(
    data: stim_cxx::GateTargetWithCoordsData,
) -> GateTargetWithCoords {
    GateTargetWithCoords::new(GateTarget::from_raw_data(data.raw_target), data.coords)
}

fn convert_dem_target_with_coords(
    data: stim_cxx::DemTargetWithCoordsData,
) -> Result<DemTargetWithCoords> {
    Ok(DemTargetWithCoords::new(
        DemTarget::from_text(&data.dem_target)?,
        data.coords,
    ))
}

fn convert_circuit_error_location_stack_frame(
    data: stim_cxx::CircuitErrorLocationStackFrameData,
) -> CircuitErrorLocationStackFrame {
    CircuitErrorLocationStackFrame::new(
        data.instruction_offset,
        data.iteration_index,
        data.instruction_repetitions_arg,
    )
}

fn convert_circuit_targets_inside_instruction(
    data: stim_cxx::CircuitTargetsInsideInstructionData,
) -> Result<CircuitTargetsInsideInstruction> {
    Ok(CircuitTargetsInsideInstruction::new(
        data.gate,
        data.tag,
        data.args,
        usize::try_from(data.target_range_start)
            .map_err(|_| StimError::new("target_range_start overflow"))?,
        usize::try_from(data.target_range_end)
            .map_err(|_| StimError::new("target_range_end overflow"))?,
        data.targets_in_range
            .into_iter()
            .map(convert_gate_target_with_coords)
            .collect(),
    ))
}

fn convert_flipped_measurement(data: stim_cxx::FlippedMeasurementData) -> FlippedMeasurement {
    FlippedMeasurement::new(
        if data.record_index == u64::MAX {
            None
        } else {
            Some(data.record_index)
        },
        data.observable
            .into_iter()
            .map(convert_gate_target_with_coords)
            .collect::<Vec<_>>(),
    )
}

fn convert_circuit_error_location(
    data: stim_cxx::CircuitErrorLocationData,
) -> Result<CircuitErrorLocation> {
    Ok(CircuitErrorLocation::new(
        data.tick_offset,
        data.flipped_pauli_product
            .into_iter()
            .map(convert_gate_target_with_coords)
            .collect::<Vec<_>>(),
        {
            let measurement = convert_flipped_measurement(data.flipped_measurement);
            if measurement.record_index().is_none() && measurement.observable().is_empty() {
                None
            } else {
                Some(measurement)
            }
        },
        convert_circuit_targets_inside_instruction(data.instruction_targets)?,
        data.stack_frames
            .into_iter()
            .map(convert_circuit_error_location_stack_frame)
            .collect::<Vec<_>>(),
        data.noise_tag,
    ))
}

pub fn convert_explained_error(data: stim_cxx::ExplainedErrorData) -> Result<ExplainedError> {
    Ok(ExplainedError::new(
        data.dem_error_terms
            .into_iter()
            .map(convert_dem_target_with_coords)
            .collect::<Result<Vec<_>>>()?,
        data.circuit_error_locations
            .into_iter()
            .map(convert_circuit_error_location)
            .collect::<Result<Vec<_>>>()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::{convert_explained_error, detecting_region_entries_to_map};

    #[test]
    fn support_parsers_and_converters_cover_blank_lines_and_measurement_details() {
        let parsed = detecting_region_entries_to_map(vec![stim_cxx::DetectingRegionEntryData {
            target_index: 0,
            target_is_observable: false,
            tick: 4,
            pauli: "+X".to_string(),
        }])
        .unwrap();
        assert_eq!(
            parsed,
            std::collections::BTreeMap::from([(
                crate::DemTarget::relative_detector_id(0).unwrap(),
                std::collections::BTreeMap::from([(
                    4,
                    crate::PauliString::from_text("+X").unwrap()
                )]),
            )])
        );

        let explained = convert_explained_error(stim_cxx::ExplainedErrorData {
            dem_error_terms: vec![stim_cxx::DemTargetWithCoordsData {
                dem_target: "D0".to_string(),
                coords: vec![1.0, 2.0],
            }],
            circuit_error_locations: vec![stim_cxx::CircuitErrorLocationData {
                tick_offset: 7,
                flipped_pauli_product: vec![],
                flipped_measurement: stim_cxx::FlippedMeasurementData {
                    record_index: 3,
                    observable: vec![],
                },
                instruction_targets: stim_cxx::CircuitTargetsInsideInstructionData {
                    gate: "M".to_string(),
                    tag: String::new(),
                    args: vec![],
                    target_range_start: 0,
                    target_range_end: 1,
                    targets_in_range: vec![stim_cxx::GateTargetWithCoordsData {
                        raw_target: crate::GateTarget::from(0u32).raw_data(),
                        coords: vec![0.0],
                    }],
                },
                stack_frames: vec![],
                noise_tag: "noise".to_string(),
            }],
        })
        .unwrap();

        assert_eq!(
            explained.circuit_error_locations()[0]
                .flipped_measurement()
                .unwrap()
                .record_index(),
            Some(3)
        );
    }
}
