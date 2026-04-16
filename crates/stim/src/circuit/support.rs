use std::collections::BTreeMap;

use crate::{
    CircuitErrorLocation, CircuitErrorLocationStackFrame, CircuitTargetsInsideInstruction,
    DemTarget, DemTargetWithCoords, ExplainedError, FlippedMeasurement, GateTarget,
    GateTargetWithCoords, PauliString, Result, StimError,
};

pub fn parse_detecting_regions_text(
    text: &str,
) -> Result<BTreeMap<DemTarget, BTreeMap<u64, PauliString>>> {
    let mut result: BTreeMap<DemTarget, BTreeMap<u64, PauliString>> = BTreeMap::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(3, '\t');
        let target = parts
            .next()
            .ok_or_else(|| StimError::new("missing detecting-region target"))?;
        let tick = parts
            .next()
            .ok_or_else(|| StimError::new("missing detecting-region tick"))?;
        let pauli = parts
            .next()
            .ok_or_else(|| StimError::new("missing detecting-region pauli string"))?;

        let target = DemTarget::from_text(target)?;
        let tick = tick
            .parse::<u64>()
            .map_err(|_| StimError::new("failed to parse detecting-region tick"))?;
        let pauli = PauliString::from_text(pauli)?;
        result.entry(target).or_default().insert(tick, pauli);
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
