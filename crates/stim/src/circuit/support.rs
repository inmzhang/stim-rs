use std::collections::BTreeMap;
use std::str::FromStr;

use crate::{
    Circuit, CircuitErrorLocation, CircuitErrorLocationStackFrame, CircuitInstruction, CircuitItem,
    CircuitRepeatBlock, CircuitTargetsInsideInstruction, DemTarget, DemTargetWithCoords,
    ExplainedError, FlippedMeasurement, GateTarget, GateTargetWithCoords, PauliString, Result,
    StimError,
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

pub fn split_top_level_circuit_items(text: &str) -> Result<Vec<String>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let mut items = Vec::new();
    let mut current = Vec::new();
    let mut depth = 0isize;

    for line in text.lines() {
        if depth == 0 && current.is_empty() && !line.starts_with("REPEAT") {
            items.push(line.to_string());
            continue;
        }

        depth += line.matches('{').count() as isize;
        current.push(line.to_string());
        depth -= line.matches('}').count() as isize;

        if depth < 0 {
            return Err(StimError::new("unbalanced circuit repeat block braces"));
        }
        if depth == 0 {
            items.push(current.join("\n"));
            current.clear();
        }
    }

    if depth != 0 || !current.is_empty() {
        return Err(StimError::new("unterminated circuit repeat block"));
    }

    Ok(items)
}

pub fn parse_circuit_item(text: &str) -> Result<CircuitItem> {
    if let Some((header, body)) = text.split_once("{\n") {
        let header = header.trim();
        let rest = header.strip_prefix("REPEAT").ok_or_else(|| {
            StimError::new(format!("invalid circuit repeat block header: {header}"))
        })?;
        let (tag, count_text) = if let Some(rest) = rest.strip_prefix('[') {
            let close = rest
                .find(']')
                .ok_or_else(|| StimError::new("unterminated circuit repeat block tag"))?;
            let tag = &rest[..close];
            let count_text = rest[close + 1..].trim();
            (tag.to_string(), count_text.to_string())
        } else {
            (String::new(), rest.trim().to_string())
        };
        let repeat_count = count_text
            .parse::<u64>()
            .map_err(|_| StimError::new(format!("invalid repeat count: {count_text}")))?;
        let inner = body
            .strip_suffix("\n}")
            .ok_or_else(|| StimError::new("invalid circuit repeat block body"))?;
        let inner = inner
            .lines()
            .map(|line| line.strip_prefix("    ").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(CircuitItem::repeat_block(CircuitRepeatBlock::new(
            repeat_count,
            &Circuit::from_str(&inner)?,
            tag,
        )?))
    } else {
        Ok(CircuitItem::instruction(CircuitInstruction::from_str(
            text,
        )?))
    }
}
