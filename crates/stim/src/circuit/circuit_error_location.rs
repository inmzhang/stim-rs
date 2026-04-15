use std::fmt::{self, Display, Formatter};

use crate::{
    CircuitErrorLocationStackFrame, CircuitTargetsInsideInstruction, FlippedMeasurement,
    GateTargetWithCoords,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CircuitErrorLocation {
    tick_offset: u64,
    flipped_pauli_product: Vec<GateTargetWithCoords>,
    flipped_measurement: Option<FlippedMeasurement>,
    instruction_targets: CircuitTargetsInsideInstruction,
    stack_frames: Vec<CircuitErrorLocationStackFrame>,
    noise_tag: String,
}

impl CircuitErrorLocation {
    #[must_use]
    pub fn new(
        tick_offset: u64,
        flipped_pauli_product: impl IntoIterator<Item = GateTargetWithCoords>,
        flipped_measurement: Option<FlippedMeasurement>,
        instruction_targets: CircuitTargetsInsideInstruction,
        stack_frames: impl IntoIterator<Item = CircuitErrorLocationStackFrame>,
        noise_tag: impl Into<String>,
    ) -> Self {
        Self {
            tick_offset,
            flipped_pauli_product: flipped_pauli_product.into_iter().collect(),
            flipped_measurement,
            instruction_targets,
            stack_frames: stack_frames.into_iter().collect(),
            noise_tag: noise_tag.into(),
        }
    }

    #[must_use]
    pub fn tick_offset(&self) -> u64 {
        self.tick_offset
    }

    #[must_use]
    pub fn flipped_pauli_product(&self) -> &[GateTargetWithCoords] {
        &self.flipped_pauli_product
    }

    #[must_use]
    pub fn flipped_measurement(&self) -> Option<&FlippedMeasurement> {
        self.flipped_measurement.as_ref()
    }

    #[must_use]
    pub fn instruction_targets(&self) -> &CircuitTargetsInsideInstruction {
        &self.instruction_targets
    }

    #[must_use]
    pub fn stack_frames(&self) -> &[CircuitErrorLocationStackFrame] {
        &self.stack_frames
    }

    #[must_use]
    pub fn noise_tag(&self) -> &str {
        &self.noise_tag
    }
}

impl Display for CircuitErrorLocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(
            "CircuitErrorLocation {
",
        )?;
        if !self.noise_tag.is_empty() {
            writeln!(f, "    noise_tag: {}", self.noise_tag)?;
        }
        if !self.flipped_pauli_product.is_empty() {
            f.write_str("    flipped_pauli_product: ")?;
            write_gate_target_product(f, &self.flipped_pauli_product)?;
            f.write_str(
                "
",
            )?;
        }
        if let Some(flipped_measurement) = &self.flipped_measurement {
            if let Some(record_index) = flipped_measurement.record_index() {
                writeln!(f, "    flipped_measurement.record_index: {record_index}")?;
            }
            if !flipped_measurement.observable().is_empty() {
                f.write_str("    flipped_measurement.observable: ")?;
                write_gate_target_product(f, flipped_measurement.observable())?;
                f.write_str(
                    "
",
                )?;
            }
        }

        f.write_str(
            "    Circuit location stack trace:
",
        )?;
        writeln!(f, "        (after {} TICKs)", self.tick_offset)?;
        for (index, frame) in self.stack_frames.iter().enumerate() {
            if index > 0 {
                writeln!(
                    f,
                    "        after {} completed iterations",
                    frame.iteration_index()
                )?;
            }
            write!(
                f,
                "        at instruction #{}",
                frame.instruction_offset() + 1
            )?;
            if index + 1 < self.stack_frames.len() {
                write!(
                    f,
                    " (a REPEAT {} block)",
                    frame.instruction_repetitions_arg()
                )?;
            } else {
                write!(f, " ({})", self.instruction_targets.gate())?;
            }
            if index > 0 {
                f.write_str(
                    " in the REPEAT block
",
                )?;
            } else {
                f.write_str(
                    " in the circuit
",
                )?;
            }
        }
        if self.instruction_targets.target_range_start() + 1
            == self.instruction_targets.target_range_end()
        {
            writeln!(
                f,
                "        at target #{} of the instruction",
                self.instruction_targets.target_range_start() + 1
            )?;
        } else {
            writeln!(
                f,
                "        at targets #{} to #{} of the instruction",
                self.instruction_targets.target_range_start() + 1,
                self.instruction_targets.target_range_end()
            )?;
        }
        write!(
            f,
            "        resolving to {}
}}",
            self.instruction_targets
        )
    }
}

impl fmt::Debug for CircuitErrorLocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("stim::CircuitErrorLocation")
            .field("tick_offset", &self.tick_offset)
            .field("flipped_pauli_product", &self.flipped_pauli_product)
            .field("flipped_measurement", &self.flipped_measurement)
            .field("instruction_targets", &self.instruction_targets)
            .field("stack_frames", &self.stack_frames)
            .field("noise_tag", &self.noise_tag)
            .finish()
    }
}

fn write_gate_target_product(
    f: &mut Formatter<'_>,
    product: &[GateTargetWithCoords],
) -> fmt::Result {
    for (index, target) in product.iter().enumerate() {
        if index > 0 && !target.gate_target().is_combiner() {
            f.write_str(" ")?;
        }
        write!(f, "{target}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    use crate::{GateTarget, target_x, target_y};

    fn sample_instruction_targets() -> CircuitTargetsInsideInstruction {
        CircuitTargetsInsideInstruction::new(
            "X_ERROR",
            "",
            vec![0.25],
            2,
            5,
            vec![
                GateTargetWithCoords::new(GateTarget::new(5u32), vec![1.0, 2.0]),
                GateTargetWithCoords::new(GateTarget::new(6u32), vec![1.0, 3.0]),
                GateTargetWithCoords::new(GateTarget::new(7u32), vec![]),
            ],
        )
    }

    fn sample_flipped_measurement() -> FlippedMeasurement {
        FlippedMeasurement::new(
            Some(5),
            [GateTargetWithCoords::new(
                target_x(5u32, false).expect("X target should build"),
                vec![1.0, 2.0, 3.0],
            )],
        )
    }

    fn sample_stack_frames() -> Vec<CircuitErrorLocationStackFrame> {
        vec![
            CircuitErrorLocationStackFrame::new(1, 0, 3),
            CircuitErrorLocationStackFrame::new(1, 2, 0),
        ]
    }

    #[test]
    fn constructor_and_accessors_round_trip() {
        let flipped_pauli_product = vec![GateTargetWithCoords::new(
            target_y(6u32, false).expect("Y target should build"),
            vec![1.0, 2.0, 3.0],
        )];
        let flipped_measurement = sample_flipped_measurement();
        let instruction_targets = sample_instruction_targets();
        let stack_frames = sample_stack_frames();

        let location = CircuitErrorLocation::new(
            5,
            flipped_pauli_product.clone(),
            Some(flipped_measurement.clone()),
            instruction_targets.clone(),
            stack_frames.clone(),
            "test-tag",
        );

        assert_eq!(location.tick_offset(), 5);
        assert_eq!(
            location.flipped_pauli_product(),
            flipped_pauli_product.as_slice()
        );
        assert_eq!(location.flipped_measurement(), Some(&flipped_measurement));
        assert_eq!(location.instruction_targets(), &instruction_targets);
        assert_eq!(location.stack_frames(), stack_frames.as_slice());
        assert_eq!(location.noise_tag(), "test-tag");
    }

    #[test]
    fn supports_empty_measurement_details() {
        let instruction_targets = sample_instruction_targets();
        let no_measurement = CircuitErrorLocation::new(
            0,
            vec![],
            None,
            instruction_targets.clone(),
            vec![CircuitErrorLocationStackFrame::new(2, 0, 0)],
            "",
        );
        let empty_measurement = CircuitErrorLocation::new(
            1,
            vec![],
            Some(FlippedMeasurement::new(None, Vec::new())),
            instruction_targets,
            vec![CircuitErrorLocationStackFrame::new(2, 0, 0)],
            "noise-tag",
        );

        assert_eq!(no_measurement.flipped_measurement(), None);
        let flipped_measurement = empty_measurement
            .flipped_measurement()
            .expect("measurement should exist");
        assert_eq!(flipped_measurement.record_index(), None);
        assert!(flipped_measurement.observable().is_empty());
        assert_eq!(
            empty_measurement.to_string(),
            "CircuitErrorLocation {
    noise_tag: noise-tag
    Circuit location stack trace:
        (after 1 TICKs)
        at instruction #3 (X_ERROR) in the circuit
        at targets #3 to #5 of the instruction
        resolving to X_ERROR(0.25) 5[coords 1,2] 6[coords 1,3] 7
}"
        );
    }

    #[test]
    fn supports_equality_order_clone_and_hash() {
        let first = CircuitErrorLocation::new(
            5,
            vec![GateTargetWithCoords::new(
                target_y(6u32, false).expect("Y target should build"),
                vec![1.0, 2.0, 3.0],
            )],
            Some(sample_flipped_measurement()),
            sample_instruction_targets(),
            sample_stack_frames(),
            "tag-a",
        );
        let same_as_first = first.clone();
        let later_tick = CircuitErrorLocation::new(
            6,
            vec![GateTargetWithCoords::new(
                target_y(6u32, false).expect("Y target should build"),
                vec![1.0, 2.0, 3.0],
            )],
            Some(sample_flipped_measurement()),
            sample_instruction_targets(),
            sample_stack_frames(),
            "tag-a",
        );
        let different_tag = CircuitErrorLocation::new(
            5,
            vec![GateTargetWithCoords::new(
                target_y(6u32, false).expect("Y target should build"),
                vec![1.0, 2.0, 3.0],
            )],
            Some(sample_flipped_measurement()),
            sample_instruction_targets(),
            sample_stack_frames(),
            "tag-b",
        );

        assert_eq!(first, same_as_first);
        assert_ne!(first, later_tick);
        assert_ne!(first, different_tag);

        let ordered = [
            different_tag.clone(),
            later_tick.clone(),
            same_as_first.clone(),
            first.clone(),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
        assert_eq!(
            ordered,
            vec![first.clone(), different_tag.clone(), later_tick]
        );

        let hashed = [first.clone(), same_as_first, different_tag.clone()]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 2);
        assert!(hashed.contains(&first));
        assert!(hashed.contains(&different_tag));
    }

    #[test]
    fn display_and_debug_are_stable() {
        let location = CircuitErrorLocation::new(
            5,
            vec![GateTargetWithCoords::new(
                target_y(6u32, false).expect("Y target should build"),
                vec![1.0, 2.0, 3.0],
            )],
            Some(sample_flipped_measurement()),
            sample_instruction_targets(),
            sample_stack_frames(),
            "test-tag",
        );

        assert_eq!(
            location.to_string(),
            "CircuitErrorLocation {
    noise_tag: test-tag
    flipped_pauli_product: Y6[coords 1,2,3]
    flipped_measurement.record_index: 5
    flipped_measurement.observable: X5[coords 1,2,3]
    Circuit location stack trace:
        (after 5 TICKs)
        at instruction #2 (a REPEAT 3 block) in the circuit
        after 2 completed iterations
        at instruction #2 (X_ERROR) in the REPEAT block
        at targets #3 to #5 of the instruction
        resolving to X_ERROR(0.25) 5[coords 1,2] 6[coords 1,3] 7
}"
        );
        assert_eq!(
            format!("{location:?}"),
            r#"stim::CircuitErrorLocation { tick_offset: 5, flipped_pauli_product: [stim::GateTargetWithCoords { gate_target: stim::target_y(6), coords: [1.0, 2.0, 3.0] }], flipped_measurement: Some(stim::FlippedMeasurement(
    record_index=5,
    observable=(stim::GateTargetWithCoords(stim::target_x(5), [1.0, 2.0, 3.0]),),
)), instruction_targets: stim::CircuitTargetsInsideInstruction(gate="X_ERROR", tag="", args=[0.25], target_range_start=2, target_range_end=5, targets_in_range=(stim::GateTargetWithCoords(stim::GateTarget(5), [1.0, 2.0]), stim::GateTargetWithCoords(stim::GateTarget(6), [1.0, 3.0]), stim::GateTargetWithCoords(stim::GateTarget(7), []))), stack_frames: [stim::CircuitErrorLocationStackFrame { instruction_offset: 1, iteration_index: 0, instruction_repetitions_arg: 3 }, stim::CircuitErrorLocationStackFrame { instruction_offset: 1, iteration_index: 2, instruction_repetitions_arg: 0 }], noise_tag: "test-tag" }"#
        );
    }
}
