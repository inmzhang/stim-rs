use std::fmt::{self, Display, Formatter};

use crate::{
    CircuitErrorLocationStackFrame, CircuitTargetsInsideInstruction, FlippedMeasurement,
    GateTargetWithCoords,
};

/// Describes the location of an error mechanism within a Stim circuit.
///
/// When Stim explains how a particular fault affects detectors and
/// observables (e.g. via [`Circuit::shortest_graphlike_error`]), each
/// fault is localized to a specific instruction and target range inside
/// the circuit. This struct bundles that location information together
/// with the Pauli product that was flipped, the measurement that was
/// flipped (for measurement errors), a stack trace through nested
/// `REPEAT` blocks, and a noise tag string.
///
/// The location uniquely identifies a single error site by providing:
///
/// - **`tick_offset`** -- the number of `TICK` instructions that executed
///   before the error, counting TICKs inside loops multiple times.
/// - **`flipped_pauli_product`** -- the Pauli operators (with qubit
///   coordinates) applied by the error mechanism. Empty for pure
///   measurement errors.
/// - **`flipped_measurement`** -- for measurement errors, which
///   measurement record entry was flipped and what observable it
///   corresponds to. `None` for purely Pauli errors.
/// - **`instruction_targets`** -- the gate name, arguments, and the
///   subset of that instruction's targets that participated in the error.
/// - **`stack_frames`** -- a stack trace from the top-level circuit down
///   through nested `REPEAT` blocks to the instruction that caused the
///   error.
/// - **`noise_tag`** -- the custom `[tag]` annotation on the noise
///   instruction, or `""` if none was set.
///
/// Instances are typically obtained from [`crate::ExplainedError`] rather
/// than constructed directly.
///
/// # Examples
///
/// ```
/// use stim::{
///     CircuitErrorLocation, CircuitErrorLocationStackFrame,
///     CircuitTargetsInsideInstruction, GateTarget, GateTargetWithCoords,
/// };
///
/// let location = CircuitErrorLocation::new(
///     3, // after 3 TICKs
///     vec![GateTargetWithCoords::new(
///         stim::GateTarget::x(0u32, false).expect("valid target"),
///         vec![],
///     )],
///     None, // no flipped measurement (Pauli error)
///     CircuitTargetsInsideInstruction::new(
///         "DEPOLARIZE1",
///         "",
///         vec![0.001],
///         0,
///         1,
///         vec![GateTargetWithCoords::new(GateTarget::from(0u32), vec![])],
///     ),
///     vec![CircuitErrorLocationStackFrame::new(2, 0, 0)],
///     "",
/// );
///
/// assert_eq!(location.tick_offset(), 3);
/// assert!(location.flipped_measurement().is_none());
/// assert_eq!(location.flipped_pauli_product().len(), 1);
/// assert_eq!(location.stack_frames().len(), 1);
/// ```
///
/// [`Circuit::shortest_graphlike_error`]: crate::Circuit::shortest_graphlike_error
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
    /// Creates a new error location from its constituent parts.
    ///
    /// This constructor assembles all the pieces that identify where an
    /// error occurred in a circuit, what Pauli operators it flipped, and
    /// whether it also flipped a measurement.
    ///
    /// # Arguments
    ///
    /// - `tick_offset` -- number of `TICK` instructions that preceded the
    ///   error in the circuit's execution timeline (counting loop
    ///   iterations).
    /// - `flipped_pauli_product` -- the Pauli operators (with qubit
    ///   coordinates) applied by the error. Pass an empty collection for
    ///   pure measurement errors.
    /// - `flipped_measurement` -- `Some(...)` when the error flips a
    ///   measurement outcome, `None` for purely Pauli errors.
    /// - `instruction_targets` -- the gate name, arguments, and target
    ///   sub-range that identify the specific operation that caused the
    ///   error.
    /// - `stack_frames` -- the nesting stack trace from the top-level
    ///   circuit down through `REPEAT` blocks to the instruction. The
    ///   outermost frame should come first.
    /// - `noise_tag` -- the `[tag]` annotation on the noise instruction,
    ///   or `""` if none.
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

    /// Returns the number of `TICK` instructions that precede this error
    /// location in the circuit's execution timeline.
    ///
    /// This counts TICKs occurring multiple times during loops. For
    /// example, a `TICK` inside a `REPEAT 5 { ... }` block that has
    /// fully completed contributes 5 to the tick offset.
    ///
    /// # Examples
    ///
    /// ```
    /// # use stim::*;
    /// let location = CircuitErrorLocation::new(
    ///     3,
    ///     Vec::<GateTargetWithCoords>::new(),
    ///     None,
    ///     CircuitTargetsInsideInstruction::new(
    ///         "X_ERROR", "", vec![0.1], 0, 1,
    ///         vec![GateTargetWithCoords::new(GateTarget::from(0u32), vec![])],
    ///     ),
    ///     vec![CircuitErrorLocationStackFrame::new(2, 0, 0)],
    ///     "",
    /// );
    /// assert_eq!(location.tick_offset(), 3);
    /// ```
    #[must_use]
    pub fn tick_offset(&self) -> u64 {
        self.tick_offset
    }

    /// Returns the Pauli operators (with qubit coordinates) that are
    /// flipped by this error mechanism.
    ///
    /// Each element describes a single-qubit Pauli (X, Y, or Z) applied
    /// to a specific qubit, together with the qubit's coordinate data
    /// from any `QUBIT_COORDS` instructions in the circuit.
    ///
    /// When the error is a pure measurement error, this slice is empty.
    #[must_use]
    pub fn flipped_pauli_product(&self) -> &[GateTargetWithCoords] {
        &self.flipped_pauli_product
    }

    /// Returns the measurement that is flipped by this error, if the error
    /// is a measurement error.
    ///
    /// For measurement errors (e.g. from `M(p)` instructions), this
    /// returns the measurement record index and the observable that the
    /// measurement corresponds to. For purely Pauli errors that do not
    /// flip any measurement outcome (e.g. from `X_ERROR` or
    /// `DEPOLARIZE1`), this returns `None`.
    #[must_use]
    pub fn flipped_measurement(&self) -> Option<&FlippedMeasurement> {
        self.flipped_measurement.as_ref()
    }

    /// Returns the resolved instruction and target range where the error
    /// occurs.
    ///
    /// An error instruction (such as `X_ERROR(0.25) 0 1 2 3`) may have
    /// many targets, but only a subset of those targets is involved in a
    /// given error. This accessor provides the gate name, its arguments,
    /// and the specific target sub-range (with coordinates) that
    /// produced the error.
    #[must_use]
    pub fn instruction_targets(&self) -> &CircuitTargetsInsideInstruction {
        &self.instruction_targets
    }

    /// Returns the stack of frames that locate this error within nested
    /// `REPEAT` blocks.
    ///
    /// The outermost frame (index 0) refers to the top-level circuit. Each
    /// subsequent frame descends one level into a `REPEAT` block. The
    /// innermost frame identifies the actual instruction that caused the
    /// error.
    ///
    /// Multiple frames are needed because the error may occur within a
    /// loop, or a loop nested inside a loop, etc. Each frame records
    /// the instruction offset at that nesting level, the iteration
    /// index within the enclosing loop, and the loop's repetition count.
    #[must_use]
    pub fn stack_frames(&self) -> &[CircuitErrorLocationStackFrame] {
        &self.stack_frames
    }

    /// Returns the noise tag associated with this error location, or `""`
    /// if no tag was set on the noise instruction.
    ///
    /// Tags are the `[...]` annotation that can appear after a noise
    /// instruction's name, e.g. `Y_ERROR[test-tag](0.125) 0`. They are
    /// arbitrary strings that Stim propagates but otherwise ignores,
    /// allowing user code to attach metadata to specific noise channels.
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

    use crate::GateTarget;

    fn sample_instruction_targets() -> CircuitTargetsInsideInstruction {
        CircuitTargetsInsideInstruction::new(
            "X_ERROR",
            "",
            vec![0.25],
            2,
            5,
            vec![
                GateTargetWithCoords::new(GateTarget::from(5u32), vec![1.0, 2.0]),
                GateTargetWithCoords::new(GateTarget::from(6u32), vec![1.0, 3.0]),
                GateTargetWithCoords::new(GateTarget::from(7u32), vec![]),
            ],
        )
    }

    fn sample_flipped_measurement() -> FlippedMeasurement {
        FlippedMeasurement::new(
            Some(5),
            [GateTargetWithCoords::new(
                GateTarget::x(5u32, false).expect("X target should build"),
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
            GateTarget::y(6u32, false).expect("Y target should build"),
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
                GateTarget::y(6u32, false).expect("Y target should build"),
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
                GateTarget::y(6u32, false).expect("Y target should build"),
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
                GateTarget::y(6u32, false).expect("Y target should build"),
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
                GateTarget::y(6u32, false).expect("Y target should build"),
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
            r#"stim::CircuitErrorLocation { tick_offset: 5, flipped_pauli_product: [stim::GateTargetWithCoords { gate_target: stim::GateTarget::y(6, false).unwrap(), coords: [1.0, 2.0, 3.0] }], flipped_measurement: Some(stim::FlippedMeasurement(
    record_index=5,
    observable=(stim::GateTargetWithCoords(stim::GateTarget::x(5, false).unwrap(), [1.0, 2.0, 3.0]),),
)), instruction_targets: stim::CircuitTargetsInsideInstruction(gate="X_ERROR", tag="", args=[0.25], target_range_start=2, target_range_end=5, targets_in_range=(stim::GateTargetWithCoords(stim::GateTarget::qubit(5, false).unwrap(), [1.0, 2.0]), stim::GateTargetWithCoords(stim::GateTarget::qubit(6, false).unwrap(), [1.0, 3.0]), stim::GateTargetWithCoords(stim::GateTarget::qubit(7, false).unwrap(), []))), stack_frames: [stim::CircuitErrorLocationStackFrame { instruction_offset: 1, iteration_index: 0, instruction_repetitions_arg: 3 }, stim::CircuitErrorLocationStackFrame { instruction_offset: 1, iteration_index: 2, instruction_repetitions_arg: 0 }], noise_tag: "test-tag" }"#
        );
    }
}
