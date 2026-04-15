use std::fmt::{self, Display, Formatter};

/// A single frame in a [`CircuitErrorLocation`]'s stack trace.
///
/// Each frame identifies a position within a level of the circuit's nesting
/// structure. The outermost frame refers to the top-level circuit, and each
/// inner frame descends into a `REPEAT` block.
///
/// [`CircuitErrorLocation`]: crate::CircuitErrorLocation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CircuitErrorLocationStackFrame {
    instruction_offset: u64,
    iteration_index: u64,
    instruction_repetitions_arg: u64,
}

impl CircuitErrorLocationStackFrame {
    /// Creates a new stack frame.
    ///
    /// - `instruction_offset`: the 0-based index of the instruction within
    ///   the circuit (or `REPEAT` body) at this nesting level.
    /// - `iteration_index`: which iteration of the enclosing `REPEAT` block
    ///   had completed before the error occurred (0 for the top-level
    ///   circuit).
    /// - `instruction_repetitions_arg`: the repeat count of the `REPEAT`
    ///   block at this level, or 0 if this frame is the innermost
    ///   (instruction-level) frame.
    #[must_use]
    pub const fn new(
        instruction_offset: u64,
        iteration_index: u64,
        instruction_repetitions_arg: u64,
    ) -> Self {
        Self {
            instruction_offset,
            iteration_index,
            instruction_repetitions_arg,
        }
    }

    /// Returns the 0-based index of the instruction at this nesting level.
    #[must_use]
    pub const fn instruction_offset(&self) -> u64 {
        self.instruction_offset
    }

    /// Returns the number of completed iterations of the enclosing `REPEAT`
    /// block before the error occurred.
    ///
    /// For the top-level circuit frame this is always 0.
    #[must_use]
    pub const fn iteration_index(&self) -> u64 {
        self.iteration_index
    }

    /// Returns the repeat count of the `REPEAT` block at this nesting level.
    ///
    /// For the innermost (instruction-level) frame, this is 0.
    #[must_use]
    pub const fn instruction_repetitions_arg(&self) -> u64 {
        self.instruction_repetitions_arg
    }
}

impl Display for CircuitErrorLocationStackFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stim::CircuitErrorLocationStackFrame(
    instruction_offset={},
    iteration_index={},
    instruction_repetitions_arg={},
)",
            self.instruction_offset, self.iteration_index, self.instruction_repetitions_arg
        )
    }
}

impl fmt::Debug for CircuitErrorLocationStackFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("stim::CircuitErrorLocationStackFrame")
            .field("instruction_offset", &self.instruction_offset)
            .field("iteration_index", &self.iteration_index)
            .field(
                "instruction_repetitions_arg",
                &self.instruction_repetitions_arg,
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn constructor_and_accessors_preserve_values() {
        let frame = CircuitErrorLocationStackFrame::new(1, 2, 3);

        assert_eq!(frame.instruction_offset(), 1);
        assert_eq!(frame.iteration_index(), 2);
        assert_eq!(frame.instruction_repetitions_arg(), 3);
    }

    #[test]
    fn equality_hash_and_order_follow_field_order() {
        let first = CircuitErrorLocationStackFrame::new(1, 2, 3);
        let same = CircuitErrorLocationStackFrame::new(1, 2, 3);
        let different_offset = CircuitErrorLocationStackFrame::new(2, 2, 3);
        let different_iteration = CircuitErrorLocationStackFrame::new(1, 4, 3);
        let different_repetitions = CircuitErrorLocationStackFrame::new(1, 2, 9);

        assert_eq!(first, same);
        assert_ne!(first, different_offset);
        assert_ne!(first, different_iteration);
        assert_ne!(first, different_repetitions);

        let ordered = [
            different_offset,
            different_repetitions,
            first,
            different_iteration,
            same,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
        assert_eq!(
            ordered,
            vec![
                CircuitErrorLocationStackFrame::new(1, 2, 3),
                CircuitErrorLocationStackFrame::new(1, 2, 9),
                CircuitErrorLocationStackFrame::new(1, 4, 3),
                CircuitErrorLocationStackFrame::new(2, 2, 3),
            ]
        );

        let hashed = [
            CircuitErrorLocationStackFrame::new(1, 2, 3),
            CircuitErrorLocationStackFrame::new(1, 2, 3),
            CircuitErrorLocationStackFrame::new(2, 2, 3),
        ]
        .into_iter()
        .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 2);
    }

    #[test]
    fn display_and_debug_match_binding_conventions() {
        let frame = CircuitErrorLocationStackFrame::new(1, 2, 3);

        assert_eq!(
            frame.to_string(),
            "stim::CircuitErrorLocationStackFrame(
    instruction_offset=1,
    iteration_index=2,
    instruction_repetitions_arg=3,
)"
        );
        assert_eq!(
            format!("{frame:?}"),
            "stim::CircuitErrorLocationStackFrame { instruction_offset: 1, iteration_index: 2, instruction_repetitions_arg: 3 }"
        );
    }
}
