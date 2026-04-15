use std::fmt::{self, Display, Formatter};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CircuitErrorLocationStackFrame {
    instruction_offset: u64,
    iteration_index: u64,
    instruction_repetitions_arg: u64,
}

impl CircuitErrorLocationStackFrame {
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

    #[must_use]
    pub const fn instruction_offset(&self) -> u64 {
        self.instruction_offset
    }

    #[must_use]
    pub const fn iteration_index(&self) -> u64 {
        self.iteration_index
    }

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
