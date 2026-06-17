use std::fmt::{self, Display, Formatter};

/// A single frame in a [`CircuitErrorLocation`]'s stack trace, describing
/// the position of an instruction within one level of a circuit's nesting
/// structure.
///
/// When Stim locates an error inside a circuit, the full location is a
/// list of these frames that drill down from the top-level circuit to the
/// innermost loop containing the instruction. Each frame records:
///
/// - **`instruction_offset`** -- the 0-based index of the instruction
///   within the circuit or `REPEAT` body at this nesting level. This is
///   slightly different from a line number because blank lines and
///   comments are not counted.
/// - **`iteration_index`** -- which iteration of the enclosing `REPEAT`
///   block had completed before the error occurred. For the outermost
///   (top-level) frame this is always 0.
/// - **`instruction_repetitions_arg`** -- the repeat count (`N`) of the
///   `REPEAT N { ... }` block at this level. For the innermost
///   (instruction-level) frame that does not itself refer to a `REPEAT`,
///   this is 0.
///
/// # Examples
///
/// A circuit with a `REPEAT 5 { ... }` block produces two stack frames
/// when an error occurs inside the loop:
///
/// ```
/// use stim::CircuitErrorLocationStackFrame;
///
/// // Outer frame: instruction #0 in the top-level circuit is the
/// // REPEAT 5 block. iteration_index is 0 for the top-level frame.
/// let outer = CircuitErrorLocationStackFrame::new(0, 0, 5);
/// assert_eq!(outer.instruction_offset(), 0);
/// assert_eq!(outer.instruction_repetitions_arg(), 5);
///
/// // Inner frame: instruction #1 inside the REPEAT body.
/// // iteration_index=4 means the error is in the 5th (last) iteration.
/// let inner = CircuitErrorLocationStackFrame::new(1, 4, 0);
/// assert_eq!(inner.iteration_index(), 4);
/// assert_eq!(inner.instruction_repetitions_arg(), 0);
/// ```
///
/// [`CircuitErrorLocation`]: crate::CircuitErrorLocation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CircuitErrorLocationStackFrame {
    instruction_offset: u64,
    iteration_index: u64,
    instruction_repetitions_arg: u64,
}

impl CircuitErrorLocationStackFrame {
    /// Creates a new stack frame from its three components.
    ///
    /// # Arguments
    ///
    /// - `instruction_offset` -- the 0-based index of the instruction
    ///   within the circuit (or `REPEAT` body) at this nesting level.
    /// - `iteration_index` -- which iteration of the enclosing `REPEAT`
    ///   block had completed before the error occurred. Pass 0 for the
    ///   top-level circuit frame (which has no enclosing loop).
    /// - `instruction_repetitions_arg` -- the repeat count of the
    ///   `REPEAT` block at this level. Pass 0 if this frame is the
    ///   innermost (instruction-level) frame that does not itself refer
    ///   to a `REPEAT` block.
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
    ///
    /// This is the position within the parent circuit or `REPEAT` body,
    /// starting from 0. Blank lines and comments in the circuit text are
    /// not counted, so this may differ from the line number in the
    /// source text.
    ///
    /// # Examples
    ///
    /// ```
    /// use stim::CircuitErrorLocationStackFrame;
    ///
    /// let frame = CircuitErrorLocationStackFrame::new(2, 0, 0);
    /// assert_eq!(frame.instruction_offset(), 2);
    /// ```
    #[must_use]
    pub const fn instruction_offset(&self) -> u64 {
        self.instruction_offset
    }

    /// Returns the number of completed iterations of the enclosing `REPEAT`
    /// block before the error occurred.
    ///
    /// Disambiguates which iteration of the loop containing this
    /// instruction is being referred to. For the top-level circuit frame
    /// (which has no enclosing loop) this is always 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use stim::CircuitErrorLocationStackFrame;
    ///
    /// // Error occurs in the 5th iteration (index 4) of a REPEAT block.
    /// let frame = CircuitErrorLocationStackFrame::new(1, 4, 0);
    /// assert_eq!(frame.iteration_index(), 4);
    /// ```
    #[must_use]
    pub const fn iteration_index(&self) -> u64 {
        self.iteration_index
    }

    /// Returns the repeat count of the `REPEAT` block at this nesting level.
    ///
    /// If the instruction being referred to is a `REPEAT` block, this is
    /// the repetition count argument (`N` in `REPEAT N { ... }`).
    /// Otherwise, for the innermost (instruction-level) frame, this is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use stim::CircuitErrorLocationStackFrame;
    ///
    /// // This frame refers to a REPEAT 5 block.
    /// let repeat_frame = CircuitErrorLocationStackFrame::new(0, 0, 5);
    /// assert_eq!(repeat_frame.instruction_repetitions_arg(), 5);
    ///
    /// // Innermost frame (not a REPEAT block itself).
    /// let inner_frame = CircuitErrorLocationStackFrame::new(1, 4, 0);
    /// assert_eq!(inner_frame.instruction_repetitions_arg(), 0);
    /// ```
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
