use crate::{CircuitInstruction, CircuitRepeatBlock};

/// An item that can appear at the top level of a [`crate::Circuit`].
///
/// A circuit is a sequence of items, where each item is either a single gate
/// instruction (like `H 0 1`) or a `REPEAT` block that loops a sub-circuit a
/// fixed number of times.
///
/// # Examples
///
/// ```
/// let instruction = stim::CircuitInstruction::new(
///     "H", [0u32, 1u32], std::iter::empty::<f64>(), "",
/// ).expect("H is a valid gate");
///
/// let item = stim::CircuitItem::instruction(instruction.clone());
/// assert_eq!(item, stim::CircuitItem::Instruction(instruction));
/// ```
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum CircuitItem {
    /// A single gate instruction, such as `H 0 1` or `CNOT rec[-1] 5`.
    Instruction(CircuitInstruction),
    /// A `REPEAT` block that loops a sub-circuit a fixed number of times.
    RepeatBlock(CircuitRepeatBlock),
}

impl CircuitItem {
    /// Wraps a [`CircuitInstruction`] into a [`CircuitItem::Instruction`].
    ///
    /// This is a convenience constructor equivalent to
    /// `CircuitItem::Instruction(instruction)`.
    #[must_use]
    pub fn instruction(instruction: CircuitInstruction) -> Self {
        Self::Instruction(instruction)
    }

    /// Wraps a [`CircuitRepeatBlock`] into a [`CircuitItem::RepeatBlock`].
    ///
    /// This is a convenience constructor equivalent to
    /// `CircuitItem::RepeatBlock(repeat_block)`.
    #[must_use]
    pub fn repeat_block(repeat_block: CircuitRepeatBlock) -> Self {
        Self::RepeatBlock(repeat_block)
    }
}
