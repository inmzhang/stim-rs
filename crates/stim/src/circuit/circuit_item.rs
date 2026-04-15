use crate::{CircuitInstruction, CircuitRepeatBlock};

/// An item that can appear at the top level of a [`crate::Circuit`].
///
/// A Stim circuit is a flat sequence of items, where each item is either a
/// single gate instruction (like `H 0 1` or `X_ERROR(0.01) 5`) or a
/// `REPEAT` block that loops a sub-circuit a fixed number of times. This
/// enum captures that two-variant structure.
///
/// `CircuitItem` is the Rust equivalent of the Python pattern where
/// iterating a `stim.Circuit` yields either `stim.CircuitInstruction` or
/// `stim.CircuitRepeatBlock` objects. In Rust the distinction is made
/// explicit through this enum rather than dynamic dispatch.
///
/// # When you encounter a `CircuitItem`
///
/// You will typically receive `CircuitItem` values when iterating over a
/// circuit. Match on the two variants to handle instructions and repeat
/// blocks differently, or use the shared `name()` / `tag()` methods on
/// the inner types for duck-typed access.
///
/// # Ordering and hashing
///
/// `CircuitItem` derives [`Eq`], [`Ord`], and [`Hash`] by delegating to
/// the inner types. The `Instruction` variant sorts before `RepeatBlock`
/// (enum variant order).
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
///
/// Matching on variants:
///
/// ```
/// let circuit: stim::Circuit = "H 0\nREPEAT 3 {\n    M 0\n}".parse().unwrap();
/// match circuit.get(0).unwrap() {
///     stim::CircuitItem::Instruction(inst) => assert_eq!(inst.name(), "H"),
///     stim::CircuitItem::RepeatBlock(_) => panic!("expected instruction"),
/// }
/// match circuit.get(1).unwrap() {
///     stim::CircuitItem::Instruction(_) => panic!("expected repeat block"),
///     stim::CircuitItem::RepeatBlock(block) => assert_eq!(block.repeat_count(), 3),
/// }
/// ```
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum CircuitItem {
    /// A single gate instruction, such as `H 0 1`, `CNOT rec[-1] 5`, or
    /// `X_ERROR(0.01) 5 7`.
    ///
    /// This variant wraps a [`CircuitInstruction`], which carries the gate
    /// name, gate arguments, targets, and optional tag. It represents any
    /// line in a Stim circuit file that is *not* a `REPEAT` block.
    Instruction(CircuitInstruction),
    /// A `REPEAT` block that loops a sub-circuit a fixed number of times.
    ///
    /// This variant wraps a [`CircuitRepeatBlock`], which carries the
    /// repeat count, the body sub-circuit, and an optional tag. It
    /// represents Stim's only looping construct:
    ///
    /// ```text
    /// REPEAT 100 {
    ///     CX 0 1
    ///     M 0
    /// }
    /// ```
    RepeatBlock(CircuitRepeatBlock),
}

impl CircuitItem {
    /// Wraps a [`CircuitInstruction`] into a [`CircuitItem::Instruction`].
    ///
    /// This is a convenience constructor equivalent to
    /// `CircuitItem::Instruction(instruction)`. It is useful when building
    /// a `CircuitItem` from a variable without needing to name the variant
    /// explicitly.
    #[must_use]
    pub fn instruction(instruction: CircuitInstruction) -> Self {
        Self::Instruction(instruction)
    }

    /// Wraps a [`CircuitRepeatBlock`] into a [`CircuitItem::RepeatBlock`].
    ///
    /// This is a convenience constructor equivalent to
    /// `CircuitItem::RepeatBlock(repeat_block)`. It is useful when building
    /// a `CircuitItem` from a variable without needing to name the variant
    /// explicitly.
    #[must_use]
    pub fn repeat_block(repeat_block: CircuitRepeatBlock) -> Self {
        Self::RepeatBlock(repeat_block)
    }
}
