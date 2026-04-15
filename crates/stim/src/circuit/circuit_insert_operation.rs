use crate::{Circuit, CircuitInstruction, CircuitRepeatBlock};

/// An operation that can be inserted into a [`Circuit`] via methods like
/// [`Circuit::insert()`](Circuit::insert).
///
/// When you modify a circuit by inserting content at a given position, the
/// content can take one of three forms:
///
/// | Variant | What it inserts |
/// |---------|-----------------|
/// | [`Instruction`](Self::Instruction) | A single gate instruction (e.g. `H 0`) |
/// | [`Circuit`](Self::Circuit) | An entire sub-circuit whose operations are inlined — they are **not** wrapped in a `REPEAT` block |
/// | [`RepeatBlock`](Self::RepeatBlock) | A `REPEAT` block (sub-circuit + repeat count) |
///
/// In Python's Stim API, the `Circuit.insert()` method accepts either a
/// `stim.CircuitInstruction` or a `stim.Circuit` directly. This enum
/// makes the Rust equivalent type-safe while remaining ergonomic thanks
/// to blanket [`From`] conversions for all three types (both owned and
/// borrowed).
///
/// # `From` conversions
///
/// You rarely need to construct this enum explicitly. All three inner
/// types (and their references) implement `Into<CircuitInsertOperation>`,
/// so you can pass them directly to any function that accepts
/// `impl Into<CircuitInsertOperation>`:
///
/// - `CircuitInstruction` / `&CircuitInstruction` → [`Instruction`](Self::Instruction)
/// - `Circuit` / `&Circuit` → [`Circuit`](Self::Circuit)
/// - `CircuitRepeatBlock` / `&CircuitRepeatBlock` → [`RepeatBlock`](Self::RepeatBlock)
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CircuitInsertOperation {
    /// A single gate instruction to insert.
    ///
    /// When this variant is inserted into a circuit, exactly one
    /// instruction line is added at the target position. For example,
    /// inserting `CircuitInstruction::from_stim_program_text("Y 3 4 5")`
    /// adds a `Y 3 4 5` line.
    Instruction(CircuitInstruction),
    /// A sub-circuit whose operations are inserted inline (not wrapped in a
    /// `REPEAT` block).
    ///
    /// All top-level items from the inner [`Circuit`] are spliced into the
    /// target circuit at the insertion index. This is conceptually similar
    /// to Python's `Circuit.insert(index, another_circuit)` — the
    /// inserted circuit's instructions appear as if they had been written
    /// directly into the host circuit at that position.
    Circuit(Circuit),
    /// A `REPEAT` block to insert.
    ///
    /// When this variant is inserted, a single `REPEAT` block (with its
    /// body sub-circuit and repetition count) is added at the target
    /// position. This is how you insert looping constructs
    /// programmatically.
    RepeatBlock(CircuitRepeatBlock),
}

impl From<CircuitInstruction> for CircuitInsertOperation {
    fn from(value: CircuitInstruction) -> Self {
        Self::Instruction(value)
    }
}

impl From<&CircuitInstruction> for CircuitInsertOperation {
    fn from(value: &CircuitInstruction) -> Self {
        Self::Instruction(value.clone())
    }
}

impl From<Circuit> for CircuitInsertOperation {
    fn from(value: Circuit) -> Self {
        Self::Circuit(value)
    }
}

impl From<&Circuit> for CircuitInsertOperation {
    fn from(value: &Circuit) -> Self {
        Self::Circuit(value.clone())
    }
}

impl From<CircuitRepeatBlock> for CircuitInsertOperation {
    fn from(value: CircuitRepeatBlock) -> Self {
        Self::RepeatBlock(value)
    }
}

impl From<&CircuitRepeatBlock> for CircuitInsertOperation {
    fn from(value: &CircuitRepeatBlock) -> Self {
        Self::RepeatBlock(value.clone())
    }
}
