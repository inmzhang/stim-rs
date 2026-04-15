use crate::{Circuit, CircuitInstruction, CircuitRepeatBlock};

/// An operation that can be inserted into a [`Circuit`].
///
/// This enum unifies the three kinds of content that can be spliced into a
/// circuit at a given position: a single instruction, an entire sub-circuit
/// (whose operations are inlined), or a repeat block. `From` conversions are
/// provided for each variant so callers can pass any of the three types
/// directly.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CircuitInsertOperation {
    /// A single gate instruction to insert.
    Instruction(CircuitInstruction),
    /// A sub-circuit whose operations are inserted inline (not wrapped in a
    /// `REPEAT` block).
    Circuit(Circuit),
    /// A `REPEAT` block to insert.
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
