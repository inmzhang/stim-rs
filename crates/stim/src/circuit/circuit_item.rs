use crate::{CircuitInstruction, CircuitRepeatBlock};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum CircuitItem {
    Instruction(CircuitInstruction),
    RepeatBlock(CircuitRepeatBlock),
}

impl CircuitItem {
    #[must_use]
    pub fn instruction(instruction: CircuitInstruction) -> Self {
        Self::Instruction(instruction)
    }

    #[must_use]
    pub fn repeat_block(repeat_block: CircuitRepeatBlock) -> Self {
        Self::RepeatBlock(repeat_block)
    }
}
