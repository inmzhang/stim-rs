use crate::{DemInstruction, DemRepeatBlock};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DemItem {
    Instruction(DemInstruction),
    RepeatBlock(DemRepeatBlock),
}

impl DemItem {
    #[must_use]
    pub fn instruction(instruction: DemInstruction) -> Self {
        Self::Instruction(instruction)
    }

    #[must_use]
    pub fn repeat_block(repeat_block: DemRepeatBlock) -> Self {
        Self::RepeatBlock(repeat_block)
    }
}
