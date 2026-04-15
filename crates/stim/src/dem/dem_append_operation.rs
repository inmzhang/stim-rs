use crate::{DemInstruction, DemRepeatBlock, DetectorErrorModel};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DemAppendOperation {
    Instruction(DemInstruction),
    RepeatBlock(DemRepeatBlock),
    DetectorErrorModel(DetectorErrorModel),
}

impl From<DemInstruction> for DemAppendOperation {
    fn from(value: DemInstruction) -> Self {
        Self::Instruction(value)
    }
}

impl From<&DemInstruction> for DemAppendOperation {
    fn from(value: &DemInstruction) -> Self {
        Self::Instruction(value.clone())
    }
}

impl From<DemRepeatBlock> for DemAppendOperation {
    fn from(value: DemRepeatBlock) -> Self {
        Self::RepeatBlock(value)
    }
}

impl From<&DemRepeatBlock> for DemAppendOperation {
    fn from(value: &DemRepeatBlock) -> Self {
        Self::RepeatBlock(value.clone())
    }
}

impl From<DetectorErrorModel> for DemAppendOperation {
    fn from(value: DetectorErrorModel) -> Self {
        Self::DetectorErrorModel(value)
    }
}

impl From<&DetectorErrorModel> for DemAppendOperation {
    fn from(value: &DetectorErrorModel) -> Self {
        Self::DetectorErrorModel(value.clone())
    }
}
