use crate::{DemInstruction, DemRepeatBlock, DetectorErrorModel};

/// An operation that can be appended to a detector error model.
///
/// This enum unifies the three types that [`DetectorErrorModel::append_operation`]
/// accepts:
/// - A single [`DemInstruction`].
/// - A [`DemRepeatBlock`].
/// - An entire [`DetectorErrorModel`] (whose items are concatenated).
///
/// Values of each inner type convert into `DemAppendOperation` via `From`.
///
/// # Examples
///
/// ```
/// let inst: stim::DemInstruction = "error(0.125) D1".parse().expect("valid");
/// let op = stim::DemAppendOperation::from(inst);
/// assert!(matches!(op, stim::DemAppendOperation::Instruction(_)));
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DemAppendOperation {
    /// Append a single instruction.
    Instruction(DemInstruction),
    /// Append a repeat block.
    RepeatBlock(DemRepeatBlock),
    /// Append all items from another detector error model.
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
