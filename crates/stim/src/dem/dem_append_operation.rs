use crate::{DemInstruction, DemRepeatBlock, DetectorErrorModel};

/// An operation that can be appended to a
/// [`DetectorErrorModel`](crate::DetectorErrorModel).
///
/// When building a detector error model programmatically, you append
/// items one at a time via
/// [`DetectorErrorModel::append_operation`](crate::DetectorErrorModel::append_operation).
/// This enum unifies the three types that method accepts:
///
/// - [`Instruction`](Self::Instruction) -- a single
///   [`DemInstruction`] (e.g. `error(0.125) D0 D1`).
/// - [`RepeatBlock`](Self::RepeatBlock) -- a [`DemRepeatBlock`]
///   (`repeat N { ... }`).
/// - [`DetectorErrorModel`](Self::DetectorErrorModel) -- an entire
///   [`DetectorErrorModel`](crate::DetectorErrorModel) whose items are
///   concatenated into the target model.
///
/// Each inner type converts into `DemAppendOperation` via `From`, for
/// both owned and borrowed values.
///
/// # Examples
///
/// ```
/// let inst: stim::DemInstruction = "error(0.125) D1".parse().expect("valid");
/// let op = stim::DemAppendOperation::from(inst);
/// assert!(matches!(op, stim::DemAppendOperation::Instruction(_)));
///
/// let body: stim::DetectorErrorModel = "error(0.1) D0".parse().expect("valid");
/// let block = stim::DemRepeatBlock::new(5, &body).expect("valid");
/// let op = stim::DemAppendOperation::from(block);
/// assert!(matches!(op, stim::DemAppendOperation::RepeatBlock(_)));
///
/// let model: stim::DetectorErrorModel = "error(0.1) D0\nerror(0.2) D1"
///     .parse().expect("valid");
/// let op = stim::DemAppendOperation::from(model);
/// assert!(matches!(op, stim::DemAppendOperation::DetectorErrorModel(_)));
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DemAppendOperation {
    /// Append a single DEM instruction (e.g. `error`, `detector`,
    /// `shift_detectors`).
    Instruction(DemInstruction),
    /// Append a `repeat N { ... }` block.
    RepeatBlock(DemRepeatBlock),
    /// Append all items from another detector error model, concatenating
    /// them into the target model in order.
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
