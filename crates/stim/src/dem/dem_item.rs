use crate::{DemInstruction, DemRepeatBlock};

/// An item that can appear at the top level of a detector error model.
///
/// A detector error model is a sequence of `DemItem`s, each of which is
/// either a single [`DemInstruction`] or a [`DemRepeatBlock`].
///
/// # Examples
///
/// ```
/// let inst: stim::DemInstruction = "error(0.125) D0 D1".parse().expect("valid");
/// let item = stim::DemItem::instruction(inst.clone());
/// assert_eq!(item, stim::DemItem::Instruction(inst));
/// ```
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DemItem {
    /// A single DEM instruction (e.g. `error`, `detector`, `shift_detectors`).
    Instruction(DemInstruction),
    /// A repeat block containing a sub-model that is repeated a given number
    /// of times.
    RepeatBlock(DemRepeatBlock),
}

impl DemItem {
    /// Creates an [`Instruction`](Self::Instruction) item.
    #[must_use]
    pub fn instruction(instruction: DemInstruction) -> Self {
        Self::Instruction(instruction)
    }

    /// Creates a [`RepeatBlock`](Self::RepeatBlock) item.
    #[must_use]
    pub fn repeat_block(repeat_block: DemRepeatBlock) -> Self {
        Self::RepeatBlock(repeat_block)
    }
}
