use crate::{DemInstruction, DemRepeatBlock};

/// An item that can appear at the top level of a detector error model.
///
/// A [`DetectorErrorModel`](crate::DetectorErrorModel) is a sequence of
/// `DemItem`s. Each item is one of:
///
/// - [`Instruction`](Self::Instruction) -- a single DEM instruction such
///   as `error(0.125) D0 D1`, `detector(1.5, 2.0) D3`, or
///   `shift_detectors 1`.
/// - [`RepeatBlock`](Self::RepeatBlock) -- a `repeat N { ... }` construct
///   containing a sub-model that is repeated `N` times.
///
/// This enum is the DEM analog of
/// [`CircuitItem`](crate::CircuitItem), which plays the same role for
/// circuit instructions. You can iterate over the items of a
/// `DetectorErrorModel` to inspect or transform its contents.
///
/// # Examples
///
/// ```
/// let inst: stim::DemInstruction = "error(0.125) D0 D1".parse().expect("valid");
/// let item = stim::DemItem::instruction(inst.clone());
/// assert_eq!(item, stim::DemItem::Instruction(inst));
///
/// let body: stim::DetectorErrorModel = "error(0.1) D0".parse().expect("valid");
/// let block = stim::DemRepeatBlock::new(10, &body).expect("valid");
/// let item = stim::DemItem::repeat_block(block.clone());
/// assert_eq!(item, stim::DemItem::RepeatBlock(block));
/// ```
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DemItem {
    /// A single DEM instruction (e.g. `error`, `detector`,
    /// `logical_observable`, `shift_detectors`).
    Instruction(DemInstruction),
    /// A repeat block containing a sub-model that is repeated a given number
    /// of times.
    RepeatBlock(DemRepeatBlock),
}

impl DemItem {
    /// Creates an [`Instruction`](Self::Instruction) item wrapping the
    /// given [`DemInstruction`].
    #[must_use]
    pub fn instruction(instruction: DemInstruction) -> Self {
        Self::Instruction(instruction)
    }

    /// Creates a [`RepeatBlock`](Self::RepeatBlock) item wrapping the
    /// given [`DemRepeatBlock`].
    #[must_use]
    pub fn repeat_block(repeat_block: DemRepeatBlock) -> Self {
        Self::RepeatBlock(repeat_block)
    }
}
