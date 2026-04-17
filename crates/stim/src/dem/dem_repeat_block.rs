use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

use crate::{DemInstructionType, DetectorErrorModel, Result, StimError};

/// A repeat block from a detector error model.
///
/// Represents a `repeat N { ... }` construct in a
/// [`DetectorErrorModel`], where a sub-model (the body) is repeated
/// `N` times. Repeat blocks are commonly used to express the periodic
/// structure of error-correction rounds: the body typically contains
/// `error` instructions and a `shift_detectors` instruction that
/// advances detector indices by a fixed stride each iteration.
///
/// The repetition count must be at least 1; attempting to create a
/// block with `repeat_count == 0` returns an error.
///
/// # Examples
///
/// ```
/// let body: stim::DetectorErrorModel = "error(0.125) D0 D1\nshift_detectors 1"
///     .parse()
///     .expect("valid body");
/// let block = stim::DemRepeatBlock::new(100, &body).expect("valid repeat block");
/// assert_eq!(block.repeat_count(), 100);
/// assert_eq!(block.r#type(), stim::DemInstructionType::Repeat);
/// assert_eq!(block.body(), &body);
///
/// // Display renders the block in DEM text format.
/// assert!(block.to_string().starts_with("repeat 100 {"));
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct DemRepeatBlock {
    repeat_count: u64,
    block: DetectorErrorModel,
}

impl DemRepeatBlock {
    /// Creates a new repeat block with the given count and body.
    ///
    /// The body is cloned into the new block. Subsequent mutations to
    /// the original `DetectorErrorModel` will not affect the block.
    ///
    /// # Errors
    ///
    /// Returns an error if `repeat_count` is zero, because repeating a
    /// block zero times has no meaning in a detector error model.
    ///
    /// # Examples
    ///
    /// ```
    /// let body: stim::DetectorErrorModel = "error(0.125) D0".parse().expect("valid");
    /// let block = stim::DemRepeatBlock::new(5, &body).expect("valid repeat");
    /// assert_eq!(block.repeat_count(), 5);
    ///
    /// // Zero repetitions are rejected.
    /// assert!(stim::DemRepeatBlock::new(0, &body).is_err());
    /// ```
    pub fn new(repeat_count: u64, block: &DetectorErrorModel) -> Result<Self> {
        if repeat_count == 0 {
            return Err(StimError::new("Can't repeat 0 times."));
        }
        Ok(Self {
            repeat_count,
            block: block.clone(),
        })
    }

    /// Returns the number of times the body is supposed to execute.
    #[must_use]
    pub fn repeat_count(&self) -> u64 {
        self.repeat_count
    }

    /// Returns the repeated body model.
    #[must_use]
    pub fn body(&self) -> &DetectorErrorModel {
        &self.block
    }

    /// Returns the type of this block, always [`DemInstructionType::Repeat`].
    ///
    /// This is a duck-typing convenience method. It exists so that code
    /// that doesn't know whether it has a
    /// [`DemInstruction`](crate::DemInstruction) or a `DemRepeatBlock`
    /// can check the type field without doing a pattern match first.
    #[must_use]
    pub fn r#type(&self) -> DemInstructionType {
        DemInstructionType::Repeat
    }
}

impl Display for DemRepeatBlock {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "repeat {} {{", self.repeat_count)?;
        let body = self.block.to_string();
        if !body.is_empty() {
            f.write_str("\n")?;
            for line in body.lines() {
                writeln!(f, "    {line}")?;
            }
        } else {
            f.write_str("\n")?;
        }
        f.write_str("}")
    }
}

impl fmt::Debug for DemRepeatBlock {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stim::DemRepeatBlock({}, {:?})",
            self.repeat_count, self.block
        )
    }
}

impl PartialOrd for DemRepeatBlock {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DemRepeatBlock {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.repeat_count
            .cmp(&other.repeat_count)
            .then_with(|| self.block.to_string().cmp(&other.block.to_string()))
    }
}

impl Hash for DemRepeatBlock {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.repeat_count.hash(state);
        self.block.to_string().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::str::FromStr;

    #[test]
    fn constructor_and_accessors_preserve_values() {
        let body = DetectorErrorModel::from_str("error(0.125) D0 D1\nshift_detectors 1").unwrap();
        let repeat = DemRepeatBlock::new(100, &body).unwrap();

        assert_eq!(repeat.repeat_count(), 100);
        assert_eq!(repeat.r#type(), DemInstructionType::Repeat);
        assert_eq!(repeat.body(), &body);
    }

    #[test]
    fn equality_and_debug_display_match_binding_conventions() {
        let body = DetectorErrorModel::from_str("error(0.125) D0 D1\nshift_detectors 1").unwrap();
        let first = DemRepeatBlock::new(100, &body).unwrap();
        let same = DemRepeatBlock::new(100, &body).unwrap();
        let different = DemRepeatBlock::new(
            50,
            &DetectorErrorModel::from_str("error(0.125) D0 D1").unwrap(),
        )
        .unwrap();

        assert_eq!(first, same);
        assert_ne!(first, different);
        assert_eq!(
            first.to_string(),
            "repeat 100 {\n    error(0.125) D0 D1\n    shift_detectors 1\n}"
        );
        assert_eq!(
            format!("{first:?}"),
            "stim::DemRepeatBlock(100, stim::DetectorErrorModel(\"\"\"\nerror(0.125) D0 D1\nshift_detectors 1\n\"\"\"))"
        );

        let ordered = [different.clone(), same.clone(), first.clone()]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(ordered, vec![different.clone(), first.clone()]);

        let hash_of = |value: &DemRepeatBlock| {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash_of(&first), hash_of(&same));
        assert_ne!(hash_of(&first), hash_of(&different));
    }

    #[test]
    fn body_is_borrowed_and_zero_repetitions_fail() {
        let body = DetectorErrorModel::from_str("error(0.125) D0 D1").unwrap();
        let repeat = DemRepeatBlock::new(5, &body).unwrap();
        let mut copied = repeat.body().clone();
        copied.clear();

        assert_eq!(repeat.body(), &body);
        assert_eq!(
            DemRepeatBlock::new(0, &body)
                .expect_err("zero repetitions should fail")
                .message(),
            "Can't repeat 0 times."
        );
    }
}
