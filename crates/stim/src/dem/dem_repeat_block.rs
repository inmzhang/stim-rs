use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

use crate::{DetectorErrorModel, Result, StimError};

#[derive(Clone, PartialEq, Eq)]
pub struct DemRepeatBlock {
    repeat_count: u64,
    block: DetectorErrorModel,
}

impl DemRepeatBlock {
    pub fn new(repeat_count: u64, block: &DetectorErrorModel) -> Result<Self> {
        if repeat_count == 0 {
            return Err(StimError::new("Can't repeat 0 times."));
        }
        Ok(Self {
            repeat_count,
            block: block.clone(),
        })
    }

    #[must_use]
    pub fn repeat_count(&self) -> u64 {
        self.repeat_count
    }

    #[must_use]
    pub fn body_copy(&self) -> DetectorErrorModel {
        self.block.clone()
    }

    #[must_use]
    pub fn r#type(&self) -> &str {
        "repeat"
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
    use std::collections::{BTreeSet, HashSet};
    use std::str::FromStr;

    #[test]
    fn constructor_and_accessors_preserve_values() {
        let body = DetectorErrorModel::from_str("error(0.125) D0 D1\nshift_detectors 1").unwrap();
        let repeat = DemRepeatBlock::new(100, &body).unwrap();

        assert_eq!(repeat.repeat_count(), 100);
        assert_eq!(repeat.r#type(), "repeat");
        assert_eq!(repeat.body_copy(), body);
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

        let hashed = [first.clone(), same, different.clone()]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 2);
        assert!(hashed.contains(&first));
        assert!(hashed.contains(&different));
    }

    #[test]
    fn body_copy_is_independent_and_zero_repetitions_fail() {
        let body = DetectorErrorModel::from_str("error(0.125) D0 D1").unwrap();
        let repeat = DemRepeatBlock::new(5, &body).unwrap();
        let mut copied = repeat.body_copy();
        copied.clear();

        assert_eq!(repeat.body_copy(), body);
        assert_eq!(
            DemRepeatBlock::new(0, &body)
                .expect_err("zero repetitions should fail")
                .message(),
            "Can't repeat 0 times."
        );
    }
}
