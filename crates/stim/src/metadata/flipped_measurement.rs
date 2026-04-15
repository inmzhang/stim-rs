use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

use crate::GateTargetWithCoords;

#[derive(Clone, PartialEq)]
pub struct FlippedMeasurement {
    record_index: Option<u64>,
    observable: Vec<GateTargetWithCoords>,
}

impl FlippedMeasurement {
    #[must_use]
    pub fn new(
        record_index: Option<u64>,
        observable: impl IntoIterator<Item = GateTargetWithCoords>,
    ) -> Self {
        Self {
            record_index,
            observable: observable.into_iter().collect(),
        }
    }

    #[must_use]
    pub fn record_index(&self) -> Option<u64> {
        self.record_index
    }

    #[must_use]
    pub fn observable(&self) -> &[GateTargetWithCoords] {
        &self.observable
    }

    fn fmt_repr(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(
            "stim::FlippedMeasurement(
    record_index=",
        )?;
        match self.record_index {
            Some(record_index) => write!(f, "{record_index}")?,
            None => f.write_str("None")?,
        }
        f.write_str(
            ",
    observable=(",
        )?;
        for target in &self.observable {
            write!(
                f,
                "stim::GateTargetWithCoords({:?}, {:?}),",
                target.gate_target(),
                target.coords()
            )?;
        }
        f.write_str(
            "),
)",
        )
    }
}

impl Eq for FlippedMeasurement {}

impl PartialOrd for FlippedMeasurement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FlippedMeasurement {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.record_index, other.record_index) {
            (Some(left), Some(right)) => left.cmp(&right),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
        .then_with(|| self.observable.cmp(&other.observable))
    }
}

impl Hash for FlippedMeasurement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.record_index.hash(state);
        self.observable.hash(state);
    }
}

impl Display for FlippedMeasurement {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_repr(f)
    }
}

impl fmt::Debug for FlippedMeasurement {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_repr(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    use crate::target_x;

    #[test]
    fn constructor_and_accessors_preserve_record_index_and_observable() {
        let first = GateTargetWithCoords::new(
            target_x(5, false).expect("X target should build"),
            vec![1.0],
        );
        let second =
            GateTargetWithCoords::new(target_x(7, false).expect("X target should build"), vec![]);

        let flipped = FlippedMeasurement::new(Some(5), [first.clone(), second.clone()]);

        assert_eq!(flipped.record_index(), Some(5));
        assert_eq!(flipped.observable(), &[first, second]);
    }

    #[test]
    fn equality_hash_and_order_follow_record_index_then_observable() {
        let first = FlippedMeasurement::new(
            Some(2),
            [GateTargetWithCoords::new(
                target_x(5, false).expect("X target should build"),
                vec![1.0],
            )],
        );
        let same = FlippedMeasurement::new(
            Some(2),
            [GateTargetWithCoords::new(
                target_x(5, false).expect("X target should build"),
                vec![1.0],
            )],
        );
        let different_observable = FlippedMeasurement::new(
            Some(2),
            [GateTargetWithCoords::new(
                target_x(5, false).expect("X target should build"),
                vec![2.0],
            )],
        );
        let later_record = FlippedMeasurement::new(Some(4), Vec::new());
        let no_record_index = FlippedMeasurement::new(None, Vec::new());

        assert_eq!(first, same);
        assert_ne!(first, different_observable);
        assert_ne!(later_record, no_record_index);

        let ordered = [
            no_record_index.clone(),
            later_record.clone(),
            different_observable.clone(),
            same.clone(),
            first.clone(),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
        assert_eq!(
            ordered,
            vec![
                first.clone(),
                different_observable,
                later_record,
                no_record_index.clone()
            ]
        );

        let hashed = [first.clone(), same, no_record_index.clone()]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 2);
        assert!(hashed.contains(&first));
        assert!(hashed.contains(&no_record_index));
    }

    #[test]
    fn display_and_debug_match_binding_conventions() {
        let flipped = FlippedMeasurement::new(
            Some(5),
            [GateTargetWithCoords::new(
                target_x(5, false).expect("X target should build"),
                vec![1.0, -2.5],
            )],
        );

        let expected = "stim::FlippedMeasurement(
    record_index=5,
    observable=(stim::GateTargetWithCoords(stim::target_x(5), [1.0, -2.5]),),
)";
        assert_eq!(flipped.to_string(), expected);
        assert_eq!(format!("{flipped:?}"), expected);
    }

    #[test]
    fn none_record_index_and_empty_observable_are_supported() {
        let flipped = FlippedMeasurement::new(None, Vec::new());

        assert_eq!(flipped.record_index(), None);
        assert!(flipped.observable().is_empty());
        assert_eq!(
            flipped.to_string(),
            "stim::FlippedMeasurement(
    record_index=None,
    observable=(),
)"
        );
    }
}
