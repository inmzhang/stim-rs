use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

pub use crate::GateTargetWithCoords;

/// The resolved targets from a specific instruction at a specific location
/// inside a circuit.
///
/// When Stim explains a circuit error, it needs to identify not just which
/// instruction caused the error, but which subset of that instruction's
/// targets were involved. This struct captures the gate name, its arguments,
/// the relevant target range, and the fully resolved targets (with
/// coordinates) within that range.
///
/// Instances are typically found inside [`CircuitErrorLocation`] rather than
/// constructed directly.
///
/// [`CircuitErrorLocation`]: crate::CircuitErrorLocation
#[derive(Clone, PartialEq)]
pub struct CircuitTargetsInsideInstruction {
    gate: String,
    tag: String,
    args: Vec<f64>,
    target_range_start: usize,
    target_range_end: usize,
    targets_in_range: Vec<GateTargetWithCoords>,
}

impl CircuitTargetsInsideInstruction {
    /// Creates a new resolved-targets descriptor.
    ///
    /// - `gate` / `tag` / `args`: identify the instruction.
    /// - `target_range_start` / `target_range_end`: the half-open range
    ///   `[start, end)` of target indices within the instruction that
    ///   participated in the error.
    /// - `targets_in_range`: the resolved targets (with coordinates) in the
    ///   selected range.
    #[must_use]
    pub fn new(
        gate: impl Into<String>,
        tag: impl Into<String>,
        args: Vec<f64>,
        target_range_start: usize,
        target_range_end: usize,
        targets_in_range: Vec<GateTargetWithCoords>,
    ) -> Self {
        Self {
            gate: gate.into(),
            tag: tag.into(),
            args,
            target_range_start,
            target_range_end,
            targets_in_range,
        }
    }

    /// Returns the canonical gate name (e.g. `"X_ERROR"`, `"MPP"`).
    #[must_use]
    pub fn gate(&self) -> &str {
        &self.gate
    }

    /// Returns the instruction's tag string, or `""` if untagged.
    #[must_use]
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Returns the gate's numeric arguments (e.g. error probability).
    #[must_use]
    pub fn args(&self) -> &[f64] {
        &self.args
    }

    /// Returns the start of the half-open target index range `[start, end)`
    /// within the instruction.
    #[must_use]
    pub fn target_range_start(&self) -> usize {
        self.target_range_start
    }

    /// Returns the end of the half-open target index range `[start, end)`
    /// within the instruction.
    #[must_use]
    pub fn target_range_end(&self) -> usize {
        self.target_range_end
    }

    /// Returns the resolved targets (with coordinates) that fall within
    /// the selected range.
    #[must_use]
    pub fn targets_in_range(&self) -> &[GateTargetWithCoords] {
        &self.targets_in_range
    }
}

impl Eq for CircuitTargetsInsideInstruction {}

impl PartialOrd for CircuitTargetsInsideInstruction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CircuitTargetsInsideInstruction {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gate
            .cmp(&other.gate)
            .then_with(|| self.tag.cmp(&other.tag))
            .then_with(|| compare_f64_slices(&self.args, &other.args))
            .then_with(|| self.target_range_start.cmp(&other.target_range_start))
            .then_with(|| self.target_range_end.cmp(&other.target_range_end))
            .then_with(|| self.targets_in_range.cmp(&other.targets_in_range))
    }
}

impl Hash for CircuitTargetsInsideInstruction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.gate.hash(state);
        self.tag.hash(state);
        for arg in &self.args {
            arg.to_bits().hash(state);
        }
        self.target_range_start.hash(state);
        self.target_range_end.hash(state);
        self.targets_in_range.hash(state);
    }
}

impl Display for CircuitTargetsInsideInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.gate)?;
        if !self.tag.is_empty() {
            write!(f, "[{}]", self.tag)?;
        }
        if !self.args.is_empty() {
            f.write_str("(")?;
            for (index, arg) in self.args.iter().enumerate() {
                if index > 0 {
                    f.write_str(",")?;
                }
                write!(f, "{arg}")?;
            }
            f.write_str(")")?;
        }
        for target in &self.targets_in_range {
            if target.gate_target().is_combiner() {
                write!(f, "{target}")?;
            } else {
                write!(f, " {target}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for CircuitTargetsInsideInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stim::CircuitTargetsInsideInstruction(gate={:?}, tag={:?}, args={:?}, target_range_start={}, target_range_end={}, targets_in_range=(",
            self.gate, self.tag, self.args, self.target_range_start, self.target_range_end
        )?;
        for (index, target) in self.targets_in_range.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(
                f,
                "stim::GateTargetWithCoords({:?}, {:?})",
                target.gate_target(),
                target.coords()
            )?;
        }
        f.write_str("))")
    }
}

fn compare_f64_slices(left: &[f64], right: &[f64]) -> std::cmp::Ordering {
    for (l, r) in left.iter().zip(right.iter()) {
        let cmp = l.to_bits().cmp(&r.to_bits());
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
    }
    left.len().cmp(&right.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    use crate::{GateTarget, target_x};

    #[test]
    fn constructor_and_field_accessors_preserve_values() {
        let first = GateTargetWithCoords::new(GateTarget::new(5), vec![1.0, 2.0]);
        let second = GateTargetWithCoords::new(target_x(6, false).unwrap(), vec![3.5]);

        let inside = CircuitTargetsInsideInstruction::new(
            "X_ERROR",
            "look-at-me",
            vec![0.25],
            2,
            5,
            vec![first.clone(), second.clone()],
        );

        assert_eq!(inside.gate(), "X_ERROR");
        assert_eq!(inside.tag(), "look-at-me");
        assert_eq!(inside.args(), &[0.25]);
        assert_eq!(inside.target_range_start(), 2);
        assert_eq!(inside.target_range_end(), 5);
        assert_eq!(inside.targets_in_range(), &[first, second]);
    }

    #[test]
    fn equality_hash_and_order_follow_all_fields() {
        let first = CircuitTargetsInsideInstruction::new(
            "X_ERROR",
            "",
            vec![0.25],
            0,
            1,
            vec![GateTargetWithCoords::new(GateTarget::new(5), vec![1.0])],
        );
        let same = CircuitTargetsInsideInstruction::new(
            "X_ERROR",
            "",
            vec![0.25],
            0,
            1,
            vec![GateTargetWithCoords::new(GateTarget::new(5), vec![1.0])],
        );
        let different = CircuitTargetsInsideInstruction::new(
            "Z_ERROR",
            "",
            vec![0.125],
            0,
            1,
            vec![GateTargetWithCoords::new(GateTarget::new(5), vec![1.0])],
        );

        assert_eq!(first, same);
        assert_ne!(first, different);

        let mut hashed = HashSet::new();
        hashed.insert(first.clone());
        hashed.insert(same);
        hashed.insert(different.clone());
        assert_eq!(hashed.len(), 2);

        let ordered = BTreeSet::from([different.clone(), first.clone()]);
        let ordered_vec: Vec<_> = ordered.into_iter().collect();
        assert_eq!(ordered_vec, vec![first, different]);
    }

    #[test]
    fn display_and_debug_match_upstream_like_shapes() {
        let inside = CircuitTargetsInsideInstruction::new(
            "MPP",
            "annotated",
            vec![0.5, 1.5],
            3,
            5,
            vec![
                GateTargetWithCoords::new(target_x(2, false).unwrap(), vec![1.0, 2.0]),
                GateTargetWithCoords::new(GateTarget::combiner(), vec![]),
                GateTargetWithCoords::new(GateTarget::new(3), vec![]),
            ],
        );

        assert_eq!(
            inside.to_string(),
            "MPP[annotated](0.5,1.5) X2[coords 1,2]* 3"
        );
        assert_eq!(
            format!("{inside:?}"),
            "stim::CircuitTargetsInsideInstruction(gate=\"MPP\", tag=\"annotated\", args=[0.5, 1.5], target_range_start=3, target_range_end=5, targets_in_range=(stim::GateTargetWithCoords(stim::target_x(2), [1.0, 2.0]), stim::GateTargetWithCoords(stim::target_combiner(), []), stim::GateTargetWithCoords(stim::GateTarget(3), [])))"
        );
    }
}
