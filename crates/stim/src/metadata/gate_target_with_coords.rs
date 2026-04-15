use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

use super::GateTarget;

#[derive(Clone, PartialEq)]
pub struct GateTargetWithCoords {
    gate_target: GateTarget,
    coords: Vec<f64>,
}

impl GateTargetWithCoords {
    #[must_use]
    pub fn new(gate_target: impl Into<GateTarget>, coords: Vec<f64>) -> Self {
        Self {
            gate_target: gate_target.into(),
            coords,
        }
    }

    #[must_use]
    pub fn gate_target(&self) -> GateTarget {
        self.gate_target
    }

    #[must_use]
    pub fn coords(&self) -> &[f64] {
        &self.coords
    }
}

impl Eq for GateTargetWithCoords {}

impl PartialOrd for GateTargetWithCoords {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GateTargetWithCoords {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gate_target
            .cmp(&other.gate_target)
            .then_with(|| compare_coord_slices(&self.coords, &other.coords))
    }
}

impl Hash for GateTargetWithCoords {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.gate_target.hash(state);
        for coord in &self.coords {
            coord.to_bits().hash(state);
        }
    }
}

impl Display for GateTargetWithCoords {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.gate_target)?;
        if !self.coords.is_empty() {
            f.write_str("[coords ")?;
            for (index, coord) in self.coords.iter().enumerate() {
                if index > 0 {
                    f.write_str(",")?;
                }
                write!(f, "{coord}")?;
            }
            f.write_str("]")?;
        }
        Ok(())
    }
}

impl fmt::Debug for GateTargetWithCoords {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("stim::GateTargetWithCoords")
            .field("gate_target", &self.gate_target)
            .field("coords", &self.coords)
            .finish()
    }
}

fn compare_coord_slices(left: &[f64], right: &[f64]) -> std::cmp::Ordering {
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
    use std::collections::{BTreeSet, HashSet};

    use crate::{GateTarget, target_x};

    use super::GateTargetWithCoords;

    #[test]
    fn gate_target_with_coords_constructor_exposes_target_and_coords() {
        let gate_target = target_x(5u32, false).expect("X target should build");
        let target_with_coords = GateTargetWithCoords::new(gate_target, vec![1.0, -2.5, 3.25]);

        assert_eq!(target_with_coords.gate_target(), gate_target);
        assert_eq!(target_with_coords.coords(), &[1.0, -2.5, 3.25]);
    }

    #[test]
    fn gate_target_with_coords_supports_equality_order_and_hash() {
        let qubit = GateTarget::new(5u32);
        let pauli = target_x(5u32, false).expect("X target should build");

        let first = GateTargetWithCoords::new(qubit, vec![1.0]);
        let same_as_first = GateTargetWithCoords::new(qubit, vec![1.0]);
        let second = GateTargetWithCoords::new(qubit, vec![2.0]);
        let third = GateTargetWithCoords::new(pauli, vec![]);

        assert_eq!(first, same_as_first);
        assert_ne!(first, second);
        assert_ne!(first, third);

        let ordered = [
            third.clone(),
            second.clone(),
            same_as_first.clone(),
            first.clone(),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
        assert_eq!(ordered, vec![first.clone(), second, third.clone()]);

        let hashed = [first.clone(), same_as_first, third.clone()]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 2);
        assert!(hashed.contains(&first));
        assert!(hashed.contains(&third));
    }

    #[test]
    fn gate_target_with_coords_display_and_debug_match_rust_binding_conventions() {
        let qubit = GateTargetWithCoords::new(GateTarget::new(5u32), vec![1.0, -2.5]);
        let pauli = GateTargetWithCoords::new(
            target_x(5u32, false).expect("X target should build"),
            vec![],
        );

        assert_eq!(qubit.to_string(), "5[coords 1,-2.5]");
        assert_eq!(
            format!("{qubit:?}"),
            "stim::GateTargetWithCoords { gate_target: stim::GateTarget(5), coords: [1.0, -2.5] }"
        );

        assert_eq!(pauli.to_string(), "X5");
        assert_eq!(
            format!("{pauli:?}"),
            "stim::GateTargetWithCoords { gate_target: stim::target_x(5), coords: [] }"
        );
    }
}
