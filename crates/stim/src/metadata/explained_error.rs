use std::fmt::{self, Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

use crate::{CircuitErrorLocation, DemTargetWithCoords};

/// Describes a single error mechanism from a Stim circuit, pairing the
/// detectors and observables it affects with the location(s) where it
/// occurs in the circuit.
///
/// When you call methods like
/// [`Circuit::shortest_graphlike_error`](crate::Circuit::shortest_graphlike_error),
/// Stim returns a list of `ExplainedError` values. Each one represents
/// one fault mechanism that was needed to produce the logical error,
/// containing:
///
/// - **`dem_error_terms`** -- the detectors and logical observables
///   flipped by this error, each annotated with coordinate data (a
///   list of [`DemTargetWithCoords`](crate::DemTargetWithCoords)).
/// - **`circuit_error_locations`** -- the physical location(s) in the
///   circuit where this error can occur, as
///   [`CircuitErrorLocation`] values.
///
/// If `circuit_error_locations` is empty, it means there was a DEM
/// error that was decomposed into parts where one of the parts is
/// impossible to produce from a single circuit error on its own.
///
/// If the list contains a single entry, it may be because only a
/// single representative circuit error was requested (as opposed to
/// all possible errors that produce the same syndrome).
///
/// # Examples
///
/// ```
/// use stim::{
///     CircuitErrorLocation, CircuitErrorLocationStackFrame,
///     CircuitTargetsInsideInstruction, DemTargetWithCoords,
///     ExplainedError, GateTarget, GateTargetWithCoords,
/// };
///
/// let explained = ExplainedError::new(
///     vec![DemTargetWithCoords::new(
///         stim::DemTarget::logical_observable_id(0).expect("valid id"),
///         vec![],
///     )],
///     vec![CircuitErrorLocation::new(
///         1,
///         vec![GateTargetWithCoords::new(
///             stim::GateTarget::y(0u32, false).expect("valid target"),
///             vec![],
///         )],
///         None,
///         CircuitTargetsInsideInstruction::new(
///             "Y_ERROR", "", vec![0.125], 0, 1,
///             vec![GateTargetWithCoords::new(GateTarget::from(0u32), vec![])],
///         ),
///         vec![CircuitErrorLocationStackFrame::new(2, 0, 0)],
///         "",
///     )],
/// );
///
/// assert_eq!(explained.dem_error_terms().len(), 1);
/// assert_eq!(explained.circuit_error_locations().len(), 1);
/// ```
///
/// [`CircuitErrorLocation`]: crate::CircuitErrorLocation
#[derive(Clone, PartialEq)]
pub struct ExplainedError {
    dem_error_terms: Vec<DemTargetWithCoords>,
    circuit_error_locations: Vec<CircuitErrorLocation>,
}

impl ExplainedError {
    /// Creates a new `ExplainedError` from its constituent parts.
    ///
    /// # Arguments
    ///
    /// - `dem_error_terms` -- the detectors and logical observables
    ///   flipped by this error mechanism, each paired with coordinate
    ///   data.
    /// - `circuit_error_locations` -- the physical circuit location(s)
    ///   where this error can occur. May be empty if the error was
    ///   decomposed into parts that cannot individually be produced by
    ///   a single circuit error.
    #[must_use]
    pub fn new(
        dem_error_terms: impl IntoIterator<Item = DemTargetWithCoords>,
        circuit_error_locations: impl IntoIterator<Item = CircuitErrorLocation>,
    ) -> Self {
        Self {
            dem_error_terms: dem_error_terms.into_iter().collect(),
            circuit_error_locations: circuit_error_locations.into_iter().collect(),
        }
    }

    /// Returns the detectors and observables flipped by this error
    /// mechanism.
    ///
    /// Each element is a [`DemTargetWithCoords`](crate::DemTargetWithCoords)
    /// pairing a detector (`D5`) or logical observable (`L0`) with its
    /// coordinate data from `DETECTOR` or other coordinate-assigning
    /// instructions.
    #[must_use]
    pub fn dem_error_terms(&self) -> &[DemTargetWithCoords] {
        &self.dem_error_terms
    }

    /// Returns the locations of circuit errors that produce the symptoms
    /// described by [`dem_error_terms`](Self::dem_error_terms).
    ///
    /// Each element is a [`CircuitErrorLocation`] that identifies a
    /// specific instruction, target range, and nesting position within
    /// the circuit.
    ///
    /// If this slice is empty, it means the DEM error was decomposed
    /// into parts where one part is impossible to produce from a single
    /// circuit error on its own. If it contains a single entry, it may
    /// be because only a single representative was requested.
    ///
    /// [`CircuitErrorLocation`]: crate::CircuitErrorLocation
    #[must_use]
    pub fn circuit_error_locations(&self) -> &[CircuitErrorLocation] {
        &self.circuit_error_locations
    }
}

impl Eq for ExplainedError {}

impl PartialOrd for ExplainedError {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExplainedError {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dem_error_terms
            .cmp(&other.dem_error_terms)
            .then_with(|| {
                self.circuit_error_locations
                    .cmp(&other.circuit_error_locations)
            })
    }
}

impl Hash for ExplainedError {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dem_error_terms.hash(state);
        self.circuit_error_locations.hash(state);
    }
}

impl Display for ExplainedError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(
            "ExplainedError {
    dem_error_terms: ",
        )?;
        for (index, term) in self.dem_error_terms.iter().enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{term}")?;
        }
        if self.circuit_error_locations.is_empty() {
            f.write_str(
                "
    [no single circuit error had these exact symptoms]",
            )?;
        } else {
            for location in &self.circuit_error_locations {
                f.write_str(
                    "
",
                )?;
                write_indented_lines(f, &location.to_string(), "    ")?;
            }
        }
        f.write_str(
            "
}",
        )
    }
}

impl Debug for ExplainedError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("stim::ExplainedError")
            .field("dem_error_terms", &self.dem_error_terms)
            .field("circuit_error_locations", &self.circuit_error_locations)
            .finish()
    }
}

fn write_indented_lines(f: &mut Formatter<'_>, value: &str, indent: &str) -> fmt::Result {
    for (index, line) in value.lines().enumerate() {
        if index > 0 {
            f.write_str(
                "
",
            )?;
        }
        f.write_str(indent)?;
        f.write_str(line)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use super::ExplainedError;
    use crate::{
        CircuitErrorLocation, CircuitErrorLocationStackFrame, CircuitTargetsInsideInstruction,
        DemTarget, DemTargetWithCoords, GateTarget, GateTargetWithCoords,
    };

    fn sample_circuit_error_location() -> CircuitErrorLocation {
        CircuitErrorLocation::new(
            5,
            vec![GateTargetWithCoords::new(
                GateTarget::y(6, false).expect("pauli target should build"),
                vec![1.0, 2.0, 3.0],
            )],
            None,
            CircuitTargetsInsideInstruction::new(
                "X_ERROR",
                "",
                vec![0.25],
                2,
                5,
                vec![
                    GateTargetWithCoords::new(GateTarget::from(5), vec![1.0, 2.0]),
                    GateTargetWithCoords::new(GateTarget::from(6), vec![1.0, 3.0]),
                    GateTargetWithCoords::new(GateTarget::from(7), vec![]),
                ],
            ),
            vec![
                CircuitErrorLocationStackFrame::new(1, 0, 3),
                CircuitErrorLocationStackFrame::new(1, 2, 0),
            ],
            "",
        )
    }

    #[test]
    fn explained_error_constructor_and_accessors_preserve_values() {
        let detector = DemTargetWithCoords::new(
            DemTarget::relative_detector_id(5).expect("detector target should build"),
            vec![1.0, -2.5],
        );
        let logical = DemTargetWithCoords::new(
            DemTarget::logical_observable_id(2).expect("logical target should build"),
            vec![],
        );
        let location = sample_circuit_error_location();

        let explained = ExplainedError::new(
            vec![detector.clone(), logical.clone()],
            vec![location.clone()],
        );

        assert_eq!(explained.dem_error_terms(), &[detector, logical]);
        assert_eq!(explained.circuit_error_locations(), &[location]);
    }

    #[test]
    fn explained_error_supports_equality_hash_and_order() {
        let detector = DemTargetWithCoords::new(
            DemTarget::relative_detector_id(5).expect("detector target should build"),
            vec![1.0],
        );
        let logical = DemTargetWithCoords::new(
            DemTarget::logical_observable_id(0).expect("logical target should build"),
            vec![],
        );
        let location = sample_circuit_error_location();

        let first = ExplainedError::new(vec![detector.clone()], vec![]);
        let same_as_first = ExplainedError::new(vec![detector.clone()], vec![]);
        let second = ExplainedError::new(vec![detector.clone()], vec![location.clone()]);
        let third = ExplainedError::new(vec![logical.clone()], vec![]);

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
    fn explained_error_display_and_debug_have_stable_shapes() {
        let empty = ExplainedError::new(
            vec![DemTargetWithCoords::new(
                DemTarget::relative_detector_id(5).expect("detector target should build"),
                vec![1.0, 2.0, 3.0],
            )],
            vec![],
        );
        let populated = ExplainedError::new(
            vec![DemTargetWithCoords::new(
                DemTarget::relative_detector_id(5).expect("detector target should build"),
                vec![1.0, -2.5],
            )],
            vec![sample_circuit_error_location()],
        );

        assert_eq!(
            empty.to_string(),
            "ExplainedError {\n    dem_error_terms: D5[coords 1,2,3]\n    [no single circuit error had these exact symptoms]\n}"
        );
        assert_eq!(
            format!("{empty:?}"),
            "stim::ExplainedError { dem_error_terms: [stim::DemTargetWithCoords { dem_target: stim::DemTarget('D5'), coords: [1.0, 2.0, 3.0] }], circuit_error_locations: [] }"
        );

        assert_eq!(
            populated.to_string(),
            "ExplainedError {\n    dem_error_terms: D5[coords 1,-2.5]\n    CircuitErrorLocation {\n        flipped_pauli_product: Y6[coords 1,2,3]\n        Circuit location stack trace:\n            (after 5 TICKs)\n            at instruction #2 (a REPEAT 3 block) in the circuit\n            after 2 completed iterations\n            at instruction #2 (X_ERROR) in the REPEAT block\n            at targets #3 to #5 of the instruction\n            resolving to X_ERROR(0.25) 5[coords 1,2] 6[coords 1,3] 7\n    }\n}"
        );

        let debug_output = format!("{populated:?}");
        assert!(debug_output.starts_with("stim::ExplainedError { dem_error_terms: [stim::DemTargetWithCoords { dem_target: stim::DemTarget('D5'), coords: [1.0, -2.5] }], circuit_error_locations: ["));
        assert!(debug_output.ends_with("] }"));
        assert!(debug_output.contains("CircuitErrorLocation"));
        assert!(debug_output.contains("flipped_pauli_product"));
    }

    #[test]
    fn explained_error_can_store_multiple_dem_terms() {
        let explained = ExplainedError::new(
            vec![
                DemTargetWithCoords::new(
                    DemTarget::logical_observable_id(0).expect("logical target should build"),
                    vec![],
                ),
                DemTargetWithCoords::new(
                    DemTarget::relative_detector_id(5).expect("detector target should build"),
                    vec![1.0],
                ),
            ],
            vec![sample_circuit_error_location()],
        );

        assert_eq!(explained.dem_error_terms().len(), 2);
        assert_eq!(explained.dem_error_terms()[0].to_string(), "L0");
        assert_eq!(explained.dem_error_terms()[1].to_string(), "D5[coords 1]");
    }
}
