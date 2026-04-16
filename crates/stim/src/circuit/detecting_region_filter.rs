use crate::{DemTarget, Result};

/// A filter that selects which detecting regions to return from a circuit
/// analysis.
///
/// When querying a circuit for its detecting regions (via methods like
/// `Circuit::detecting_regions`), you often want only a subset of the
/// results rather than every detector and observable in the entire
/// circuit. This enum encodes the available filtering strategies:
///
/// - [`AllDetectors`](Self::AllDetectors) -- include every detector.
/// - [`AllObservables`](Self::AllObservables) -- include every logical
///   observable.
/// - [`Target`](Self::Target) -- include a single specific
///   [`DemTarget`](crate::DemTarget) (one detector or one observable).
/// - [`DetectorCoordinatePrefix`](Self::DetectorCoordinatePrefix) --
///   include only detectors whose coordinate vector starts with a given
///   prefix. This is useful for spatially filtering detectors in a
///   layout where coordinates encode physical qubit positions.
///
/// # Examples
///
/// ```
/// use stim::DetectingRegionFilter;
///
/// // Select all detectors in the circuit.
/// let filter = DetectingRegionFilter::AllDetectors;
///
/// // Select only detectors whose first two coordinates are (1.0, 2.0).
/// let spatial = DetectingRegionFilter::DetectorCoordinatePrefix(vec![1.0, 2.0]);
///
/// // Select a single specific target.
/// let target = stim::DemTarget::relative_detector_id(3).expect("valid id");
/// let single = DetectingRegionFilter::Target(target);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum DetectingRegionFilter {
    /// Match all detectors in the circuit.
    AllDetectors,
    /// Match all logical observables in the circuit.
    AllObservables,
    /// Match a single specific [`DemTarget`](crate::DemTarget) (either a
    /// detector or a logical observable).
    Target(DemTarget),
    /// Match detectors whose coordinate vector starts with the given
    /// floating-point prefix.
    ///
    /// For example, a prefix of `[1.0, 2.0]` matches any detector whose
    /// first two coordinates are `1.0` and `2.0`, regardless of how many
    /// additional coordinate values follow. This is especially useful in
    /// surface-code-like layouts where coordinates encode spatial
    /// positions and you want to filter detectors by region.
    DetectorCoordinatePrefix(Vec<f64>),
}

impl DetectingRegionFilter {
    pub(crate) fn matching_targets(
        &self,
        num_detectors: u64,
        num_observables: u64,
        detector_coordinates: impl FnOnce() -> Result<std::collections::BTreeMap<u64, Vec<f64>>>,
    ) -> Result<Vec<DemTarget>> {
        match self {
            Self::AllDetectors => (0..num_detectors)
                .map(DemTarget::relative_detector_id)
                .collect(),
            Self::AllObservables => (0..num_observables)
                .map(DemTarget::logical_observable_id)
                .collect(),
            Self::Target(target) => Ok(vec![*target]),
            Self::DetectorCoordinatePrefix(prefix) => {
                detector_coordinates().map(|coords| {
                    coords
                        .into_iter()
                        .filter_map(|(index, candidate)| {
                            if candidate.starts_with(prefix) {
                                Some(DemTarget::relative_detector_id(index).expect(
                                    "detector ids from existing coordinates should be valid",
                                ))
                            } else {
                                None
                            }
                        })
                        .collect()
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DetectingRegionFilter;

    #[test]
    fn coordinate_prefix_filter_skips_non_matching_detectors() {
        let matches = DetectingRegionFilter::DetectorCoordinatePrefix(vec![1.0, 2.0])
            .matching_targets(3, 0, || {
                Ok(std::collections::BTreeMap::from([
                    (0, vec![1.0, 2.0]),
                    (1, vec![1.0, 3.0]),
                ]))
            })
            .unwrap();
        assert_eq!(
            matches,
            vec![crate::DemTarget::relative_detector_id(0).unwrap()]
        );
    }
}
