use crate::{DemTarget, Result, target_logical_observable_id, target_relative_detector_id};

/// A filter that selects which detecting regions to return from a circuit
/// analysis.
///
/// When querying a circuit for its detecting regions, you often want only a
/// subset — for example, only detectors, only observables, a specific target,
/// or detectors whose coordinates match a given prefix. This enum encodes
/// those filtering strategies.
#[derive(Clone, Debug, PartialEq)]
pub enum DetectingRegionFilter {
    /// Match all detectors in the circuit.
    AllDetectors,
    /// Match all logical observables in the circuit.
    AllObservables,
    /// Match a single specific [`DemTarget`] (detector or observable).
    Target(DemTarget),
    /// Match detectors whose coordinate vector starts with the given prefix.
    ///
    /// For example, a prefix of `[1.0, 2.0]` matches any detector whose
    /// first two coordinates are `1.0` and `2.0`, regardless of subsequent
    /// coordinate values.
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
                .map(target_relative_detector_id)
                .collect(),
            Self::AllObservables => (0..num_observables)
                .map(target_logical_observable_id)
                .collect(),
            Self::Target(target) => Ok(vec![*target]),
            Self::DetectorCoordinatePrefix(prefix) => {
                detector_coordinates().map(|coords| {
                    coords
                        .into_iter()
                        .filter_map(|(index, candidate)| {
                            if candidate.starts_with(prefix) {
                                Some(target_relative_detector_id(index).expect(
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
