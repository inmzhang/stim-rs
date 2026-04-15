use crate::{DemTarget, Result, target_logical_observable_id, target_relative_detector_id};

#[derive(Clone, Debug, PartialEq)]
pub enum DetectingRegionFilter {
    AllDetectors,
    AllObservables,
    Target(DemTarget),
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
