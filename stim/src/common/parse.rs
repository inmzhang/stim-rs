use std::collections::BTreeMap;

use super::{Result, StimError};

pub fn decode_measurement_solution(text: String) -> Result<Option<Vec<i32>>> {
    if text == "!" {
        return Ok(None);
    }
    if text.is_empty() {
        return Ok(Some(Vec::new()));
    }

    text.split(',')
        .map(|part| {
            part.parse::<i32>()
                .map_err(|_| StimError::new("invalid measurement solution index from stim-cxx"))
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

pub fn coordinate_entries_to_map(
    entries: Vec<stim_cxx::CoordinateEntryData>,
) -> BTreeMap<u64, Vec<f64>> {
    entries
        .into_iter()
        .map(|entry| (entry.index, entry.coords))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{coordinate_entries_to_map, decode_measurement_solution};

    #[test]
    fn parse_helpers_accept_empty_markers_and_blank_lines() {
        assert_eq!(decode_measurement_solution("!".to_string()).unwrap(), None);
        assert_eq!(
            decode_measurement_solution(String::new()).unwrap(),
            Some(Vec::new())
        );
        assert_eq!(
            coordinate_entries_to_map(vec![stim_cxx::CoordinateEntryData {
                index: 4,
                coords: vec![1.5, 2.5],
            }]),
            std::collections::BTreeMap::from([(4, vec![1.5, 2.5])])
        );
    }
}
