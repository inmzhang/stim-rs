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

pub fn parse_detector_coordinate_map(serialized: &str) -> Result<BTreeMap<u64, Vec<f64>>> {
    let mut result = BTreeMap::new();
    for line in serialized.lines() {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split('\t');
        let key = parts
            .next()
            .ok_or_else(|| StimError::new("missing detector coordinate key"))?
            .parse::<u64>()
            .map_err(|_| StimError::new(format!("invalid detector coordinate key: {line}")))?;
        let coords = parts
            .map(|part| {
                part.parse::<f64>()
                    .map_err(|_| StimError::new(format!("invalid detector coordinate: {part}")))
            })
            .collect::<Result<Vec<_>>>()?;
        result.insert(key, coords);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{decode_measurement_solution, parse_detector_coordinate_map};

    #[test]
    fn parse_helpers_accept_empty_markers_and_blank_lines() {
        assert_eq!(decode_measurement_solution("!".to_string()).unwrap(), None);
        assert_eq!(
            decode_measurement_solution(String::new()).unwrap(),
            Some(Vec::new())
        );
        assert_eq!(
            parse_detector_coordinate_map("\n1\t2\t3\n").unwrap(),
            std::collections::BTreeMap::from([(1, vec![2.0, 3.0])])
        );
    }
}
