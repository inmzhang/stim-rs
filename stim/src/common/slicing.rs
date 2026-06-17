pub fn normalize_index(index: isize, len: usize) -> Option<usize> {
    let len = len as isize;
    let index = if index < 0 { len + index } else { index };
    (0..len).contains(&index).then_some(index as usize)
}

pub fn compute_slice_indices(
    len: isize,
    start: Option<isize>,
    stop: Option<isize>,
    step: isize,
) -> Vec<isize> {
    let normalize = |value: isize| if value < 0 { len + value } else { value };

    let (mut start, stop) = match step.cmp(&0) {
        std::cmp::Ordering::Greater => (
            start.map(normalize).unwrap_or(0).clamp(0, len),
            stop.map(normalize).unwrap_or(len).clamp(0, len),
        ),
        std::cmp::Ordering::Less => (
            start.map(normalize).unwrap_or(len - 1).clamp(-1, len - 1),
            stop.map(normalize).unwrap_or(-1).clamp(-1, len - 1),
        ),
        std::cmp::Ordering::Equal => unreachable!(),
    };

    let mut indices = Vec::new();
    if step > 0 {
        while start < stop {
            indices.push(start);
            start += step;
        }
    } else {
        while start > stop {
            indices.push(start);
            start += step;
        }
    }
    indices
}
