use std::path::Path;

use super::bit_packing::{pack_rows_array, unpack_rows_array};
use super::{Result, ShotDataFormat, StimError};
use ndarray::{Array2, ArrayView2};

/// Reads shot data from a file in one of Stim's supported formats.
///
/// This function is the Rust equivalent of Python's `stim.read_shot_data_file`.
/// It reads measurement, detector, and observable data that was previously
/// written by Stim (or by [`write_shot_data_file`]) and returns it as a 2D
/// boolean array.
///
/// Each row of the returned array contains the concatenated measurement,
/// detector, and observable bits for a single shot. The total number of
/// columns equals `num_measurements + num_detectors + num_observables`.
///
/// # Arguments
///
/// * `filepath` - Path to the file to read data from.
/// * `format_name` - The format the data is stored in (see below).
/// * `num_measurements` - How many measurement bits there are per shot.
/// * `num_detectors` - How many detector bits there are per shot.
/// * `num_observables` - How many observable bits there are per shot.
///   Note that this refers to observables *stored in the file*, not
///   necessarily all observables from the original circuit.
///
/// # Supported formats
///
/// Common format names include:
/// - `"01"` -- one ASCII character per bit, one line per shot
/// - `"b8"` -- bit-packed binary, 8 bits per byte, little-endian within bytes
/// - `"r8"` -- run-length encoded binary
/// - `"ptb64"` -- partially transposed bit-packed format
/// - `"dets"` -- detector-event format (`shot D0 D3 L1`)
/// - `"hits"` -- space-separated indices of set bits
///
/// See the [Stim result formats documentation](https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md)
/// for full details.
///
/// # Errors
///
/// Returns a [`StimError`] if the file path is not valid UTF-8, the file
/// cannot be read, or its contents do not match the declared format and
/// dimensions.
///
/// # Examples
///
/// Round-trip through the `"01"` plain-text format:
///
/// ```
/// use stim::{read_shot_data_file, write_shot_data_file};
/// use ndarray::Array2;
///
/// let path = std::env::temp_dir().join("stim-doctest-read.01");
///
/// let data = Array2::from_shape_vec(
///     (2, 3),
///     vec![true, false, true, false, true, false],
/// ).expect("shape should be valid");
///
/// write_shot_data_file(data.view(), &path, stim::ShotDataFormat::Bits01, 3, 0, 0)
///     .expect("write should succeed");
///
/// let read_back = read_shot_data_file(&path, stim::ShotDataFormat::Bits01, 3, 0, 0)
///     .expect("read should succeed");
/// assert_eq!(read_back, data);
///
/// std::fs::remove_file(&path).expect("cleanup should succeed");
/// ```
pub fn read_shot_data_file(
    filepath: impl AsRef<Path>,
    format_name: ShotDataFormat,
    num_measurements: usize,
    num_detectors: usize,
    num_observables: usize,
) -> Result<Array2<bool>> {
    let path = filepath
        .as_ref()
        .to_str()
        .ok_or_else(|| StimError::new("filepath must be valid UTF-8"))?;
    let bit_len = num_measurements + num_detectors + num_observables;
    let packed = stim_cxx::read_shot_data_file_bit_packed(
        path,
        format_name.as_str(),
        num_measurements as u64,
        num_detectors as u64,
        num_observables as u64,
    )
    .map_err(StimError::from)?;
    Ok(unpack_rows_array(&packed, bit_len))
}

/// Writes shot data to a file in one of Stim's supported formats.
///
/// This function is the Rust equivalent of Python's `stim.write_shot_data_file`.
/// It serializes a 2D boolean array of measurement/detector/observable data
/// into one of Stim's shot-data file formats.
///
/// Each row of `data` must contain exactly
/// `num_measurements + num_detectors + num_observables` columns. The columns
/// are interpreted in that order: measurements first, then detectors, then
/// observables.
///
/// # Arguments
///
/// * `data` - Boolean matrix of shape `(num_shots, total_bits)` where
///   `total_bits = num_measurements + num_detectors + num_observables`.
/// * `filepath` - Path to the file to write data to.
/// * `format_name` - The output format (see [`read_shot_data_file`] for the
///   list of supported format names).
/// * `num_measurements` - How many measurement bits there are per shot.
/// * `num_detectors` - How many detector bits there are per shot.
/// * `num_observables` - How many observable bits there are per shot.
///   Note that this refers to observables *in the given shot data*, not
///   necessarily all observables from the original circuit.
///
/// # Errors
///
/// Returns a [`StimError`] if the file path is not valid UTF-8, the row
/// width does not match the declared dimensions, or the file cannot be
/// written.
///
/// # Examples
///
/// Write two shots of three measurement bits to a `"01"` file:
///
/// ```
/// use stim::write_shot_data_file;
/// use ndarray::Array2;
///
/// let path = std::env::temp_dir().join("stim-doctest-write.01");
///
/// let data = Array2::from_shape_vec(
///     (2, 3),
///     vec![true, false, true, false, true, false],
/// ).expect("shape should be valid");
///
/// write_shot_data_file(data.view(), &path, stim::ShotDataFormat::Bits01, 3, 0, 0)
///     .expect("write should succeed");
///
/// let contents = std::fs::read_to_string(&path)
///     .expect("file should be readable");
/// assert_eq!(contents, "101\n010\n");
///
/// std::fs::remove_file(&path).expect("cleanup should succeed");
/// ```
pub fn write_shot_data_file(
    data: ArrayView2<'_, bool>,
    filepath: impl AsRef<Path>,
    format_name: ShotDataFormat,
    num_measurements: usize,
    num_detectors: usize,
    num_observables: usize,
) -> Result<()> {
    let path = filepath
        .as_ref()
        .to_str()
        .ok_or_else(|| StimError::new("filepath must be valid UTF-8"))?;
    let bit_len = num_measurements + num_detectors + num_observables;
    let packed = pack_rows_array(data, bit_len)?;
    stim_cxx::write_shot_data_file_bit_packed(
        &packed,
        data.nrows() as u64,
        path,
        format_name.as_str(),
        num_measurements as u64,
        num_detectors as u64,
        num_observables as u64,
    )
    .map_err(StimError::from)
}

#[cfg(test)]
mod tests {
    use super::{ShotDataFormat, read_shot_data_file, write_shot_data_file};
    use ndarray::Array2;
    use std::fs;
    use std::path::PathBuf;
    use std::time::SystemTime;

    fn bool_matrix(rows: Vec<Vec<bool>>) -> Array2<bool> {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, Vec::len);
        Array2::from_shape_vec((nrows, ncols), rows.into_iter().flatten().collect())
            .expect("rows should be rectangular")
    }

    fn unique_temp_path_with_extension(name: &str, extension: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        std::env::temp_dir().join(format!("stim-rs-{name}-{nanos}.{extension}"))
    }

    #[test]
    fn shot_data_file_round_trips_measurement_rows_in_01_format() {
        let path = unique_temp_path_with_extension("shot-data-01", "01");
        let data = vec![
            vec![false, true, false, true],
            vec![true, true, false, false],
            vec![false, false, false, true],
        ];

        write_shot_data_file(
            bool_matrix(data.clone()).view(),
            &path,
            ShotDataFormat::Bits01,
            4,
            0,
            0,
        )
        .expect("write should succeed");

        let read_back = read_shot_data_file(&path, ShotDataFormat::Bits01, 4, 0, 0)
            .expect("read should succeed");
        assert_eq!(read_back, bool_matrix(data.clone()));
        assert_eq!(
            fs::read_to_string(&path).expect("written file should read as text"),
            "0101\n1100\n0001\n"
        );

        fs::remove_file(path).expect("temporary file should delete");
    }

    #[test]
    fn shot_data_file_round_trips_all_width_kinds_in_b8_format() {
        let path = unique_temp_path_with_extension("shot-data-b8", "b8");
        let data = vec![
            vec![false, true, true, false, true, false, true],
            vec![true, false, false, true, false, true, false],
            vec![true, true, false, false, false, false, true],
        ];

        write_shot_data_file(
            bool_matrix(data.clone()).view(),
            &path,
            ShotDataFormat::B8,
            2,
            3,
            2,
        )
        .expect("write should succeed");

        let read_back = read_shot_data_file(&path, ShotDataFormat::B8, 2, 3, 2)
            .expect("mixed-width read should succeed");
        assert_eq!(read_back, bool_matrix(data.clone()));

        fs::remove_file(path).expect("temporary file should delete");
    }

    #[test]
    fn shot_data_file_round_trips_detector_and_observable_rows_in_dets_format() {
        let path = unique_temp_path_with_extension("shot-data-dets", "dets");
        let data = vec![
            vec![false, false, false, false, false, false],
            vec![true, false, true, false, true, false],
            vec![false, true, false, true, false, true],
        ];

        write_shot_data_file(
            bool_matrix(data.clone()).view(),
            &path,
            ShotDataFormat::Dets,
            0,
            4,
            2,
        )
        .expect("write should succeed");

        let read_back = read_shot_data_file(&path, ShotDataFormat::Dets, 0, 4, 2)
            .expect("dets read should succeed");
        assert_eq!(read_back, bool_matrix(data.clone()));
        assert_eq!(
            fs::read_to_string(&path).expect("written file should read as text"),
            "shot\nshot D0 D2 L0\nshot D1 D3 L1\n"
        );

        fs::remove_file(path).expect("temporary file should delete");
    }

    #[test]
    fn write_shot_data_file_rejects_inconsistent_row_widths() {
        let path = unique_temp_path_with_extension("shot-data-invalid-width", "01");
        let data = bool_matrix(vec![vec![true, false, true]]);

        let error = write_shot_data_file(data.view(), &path, ShotDataFormat::Bits01, 4, 0, 0)
            .expect_err("write should fail");

        assert_eq!(error.message(), "expected 4 bits per shot, got 3");
        assert!(
            !path.exists(),
            "validation should fail before any file is created"
        );
    }

    #[cfg(unix)]
    #[test]
    fn shot_data_file_rejects_non_utf8_paths() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let invalid_path = PathBuf::from(OsString::from_vec(vec![0x66, 0x6f, 0x80, 0x6f]));
        let data = vec![vec![true, false, true]];

        let write_error = write_shot_data_file(
            bool_matrix(data).view(),
            &invalid_path,
            ShotDataFormat::Bits01,
            3,
            0,
            0,
        )
        .expect_err("write should fail");
        assert_eq!(write_error.message(), "filepath must be valid UTF-8");

        let read_error = read_shot_data_file(&invalid_path, ShotDataFormat::Bits01, 3, 0, 0)
            .expect_err("read should fail");
        assert_eq!(read_error.message(), "filepath must be valid UTF-8");
    }
}
