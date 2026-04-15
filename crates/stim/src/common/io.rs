use std::path::Path;

use super::bit_packing::{pack_rows_array, unpack_rows_array};
use super::{Result, StimError};
use ndarray::{Array2, ArrayView2};

pub fn read_shot_data_file(
    filepath: impl AsRef<Path>,
    format_name: &str,
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
        format_name,
        num_measurements as u64,
        num_detectors as u64,
        num_observables as u64,
    )
    .map_err(StimError::from)?;
    Ok(unpack_rows_array(&packed, bit_len))
}

pub fn write_shot_data_file(
    data: ArrayView2<'_, bool>,
    filepath: impl AsRef<Path>,
    format_name: &str,
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
        format_name,
        num_measurements as u64,
        num_detectors as u64,
        num_observables as u64,
    )
    .map_err(StimError::from)
}

#[cfg(test)]
mod tests {
    use super::{read_shot_data_file, write_shot_data_file};
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

        write_shot_data_file(bool_matrix(data.clone()).view(), &path, "01", 4, 0, 0)
            .expect("write should succeed");

        let read_back = read_shot_data_file(&path, "01", 4, 0, 0).expect("read should succeed");
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

        write_shot_data_file(bool_matrix(data.clone()).view(), &path, "b8", 2, 3, 2)
            .expect("write should succeed");

        let read_back =
            read_shot_data_file(&path, "b8", 2, 3, 2).expect("mixed-width read should succeed");
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

        write_shot_data_file(bool_matrix(data.clone()).view(), &path, "dets", 0, 4, 2)
            .expect("write should succeed");

        let read_back =
            read_shot_data_file(&path, "dets", 0, 4, 2).expect("dets read should succeed");
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

        let error =
            write_shot_data_file(data.view(), &path, "01", 4, 0, 0).expect_err("write should fail");

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

        let write_error =
            write_shot_data_file(bool_matrix(data).view(), &invalid_path, "01", 3, 0, 0)
                .expect_err("write should fail");
        assert_eq!(write_error.message(), "filepath must be valid UTF-8");

        let read_error =
            read_shot_data_file(&invalid_path, "01", 3, 0, 0).expect_err("read should fail");
        assert_eq!(read_error.message(), "filepath must be valid UTF-8");
    }
}
