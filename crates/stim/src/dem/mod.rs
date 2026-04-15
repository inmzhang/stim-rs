mod dem_append_operation;
mod dem_instruction;
mod dem_item;
mod dem_repeat_block;
mod dem_target;

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::path::Path;
use std::str::FromStr;

use crate::common::parse::parse_detector_coordinate_map;
use crate::common::slicing::{compute_slice_indices, normalize_index};
use crate::{DemSampler, Result, StimError};

pub use dem_append_operation::DemAppendOperation;
pub use dem_instruction::{DemInstruction, DemInstructionTarget};
pub use dem_item::DemItem;
pub use dem_repeat_block::DemRepeatBlock;
pub use dem_target::{
    DemTarget, DemTargetWithCoords, target_logical_observable_id, target_relative_detector_id,
    target_separator,
};

/// A detector error model describing how faults trigger detectors and observables.
pub struct DetectorErrorModel {
    pub(crate) inner: stim_cxx::DetectorErrorModel,
}

impl DetectorErrorModel {
    /// Creates an empty detector error model.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem = stim::DetectorErrorModel::new();
    /// assert!(dem.is_empty());
    /// assert_eq!(dem.len(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: stim_cxx::DetectorErrorModel::new(),
        }
    }

    /// Returns the number of top-level items in the detector error model.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the detector error model is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the total number of detectors mentioned by the model.
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        self.inner.num_detectors()
    }

    /// Returns the number of error mechanisms in the model.
    #[must_use]
    pub fn num_errors(&self) -> u64 {
        self.inner.num_errors()
    }

    /// Returns the number of observables mentioned by the model.
    #[must_use]
    pub fn num_observables(&self) -> u64 {
        self.inner.num_observables()
    }

    /// Clears all items from the detector error model.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Reads a detector error model from a file.
    ///
    /// # Examples
    ///
    /// ```
    /// let path = std::env::temp_dir().join("stim-rs-dem-from-file.dem");
    /// std::fs::write(&path, "error(0.25) D2 D3\nshift_detectors 4\n").unwrap();
    /// let model = stim::DetectorErrorModel::from_file(&path).unwrap();
    /// assert_eq!(model.to_string(), "error(0.25) D2 D3\nshift_detectors 4");
    /// std::fs::remove_file(path).unwrap();
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(|error| StimError::new(error.to_string()))?;
        Self::from_str(&text)
    }

    /// Writes the detector error model to a file with a trailing newline.
    ///
    /// # Examples
    ///
    /// ```
    /// let path = std::env::temp_dir().join("stim-rs-dem-to-file.dem");
    /// let model: stim::DetectorErrorModel = "error(0.25) D2 D3\nlogical_observable L1".parse().unwrap();
    /// model.to_file(&path).unwrap();
    /// assert_eq!(
    ///     std::fs::read_to_string(&path).unwrap(),
    ///     "error(0.25) D2 D3\nlogical_observable L1\n"
    /// );
    /// std::fs::remove_file(path).unwrap();
    /// ```
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, format!("{self}\n")).map_err(|error| StimError::new(error.to_string()))
    }

    /// Returns an owned copy of the detector error model.
    #[must_use]
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Compares two detector error models using an absolute tolerance on probabilities.
    ///
    /// # Examples
    ///
    /// ```
    /// let base: stim::DetectorErrorModel = "error(0.099) D0 D1".parse().unwrap();
    /// let near: stim::DetectorErrorModel = "error(0.101) D0 D1".parse().unwrap();
    /// assert!(base.approx_equals(&near, 0.01));
    /// assert!(!base.approx_equals(&near, 0.0001));
    /// ```
    #[must_use]
    pub fn approx_equals(&self, other: &Self, atol: f64) -> bool {
        self.inner.approx_equals(&other.inner, atol)
    }

    /// Returns the model with all tags removed.
    ///
    /// # Examples
    ///
    /// ```
    /// let model: stim::DetectorErrorModel =
    ///     "error[tag](0.25) D0\nrepeat[loop] 2 {\n    error[nested](0.125) D1\n}"
    ///         .parse()
    ///         .unwrap();
    /// assert_eq!(
    ///     model.without_tags().to_string(),
    ///     "error(0.25) D0\nrepeat 2 {\n    error(0.125) D1\n}"
    /// );
    /// ```
    #[must_use]
    pub fn without_tags(&self) -> Self {
        Self {
            inner: self.inner.without_tags(),
        }
    }

    /// Returns the model with repeat blocks flattened and shifts propagated.
    ///
    /// # Examples
    ///
    /// ```
    /// let model: stim::DetectorErrorModel =
    ///     "error(0.125) D0\nrepeat 3 {\n    error(0.25) D0 D1\n    shift_detectors 1\n}\nerror(0.125) D0 L0"
    ///         .parse()
    ///         .unwrap();
    /// assert_eq!(
    ///     model.flattened().to_string(),
    ///     "error(0.125) D0\nerror(0.25) D0 D1\nerror(0.25) D1 D2\nerror(0.25) D2 D3\nerror(0.125) D3 L0"
    /// );
    /// ```
    #[must_use]
    pub fn flattened(&self) -> Self {
        Self {
            inner: self.inner.flattened(),
        }
    }

    /// Returns the model with probabilities rounded to a given number of digits.
    #[must_use]
    pub fn rounded(&self, digits: u8) -> Self {
        Self {
            inner: self.inner.rounded(digits),
        }
    }

    /// Compiles a sampler for the detector error model using the default seed.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "error(0) D0\nerror(1) D1 D2 L0".parse().unwrap();
    /// let mut sampler = dem.compile_sampler();
    /// let (detectors, observables, errors) = sampler.sample(2);
    /// assert_eq!(detectors, ndarray::array![[false, true, true], [false, true, true]]);
    /// assert_eq!(observables, ndarray::array![[true], [true]]);
    /// assert_eq!(errors, ndarray::array![[false, true], [false, true]]);
    /// ```
    #[must_use]
    pub fn compile_sampler(&self) -> DemSampler {
        DemSampler {
            inner: self.inner.compile_sampler(),
        }
    }

    /// Compiles a sampler for the detector error model using an explicit seed.
    #[must_use]
    pub fn compile_sampler_with_seed(&self, seed: u64) -> DemSampler {
        DemSampler {
            inner: self.inner.compile_sampler_with_seed(seed),
        }
    }

    /// Returns a shortest graphlike logical error of the detector error model.
    ///
    /// # Examples
    ///
    /// ```
    /// let model: stim::DetectorErrorModel =
    ///     "error(0.125) D0\nerror(0.125) D0 D1\nerror(0.125) D1 L55\nerror(0.125) D1"
    ///         .parse()
    ///         .unwrap();
    /// assert_eq!(
    ///     model.shortest_graphlike_error(true).unwrap(),
    ///     "error(1) D1\nerror(1) D1 L55".parse::<stim::DetectorErrorModel>().unwrap()
    /// );
    /// ```
    pub fn shortest_graphlike_error(&self, ignore_ungraphlike_errors: bool) -> Result<Self> {
        self.inner
            .shortest_graphlike_error(ignore_ungraphlike_errors)
            .map(|inner| Self { inner })
            .map_err(StimError::from)
    }

    /// Returns a SAT problem encoding the shortest error search.
    pub fn shortest_error_sat_problem(&self) -> Result<String> {
        self.shortest_error_sat_problem_with_format("WDIMACS")
    }

    /// Returns a SAT problem encoding in the requested format.
    pub fn shortest_error_sat_problem_with_format(&self, format_name: &str) -> Result<String> {
        self.inner
            .shortest_error_sat_problem(format_name)
            .map_err(StimError::from)
    }

    /// Returns a SAT problem encoding the likeliest logical error search.
    pub fn likeliest_error_sat_problem(&self) -> Result<String> {
        self.likeliest_error_sat_problem_with_options(100, "WDIMACS")
    }

    /// Returns a likeliest-error SAT problem encoding with explicit options.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "error(0.125) D0\nerror(0.25) D0 L0".parse().unwrap();
    /// let sat = dem.likeliest_error_sat_problem_with_options(100, "WDIMACS").unwrap();
    /// assert!(sat.contains("p wcnf") || sat.contains("p cnf"));
    /// ```
    pub fn likeliest_error_sat_problem_with_options(
        &self,
        quantization: i32,
        format_name: &str,
    ) -> Result<String> {
        self.inner
            .likeliest_error_sat_problem(quantization, format_name)
            .map_err(StimError::from)
    }

    /// Returns detector coordinates, optionally filtered to specific ids.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "\
    /// error(0.25) D0 D1
    /// detector(1, 2, 3) D1
    /// shift_detectors(5) 1
    /// detector(1, 2) D2"
    ///     .parse()
    ///     .unwrap();
    ///
    /// assert_eq!(
    ///     dem.get_detector_coordinates(None).unwrap(),
    ///     std::collections::BTreeMap::from([
    ///         (0, vec![]),
    ///         (1, vec![1.0, 2.0, 3.0]),
    ///         (2, vec![]),
    ///         (3, vec![6.0, 2.0]),
    ///     ])
    /// );
    /// assert_eq!(
    ///     dem.get_detector_coordinates(Some(&[1])).unwrap(),
    ///     std::collections::BTreeMap::from([(1, vec![1.0, 2.0, 3.0])])
    /// );
    /// ```
    pub fn get_detector_coordinates(
        &self,
        only: Option<&[u64]>,
    ) -> Result<BTreeMap<u64, Vec<f64>>> {
        let included = only
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| (0..self.num_detectors()).collect());
        let serialized = self
            .inner
            .get_detector_coordinates_text(&included)
            .map_err(StimError::from)?;
        parse_detector_coordinate_map(&serialized)
    }

    /// Returns a visualization of the detector error model.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit = stim::Circuit::generated("repetition_code:memory", 3, 2).unwrap();
    /// let dem = circuit
    ///     .detector_error_model_with_options(true, false, false, 0.0, false, false)
    ///     .unwrap();
    /// let svg = dem.diagram("matchgraph-svg").unwrap();
    /// assert!(svg.contains("<svg"));
    /// ```
    pub fn diagram(&self, type_name: &str) -> Result<String> {
        self.inner.diagram(type_name).map_err(StimError::from)
    }

    /// Appends a detector error model instruction built from parts.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut dem = stim::DetectorErrorModel::new();
    /// dem.append("error", [0.125], [stim::target_relative_detector_id(1).unwrap()], "")
    ///     .unwrap();
    /// dem.append(
    ///     "shift_detectors",
    ///     [1.0, 2.0, 3.0],
    ///     [5u64],
    ///     "",
    /// )
    /// .unwrap();
    /// assert_eq!(dem.to_string(), "error(0.125) D1\nshift_detectors(1, 2, 3) 5");
    /// ```
    pub fn append(
        &mut self,
        instruction_type: &str,
        parens_arguments: impl IntoIterator<Item = f64>,
        targets: impl IntoIterator<Item = impl Into<DemInstructionTarget>>,
        tag: &str,
    ) -> Result<()> {
        let instruction = DemInstruction::new(instruction_type, parens_arguments, targets, tag)?;
        self.append_dem_instruction(&instruction)
    }

    /// Appends an existing instruction item.
    pub fn append_dem_instruction(&mut self, instruction: &DemInstruction) -> Result<()> {
        self.append_text_item(&instruction.to_string())
    }

    /// Appends an existing repeat block item.
    pub fn append_dem_repeat_block(&mut self, repeat_block: &DemRepeatBlock) -> Result<()> {
        self.append_text_item(&repeat_block.to_string())
    }

    /// Appends another detector error model.
    pub fn append_detector_error_model(&mut self, model: &DetectorErrorModel) -> Result<()> {
        if model.is_empty() {
            return Ok(());
        }
        self.append_text_item(&model.to_string())
    }

    /// Appends a detector-error-model operation of any supported owned type.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut model = stim::DetectorErrorModel::new();
    /// let instruction: stim::DemInstruction = "error(0.125) D1".parse().unwrap();
    /// model.append_operation(stim::DemAppendOperation::Instruction(instruction)).unwrap();
    /// assert_eq!(model.to_string(), "error(0.125) D1");
    /// ```
    pub fn append_operation(&mut self, operation: impl Into<DemAppendOperation>) -> Result<()> {
        match operation.into() {
            DemAppendOperation::Instruction(instruction) => {
                self.append_dem_instruction(&instruction)
            }
            DemAppendOperation::RepeatBlock(repeat_block) => {
                self.append_dem_repeat_block(&repeat_block)
            }
            DemAppendOperation::DetectorErrorModel(model) => {
                self.append_detector_error_model(&model)
            }
        }
    }

    /// Returns a top-level item by index.
    pub fn get(&self, index: isize) -> Result<DemItem> {
        let items = self.top_level_item_texts()?;
        let normalized = normalize_index(index, items.len())
            .ok_or_else(|| StimError::new(format!("index {index} out of range")))?;
        parse_dem_item(&items[normalized])
    }

    /// Returns a sliced detector error model over top-level items.
    pub fn slice(&self, start: Option<isize>, stop: Option<isize>, step: isize) -> Result<Self> {
        if step == 0 {
            return Err(StimError::new("slice step cannot be zero"));
        }
        let items = self.top_level_item_texts()?;
        let len = items.len() as isize;
        let indices = compute_slice_indices(len, start, stop, step);
        if indices.is_empty() {
            return Ok(Self::new());
        }
        let text = indices
            .into_iter()
            .map(|index| items[index as usize].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        Self::from_str(&text)
    }

    fn top_level_item_texts(&self) -> Result<Vec<String>> {
        split_top_level_dem_items(&self.to_string())
    }

    fn append_text_item(&mut self, text: &str) -> Result<()> {
        let combined = if self.is_empty() {
            text.to_string()
        } else {
            format!("{self}\n{text}")
        };
        *self = Self::from_str(&combined)?;
        Ok(())
    }
}

impl Clone for DetectorErrorModel {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Default for DetectorErrorModel {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for DetectorErrorModel {
    fn eq(&self, other: &Self) -> bool {
        self.inner.equals(&other.inner)
    }
}

impl Eq for DetectorErrorModel {}

impl Add for DetectorErrorModel {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner.add(&rhs.inner),
        }
    }
}

impl AddAssign for DetectorErrorModel {
    fn add_assign(&mut self, rhs: Self) {
        self.inner.add_assign(&rhs.inner);
    }
}

impl Mul<u64> for DetectorErrorModel {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self::Output {
        Self {
            inner: self.inner.repeat(rhs),
        }
    }
}

impl Mul<DetectorErrorModel> for u64 {
    type Output = DetectorErrorModel;

    fn mul(self, rhs: DetectorErrorModel) -> Self::Output {
        rhs * self
    }
}

impl MulAssign<u64> for DetectorErrorModel {
    fn mul_assign(&mut self, rhs: u64) {
        self.inner.repeat_assign(rhs);
    }
}

impl Display for DetectorErrorModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.to_dem_text())
    }
}

impl fmt::Debug for DetectorErrorModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self == &Self::new() {
            return f.write_str("stim::DetectorErrorModel()");
        }

        write!(f, "stim::DetectorErrorModel(\"\"\"\n{}\n\"\"\")", self)
    }
}

impl FromStr for DetectorErrorModel {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self> {
        stim_cxx::DetectorErrorModel::from_dem_text(s)
            .map(|inner| Self { inner })
            .map_err(StimError::from)
    }
}

fn split_top_level_dem_items(text: &str) -> Result<Vec<String>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let mut items = Vec::new();
    let mut current = Vec::new();
    let mut depth = 0isize;

    for line in text.lines() {
        if depth == 0 && current.is_empty() && !line.starts_with("repeat ") {
            items.push(line.to_string());
            continue;
        }

        depth += line.matches('{').count() as isize;
        current.push(line.to_string());
        depth -= line.matches('}').count() as isize;

        if depth < 0 {
            return Err(StimError::new("unbalanced DEM repeat block braces"));
        }
        if depth == 0 {
            items.push(current.join("\n"));
            current.clear();
        }
    }

    if depth != 0 || !current.is_empty() {
        return Err(StimError::new("unterminated DEM repeat block"));
    }

    Ok(items)
}

fn parse_dem_item(text: &str) -> Result<DemItem> {
    if let Some((header, body)) = text.split_once("{\n") {
        let repeat_count = header
            .trim()
            .strip_prefix("repeat ")
            .ok_or_else(|| StimError::new(format!("invalid DEM repeat block header: {header}")))?;
        let repeat_count = repeat_count
            .parse::<u64>()
            .map_err(|_| StimError::new(format!("invalid repeat count: {repeat_count}")))?;
        let inner = body
            .strip_suffix("\n}")
            .ok_or_else(|| StimError::new("invalid DEM repeat block body"))?;
        let inner = inner
            .lines()
            .map(|line| line.strip_prefix("    ").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(DemItem::repeat_block(DemRepeatBlock::new(
            repeat_count,
            &DetectorErrorModel::from_str(&inner)?,
        )?))
    } else {
        Ok(DemItem::instruction(DemInstruction::from_str(text)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::SystemTime;

    use crate::{
        Circuit, DemInstructionTarget, DemRepeatBlock, target_logical_observable_id,
        target_relative_detector_id, target_separator,
    };

    fn unique_temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        std::env::temp_dir().join(format!("stim-rs-{name}-{nanos}.01"))
    }

    fn unpack_rows(packed: &[u8], bits_per_row: usize) -> Vec<Vec<bool>> {
        packed
            .iter()
            .map(|byte| {
                (0..bits_per_row)
                    .map(|bit| ((byte >> bit) & 1) == 1)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn bool_matrix(rows: Vec<Vec<bool>>) -> Array2<bool> {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, Vec::len);
        Array2::from_shape_vec((nrows, ncols), rows.into_iter().flatten().collect())
            .expect("rows should be rectangular")
    }
    #[test]
    fn detector_error_model_compile_sampler_matches_documented_basic_contract() {
        let dem: DetectorErrorModel = "error(0) D0\nerror(1) D1 D2 L0".parse().unwrap();

        let mut sampler = dem.compile_sampler_with_seed(7);
        assert_eq!(sampler.num_detectors(), 3);
        assert_eq!(sampler.num_observables(), 1);
        assert_eq!(sampler.num_errors(), 2);

        let (packed_detectors, packed_observables, packed_errors) = sampler.sample_bit_packed(4);
        assert_eq!(
            unpack_rows(&packed_detectors, 3),
            vec![vec![false, true, true]; 4]
        );
        assert_eq!(unpack_rows(&packed_observables, 1), vec![vec![true]; 4]);
        assert_eq!(unpack_rows(&packed_errors, 2), vec![vec![false, true]; 4]);

        let mut sampler = dem.compile_sampler_with_seed(7);
        let (detectors, observables, errors) = sampler.sample(4);
        assert_eq!(detectors, bool_matrix(vec![vec![false, true, true]; 4]));
        assert_eq!(observables, bool_matrix(vec![vec![true]; 4]));
        assert_eq!(errors, bool_matrix(vec![vec![false, true]; 4]));

        let noisy_dem: DetectorErrorModel = "error(0.125) D0\nerror(0.25) D1".parse().unwrap();
        let mut noisy_sampler = noisy_dem.compile_sampler_with_seed(9);
        let (det_data, obs_data, err_data) = noisy_sampler.sample_bit_packed(100);
        let mut replay_sampler = noisy_dem.compile_sampler_with_seed(999);
        let (replay_det_data, replay_obs_data, replay_err_data) =
            replay_sampler.sample_bit_packed_replay(&err_data, 100);
        assert_eq!(det_data, replay_det_data);
        assert_eq!(obs_data, replay_obs_data);
        assert_eq!(err_data, replay_err_data);

        let dets_path = unique_temp_path("dem-dets");
        let obs_path = unique_temp_path("dem-obs");
        let err_path = unique_temp_path("dem-err");

        let mut sampler = dem.compile_sampler_with_seed(7);
        sampler
            .sample_write_with_errors(1, &dets_path, "01", &obs_path, "01", &err_path, "hits")
            .unwrap();
        assert_eq!(fs::read_to_string(&dets_path).unwrap(), "011\n");
        assert_eq!(fs::read_to_string(&obs_path).unwrap(), "1\n");
        assert_eq!(fs::read_to_string(&err_path).unwrap(), "1\n");

        let dets_path = unique_temp_path("dem-dets-noerr");
        let obs_path = unique_temp_path("dem-obs-noerr");
        let mut sampler = dem.compile_sampler_with_seed(7);
        sampler
            .sample_write(1, &dets_path, "01", &obs_path, "01")
            .unwrap();
        assert_eq!(fs::read_to_string(&dets_path).unwrap(), "011\n");
        assert_eq!(fs::read_to_string(&obs_path).unwrap(), "1\n");

        let replay_err_path = unique_temp_path("dem-replay-err");
        let replay_det_path = unique_temp_path("dem-replay-dets");
        let replay_obs_path = unique_temp_path("dem-replay-obs");
        let replay_err_out_path = unique_temp_path("dem-replay-err-out");

        let mut writer = noisy_dem.compile_sampler_with_seed(9);
        writer
            .sample_write_with_errors(
                5,
                &replay_det_path,
                "01",
                &replay_obs_path,
                "01",
                &replay_err_path,
                "b8",
            )
            .unwrap();
        let expected_dets = fs::read_to_string(&replay_det_path).unwrap();
        let expected_obs = fs::read_to_string(&replay_obs_path).unwrap();

        let replay_det_path = unique_temp_path("dem-replay-dets-2");
        let replay_obs_path = unique_temp_path("dem-replay-obs-2");
        let mut replay_writer = noisy_dem.compile_sampler_with_seed(123456);
        replay_writer
            .sample_write_replay_with_errors(
                5,
                &replay_det_path,
                "01",
                &replay_obs_path,
                "01",
                &replay_err_out_path,
                "b8",
                &replay_err_path,
                "b8",
            )
            .unwrap();
        assert_eq!(fs::read_to_string(&replay_det_path).unwrap(), expected_dets);
        assert_eq!(fs::read_to_string(&replay_obs_path).unwrap(), expected_obs);
        assert_eq!(
            fs::read(&replay_err_out_path).unwrap(),
            fs::read(&replay_err_path).unwrap()
        );
    }

    #[test]
    fn detector_error_model_sat_problem_helpers_match_upstream_examples() {
        let dem: DetectorErrorModel = "error(0.1) L0".parse().unwrap();
        assert_eq!(
            dem.shortest_error_sat_problem().unwrap(),
            "p wcnf 1 2 3\n1 -1 0\n3 1 0\n"
        );

        let dem: DetectorErrorModel = "error(0.1) D0 L0\nerror(0.1) D0".parse().unwrap();
        assert_eq!(
            dem.likeliest_error_sat_problem_with_options(100, "WDIMACS")
                .unwrap(),
            "p wcnf 3 8 801\n100 -1 0\n801 1 2 -3 0\n801 1 -2 3 0\n801 -1 2 3 0\n801 -1 -2 -3 0\n100 -2 0\n801 -3 0\n801 1 0\n"
        );
    }

    #[test]
    fn detector_error_model_get_and_slice_return_independent_copies() {
        let model: DetectorErrorModel = "\
    error(0.125) D0
    error(0.125) D1 L1
    repeat 100 {
        error(0.125) D1 D2
        shift_detectors 1
    }
    error(0.125) D2
    logical_observable L0
    detector D5"
            .parse()
            .unwrap();

        assert_eq!(
            model.get(0).unwrap(),
            DemItem::instruction(
                DemInstruction::new(
                    "error",
                    [0.125],
                    [target_relative_detector_id(0).unwrap()],
                    ""
                )
                .unwrap()
            )
        );
        assert_eq!(
            model.get(2).unwrap(),
            DemItem::repeat_block(
                DemRepeatBlock::new(
                    100,
                    &"error(0.125) D1 D2\nshift_detectors 1"
                        .parse::<DetectorErrorModel>()
                        .unwrap(),
                )
                .unwrap()
            )
        );
        assert_eq!(
            model.get(-1).unwrap(),
            DemItem::instruction(
                DemInstruction::new(
                    "detector",
                    [],
                    [target_relative_detector_id(5).unwrap()],
                    ""
                )
                .unwrap()
            )
        );

        assert_eq!(
            model.slice(Some(1), None, 2).unwrap(),
            "\
    error(0.125) D1 L1
    error(0.125) D2
    detector D5"
                .parse::<DetectorErrorModel>()
                .unwrap()
        );
        assert_eq!(
            model.slice(Some(-3), None, 1).unwrap(),
            "\
    error(0.125) D2
    logical_observable L0
    detector D5"
                .parse::<DetectorErrorModel>()
                .unwrap()
        );

        let mut sliced = model.slice(None, None, 1).unwrap();
        sliced.clear();
        assert_eq!(sliced, DetectorErrorModel::new());
        assert_eq!(model.len(), 6);
    }

    #[test]
    fn detector_error_model_shortest_graphlike_error_matches_documented_examples() {
        assert_eq!(
            "error(0.125) D0\nerror(0.125) D0 D1\nerror(0.125) D1 L55\nerror(0.125) D1"
                .parse::<DetectorErrorModel>()
                .unwrap()
                .shortest_graphlike_error(true)
                .unwrap(),
            "error(1) D1\nerror(1) D1 L55"
                .parse::<DetectorErrorModel>()
                .unwrap()
        );

        assert_eq!(
            "error(0.125) D0 D1 D2\nerror(0.125) L0"
                .parse::<DetectorErrorModel>()
                .unwrap()
                .shortest_graphlike_error(true)
                .unwrap(),
            "error(1) L0".parse::<DetectorErrorModel>().unwrap()
        );

        let circuit =
            Circuit::generated_with_noise("repetition_code:memory", 7, 10, 0.0, 0.01, 0.0, 0.0)
                .unwrap();
        let model = circuit
            .detector_error_model_with_options(true, false, false, 0.0, false, false)
            .unwrap();
        assert_eq!(model.shortest_graphlike_error(true).unwrap().len(), 7);
    }

    #[test]
    fn detector_error_model_append_builds_instructions_from_parts() {
        let mut model = DetectorErrorModel::new();
        model
            .append(
                "error",
                [0.125],
                [target_relative_detector_id(1).unwrap()],
                "",
            )
            .unwrap();
        model
            .append(
                "error",
                [0.25],
                [
                    target_relative_detector_id(1).unwrap(),
                    target_separator(),
                    target_relative_detector_id(2).unwrap(),
                    target_logical_observable_id(3).unwrap(),
                ],
                "test-tag",
            )
            .unwrap();
        model
            .append("shift_detectors", [1.0, 2.0, 3.0], [5u64], "")
            .unwrap();

        assert_eq!(
            model,
            DetectorErrorModel::from_str(
                "\
    error(0.125) D1
    error[test-tag](0.25) D1 ^ D2 L3
    shift_detectors(1,2,3) 5"
            )
            .unwrap()
        );
    }

    #[test]
    fn detector_error_model_append_accepts_existing_instruction_repeat_block_and_model() {
        let mut model = DetectorErrorModel::new();
        model
            .append(
                "error",
                [0.125],
                [target_relative_detector_id(1).unwrap()],
                "",
            )
            .unwrap();
        model
            .append("shift_detectors", [1.0, 2.0, 3.0], [5u64], "")
            .unwrap();

        let repeated = model.clone() * 3;
        let first = DemInstruction::new(
            "error",
            [0.125],
            [target_relative_detector_id(1).unwrap()],
            "",
        )
        .unwrap();
        let repeat_block = DemRepeatBlock::new(3, &model).unwrap();

        model.append_dem_repeat_block(&repeat_block).unwrap();
        model.append_dem_instruction(&first).unwrap();
        model.append_detector_error_model(&repeated).unwrap();

        assert_eq!(
            model,
            DetectorErrorModel::from_str(
                "\
    error(0.125) D1
    shift_detectors(1,2,3) 5
    repeat 3 {
        error(0.125) D1
        shift_detectors(1,2,3) 5
    }
    error(0.125) D1
    repeat 3 {
        error(0.125) D1
        shift_detectors(1,2,3) 5
    }"
            )
            .unwrap()
        );
    }

    #[test]
    fn dem_instruction_append_path_supports_raw_numeric_targets() {
        let instruction = DemInstruction::new("shift_detectors", [1.0], [5u64], "").unwrap();
        assert_eq!(
            instruction.targets_copy(),
            vec![DemInstructionTarget::from(5u64)]
        );
    }

    #[test]
    fn detector_error_model_append_empty_model_is_noop_and_invalid_instruction_fails() {
        let mut model = DetectorErrorModel::from_str("error(0.125) D1").unwrap();
        model
            .append_detector_error_model(&DetectorErrorModel::new())
            .unwrap();
        assert_eq!(
            model,
            DetectorErrorModel::from_str("error(0.125) D1").unwrap()
        );

        assert!(
            model
                .append("", [], Vec::<DemInstructionTarget>::new(), "")
                .is_err()
        );
    }

    #[test]
    fn detector_error_model_append_operation_accepts_instruction_values() {
        let mut model = DetectorErrorModel::new();

        model
            .append_operation(DemInstruction::from_str("error(0.125) D1").unwrap())
            .unwrap();
        model
            .append_operation(DemInstruction::from_str("shift_detectors(1,2,3) 5").unwrap())
            .unwrap();

        assert_eq!(
            model,
            DetectorErrorModel::from_str("error(0.125) D1\nshift_detectors(1,2,3) 5").unwrap()
        );
    }

    #[test]
    fn detector_error_model_append_operation_accepts_repeat_blocks() {
        let mut model = DetectorErrorModel::from_str("error(0.125) D1").unwrap();
        let repeat = DemRepeatBlock::new(
            2,
            &DetectorErrorModel::from_str("shift_detectors 1").unwrap(),
        )
        .unwrap();

        model.append_operation(repeat).unwrap();

        assert_eq!(
            model,
            DetectorErrorModel::from_str("error(0.125) D1\nrepeat 2 {\n    shift_detectors 1\n}")
                .unwrap()
        );
    }

    #[test]
    fn detector_error_model_append_operation_accepts_models_and_ignores_empty_input() {
        let mut model = DetectorErrorModel::from_str("error(0.125) D1").unwrap();

        model
            .append_operation(
                DetectorErrorModel::from_str("shift_detectors 1\nlogical_observable L0").unwrap(),
            )
            .unwrap();
        model.append_operation(DetectorErrorModel::new()).unwrap();

        assert_eq!(
            model,
            DetectorErrorModel::from_str(
                "error(0.125) D1\nshift_detectors 1\nlogical_observable L0"
            )
            .unwrap()
        );
    }

    #[test]
    fn detector_error_model_append_operation_copies_inputs_instead_of_aliasing() {
        let mut model = DetectorErrorModel::new();
        let instruction = DemInstruction::from_str("error(0.125) D1").unwrap();
        let appended = DetectorErrorModel::from_str("shift_detectors 1").unwrap();

        model.append_operation(&instruction).unwrap();
        model.append_operation(&appended).unwrap();

        assert_eq!(
            instruction,
            DemInstruction::from_str("error(0.125) D1").unwrap()
        );
        assert_eq!(
            appended,
            DetectorErrorModel::from_str("shift_detectors 1").unwrap()
        );
        assert_eq!(
            model,
            DetectorErrorModel::from_str("error(0.125) D1\nshift_detectors 1").unwrap()
        );
    }

    #[test]
    fn detector_error_model_get_detector_coordinates_matches_documented_examples() {
        let dem = DetectorErrorModel::from_str(
            "\
    error(0.25) D0 D1
    detector(1, 2, 3) D1
    shift_detectors(5) 1
    detector(1, 2) D2",
        )
        .unwrap();

        assert_eq!(
            dem.get_detector_coordinates(None).unwrap(),
            BTreeMap::from([
                (0, vec![]),
                (1, vec![1.0, 2.0, 3.0]),
                (2, vec![]),
                (3, vec![6.0, 2.0]),
            ])
        );
        assert_eq!(
            dem.get_detector_coordinates(Some(&[1])).unwrap(),
            BTreeMap::from([(1, vec![1.0, 2.0, 3.0])])
        );
    }

    #[test]
    fn detector_error_model_get_detector_coordinates_supports_filters_and_errors() {
        let dem = DetectorErrorModel::from_str(
            "\
    detector(1, 2) D0
    shift_detectors 1
    detector(4) D0",
        )
        .unwrap();

        assert_eq!(
            dem.get_detector_coordinates(Some(&[0, 1])).unwrap(),
            BTreeMap::from([(0, vec![1.0, 2.0]), (1, vec![4.0])])
        );
        assert!(dem.get_detector_coordinates(Some(&[2])).is_err());
    }

    #[test]
    fn detector_error_model_supports_core_value_type_behavior() {
        let empty = DetectorErrorModel::new();
        let also_empty = DetectorErrorModel::default();
        let model = DetectorErrorModel::from_str(
            "error(0.125) D0 L1\nshift_detectors 4\ndetector(1, 2) D2\nlogical_observable L5",
        )
        .expect("detector error model should parse");

        assert_eq!(empty, also_empty);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.num_detectors(), 0);
        assert_eq!(empty.num_errors(), 0);
        assert_eq!(empty.num_observables(), 0);
        assert_eq!(empty.to_string(), "");
        assert_eq!(format!("{empty:?}"), "stim::DetectorErrorModel()");

        assert!(!model.is_empty());
        assert_eq!(model.len(), 4);
        assert_eq!(model.num_detectors(), 7);
        assert_eq!(model.num_errors(), 1);
        assert_eq!(model.num_observables(), 6);
        assert_eq!(
            model.to_string(),
            "error(0.125) D0 L1\nshift_detectors 4\ndetector(1, 2) D2\nlogical_observable L5"
        );
        assert_eq!(
            format!("{model:?}"),
            "stim::DetectorErrorModel(\"\"\"\nerror(0.125) D0 L1\nshift_detectors 4\ndetector(1, 2) D2\nlogical_observable L5\n\"\"\")"
        );
    }

    #[test]
    fn detector_error_model_copy_clone_clear_and_roundtrip_parsing_are_independent() {
        let model = DetectorErrorModel::from_str("error(0.25) D2 D3\nlogical_observable L0")
            .expect("detector error model should parse");

        let copy = model.copy();
        let clone = model.clone();
        let roundtrip = DetectorErrorModel::from_str(&model.to_string())
            .expect("display output should parse back into an equivalent model");

        assert_eq!(copy, model);
        assert_ne!(
            (&copy as *const DetectorErrorModel),
            (&model as *const DetectorErrorModel)
        );
        assert_eq!(clone, model);
        assert_eq!(roundtrip, model);

        let mut cleared = clone;
        cleared.clear();

        assert_eq!(cleared, DetectorErrorModel::new());
        assert_eq!(copy, model);
        assert_eq!(roundtrip, model);
    }

    #[test]
    fn detector_error_model_diagram_supports_documented_matchgraph_variants() {
        let circuit =
            Circuit::generated("repetition_code:memory", 7, 10).expect("generated circuit");
        let dem = circuit
            .detector_error_model_with_options(true, false, false, 0.0, false, false)
            .expect("detector error model");

        let svg = dem.diagram("matchgraph-svg").expect("svg");
        let svg_alias = dem.diagram("match-graph-svg").expect("svg alias");
        let svg_html = dem.diagram("match-graph-svg-html").expect("svg html");
        let gltf = dem.diagram("matchgraph-3d").expect("gltf");
        let gltf_html = dem.diagram("matchgraph-3d-html").expect("gltf html");

        assert!(svg.contains("<svg"));
        assert_eq!(svg, svg_alias);
        assert!(svg_html.contains("iframe"));
        assert!(gltf.contains("\"nodes\"") || gltf.contains("\"scenes\""));
        assert!(gltf_html.contains("<html") || gltf_html.contains("iframe"));
    }

    #[test]
    fn detector_error_model_diagram_rejects_unknown_types() {
        let circuit =
            Circuit::generated("repetition_code:memory", 3, 2).expect("generated circuit");
        let dem = circuit
            .detector_error_model_with_options(true, false, false, 0.0, false, false)
            .expect("detector error model");

        assert!(dem.diagram("not-a-diagram").is_err());
    }

    #[test]
    fn detector_error_model_from_file_reads_dem_text() {
        let path = unique_temp_path("detector-error-model-from-file");
        fs::write(&path, "error(0.25) D2 D3\nshift_detectors 4\n").expect("temp file should write");

        let model = DetectorErrorModel::from_file(&path).expect("detector error model should read");

        assert_eq!(model.to_string(), "error(0.25) D2 D3\nshift_detectors 4");

        fs::remove_file(path).expect("temp file should delete");
    }

    #[test]
    fn detector_error_model_to_file_writes_trailing_newline() {
        let path = unique_temp_path("detector-error-model-to-file");
        let model = DetectorErrorModel::from_str("error(0.25) D2 D3\nlogical_observable L1")
            .expect("detector error model should parse");

        model
            .to_file(&path)
            .expect("detector error model should write");

        assert_eq!(
            fs::read_to_string(&path).expect("temp file should read"),
            "error(0.25) D2 D3\nlogical_observable L1\n"
        );

        fs::remove_file(path).expect("temp file should delete");
    }

    #[test]
    fn detector_error_model_add_and_add_assign_append_models() {
        let left = DetectorErrorModel::from_str("error(0.125) D0\nlogical_observable L0")
            .expect("left model should parse");
        let right = DetectorErrorModel::from_str("error(0.25) D1\nshift_detectors 2")
            .expect("right model should parse");

        let combined = left.clone() + right.clone();

        assert_eq!(
            combined.to_string(),
            "error(0.125) D0\nlogical_observable L0\nerror(0.25) D1\nshift_detectors 2"
        );
        assert_eq!(left.to_string(), "error(0.125) D0\nlogical_observable L0");
        assert_eq!(right.to_string(), "error(0.25) D1\nshift_detectors 2");

        let mut appended = left.clone();
        appended += right.clone();
        assert_eq!(appended, combined);

        let mut doubled = right.clone();
        doubled += right.clone();
        assert_eq!(
            doubled.to_string(),
            "error(0.25) D1\nshift_detectors 2\nerror(0.25) D1\nshift_detectors 2"
        );
    }

    #[test]
    fn detector_error_model_mul_and_mul_assign_wrap_in_repeat_blocks() {
        let model = DetectorErrorModel::from_str("error(0.25) D0\nshift_detectors 1")
            .expect("model should parse");

        let zero = u64::default();
        let one = 1_u64;
        assert_eq!((model.clone() * zero), DetectorErrorModel::new());
        assert_eq!((model.clone() * one), model);
        assert_eq!(
            (model.clone() * 3).to_string(),
            "repeat 3 {\n    error(0.25) D0\n    shift_detectors 1\n}"
        );

        let mut repeated = model.clone();
        repeated *= 3;
        assert_eq!(
            repeated.to_string(),
            "repeat 3 {\n    error(0.25) D0\n    shift_detectors 1\n}"
        );
    }

    #[test]
    fn detector_error_model_approx_equals_uses_absolute_tolerance() {
        let base =
            DetectorErrorModel::from_str("error(0.099) D0 D1").expect("base model should parse");
        let near = DetectorErrorModel::from_str("error(0.101) D0 D1")
            .expect("comparison model should parse");
        let different_targets = DetectorErrorModel::from_str("error(0.099) D0 D1 L2")
            .expect("comparison model should parse");

        assert!(base.approx_equals(&base, 0.0));
        assert!(!base.approx_equals(&near, 0.0001));
        assert!(base.approx_equals(&near, 0.01));
        assert!(!base.approx_equals(&different_targets, 9999.0));
    }

    #[test]
    fn detector_error_model_without_tags_removes_all_tags() {
        let model = DetectorErrorModel::from_str(
            "error[test-tag](0.25) D0\nrepeat[loop-tag] 2 {\n    error[nested-tag](0.125) D1\n}",
        )
        .expect("detector error model should parse");

        let untagged = model.without_tags();

        assert_eq!(
            untagged.to_string(),
            "error(0.25) D0\nrepeat 2 {\n    error(0.125) D1\n}"
        );
        assert_eq!(
            model.to_string(),
            "error[test-tag](0.25) D0\nrepeat[loop-tag] 2 {\n    error[nested-tag](0.125) D1\n}"
        );
    }

    #[test]
    fn detector_error_model_flattened_inlines_repeat_blocks_and_detector_shifts() {
        let model = DetectorErrorModel::from_str(
                "error(0.125) D0\nrepeat 3 {\n    error(0.25) D0 D1\n    shift_detectors 1\n}\nerror(0.125) D0 L0",
            )
            .expect("detector error model should parse");

        let flattened = model.flattened();

        assert_eq!(
            flattened.to_string(),
            "error(0.125) D0\nerror(0.25) D0 D1\nerror(0.25) D1 D2\nerror(0.25) D2 D3\nerror(0.125) D3 L0"
        );
    }

    #[test]
    fn detector_error_model_rounded_only_rounds_error_probabilities() {
        let model = DetectorErrorModel::from_str(
                "error(0.01000002) D0 D1\nrepeat 2 {\n    error(0.123456789) D1 D2 L3\n}\ndetector(0.0200000334,0.12345) D0\nshift_detectors(5.0300004,0.12345) 3",
            )
            .expect("detector error model should parse");

        assert_eq!(
                model.rounded(2),
                DetectorErrorModel::from_str(
                    "error(0.01) D0 D1\nrepeat 2 {\n    error(0.12) D1 D2 L3\n}\ndetector(0.0200000334,0.12345) D0\nshift_detectors(5.0300004,0.12345) 3"
                )
                .expect("expected rounded model should parse")
            );
        assert_eq!(
                model.rounded(3),
                DetectorErrorModel::from_str(
                    "error(0.010) D0 D1\nrepeat 2 {\n    error(0.123) D1 D2 L3\n}\ndetector(0.0200000334,0.12345) D0\nshift_detectors(5.0300004,0.12345) 3"
                )
                .expect("expected rounded model should parse")
            );
    }
}

#[cfg(test)]
mod debug_repr_tests {
    use super::DetectorErrorModel;

    #[test]
    fn debug_representation_marks_non_empty_detector_error_models() {
        let model: DetectorErrorModel = "error(0.125) D0".parse().expect("model should parse");

        assert_eq!(
            format!("{model:?}"),
            "stim::DetectorErrorModel(\"\"\"\nerror(0.125) D0\n\"\"\")"
        );
    }
}
