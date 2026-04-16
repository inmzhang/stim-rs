mod dem_append_operation;
mod dem_instruction;
mod dem_item;
mod dem_repeat_block;
mod dem_target;

use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::ops::{Add, AddAssign, Index, Mul, MulAssign};
use std::path::Path;
use std::str::FromStr;

use crate::common::parse::parse_detector_coordinate_map;
use crate::common::slicing::{compute_slice_indices, normalize_index};
use crate::{DemSampler, Result, StimError};

pub use dem_append_operation::DemAppendOperation;
pub use dem_instruction::{DemInstruction, DemInstructionTarget};
pub use dem_item::DemItem;
pub use dem_repeat_block::DemRepeatBlock;
pub use dem_target::{DemTarget, DemTargetWithCoords};

/// An error model built out of independent error mechanisms, describing how faults
/// trigger detectors and logical observables.
///
/// `DetectorErrorModel` is one of the most important types in Stim, because it is
/// the mechanism used to explain quantum error correction circuits to decoders. A
/// typical quantum error correction workflow looks like:
///
/// 1. Create a quantum error correction [`Circuit`](crate::Circuit) annotated with
///    detectors and observables.
/// 2. Call [`Circuit::detector_error_model`](crate::Circuit::detector_error_model)
///    (with `decompose_errors=true` if working with a matching-based decoder). This
///    converts the circuit's error mechanisms into a straightforward list of
///    independent "with probability *p*, these detectors and observables get flipped"
///    terms.
/// 3. Feed the detector error model into your decoder of choice (e.g., pymatching,
///    fusion_blossom, or another graph-based decoder).
/// 4. Sample detection events with
///    [`Circuit::compile_detector_sampler`](crate::Circuit::compile_detector_sampler),
///    feed them to the decoder, and compare its observable-flip predictions to the
///    actual flips.
///
/// Error mechanisms are described in terms of the visible detection events (e.g.
/// `D0`, `D1`) and the hidden observable frame changes (e.g. `L0`) they cause.
/// Error mechanisms can also suggest decompositions of their effects into
/// components, which is helpful for decoders that want to work with a simpler
/// decomposed error model instead of the full error model.
///
/// # Key operations
///
/// - **Construction**: parse from a `.dem`-format string via [`FromStr`], read from a
///   file with [`from_file`](Self::from_file), or build programmatically with
///   [`append`](Self::append).
/// - **Sampling**: compile a [`DemSampler`] with
///   [`compile_sampler`](Self::compile_sampler) to generate detection event and
///   observable-flip samples directly from the error model.
/// - **Error analysis**: find the shortest graphlike logical error with
///   [`shortest_graphlike_error`](Self::shortest_graphlike_error), or encode the
///   distance problem as a maxSAT instance with
///   [`shortest_error_sat_problem`](Self::shortest_error_sat_problem).
/// - **Arithmetic**: concatenate models with `+`, wrap in repeat blocks with `*`.
///
/// # Examples
///
/// ```
/// let model: stim::DetectorErrorModel = "
///     error(0.125) D0
///     error(0.125) D0 D1 L0
///     error(0.125) D1 D2
///     error(0.125) D2 D3
///     error(0.125) D3
/// ".parse().unwrap();
/// assert_eq!(model.len(), 5);
/// ```
pub struct DetectorErrorModel {
    pub(crate) inner: stim_cxx::DetectorErrorModel,
    item_cache: OnceCell<Vec<DemItem>>,
}

impl DetectorErrorModel {
    pub(crate) fn from_inner(inner: stim_cxx::DetectorErrorModel) -> Self {
        Self {
            inner,
            item_cache: OnceCell::new(),
        }
    }

    fn invalidate_item_cache(&mut self) {
        let _ = self.item_cache.take();
    }

    /// Creates an empty detector error model with no instructions.
    ///
    /// This is the starting point for programmatically building a detector error
    /// model. Use [`append`](Self::append) or the `+=` operator to add
    /// instructions after construction.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem = stim::DetectorErrorModel::new();
    /// assert!(dem.is_empty());
    /// assert_eq!(dem.len(), 0);
    /// assert_eq!(dem.num_detectors(), 0);
    /// assert_eq!(dem.num_errors(), 0);
    /// assert_eq!(dem.num_observables(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::from_inner(stim_cxx::DetectorErrorModel::new())
    }

    /// Returns the number of top-level instructions and blocks in the detector
    /// error model.
    ///
    /// Instructions inside of repeat blocks are **not** included in this count.
    /// A repeat block counts as a single item regardless of how many
    /// instructions it contains or how many times it repeats.
    ///
    /// # Examples
    ///
    /// ```
    /// // Each top-level instruction counts as one item.
    /// let dem: stim::DetectorErrorModel = "
    ///     error(0.1) D0 D1
    ///     shift_detectors 100
    ///     logical_observable L5
    /// ".parse().unwrap();
    /// assert_eq!(dem.len(), 3);
    ///
    /// // A repeat block containing two instructions still counts as one item.
    /// let dem: stim::DetectorErrorModel = "
    ///     repeat 100 {
    ///         error(0.1) D0 D1
    ///         error(0.1) D1 D2
    ///     }
    /// ".parse().unwrap();
    /// assert_eq!(dem.len(), 1);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the detector error model contains no instructions.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::DetectorErrorModel::new().is_empty());
    /// assert!("error(0.1) D0".parse::<stim::DetectorErrorModel>().unwrap().is_empty() == false);
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Counts the total number of detectors (e.g. `D2`) mentioned by the model.
    ///
    /// Detector indices are assumed to be contiguous from 0 up to the maximum
    /// detector id. If the largest detector's absolute id is *n* − 1, the number
    /// of detectors is *n*. The `shift_detectors` instruction is taken into
    /// account when computing absolute ids.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "error(0.1) D0 D199".parse().unwrap();
    /// assert_eq!(dem.num_detectors(), 200);
    ///
    /// // shift_detectors offsets subsequent detector ids.
    /// let dem: stim::DetectorErrorModel = "
    ///     shift_detectors 1000
    ///     error(0.1) D0 D199
    /// ".parse().unwrap();
    /// assert_eq!(dem.num_detectors(), 1200);
    /// ```
    #[must_use]
    pub fn num_detectors(&self) -> u64 {
        self.inner.num_detectors()
    }

    /// Counts the total number of error mechanisms (e.g. `error(0.1) D0`) in
    /// the model.
    ///
    /// Error instructions inside repeat blocks count once per repetition.
    /// Redundant errors with the same targets count as separate errors.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "
    ///     error(0.125) D0
    ///     repeat 100 {
    ///         repeat 5 {
    ///             error(0.25) D1
    ///         }
    ///     }
    /// ".parse().unwrap();
    /// assert_eq!(dem.num_errors(), 501);
    /// ```
    #[must_use]
    pub fn num_errors(&self) -> u64 {
        self.inner.num_errors()
    }

    /// Counts the number of logical observables (e.g. `L2`) referenced by the
    /// model.
    ///
    /// Observable indices are assumed to be contiguous from 0 up to the maximum
    /// observable id. If the largest observable's id is *n* − 1, the number of
    /// observables is *n*.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "error(0.1) L399".parse().unwrap();
    /// assert_eq!(dem.num_observables(), 400);
    /// ```
    #[must_use]
    pub fn num_observables(&self) -> u64 {
        self.inner.num_observables()
    }

    /// Clears all instructions from the detector error model, making it empty.
    ///
    /// After calling this method, [`is_empty`](Self::is_empty) returns `true`
    /// and [`len`](Self::len) returns `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut model: stim::DetectorErrorModel = "error(0.1) D0 D1".parse().unwrap();
    /// assert!(!model.is_empty());
    /// model.clear();
    /// assert!(model.is_empty());
    /// assert_eq!(model, stim::DetectorErrorModel::new());
    /// ```
    pub fn clear(&mut self) {
        self.inner.clear();
        self.invalidate_item_cache();
    }

    /// Reads a detector error model from a file on disk.
    ///
    /// The file should contain text in the detector error model (`.dem`) format,
    /// as defined at
    /// <https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md>.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read (e.g. the path does not
    /// exist or permissions are insufficient), or if the file contents are not
    /// valid `.dem` text.
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

    /// Writes the detector error model to a file on disk, with a trailing newline.
    ///
    /// The output uses the detector error model (`.dem`) format, as defined at
    /// <https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md>.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written (e.g. the parent
    /// directory does not exist or permissions are insufficient).
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

    /// Returns an independent owned copy of the detector error model.
    ///
    /// The copy has identical contents but is a separate allocation, so
    /// mutating one will not affect the other. This is equivalent to
    /// [`Clone::clone`].
    ///
    /// # Examples
    ///
    /// Checks whether two detector error models are approximately equal,
    /// using an absolute tolerance on numeric arguments such as probabilities.
    ///
    /// Two models are approximately equal if they are equal up to slight
    /// perturbations of instruction arguments (e.g. probabilities). For
    /// example, `error(0.100) D0` is approximately equal to `error(0.099) D0`
    /// within an absolute tolerance of `0.002`. All other details of the
    /// models — including the ordering of errors, their targets, and their
    /// structure — must be exactly the same.
    ///
    /// # Examples
    ///
    /// ```
    /// let base: stim::DetectorErrorModel = "error(0.099) D0 D1".parse().unwrap();
    /// let near: stim::DetectorErrorModel = "error(0.101) D0 D1".parse().unwrap();
    /// assert!(base.approx_equals(&near, 0.01));
    /// assert!(!base.approx_equals(&near, 0.0001));
    ///
    /// // Structural differences (different targets) are never approximately equal,
    /// // regardless of the tolerance.
    /// let different: stim::DetectorErrorModel = "error(0.099) D0 D1 L2".parse().unwrap();
    /// assert!(!base.approx_equals(&different, 9999.0));
    /// ```
    #[must_use]
    pub fn approx_equals(&self, other: &Self, atol: f64) -> bool {
        self.inner.approx_equals(&other.inner, atol)
    }

    /// Returns a copy of the detector error model with all tags removed from
    /// every instruction.
    ///
    /// Tags are arbitrary text annotations attached to instructions in square
    /// brackets, such as the `[tag]` in `error[tag](0.25) D0`. This method
    /// strips all such annotations while preserving every other aspect of the
    /// model, including instructions inside repeat blocks.
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
        Self::from_inner(self.inner.without_tags())
    }

    /// Returns an equivalent detector error model with repeat blocks fully
    /// unrolled and `shift_detectors` instructions inlined.
    ///
    /// The returned model contains the same errors in the same order, but with
    /// all repeat loops expanded into their individual iterations and all
    /// coordinate/index shifts folded into the detector and observable ids of
    /// each error instruction. The result contains no `repeat` or
    /// `shift_detectors` instructions.
    ///
    /// This is useful for analysis that needs to enumerate every error
    /// mechanism with its absolute detector ids, at the cost of a potentially
    /// much larger model.
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
        Self::from_inner(self.inner.flattened())
    }

    /// Creates an equivalent detector error model with error probabilities
    /// rounded to a given number of decimal digits.
    ///
    /// Only the parenthesized arguments of `error` instructions are rounded;
    /// arguments on other instructions (e.g. `detector` coordinates,
    /// `shift_detectors` offsets) are left unchanged. Error instructions whose
    /// probability rounds to zero are still included in the output.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "
    ///     error(0.019499) D0
    ///     error(0.000001) D0 D1
    /// ".parse().unwrap();
    ///
    /// // Rounding is applied to error probabilities only. Use approx_equals
    /// // to verify the rounded values are within tolerance.
    /// assert!(dem.rounded(2).approx_equals(
    ///     &"error(0.02) D0\nerror(0) D0 D1".parse().unwrap(),
    ///     1e-15,
    /// ));
    /// assert!(dem.rounded(3).approx_equals(
    ///     &"error(0.019) D0\nerror(0) D0 D1".parse().unwrap(),
    ///     1e-15,
    /// ));
    /// ```
    #[must_use]
    pub fn rounded(&self, digits: u8) -> Self {
        Self::from_inner(self.inner.rounded(digits))
    }

    /// Compiles a [`DemSampler`] for the detector error model, seeded from
    /// system entropy.
    ///
    /// The returned sampler can efficiently generate batches of detection event
    /// samples and observable flip samples directly from the error model,
    /// without simulating a full circuit. Each call to
    /// [`DemSampler::sample`](crate::DemSampler::sample) independently fires
    /// each error mechanism with its stated probability, then computes which
    /// detectors and observables are flipped.
    ///
    /// Because the seed is drawn from system entropy, results are
    /// non-deterministic. Use [`compile_sampler_with_seed`](Self::compile_sampler_with_seed)
    /// for reproducible sampling.
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

    /// Compiles a [`DemSampler`] for the detector error model using an
    /// explicit seed for the random number generator.
    ///
    /// When set to the same seed, making the exact same series of calls on the
    /// exact same machine with the exact same version of Stim will produce the
    /// exact same simulation results.
    ///
    /// **Caution:** Simulation results are *not* guaranteed to be consistent
    /// between different versions of Stim, across machines with different SIMD
    /// instruction widths (e.g. AVX vs. SSE), or if you vary the number of
    /// shots per call.
    ///
    /// # Examples
    ///
    /// ```
    /// let dem: stim::DetectorErrorModel = "error(0.5) D0 L0".parse().unwrap();
    /// let mut s1 = dem.compile_sampler_with_seed(42);
    /// let mut s2 = dem.compile_sampler_with_seed(42);
    /// let (d1, o1, e1) = s1.sample(10);
    /// let (d2, o2, e2) = s2.sample(10);
    /// assert_eq!(d1, d2);
    /// assert_eq!(o1, o2);
    /// assert_eq!(e1, e2);
    /// ```
    #[must_use]
    pub fn compile_sampler_with_seed(&self, seed: u64) -> DemSampler {
        DemSampler {
            inner: self.inner.compile_sampler_with_seed(seed),
        }
    }

    /// Finds a minimum-weight set of graphlike errors that produces an
    /// undetected logical error (a logical error that flips at least one
    /// observable while triggering no detection events).
    ///
    /// This method does **not** pay attention to error probabilities (other
    /// than ignoring errors with probability 0). It searches for a logical
    /// error with the minimum *number* of physical error mechanisms, not the
    /// maximum probability of those mechanisms all occurring. Use
    /// [`likeliest_error_sat_problem`](Self::likeliest_error_sat_problem) if
    /// you need a probability-aware search.
    ///
    /// A "graphlike error" is an error that produces at most two detection
    /// events (symptoms). Errors that decompose into graphlike components via
    /// the `^` separator are also accepted. Non-graphlike errors (e.g.
    /// `error(0.1) D0 D1 D2` without decomposition) are either skipped or
    /// cause an error, depending on `ignore_ungraphlike_errors`.
    ///
    /// The search works by converting each frame-changing error into one or
    /// two symptoms and a net frame change, then performing a breadth-first
    /// search that moves symptoms along error edges until they cancel or
    /// reach a boundary. If a net frame change remains when all symptoms are
    /// gone, a logical error has been found.
    ///
    /// The returned model contains only `error` instructions (no `repeat` or
    /// `shift_detectors`), all with probability set to 1. The `len()` of the
    /// returned model is the graphlike code distance.
    ///
    /// **Note:** The true code distance may be smaller than the graphlike code
    /// distance. For example, in the XZ surface code with twists, the true
    /// minimum-sized logical error likely uses Y errors, each of which
    /// decomposes into two graphlike components.
    ///
    /// # Errors
    ///
    /// Returns an error if `ignore_ungraphlike_errors` is `false` and the
    /// model contains errors that are not graphlike and not decomposed into
    /// graphlike components.
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
            .map(Self::from_inner)
            .map_err(StimError::from)
    }

    /// Encodes the shortest-error search as a weighted partial maxSAT problem
    /// in WDIMACS format.
    ///
    /// The optimal solution to the returned problem is the fault distance of
    /// the model: the minimum number of error mechanisms that combine to flip
    /// at least one logical observable while producing no detection events.
    /// This method ignores error probabilities — it only minimises the count
    /// of triggered mechanisms.
    ///
    /// The output string can be fed to any maxSAT solver that accepts WDIMACS
    /// format (see <http://www.maxhs.org/docs/wdimacs.html>).
    ///
    /// This is a convenience wrapper that calls
    /// [`shortest_error_sat_problem_with_format`](Self::shortest_error_sat_problem_with_format)
    /// with `"WDIMACS"`.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be converted to a SAT problem
    /// (e.g. it contains no observables).
    pub fn shortest_error_sat_problem(&self) -> Result<String> {
        self.shortest_error_sat_problem_with_format("WDIMACS")
    }

    /// Encodes the shortest-error search as a maxSAT problem in the specified
    /// format.
    ///
    /// See [`shortest_error_sat_problem`](Self::shortest_error_sat_problem)
    /// for a description of the problem being encoded.
    ///
    /// # Errors
    ///
    /// Returns an error if `format_name` is not a recognised format string, or
    /// if the model cannot be converted to a SAT problem.
    pub fn shortest_error_sat_problem_with_format(&self, format_name: &str) -> Result<String> {
        self.inner
            .shortest_error_sat_problem(format_name)
            .map_err(StimError::from)
    }

    /// Encodes the likeliest-error search as a weighted partial maxSAT problem
    /// in WDIMACS format, using the default quantization of 100.
    ///
    /// The optimal solution to the returned problem is the highest-likelihood
    /// set of error mechanisms that combine to flip at least one logical
    /// observable while producing no detection events. Unlike
    /// [`shortest_error_sat_problem`](Self::shortest_error_sat_problem), this
    /// method **does** take error probabilities into account: each error is
    /// weighted by the log-odds of its probability, so that more likely errors
    /// are preferred.
    ///
    /// Error probabilities are converted to log-odds and scaled/rounded to
    /// positive integers. If any error has probability *p* > 0.5, it is
    /// inverted so the weight remains positive. Errors with probability close
    /// to 0.5 may receive a weight of 0, meaning they can be included or
    /// excluded with no effect on the objective.
    ///
    /// This is a convenience wrapper that calls
    /// [`likeliest_error_sat_problem_with_options`](Self::likeliest_error_sat_problem_with_options)
    /// with `quantization = 100` and `format_name = "WDIMACS"`.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be converted to a SAT problem.
    pub fn likeliest_error_sat_problem(&self) -> Result<String> {
        self.likeliest_error_sat_problem_with_options(100, "WDIMACS")
    }

    /// Encodes the likeliest-error search as a maxSAT problem with explicit
    /// quantization and format options.
    ///
    /// `quantization` controls the precision of the log-odds weights: error
    /// probabilities are converted to log-odds and scaled/rounded to positive
    /// integers at most this large. A larger value gives more accurate
    /// quantization (the returned error set's likelihood will be closer to the
    /// true optimum) at the cost of potentially slower maxSAT solving.
    ///
    /// `format_name` selects the output format. `"WDIMACS"` is the standard
    /// weighted partial maxSAT format (see
    /// <http://www.maxhs.org/docs/wdimacs.html>).
    ///
    /// # Errors
    ///
    /// Returns an error if `format_name` is not a recognised format string, or
    /// if the model cannot be converted to a SAT problem.
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

    /// Returns the coordinate metadata of detectors in the error model.
    ///
    /// Detector coordinates are set by `detector(...)` and
    /// `shift_detectors(...)` instructions. This method resolves all shifts
    /// and returns a map from detector index to its coordinate list. Detectors
    /// with no explicitly specified coordinates are mapped to an empty `Vec`.
    ///
    /// If `only` is `Some`, only the listed detector indices are included in
    /// the result and `set(result.keys()) == set(only)`. If `only` is `None`,
    /// all detectors from index 0 up to [`num_detectors`](Self::num_detectors)
    /// − 1 are included.
    ///
    /// # Errors
    ///
    /// Returns an error if any detector index in `only` exceeds the model's
    /// detector count.
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

    /// Returns a diagram of the detector error model as a string.
    ///
    /// The diagram visualises the decoding graph implied by the model.
    /// Available diagram types are:
    ///
    /// - `"matchgraph-svg"` (or `"match-graph-svg"`): An SVG image of the
    ///   decoding graph. Red lines are errors crossing a logical observable;
    ///   blue lines are undecomposed hyper-errors.
    /// - `"match-graph-svg-html"`: Same as `matchgraph-svg` but wrapped in a
    ///   resizable HTML iframe.
    /// - `"matchgraph-3d"`: A GLTF 3D model of the decoding graph. GLTF files
    ///   can be opened in viewers such as
    ///   <https://gltf-viewer.donmccurdy.com/>. Red lines are errors crossing
    ///   a logical observable.
    /// - `"matchgraph-3d-html"`: The 3D model embedded in an HTML page with an
    ///   interactive THREE.js viewer.
    ///
    /// # Errors
    ///
    /// Returns an error if `type_name` is not a recognized diagram type.
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

    /// Appends a new detector error model instruction built from its
    /// constituent parts: instruction type name, parenthesized arguments,
    /// targets, and an optional tag.
    ///
    /// This is the primary way to programmatically build up a detector error
    /// model instruction by instruction. The `instruction_type` is a string
    /// like `"error"`, `"shift_detectors"`, `"detector"`, or
    /// `"logical_observable"`. The `parens_arguments` are the numeric values
    /// inside the parentheses (e.g. the `0.25` in `error(0.25) D0`). The
    /// `targets` are the instruction targets (e.g. detector ids, observable
    /// ids, separators, or raw integers). The `tag` is an arbitrary string
    /// annotation placed in square brackets.
    ///
    /// # Errors
    ///
    /// Returns an error if the instruction type is not recognised, or if the
    /// combination of arguments and targets is invalid (e.g. an empty
    /// instruction type string).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut dem = stim::DetectorErrorModel::new();
    /// dem.append("error", [0.125], [stim::DemTarget::relative_detector_id(1).unwrap()], "")
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

    /// Appends an existing [`DemInstruction`] to the end of this model.
    ///
    /// The instruction is serialized and re-parsed, so the appended
    /// instruction is an independent copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the serialized instruction text is invalid (this
    /// should not happen for well-formed `DemInstruction` values).
    pub fn append_dem_instruction(&mut self, instruction: &DemInstruction) -> Result<()> {
        let raw_targets = instruction
            .targets_copy()
            .into_iter()
            .map(|target| match target {
                DemInstructionTarget::DemTarget(target) => target.raw_data(),
                DemInstructionTarget::RelativeOffset(value) => value,
            })
            .collect::<Vec<_>>();
        self.inner
            .append_instruction(
                instruction.r#type(),
                &instruction.args_copy(),
                &raw_targets,
                instruction.tag(),
            )
            .map_err(StimError::from)?;
        self.invalidate_item_cache();
        Ok(())
    }

    /// Appends an existing [`DemRepeatBlock`] to the end of this model.
    ///
    /// The repeat block is serialized and re-parsed, so the appended block is
    /// an independent copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the serialized repeat block text is invalid.
    pub fn append_dem_repeat_block(&mut self, repeat_block: &DemRepeatBlock) -> Result<()> {
        self.inner
            .append_repeat_block(repeat_block.repeat_count(), &repeat_block.body_copy().inner)
            .map_err(StimError::from)?;
        self.invalidate_item_cache();
        Ok(())
    }

    /// Appends all instructions from another [`DetectorErrorModel`] to the end
    /// of this model.
    ///
    /// If the other model is empty, this is a no-op. The appended model is
    /// serialized and re-parsed, so mutations to the source after appending
    /// have no effect on this model.
    ///
    /// # Errors
    ///
    /// Returns an error if re-parsing the combined model text fails.
    pub fn append_detector_error_model(&mut self, model: &DetectorErrorModel) -> Result<()> {
        if model.is_empty() {
            return Ok(());
        }
        self.inner.add_assign(&model.inner);
        self.invalidate_item_cache();
        Ok(())
    }

    /// Appends a detector-error-model operation of any supported owned type:
    /// a [`DemInstruction`], a [`DemRepeatBlock`], or an entire
    /// [`DetectorErrorModel`].
    ///
    /// This is a convenience method that dispatches to
    /// [`append_dem_instruction`](Self::append_dem_instruction),
    /// [`append_dem_repeat_block`](Self::append_dem_repeat_block), or
    /// [`append_detector_error_model`](Self::append_detector_error_model)
    /// depending on the variant of the [`DemAppendOperation`]. It also accepts
    /// references and owned values of the underlying types via `Into`
    /// conversions.
    ///
    /// # Errors
    ///
    /// Returns an error if the serialized text of the operation is invalid.
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

    /// Returns a copy of a single top-level item by index.
    ///
    /// The `index` supports Python-style negative indexing: `-1` is the last
    /// item, `-2` is the second-to-last, and so on. The returned [`DemItem`]
    /// is an independent copy; mutating it does not affect this model.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of range for the number of
    /// top-level items.
    ///
    /// # Examples
    ///
    /// ```
    /// let model: stim::DetectorErrorModel = "
    ///     error(0.125) D0
    ///     error(0.125) D1 L1
    ///     repeat 100 {
    ///         error(0.125) D1 D2
    ///         shift_detectors 1
    ///     }
    ///     detector D5
    /// ".parse().unwrap();
    ///
    /// // First item is an error instruction.
    /// assert!(matches!(model.get(0).unwrap(), stim::DemItem::Instruction(_)));
    ///
    /// // Third item (index 2) is a repeat block.
    /// assert!(matches!(model.get(2).unwrap(), stim::DemItem::RepeatBlock(_)));
    ///
    /// // Negative indexing: last item.
    /// assert!(matches!(model.get(-1).unwrap(), stim::DemItem::Instruction(_)));
    /// ```
    pub fn get(&self, index: isize) -> Result<DemItem> {
        let normalized = normalize_index(index, self.len())
            .ok_or_else(|| StimError::new(format!("index {index} out of range")))?;
        Ok(self.cached_top_level_items()[normalized].clone())
    }

    /// Returns a new detector error model containing a slice of top-level
    /// items, analogous to Python's `model[start:stop:step]` syntax.
    ///
    /// The `start`, `stop`, and `step` parameters follow Python slice
    /// semantics, including support for negative indices.
    ///
    /// # Errors
    ///
    /// Returns an error if `step` is zero, or if the resulting text cannot be
    /// parsed back into a valid model.
    ///
    /// # Examples
    ///
    /// ```
    /// let model: stim::DetectorErrorModel = "
    ///     error(0.125) D0
    ///     error(0.125) D1 L1
    ///     error(0.125) D2
    ///     detector D5
    /// ".parse().unwrap();
    ///
    /// // Every other item starting from index 1.
    /// let sliced = model.slice(Some(1), None, 2).unwrap();
    /// assert_eq!(sliced.len(), 2);
    ///
    /// // Last two items.
    /// let tail = model.slice(Some(-2), None, 1).unwrap();
    /// assert_eq!(tail.len(), 2);
    /// ```
    pub fn slice(&self, start: Option<isize>, stop: Option<isize>, step: isize) -> Result<Self> {
        if step == 0 {
            return Err(StimError::new("slice step cannot be zero"));
        }
        let len = self.len() as isize;
        let indices = compute_slice_indices(len, start, stop, step);
        if indices.is_empty() {
            return Ok(Self::new());
        }
        let slice_start = indices[0];
        let slice_step = if indices.len() >= 2 {
            indices[1] - indices[0]
        } else {
            step
        };
        Ok(Self::from_inner(self.inner.get_slice(
            slice_start as i64,
            slice_step as i64,
            indices.len() as i64,
        )))
    }

    fn top_level_items(&self) -> Result<Vec<DemItem>> {
        Ok(self.cached_top_level_items().clone())
    }

    fn top_level_item(&self, index: usize) -> DemItem {
        Self::build_top_level_item(&self.inner, index)
    }

    fn build_top_level_item(inner: &stim_cxx::DetectorErrorModel, index: usize) -> DemItem {
        let item = inner.top_level_item(index);
        if item.is_repeat_block {
            DemItem::RepeatBlock(
                DemRepeatBlock::new(
                    item.repeat_count,
                    &Self::from_inner(inner.top_level_repeat_block_body(index)),
                )
                .expect("stim-cxx repeat block data should be valid"),
            )
        } else {
            let targets: Vec<_> = if item.instruction_type == "shift_detectors" {
                item.targets
                    .into_iter()
                    .map(DemInstructionTarget::from)
                    .collect()
            } else {
                item.targets
                    .into_iter()
                    .map(|raw| DemInstructionTarget::from(DemTarget::from_raw_data(raw)))
                    .collect()
            };
            DemItem::Instruction(
                DemInstruction::new(item.instruction_type, item.args, targets, item.tag)
                    .expect("stim-cxx instruction data should be valid"),
            )
        }
    }

    fn cached_top_level_items(&self) -> &Vec<DemItem> {
        self.item_cache.get_or_init(|| {
            (0..self.len())
                .map(|index| self.top_level_item(index))
                .collect()
        })
    }
}

impl IntoIterator for DetectorErrorModel {
    type Item = DemItem;
    type IntoIter = std::vec::IntoIter<DemItem>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.len();
        let Self { inner, item_cache } = self;
        item_cache
            .into_inner()
            .unwrap_or_else(|| {
                (0..len)
                    .map(|index| Self::build_top_level_item(&inner, index))
                    .collect()
            })
            .into_iter()
    }
}

impl IntoIterator for &DetectorErrorModel {
    type Item = DemItem;
    type IntoIter = std::vec::IntoIter<DemItem>;

    fn into_iter(self) -> Self::IntoIter {
        self.top_level_items()
            .expect("valid DetectorErrorModel values must iterate as valid top-level items")
            .into_iter()
    }
}

impl IntoIterator for &mut DetectorErrorModel {
    type Item = DemItem;
    type IntoIter = std::vec::IntoIter<DemItem>;

    fn into_iter(self) -> Self::IntoIter {
        self.top_level_items()
            .expect("valid DetectorErrorModel values must iterate as valid top-level items")
            .into_iter()
    }
}

impl Clone for DetectorErrorModel {
    fn clone(&self) -> Self {
        Self::from_inner(self.inner.clone())
    }
}

impl Default for DetectorErrorModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Determines if two detector error models have identical contents.
///
/// Two models are equal if and only if they have the exact same sequence of
/// instructions with the exact same arguments and targets. Approximate
/// equality (e.g. within a tolerance on probabilities) is available via
/// [`approx_equals`](DetectorErrorModel::approx_equals).
impl PartialEq for DetectorErrorModel {
    fn eq(&self, other: &Self) -> bool {
        self.inner.equals(&other.inner)
    }
}

impl Eq for DetectorErrorModel {}

/// Creates a new detector error model by concatenating two models.
///
/// The resulting model contains all instructions from the left-hand side
/// followed by all instructions from the right-hand side. Neither operand
/// is modified.
///
/// # Examples
///
/// ```
/// let m1: stim::DetectorErrorModel = "error(0.125) D0".parse().unwrap();
/// let m2: stim::DetectorErrorModel = "error(0.25) D1".parse().unwrap();
/// let combined = m1 + m2;
/// assert_eq!(combined.to_string(), "error(0.125) D0\nerror(0.25) D1");
/// ```
impl Add for DetectorErrorModel {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_inner(self.inner.add(&rhs.inner))
    }
}

/// Appends a detector error model into the receiving model, mutating it
/// in place.
///
/// After `a += b`, `a` contains all of its original instructions followed
/// by all instructions from `b`.
///
/// # Examples
///
/// ```
/// let mut m1: stim::DetectorErrorModel = "error(0.125) D0".parse().unwrap();
/// let m2: stim::DetectorErrorModel = "error(0.25) D1".parse().unwrap();
/// m1 += m2;
/// assert_eq!(m1.to_string(), "error(0.125) D0\nerror(0.25) D1");
/// ```
impl AddAssign for DetectorErrorModel {
    fn add_assign(&mut self, rhs: Self) {
        self.inner.add_assign(&rhs.inner);
        self.invalidate_item_cache();
    }
}

/// Repeats the detector error model by wrapping its contents in a repeat
/// block.
///
/// Has special cases:
/// - `model * 0` returns an empty model.
/// - `model * 1` returns a copy of the model (no repeat block wrapper).
/// - `model * n` (for `n >= 2`) returns a model with a single `repeat n { … }` block.
///
/// # Examples
///
/// ```
/// let m: stim::DetectorErrorModel = "error(0.25) D0\nshift_detectors 1".parse().unwrap();
/// assert_eq!(
///     (m * 3).to_string(),
///     "repeat 3 {\n    error(0.25) D0\n    shift_detectors 1\n}"
/// );
/// ```
impl Mul<u64> for DetectorErrorModel {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self::Output {
        Self::from_inner(self.inner.repeat(rhs))
    }
}

/// Repeats the detector error model (right-multiply variant), allowing
/// `3 * model` as an alternative spelling of `model * 3`.
///
/// See [`Mul<u64> for DetectorErrorModel`] for full semantics.
///
/// # Examples
///
/// ```
/// let m: stim::DetectorErrorModel = "error(0.25) D0\nshift_detectors 1".parse().unwrap();
/// assert_eq!(
///     (3 * m).to_string(),
///     "repeat 3 {\n    error(0.25) D0\n    shift_detectors 1\n}"
/// );
/// ```
impl Mul<DetectorErrorModel> for u64 {
    type Output = DetectorErrorModel;

    fn mul(self, rhs: DetectorErrorModel) -> Self::Output {
        rhs * self
    }
}

/// Mutates the detector error model by wrapping its contents in a repeat
/// block in place.
///
/// - `model *= 0` clears the model.
/// - `model *= 1` is a no-op.
/// - `model *= n` (for `n >= 2`) replaces the model's contents with a
///   single `repeat n { … }` block.
///
/// # Examples
///
/// ```
/// let mut m: stim::DetectorErrorModel = "error(0.25) D0\nshift_detectors 1".parse().unwrap();
/// m *= 3;
/// assert_eq!(
///     m.to_string(),
///     "repeat 3 {\n    error(0.25) D0\n    shift_detectors 1\n}"
/// );
/// ```
impl MulAssign<u64> for DetectorErrorModel {
    fn mul_assign(&mut self, rhs: u64) {
        self.inner.repeat_assign(rhs);
        self.invalidate_item_cache();
    }
}

impl Index<usize> for DetectorErrorModel {
    type Output = DemItem;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cached_top_level_items()[index]
    }
}

/// Returns the contents of the detector error model in the `.dem` file
/// format.
///
/// The output can be parsed back into an equivalent model via
/// [`FromStr`].
impl Display for DetectorErrorModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.to_dem_text())
    }
}

/// Produces a debug representation that resembles the Rust constructor
/// syntax: `stim::DetectorErrorModel("""...""")` for non-empty models, or
/// `stim::DetectorErrorModel()` for empty ones.
impl fmt::Debug for DetectorErrorModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self == &Self::new() {
            return f.write_str("stim::DetectorErrorModel()");
        }

        write!(f, "stim::DetectorErrorModel(\"\"\"\n{}\n\"\"\")", self)
    }
}

/// Parses a detector error model from its `.dem` text representation.
///
/// # Examples
///
/// ```
/// let model: stim::DetectorErrorModel = "error(0.125) D0 D1 L0".parse().unwrap();
/// assert_eq!(model.len(), 1);
/// ```
impl FromStr for DetectorErrorModel {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self> {
        stim_cxx::DetectorErrorModel::from_dem_text(s)
            .map(Self::from_inner)
            .map_err(StimError::from)
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

    use crate::{Circuit, DemInstructionTarget, DemRepeatBlock, DemTarget};

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
            DemItem::Instruction(
                DemInstruction::new(
                    "error",
                    [0.125],
                    [DemTarget::relative_detector_id(0).unwrap()],
                    ""
                )
                .unwrap()
            )
        );
        assert_eq!(
            model.get(2).unwrap(),
            DemItem::RepeatBlock(
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
            DemItem::Instruction(
                DemInstruction::new(
                    "detector",
                    [],
                    [DemTarget::relative_detector_id(5).unwrap()],
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
    fn detector_error_model_supports_borrowed_and_owned_iteration() {
        let model: DetectorErrorModel = "\
    error(0.125) D0
    repeat 2 {
        shift_detectors 1
    }
    logical_observable L0"
            .parse()
            .unwrap();

        let expected = vec![
            DemItem::Instruction(
                DemInstruction::new(
                    "error",
                    [0.125],
                    [DemTarget::relative_detector_id(0).unwrap()],
                    "",
                )
                .unwrap(),
            ),
            DemItem::RepeatBlock(
                DemRepeatBlock::new(
                    2,
                    &"shift_detectors 1".parse::<DetectorErrorModel>().unwrap(),
                )
                .unwrap(),
            ),
            DemItem::Instruction(
                DemInstruction::new(
                    "logical_observable",
                    [],
                    [DemTarget::logical_observable_id(0).unwrap()],
                    "",
                )
                .unwrap(),
            ),
        ];
        let mut mutable = model.clone();

        assert_eq!((&model).into_iter().collect::<Vec<_>>(), expected);
        assert_eq!((&mut mutable).into_iter().collect::<Vec<_>>(), expected);
        assert_eq!(model.clone().into_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn detector_error_model_supports_indexing_top_level_items() {
        let model: DetectorErrorModel = "\
    error(0.125) D0
    repeat 2 {
        shift_detectors 1
    }
    logical_observable L0"
            .parse()
            .unwrap();

        assert_eq!(
            model[0],
            DemItem::Instruction(
                DemInstruction::new(
                    "error",
                    [0.125],
                    [DemTarget::relative_detector_id(0).unwrap()],
                    "",
                )
                .unwrap(),
            )
        );
        assert_eq!(
            model[1],
            DemItem::RepeatBlock(
                DemRepeatBlock::new(
                    2,
                    &"shift_detectors 1".parse::<DetectorErrorModel>().unwrap()
                )
                .unwrap(),
            )
        );
    }

    #[test]
    fn detector_error_model_index_cache_invalidates_after_mutation() {
        let mut model: DetectorErrorModel = "error(0.125) D0".parse().unwrap();
        assert_eq!(
            model[0],
            DemItem::Instruction(
                DemInstruction::new(
                    "error",
                    [0.125],
                    [DemTarget::relative_detector_id(0).unwrap()],
                    "",
                )
                .unwrap(),
            )
        );

        model += "logical_observable L0"
            .parse::<DetectorErrorModel>()
            .unwrap();

        assert_eq!(model.len(), 2);
        assert_eq!(
            model[1],
            DemItem::Instruction(
                DemInstruction::new(
                    "logical_observable",
                    [],
                    [DemTarget::logical_observable_id(0).unwrap()],
                    "",
                )
                .unwrap(),
            )
        );
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
                [DemTarget::relative_detector_id(1).unwrap()],
                "",
            )
            .unwrap();
        model
            .append(
                "error",
                [0.25],
                [
                    DemTarget::relative_detector_id(1).unwrap(),
                    DemTarget::separator(),
                    DemTarget::relative_detector_id(2).unwrap(),
                    DemTarget::logical_observable_id(3).unwrap(),
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
                [DemTarget::relative_detector_id(1).unwrap()],
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
            [DemTarget::relative_detector_id(1).unwrap()],
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

        let copy = model.clone();
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

    #[test]
    fn detector_error_model_remaining_sampler_and_slice_paths_are_covered() {
        let dem: DetectorErrorModel = "error(0.125) D0\nshift_detectors(1, 2) 5\nerror(0.25) D1 L0"
            .parse()
            .unwrap();
        let sampler = dem.compile_sampler();
        assert_eq!(sampler.num_detectors(), 7);
        let sat = dem.likeliest_error_sat_problem().unwrap();
        assert!(sat.contains("p wcnf") || sat.contains("p cnf"));

        let zero_step = dem.slice(None, None, 0).unwrap_err();
        assert!(zero_step.message().contains("slice step cannot be zero"));
        assert!(dem.slice(Some(10), Some(11), 1).unwrap().is_empty());
        assert_eq!(
            dem.slice(Some(1), Some(2), 9).unwrap().to_string(),
            "shift_detectors(1, 2) 5"
        );
        assert_eq!(
            dem.get(1).unwrap(),
            DemItem::Instruction("shift_detectors(1, 2) 5".parse().unwrap())
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
