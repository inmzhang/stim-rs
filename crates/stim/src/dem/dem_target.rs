use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::{Result, StimError};

const MAX_OBS: u64 = 0xFFFF_FFFF;
const MAX_DET: u64 = (1u64 << 62) - 1;
const OBSERVABLE_BIT: u64 = 1u64 << 63;
const SEPARATOR_SYGIL: u64 = u64::MAX;

/// An instruction target from a detector error model (`.dem`) file.
///
/// A `DemTarget` represents one of the three fundamental target types that can
/// appear in a detector error model instruction. Every target in a DEM
/// instruction is exactly one of:
///
/// - **Relative detector id** (`D0`, `D1`, …) — references a detector by its
///   index relative to the current detector offset. In a `.dem` file these are
///   written as `D<index>`. For example, `D5` refers to the detector whose index
///   is 5 greater than the current detector offset (which is 0 at the top level,
///   but shifts inside `repeat` blocks). Detector indices must be in the range
///   `0..2^62 - 1`.
///
/// - **Logical observable id** (`L0`, `L1`, …) — references a logical
///   observable that tracks frame changes across the circuit. In a `.dem` file
///   these are written as `L<index>`. For example, in `error(0.25) D0 L1` the
///   `L1` identifies observable 1. Observable indices must be in the range
///   `0..0xFFFF_FFFF`.
///
/// - **Separator** (`^`) — delimits groups of targets within a single DEM
///   instruction. Separators are used to specify *suggested decompositions* of a
///   multi-detector error into simpler components. For example,
///   `error(0.25) D1 D2 ^ D3 D4` suggests decomposing the error into the pair
///   `{D1, D2}` and the pair `{D3, D4}`.
///
/// `DemTarget` is a lightweight `Copy` value (backed by a single `u64`) that
/// implements `Eq`, `Ord`, `Hash`, `Display`, and `Debug`, making it cheap to
/// store in collections and easy to inspect.
///
/// # Construction
///
/// Targets can be constructed via:
/// - [`DemTarget::relative_detector_id`]
/// - [`DemTarget::logical_observable_id`]
/// - [`DemTarget::separator`]
/// - [`DemTarget::from_text`] (parses `"D5"`, `"L2"`, `"^"`)
///
/// # Classification
///
/// Once constructed, use the `is_*` predicates to determine which variant a
/// target is, and [`val`](Self::val) or [`raw_id`](Self::raw_id) to extract
/// the numeric index.
///
/// # Examples
///
/// ```
/// let det = stim::DemTarget::relative_detector_id(5).expect("valid detector id");
/// assert!(det.is_relative_detector_id());
/// assert_eq!(det.to_string(), "D5");
///
/// let obs = stim::DemTarget::logical_observable_id(2).expect("valid observable id");
/// assert!(obs.is_logical_observable_id());
/// assert_eq!(obs.to_string(), "L2");
///
/// let sep = stim::DemTarget::separator();
/// assert!(sep.is_separator());
/// assert_eq!(sep.to_string(), "^");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DemTarget {
    data: u64,
}

/// A DEM target paired with coordinate metadata.
///
/// Wraps a [`DemTarget`] together with an optional vector of floating-point
/// coordinates describing its spatial position. This is the type returned by
/// APIs that resolve detector indices to their physical or logical coordinates
/// — for example, when a circuit attaches coordinate arguments to its
/// `DETECTOR` instructions.
///
/// When the enclosed target is a relative detector id, it is guaranteed to be
/// *absolute* (i.e., relative to detector offset 0), so you can use it directly
/// as a global index without further adjustment.
///
/// Having the coordinates readily available alongside the target is especially
/// helpful when debugging a problem in a circuit: instead of manually looking up
/// the coordinates of a detector index to understand where an error lives in the
/// physical layout, you can inspect the `coords` field directly. Coordinates are
/// also used by visualization tools and matching-graph construction routines.
///
/// The coordinate vector may be empty if no coordinate metadata was attached to
/// the detector in the original circuit. The number and meaning of coordinates
/// is user-defined, but conventionally they represent spatial positions (e.g.,
/// `[x, y]` or `[x, y, t]` for space-time coordinates).
///
/// `DemTargetWithCoords` implements `Eq`, `Ord`, `Hash`, `Display`, and `Debug`
/// (floating-point coordinates are compared bitwise for ordering and hashing
/// purposes).
///
/// # Examples
///
/// ```
/// let det = stim::DemTarget::relative_detector_id(5).expect("valid detector id");
/// let with_coords = stim::DemTargetWithCoords::new(det, vec![1.0, -2.5, 3.25]);
/// assert_eq!(with_coords.dem_target(), det);
/// assert_eq!(with_coords.coords(), &[1.0, -2.5, 3.25]);
/// assert_eq!(with_coords.to_string(), "D5[coords 1,-2.5,3.25]");
/// ```
#[derive(Clone, PartialEq)]
pub struct DemTargetWithCoords {
    dem_target: DemTarget,
    coords: Vec<f64>,
}

impl DemTarget {
    #[must_use]
    pub(crate) const fn from_raw_data(data: u64) -> Self {
        Self { data }
    }

    #[must_use]
    pub(crate) const fn raw_data(self) -> u64 {
        self.data
    }

    /// Parses a `DemTarget` from its textual representation.
    ///
    /// This method recognizes the three canonical DEM target text formats:
    ///
    /// - `D<id>` — parsed as a relative detector id via
    ///   [`DemTarget::relative_detector_id`].
    /// - `L<id>` — parsed as a logical observable id via
    ///   [`DemTarget::logical_observable_id`].
    /// - `^` — parsed as the separator via [`DemTarget::separator`].
    ///
    /// The `<id>` portion must be a valid non-negative integer that fits within
    /// the range allowed for that target kind.
    ///
    /// This method is also available via the [`FromStr`] trait, so you can use
    /// `"D7".parse::<DemTarget>()` interchangeably with `DemTarget::from_text("D7")`.
    ///
    /// # Errors
    ///
    /// Returns an error if `text` does not match a known target format or
    /// if the numeric id is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// let det = stim::DemTarget::from_text("D7").expect("valid detector");
    /// assert!(det.is_relative_detector_id());
    /// assert_eq!(det.val().expect("has value"), 7);
    ///
    /// let obs = stim::DemTarget::from_text("L5").expect("valid observable");
    /// assert!(obs.is_logical_observable_id());
    ///
    /// let sep = stim::DemTarget::from_text("^").expect("valid separator");
    /// assert!(sep.is_separator());
    /// ```
    pub fn from_text(text: &str) -> Result<Self> {
        if text == "^" {
            return Ok(Self::separator());
        }
        if let Some(rest) = text.strip_prefix('D') {
            let id = parse_u64(rest, "relative detector id")?;
            return Self::relative_detector_id(id);
        }
        if let Some(rest) = text.strip_prefix('L') {
            let id = parse_u64(rest, "logical observable id")?;
            return Self::logical_observable_id(id);
        }
        Err(StimError::new(format!(
            "failed to parse as a stim.DemTarget: '{text}'"
        )))
    }

    /// Creates a logical observable target (`L<id>`).
    ///
    /// Returns a `DemTarget` representing a logical observable — a frame-change
    /// observable tracked across the circuit. In a DEM file, observable targets
    /// are written as `L<id>`. For example, in the DEM instruction
    /// `error(0.25) D0 L1`, the `L1` identifies observable index 1, meaning that
    /// the error mechanism flips the frame of logical observable 1 with
    /// probability 0.25.
    ///
    /// Observable indices are stored in 32 bits, so `id` must be at most
    /// `0xFFFF_FFFF` (4,294,967,295).
    ///
    /// # Errors
    ///
    /// Returns an error if `id` exceeds `0xFFFF_FFFF`.
    ///
    /// # Examples
    ///
    /// ```
    /// let obs = stim::DemTarget::logical_observable_id(3).expect("valid id");
    /// assert!(obs.is_logical_observable_id());
    /// assert_eq!(obs.val().expect("has value"), 3);
    /// assert_eq!(obs.to_string(), "L3");
    /// ```
    pub fn logical_observable_id(id: u64) -> Result<Self> {
        if id > MAX_OBS {
            return Err(StimError::new("id > 0xFFFFFFFF"));
        }
        Ok(Self {
            data: OBSERVABLE_BIT | id,
        })
    }

    /// Creates a relative detector target (`D<id>`).
    ///
    /// Returns a `DemTarget` representing a detector. In a DEM file, detector
    /// targets are written as `D<id>`, where `<id>` is the detector's index
    /// relative to the current detector offset. At the top level the offset is 0,
    /// so `D5` refers to detector 5 globally. Inside a `repeat` block the offset
    /// accumulates with each iteration, making the same `D5` refer to a different
    /// global detector on each pass.
    ///
    /// The maximum supported detector index is `2^62 - 1`
    /// (4,611,686,018,427,387,903).
    ///
    /// # Errors
    ///
    /// Returns an error if `id` exceeds the maximum supported detector index
    /// (2^62 - 1).
    ///
    /// # Examples
    ///
    /// ```
    /// let det = stim::DemTarget::relative_detector_id(5).expect("valid id");
    /// assert!(det.is_relative_detector_id());
    /// assert_eq!(det.val().expect("has value"), 5);
    /// assert_eq!(det.to_string(), "D5");
    /// ```
    pub fn relative_detector_id(id: u64) -> Result<Self> {
        if id > MAX_DET {
            return Err(StimError::new("Relative detector id too large."));
        }
        Ok(Self { data: id })
    }

    /// Returns the separator target (`^`).
    ///
    /// Separators delimit groups of targets within a single DEM instruction. They
    /// are used to express *suggested decompositions* of a multi-detector error
    /// into simpler components. For example, the DEM instruction:
    ///
    /// ```text
    /// error(0.25) D1 D2 ^ D3 D4
    /// ```
    ///
    /// describes a single error mechanism that flips detectors `{D1, D2, D3, D4}`
    /// with probability 0.25, and *suggests* that it can be decomposed into two
    /// independent pieces `{D1, D2}` and `{D3, D4}` separated by the `^`.
    ///
    /// There is exactly one separator value; it carries no numeric index.
    ///
    /// # Examples
    ///
    /// ```
    /// let sep = stim::DemTarget::separator();
    /// assert!(sep.is_separator());
    /// assert_eq!(sep.to_string(), "^");
    /// ```
    #[must_use]
    pub const fn separator() -> Self {
        Self {
            data: SEPARATOR_SYGIL,
        }
    }

    /// Returns `true` if this target is a logical observable id (`L<id>`).
    ///
    /// In a detector error model file, observable targets are prefixed by `L`.
    /// For example, in `error(0.25) D0 L1` the `L1` is an observable target, so
    /// calling `is_logical_observable_id()` on it returns `true`.
    ///
    /// Returns `false` for relative detector ids and for the separator.
    #[must_use]
    pub fn is_logical_observable_id(self) -> bool {
        self.data != SEPARATOR_SYGIL && (self.data & OBSERVABLE_BIT) != 0
    }

    /// Returns `true` if this target is the separator (`^`).
    ///
    /// Separators separate the components of a suggested decomposition within an
    /// error. For example, the `^` in `error(0.25) D1 D2 ^ D3 D4` is the
    /// separator. Calling `is_separator()` on the `^` target returns `true`.
    ///
    /// Returns `false` for relative detector ids and logical observable ids.
    #[must_use]
    pub fn is_separator(self) -> bool {
        self.data == SEPARATOR_SYGIL
    }

    /// Returns `true` if this target is a relative detector id (`D<id>`).
    ///
    /// In a detector error model file, detectors are prefixed by `D`. For
    /// example, in `error(0.25) D0 L1` the `D0` is a relative detector target,
    /// so calling `is_relative_detector_id()` on it returns `true`.
    ///
    /// Returns `false` for logical observable ids and for the separator.
    #[must_use]
    pub fn is_relative_detector_id(self) -> bool {
        self.data != SEPARATOR_SYGIL && (self.data & OBSERVABLE_BIT) == 0
    }

    /// Returns the numeric id with the observable/detector classification bit
    /// stripped.
    ///
    /// For detector targets this equals the detector index (`D5` → 5). For
    /// observable targets this equals the observable index (`L3` → 3). For the
    /// separator the return value is a large sentinel that is not meaningful as
    /// an index — prefer [`val`](Self::val) when you need a checked id that
    /// rejects separators.
    ///
    /// Unlike [`val`](Self::val), this method never fails, which makes it useful
    /// in contexts where you have already classified the target with one of the
    /// `is_*` predicates and want a cheap, infallible extraction.
    #[must_use]
    pub fn raw_id(self) -> u64 {
        self.data & !OBSERVABLE_BIT
    }

    /// Returns the numeric id of this target, failing for separators.
    ///
    /// For a relative detector id (`D5`), returns `5`. For a logical observable
    /// id (`L3`), returns `3`. Separators do not carry an integer value, so
    /// calling `val()` on one returns an error.
    ///
    /// This is the checked counterpart to [`raw_id`](Self::raw_id). Use `val()`
    /// when you are not certain of the target's kind and want to detect
    /// separators as errors rather than receiving a meaningless sentinel.
    ///
    /// # Errors
    ///
    /// Returns an error if this target is a separator, since separators do
    /// not carry an integer value.
    ///
    /// # Examples
    ///
    /// ```
    /// let det = stim::DemTarget::relative_detector_id(8).expect("valid id");
    /// assert_eq!(det.val().expect("has value"), 8);
    ///
    /// let sep = stim::DemTarget::separator();
    /// assert!(sep.val().is_err());
    /// ```
    pub fn val(self) -> Result<u64> {
        if self.is_separator() {
            return Err(StimError::new("Separator doesn't have an integer value."));
        }
        Ok(self.raw_id())
    }

    /// Adds `offset` to this target's detector index, if it is a relative
    /// detector id. Observable and separator targets are left unchanged.
    ///
    /// This is the primary mechanism for translating detector indices when
    /// flattening `repeat` blocks or composing detector error models. When a
    /// DEM repeat block is unrolled, each iteration shifts the relative detector
    /// ids in its body by a cumulative offset so that they map to the correct
    /// global detectors.
    ///
    /// The `offset` is a signed integer, allowing both forward and backward
    /// shifts. A negative offset is useful when rewriting instructions to
    /// reference earlier detectors.
    ///
    /// If this target is a logical observable id or a separator, the method is a
    /// no-op and always succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error if the shift overflows `i64` arithmetic or produces a
    /// negative detector index (negative detector ids are not representable).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut det = stim::DemTarget::relative_detector_id(4).expect("valid id");
    /// det.shift_if_detector_id(9).expect("shift succeeds");
    /// assert_eq!(det.val().expect("has value"), 13);
    ///
    /// // Observable targets are unaffected:
    /// let mut obs = stim::DemTarget::logical_observable_id(7).expect("valid id");
    /// obs.shift_if_detector_id(100).expect("no-op for observables");
    /// assert_eq!(obs.val().expect("has value"), 7);
    /// ```
    pub fn shift_if_detector_id(&mut self, offset: i64) -> Result<()> {
        if self.is_relative_detector_id() {
            let shifted = (self.data as i64)
                .checked_add(offset)
                .ok_or_else(|| StimError::new("detector id shift overflowed"))?;
            if shifted < 0 {
                return Err(StimError::new(
                    "detector id shift produced a negative value",
                ));
            }
            *self = Self::relative_detector_id(shifted as u64)?;
        }
        Ok(())
    }
}

impl FromStr for DemTarget {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_text(s)
    }
}

impl Display for DemTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_separator() {
            f.write_str("^")
        } else if self.is_relative_detector_id() {
            write!(f, "D{}", self.raw_id())
        } else {
            write!(f, "L{}", self.raw_id())
        }
    }
}

impl fmt::Debug for DemTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_separator() {
            f.write_str("stim::DemTarget::separator()")
        } else if self.is_relative_detector_id() {
            write!(f, "stim::DemTarget('D{}')", self.raw_id())
        } else {
            write!(f, "stim::DemTarget('L{}')", self.raw_id())
        }
    }
}

impl DemTargetWithCoords {
    /// Creates a new target-with-coordinates pairing.
    ///
    /// Bundles the given [`DemTarget`] together with a vector of floating-point
    /// coordinates. If the target has no associated coordinate metadata, pass an
    /// empty `Vec` — the coordinates will simply be omitted from the `Display`
    /// output.
    #[must_use]
    pub fn new(dem_target: DemTarget, coords: Vec<f64>) -> Self {
        Self { dem_target, coords }
    }

    /// Returns the underlying [`DemTarget`].
    ///
    /// If the target is a relative detector id, it is guaranteed to be absolute
    /// (i.e., relative to detector offset 0), so you can use the returned value
    /// as a global detector index directly.
    #[must_use]
    pub fn dem_target(&self) -> DemTarget {
        self.dem_target
    }

    /// Returns the coordinate metadata as a slice.
    ///
    /// The coordinates are the floating-point values that were attached to this
    /// target when it was created — typically originating from the `DETECTOR`
    /// instruction's coordinate arguments in the source circuit. The number of
    /// coordinates and their semantic meaning are user-defined, but by convention
    /// they represent spatial (and optionally temporal) positions, e.g.
    /// `[x, y]` or `[x, y, t]`.
    ///
    /// Returns an empty slice if no coordinate metadata was associated with the
    /// target.
    #[must_use]
    pub fn coords(&self) -> &[f64] {
        &self.coords
    }
}

impl Eq for DemTargetWithCoords {}

impl PartialOrd for DemTargetWithCoords {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DemTargetWithCoords {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dem_target
            .cmp(&other.dem_target)
            .then_with(|| compare_coord_slices(&self.coords, &other.coords))
    }
}

impl std::hash::Hash for DemTargetWithCoords {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dem_target.hash(state);
        for coord in &self.coords {
            coord.to_bits().hash(state);
        }
    }
}

impl Display for DemTargetWithCoords {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.dem_target)?;
        if !self.coords.is_empty() {
            f.write_str("[coords ")?;
            for (index, coord) in self.coords.iter().enumerate() {
                if index > 0 {
                    f.write_str(",")?;
                }
                write!(f, "{coord}")?;
            }
            f.write_str("]")?;
        }
        Ok(())
    }
}

impl fmt::Debug for DemTargetWithCoords {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("stim::DemTargetWithCoords")
            .field("dem_target", &self.dem_target)
            .field("coords", &self.coords)
            .finish()
    }
}

fn parse_u64(text: &str, kind: &str) -> Result<u64> {
    text.parse::<u64>()
        .map_err(|_| StimError::new(format!("failed to parse {kind}: {text:?}")))
}

fn compare_coord_slices(left: &[f64], right: &[f64]) -> std::cmp::Ordering {
    for (l, r) in left.iter().zip(right.iter()) {
        let cmp = l.to_bits().cmp(&r.to_bits());
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
    }
    left.len().cmp(&right.len())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};
    use std::str::FromStr;

    use super::{DemTarget, DemTargetWithCoords};

    #[test]
    fn dem_target_supports_equality_order_hash_and_representation() {
        let det_three = DemTarget::relative_detector_id(3).expect("detector target should build");
        let same_det_three = det_three;
        let det_four = DemTarget::relative_detector_id(4).expect("detector target should build");
        let obs_three =
            DemTarget::logical_observable_id(3).expect("observable target should build");
        let separator = DemTarget::separator();

        assert_eq!(det_three, same_det_three);
        assert_ne!(det_three, det_four);
        assert_ne!(det_three, obs_three);
        assert_ne!(obs_three, separator);

        let ordered = [separator, obs_three, det_four, det_three, same_det_three]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(ordered, vec![det_three, det_four, obs_three, separator]);

        let hashed = [det_three, det_four, same_det_three, obs_three, separator]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 4);
        assert!(hashed.contains(&det_three));
        assert!(hashed.contains(&det_four));
        assert!(hashed.contains(&obs_three));
        assert!(hashed.contains(&separator));

        assert_eq!(det_three.to_string(), "D3");
        assert_eq!(format!("{det_three:?}"), "stim::DemTarget('D3')");
        assert_eq!(obs_three.to_string(), "L3");
        assert_eq!(format!("{obs_three:?}"), "stim::DemTarget('L3')");
        assert_eq!(separator.to_string(), "^");
        assert_eq!(format!("{separator:?}"), "stim::DemTarget::separator()");
    }

    #[test]
    fn dem_target_exposes_detector_observable_and_separator_classification() {
        let detector = DemTarget::relative_detector_id(8).expect("detector target should build");
        let observable =
            DemTarget::logical_observable_id(2).expect("logical observable target should build");
        let separator = DemTarget::separator();

        assert!(detector.is_relative_detector_id());
        assert!(!detector.is_logical_observable_id());
        assert!(!detector.is_separator());
        assert_eq!(detector.raw_id(), 8);
        assert_eq!(
            detector.val().expect("detector target should have a value"),
            8
        );

        assert!(observable.is_logical_observable_id());
        assert!(!observable.is_relative_detector_id());
        assert!(!observable.is_separator());
        assert_eq!(observable.raw_id(), 2);
        assert_eq!(
            observable
                .val()
                .expect("logical observable target should have a value"),
            2
        );

        assert!(separator.is_separator());
        assert!(!separator.is_relative_detector_id());
        assert!(!separator.is_logical_observable_id());
        assert_eq!(separator.raw_id(), u64::MAX >> 1);

        let error = separator
            .val()
            .expect_err("separator should not expose an integer value");
        assert!(
            error
                .message()
                .contains("Separator doesn't have an integer value")
        );
    }

    #[test]
    fn dem_target_parses_representative_strings() {
        assert_eq!(
            DemTarget::from_str("D7").expect("detector target should parse"),
            DemTarget::relative_detector_id(7).expect("detector target should build")
        );
        assert_eq!(
            DemTarget::from_str("L5").expect("logical observable target should parse"),
            DemTarget::logical_observable_id(5).expect("logical observable target should build")
        );
        assert_eq!(
            DemTarget::from_str("^").expect("separator target should parse"),
            DemTarget::separator()
        );

        let error = DemTarget::from_str("rec[-1]").expect_err("invalid DEM target should fail");
        assert!(
            error
                .message()
                .contains("failed to parse as a stim.DemTarget: 'rec[-1]'")
        );
    }

    #[test]
    fn dem_target_with_coords_constructor_exposes_target_and_coords() {
        let detector = DemTarget::relative_detector_id(5).expect("detector target should build");
        let target_with_coords = DemTargetWithCoords::new(detector, vec![1.0, -2.5, 3.25]);

        assert_eq!(target_with_coords.dem_target(), detector);
        assert_eq!(target_with_coords.coords(), &[1.0, -2.5, 3.25]);
    }

    #[test]
    fn dem_target_with_coords_supports_equality_order_and_hash() {
        let detector_one =
            DemTarget::relative_detector_id(1).expect("detector target should build");
        let detector_two =
            DemTarget::relative_detector_id(2).expect("detector target should build");
        let logical = DemTarget::logical_observable_id(0).expect("logical target should build");

        let first = DemTargetWithCoords::new(detector_one, vec![1.0]);
        let same_as_first = DemTargetWithCoords::new(detector_one, vec![1.0]);
        let second = DemTargetWithCoords::new(detector_one, vec![2.0]);
        let third = DemTargetWithCoords::new(detector_two, vec![]);
        let fourth = DemTargetWithCoords::new(logical, vec![]);

        assert_eq!(first, same_as_first);
        assert_ne!(first, second);
        assert_ne!(first, third);

        let ordered = [
            fourth.clone(),
            second.clone(),
            third.clone(),
            same_as_first.clone(),
            first.clone(),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
        assert_eq!(ordered, vec![first.clone(), second, third, fourth.clone()]);

        let hashed = [first.clone(), same_as_first, fourth.clone()]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 2);
        assert!(hashed.contains(&first));
        assert!(hashed.contains(&fourth));
    }

    #[test]
    fn dem_target_with_coords_display_and_debug_are_informative() {
        let detector = DemTargetWithCoords::new(
            DemTarget::relative_detector_id(5).expect("detector target should build"),
            vec![1.0, -2.5],
        );
        let logical = DemTargetWithCoords::new(
            DemTarget::logical_observable_id(2).expect("logical target should build"),
            vec![],
        );

        assert_eq!(detector.to_string(), "D5[coords 1,-2.5]");
        assert_eq!(
            format!("{detector:?}"),
            "stim::DemTargetWithCoords { dem_target: stim::DemTarget('D5'), coords: [1.0, -2.5] }"
        );

        assert_eq!(logical.to_string(), "L2");
        assert_eq!(
            format!("{logical:?}"),
            "stim::DemTargetWithCoords { dem_target: stim::DemTarget('L2'), coords: [] }"
        );
    }

    #[test]
    fn dem_target_helpers_build_expected_targets_and_classify_them() {
        let detector = DemTarget::relative_detector_id(12).expect("detector helper should succeed");
        let observable =
            DemTarget::logical_observable_id(34).expect("logical observable helper should succeed");
        let separator = DemTarget::separator();

        assert_eq!(detector, DemTarget::relative_detector_id(12).unwrap());
        assert!(detector.is_relative_detector_id());
        assert!(!detector.is_logical_observable_id());
        assert!(!detector.is_separator());
        assert_eq!(detector.raw_id(), 12);
        assert_eq!(detector.val().unwrap(), 12);
        assert_eq!(detector.to_string(), "D12");
        assert_eq!(format!("{detector:?}"), "stim::DemTarget('D12')");

        assert_eq!(observable, DemTarget::logical_observable_id(34).unwrap());
        assert!(observable.is_logical_observable_id());
        assert!(!observable.is_relative_detector_id());
        assert!(!observable.is_separator());
        assert_eq!(observable.raw_id(), 34);
        assert_eq!(observable.val().unwrap(), 34);
        assert_eq!(observable.to_string(), "L34");
        assert_eq!(format!("{observable:?}"), "stim::DemTarget('L34')");

        assert!(separator.is_separator());
        assert!(!separator.is_relative_detector_id());
        assert!(!separator.is_logical_observable_id());
        assert_eq!(separator.to_string(), "^");
        assert_eq!(format!("{separator:?}"), "stim::DemTarget::separator()");
    }

    #[test]
    fn dem_target_helpers_support_maximum_constructor_bounds() {
        let max_detector = DemTarget::relative_detector_id((1u64 << 62) - 1)
            .expect("maximum relative detector id should succeed");
        let max_observable = DemTarget::logical_observable_id(0xFFFF_FFFF)
            .expect("maximum logical observable id should succeed");

        assert_eq!(max_detector.val().unwrap(), (1u64 << 62) - 1);
        assert_eq!(max_detector.to_string(), format!("D{}", (1u64 << 62) - 1));

        assert_eq!(max_observable.val().unwrap(), 0xFFFF_FFFF);
        assert_eq!(max_observable.to_string(), "L4294967295");
    }

    #[test]
    fn dem_target_helpers_reject_out_of_range_constructor_inputs() {
        let detector_error = DemTarget::relative_detector_id(1u64 << 62)
            .expect_err("out-of-range detector id should fail");
        let observable_error = DemTarget::logical_observable_id(0x1_0000_0000)
            .expect_err("out-of-range observable id should fail");

        assert_eq!(
            detector_error.to_string(),
            "Relative detector id too large."
        );
        assert_eq!(observable_error.to_string(), "id > 0xFFFFFFFF");
    }

    #[test]
    fn target_separator_rejects_integer_value_access() {
        let separator = DemTarget::separator();

        let error = separator
            .val()
            .expect_err("separator should not expose an integer value");

        assert_eq!(
            error.to_string(),
            "Separator doesn't have an integer value."
        );
    }

    #[test]
    fn dem_target_from_str_matches_from_text_for_representative_cases() {
        for text in ["D0", "D17", "L0", "L99", "^"] {
            assert_eq!(
                DemTarget::from_str(text).expect("FromStr should parse representative target"),
                DemTarget::from_text(text).expect("from_text should parse representative target")
            );
        }
    }

    #[test]
    fn dem_target_rejects_malformed_inputs() {
        for text in ["", "5", "d3", "l2", "^^", "rec[-1]", "*"] {
            let error = DemTarget::from_text(text).expect_err("malformed target should fail");
            assert!(
                error
                    .message()
                    .contains(&format!("failed to parse as a stim.DemTarget: '{text}'")),
                "unexpected error for {text:?}: {}",
                error.message()
            );
        }

        for (text, expected_fragment) in [
            ("D", "failed to parse relative detector id"),
            ("D-1", "failed to parse relative detector id"),
            ("L", "failed to parse logical observable id"),
            ("L-1", "failed to parse logical observable id"),
            ("D 1", "failed to parse relative detector id"),
            ("L 1", "failed to parse logical observable id"),
        ] {
            let error =
                DemTarget::from_str(text).expect_err("malformed prefixed target should fail");
            assert!(
                error.message().contains(expected_fragment),
                "unexpected error for {text:?}: {}",
                error.message()
            );
        }
    }

    #[test]
    fn dem_target_rejects_ids_outside_supported_bounds() {
        let max_detector = (1u64 << 62) - 1;
        let max_observable = 0xFFFF_FFFFu64;

        assert_eq!(
            DemTarget::from_text(&format!("D{max_detector}"))
                .expect("largest supported detector id should parse"),
            DemTarget::relative_detector_id(max_detector).expect("helper should build")
        );
        assert_eq!(
            DemTarget::from_text(&format!("L{max_observable}"))
                .expect("largest supported observable id should parse"),
            DemTarget::logical_observable_id(max_observable).expect("helper should build")
        );

        let detector_error = DemTarget::from_text(&format!("D{}", max_detector + 1))
            .expect_err("too-large detector id should fail");
        assert!(
            detector_error
                .message()
                .contains("Relative detector id too large."),
            "unexpected detector error: {}",
            detector_error.message()
        );

        let observable_error = DemTarget::from_text(&format!("L{}", max_observable + 1))
            .expect_err("too-large observable id should fail");
        assert!(
            observable_error.message().contains("id > 0xFFFFFFFF"),
            "unexpected observable error: {}",
            observable_error.message()
        );
    }

    #[test]
    fn dem_target_shift_if_detector_id_offsets_relative_detector_targets() {
        let mut target = DemTarget::relative_detector_id(4).expect("detector target should build");

        target
            .shift_if_detector_id(9)
            .expect("positive shift should succeed");
        assert_eq!(target, DemTarget::relative_detector_id(13).unwrap());
        assert!(target.is_relative_detector_id());
        assert!(!target.is_logical_observable_id());
        assert!(!target.is_separator());
        assert_eq!(target.val().unwrap(), 13);
        assert_eq!(target.to_string(), "D13");

        target
            .shift_if_detector_id(-5)
            .expect("in-range negative shift should succeed");
        assert_eq!(target, DemTarget::relative_detector_id(8).unwrap());
        assert_eq!(target.val().unwrap(), 8);
        assert_eq!(target.to_string(), "D8");
    }

    #[test]
    fn dem_target_shift_if_detector_id_is_a_no_op_for_observables_and_separators() {
        let observable =
            DemTarget::logical_observable_id(7).expect("observable target should build");
        let separator = DemTarget::separator();

        let mut shifted_observable = observable;
        shifted_observable
            .shift_if_detector_id(i64::MIN)
            .expect("observable targets should ignore detector shifts");
        assert_eq!(shifted_observable, observable);
        assert!(shifted_observable.is_logical_observable_id());
        assert_eq!(shifted_observable.val().unwrap(), 7);
        assert_eq!(shifted_observable.to_string(), "L7");

        let mut shifted_separator = separator;
        shifted_separator
            .shift_if_detector_id(i64::MAX)
            .expect("separator targets should ignore detector shifts");
        assert_eq!(shifted_separator, separator);
        assert!(shifted_separator.is_separator());
        assert_eq!(shifted_separator.to_string(), "^");
    }

    #[test]
    fn dem_target_shift_if_detector_id_rejects_negative_results_without_mutating_target() {
        let mut target = DemTarget::relative_detector_id(2).expect("detector target should build");

        let error = target
            .shift_if_detector_id(-3)
            .expect_err("shifts below zero should fail");

        assert!(
            error
                .message()
                .contains("detector id shift produced a negative value")
        );
        assert_eq!(target, DemTarget::relative_detector_id(2).unwrap());
        assert_eq!(target.to_string(), "D2");
    }
}
