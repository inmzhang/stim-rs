use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::{Result, StimError};

const TARGET_VALUE_MASK: u32 = (1u32 << 24) - 1;
const TARGET_INVERTED_BIT: u32 = 1u32 << 31;
const TARGET_PAULI_X_BIT: u32 = 1u32 << 30;
const TARGET_PAULI_Z_BIT: u32 = 1u32 << 29;
const TARGET_RECORD_BIT: u32 = 1u32 << 28;
const TARGET_COMBINER: u32 = 1u32 << 27;
const TARGET_SWEEP_BIT: u32 = 1u32 << 26;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A target operand used inside Stim circuit instructions.
///
/// Stim instructions act on *targets* that identify what the gate operates on. A
/// `GateTarget` can represent any of the following target kinds:
///
/// - **Qubit target** — a plain qubit index such as `5`, created with
///   [`GateTarget::new`] or [`GateTarget::qubit`]. Qubit targets can be optionally
///   inverted with a `!` prefix (e.g. `!5`), which flips the measurement result.
/// - **Pauli X / Y / Z target** — a qubit index tagged with a Pauli basis, such as
///   `X3`, `Y7`, or `!Z2`. These are used by instructions that operate on
///   Pauli-product targets, like `CORRELATED_ERROR` and `MPP`. Created with
///   [`GateTarget::x`], [`GateTarget::y`], [`GateTarget::z`], [`target_x`],
///   [`target_y`], [`target_z`], or [`target_pauli`].
/// - **Measurement-record target** — a backward reference into the measurement record
///   such as `rec[-1]` (the most recent measurement). Created with [`GateTarget::rec`]
///   or [`target_rec`].
/// - **Sweep-bit target** — an index into a per-shot configuration bitstring such as
///   `sweep[4]`. Created with [`GateTarget::sweep_bit`] or [`target_sweep_bit`].
/// - **Combiner** — the special `*` token that separates factors of a Pauli product
///   inside `MPP` instructions (e.g. `MPP X0*Y1*Z2`). Created with
///   [`GateTarget::combiner`] or [`target_combiner`].
///
/// `GateTarget` values are cheap to copy (they are a single `u32` internally) and
/// support equality, ordering, and hashing. They can be parsed from Stim text syntax
/// with [`GateTarget::from_target_str`] or via the [`FromStr`](std::str::FromStr)
/// trait.
///
/// # Examples
///
/// ```
/// // Qubit target
/// let qubit = stim::GateTarget::new(5u32);
/// assert!(qubit.is_qubit_target());
///
/// // Pauli target
/// let pauli_x = stim::target_x(3u32, false).unwrap();
/// assert!(pauli_x.is_x_target());
///
/// // Measurement-record target
/// let record = stim::target_rec(-1).unwrap();
/// assert!(record.is_measurement_record_target());
///
/// // Combiner
/// let combiner = stim::target_combiner();
/// assert!(combiner.is_combiner());
/// ```
pub struct GateTarget {
    data: u32,
}

impl GateTarget {
    /// Creates a gate target from a qubit index, an existing `GateTarget`, or any
    /// type that implements `Into<GateTarget>`.
    ///
    /// When given a `u32`, this creates a plain (non-inverted) qubit target. When
    /// given an existing `GateTarget`, it simply returns a copy.
    ///
    /// # Examples
    ///
    /// ```
    /// let qubit = stim::GateTarget::new(5u32);
    /// assert_eq!(qubit.to_string(), "5");
    ///
    /// // Passing an existing GateTarget returns a copy.
    /// let copy = stim::GateTarget::new(qubit);
    /// assert_eq!(copy, qubit);
    /// ```
    #[must_use]
    pub fn new(value: impl Into<Self>) -> Self {
        value.into()
    }

    #[must_use]
    pub(crate) const fn from_raw_data(data: u32) -> Self {
        Self { data }
    }

    /// Parses a gate target from Stim text syntax.
    ///
    /// Accepts all target forms used in Stim circuit files:
    /// - Plain qubit: `"5"`, `"0"`
    /// - Inverted qubit: `"!5"`
    /// - Pauli targets: `"X3"`, `"Y7"`, `"Z2"`, `"!Z2"` (case-insensitive)
    /// - Measurement records: `"rec[-4]"`
    /// - Sweep bits: `"sweep[6]"`
    /// - Combiner: `"*"`
    ///
    /// # Errors
    ///
    /// Returns an error if `text` does not match any recognised target syntax.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(
    ///     stim::GateTarget::from_target_str("rec[-4]").unwrap(),
    ///     stim::target_rec(-4).unwrap()
    /// );
    /// assert_eq!(
    ///     stim::GateTarget::from_target_str("!Z3").unwrap(),
    ///     stim::target_z(3u32, true).unwrap()
    /// );
    /// ```
    pub fn from_target_str(text: &str) -> Result<Self> {
        if text == "*" {
            return Ok(Self::combiner());
        }

        let (inverted, rest) = match text.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, text),
        };

        if let Some(rest) = rest.strip_prefix("rec[-") {
            let lookback = rest
                .strip_suffix(']')
                .ok_or_else(|| StimError::new(format!("invalid record target: {text}")))?;
            let lookback = parse_u24(lookback)?;
            return Self::rec(-(lookback as i32));
        }

        if let Some(rest) = rest.strip_prefix("sweep[") {
            let index = rest
                .strip_suffix(']')
                .ok_or_else(|| StimError::new(format!("invalid sweep target: {text}")))?;
            return Self::sweep_bit(parse_u24(index)?);
        }

        let mut chars = rest.chars();
        match chars.next() {
            Some(pauli @ ('X' | 'x' | 'Y' | 'y' | 'Z' | 'z')) => {
                let qubit = parse_u24(chars.as_str())?;
                return Self::pauli(qubit, pauli.to_ascii_uppercase(), inverted);
            }
            Some(digit) if digit.is_ascii_digit() => {
                let qubit = parse_u24(rest)?;
                return Self::qubit(qubit, inverted);
            }
            _ => {}
        }

        Err(StimError::new(format!("unrecognized target: {text}")))
    }

    /// Creates a plain qubit target, optionally inverted.
    ///
    /// A qubit target identifies a qubit by its zero-based index. When `inverted` is
    /// `true`, the target is prefixed with `!` in Stim syntax (e.g. `!5`), which
    /// causes the measurement result for that qubit to be flipped.
    ///
    /// # Errors
    ///
    /// Returns an error if `qubit` exceeds the maximum supported target value
    /// (2²⁴ − 1 = 16,777,215).
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(stim::GateTarget::qubit(5, false).unwrap().to_string(), "5");
    /// assert_eq!(stim::GateTarget::qubit(5, true).unwrap().to_string(), "!5");
    /// ```
    pub fn qubit(qubit: u32, inverted: bool) -> Result<Self> {
        ensure_target_value_range(qubit)?;
        Ok(Self {
            data: qubit | (TARGET_INVERTED_BIT * u32::from(inverted)),
        })
    }

    /// Creates an X-basis Pauli target on a qubit.
    ///
    /// In Stim syntax this produces targets like `X3` or `!X3` (when inverted). Used
    /// by instructions that take Pauli-product targets, such as `CORRELATED_ERROR` and
    /// `MPP`.
    ///
    /// # Errors
    ///
    /// Returns an error if `qubit` exceeds the maximum supported target value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x3 = stim::GateTarget::x(3, false).unwrap();
    /// assert!(x3.is_x_target());
    /// assert_eq!(x3.pauli_type(), 'X');
    /// assert_eq!(x3.to_string(), "X3");
    /// ```
    pub fn x(qubit: u32, inverted: bool) -> Result<Self> {
        ensure_target_value_range(qubit)?;
        Ok(Self {
            data: qubit | (TARGET_INVERTED_BIT * u32::from(inverted)) | TARGET_PAULI_X_BIT,
        })
    }

    /// Creates a Y target on a qubit.
    pub fn y(qubit: u32, inverted: bool) -> Result<Self> {
        ensure_target_value_range(qubit)?;
        Ok(Self {
            data: qubit
                | (TARGET_INVERTED_BIT * u32::from(inverted))
                | TARGET_PAULI_X_BIT
                | TARGET_PAULI_Z_BIT,
        })
    }

    /// Creates a Z target on a qubit.
    pub fn z(qubit: u32, inverted: bool) -> Result<Self> {
        ensure_target_value_range(qubit)?;
        Ok(Self {
            data: qubit | (TARGET_INVERTED_BIT * u32::from(inverted)) | TARGET_PAULI_Z_BIT,
        })
    }

    /// Creates a Pauli target from a qubit and Pauli letter.
    pub fn pauli(qubit: u32, pauli: char, inverted: bool) -> Result<Self> {
        match pauli {
            'I' | 'i' => Self::qubit(qubit, inverted),
            'X' | 'x' => Self::x(qubit, inverted),
            'Y' | 'y' => Self::y(qubit, inverted),
            'Z' | 'z' => Self::z(qubit, inverted),
            _ => Err(StimError::new(format!(
                "expected pauli in ['I', 'X', 'Y', 'Z'], got {pauli:?}"
            ))),
        }
    }

    /// Creates a `rec[-k]` target.
    pub fn rec(lookback: i32) -> Result<Self> {
        if !(-(1 << 24)..0).contains(&lookback) {
            return Err(StimError::new("need -16777215 <= lookback <= -1"));
        }
        Ok(Self {
            data: (-lookback as u32) | TARGET_RECORD_BIT,
        })
    }

    /// Creates a `sweep[k]` target.
    pub fn sweep_bit(index: u32) -> Result<Self> {
        ensure_target_value_range(index)?;
        Ok(Self {
            data: index | TARGET_SWEEP_BIT,
        })
    }

    #[must_use]
    pub const fn combiner() -> Self {
        Self {
            data: TARGET_COMBINER,
        }
    }

    #[must_use]
    pub const fn raw_data(self) -> u32 {
        self.data
    }

    /// Returns the target value as a qubit index or negative record lookback.
    #[must_use]
    pub fn value(self) -> i32 {
        let result = (self.data & TARGET_VALUE_MASK) as i32;
        if self.is_measurement_record_target() {
            -result
        } else {
            result
        }
    }

    /// Returns the underlying qubit index when the target is qubit-like.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(stim::GateTarget::new(5u32).qubit_value(), Some(5));
    /// assert_eq!(stim::target_rec(-2).unwrap().qubit_value(), None);
    /// ```
    #[must_use]
    pub fn qubit_value(self) -> Option<u32> {
        if self.has_qubit_value() {
            Some(self.data & TARGET_VALUE_MASK)
        } else {
            None
        }
    }

    /// Returns whether the target carries a qubit index.
    #[must_use]
    pub fn has_qubit_value(self) -> bool {
        (self.data & (TARGET_RECORD_BIT | TARGET_SWEEP_BIT | TARGET_COMBINER)) == 0
    }

    /// Returns whether the target is the `*` combiner.
    #[must_use]
    pub fn is_combiner(self) -> bool {
        self.data == TARGET_COMBINER
    }

    /// Returns whether the target is an X target.
    #[must_use]
    pub fn is_x_target(self) -> bool {
        (self.data & TARGET_PAULI_X_BIT) != 0 && (self.data & TARGET_PAULI_Z_BIT) == 0
    }

    /// Returns whether the target is a Y target.
    #[must_use]
    pub fn is_y_target(self) -> bool {
        (self.data & TARGET_PAULI_X_BIT) != 0 && (self.data & TARGET_PAULI_Z_BIT) != 0
    }

    /// Returns whether the target is a Z target.
    #[must_use]
    pub fn is_z_target(self) -> bool {
        (self.data & TARGET_PAULI_X_BIT) == 0 && (self.data & TARGET_PAULI_Z_BIT) != 0
    }

    /// Returns whether the target is inverted with `!`.
    #[must_use]
    pub fn is_inverted_result_target(self) -> bool {
        (self.data & TARGET_INVERTED_BIT) != 0
    }

    /// Returns whether the target is a measurement-record target.
    #[must_use]
    pub fn is_measurement_record_target(self) -> bool {
        (self.data & TARGET_RECORD_BIT) != 0
    }

    /// Returns whether the target is a plain qubit target.
    #[must_use]
    pub fn is_qubit_target(self) -> bool {
        (self.data
            & (TARGET_PAULI_X_BIT
                | TARGET_PAULI_Z_BIT
                | TARGET_RECORD_BIT
                | TARGET_SWEEP_BIT
                | TARGET_COMBINER))
            == 0
    }

    /// Returns whether the target is a sweep-bit target.
    #[must_use]
    pub fn is_sweep_bit_target(self) -> bool {
        (self.data & TARGET_SWEEP_BIT) != 0
    }

    /// Returns whether the target is a classical-bit target.
    #[must_use]
    pub fn is_classical_bit_target(self) -> bool {
        (self.data & (TARGET_SWEEP_BIT | TARGET_RECORD_BIT)) != 0
    }

    /// Returns whether the target is a Pauli target.
    #[must_use]
    pub fn is_pauli_target(self) -> bool {
        (self.data & (TARGET_PAULI_X_BIT | TARGET_PAULI_Z_BIT)) != 0
    }

    /// Returns the Pauli type of the target (`I`, `X`, `Y`, or `Z`).
    #[must_use]
    pub fn pauli_type(self) -> char {
        match (self.data >> 29) & 3 {
            0 => 'I',
            1 => 'Z',
            2 => 'X',
            3 => 'Y',
            _ => unreachable!(),
        }
    }

    /// Toggles target inversion when that operation is defined.
    ///
    /// # Examples
    ///
    /// ```
    /// let target = stim::GateTarget::new(7u32);
    /// assert_eq!(target.inverted().unwrap().to_string(), "!7");
    /// ```
    pub fn inverted(self) -> Result<Self> {
        if self.data & (TARGET_COMBINER | TARGET_RECORD_BIT | TARGET_SWEEP_BIT) != 0 {
            return Err(StimError::new(format!(
                "target '{}' doesn't have a defined inverse",
                self.target_str()
            )));
        }
        Ok(Self {
            data: self.data ^ TARGET_INVERTED_BIT,
        })
    }

    /// Returns the canonical Stim text form of the target.
    #[must_use]
    pub fn target_str(self) -> String {
        if self.is_combiner() {
            return "*".to_string();
        }

        let mut result = String::new();
        if self.is_inverted_result_target() {
            result.push('!');
        }
        if self.is_x_target() {
            result.push('X');
        } else if self.is_y_target() {
            result.push('Y');
        } else if self.is_z_target() {
            result.push('Z');
        }

        if self.is_measurement_record_target() {
            result.push_str(&format!("rec[{}]", self.value()));
        } else if self.is_sweep_bit_target() {
            result.push_str(&format!("sweep[{}]", self.value()));
        } else {
            result.push_str(&(self.data & TARGET_VALUE_MASK).to_string());
        }
        result
    }
}

impl From<u32> for GateTarget {
    fn from(value: u32) -> Self {
        Self::qubit(value, false).expect("raw qubit target should fit within Stim target range")
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::{
        GateTarget, target_combined_paulis, target_combiner, target_inv, target_pauli, target_rec,
        target_sweep_bit, target_x, target_y, target_z,
    };
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn gate_target_supports_equality_order_hash_and_representation() {
        let qubit_five = GateTarget::new(5u32);
        let same_qubit_five = GateTarget::new(qubit_five);
        let qubit_six = GateTarget::new(6u32);
        let x_five = target_x(5u32, false).expect("X target should build");
        let inverted_qubit_five = target_inv(5u32).expect("inverted qubit target should build");
        let record = target_rec(-4).expect("record target should build");
        let sweep = target_sweep_bit(6).expect("sweep target should build");
        let combiner = target_combiner();
        let inverted_z = target_z(3u32, true).expect("inverted Z target should build");

        assert_eq!(qubit_five, same_qubit_five);
        assert_ne!(qubit_five, qubit_six);
        assert_ne!(qubit_five, x_five);

        let ordered = [x_five, qubit_six, qubit_five, same_qubit_five]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(ordered, vec![qubit_five, qubit_six, x_five]);

        let hashed = [qubit_five, qubit_six, same_qubit_five]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 2);
        assert!(hashed.contains(&qubit_five));
        assert!(hashed.contains(&qubit_six));

        assert_eq!(qubit_five.to_string(), "5");
        assert_eq!(format!("{qubit_five:?}"), "stim::GateTarget(5)");
        assert_eq!(inverted_qubit_five.to_string(), "!5");
        assert_eq!(format!("{inverted_qubit_five:?}"), "stim::target_inv(5)");
        assert_eq!(record.to_string(), "rec[-4]");
        assert_eq!(format!("{record:?}"), "stim::target_rec(-4)");
        assert_eq!(sweep.to_string(), "sweep[6]");
        assert_eq!(format!("{sweep:?}"), "stim::target_sweep_bit(6)");
        assert_eq!(combiner.to_string(), "*");
        assert_eq!(format!("{combiner:?}"), "stim::target_combiner()");
        assert_eq!(x_five.to_string(), "X5");
        assert_eq!(format!("{x_five:?}"), "stim::target_x(5)");
        assert_eq!(inverted_z.to_string(), "!Z3");
        assert_eq!(format!("{inverted_z:?}"), "stim::target_z(3, true)");
    }

    #[test]
    fn gate_target_exposes_value_qubit_value_and_classification() {
        let qubit = GateTarget::new(5u32);
        let inverted_qubit = target_inv(5u32).expect("inverted qubit target should build");
        let measurement_record = target_rec(-4).expect("record target should build");
        let combiner = target_combiner();
        let sweep = target_sweep_bit(6).expect("sweep target should build");
        let x_target = target_x(3u32, false).expect("X target should build");

        assert_eq!(qubit.value(), 5);
        assert_eq!(qubit.qubit_value(), Some(5));
        assert!(qubit.is_qubit_target());
        assert!(!qubit.is_measurement_record_target());
        assert!(!qubit.is_combiner());
        assert!(!qubit.is_classical_bit_target());
        assert!(!qubit.is_pauli_target());

        assert_eq!(inverted_qubit.value(), 5);
        assert_eq!(inverted_qubit.qubit_value(), Some(5));
        assert!(inverted_qubit.is_qubit_target());
        assert!(inverted_qubit.is_inverted_result_target());
        assert!(!inverted_qubit.is_pauli_target());

        assert_eq!(measurement_record.value(), -4);
        assert_eq!(measurement_record.qubit_value(), None);
        assert!(measurement_record.is_measurement_record_target());
        assert!(measurement_record.is_classical_bit_target());
        assert!(!measurement_record.is_qubit_target());
        assert!(!measurement_record.is_combiner());

        assert_eq!(combiner.value(), 0);
        assert_eq!(combiner.qubit_value(), None);
        assert!(combiner.is_combiner());
        assert!(!combiner.is_qubit_target());
        assert!(!combiner.is_measurement_record_target());
        assert!(!combiner.is_classical_bit_target());

        assert_eq!(sweep.value(), 6);
        assert_eq!(sweep.qubit_value(), None);
        assert!(sweep.is_sweep_bit_target());
        assert!(sweep.is_classical_bit_target());
        assert!(!sweep.is_qubit_target());
        assert!(!sweep.is_measurement_record_target());
        assert!(!sweep.is_combiner());

        assert_eq!(x_target.value(), 3);
        assert_eq!(x_target.qubit_value(), Some(3));
        assert!(x_target.is_x_target());
        assert!(x_target.is_pauli_target());
        assert!(!x_target.is_qubit_target());
        assert!(!x_target.is_measurement_record_target());
        assert!(!x_target.is_combiner());
    }

    #[test]
    fn gate_target_parses_representative_strings() {
        assert_eq!(
            "7".parse::<GateTarget>()
                .expect("qubit target should parse"),
            GateTarget::new(7u32)
        );
        assert_eq!(
            "!7".parse::<GateTarget>()
                .expect("inverted qubit target should parse"),
            target_inv(7u32).expect("inverted qubit target should build")
        );
        assert_eq!(
            "rec[-4]"
                .parse::<GateTarget>()
                .expect("record target should parse"),
            target_rec(-4).expect("record target should build")
        );
        assert_eq!(
            "sweep[6]"
                .parse::<GateTarget>()
                .expect("sweep target should parse"),
            target_sweep_bit(6).expect("sweep target should build")
        );
        assert_eq!(
            "X5".parse::<GateTarget>().expect("X target should parse"),
            target_x(5u32, false).expect("X target should build")
        );
        assert_eq!(
            "!z3"
                .parse::<GateTarget>()
                .expect("inverted lowercase Z target should parse"),
            target_z(3u32, true).expect("inverted Z target should build")
        );
        assert_eq!(
            "*".parse::<GateTarget>().expect("combiner should parse"),
            target_combiner()
        );
    }

    #[test]
    fn target_rec_accepts_valid_lookbacks_and_roundtrips_through_text() {
        let latest = target_rec(-1).expect("latest measurement record target should succeed");
        let older = target_rec(-15).expect("older measurement record target should succeed");

        assert!(latest.is_measurement_record_target());
        assert!(latest.is_classical_bit_target());
        assert!(!latest.is_sweep_bit_target());
        assert!(!latest.is_combiner());
        assert!(!latest.is_inverted_result_target());
        assert_eq!(latest.value(), -1);
        assert_eq!(latest.qubit_value(), None);
        assert_eq!(latest.pauli_type(), 'I');
        assert_eq!(latest.to_string(), "rec[-1]");
        assert_eq!(format!("{latest:?}"), "stim::target_rec(-1)");

        assert_eq!(older.value(), -15);
        assert_eq!(older.to_string(), "rec[-15]");
        assert_eq!(
            "rec[-15]"
                .parse::<GateTarget>()
                .expect("record target should parse"),
            older
        );
    }

    #[test]
    fn target_rec_rejects_non_negative_and_out_of_range_lookbacks() {
        for invalid in [0, 1, -(1 << 30)] {
            let error = target_rec(invalid).expect_err("invalid lookback should fail");
            assert!(error.message().contains("need -16777215 <= lookback <= -1"));
        }
    }

    #[test]
    fn target_sweep_bit_accepts_u24_values_and_roundtrips_through_text() {
        let first = target_sweep_bit(0).expect("first sweep bit should succeed");
        let indexed = target_sweep_bit(9).expect("representative sweep bit should succeed");
        let last =
            target_sweep_bit(16_777_215).expect("maximum supported sweep bit should succeed");

        assert!(first.is_sweep_bit_target());
        assert!(first.is_classical_bit_target());
        assert!(!first.is_measurement_record_target());
        assert!(!first.is_combiner());
        assert!(!first.is_inverted_result_target());
        assert_eq!(first.value(), 0);
        assert_eq!(first.qubit_value(), None);
        assert_eq!(first.pauli_type(), 'I');
        assert_eq!(first.to_string(), "sweep[0]");
        assert_eq!(format!("{first:?}"), "stim::target_sweep_bit(0)");

        assert_eq!(indexed.to_string(), "sweep[9]");
        assert_eq!(
            "sweep[9]"
                .parse::<GateTarget>()
                .expect("sweep target should parse"),
            indexed
        );

        assert_eq!(last.value(), 16_777_215);
    }

    #[test]
    fn target_sweep_bit_rejects_values_outside_the_supported_u24_range() {
        let error = target_sweep_bit(16_777_216).expect_err("out-of-range sweep bit should fail");

        assert!(
            error
                .message()
                .contains("target value 16777216 exceeds maximum 16777215")
        );
    }

    #[test]
    fn target_combiner_has_expected_representation_and_inverse_failures() {
        let combiner = target_combiner();
        let inverted = target_inv(7).expect("qubit targets should invert");

        assert!(combiner.is_combiner());
        assert!(!combiner.is_classical_bit_target());
        assert!(!combiner.is_measurement_record_target());
        assert!(!combiner.is_sweep_bit_target());
        assert!(!combiner.is_qubit_target());
        assert_eq!(combiner.value(), 0);
        assert_eq!(combiner.qubit_value(), None);
        assert_eq!(combiner.to_string(), "*");
        assert_eq!(format!("{combiner:?}"), "stim::target_combiner()");
        assert_eq!(
            "*".parse::<GateTarget>().expect("combiner should parse"),
            combiner
        );

        assert!(inverted.is_inverted_result_target());
        assert_eq!(inverted.to_string(), "!7");
        assert_eq!(
            target_inv(inverted).expect("double inversion should unwrap"),
            GateTarget::new(7)
        );

        let rec_error = target_inv(target_rec(-2).expect("record target should build"))
            .expect_err("record targets should not invert");
        let sweep_error = target_inv(target_sweep_bit(3).expect("sweep target should build"))
            .expect_err("sweep targets should not invert");
        let combiner_error = target_inv(combiner).expect_err("combiner should not invert");

        assert!(
            rec_error
                .message()
                .contains("target 'rec[-2]' doesn't have a defined inverse")
        );
        assert!(
            sweep_error
                .message()
                .contains("target 'sweep[3]' doesn't have a defined inverse")
        );
        assert!(
            combiner_error
                .message()
                .contains("target '*' doesn't have a defined inverse")
        );
    }

    #[test]
    fn target_xyz_helpers_report_expected_pauli_types() {
        let x = target_x(2, false).expect("X target should build");
        let y = target_y(3, true).expect("Y target should build");
        let z = target_z(5, false).expect("Z target should build");

        assert_eq!(x.pauli_type(), 'X');
        assert_eq!(y.pauli_type(), 'Y');
        assert_eq!(z.pauli_type(), 'Z');

        assert!(x.is_x_target());
        assert!(y.is_y_target());
        assert!(z.is_z_target());

        assert_eq!(x.to_string(), "X2");
        assert_eq!(y.to_string(), "!Y3");
        assert_eq!(z.to_string(), "Z5");
    }

    #[test]
    fn target_pauli_matches_specific_helpers_and_inversion_propagates() {
        let inverted_qubit = target_inv(7u32).expect("qubit inversion should succeed");

        assert_eq!(
            target_x(inverted_qubit, false).unwrap(),
            target_x(7, true).unwrap()
        );
        assert_eq!(
            target_y(inverted_qubit, false).unwrap(),
            target_y(7, true).unwrap()
        );
        assert_eq!(
            target_z(inverted_qubit, false).unwrap(),
            target_z(7, true).unwrap()
        );

        assert_eq!(
            target_x(inverted_qubit, true).unwrap(),
            target_pauli(7, 'X', false).unwrap()
        );
        assert_eq!(
            target_y(inverted_qubit, true).unwrap(),
            target_pauli(7, 'Y', false).unwrap()
        );
        assert_eq!(
            target_z(inverted_qubit, true).unwrap(),
            target_pauli(7, 'Z', false).unwrap()
        );

        assert_eq!(
            target_pauli(7, 'X', true).unwrap(),
            target_x(7, true).unwrap()
        );
        assert_eq!(
            target_pauli(7, 'Y', false).unwrap(),
            target_y(7, false).unwrap()
        );
        assert_eq!(
            target_pauli(7, 'Z', true).unwrap(),
            target_z(7, true).unwrap()
        );
    }

    #[test]
    fn target_xyz_reject_non_qubit_inputs() {
        let record = target_rec(-1).expect("record target should build");
        let sweep = target_sweep_bit(4).expect("sweep target should build");
        let combiner = target_combiner();

        let x_err = target_x(record, false).expect_err("record target should be rejected");
        assert_eq!(
            x_err.to_string(),
            "result of stim::target_x(rec[-1]) is not defined"
        );

        let y_err = target_y(sweep, true).expect_err("sweep target should be rejected");
        assert_eq!(
            y_err.to_string(),
            "result of stim::target_y(sweep[4]) is not defined"
        );

        let z_err = target_z(combiner, false).expect_err("combiner target should be rejected");
        assert_eq!(
            z_err.to_string(),
            "result of stim::target_z(*) is not defined"
        );
    }

    #[test]
    fn target_pauli_rejects_unknown_pauli_type() {
        let err = target_pauli(1, 'Q', false).expect_err("unknown pauli should be rejected");

        assert_eq!(
            err.to_string(),
            "expected pauli in ['I', 'X', 'Y', 'Z'], got 'Q'"
        );
    }

    #[test]
    fn target_combined_paulis_inserts_combiners_and_applies_sign_rules() {
        let x5 = target_x(5, false).expect("X target should be constructible");
        let z9 = target_z(9, false).expect("Z target should be constructible");

        assert_eq!(
            target_combined_paulis(&[x5, z9], false).expect("pauli product should combine"),
            vec![x5, target_combiner(), z9]
        );
        assert_eq!(
            target_combined_paulis(&[x5, z9], true)
                .expect("explicit inversion should flip the product sign"),
            vec![
                target_x(5, true).expect("inverted X target should be constructible"),
                target_combiner(),
                z9,
            ]
        );

        assert_eq!(
            target_combined_paulis(
                &[
                    target_x(5, false).expect("X target should be constructible"),
                    target_z(9, true).expect("inverted Z target should be constructible"),
                ],
                false,
            )
            .expect("inverted member should fold into overall sign"),
            vec![
                target_x(5, true).expect("inverted X target should be constructible"),
                target_combiner(),
                target_z(9, false).expect("Z target should be constructible"),
            ]
        );
    }

    #[test]
    fn target_combined_paulis_rejects_identity_and_non_pauli_targets() {
        let identity =
            target_combined_paulis(&[], false).expect_err("empty pauli product should be rejected");
        assert!(
            identity.message().contains("identity pauli product"),
            "unexpected identity error: {identity}"
        );

        let qubit = target_combined_paulis(&[GateTarget::new(5u32)], false)
            .expect_err("plain qubit targets are not pauli targets");
        assert!(
            qubit.message().contains("expected pauli targets"),
            "unexpected qubit error: {qubit}"
        );

        let record = target_combined_paulis(
            &[target_rec(-2).expect("record target should be constructible")],
            false,
        )
        .expect_err("record targets are not pauli targets");
        assert!(
            record.message().contains("expected pauli targets"),
            "unexpected record error: {record}"
        );
    }
}

impl Display for GateTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.target_str())
    }
}

impl fmt::Debug for GateTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_combiner() {
            return f.write_str("stim::target_combiner()");
        }
        if self.is_qubit_target() && !self.is_inverted_result_target() {
            return write!(f, "stim::GateTarget({})", self.value());
        }
        if self.is_qubit_target() && self.is_inverted_result_target() {
            return write!(f, "stim::target_inv({})", self.value());
        }
        if self.is_measurement_record_target() {
            return write!(f, "stim::target_rec({})", self.value());
        }
        if self.is_sweep_bit_target() {
            return write!(f, "stim::target_sweep_bit({})", self.value());
        }

        let prefix = match self.pauli_type() {
            'X' => "target_x",
            'Y' => "target_y",
            'Z' => "target_z",
            _ => unreachable!(),
        };
        if self.is_inverted_result_target() {
            write!(f, "stim::{prefix}({}, true)", self.value())
        } else {
            write!(f, "stim::{prefix}({})", self.value())
        }
    }
}

impl FromStr for GateTarget {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_target_str(s)
    }
}

/// Creates a measurement-record target like `rec[-k]`.
///
/// # Examples
///
/// ```
/// assert_eq!(stim::target_rec(-4).unwrap().to_string(), "rec[-4]");
/// ```
pub fn target_rec(lookback_index: i32) -> Result<GateTarget> {
    GateTarget::rec(lookback_index)
}

/// Inverts a qubit or Pauli target.
///
/// # Examples
///
/// ```
/// assert_eq!(stim::target_inv(stim::GateTarget::new(7u32)).unwrap().to_string(), "!7");
/// ```
pub fn target_inv(target: impl Into<GateTarget>) -> Result<GateTarget> {
    target.into().inverted()
}

/// Creates an X-basis target.
///
/// # Examples
///
/// ```
/// assert_eq!(stim::target_x(5u32, false).unwrap().to_string(), "X5");
/// ```
pub fn target_x(target: impl Into<GateTarget>, invert: bool) -> Result<GateTarget> {
    target_to_pauli(target.into(), 'X', invert)
}

/// Creates a Y-basis target.
pub fn target_y(target: impl Into<GateTarget>, invert: bool) -> Result<GateTarget> {
    target_to_pauli(target.into(), 'Y', invert)
}

/// Creates a Z-basis target.
pub fn target_z(target: impl Into<GateTarget>, invert: bool) -> Result<GateTarget> {
    target_to_pauli(target.into(), 'Z', invert)
}

/// Returns the target combiner used inside `MPP` products.
#[must_use]
pub fn target_combiner() -> GateTarget {
    GateTarget::combiner()
}

/// Creates a `sweep[k]` target.
pub fn target_sweep_bit(index: u32) -> Result<GateTarget> {
    GateTarget::sweep_bit(index)
}

/// Creates a Pauli target from a qubit index and Pauli letter.
pub fn target_pauli(qubit_index: u32, pauli: char, invert: bool) -> Result<GateTarget> {
    GateTarget::pauli(qubit_index, pauli, invert)
}

/// Builds a combined Pauli product target list suitable for `MPP`.
///
/// # Examples
///
/// ```
/// let targets = stim::target_combined_paulis(
///     &[
///         stim::target_x(1u32, false).unwrap(),
///         stim::target_z(2u32, false).unwrap(),
///     ],
///     false,
/// )
/// .unwrap();
/// assert_eq!(
///     targets
///         .iter()
///         .map(ToString::to_string)
///         .collect::<Vec<_>>(),
///     vec!["X1".to_string(), "*".to_string(), "Z2".to_string()]
/// );
/// ```
pub fn target_combined_paulis(paulis: &[GateTarget], invert: bool) -> Result<Vec<GateTarget>> {
    let mut result = Vec::new();
    let mut invert = invert;
    for target in paulis.iter().copied() {
        if target.pauli_type() == 'I' {
            return Err(StimError::new(format!(
                "expected pauli targets but got '{target}'"
            )));
        }
        let mut target = target;
        if target.is_inverted_result_target() {
            invert ^= true;
            target = target.inverted()?;
        }
        result.push(target);
        result.push(GateTarget::combiner());
    }
    if result.is_empty() {
        return Err(StimError::new("identity pauli product is not allowed"));
    }
    result.pop();
    if invert {
        result[0] = result[0].inverted()?;
    }
    Ok(result)
}

fn ensure_target_value_range(value: u32) -> Result<()> {
    if value == (value & TARGET_VALUE_MASK) {
        Ok(())
    } else {
        Err(StimError::new(format!(
            "target value {value} exceeds maximum {}",
            TARGET_VALUE_MASK
        )))
    }
}

fn parse_u24(text: &str) -> Result<u32> {
    let value: u32 = text
        .parse()
        .map_err(|_| StimError::new(format!("expected integer target value, got {text:?}")))?;
    ensure_target_value_range(value)?;
    Ok(value)
}

fn target_to_pauli(target: GateTarget, pauli: char, invert: bool) -> Result<GateTarget> {
    if !target.is_qubit_target() {
        return Err(StimError::new(format!(
            "result of stim::target_{}({target}) is not defined",
            pauli.to_ascii_lowercase()
        )));
    }
    GateTarget::pauli(
        target
            .qubit_value()
            .expect("qubit target should have value"),
        pauli,
        target.is_inverted_result_target() ^ invert,
    )
}
