use std::fmt::{self, Display, Formatter};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg};
use std::str::FromStr;

use ndarray::{Array1, ArrayView1};

/// A single Pauli operator value accepted by [`PauliString::set`].
///
/// `PauliValue` provides a type-safe way to specify individual Pauli operators
/// when mutating entries of a [`PauliString`]. It supports two encodings that
/// mirror the conventions used throughout the Stim ecosystem:
///
/// - **Numeric codes** (`0` = I, `1` = X, `2` = Y, `3` = Z) — the same
///   integer encoding returned by [`PauliString::get`] and used internally in
///   the XZ bit representation.
/// - **Character symbols** (`'I'`, `'_'`, `'X'`, `'Y'`, `'Z'`) — the
///   human-readable characters used in Pauli string text notation. The
///   underscore `'_'` is an alias for `'I'` (identity).
///
/// You rarely need to construct `PauliValue` directly. Thanks to the blanket
/// `From<u8>` and `From<char>` implementations, you can pass a `u8` or `char`
/// directly to any method that expects `impl Into<PauliValue>`.
///
/// # Examples
///
/// ```
/// let mut p = stim::PauliString::new(3);
///
/// // Set via numeric code
/// p.set(0, 1u8).unwrap();   // qubit 0 → X
///
/// // Set via character symbol
/// p.set(1, 'Y').unwrap();   // qubit 1 → Y
/// p.set(2, '_').unwrap();   // qubit 2 → I (identity)
///
/// assert_eq!(p.to_string(), "+XY_");
/// ```
pub enum PauliValue {
    /// A numeric Pauli code using the XZ encoding: `0` = I (identity),
    /// `1` = X, `2` = Y, `3` = Z. Values outside `0..=3` will produce
    /// an error when passed to [`PauliString::set`].
    Code(u8),
    /// A character Pauli symbol. Accepted characters are `'I'`, `'_'`
    /// (both meaning identity), `'X'`, `'Y'`, and `'Z'`. Any other
    /// character will produce an error when passed to [`PauliString::set`].
    Symbol(char),
}

/// The unit complex phase carried by a [`PauliString`].
///
/// A Pauli string's global phase is always one of the four unit complex
/// numbers `{+1, +i, −1, −i}`. This enum represents that phase factor and
/// provides conversions to [`crate::Complex32`] for use in matrix
/// computations.
///
/// In the Stim model, every [`PauliString`] stores its phase as the
/// combination of a real sign (`±1`) and an imaginary flag. The four
/// resulting possibilities are exactly the four variants of `PauliPhase`.
///
/// When the phase is real (`+1` or `−1`), the Pauli string is Hermitian
/// (i.e., it equals its own conjugate transpose). Imaginary phases (`+i`
/// or `−i`) arise from products of non-commuting Paulis — for example,
/// `X · Y = iZ`.
///
/// # Relationship to `PauliString` methods
///
/// - [`PauliString::phase`] returns this enum.
/// - [`PauliString::complex_phase`] returns the equivalent
///   [`crate::Complex32`] value.
/// - [`PauliString::has_imaginary_phase`] is a convenience shortcut for
///   [`PauliPhase::is_imaginary`].
///
/// # Examples
///
/// ```
/// use stim::{PauliPhase, PauliString};
///
/// let p = PauliString::from_text("-iXYZ").unwrap();
/// assert_eq!(p.phase(), PauliPhase::NegativeImaginary);
/// assert!(p.has_imaginary_phase());
///
/// let q = PauliString::from_text("+X").unwrap();
/// assert_eq!(q.phase(), PauliPhase::Positive);
/// assert!(!q.has_imaginary_phase());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliPhase {
    /// The phase `+1`. The Pauli string is Hermitian with positive sign.
    Positive,
    /// The phase `+i`. The Pauli string is non-Hermitian.
    PositiveImaginary,
    /// The phase `-1`. The Pauli string is Hermitian with negative sign.
    Negative,
    /// The phase `-i`. The Pauli string is non-Hermitian.
    NegativeImaginary,
}

impl PauliPhase {
    /// Returns the phase as a complex number.
    ///
    /// Converts this `PauliPhase` variant into its corresponding unit complex
    /// value as a [`crate::Complex32`]:
    ///
    /// | Variant                | Complex value |
    /// |------------------------|---------------|
    /// | `Positive`             | `1 + 0i`      |
    /// | `PositiveImaginary`    | `0 + 1i`      |
    /// | `Negative`             | `-1 + 0i`     |
    /// | `NegativeImaginary`    | `0 - 1i`      |
    ///
    /// This is useful when you need the numeric phase factor for matrix
    /// arithmetic — for example, when constructing the unitary matrix of
    /// a Pauli string.
    ///
    /// # Examples
    ///
    /// ```
    /// use stim::PauliPhase;
    ///
    /// assert_eq!(
    ///     PauliPhase::PositiveImaginary.as_complex32(),
    ///     stim::Complex32::new(0.0, 1.0),
    /// );
    /// assert_eq!(
    ///     PauliPhase::Negative.as_complex32(),
    ///     stim::Complex32::new(-1.0, 0.0),
    /// );
    /// ```
    #[must_use]
    pub fn as_complex32(self) -> crate::Complex32 {
        match self {
            Self::Positive => crate::Complex32::new(1.0, 0.0),
            Self::PositiveImaginary => crate::Complex32::new(0.0, 1.0),
            Self::Negative => crate::Complex32::new(-1.0, 0.0),
            Self::NegativeImaginary => crate::Complex32::new(0.0, -1.0),
        }
    }

    /// Returns whether the phase is imaginary (`±i`).
    ///
    /// A Pauli string with an imaginary phase is non-Hermitian, meaning it
    /// does not equal its own conjugate transpose. Imaginary phases arise
    /// naturally when multiplying non-commuting Pauli operators (e.g.,
    /// `X · Y = iZ`).
    ///
    /// Returns `true` for [`PauliPhase::PositiveImaginary`] and
    /// [`PauliPhase::NegativeImaginary`]; `false` for [`PauliPhase::Positive`]
    /// and [`PauliPhase::Negative`].
    ///
    /// # Examples
    ///
    /// ```
    /// use stim::PauliPhase;
    ///
    /// assert!(PauliPhase::PositiveImaginary.is_imaginary());
    /// assert!(PauliPhase::NegativeImaginary.is_imaginary());
    /// assert!(!PauliPhase::Positive.is_imaginary());
    /// assert!(!PauliPhase::Negative.is_imaginary());
    /// ```
    #[must_use]
    pub fn is_imaginary(self) -> bool {
        matches!(self, Self::PositiveImaginary | Self::NegativeImaginary)
    }
}

impl From<u8> for PauliValue {
    fn from(value: u8) -> Self {
        Self::Code(value)
    }
}

impl From<char> for PauliValue {
    fn from(value: char) -> Self {
        Self::Symbol(value)
    }
}

/// The type of Clifford operation used to conjugate a [`PauliString`].
///
/// In the stabilizer formalism, Clifford operations act on Pauli strings by
/// *conjugation*: given a Clifford unitary *U* and a Pauli string *P*, the
/// conjugated Pauli string is *U · P · U†*. This is the central operation
/// that makes Pauli tracking and stabilizer simulation efficient — rather
/// than manipulating exponentially large state vectors, Clifford operations
/// simply map Pauli strings to other Pauli strings.
///
/// `PauliStringConjugation` enumerates the three sources from which a
/// Clifford conjugation can be performed:
///
/// - **Circuit**: Apply every instruction in a [`crate::Circuit`] (the
///   circuit must be entirely Clifford — no noise or measurements).
/// - **Instruction**: Apply a single [`crate::CircuitInstruction`].
/// - **Tableau**: Apply a [`crate::Tableau`] to specific target qubits.
///
/// You rarely need to construct this enum yourself. It implements
/// [`From`] conversions for each variant, so you can pass a `&Circuit`,
/// `&CircuitInstruction`, or `(&Tableau, &[usize])` directly to
/// [`PauliString::after`] and [`PauliString::before`].
///
/// # Forward vs. backward conjugation
///
/// - [`PauliString::after`] performs *forward* (Heisenberg-picture)
///   conjugation: "what does this Pauli string become *after* the
///   operation is applied?"
/// - [`PauliString::before`] performs *backward* conjugation: "what
///   Pauli string *before* the operation is equivalent to this one
///   after?"
///
/// # Examples
///
/// ```
/// let p = stim::PauliString::from_text("_XYZ").unwrap();
/// let cz = stim::Tableau::from_named_gate("CZ").unwrap();
///
/// // Pass a (&Tableau, &[usize]) tuple — it converts automatically.
/// let result = p.after((&cz, &[0, 1][..])).unwrap();
/// assert_eq!(result, stim::PauliString::from_text("+ZXYZ").unwrap());
/// ```
pub enum PauliStringConjugation<'a> {
    /// Conjugate through an entire Clifford circuit. The circuit must not
    /// contain noise channels or measurements.
    Circuit(&'a crate::Circuit),
    /// Conjugate through a single Clifford instruction (e.g., a gate
    /// applied to specific qubits).
    Instruction(&'a crate::CircuitInstruction),
    /// Conjugate through a Clifford tableau applied to an explicit set of
    /// target qubit indices.
    Tableau {
        /// The Clifford tableau to conjugate through.
        tableau: &'a crate::Tableau,
        /// The qubit indices the tableau acts on, in order. The length
        /// must match the number of qubits in the tableau.
        targets: &'a [usize],
    },
}

impl<'a> From<&'a crate::Circuit> for PauliStringConjugation<'a> {
    fn from(value: &'a crate::Circuit) -> Self {
        Self::Circuit(value)
    }
}

impl<'a> From<&'a crate::CircuitInstruction> for PauliStringConjugation<'a> {
    fn from(value: &'a crate::CircuitInstruction) -> Self {
        Self::Instruction(value)
    }
}

impl<'a> From<(&'a crate::Tableau, &'a [usize])> for PauliStringConjugation<'a> {
    fn from(value: (&'a crate::Tableau, &'a [usize])) -> Self {
        Self::Tableau {
            tableau: value.0,
            targets: value.1,
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
/// A signed tensor product of Pauli operators (e.g., `+X ⊗ Y ⊗ Z` or `-iX ⊗ I ⊗ Z`).
///
/// A `PauliString` represents a collection of single-qubit Pauli operators
/// (I, X, Y, Z) applied to a sequence of qubits, together with a global phase
/// factor drawn from the set `{+1, +i, −1, −i}`. Pauli strings are fundamental
/// objects in the **stabilizer formalism** of quantum error correction: every
/// stabilizer code is defined by a group of commuting Pauli strings (the
/// *stabilizer generators*), and the errors that the code can detect and
/// correct are also described as Pauli strings.
///
/// # Representation
///
/// Internally, each qubit's Pauli operator is encoded using two bits (the "XZ
/// encoding"):
///
/// | Pauli | X bit | Z bit |
/// |-------|-------|-------|
/// | I     | 0     | 0     |
/// | X     | 1     | 0     |
/// | Y     | 1     | 1     |
/// | Z     | 0     | 1     |
///
/// This encoding is efficient and naturally captures the algebraic structure
/// of the Pauli group. The global phase is stored as a real sign (`±1`)
/// combined with an imaginary flag, yielding the four unit complex phases
/// represented by [`PauliPhase`].
///
/// # Text notation
///
/// Pauli strings can be created from a human-readable text format. A text
/// representation begins with an optional sign prefix (`+`, `-`, `i`, `+i`,
/// or `-i`) followed by a sequence of Pauli characters (`I`, `_`, `X`, `Y`,
/// `Z`), where `_` is an alias for `I` (identity). For example:
///
/// - `"+XYZ"` — a 3-qubit Pauli string with phase `+1`
/// - `"-iX_Z"` — a 3-qubit Pauli string with phase `−i`
/// - `"XXXX"` — phase defaults to `+1` when omitted
///
/// # Key operations
///
/// - **Creation**: [`PauliString::new`] (identity), [`PauliString::from_text`]
///   (parse from string), [`PauliString::random`] (uniformly random),
///   [`PauliString::from_ndarray`] / [`PauliString::from_ndarray_bit_packed`]
///   (from X/Z indicator arrays), [`PauliString::from_unitary_matrix`]
///   (infer from a unitary matrix).
/// - **Indexing**: [`PauliString::get`] and [`PauliString::set`] read and
///   write individual Pauli entries using numeric codes or character symbols.
///   [`PauliString::slice`] extracts sub-strings with Python-style slice
///   semantics.
/// - **Algebra**: The `+` operator computes tensor products (concatenation
///   with sign multiplication). The unary `-` operator negates the phase.
///   The `*` operator with a `u64` repeats the string (tensor power) with
///   appropriate phase arithmetic.
/// - **Conjugation**: [`PauliString::after`] and [`PauliString::before`]
///   conjugate the Pauli string through Clifford operations (circuits,
///   instructions, or tableaux), which is the core operation of Pauli
///   tracking in stabilizer simulation.
/// - **Queries**: [`PauliString::weight`] counts non-identity terms,
///   [`PauliString::commutes`] checks commutativity with another Pauli
///   string, and [`PauliString::pauli_indices`] finds positions of
///   specified Pauli types.
/// - **Conversion**: [`PauliString::to_unitary_matrix`] builds the full
///   unitary matrix, [`PauliString::to_tableau`] wraps the Pauli string
///   as a Clifford tableau, and [`PauliString::to_ndarray`] /
///   [`PauliString::to_ndarray_bit_packed`] export to X/Z bit arrays.
///
/// # Examples
///
/// ```
/// use stim::PauliString;
///
/// // Create from text and inspect properties
/// let p = PauliString::from_text("-XYZ").unwrap();
/// assert_eq!(p.num_qubits(), 3);
/// assert_eq!(p.weight(), 3);
/// assert_eq!(p.phase(), stim::PauliPhase::Negative);
///
/// // Tensor product via the + operator
/// let a = PauliString::from_text("X").unwrap();
/// let b = PauliString::from_text("YZ").unwrap();
/// assert_eq!((a + b).to_string(), "+XYZ");
///
/// // Identity Pauli string
/// let id = PauliString::new(5);
/// assert_eq!(id.to_string(), "+_____");
/// assert_eq!(id.weight(), 0);
/// ```
pub struct PauliString {
    pub(crate) inner: stim_cxx::PauliString,
    pub(crate) imag: bool,
}

impl PauliString {
    fn from_phase_code_and_body(phase: u8, body: &str) -> Self {
        let prefix = if phase & 2 != 0 { "-" } else { "+" };
        let inner = stim_cxx::PauliString::from_text(&format!("{prefix}{body}"))
            .expect("internal PauliString reconstruction should succeed");
        Self {
            inner,
            imag: phase & 1 != 0,
        }
    }

    fn phase_code(&self) -> u8 {
        (u8::from(self.imag)) | (u8::from(self.sign() < 0) << 1)
    }

    fn phase_from_code(code: u8) -> PauliPhase {
        match code & 3 {
            0 => PauliPhase::Positive,
            1 => PauliPhase::PositiveImaginary,
            2 => PauliPhase::Negative,
            _ => PauliPhase::NegativeImaginary,
        }
    }

    fn unit_phase_code(value: crate::Complex32) -> crate::Result<u8> {
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        if close(value.re, 1.0) && close(value.im, 0.0) {
            Ok(0)
        } else if close(value.re, 0.0) && close(value.im, 1.0) {
            Ok(1)
        } else if close(value.re, -1.0) && close(value.im, 0.0) {
            Ok(2)
        } else if close(value.re, 0.0) && close(value.im, -1.0) {
            Ok(3)
        } else {
            Err(crate::StimError::new("divisor not in (1, -1, 1j, -1j)"))
        }
    }

    pub(crate) fn from_real_sign_and_body(sign: i32, body: &str) -> Self {
        Self::from_phase_code_and_body(if sign < 0 { 2 } else { 0 }, body)
    }

    pub(crate) fn dense_body_text(&self) -> String {
        let text = self.inner.to_text();
        text.strip_prefix(['+', '-']).unwrap_or(&text).to_string()
    }

    /// Creates an identity Pauli string over `num_qubits` qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::new(3);
    /// assert_eq!(p.to_string(), "+___");
    /// assert_eq!(p.weight(), 0);
    /// ```
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            inner: stim_cxx::PauliString::new(num_qubits),
            imag: false,
        }
    }

    /// Parses a Pauli string from text.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("-iXYZ").unwrap();
    /// assert_eq!(p.phase(), stim::PauliPhase::NegativeImaginary);
    /// assert_eq!(p.to_string(), "-iXYZ");
    /// ```
    pub fn from_text(text: &str) -> crate::Result<Self> {
        let text = text.trim();
        let (phase, body) = if let Some(rest) = text.strip_prefix("-i") {
            (3, rest)
        } else if let Some(rest) = text.strip_prefix("+i") {
            (1, rest)
        } else if let Some(rest) = text.strip_prefix('i') {
            (1, rest)
        } else if let Some(rest) = text.strip_prefix('-') {
            (2, rest)
        } else if let Some(rest) = text.strip_prefix('+') {
            (0, rest)
        } else {
            (0, text)
        };
        let body = if body.is_empty() { "" } else { body };
        let inner = if body.is_empty() {
            stim_cxx::PauliString::new(0)
        } else {
            let prefix = if phase & 2 != 0 { "-" } else { "+" };
            stim_cxx::PauliString::from_text(&format!("{prefix}{body}"))
                .map_err(crate::StimError::from)?
        };
        Ok(Self {
            inner,
            imag: phase & 1 != 0,
        })
    }

    /// Returns a random Pauli string of the requested length.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::random(4);
    /// assert_eq!(p.num_qubits(), 4);
    /// ```
    #[must_use]
    pub fn random(num_qubits: usize) -> Self {
        Self {
            inner: stim_cxx::PauliString::random(num_qubits),
            imag: false,
        }
    }

    /// Iterates over all Pauli strings matching the requested filters.
    ///
    /// # Examples
    ///
    /// ```
    /// let all: Vec<_> = stim::PauliString::iter_all(1, 0, 1, "XYZ").collect();
    /// assert_eq!(all.len(), 4);
    /// assert_eq!(all[0], stim::PauliString::from_text("+_").unwrap());
    /// ```
    #[must_use]
    pub fn iter_all(
        num_qubits: usize,
        min_weight: usize,
        max_weight: usize,
        allowed_paulis: &str,
    ) -> crate::PauliStringIterator {
        let allow_x = allowed_paulis.contains(['x', 'X']);
        let allow_y = allowed_paulis.contains(['y', 'Y']);
        let allow_z = allowed_paulis.contains(['z', 'Z']);
        crate::PauliStringIterator {
            inner: stim_cxx::PauliString::iter_all(
                num_qubits, min_weight, max_weight, allow_x, allow_y, allow_z,
            ),
        }
    }

    /// Returns the number of qubits described by the Pauli string.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of qubits described by the Pauli string.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_qubits()
    }

    /// Returns whether the Pauli string is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of non-identity Paulis.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(stim::PauliString::from_text("+_XYZ").unwrap().weight(), 3);
    /// ```
    #[must_use]
    pub fn weight(&self) -> usize {
        self.inner.weight()
    }

    /// Exports the Pauli string as unpacked X/Z indicator vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("_XYZ").unwrap();
    /// let (xs, zs) = p.to_ndarray();
    /// assert_eq!(xs, ndarray::array![false, true, true, false]);
    /// assert_eq!(zs, ndarray::array![false, false, true, true]);
    /// ```
    #[must_use]
    pub fn to_ndarray(&self) -> (Array1<bool>, Array1<bool>) {
        let mut xs = Vec::with_capacity(self.len());
        let mut zs = Vec::with_capacity(self.len());
        for k in 0..self.len() {
            let p = self
                .get(k as isize)
                .expect("in-bounds PauliString indexing should succeed");
            xs.push(matches!(p, 1 | 2));
            zs.push(matches!(p, 2 | 3));
        }
        (Array1::from_vec(xs), Array1::from_vec(zs))
    }

    /// Exports the Pauli string as bit-packed X/Z indicator vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("_XYZ___XYXZYZ").unwrap();
    /// let (xs, zs) = p.to_ndarray_bit_packed();
    /// assert_eq!(xs, ndarray::array![0x86, 0x0B]);
    /// assert_eq!(zs, ndarray::array![0x0C, 0x1D]);
    /// ```
    #[must_use]
    pub fn to_ndarray_bit_packed(&self) -> (Array1<u8>, Array1<u8>) {
        let n = self.len();
        let mut xs = vec![0u8; n.div_ceil(8)];
        let mut zs = vec![0u8; n.div_ceil(8)];
        for k in 0..n {
            let p = self
                .get(k as isize)
                .expect("in-bounds PauliString indexing should succeed");
            if matches!(p, 1 | 2) {
                xs[k / 8] |= 1 << (k % 8);
            }
            if matches!(p, 2 | 3) {
                zs[k / 8] |= 1 << (k % 8);
            }
        }
        (Array1::from_vec(xs), Array1::from_vec(zs))
    }

    /// Builds a Pauli string from unpacked X/Z indicator vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_ndarray(
    ///     ndarray::array![false, true, true, false].view(),
    ///     ndarray::array![false, false, true, true].view(),
    ///     -1,
    /// )
    /// .unwrap();
    /// assert_eq!(p, stim::PauliString::from_text("-_XYZ").unwrap());
    /// ```
    pub fn from_ndarray(
        xs: ArrayView1<'_, bool>,
        zs: ArrayView1<'_, bool>,
        sign: i32,
    ) -> crate::Result<Self> {
        if xs.len() != zs.len() {
            return Err(crate::StimError::new("Inconsistent xs/zs lengths"));
        }
        if sign != 1 && sign != -1 {
            return Err(crate::StimError::new("sign must be +1 or -1"));
        }
        let body: String = xs
            .iter()
            .zip(zs.iter())
            .map(|(x, z)| match (*x, *z) {
                (false, false) => '_',
                (true, false) => 'X',
                (true, true) => 'Y',
                (false, true) => 'Z',
            })
            .collect();
        Ok(Self::from_real_sign_and_body(sign, &body))
    }

    /// Builds a Pauli string from bit-packed X/Z indicator vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_ndarray_bit_packed(
    ///     ndarray::array![0x86, 0x0B].view(),
    ///     ndarray::array![0x0C, 0x1D].view(),
    ///     13,
    ///     1,
    /// )
    /// .unwrap();
    /// assert_eq!(p, stim::PauliString::from_text("+_XYZ___XYXZYZ").unwrap());
    /// ```
    pub fn from_ndarray_bit_packed(
        xs: ArrayView1<'_, u8>,
        zs: ArrayView1<'_, u8>,
        num_qubits: usize,
        sign: i32,
    ) -> crate::Result<Self> {
        let expected = num_qubits.div_ceil(8);
        if xs.len() != expected || zs.len() != expected {
            return Err(crate::StimError::new(format!(
                "bit-packed arrays must each have length {} for {} qubits",
                expected, num_qubits
            )));
        }
        if sign != 1 && sign != -1 {
            return Err(crate::StimError::new("sign must be +1 or -1"));
        }
        let mut body = String::with_capacity(num_qubits);
        for k in 0..num_qubits {
            let x = ((xs[k / 8] >> (k % 8)) & 1) != 0;
            let z = ((zs[k / 8] >> (k % 8)) & 1) != 0;
            body.push(match (x, z) {
                (false, false) => '_',
                (true, false) => 'X',
                (true, true) => 'Y',
                (false, true) => 'Z',
            });
        }
        Ok(Self::from_real_sign_and_body(sign, &body))
    }

    /// Converts the Pauli string into a unitary matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(
    ///     stim::PauliString::from_text("-YZ")
    ///         .unwrap()
    ///         .to_unitary_matrix(stim::Endian::Little)
    ///         .unwrap()[0][1],
    ///     stim::Complex32::new(0.0, 1.0)
    /// );
    /// ```
    pub fn to_unitary_matrix(
        &self,
        endian: crate::Endian,
    ) -> crate::Result<Vec<Vec<crate::Complex32>>> {
        let q = self.len();
        if q >= usize::BITS as usize {
            return Err(crate::StimError::new("Too many qubits."));
        }

        let mut x_mask = 0usize;
        let mut z_mask = 0usize;
        for k in 0..q {
            let p = self
                .get(k as isize)
                .expect("in-bounds PauliString indexing should succeed");
            let bit = match endian {
                crate::Endian::Little => 1usize << k,
                crate::Endian::Big => 1usize << (q - k - 1),
            };
            if matches!(p, 1 | 2) {
                x_mask |= bit;
            }
            if matches!(p, 2 | 3) {
                z_mask |= bit;
            }
        }

        let n = 1usize << q;
        let mut result = vec![vec![crate::Complex32::new(0.0, 0.0); n]; n];
        let mut start_phase = (x_mask & z_mask).count_ones() as u8;
        start_phase = start_phase.wrapping_add(self.phase_code());
        for (row_index, row) in result.iter_mut().enumerate() {
            let col = row_index ^ x_mask;
            let phase = start_phase.wrapping_add(((col & z_mask).count_ones() as u8) * 2);
            row[col] = match phase & 3 {
                0 => crate::Complex32::new(1.0, 0.0),
                1 => crate::Complex32::new(0.0, 1.0),
                2 => crate::Complex32::new(-1.0, 0.0),
                _ => crate::Complex32::new(0.0, -1.0),
            };
        }
        Ok(result)
    }

    /// Returns the Pauli code at an index.
    ///
    /// Codes are `0=I`, `1=X`, `2=Y`, `3=Z`.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("+_XYZ").unwrap();
    /// assert_eq!(p.get(1).unwrap(), 1);
    /// assert_eq!(p.get(-1).unwrap(), 3);
    /// ```
    pub fn get(&self, index: isize) -> crate::Result<u8> {
        let normalized = crate::normalize_index(index, self.len())
            .ok_or_else(|| crate::StimError::new(format!("index {index} out of range")))?;
        self.inner
            .get_item(normalized as i64)
            .map_err(crate::StimError::from)
    }

    /// Replaces the Pauli at an index.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut p = stim::PauliString::from_text("+_XYZ").unwrap();
    /// p.set(0, 'Z').unwrap();
    /// p.set(-1, 1u8).unwrap();
    /// assert_eq!(p, stim::PauliString::from_text("+ZXYX").unwrap());
    /// ```
    pub fn set(&mut self, index: isize, new_pauli: impl Into<PauliValue>) -> crate::Result<()> {
        let normalized = crate::normalize_index(index, self.len())
            .ok_or_else(|| crate::StimError::new(format!("index {index} out of range")))?;
        let code = match new_pauli.into() {
            PauliValue::Code(value @ 0..=3) => value,
            PauliValue::Code(_) => {
                return Err(crate::StimError::new(
                    "Expected new_pauli in [0, 1, 2, 3, '_', 'I', 'X', 'Y', 'Z']",
                ));
            }
            PauliValue::Symbol('_') | PauliValue::Symbol('I') => 0,
            PauliValue::Symbol('X') => 1,
            PauliValue::Symbol('Y') => 2,
            PauliValue::Symbol('Z') => 3,
            PauliValue::Symbol(_) => {
                return Err(crate::StimError::new(
                    "Expected new_pauli in [0, 1, 2, 3, '_', 'I', 'X', 'Y', 'Z']",
                ));
            }
        };
        self.inner
            .set_item(normalized as i64, code)
            .map_err(crate::StimError::from)
    }

    /// Returns a sliced Pauli string.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("+_XYZ___XYXZYZ").unwrap();
    /// assert_eq!(p.slice(None, Some(-1), 1).unwrap().to_string(), "+_XYZ___XYXZY");
    /// assert_eq!(p.slice(None, None, 2).unwrap().to_string(), "+_Y__YZZ");
    /// ```
    pub fn slice(
        &self,
        start: Option<isize>,
        stop: Option<isize>,
        step: isize,
    ) -> crate::Result<Self> {
        if step == 0 {
            return Err(crate::StimError::new("slice step cannot be zero"));
        }
        let indices = crate::compute_slice_indices(self.len() as isize, start, stop, step);
        let slice_length = i64::try_from(indices.len())
            .map_err(|_| crate::StimError::new("slice result length overflow"))?;
        let start = indices.first().copied().unwrap_or(0) as i64;
        Ok(Self {
            inner: self.inner.get_slice(start, step as i64, slice_length),
            imag: self.imag,
        })
    }

    /// Returns whether two Pauli strings commute.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = stim::PauliString::from_text("+X").unwrap();
    /// let z = stim::PauliString::from_text("+Z").unwrap();
    /// assert!(!x.commutes(&z));
    /// assert!(x.commutes(&x));
    /// ```
    #[must_use]
    pub fn commutes(&self, other: &Self) -> bool {
        self.inner.commutes(&other.inner)
    }

    /// Returns indices whose Pauli appears in `included_paulis`.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("+_XYZ___XYXZYZ").unwrap();
    /// assert_eq!(p.pauli_indices("XZ").unwrap(), vec![1, 3, 7, 9, 10, 12]);
    /// ```
    pub fn pauli_indices(&self, included_paulis: &str) -> crate::Result<Vec<usize>> {
        self.inner
            .pauli_indices(included_paulis)
            .map(|indices| indices.into_iter().map(|index| index as usize).collect())
            .map_err(crate::StimError::from)
    }

    /// Returns the real sign component (`±1`) of the Pauli string.
    #[must_use]
    pub fn sign(&self) -> i32 {
        self.inner.sign_code()
    }

    /// Returns the real sign component of the phase.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(stim::PauliString::from_text("-iX").unwrap().real_sign(), -1);
    /// ```
    #[must_use]
    pub fn real_sign(&self) -> i32 {
        self.sign()
    }

    /// Returns the full unit phase carried by the Pauli string.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("iX").unwrap();
    /// assert_eq!(p.phase(), stim::PauliPhase::PositiveImaginary);
    /// assert!(p.has_imaginary_phase());
    /// ```
    #[must_use]
    pub fn phase(&self) -> PauliPhase {
        Self::phase_from_code(self.phase_code())
    }

    /// Returns the phase as a complex number.
    #[must_use]
    pub fn complex_phase(&self) -> crate::Complex32 {
        self.phase().as_complex32()
    }

    /// Returns whether the phase is imaginary (`±i`).
    #[must_use]
    pub fn has_imaginary_phase(&self) -> bool {
        self.phase().is_imaginary()
    }

    /// Returns a copy of the Pauli string.
    #[must_use]
    pub fn pos(&self) -> Self {
        self.clone()
    }

    /// Infers a Pauli string from a unitary matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// let matrix = stim::PauliString::from_text("-iX")
    ///     .unwrap()
    ///     .to_unitary_matrix(stim::Endian::Little)
    ///     .unwrap();
    /// let p = stim::PauliString::from_unitary_matrix(&matrix, stim::Endian::Little, false).unwrap();
    /// assert_eq!(p, stim::PauliString::from_text("-iX").unwrap());
    /// ```
    pub fn from_unitary_matrix(
        matrix: &[Vec<crate::Complex32>],
        endian: crate::Endian,
        unsigned: bool,
    ) -> crate::Result<Self> {
        let little_endian = endian.is_little();
        let n = matrix.len();
        if n == 0 {
            return Ok(Self::new(0));
        }
        if matrix.iter().any(|row| row.len() != n) {
            return Err(crate::StimError::new("matrix must be square"));
        }
        if !n.is_power_of_two() {
            return Err(crate::StimError::new(
                "The given unitary matrix isn't a Pauli string matrix.",
            ));
        }
        let mut unphase: Option<crate::Complex32> = None;
        let mut row_index_phase = |row: &[crate::Complex32]| -> crate::Result<(u64, u8, u64)> {
            let mut num_cells = 0u64;
            let mut non_zero_index = None;
            let mut phase = 0u8;
            for cell in row {
                let mut c = *cell;
                if c.re.abs().max(c.im.abs()) < 1e-2 {
                    num_cells += 1;
                    continue;
                }
                if unphase.is_none() {
                    unphase = Some(if unsigned {
                        crate::Complex32::new(c.re, -c.im)
                    } else {
                        crate::Complex32::new(1.0, 0.0)
                    });
                }
                let u = unphase.expect("just set or already present");
                c = crate::Complex32::new(c.re * u.re - c.im * u.im, c.re * u.im + c.im * u.re);

                phase = if (c.re - 1.0).abs() < 1e-2 && c.im.abs() < 1e-2 {
                    0
                } else if c.re.abs() < 1e-2 && (c.im - 1.0).abs() < 1e-2 {
                    1
                } else if (c.re + 1.0).abs() < 1e-2 && c.im.abs() < 1e-2 {
                    2
                } else if c.re.abs() < 1e-2 && (c.im + 1.0).abs() < 1e-2 {
                    3
                } else {
                    let mut msg = String::from(
                        "The given unitary matrix isn't a Pauli string matrix. It has values besides 0, 1, -1, 1j, and -1j",
                    );
                    if unsigned {
                        msg.push_str(" (up to global phase)");
                    }
                    msg.push('.');
                    return Err(crate::StimError::new(msg));
                };
                if non_zero_index.is_some() {
                    return Err(crate::StimError::new(
                        "The given unitary matrix isn't a Pauli string matrix. It has two non-zero entries in the same row.",
                    ));
                }
                non_zero_index = Some(num_cells);
                num_cells += 1;
            }
            let idx = non_zero_index.ok_or_else(|| {
                crate::StimError::new(
                    "The given unitary matrix isn't a Pauli string matrix. It has a row with no non-zero entries.",
                )
            })?;
            Ok((idx, phase, num_cells))
        };

        let mut x = 0u64;
        let mut width = 0u64;
        let mut phases = Vec::new();
        for row in matrix {
            let (non_zero_index, phase, num_cells) = row_index_phase(row)?;
            let index = non_zero_index ^ phases.len() as u64;
            phases.push(phase);
            if phases.len() == 1 {
                x = index;
                width = num_cells;
            } else {
                if x != index {
                    return Err(crate::StimError::new(
                        "The given unitary matrix isn't a Pauli string matrix. Rows disagree about which qubits are flipped.",
                    ));
                }
                if width != num_cells {
                    return Err(crate::StimError::new(
                        "The given unitary matrix isn't a Pauli string matrix. Rows have different lengths.",
                    ));
                }
            }
        }

        if phases.len() as u64 != width {
            return Err(crate::StimError::new(
                "The given unitary matrix isn't a Pauli string matrix. It isn't square.",
            ));
        }
        if width == 0 || (width & (width - 1)) != 0 {
            return Err(crate::StimError::new(
                "The given unitary matrix isn't a Pauli string matrix. Its height isn't a power of 2.",
            ));
        }

        let mut z = 0u64;
        let mut q = 0usize;
        let mut p = width >> 1;
        while p > 0 {
            z <<= 1;
            z |= u64::from((phases[p as usize].wrapping_sub(phases[0]) & 2) != 0);
            q += 1;
            p >>= 1;
        }
        for k in 0..width as usize {
            let mut expected_phase = phases[0];
            if (((k as u64) & z).count_ones() & 1) != 0 {
                expected_phase = expected_phase.wrapping_add(2);
            }
            if (expected_phase & 3) != phases[k] {
                return Err(crate::StimError::new(
                    "The given unitary matrix isn't a Pauli string matrix. It doesn't have consistent phase flips.",
                ));
            }
        }

        let phase = if unsigned {
            0
        } else {
            phases[0].wrapping_add((x & z).count_ones() as u8) & 3
        };

        let mut body = String::with_capacity(q);
        if little_endian {
            for k in 0..q {
                let xb = (x >> k) & 1;
                let zb = (z >> k) & 1;
                body.push(match (xb != 0, zb != 0) {
                    (false, false) => '_',
                    (true, false) => 'X',
                    (true, true) => 'Y',
                    (false, true) => 'Z',
                });
            }
        } else {
            for k in (0..q).rev() {
                let xb = (x >> k) & 1;
                let zb = (z >> k) & 1;
                body.push(match (xb != 0, zb != 0) {
                    (false, false) => '_',
                    (true, false) => 'X',
                    (true, true) => 'Y',
                    (false, true) => 'Z',
                });
            }
        }

        Ok(Self::from_phase_code_and_body(phase, &body))
    }

    /// Divides the Pauli string by a unit complex phase.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("iX").unwrap();
    /// assert_eq!(
    ///     p.div_complex_unit(stim::Complex32::new(0.0, 1.0)).unwrap(),
    ///     stim::PauliString::from_text("+X").unwrap()
    /// );
    /// ```
    pub fn div_complex_unit(&self, divisor: crate::Complex32) -> crate::Result<Self> {
        let divisor_phase = Self::unit_phase_code(divisor)?;
        Ok(Self::from_phase_code_and_body(
            self.phase_code().wrapping_sub(divisor_phase) & 3,
            &self.dense_body_text(),
        ))
    }

    /// Divides the Pauli string by a unit complex phase in place.
    pub fn div_assign_complex_unit(&mut self, divisor: crate::Complex32) -> crate::Result<()> {
        *self = self.div_complex_unit(divisor)?;
        Ok(())
    }

    /// Converts the Pauli string into a tableau.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("+X").unwrap();
    /// let t = p.to_tableau();
    /// assert_eq!(t.to_pauli_string().unwrap(), p);
    /// ```
    #[must_use]
    pub fn to_tableau(&self) -> crate::Tableau {
        crate::Tableau {
            inner: self.inner.to_tableau(),
        }
    }

    /// Conjugates the Pauli string by a tableau applied on the given targets.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("_XYZ").unwrap();
    /// let cz = stim::Tableau::from_named_gate("CZ").unwrap();
    /// assert_eq!(
    ///     p.after_tableau(&cz, &[0, 1]).unwrap(),
    ///     stim::PauliString::from_text("+ZXYZ").unwrap()
    /// );
    /// ```
    pub fn after_tableau(
        &self,
        operation: &crate::Tableau,
        targets: &[usize],
    ) -> crate::Result<Self> {
        operation
            .inner
            .conjugate_pauli_string_within(&self.inner, targets, false)
            .map(|inner| Self {
                inner,
                imag: self.imag,
            })
            .map_err(crate::StimError::from)
    }

    /// Conjugates the Pauli string by a tableau before it acts on the given targets.
    pub fn before_tableau(
        &self,
        operation: &crate::Tableau,
        targets: &[usize],
    ) -> crate::Result<Self> {
        operation
            .inner
            .conjugate_pauli_string_within(&self.inner, targets, true)
            .map(|inner| Self {
                inner,
                imag: self.imag,
            })
            .map_err(crate::StimError::from)
    }

    /// Conjugates the Pauli string by a circuit.
    pub fn after_circuit(&self, operation: &crate::Circuit) -> crate::Result<Self> {
        let tableau = operation.to_tableau(false, false, false)?;
        let targets: Vec<usize> = (0..tableau.num_qubits()).collect();
        self.after_tableau(&tableau, &targets)
    }

    /// Conjugates the Pauli string by a circuit in reverse.
    pub fn before_circuit(&self, operation: &crate::Circuit) -> crate::Result<Self> {
        let tableau = operation.to_tableau(false, false, false)?;
        let targets: Vec<usize> = (0..tableau.num_qubits()).collect();
        self.before_tableau(&tableau, &targets)
    }

    /// Conjugates the Pauli string by a circuit instruction.
    pub fn after_instruction(&self, operation: &crate::CircuitInstruction) -> crate::Result<Self> {
        let mut circuit = crate::Circuit::new();
        circuit.append_instruction(operation)?;
        self.after_circuit(&circuit)
    }

    /// Conjugates the Pauli string by a circuit instruction in reverse.
    pub fn before_instruction(&self, operation: &crate::CircuitInstruction) -> crate::Result<Self> {
        let mut circuit = crate::Circuit::new();
        circuit.append_instruction(operation)?;
        self.before_circuit(&circuit)
    }

    /// Conjugates the Pauli string by any supported operation.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("_XYZ").unwrap();
    /// let cz = stim::Tableau::from_named_gate("CZ").unwrap();
    /// assert_eq!(
    ///     p.after((&cz, &[0, 1][..])).unwrap(),
    ///     stim::PauliString::from_text("+ZXYZ").unwrap()
    /// );
    /// ```
    pub fn after<'a>(
        &self,
        operation: impl Into<PauliStringConjugation<'a>>,
    ) -> crate::Result<Self> {
        match operation.into() {
            PauliStringConjugation::Circuit(circuit) => self.after_circuit(circuit),
            PauliStringConjugation::Instruction(instruction) => self.after_instruction(instruction),
            PauliStringConjugation::Tableau { tableau, targets } => {
                self.after_tableau(tableau, targets)
            }
        }
    }

    /// Conjugates the Pauli string by any supported operation in reverse.
    pub fn before<'a>(
        &self,
        operation: impl Into<PauliStringConjugation<'a>>,
    ) -> crate::Result<Self> {
        match operation.into() {
            PauliStringConjugation::Circuit(circuit) => self.before_circuit(circuit),
            PauliStringConjugation::Instruction(instruction) => {
                self.before_instruction(instruction)
            }
            PauliStringConjugation::Tableau { tableau, targets } => {
                self.before_tableau(tableau, targets)
            }
        }
    }
}

impl Display for PauliString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let sign = if self.sign() < 0 { "-" } else { "+" };
        f.write_str(sign)?;
        if self.imag {
            f.write_str("i")?;
        }
        f.write_str(&self.dense_body_text())
    }
}

impl fmt::Debug for PauliString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "stim.PauliString({:?})", self.to_string())
    }
}

impl FromStr for PauliString {
    type Err = crate::StimError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_text(s)
    }
}

impl Add for PauliString {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let phase = (self.phase_code() + rhs.phase_code()) & 3;
        let body = format!("{}{}", self.dense_body_text(), rhs.dense_body_text());
        PauliString::from_phase_code_and_body(phase, &body)
    }
}

impl AddAssign for PauliString {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl Neg for PauliString {
    type Output = Self;

    fn neg(self) -> Self::Output {
        PauliString::from_phase_code_and_body((self.phase_code() + 2) & 3, &self.dense_body_text())
    }
}

impl Mul<u64> for PauliString {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self::Output {
        let phase = ((self.phase_code() as u64 * rhs) & 3) as u8;
        let power = usize::try_from(rhs).expect("tensor power too large for this platform");
        let body = self.dense_body_text().repeat(power);
        PauliString::from_phase_code_and_body(phase, &body)
    }
}

impl MulAssign<u64> for PauliString {
    fn mul_assign(&mut self, rhs: u64) {
        *self = self.clone() * rhs;
    }
}

impl Mul<PauliString> for u64 {
    type Output = PauliString;

    fn mul(self, rhs: PauliString) -> Self::Output {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::{PauliPhase, PauliString};
    use std::str::FromStr;

    use crate::{Circuit, CircuitInstruction, Complex32, Tableau};

    #[test]
    fn pauli_string_numpy_style_bit_conversions_match_documented_examples() {
        let p = PauliString::from_text("_XYZ___XYXZYZ").unwrap();

        let (xs, zs) = p.to_ndarray();
        assert_eq!(
            xs,
            ndarray::array![
                false, true, true, false, false, false, false, true, true, true, false, true,
                false,
            ]
        );
        assert_eq!(
            zs,
            ndarray::array![
                false, false, true, true, false, false, false, false, true, false, true, true,
                true,
            ]
        );

        let (xs_packed, zs_packed) = p.to_ndarray_bit_packed();
        assert_eq!(xs_packed, ndarray::array![0x86, 0x0B]);
        assert_eq!(zs_packed, ndarray::array![0x0C, 0x1D]);

        assert_eq!(
            PauliString::from_ndarray(xs.view(), zs.view(), -1).unwrap(),
            PauliString::from_text("-_XYZ___XYXZYZ").unwrap()
        );
        assert_eq!(
            PauliString::from_ndarray_bit_packed(xs_packed.view(), zs_packed.view(), 13, 1)
                .unwrap(),
            p
        );
    }

    #[test]
    fn pauli_string_numpy_style_round_trips_and_validates_lengths() {
        for text in ["", "X", "-XYZ", "+_XYZ___XYXZYZ"] {
            let p = PauliString::from_text(text).unwrap();
            let (xs, zs) = p.to_ndarray();
            assert_eq!(
                PauliString::from_ndarray(xs.view(), zs.view(), p.sign()).unwrap(),
                p
            );

            let (xs_packed, zs_packed) = p.to_ndarray_bit_packed();
            assert_eq!(
                PauliString::from_ndarray_bit_packed(
                    xs_packed.view(),
                    zs_packed.view(),
                    p.len(),
                    p.sign(),
                )
                .unwrap(),
                p
            );
        }

        let err = PauliString::from_ndarray(
            ndarray::array![true, false].view(),
            ndarray::array![true].view(),
            1,
        )
        .unwrap_err();
        assert!(err.message().contains("Inconsistent xs/zs lengths"));

        let err = PauliString::from_ndarray_bit_packed(
            ndarray::array![0x86].view(),
            ndarray::array![0x0C, 0x1D].view(),
            13,
            1,
        )
        .unwrap_err();
        assert!(
            err.message()
                .contains("bit-packed arrays must each have length")
        );

        let err = PauliString::from_ndarray(
            ndarray::array![true].view(),
            ndarray::array![false].view(),
            0,
        )
        .unwrap_err();
        assert!(err.message().contains("sign must be +1 or -1"));
    }

    #[test]
    fn pauli_string_to_unitary_matrix_matches_documented_real_sign_examples() {
        assert_eq!(
            PauliString::from_text("-YZ")
                .unwrap()
                .to_unitary_matrix(crate::Endian::Little)
                .unwrap(),
            vec![
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 1.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, -1.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, -1.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 1.0),
                    Complex32::new(0.0, 0.0),
                ],
            ]
        );

        assert_eq!(
            PauliString::from_text("ZYX")
                .unwrap()
                .to_unitary_matrix(crate::Endian::Big)
                .unwrap(),
            vec![
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, -1.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, -1.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 1.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 1.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 1.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 1.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, -1.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, -1.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
            ]
        );
    }

    #[test]
    fn pauli_string_to_unitary_matrix_reports_documented_error_shapes() {
        assert!("middle".parse::<crate::Endian>().is_err());
    }

    #[test]
    fn pauli_string_len_matches_documented_examples() {
        assert_eq!(PauliString::from_text("XY_ZZ").unwrap().len(), 5);
        assert_eq!(
            PauliString::from_text(&format!("X{}Z", "_".repeat(98)))
                .unwrap()
                .len(),
            100
        );
    }

    #[test]
    fn pauli_string_weight_matches_documented_examples() {
        assert_eq!(PauliString::from_text("+___").unwrap().weight(), 0);
        assert_eq!(PauliString::from_text("+__X").unwrap().weight(), 1);
        assert_eq!(PauliString::from_text("+XYZ").unwrap().weight(), 3);
        assert_eq!(PauliString::from_text("-XXX___XXYZ").unwrap().weight(), 7);
    }

    #[test]
    fn pauli_string_after_matches_documented_examples() {
        let p = PauliString::from_text("_XYZ").unwrap();
        let cz = Tableau::from_named_gate("CZ").unwrap();

        assert_eq!(
            p.after_instruction(&CircuitInstruction::from_str("H 1").unwrap())
                .unwrap(),
            PauliString::from_text("+_ZYZ").unwrap()
        );
        assert_eq!(
            p.after_circuit(&Circuit::from_str("C_XYZ 1 2 3").unwrap())
                .unwrap(),
            PauliString::from_text("+_YZX").unwrap()
        );
        assert_eq!(
            p.after_tableau(&cz, &[0, 1]).unwrap(),
            PauliString::from_text("+ZXYZ").unwrap()
        );
        assert_eq!(
            p.after((&cz, &[0, 1][..])).unwrap(),
            PauliString::from_text("+ZXYZ").unwrap()
        );
    }

    #[test]
    fn pauli_string_before_matches_documented_examples() {
        let p = PauliString::from_text("_XYZ").unwrap();
        let cz = Tableau::from_named_gate("CZ").unwrap();

        assert_eq!(
            p.before_instruction(&CircuitInstruction::from_str("H 1").unwrap())
                .unwrap(),
            PauliString::from_text("+_ZYZ").unwrap()
        );
        assert_eq!(
            p.before_circuit(&Circuit::from_str("C_XYZ 1 2 3").unwrap())
                .unwrap(),
            PauliString::from_text("+_ZXY").unwrap()
        );
        assert_eq!(
            p.before_tableau(&cz, &[0, 1]).unwrap(),
            PauliString::from_text("+ZXYZ").unwrap()
        );
        assert_eq!(
            p.before((&cz, &[0, 1][..])).unwrap(),
            PauliString::from_text("+ZXYZ").unwrap()
        );
    }

    #[test]
    fn pauli_string_new_matches_documented_identity_behavior() {
        assert_eq!(PauliString::new(0).to_string(), "+");
        assert_eq!(PauliString::new(5).to_string(), "+_____");
    }

    #[test]
    fn pauli_string_copy_matches_documented_behavior() {
        let p1 = PauliString::random(2);
        let p2 = p1.clone();

        assert_eq!(p2, p1);
        assert_ne!((&p2 as *const PauliString), (&p1 as *const PauliString));
    }

    #[test]
    fn pauli_string_get_matches_documented_examples() {
        let p = PauliString::from_text("_XYZ").unwrap();

        assert_eq!(p.get(2).unwrap(), 2);
        assert_eq!(p.get(-1).unwrap(), 3);
    }

    #[test]
    fn pauli_string_slice_matches_documented_examples() {
        let p = PauliString::from_text("_XYZ").unwrap();

        assert_eq!(
            p.slice(None, Some(2), 1).unwrap(),
            PauliString::from_text("+_X").unwrap()
        );
        assert_eq!(
            p.slice(None, None, -1).unwrap(),
            PauliString::from_text("+ZYX_").unwrap()
        );
    }

    #[test]
    fn pauli_string_indexing_rejects_out_of_range_and_zero_step() {
        let p = PauliString::from_text("_XYZ").unwrap();

        assert!(p.get(4).unwrap_err().to_string().contains("out of range"));
        assert!(p.get(-5).unwrap_err().to_string().contains("out of range"));
        assert!(
            p.slice(None, None, 0)
                .unwrap_err()
                .to_string()
                .contains("step cannot be zero")
        );
    }

    #[test]
    fn pauli_string_iter_all_matches_documented_example_and_reiteration() {
        let iter = PauliString::iter_all(3, 1, 2, "XZ");
        let values: Vec<String> = iter.clone().map(|p| p.to_string()).collect();

        assert_eq!(
            values,
            vec![
                "+X__", "+Z__", "+_X_", "+_Z_", "+__X", "+__Z", "+XX_", "+XZ_", "+ZX_", "+ZZ_",
                "+X_X", "+X_Z", "+Z_X", "+Z_Z", "+_XX", "+_XZ", "+_ZX", "+_ZZ",
            ]
        );
        assert_eq!(iter.count(), 18);
    }

    #[test]
    fn pauli_string_tensor_power_matches_real_sign_examples() {
        assert_eq!(
            PauliString::from_text("X").unwrap() * 1_u64,
            PauliString::from_text("+X").unwrap()
        );
        assert_eq!(
            PauliString::from_text("X").unwrap() * 2_u64,
            PauliString::from_text("+XX").unwrap()
        );
        assert_eq!(
            PauliString::from_text("-X").unwrap() * 2_u64,
            PauliString::from_text("+XX").unwrap()
        );
        assert_eq!(
            PauliString::from_text("X").unwrap() * 3_u64,
            PauliString::from_text("+XXX").unwrap()
        );
    }

    #[test]
    fn pauli_string_tensor_power_assign_preserves_alias_identity() {
        let mut p = PauliString::from_text("-X").unwrap();
        let alias = &mut p as *mut PauliString;
        p *= 3_u64;
        assert_eq!(p, PauliString::from_text("-XXX").unwrap());
        assert_eq!(alias, &mut p as *mut PauliString);
    }

    #[test]
    fn pauli_string_left_tensor_power_matches_documented_real_sign_examples() {
        assert_eq!(
            2_u64 * PauliString::from_text("X").unwrap(),
            PauliString::from_text("+XX").unwrap()
        );
        assert_eq!(
            2_u64 * PauliString::from_text("-X").unwrap(),
            PauliString::from_text("+XX").unwrap()
        );
        assert_eq!(
            3_u64 * PauliString::from_text("X").unwrap(),
            PauliString::from_text("+XXX").unwrap()
        );
    }

    #[test]
    fn pauli_string_pos_and_set_match_documented_examples() {
        assert_eq!(
            PauliString::from_text("+X").unwrap().pos(),
            PauliString::from_text("+X").unwrap()
        );
        assert_eq!(
            PauliString::from_text("-YY").unwrap().pos(),
            PauliString::from_text("-YY").unwrap()
        );
        assert_eq!(
            PauliString::from_text("iZZZ").unwrap().pos(),
            PauliString::from_text("+iZZZ").unwrap()
        );

        let mut p = PauliString::new(4);
        p.set(2, 1u8).unwrap();
        assert_eq!(p.to_string(), "+__X_");
        p.set(0, 3u8).unwrap();
        p.set(1, 2u8).unwrap();
        p.set(3, 0u8).unwrap();
        assert_eq!(p.to_string(), "+ZYX_");
        p.set(0, 'I').unwrap();
        p.set(1, 'X').unwrap();
        p.set(2, 'Y').unwrap();
        p.set(3, 'Z').unwrap();
        assert_eq!(p.to_string(), "+_XYZ");
        p.set(-1, 'Y').unwrap();
        assert_eq!(p.to_string(), "+_XYY");

        let err = p.set(0, 4u8).unwrap_err();
        assert!(
            err.message()
                .contains("Expected new_pauli in [0, 1, 2, 3, '_', 'I', 'X', 'Y', 'Z']")
        );
    }

    #[test]
    fn pauli_string_commutes_matches_documented_examples() {
        let xx = PauliString::from_text("XX").unwrap();

        assert!(xx.commutes(&PauliString::from_text("X_").unwrap()));
        assert!(xx.commutes(&PauliString::from_text("XX").unwrap()));
        assert!(!xx.commutes(&PauliString::from_text("XY").unwrap()));
        assert!(!xx.commutes(&PauliString::from_text("XZ").unwrap()));
        assert!(xx.commutes(&PauliString::from_text("ZZ").unwrap()));
        assert!(xx.commutes(&PauliString::from_text("X_Y__").unwrap()));
        assert!(xx.commutes(&PauliString::from_text("").unwrap()));
    }

    #[test]
    fn pauli_string_pauli_indices_matches_documented_examples() {
        let ps = PauliString::from_text("_____X___Y____Z___").unwrap();

        assert_eq!(ps.pauli_indices("XYZ").unwrap(), vec![5, 9, 14]);
        assert_eq!(ps.pauli_indices("XZ").unwrap(), vec![5, 14]);
        assert_eq!(ps.pauli_indices("X").unwrap(), vec![5]);
        assert_eq!(ps.pauli_indices("Y").unwrap(), vec![9]);
        assert_eq!(
            ps.pauli_indices("IY").unwrap(),
            vec![0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
        );
        assert_eq!(
            PauliString::from_text(&format!(
                "-{}",
                (0..104)
                    .map(|k| match k {
                        100 => 'Y',
                        103 => 'X',
                        _ => '_',
                    })
                    .collect::<String>()
            ))
            .unwrap()
            .pauli_indices("XYZ")
            .unwrap(),
            vec![100, 103]
        );
    }

    #[test]
    fn pauli_string_pauli_indices_support_case_insensitive_filters_and_reject_invalid_chars() {
        let ps = PauliString::from_text("_XYZ").unwrap();

        assert_eq!(ps.pauli_indices("x").unwrap(), vec![1]);
        assert_eq!(ps.pauli_indices("y").unwrap(), vec![2]);
        assert_eq!(ps.pauli_indices("z").unwrap(), vec![3]);
        assert_eq!(ps.pauli_indices("I").unwrap(), vec![0]);
        assert_eq!(ps.pauli_indices("_").unwrap(), vec![0]);

        let err = ps.pauli_indices("k").unwrap_err();
        assert!(err.to_string().contains("Invalid character"));
    }

    #[test]
    fn pauli_string_random_matches_documented_basic_contract() {
        let p = PauliString::random(5);
        assert_eq!(p.len(), 5);
        assert!(matches!(p.sign(), -1 | 1));

        let p0 = PauliString::random(0);
        assert_eq!(p0.len(), 0);
        assert!(matches!(p0.sign(), -1 | 1));
    }

    #[test]
    fn pauli_string_phase_accessors_match_sign_model() {
        let p = PauliString::from_text("+X").unwrap();
        assert_eq!(p.phase(), PauliPhase::Positive);
        assert_eq!(p.real_sign(), 1);
        assert_eq!(p.complex_phase(), crate::Complex32::new(1.0, 0.0));
        assert!(!p.has_imaginary_phase());

        let p = PauliString::from_text("-X").unwrap();
        assert_eq!(p.phase(), PauliPhase::Negative);
        assert_eq!(p.real_sign(), -1);
        assert_eq!(p.complex_phase(), crate::Complex32::new(-1.0, 0.0));
        assert!(!p.has_imaginary_phase());

        let p = PauliString::from_text("iX").unwrap();
        assert_eq!(p.phase(), PauliPhase::PositiveImaginary);
        assert_eq!(p.real_sign(), 1);
        assert_eq!(p.complex_phase(), crate::Complex32::new(0.0, 1.0));
        assert!(p.has_imaginary_phase());

        let p = PauliString::from_text("-iX").unwrap();
        assert_eq!(p.phase(), PauliPhase::NegativeImaginary);
        assert_eq!(p.real_sign(), -1);
        assert_eq!(p.complex_phase(), crate::Complex32::new(0.0, -1.0));
        assert!(p.has_imaginary_phase());
    }

    #[test]
    fn pauli_string_add_and_add_assign_match_real_sign_tensor_product_examples() {
        assert_eq!(
            PauliString::from_text("X").unwrap() + PauliString::from_text("YZ").unwrap(),
            PauliString::from_text("+XYZ").unwrap()
        );

        let mut p = PauliString::from_text("-X").unwrap();
        let alias = &mut p as *mut PauliString;
        p += PauliString::from_text("YY").unwrap();
        assert_eq!(p, PauliString::from_text("-XYY").unwrap());
        assert_eq!(alias, &mut p as *mut PauliString);
    }

    #[test]
    fn pauli_string_neg_matches_real_sign_examples() {
        assert_eq!(
            -PauliString::from_text("X").unwrap(),
            PauliString::from_text("-X").unwrap()
        );
        assert_eq!(
            -PauliString::from_text("-Y").unwrap(),
            PauliString::from_text("+Y").unwrap()
        );
        assert_eq!(
            -PauliString::from_text("iZZZ").unwrap(),
            PauliString::from_text("-iZZZ").unwrap()
        );
    }

    #[test]
    fn pauli_string_complex_unit_division_matches_documented_examples() {
        let p = PauliString::from_text("X")
            .unwrap()
            .div_complex_unit(crate::Complex32::new(0.0, 1.0))
            .unwrap();
        assert_eq!(p, PauliString::from_text("-iX").unwrap());

        let mut p = PauliString::from_text("X").unwrap();
        p.div_assign_complex_unit(crate::Complex32::new(0.0, 1.0))
            .unwrap();
        assert_eq!(p, PauliString::from_text("-iX").unwrap());
    }

    #[test]
    fn pauli_string_remaining_public_entrypoints_and_error_paths_are_exercised() {
        let empty = PauliString::from_text("").unwrap();
        assert!(empty.is_empty());
        assert_eq!(
            PauliString::from_unitary_matrix(&[], crate::Endian::Little, false).unwrap(),
            empty
        );

        let mut p = PauliString::new(2);
        let invalid_code = p.set(0, 9u8).unwrap_err();
        assert!(invalid_code.to_string().contains("Expected new_pauli"));
        let invalid_symbol = p.set(0, 'Q').unwrap_err();
        assert!(invalid_symbol.to_string().contains("Expected new_pauli"));
        let invalid_sign = PauliString::from_ndarray_bit_packed(
            ndarray::array![0u8].view(),
            ndarray::array![0u8].view(),
            1,
            0,
        )
        .unwrap_err();
        assert!(invalid_sign.to_string().contains("sign must be +1 or -1"));

        let too_wide = PauliString::new(usize::BITS as usize)
            .to_unitary_matrix(crate::Endian::Little)
            .unwrap_err();
        assert!(too_wide.to_string().contains("Too many qubits"));

        assert_eq!(
            PauliString::from_text("-X")
                .unwrap()
                .to_unitary_matrix(crate::Endian::Little)
                .unwrap()[0][1],
            Complex32::new(-1.0, 0.0)
        );
        assert_eq!(
            PauliString::from_text("-iX")
                .unwrap()
                .to_unitary_matrix(crate::Endian::Little)
                .unwrap()[0][1],
            Complex32::new(0.0, -1.0)
        );
        assert_eq!(
            PauliString::from_unitary_matrix(
                &PauliString::from_text("+XZ")
                    .unwrap()
                    .to_unitary_matrix(crate::Endian::Big)
                    .unwrap(),
                crate::Endian::Big,
                false,
            )
            .unwrap(),
            PauliString::from_text("+XZ").unwrap()
        );

        assert_eq!(
            PauliString::from_text("iX")
                .unwrap()
                .div_complex_unit(Complex32::new(1.0, 0.0))
                .unwrap(),
            PauliString::from_text("iX").unwrap()
        );
        assert_eq!(
            PauliString::from_text("iX")
                .unwrap()
                .div_complex_unit(Complex32::new(-1.0, 0.0))
                .unwrap(),
            PauliString::from_text("-iX").unwrap()
        );
        assert_eq!(
            PauliString::from_text("iX")
                .unwrap()
                .div_complex_unit(Complex32::new(0.0, -1.0))
                .unwrap(),
            PauliString::from_text("-X").unwrap()
        );
        let invalid_divisor = PauliString::from_text("X")
            .unwrap()
            .div_complex_unit(Complex32::new(0.5, 0.0))
            .unwrap_err();
        assert!(
            invalid_divisor
                .to_string()
                .contains("divisor not in (1, -1, 1j, -1j)")
        );
        assert_eq!(format!("{}", PauliString::from_text("iX").unwrap()), "+iX");

        let circuit: Circuit = "H 0".parse().unwrap();
        let instruction = CircuitInstruction::from_stim_program_text("H 0").unwrap();
        assert_eq!(
            PauliString::from_text("Z")
                .unwrap()
                .after(&circuit)
                .unwrap(),
            PauliString::from_text("X").unwrap()
        );
        assert_eq!(
            PauliString::from_text("X")
                .unwrap()
                .before(&instruction)
                .unwrap(),
            PauliString::from_text("Z").unwrap()
        );
    }

    #[test]
    fn pauli_string_from_unitary_matrix_matches_documented_examples() {
        assert_eq!(
            PauliString::from_unitary_matrix(
                &[
                    vec![
                        crate::Complex32::new(0.0, 1.0),
                        crate::Complex32::new(0.0, 0.0)
                    ],
                    vec![
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(0.0, -1.0)
                    ],
                ],
                crate::Endian::Little,
                false,
            )
            .unwrap(),
            PauliString::from_text("+iZ").unwrap()
        );
        assert_eq!(
            PauliString::from_unitary_matrix(
                &[
                    vec![
                        crate::Complex32::new(0.98768836, 0.15643446),
                        crate::Complex32::new(0.0, 0.0)
                    ],
                    vec![
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(-0.98768836, -0.15643446)
                    ],
                ],
                crate::Endian::Little,
                true,
            )
            .unwrap(),
            PauliString::from_text("+Z").unwrap()
        );
        assert_eq!(
            PauliString::from_unitary_matrix(
                &[
                    vec![
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(1.0, 0.0),
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(0.0, 0.0),
                    ],
                    vec![
                        crate::Complex32::new(1.0, 0.0),
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(0.0, 0.0),
                    ],
                    vec![
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(-1.0, 0.0),
                    ],
                    vec![
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(0.0, 0.0),
                        crate::Complex32::new(-1.0, 0.0),
                        crate::Complex32::new(0.0, 0.0),
                    ],
                ],
                crate::Endian::Little,
                false,
            )
            .unwrap(),
            PauliString::from_text("+XZ").unwrap()
        );
    }

    #[test]
    fn pauli_string_from_unitary_matrix_reports_remaining_error_shapes() {
        assert!("middle".parse::<crate::Endian>().is_err());

        let non_square = PauliString::from_unitary_matrix(
            &[
                vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
                vec![Complex32::new(0.0, 0.0)],
            ],
            crate::Endian::Little,
            false,
        )
        .unwrap_err();
        assert!(non_square.to_string().contains("matrix must be square"));

        let non_power_of_two = PauliString::from_unitary_matrix(
            &[
                vec![
                    Complex32::new(1.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(1.0, 0.0),
                ],
            ],
            crate::Endian::Little,
            false,
        )
        .unwrap_err();
        assert!(
            non_power_of_two
                .to_string()
                .contains("isn't a Pauli string matrix")
        );

        let no_non_zero = PauliString::from_unitary_matrix(
            &[
                vec![Complex32::new(0.0, 0.0), Complex32::new(0.0, 0.0)],
                vec![Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)],
            ],
            crate::Endian::Little,
            false,
        )
        .unwrap_err();
        assert!(
            no_non_zero
                .to_string()
                .contains("row with no non-zero entries")
        );

        let two_non_zero = PauliString::from_unitary_matrix(
            &[
                vec![Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)],
                vec![Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)],
            ],
            crate::Endian::Little,
            false,
        )
        .unwrap_err();
        assert!(
            two_non_zero
                .to_string()
                .contains("two non-zero entries in the same row")
        );

        let inconsistent_flips = PauliString::from_unitary_matrix(
            &[
                vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
                vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
            ],
            crate::Endian::Little,
            false,
        )
        .unwrap_err();
        assert!(
            inconsistent_flips
                .to_string()
                .contains("Rows disagree about which qubits are flipped")
        );

        let inconsistent_phases = PauliString::from_unitary_matrix(
            &[
                vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
                vec![Complex32::new(0.0, 0.0), Complex32::new(0.0, 1.0)],
            ],
            crate::Endian::Little,
            false,
        )
        .unwrap_err();
        assert!(
            inconsistent_phases
                .to_string()
                .contains("doesn't have consistent phase flips")
        );
    }
}
