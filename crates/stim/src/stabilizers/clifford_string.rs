use std::fmt::{self, Display, Formatter};
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::str::FromStr;

/// A replacement value accepted by [`CliffordString::set`].
///
/// This enum allows `set` to accept either a gate name string (like `"H"` or
/// `"S"`) or a [`crate::Gate`] enum value. Both variants
/// are converted through the corresponding [`From`] implementations, so
/// callers can pass either type directly.
pub enum CliffordGateValue<'a> {
    /// A gate name like `"H"`, `"S"`, or `"SQRT_X"`.
    Name(&'a str),
    /// A canonical Stim gate.
    Gate(crate::Gate),
}

impl<'a> From<&'a str> for CliffordGateValue<'a> {
    fn from(value: &'a str) -> Self {
        Self::Name(value)
    }
}

impl<'a> From<crate::Gate> for CliffordGateValue<'a> {
    fn from(value: crate::Gate) -> Self {
        Self::Gate(value)
    }
}

/// A tensor product of single-qubit Clifford gates (e.g. "H x X x S").
///
/// Represents a sequence of single-qubit Clifford operations, one per qubit,
/// applied independently. This is useful for representing per-qubit Clifford
/// layers in a circuit, such as a round of single-qubit gates before or after
/// entangling operations. Global phase is ignored.
///
/// There are exactly 24 single-qubit Clifford gates (the symmetry group of
/// the octahedron). A `CliffordString` assigns one of these 24 gates to each
/// qubit position. The gates are represented by their canonical Stim names
/// (e.g. `I`, `X`, `Y`, `Z`, `H`, `S`, `S_DAG`, `SQRT_X`, `C_XYZ`, etc.).
///
/// `CliffordString` supports several arithmetic operations:
///
/// - **Concatenation** (`+`, `+=`): appends one string after another.
/// - **Pairwise multiplication** (`*`, `*=` with another `CliffordString`):
///   composes corresponding Clifford gates element-wise. When lengths differ,
///   the shorter string is implicitly padded with identity gates.
/// - **Repetition** (`*`, `*=` with a `u64`): repeats the contents.
/// - **Exponentiation** (`pow`, `ipow`): raises each gate to a power.
///
/// # Examples
///
/// ```
/// use stim::CliffordString;
///
/// // Parse from comma-separated gate names.
/// let c = "H,S,C_XYZ".parse::<CliffordString>().unwrap();
/// assert_eq!(c.len(), 3);
/// assert_eq!(c.get(0).unwrap().name(), "H");
///
/// // Pairwise composition: H*H = I, S*H = C_ZYX, C_XYZ*H = SQRT_X_DAG.
/// let composed = "H,S,C_XYZ".parse::<CliffordString>().unwrap()
///     * "H,H,H".parse::<CliffordString>().unwrap();
/// assert_eq!(composed.to_string(), "I,C_ZYX,SQRT_X_DAG");
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct CliffordString {
    pub(crate) inner: stim_cxx::CliffordString,
}

impl CliffordString {
    /// Creates an identity Clifford string over `num_qubits` qubits.
    ///
    /// Every gate in the resulting string is the single-qubit identity `I`.
    ///
    /// # Examples
    ///
    /// ```
    /// let c = stim::CliffordString::new(3);
    /// assert_eq!(c.to_string(), "I,I,I");
    /// assert_eq!(c.len(), 3);
    ///
    /// let empty = stim::CliffordString::new(0);
    /// assert!(empty.is_empty());
    /// ```
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            inner: stim_cxx::CliffordString::new(num_qubits),
        }
    }

    /// Parses a Clifford string from a comma-separated list of gate names.
    ///
    /// Each token between commas should be the canonical Stim name of a
    /// single-qubit Clifford gate. Leading and trailing whitespace around each
    /// token is trimmed, and a trailing comma is allowed. Gate name aliases
    /// (e.g. `H_XZ` for `H`) are accepted and normalized to their canonical
    /// form.
    ///
    /// # Errors
    ///
    /// Returns an error if any token is not a recognized single-qubit Clifford
    /// gate name.
    ///
    /// # Examples
    ///
    /// ```
    /// let s: stim::CliffordString = "X,Y,Z,H,SQRT_X,C_XYZ".parse().unwrap();
    /// assert_eq!(s.get(2).unwrap().name(), "Z");
    /// assert_eq!(s.get(-1).unwrap().name(), "C_XYZ");
    ///
    /// // Whitespace is trimmed.
    /// let s: stim::CliffordString = "  H  ,  S  ".parse().unwrap();
    /// assert_eq!(s.to_string(), "H,S");
    /// ```
    fn parse_text(text: &str) -> crate::Result<Self> {
        stim_cxx::CliffordString::from_text(text)
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Converts a [`crate::PauliString`] into the corresponding Clifford string.
    ///
    /// Each Pauli operator in the input string (`I`, `X`, `Y`, `Z`) becomes the
    /// corresponding single-qubit Clifford gate. The sign of the Pauli string
    /// is ignored, since `CliffordString` does not track global phase.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = "-XYZ".parse::<stim::PauliString>().unwrap();
    /// let c = stim::CliffordString::from_pauli_string(&p);
    /// // The sign is discarded.
    /// assert_eq!(c.to_string(), "X,Y,Z");
    /// ```
    #[must_use]
    pub fn from_pauli_string(pauli_string: &crate::PauliString) -> Self {
        Self {
            inner: stim_cxx::CliffordString::from_pauli_string(&pauli_string.inner),
        }
    }

    /// Extracts the per-qubit Clifford action of a circuit as a Clifford string.
    ///
    /// The circuit must contain only single-qubit unitary operations and
    /// annotations (like `TICK`). Multi-qubit gates such as `CNOT` are not
    /// allowed. The resulting string describes the composed single-qubit
    /// Clifford applied to each qubit across all layers of the circuit.
    ///
    /// The length of the returned string equals the number of qubits used by
    /// the circuit (including implicit identity qubits for unused indices up to
    /// the maximum qubit index).
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit contains multi-qubit gates or non-unitary
    /// operations.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0 1\nS 1 2".parse().unwrap();
    /// let c = stim::CliffordString::from_circuit(&circuit).unwrap();
    /// // Qubit 0: H, Qubit 1: H then S = C_ZYX, Qubit 2: S.
    /// assert_eq!(c.to_string(), "H,C_ZYX,S");
    /// ```
    pub fn from_circuit(circuit: &crate::Circuit) -> crate::Result<Self> {
        stim_cxx::CliffordString::from_circuit(&circuit.inner)
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Returns a uniformly random Clifford string of the given length.
    ///
    /// Each qubit position is independently assigned one of the 24 single-qubit
    /// Clifford gates with equal probability.
    ///
    /// # Examples
    ///
    /// ```
    /// let c = stim::CliffordString::random(4);
    /// assert_eq!(c.num_qubits(), 4);
    /// assert_eq!(c.len(), 4);
    /// ```
    #[must_use]
    pub fn random(num_qubits: usize) -> Self {
        Self {
            inner: stim_cxx::CliffordString::random(num_qubits),
        }
    }

    /// Returns a 24-element Clifford string containing each single-qubit Clifford
    /// gate exactly once, in Stim's canonical order.
    ///
    /// This is useful for enumerating or testing behavior across all 24
    /// single-qubit Cliffords. The canonical order groups the gates as:
    ///
    /// - Indices 0--3: Paulis (`I`, `X`, `Y`, `Z`)
    /// - Indices 4--7: axis-exchange half-turns (`H_XY`, `S`, `S_DAG`, `H_NXY`)
    /// - Indices 8--15: other quarter/half-turns (`H`, `SQRT_Y_DAG`, ...)
    /// - Indices 16--23: order-3 rotations (`C_XYZ`, `C_XYNZ`, ...)
    ///
    /// # Examples
    ///
    /// ```
    /// let c = stim::CliffordString::all_cliffords_string();
    /// assert_eq!(c.len(), 24);
    /// assert_eq!(
    ///     c.slice(Some(0), Some(4), 1).unwrap().to_string(),
    ///     "I,X,Y,Z"
    /// );
    /// ```
    #[must_use]
    pub fn all_cliffords_string() -> Self {
        Self {
            inner: stim_cxx::CliffordString::all_cliffords_string(),
        }
    }

    /// Returns the number of qubit positions in the Clifford string.
    ///
    /// This is identical to [`CliffordString::len`].
    ///
    /// # Examples
    ///
    /// ```
    /// let c = "H,S,I".parse::<stim::CliffordString>().unwrap();
    /// assert_eq!(c.num_qubits(), 3);
    /// ```
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of single-qubit Clifford entries in the string.
    ///
    /// This is identical to [`CliffordString::num_qubits`].
    ///
    /// # Examples
    ///
    /// ```
    /// let c = "I,X,Y,Z,H".parse::<stim::CliffordString>().unwrap();
    /// assert_eq!(c.len(), 5);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_qubits()
    }

    /// Returns whether the Clifford string contains zero entries.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::CliffordString::new(0).is_empty());
    /// assert!(!stim::CliffordString::new(1).is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the single-qubit Clifford gate at the given index as a
    /// [`crate::Gate`].
    ///
    /// Negative indices count backwards from the end of the string, following
    /// the Python convention (e.g. `-1` is the last element).
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let c = "I,X,H".parse::<stim::CliffordString>().unwrap();
    /// assert_eq!(c.get(1).unwrap().name(), "X");
    /// assert_eq!(c.get(-1).unwrap().name(), "H");
    /// ```
    pub fn get(&self, index: isize) -> crate::Result<crate::Gate> {
        self.inner
            .get_item_name(index as i64)
            .map_err(crate::StimError::from)
            .and_then(|name| crate::Gate::new(&name))
    }

    /// Replaces the single-qubit Clifford gate at the given index.
    ///
    /// The replacement `value` can be a gate name string (e.g. `"H"`) or a
    /// [`crate::Gate`] enum value. Negative indices count
    /// backwards from the end.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of bounds or `value` is not a
    /// recognized single-qubit Clifford gate.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut c = "I,I,I".parse::<stim::CliffordString>().unwrap();
    /// c.set(1, "H").unwrap();
    /// c.set(-1, stim::Gate::S).unwrap();
    /// assert_eq!(c.to_string(), "I,H,S");
    /// ```
    pub fn set<'a>(
        &mut self,
        index: isize,
        value: impl Into<CliffordGateValue<'a>>,
    ) -> crate::Result<()> {
        let normalized = crate::normalize_index(index, self.len())
            .ok_or_else(|| crate::StimError::new(format!("index {index} out of range")))?;
        let replacement = match value.into() {
            CliffordGateValue::Name(name) => crate::Gate::new(name)?.name().to_string(),
            CliffordGateValue::Gate(gate) => gate.name().to_string(),
        };
        let mut parts: Vec<String> = if self.is_empty() {
            Vec::new()
        } else {
            self.to_string().split(',').map(str::to_owned).collect()
        };
        parts[normalized] = replacement;
        *self = parts.join(",").parse::<Self>()?;
        Ok(())
    }

    /// Returns a sub-sequence of the Clifford string selected by Python-style
    /// slice parameters.
    ///
    /// The `start`, `stop`, and `step` parameters follow the same semantics as
    /// Python's `slice(start, stop, step)`. Negative values for `start` and
    /// `stop` count backwards from the end of the string. `None` means "use the
    /// natural bound" (beginning or end, depending on sign of `step`).
    ///
    /// # Errors
    ///
    /// Returns an error if `step` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let c = "I,X,H,Y,S".parse::<stim::CliffordString>().unwrap();
    ///
    /// // Omit the last element.
    /// assert_eq!(c.slice(None, Some(-1), 1).unwrap().to_string(), "I,X,H,Y");
    ///
    /// // Every other element.
    /// assert_eq!(c.slice(None, None, 2).unwrap().to_string(), "I,H,S");
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
        })
    }

    /// Returns what each Clifford in the string conjugates an X input into.
    ///
    /// For each qubit position, this computes the result of conjugating the
    /// single-qubit X operator by the Clifford gate at that position. For
    /// example, `H` conjugates X into +Z, and `Y` conjugates X into -X.
    ///
    /// Combined with [`CliffordString::z_outputs`], the X outputs completely
    /// specify the single-qubit Clifford applied at each position.
    ///
    /// Returns a `(paulis, signs)` tuple where:
    /// - `paulis` is a [`crate::PauliString`] (always positive sign) giving
    ///   the output Pauli type at each qubit.
    /// - `signs` is a `Vec<bool>` where `true` means the output has a negative
    ///   sign.
    ///
    /// # Examples
    ///
    /// ```
    /// let (paulis, signs) = "I,Y,H,S".parse::<stim::CliffordString>().unwrap().x_outputs();
    /// assert_eq!(paulis, "+XXZY".parse::<stim::PauliString>().unwrap());
    /// assert_eq!(signs, vec![false, true, false, false]);
    /// ```
    pub fn x_outputs(&self) -> (crate::PauliString, Vec<bool>) {
        let signs = self.inner.x_signs_bit_packed();
        (
            crate::PauliString {
                inner: self.inner.x_outputs(),
                imag: false,
            },
            crate::unpack_bits(&signs, self.len()),
        )
    }

    /// Returns the X outputs with sign data in bit-packed form.
    ///
    /// This is the same as [`CliffordString::x_outputs`] except the sign data
    /// is returned as a `Vec<u8>` with 8 sign bits packed into each byte in
    /// little-endian bit order. The vector length is `ceil(num_qubits / 8)`.
    ///
    /// # Examples
    ///
    /// ```
    /// let (paulis, signs) = "I,Y,H,S".parse::<stim::CliffordString>().unwrap().x_outputs_bit_packed();
    /// assert_eq!(paulis, "+XXZY".parse::<stim::PauliString>().unwrap());
    /// // Bit 1 is set (Y at index 1 negates X), giving 0b00000010 = 2.
    /// assert_eq!(signs, vec![2]);
    /// ```
    pub fn x_outputs_bit_packed(&self) -> (crate::PauliString, Vec<u8>) {
        let signs = self.inner.x_signs_bit_packed();
        (
            crate::PauliString {
                inner: self.inner.x_outputs(),
                imag: false,
            },
            signs,
        )
    }

    /// Returns what each Clifford in the string conjugates a Y input into.
    ///
    /// For each qubit position, this computes the result of conjugating the
    /// single-qubit Y operator by the Clifford gate at that position. For
    /// example, `H` conjugates Y into -Y, and `S_DAG` conjugates Y into +X.
    ///
    /// Returns a `(paulis, signs)` tuple with the same structure as
    /// [`CliffordString::x_outputs`].
    ///
    /// # Examples
    ///
    /// ```
    /// let (paulis, signs) = "I,X,H,S".parse::<stim::CliffordString>().unwrap().y_outputs();
    /// assert_eq!(paulis, "+YYYX".parse::<stim::PauliString>().unwrap());
    /// assert_eq!(signs, vec![false, true, true, true]);
    /// ```
    pub fn y_outputs(&self) -> (crate::PauliString, Vec<bool>) {
        let signs = self.inner.y_signs_bit_packed();
        (
            crate::PauliString {
                inner: self.inner.y_outputs(),
                imag: false,
            },
            crate::unpack_bits(&signs, self.len()),
        )
    }

    /// Returns the Y outputs with sign data in bit-packed form.
    ///
    /// This is the same as [`CliffordString::y_outputs`] except the sign data
    /// is returned as a `Vec<u8>` with 8 sign bits packed per byte in
    /// little-endian bit order.
    ///
    /// # Examples
    ///
    /// ```
    /// let (paulis, signs) = "I,X,H,S".parse::<stim::CliffordString>().unwrap().y_outputs_bit_packed();
    /// assert_eq!(paulis, "+YYYX".parse::<stim::PauliString>().unwrap());
    /// // Bits 1, 2, 3 are set: 0b00001110 = 14.
    /// assert_eq!(signs, vec![14]);
    /// ```
    pub fn y_outputs_bit_packed(&self) -> (crate::PauliString, Vec<u8>) {
        let signs = self.inner.y_signs_bit_packed();
        (
            crate::PauliString {
                inner: self.inner.y_outputs(),
                imag: false,
            },
            signs,
        )
    }

    /// Returns what each Clifford in the string conjugates a Z input into.
    ///
    /// For each qubit position, this computes the result of conjugating the
    /// single-qubit Z operator by the Clifford gate at that position. For
    /// example, `H` conjugates Z into +X, and `SQRT_X` conjugates Z into -Y.
    ///
    /// Combined with [`CliffordString::x_outputs`], the Z outputs completely
    /// specify the single-qubit Clifford applied at each position.
    ///
    /// Returns a `(paulis, signs)` tuple with the same structure as
    /// [`CliffordString::x_outputs`].
    ///
    /// # Examples
    ///
    /// ```
    /// let (paulis, signs) = "I,Y,H,S".parse::<stim::CliffordString>().unwrap().z_outputs();
    /// assert_eq!(paulis, "+ZZXZ".parse::<stim::PauliString>().unwrap());
    /// assert_eq!(signs, vec![false, true, false, false]);
    /// ```
    pub fn z_outputs(&self) -> (crate::PauliString, Vec<bool>) {
        let signs = self.inner.z_signs_bit_packed();
        (
            crate::PauliString {
                inner: self.inner.z_outputs(),
                imag: false,
            },
            crate::unpack_bits(&signs, self.len()),
        )
    }

    /// Returns the Z outputs with sign data in bit-packed form.
    ///
    /// This is the same as [`CliffordString::z_outputs`] except the sign data
    /// is returned as a `Vec<u8>` with 8 sign bits packed per byte in
    /// little-endian bit order.
    ///
    /// # Examples
    ///
    /// ```
    /// let (paulis, signs) = "I,Y,H,S".parse::<stim::CliffordString>().unwrap().z_outputs_bit_packed();
    /// assert_eq!(paulis, "+ZZXZ".parse::<stim::PauliString>().unwrap());
    /// // Bit 1 is set (Y at index 1 negates Z): 0b00000010 = 2.
    /// assert_eq!(signs, vec![2]);
    /// ```
    pub fn z_outputs_bit_packed(&self) -> (crate::PauliString, Vec<u8>) {
        let signs = self.inner.z_signs_bit_packed();
        (
            crate::PauliString {
                inner: self.inner.z_outputs(),
                imag: false,
            },
            signs,
        )
    }

    /// Returns a new Clifford string with each gate raised to the given power.
    ///
    /// Each single-qubit Clifford gate in the string is independently raised
    /// to `exponent`. Since single-qubit Cliffords have order dividing 24,
    /// the exponent is effectively taken modulo the gate's order. Negative
    /// exponents produce the inverse (e.g. `S` raised to `-1` gives `S_DAG`).
    ///
    /// # Examples
    ///
    /// ```
    /// let c = "I,X,H,S,C_XYZ".parse::<stim::CliffordString>().unwrap();
    /// assert_eq!(c.pow(3).to_string(), "I,X,H,S_DAG,I");
    /// assert_eq!(c.pow(-1).to_string(), "I,X,H,S_DAG,C_ZYX");
    /// ```
    #[must_use]
    pub fn pow(&self, exponent: i64) -> Self {
        Self {
            inner: self.inner.pow(exponent),
        }
    }

    /// Raises each Clifford gate in the string to the given power in place.
    ///
    /// This is the mutating variant of [`CliffordString::pow`].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut c = "I,X,H,S,C_XYZ".parse::<stim::CliffordString>().unwrap();
    /// c.ipow(3);
    /// assert_eq!(c, "I,X,H,S_DAG,I".parse::<stim::CliffordString>().unwrap());
    /// ```
    pub fn ipow(&mut self, exponent: i64) {
        self.inner.ipow(exponent);
    }
}

impl Default for CliffordString {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Display for CliffordString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.to_text())
    }
}

impl fmt::Debug for CliffordString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.to_repr_text())
    }
}

impl FromStr for CliffordString {
    type Err = crate::StimError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_text(s)
    }
}

/// Concatenates two Clifford strings end-to-end.
///
/// The result has length `self.len() + rhs.len()`, with the entries of `self`
/// followed by the entries of `rhs`.
///
/// # Examples
///
/// ```
/// use stim::CliffordString;
///
/// let ab = "I,X,H".parse::<CliffordString>().unwrap()
///     + "Y,S".parse::<CliffordString>().unwrap();
/// assert_eq!(ab.to_string(), "I,X,H,Y,S");
/// ```
impl Add for CliffordString {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner.add(&rhs.inner),
        }
    }
}

/// Concatenates another Clifford string onto the end of this one in place.
///
/// After `self += rhs`, `self` contains its original entries followed by the
/// entries of `rhs`.
impl AddAssign for CliffordString {
    fn add_assign(&mut self, rhs: Self) {
        self.inner.add_assign(&rhs.inner);
    }
}

/// Composes two Clifford strings element-wise (pairwise gate multiplication).
///
/// The Clifford gate at each qubit position is the composition of the
/// corresponding gates from `self` and `rhs`. When the strings have different
/// lengths, the shorter one is implicitly padded with identity gates.
///
/// # Examples
///
/// ```
/// use stim::CliffordString;
///
/// // S * S = Z on the first qubit, X * Z = Y on the second.
/// let product = "S,X,X".parse::<CliffordString>().unwrap()
///     * "S,Z,H,Z".parse::<CliffordString>().unwrap();
/// assert_eq!(product.to_string(), "Z,Y,SQRT_Y,Z");
/// ```
impl Mul<CliffordString> for CliffordString {
    type Output = Self;

    fn mul(self, rhs: CliffordString) -> Self::Output {
        Self {
            inner: self.inner.mul_clifford(&rhs.inner),
        }
    }
}

/// Composes another Clifford string element-wise into this one in place.
///
/// This is the in-place variant of pairwise multiplication. See the
/// `Mul<CliffordString>` impl for details.
impl MulAssign<CliffordString> for CliffordString {
    fn mul_assign(&mut self, rhs: CliffordString) {
        self.inner.mul_assign_clifford(&rhs.inner);
    }
}

/// Repeats the Clifford string's contents `rhs` times.
///
/// The result has length `self.len() * rhs`. For example, repeating
/// `"I,X,H"` by 3 gives `"I,X,H,I,X,H,I,X,H"`.
///
/// # Examples
///
/// ```
/// use stim::CliffordString;
///
/// let repeated = "I,X,H".parse::<CliffordString>().unwrap() * 3;
/// assert_eq!(repeated.to_string(), "I,X,H,I,X,H,I,X,H");
/// ```
impl Mul<u64> for CliffordString {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self::Output {
        Self {
            inner: self
                .inner
                .repeat(rhs)
                .expect("CliffordString repetition should fit platform size"),
        }
    }
}

/// Repeats the Clifford string's contents in place.
///
/// This is the in-place variant of repetition. After `self *= n`, `self`
/// contains its original contents repeated `n` times.
impl MulAssign<u64> for CliffordString {
    fn mul_assign(&mut self, rhs: u64) {
        self.inner
            .repeat_assign(rhs)
            .expect("CliffordString repetition should fit platform size");
    }
}

/// Left-multiplication by a repetition count: `n * clifford_string`.
///
/// This is the commutative counterpart to `CliffordString * u64`, allowing
/// `2 * string` as an alternative to `string * 2`.
///
/// # Examples
///
/// ```
/// use stim::CliffordString;
///
/// let repeated = 2 * "I,X,H".parse::<CliffordString>().unwrap();
/// assert_eq!(repeated.to_string(), "I,X,H,I,X,H");
/// ```
impl Mul<CliffordString> for u64 {
    type Output = CliffordString;

    fn mul(self, rhs: CliffordString) -> Self::Output {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::CliffordString;
    use crate::{Circuit, PauliString};
    use std::str::FromStr;

    #[test]
    fn clifford_string_core_examples_match_documented_behavior() {
        let c = CliffordString::new(3);
        assert_eq!(c.to_string(), "I,I,I");
        assert_eq!(format!("{c:?}"), "stim.CliffordString(\"I,I,I\")");
        assert_eq!(c.len(), 3);

        let s = "  X  ,   Y  ,  Z  , H_XZ , SQRT_X,C_XYZ,   "
            .parse::<CliffordString>()
            .unwrap();
        assert_eq!(s.to_string(), "X,Y,Z,H,SQRT_X,C_XYZ");
        assert_eq!(s.get(2).unwrap().name(), "Z");
        assert_eq!(s.get(-1).unwrap().name(), "C_XYZ");
        assert_eq!(
            s.slice(None, Some(-1), 1).unwrap().to_string(),
            "X,Y,Z,H,SQRT_X"
        );
        assert_eq!(s.slice(None, None, 2).unwrap().to_string(), "X,Z,SQRT_X");

        let from_pauli = CliffordString::from_pauli_string(&"-XYZ".parse::<PauliString>().unwrap());
        assert_eq!(from_pauli.to_string(), "X,Y,Z");

        let from_circuit = CliffordString::from_circuit(
            &Circuit::from_str(
                "
                H 0 1 2
                S 2 3
                TICK
                S 3
                I 6
                ",
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(from_circuit.to_string(), "H,H,C_ZYX,Z,I,I,I");
    }

    #[test]
    fn clifford_string_all_cliffords_and_outputs_match_documented_examples() {
        let cliffords = CliffordString::all_cliffords_string();
        assert_eq!(cliffords.len(), 24);
        assert_eq!(
            cliffords.slice(Some(0), Some(8), 1).unwrap().to_string(),
            "I,X,Y,Z,H_XY,S,S_DAG,H_NXY"
        );
        assert_eq!(
            cliffords.slice(Some(8), Some(16), 1).unwrap().to_string(),
            "H,SQRT_Y_DAG,H_NXZ,SQRT_Y,H_YZ,H_NYZ,SQRT_X,SQRT_X_DAG"
        );
        assert_eq!(
            cliffords.slice(Some(16), None, 1).unwrap().to_string(),
            "C_XYZ,C_XYNZ,C_NXYZ,C_XNYZ,C_ZYX,C_ZNYX,C_NZYX,C_ZYNX"
        );

        let (x_paulis, x_signs) = "I,Y,H,S".parse::<CliffordString>().unwrap().x_outputs();
        assert_eq!(x_paulis, "+XXZY".parse::<PauliString>().unwrap());
        assert_eq!(x_signs, vec![false, true, false, false]);
        assert_eq!(
            "I,Y,H,S"
                .parse::<CliffordString>()
                .unwrap()
                .x_outputs_bit_packed()
                .1,
            vec![2]
        );

        let (y_paulis, y_signs) = "I,X,H,S".parse::<CliffordString>().unwrap().y_outputs();
        assert_eq!(y_paulis, "+YYYX".parse::<PauliString>().unwrap());
        assert_eq!(y_signs, vec![false, true, true, true]);
        assert_eq!(
            "I,X,H,S"
                .parse::<CliffordString>()
                .unwrap()
                .y_outputs_bit_packed()
                .1,
            vec![14]
        );

        let (z_paulis, z_signs) = "I,Y,H,S".parse::<CliffordString>().unwrap().z_outputs();
        assert_eq!(z_paulis, "+ZZXZ".parse::<PauliString>().unwrap());
        assert_eq!(z_signs, vec![false, true, false, false]);
        assert_eq!(
            "I,Y,H,S"
                .parse::<CliffordString>()
                .unwrap()
                .z_outputs_bit_packed()
                .1,
            vec![2]
        );
    }

    #[test]
    fn clifford_string_arithmetic_and_powers_match_documented_examples() {
        assert_eq!(
            "I,X,H".parse::<CliffordString>().unwrap() + "Y,S".parse::<CliffordString>().unwrap(),
            "I,X,H,Y,S".parse::<CliffordString>().unwrap()
        );

        let mut concat = "I,X,H".parse::<CliffordString>().unwrap();
        concat += "Y,S".parse::<CliffordString>().unwrap();
        assert_eq!(concat, "I,X,H,Y,S".parse::<CliffordString>().unwrap());

        assert_eq!(
            "S,X,X".parse::<CliffordString>().unwrap()
                * "S,Z,H,Z".parse::<CliffordString>().unwrap(),
            "Z,Y,SQRT_Y,Z".parse::<CliffordString>().unwrap()
        );

        let mut mul_assign = "S,X,X".parse::<CliffordString>().unwrap();
        mul_assign *= "S,Z,H,Z".parse::<CliffordString>().unwrap();
        assert_eq!(
            mul_assign,
            "Z,Y,SQRT_Y,Z".parse::<CliffordString>().unwrap()
        );

        assert_eq!(
            "I,X,H".parse::<CliffordString>().unwrap() * 3,
            "I,X,H,I,X,H,I,X,H".parse::<CliffordString>().unwrap()
        );
        assert_eq!(
            2 * "I,X,H".parse::<CliffordString>().unwrap(),
            "I,X,H,I,X,H".parse::<CliffordString>().unwrap()
        );

        let mut repeated = "I,X,H".parse::<CliffordString>().unwrap();
        repeated *= 2;
        assert_eq!(repeated, "I,X,H,I,X,H".parse::<CliffordString>().unwrap());

        let mut p = "I,X,H,S,C_XYZ".parse::<CliffordString>().unwrap();
        p.ipow(3);
        assert_eq!(p, "I,X,H,S_DAG,I".parse::<CliffordString>().unwrap());
        p.ipow(2);
        assert_eq!(p, "I,I,I,Z,I".parse::<CliffordString>().unwrap());
        assert_eq!(p.pow(2), "I,I,I,I,I".parse::<CliffordString>().unwrap());
        p.ipow(2);
        assert_eq!(p, "I,I,I,I,I".parse::<CliffordString>().unwrap());
    }

    #[test]
    fn clifford_string_set_updates_single_gate_entries() {
        let mut s = "I,I,I,I,I".parse::<CliffordString>().unwrap();
        s.set(1, "H").unwrap();
        assert_eq!(s, "I,H,I,I,I".parse::<CliffordString>().unwrap());
        s.set(-1, crate::Gate::S).unwrap();
        assert_eq!(s, "I,H,I,I,S".parse::<CliffordString>().unwrap());
    }

    #[test]
    fn clifford_string_remaining_convenience_paths_are_covered() {
        let default = CliffordString::default();
        assert!(default.is_empty());

        let random = CliffordString::random(3);
        assert_eq!(random.clone().len(), 3);

        let step_error = "I,X"
            .parse::<CliffordString>()
            .unwrap()
            .slice(None, None, 0)
            .unwrap_err();
        assert!(step_error.message().contains("slice step cannot be zero"));
    }
}
