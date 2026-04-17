use std::fmt::{self, Display, Formatter};
use std::ops::{Add, AddAssign, Mul};
use std::pin::Pin;
use std::str::FromStr;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

type TableauNumpyBool = (
    Array2<bool>,
    Array2<bool>,
    Array2<bool>,
    Array2<bool>,
    Array1<bool>,
    Array1<bool>,
);
type TableauNumpyPacked = (
    Array2<u8>,
    Array2<u8>,
    Array2<u8>,
    Array2<u8>,
    Array1<u8>,
    Array1<u8>,
);

/// Circuit synthesis strategy used when converting a tableau back into a circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TableauSynthesisMethod {
    Elimination,
    GraphState,
    MppState,
    MppStateUnsigned,
}

impl TableauSynthesisMethod {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Elimination => "elimination",
            Self::GraphState => "graph_state",
            Self::MppState => "mpp_state",
            Self::MppStateUnsigned => "mpp_state_unsigned",
        }
    }
}

impl Display for TableauSynthesisMethod {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TableauSynthesisMethod {
    type Err = crate::StimError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "elimination" => Ok(Self::Elimination),
            "graph_state" => Ok(Self::GraphState),
            "mpp_state" => Ok(Self::MppState),
            "mpp_state_unsigned" => Ok(Self::MppStateUnsigned),
            _ => Err(crate::StimError::new(
                "tableau synthesis method not in ['elimination', 'graph_state', 'mpp_state', 'mpp_state_unsigned']",
            )),
        }
    }
}

/// A stabilizer tableau representing a Clifford operation.
///
/// A stabilizer tableau is a compact representation of a Clifford gate (or more generally,
/// any unitary operation in the Clifford group) that works by explicitly storing how the
/// operation conjugates each single-qubit Pauli generator into a composite Pauli product.
/// Specifically, for an n-qubit Clifford operation C, the tableau records:
///
/// - The n "X outputs": for each qubit k, the Pauli string `C · X_k · C†`
/// - The n "Z outputs": for each qubit k, the Pauli string `C · Z_k · C†`
///
/// Because the X and Z generators of the Pauli group generate the full Pauli group,
/// knowing how the operation conjugates each generator is sufficient to determine how it
/// conjugates *any* Pauli product. The Y outputs are implicitly determined since
/// `Y_k = i · X_k · Z_k`.
///
/// Tableaux are central to the Stim ecosystem. They provide efficient O(n²) simulation
/// of Clifford circuits (as opposed to the exponential cost of full state-vector simulation),
/// and serve as the bridge between several representations:
///
/// - **[`PauliString`](crate::PauliString)**: Pauli strings appear as the rows and columns
///   of a tableau. A Pauli product can itself be viewed as a trivial Clifford operation
///   (it negates anticommuting generators), and can be converted to a tableau via
///   [`PauliString::to_tableau`](crate::PauliString::to_tableau).
/// - **[`Circuit`](crate::Circuit)**: A Clifford circuit can be compiled into a single
///   tableau via [`Circuit::to_tableau`](crate::Circuit::to_tableau), and a tableau can
///   be decomposed back into a circuit via [`Tableau::to_circuit`].
/// - **[`GateData`](crate::GateData)**: The tableau for any named Clifford gate known to
///   Stim (e.g. `"H"`, `"CNOT"`, `"S"`) can be retrieved with [`Tableau::from_named_gate`].
///
/// # Examples
///
/// ```
/// // The Hadamard gate swaps X and Z:
/// let h = stim::Tableau::from_named_gate("H").unwrap();
/// assert_eq!(h.x_output(0), stim::PauliString::from_text("+Z").unwrap());
/// assert_eq!(h.z_output(0), stim::PauliString::from_text("+X").unwrap());
///
/// // Composing a random tableau with its inverse yields the identity:
/// let t = stim::Tableau::random(5);
/// let t_inv = t.inverse(false);
/// let product = t.then(&t_inv).unwrap();
/// assert_eq!(product, stim::Tableau::new(5));
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Tableau {
    pub(crate) inner: stim_cxx::Tableau,
}

impl Tableau {
    /// Creates the identity tableau over `num_qubits` qubits.
    ///
    /// The identity tableau represents the "do nothing" Clifford operation: every
    /// X generator maps to itself (`X_k → +X_k`) and every Z generator maps to
    /// itself (`Z_k → +Z_k`). This is the natural starting point when you want to
    /// incrementally build up a tableau by appending gates with [`Tableau::append`].
    ///
    /// # Examples
    ///
    /// ```
    /// let identity = stim::Tableau::new(3);
    /// assert_eq!(identity.num_qubits(), 3);
    ///
    /// // Every generator maps to itself:
    /// assert_eq!(
    ///     identity.x_output(0),
    ///     stim::PauliString::from_text("+X__").unwrap(),
    /// );
    /// assert_eq!(
    ///     identity.z_output(2),
    ///     stim::PauliString::from_text("+__Z").unwrap(),
    /// );
    /// ```
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            inner: stim_cxx::Tableau::new(num_qubits),
        }
    }

    /// Samples a uniformly random Clifford operation over `num_qubits` qubits and
    /// returns its tableau.
    ///
    /// The sampling algorithm produces a uniformly random element of the Clifford group
    /// on `num_qubits` qubits, using the method described in "Hadamard-free circuits
    /// expose the structure of the Clifford group" by Bravyi and Maslov
    /// (<https://arxiv.org/abs/2003.09412>). Each call produces a different random
    /// tableau (with astronomically high probability for non-trivial qubit counts).
    ///
    /// This is useful for randomized benchmarking, testing circuit equivalence, and
    /// generating random stabilizer states.
    ///
    /// # Examples
    ///
    /// ```
    /// let t = stim::Tableau::random(3);
    /// assert_eq!(t.num_qubits(), 3);
    /// ```
    #[must_use]
    pub fn random(num_qubits: usize) -> Self {
        Self {
            inner: stim_cxx::Tableau::random(num_qubits),
        }
    }

    /// Returns an iterator over all tableaux on `num_qubits` qubits.
    ///
    /// This method enumerates every distinct Clifford operation on the given number
    /// of qubits. The number of Clifford operations grows extremely fast: there are 24
    /// single-qubit Cliffords and 11,520 two-qubit Cliffords (including sign degrees
    /// of freedom). When `unsigned` is set to `true`, only tableaux where all output
    /// columns have a positive sign are yielded, which substantially reduces the count
    /// (e.g. 6 unsigned single-qubit Cliffords, 720 unsigned two-qubit Cliffords).
    ///
    /// This is primarily useful for exhaustive searches over small Clifford groups,
    /// such as verifying gate decompositions or testing properties of all single-qubit
    /// or two-qubit Cliffords.
    ///
    /// # Examples
    ///
    /// ```
    /// // There are 6 single-qubit Cliffords modulo Pauli signs:
    /// let all: Vec<_> = stim::Tableau::iter_all(1, true).collect();
    /// assert_eq!(all.len(), 6);
    ///
    /// // There are 24 single-qubit Cliffords including signs:
    /// assert_eq!(stim::Tableau::iter_all(1, false).count(), 24);
    /// ```
    #[must_use]
    pub fn iter_all(num_qubits: usize, unsigned: bool) -> crate::TableauIterator {
        crate::TableauIterator {
            inner: stim_cxx::Tableau::iter_all(num_qubits, unsigned),
        }
    }

    /// Converts a circuit into an equivalent stabilizer tableau.
    ///
    /// This compiles the entire circuit down to a single tableau that describes the
    /// net Clifford operation. The circuit must contain only Clifford gates; noise
    /// operations, measurements, and resets will cause an error unless the corresponding
    /// `ignore_*` flag is set to `true`, in which case those operations are skipped
    /// over as if they were not present.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The circuit contains noise operations and `ignore_noise` is `false`.
    /// - The circuit contains measurement operations and `ignore_measurement` is `false`.
    /// - The circuit contains reset operations and `ignore_reset` is `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0\nCNOT 0 1".parse().unwrap();
    /// let tableau = circuit.to_tableau(false, false, false).unwrap();
    /// assert_eq!(tableau.num_qubits(), 2);
    /// assert_eq!(
    ///     tableau.x_output(0),
    ///     stim::PauliString::from_text("+Z_").unwrap(),
    /// );
    /// ```
    /// Returns the tableau of a named Clifford gate known to Stim.
    ///
    /// Stim has a built-in library of common Clifford gates, including single-qubit
    /// gates like `"H"`, `"S"`, `"S_DAG"`, `"X"`, `"Y"`, `"Z"`, `"SQRT_X"`, `"SQRT_Y"`,
    /// and two-qubit gates like `"CNOT"` (aka `"CX"`), `"CZ"`, `"CY"`, `"SWAP"`,
    /// `"ISWAP"`, etc. This method looks up the gate by name and returns its tableau
    /// representation.
    ///
    /// Gate names are case-insensitive. Use [`GateData`](crate::GateData) to discover
    /// all available gate names.
    ///
    /// # Errors
    ///
    /// Returns an error if the given `name` does not correspond to a known gate.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// assert_eq!(h.x_output(0), stim::PauliString::from_text("+Z").unwrap());
    /// assert_eq!(h.z_output(0), stim::PauliString::from_text("+X").unwrap());
    ///
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    /// assert_eq!(cnot.num_qubits(), 2);
    /// assert_eq!(cnot.x_output(0), stim::PauliString::from_text("+XX").unwrap());
    /// ```
    pub fn from_named_gate(name: &str) -> crate::Result<Self> {
        let gate = crate::GateData::new(name)?;
        Self::from_gate(&gate)
    }

    /// Returns the tableau of a validated Clifford gate handle.
    pub fn from_gate(gate: &crate::Gate) -> crate::Result<Self> {
        stim_cxx::Tableau::from_named_gate(&gate.name())
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Creates a tableau representing the stabilizer state described by a state vector.
    ///
    /// Given a state vector of complex amplitudes (which must correspond to a stabilizer
    /// state reachable by Clifford operations from |0...0⟩), this method infers the
    /// tableau of a Clifford operation that, when applied to the all-zeros state,
    /// produces the given state. The state vector can be unnormalized.
    ///
    /// The `endian` parameter controls how qubit indices map to state-vector offsets:
    /// - `"little"`: higher-index qubits correspond to larger changes in the state
    ///   index (qubit 0 is the least significant bit).
    /// - `"big"`: higher-index qubits correspond to smaller changes in the state
    ///   index (qubit 0 is the most significant bit).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The state vector length is not a power of 2.
    /// - The state vector does not correspond to a stabilizer state.
    /// - The `endian` value is not `"little"` or `"big"`.
    ///
    /// # Examples
    ///
    /// ```
    /// let t = stim::Tableau::from_state_vector(
    ///     &[
    ///         stim::Complex32::new(0.5_f32.sqrt(), 0.0),
    ///         stim::Complex32::new(0.0, 0.5_f32.sqrt()),
    ///     ],
    ///     stim::Endian::Little,
    /// )
    /// .unwrap();
    /// assert_eq!(
    ///     t,
    ///     stim::Tableau::from_conjugated_generators(
    ///         &[stim::PauliString::from_text("+Z").unwrap()],
    ///         &[stim::PauliString::from_text("+Y").unwrap()],
    ///     )
    ///     .unwrap()
    /// );
    /// ```
    pub fn from_state_vector(
        state_vector: &[crate::Complex32],
        endian: crate::Endian,
    ) -> crate::Result<Self> {
        let mut flat = Vec::with_capacity(state_vector.len() * 2);
        for amp in state_vector {
            flat.push(amp.re);
            flat.push(amp.im);
        }
        stim_cxx::Tableau::from_state_vector_data(flat, endian.as_str())
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Creates a tableau from the unitary matrix of a Clifford operation.
    ///
    /// Given a square unitary matrix (as a slice of row vectors of complex amplitudes),
    /// this method determines the corresponding Clifford tableau. The matrix must be
    /// the unitary of a valid Clifford operation; otherwise an error is returned.
    ///
    /// Note that tableaux do not track global phase, so the resulting tableau is only
    /// determined up to a global phase factor. For example, the square of `SQRT_X`'s
    /// unitary might correspond to `-X` rather than `+X` in the tableau representation.
    ///
    /// The `endian` parameter controls how qubit indices map to matrix row/column
    /// indices:
    /// - `"little"`: higher-index qubits correspond to larger changes in row/col
    ///   indices.
    /// - `"big"`: higher-index qubits correspond to smaller changes in row/col
    ///   indices.
    ///
    /// For an n-qubit operation, this method performs O(n·4ⁿ) work.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The matrix is not square.
    /// - The matrix does not represent a Clifford operation.
    /// - The `endian` value is not `"little"` or `"big"`.
    ///
    /// # Examples
    ///
    /// ```
    /// let t = stim::Tableau::from_unitary_matrix(
    ///     &[
    ///         vec![stim::Complex32::new(1.0, 0.0), stim::Complex32::new(0.0, 0.0)],
    ///         vec![stim::Complex32::new(0.0, 0.0), stim::Complex32::new(0.0, 1.0)],
    ///     ],
    ///     stim::Endian::Little,
    /// )
    /// .unwrap();
    /// assert_eq!(
    ///     t,
    ///     stim::Tableau::from_conjugated_generators(
    ///         &[stim::PauliString::from_text("+Y").unwrap()],
    ///         &[stim::PauliString::from_text("+Z").unwrap()],
    ///     )
    ///     .unwrap()
    /// );
    /// ```
    pub fn from_unitary_matrix(
        matrix: &[Vec<crate::Complex32>],
        endian: crate::Endian,
    ) -> crate::Result<Self> {
        let n = matrix.len();
        if matrix.iter().any(|row| row.len() != n) {
            return Err(crate::StimError::new("matrix must be square"));
        }
        let mut flat = Vec::with_capacity(n * n * 2);
        for row in matrix {
            for cell in row {
                flat.push(cell.re);
                flat.push(cell.im);
            }
        }
        stim_cxx::Tableau::from_unitary_matrix_data(flat, endian.as_str())
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Builds a tableau from unpacked boolean ndarray matrices.
    ///
    /// The six arrays correspond to the four quadrants and two sign vectors defined in
    /// Aaronson and Gottesman's "Improved Simulation of Stabilizer Circuits"
    /// (<https://arxiv.org/abs/quant-ph/0406196>):
    ///
    /// - `x2x[i, j]`: whether the X output for input qubit `i` has an X or Y component
    ///   on output qubit `j` (i.e. `x_output_pauli(i, j) ∈ {1, 2}`).
    /// - `x2z[i, j]`: whether the X output for input qubit `i` has a Z or Y component
    ///   on output qubit `j` (i.e. `x_output_pauli(i, j) ∈ {2, 3}`).
    /// - `z2x[i, j]`: whether the Z output for input qubit `i` has an X or Y component
    ///   on output qubit `j` (i.e. `z_output_pauli(i, j) ∈ {1, 2}`).
    /// - `z2z[i, j]`: whether the Z output for input qubit `i` has a Z or Y component
    ///   on output qubit `j` (i.e. `z_output_pauli(i, j) ∈ {2, 3}`).
    /// - `x_signs[i]`: whether the X output for input qubit `i` is negative.
    /// - `z_signs[i]`: whether the Z output for input qubit `i` is negative.
    ///
    /// All four 2D arrays must be square with shape `(n, n)` and both sign vectors
    /// must have length `n`, where `n` is the number of qubits.
    ///
    /// # Errors
    ///
    /// Returns an error if the array dimensions are inconsistent or if the resulting
    /// generators violate the required commutation relationships for a valid tableau.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_ndarray(
    ///     ndarray::array![[false]].view(),
    ///     ndarray::array![[true]].view(),
    ///     ndarray::array![[true]].view(),
    ///     ndarray::array![[false]].view(),
    ///     ndarray::array![false].view(),
    ///     ndarray::array![false].view(),
    /// )
    /// .unwrap();
    /// assert_eq!(h, stim::Tableau::from_named_gate("H").unwrap());
    /// ```
    pub fn from_ndarray(
        x2x: ArrayView2<'_, bool>,
        x2z: ArrayView2<'_, bool>,
        z2x: ArrayView2<'_, bool>,
        z2z: ArrayView2<'_, bool>,
        x_signs: ArrayView1<'_, bool>,
        z_signs: ArrayView1<'_, bool>,
    ) -> crate::Result<Self> {
        let n = x_signs.len();
        if z_signs.len() != n {
            return Err(crate::StimError::new(
                "Inconsistent x_signs/z_signs lengths",
            ));
        }
        if x2x.nrows() != n
            || x2x.ncols() != n
            || x2z.nrows() != n
            || x2z.ncols() != n
            || z2x.nrows() != n
            || z2x.ncols() != n
            || z2z.nrows() != n
            || z2z.ncols() != n
        {
            return Err(crate::StimError::new(
                "tableau bit tables must all be square with side len(x_signs)",
            ));
        }

        let xs = (0..n)
            .map(|i| {
                let body: String = (0..n)
                    .map(|j| match (x2x[[i, j]], x2z[[i, j]]) {
                        (false, false) => '_',
                        (true, false) => 'X',
                        (true, true) => 'Y',
                        (false, true) => 'Z',
                    })
                    .collect();
                crate::PauliString::from_real_sign_and_body(if x_signs[i] { -1 } else { 1 }, &body)
            })
            .collect::<Vec<_>>();

        let zs = (0..n)
            .map(|i| {
                let body: String = (0..n)
                    .map(|j| match (z2x[[i, j]], z2z[[i, j]]) {
                        (false, false) => '_',
                        (true, false) => 'X',
                        (true, true) => 'Y',
                        (false, true) => 'Z',
                    })
                    .collect();
                crate::PauliString::from_real_sign_and_body(if z_signs[i] { -1 } else { 1 }, &body)
            })
            .collect::<Vec<_>>();

        Self::from_conjugated_generators(&xs, &zs)
    }

    /// Builds a tableau from bit-packed boolean ndarray matrices.
    ///
    /// This is the packed counterpart of [`Tableau::from_ndarray`]. Instead of using
    /// `bool` arrays, the bits are packed into `u8` bytes in little-endian order within
    /// each byte (bit `k` of the row is at byte `k / 8`, bit position `k % 8`).
    ///
    /// The 2D arrays have shape `(num_qubits, ceil(num_qubits / 8))` and the sign
    /// vectors have length `ceil(num_qubits / 8)`. The `num_qubits` parameter must be
    /// provided explicitly since it cannot be inferred from the packed dimensions alone.
    ///
    /// Bit packing is useful when working with large tableaux where memory and
    /// transfer overhead matter.
    ///
    /// # Errors
    ///
    /// Returns an error if the array dimensions do not match the expected packed shapes,
    /// or if the unpacked result fails validation as a valid tableau.
    pub fn from_ndarray_bit_packed(
        x2x: ArrayView2<'_, u8>,
        x2z: ArrayView2<'_, u8>,
        z2x: ArrayView2<'_, u8>,
        z2z: ArrayView2<'_, u8>,
        x_signs: ArrayView1<'_, u8>,
        z_signs: ArrayView1<'_, u8>,
        num_qubits: usize,
    ) -> crate::Result<Self> {
        let expected_row_bytes = num_qubits.div_ceil(8);
        if x2x.nrows() != num_qubits
            || x2x.ncols() != expected_row_bytes
            || x2z.nrows() != num_qubits
            || x2z.ncols() != expected_row_bytes
            || z2x.nrows() != num_qubits
            || z2x.ncols() != expected_row_bytes
            || z2z.nrows() != num_qubits
            || z2z.ncols() != expected_row_bytes
        {
            return Err(crate::StimError::new(
                "bit-packed tableau tables must have len(num_qubits) rows of len(ceil(num_qubits/8))",
            ));
        }
        if x_signs.len() != expected_row_bytes || z_signs.len() != expected_row_bytes {
            return Err(crate::StimError::new(
                "bit-packed tableau sign vectors must have len(ceil(num_qubits/8))",
            ));
        }

        let unpack_row = |row: ArrayView1<'_, u8>| -> Array1<bool> {
            Array1::from_iter((0..num_qubits).map(|k| ((row[k / 8] >> (k % 8)) & 1) != 0))
        };
        let unpack_signs = |packed: ArrayView1<'_, u8>| -> Array1<bool> {
            Array1::from_iter((0..num_qubits).map(|k| ((packed[k / 8] >> (k % 8)) & 1) != 0))
        };

        let unpack_table = |table: ArrayView2<'_, u8>| -> Array2<bool> {
            let mut bits = Vec::with_capacity(num_qubits * num_qubits);
            for row in table.rows() {
                bits.extend(unpack_row(row));
            }
            Array2::from_shape_vec((num_qubits, num_qubits), bits)
                .expect("unpacked tableau arrays should be rectangular")
        };

        let x2x_u = unpack_table(x2x);
        let x2z_u = unpack_table(x2z);
        let z2x_u = unpack_table(z2x);
        let z2z_u = unpack_table(z2z);
        let x_signs_u = unpack_signs(x_signs);
        let z_signs_u = unpack_signs(z_signs);

        Self::from_ndarray(
            x2x_u.view(),
            x2z_u.view(),
            z2x_u.view(),
            z2z_u.view(),
            x_signs_u.view(),
            z_signs_u.view(),
        )
    }

    /// Builds a tableau from explicit conjugated X and Z generators.
    ///
    /// This is the most direct way to construct a tableau: you specify exactly what
    /// each X and Z generator conjugates to under the Clifford operation. The `xs`
    /// slice provides the output of conjugating `X_0, X_1, ..., X_{n-1}`, and the
    /// `zs` slice provides the output of conjugating `Z_0, Z_1, ..., Z_{n-1}`.
    ///
    /// The method validates that the provided generators form a well-formed tableau
    /// by checking:
    /// - `xs` and `zs` have the same length `n`.
    /// - Every Pauli string in `xs` and `zs` has length `n`.
    /// - The outputs satisfy the required commutation relationships (e.g. `xs[i]`
    ///   anticommutes with `zs[i]`, and `xs[i]` commutes with `zs[j]` for `i ≠ j`).
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths are inconsistent or the commutation
    /// relationships are violated.
    ///
    /// # Examples
    ///
    /// ```
    /// // Construct the Hadamard gate: X → Z, Z → X
    /// let h = stim::Tableau::from_conjugated_generators(
    ///     &[stim::PauliString::from_text("+Z").unwrap()],
    ///     &[stim::PauliString::from_text("+X").unwrap()],
    /// )
    /// .unwrap();
    /// assert_eq!(h, stim::Tableau::from_named_gate("H").unwrap());
    ///
    /// // Construct the 3-qubit identity:
    /// let id3 = stim::Tableau::from_conjugated_generators(
    ///     &[
    ///         stim::PauliString::from_text("X__").unwrap(),
    ///         stim::PauliString::from_text("_X_").unwrap(),
    ///         stim::PauliString::from_text("__X").unwrap(),
    ///     ],
    ///     &[
    ///         stim::PauliString::from_text("Z__").unwrap(),
    ///         stim::PauliString::from_text("_Z_").unwrap(),
    ///         stim::PauliString::from_text("__Z").unwrap(),
    ///     ],
    /// )
    /// .unwrap();
    /// assert_eq!(id3, stim::Tableau::new(3));
    /// ```
    pub fn from_conjugated_generators(
        xs: &[crate::PauliString],
        zs: &[crate::PauliString],
    ) -> crate::Result<Self> {
        stim_cxx::Tableau::from_conjugated_generator_texts(
            xs.iter().map(ToString::to_string).collect(),
            zs.iter().map(ToString::to_string).collect(),
        )
        .map(|inner| Self { inner })
        .map_err(crate::StimError::from)
    }

    /// Creates a tableau representing a state with the given stabilizer generators.
    ///
    /// Given a list of Pauli strings that stabilize a quantum state, this method
    /// constructs a tableau which, when applied to the all-zeros state |0...0⟩,
    /// produces a state with those stabilizers. The result guarantees that
    /// `result.z_output(k)` equals the k-th independent stabilizer from the input.
    ///
    /// Stabilizers may have different lengths; shorter ones are padded with identity
    /// terms to match the longest.
    ///
    /// # Arguments
    ///
    /// * `stabilizers` - The Pauli string stabilizers describing the target state.
    /// * `allow_redundant` - When `false` (default behavior), all stabilizers must be
    ///   independent (no stabilizer is a product of the others). When `true`, redundant
    ///   stabilizers are silently ignored.
    /// * `allow_underconstrained` - When `false` (default behavior), the stabilizers
    ///   must form a complete generating set (exactly n independent stabilizers for an
    ///   n-qubit state). When `true`, missing degrees of freedom are filled in
    ///   arbitrarily.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A stabilizer is redundant but `allow_redundant` is `false`.
    /// - The stabilizers are contradictory (e.g. both `+Z` and `-Z`).
    /// - The stabilizers anticommute (e.g. both `+Z` and `+X`).
    /// - The stabilizers are underconstrained but `allow_underconstrained` is `false`.
    /// - A stabilizer has an imaginary sign (`i` or `-i`).
    ///
    /// # Examples
    ///
    /// ```
    /// let t = stim::Tableau::from_stabilizers(
    ///     &[
    ///         stim::PauliString::from_text("XX").unwrap(),
    ///         stim::PauliString::from_text("ZZ").unwrap(),
    ///     ],
    ///     false,
    ///     false,
    /// )
    /// .unwrap();
    /// // The Z outputs of the resulting tableau are the stabilizers:
    /// assert_eq!(t.z_output(0), stim::PauliString::from_text("+XX").unwrap());
    /// assert_eq!(t.z_output(1), stim::PauliString::from_text("+ZZ").unwrap());
    /// ```
    pub fn from_stabilizers(
        stabilizers: &[crate::PauliString],
        allow_redundant: bool,
        allow_underconstrained: bool,
    ) -> crate::Result<Self> {
        stim_cxx::Tableau::from_stabilizer_texts(
            stabilizers.iter().map(ToString::to_string).collect(),
            allow_redundant,
            allow_underconstrained,
        )
        .map(|inner| Self { inner })
        .map_err(crate::StimError::from)
    }

    /// Returns the result of composing two tableaux: first `self`, then `second`.
    ///
    /// If `self` represents the Clifford operation C1 and `second` represents C2,
    /// then `self.then(&second)` represents the operation "first apply C1, then apply
    /// C2", which has the unitary `C2 · C1`. This means the result conjugates a Pauli
    /// P as `second.conjugate(&self.conjugate(&P))`.
    ///
    /// Note the distinction from the `*` operator: `a * b` is equivalent to
    /// `b.then(&a)` (i.e. `*` uses standard matrix multiplication order where the
    /// right operand is applied first).
    ///
    /// Both tableaux must act on the same number of qubits.
    ///
    /// # Errors
    ///
    /// Returns an error if `self.num_qubits() != second.num_qubits()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// let z = stim::Tableau::from_named_gate("Z").unwrap();
    /// let combined = h.then(&z).unwrap();
    /// let x = stim::PauliString::from_text("+X").unwrap();
    /// assert_eq!(combined.conjugate(&x), z.conjugate(&h.conjugate(&x)));
    /// ```
    pub fn then(&self, second: &Self) -> crate::Result<Self> {
        self.inner
            .then(&second.inner)
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Returns the number of qubits that the tableau's Clifford operation acts on.
    ///
    /// This is determined by the number of X/Z generator pairs stored in the tableau.
    /// For example, a CNOT gate has `num_qubits() == 2` and a Hadamard gate has
    /// `num_qubits() == 1`.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Appends a gate operation onto selected target qubits, mutating this tableau
    /// in place.
    ///
    /// This is equivalent to composing `self` with the given `gate` applied to
    /// the specified `targets`. After the call, `self` represents the combined
    /// operation "first the old self, then the gate on the given targets".
    ///
    /// The time cost is O(n·m²) where n = `self.num_qubits()` and m = `gate.num_qubits()`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `targets.len() != gate.num_qubits()` (target count doesn't match gate size).
    /// - Any two targets collide (the same qubit appears twice).
    /// - Any target index is `>= self.num_qubits()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut t = stim::Tableau::new(2);
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// t.append(&h, &[1]).unwrap();
    /// assert_eq!(t.x_output(1), stim::PauliString::from_text("+_Z").unwrap());
    ///
    /// // Three CNOTs in a row make a SWAP:
    /// let mut t = stim::Tableau::new(2);
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    /// t.append(&cnot, &[0, 1]).unwrap();
    /// t.append(&cnot, &[1, 0]).unwrap();
    /// t.append(&cnot, &[0, 1]).unwrap();
    /// assert_eq!(t, stim::Tableau::from_named_gate("SWAP").unwrap());
    /// ```
    pub fn append(&mut self, gate: &Self, targets: &[usize]) -> crate::Result<()> {
        Pin::new(&mut self.inner)
            .append(&gate.inner, targets)
            .map_err(crate::StimError::from)
    }

    /// Prepends a gate operation onto selected target qubits, mutating this tableau
    /// in place.
    ///
    /// This is equivalent to composing the given `gate` (applied to the specified
    /// `targets`) *before* `self`. After the call, `self` represents the combined
    /// operation "first the gate on the given targets, then the old self".
    ///
    /// The time cost is O(n·m²) where n = `self.num_qubits()` and m = `gate.num_qubits()`.
    ///
    /// Note the difference from [`append`](Self::append): appending places the gate
    /// *after* the current operation, while prepending places it *before*.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `targets.len() != gate.num_qubits()` (target count doesn't match gate size).
    /// - Any two targets collide (the same qubit appears twice).
    /// - Any target index is `>= self.num_qubits()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut t = stim::Tableau::from_named_gate("H").unwrap();
    /// t.prepend(&stim::Tableau::from_named_gate("X").unwrap(), &[0]).unwrap();
    /// assert_eq!(t, stim::Tableau::from_named_gate("SQRT_Y_DAG").unwrap());
    /// ```
    pub fn prepend(&mut self, gate: &Self, targets: &[usize]) -> crate::Result<()> {
        Pin::new(&mut self.inner)
            .prepend(&gate.inner, targets)
            .map_err(crate::StimError::from)
    }

    /// Returns the number of qubits acted on by the tableau.
    ///
    /// This is an alias of [`num_qubits`](Self::num_qubits), provided to satisfy
    /// common Rust collection conventions.
    ///
    /// # Examples
    ///
    /// ```
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    /// assert_eq!(cnot.len(), 2);
    /// assert_eq!(stim::Tableau::new(0).len(), 0);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_qubits()
    }

    /// Returns whether the tableau acts on zero qubits (i.e. is a trivial 0-qubit
    /// identity operation).
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::Tableau::new(0).is_empty());
    /// assert!(!stim::Tableau::from_named_gate("H").unwrap().is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Computes the inverse of the tableau.
    ///
    /// The inverse T⁻¹ of a tableau T is the unique tableau satisfying
    /// `T * T⁻¹ = T⁻¹ * T = I`, where I is the identity tableau. If T represents
    /// a unitary Clifford operation C, then T⁻¹ represents C†.
    ///
    /// When `unsigned` is `true`, the method skips computing the signs of the output
    /// Pauli strings and instead sets them all to positive. This is a useful
    /// optimization because computing signs takes O(n³) time while the rest of the
    /// inverse computation is only O(n²), where n is the number of qubits. Use this
    /// when you only need the Pauli terms (not the signs).
    ///
    /// # Examples
    ///
    /// ```
    /// let s = stim::Tableau::from_named_gate("S").unwrap();
    /// let s_dag = stim::Tableau::from_named_gate("S_DAG").unwrap();
    /// assert_eq!(s.inverse(false), s_dag);
    ///
    /// // Multiplying by the inverse gives the identity:
    /// let t = stim::Tableau::random(5);
    /// let identity = t.then(&t.inverse(false)).unwrap();
    /// assert_eq!(identity, stim::Tableau::new(5));
    /// ```
    #[must_use]
    pub fn inverse(&self, unsigned: bool) -> Self {
        Self {
            inner: self.inner.inverse(unsigned),
        }
    }

    /// Raises the tableau to an integer power.
    ///
    /// Large powers are reached efficiently using repeated squaring. Negative powers
    /// are computed by first inverting the tableau and then exponentiating. A power of
    /// zero returns the identity tableau.
    ///
    /// Because Clifford operations on n qubits form a finite group, the result is
    /// always periodic. For example, `S.raised_to(4)` returns the identity because the
    /// S gate has order 4.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = stim::Tableau::from_named_gate("S").unwrap();
    /// assert_eq!(s.raised_to(0), stim::Tableau::new(1));
    /// assert_eq!(s.raised_to(1), s);
    /// assert_eq!(s.raised_to(2), stim::Tableau::from_named_gate("Z").unwrap());
    /// assert_eq!(s.raised_to(-1), stim::Tableau::from_named_gate("S_DAG").unwrap());
    /// assert_eq!(s.raised_to(5), s); // S has order 4
    /// ```
    #[must_use]
    pub fn raised_to(&self, exponent: i64) -> Self {
        Self {
            inner: self.inner.raised_to(exponent),
        }
    }

    /// Alias of [`raised_to`](Self::raised_to).
    ///
    /// Raises the tableau to an integer power. See [`raised_to`](Self::raised_to) for
    /// full documentation.
    #[must_use]
    pub fn pow(&self, exponent: i64) -> Self {
        self.raised_to(exponent)
    }

    /// Returns just the sign of the result of conjugating an X generator.
    ///
    /// This operation runs in constant time O(1). It returns `+1` if the X output
    /// for the given `target` qubit has a positive sign, or `-1` if it has a negative
    /// sign.
    ///
    /// # Errors
    ///
    /// Returns an error if `target >= self.num_qubits()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = stim::Tableau::from_named_gate("S").unwrap();
    /// assert_eq!(s.x_sign(0).unwrap(), 1);      // S: X → +Y
    ///
    /// let s_dag = stim::Tableau::from_named_gate("S_DAG").unwrap();
    /// assert_eq!(s_dag.x_sign(0).unwrap(), -1);  // S†: X → -Y
    /// ```
    pub fn x_sign(&self, target: usize) -> crate::Result<i32> {
        self.inner.x_sign(target).map_err(crate::StimError::from)
    }

    /// Returns just the sign of the result of conjugating a Y generator.
    ///
    /// Unlike [`x_sign`](Self::x_sign) and [`z_sign`](Self::z_sign), this operation
    /// runs in linear time O(n) rather than constant time. The Y generator must be
    /// computed by multiplying the X and Z outputs (`Y_k = i · X_k · Z_k`), and the
    /// resulting sign depends on all terms in both output strings.
    ///
    /// Returns `+1` or `-1`.
    ///
    /// # Errors
    ///
    /// Returns an error if `target >= self.num_qubits()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = stim::Tableau::from_named_gate("S").unwrap();
    /// assert_eq!(s.y_sign(0).unwrap(), -1);      // S: Y → -Y (via X→Y, Z→Z)
    ///
    /// let s_dag = stim::Tableau::from_named_gate("S_DAG").unwrap();
    /// assert_eq!(s_dag.y_sign(0).unwrap(), 1);   // S†: Y → +Y (via X→-Y, Z→Z)
    /// ```
    pub fn y_sign(&self, target: usize) -> crate::Result<i32> {
        self.inner.y_sign(target).map_err(crate::StimError::from)
    }

    /// Returns just the sign of the result of conjugating a Z generator.
    ///
    /// This operation runs in constant time O(1). It returns `+1` if the Z output
    /// for the given `target` qubit has a positive sign, or `-1` if it has a negative
    /// sign.
    ///
    /// # Errors
    ///
    /// Returns an error if `target >= self.num_qubits()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let sqrt_x = stim::Tableau::from_named_gate("SQRT_X").unwrap();
    /// assert_eq!(sqrt_x.z_sign(0).unwrap(), -1);   // √X: Z → -Y
    ///
    /// let sqrt_x_dag = stim::Tableau::from_named_gate("SQRT_X_DAG").unwrap();
    /// assert_eq!(sqrt_x_dag.z_sign(0).unwrap(), 1); // √X†: Z → +Y
    /// ```
    pub fn z_sign(&self, target: usize) -> crate::Result<i32> {
        self.inner.z_sign(target).map_err(crate::StimError::from)
    }

    /// Returns the Pauli code of a single entry in the X output, in constant time.
    ///
    /// This is a constant-time O(1) version of looking up a specific Pauli within
    /// the full X output string: equivalent to indexing into
    /// `self.x_output(input_index)` at position `output_index`, but without
    /// constructing the entire [`PauliString`](crate::PauliString).
    ///
    /// The returned integer encodes the Pauli as: 0 = I, 1 = X, 2 = Y, 3 = Z.
    ///
    /// # Errors
    ///
    /// Returns an error if either index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    /// // CNOT: X₀ → X₀X₁, so x_output_pauli(0, 0) = X = 1, x_output_pauli(0, 1) = X = 1
    /// assert_eq!(cnot.x_output_pauli(0, 0).unwrap(), 1);
    /// assert_eq!(cnot.x_output_pauli(0, 1).unwrap(), 1);
    /// // CNOT: X₁ → _X₁, so x_output_pauli(1, 0) = I = 0
    /// assert_eq!(cnot.x_output_pauli(1, 0).unwrap(), 0);
    /// ```
    pub fn x_output_pauli(&self, input_index: usize, output_index: usize) -> crate::Result<u8> {
        self.inner
            .x_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns the Pauli code of a single entry in the Y output, in constant time.
    ///
    /// This is a constant-time O(1) version of looking up a specific Pauli within
    /// the full Y output string: equivalent to indexing into
    /// `self.y_output(input_index)` at position `output_index`, but without
    /// constructing the entire [`PauliString`](crate::PauliString).
    ///
    /// The returned integer encodes the Pauli as: 0 = I, 1 = X, 2 = Y, 3 = Z.
    ///
    /// # Errors
    ///
    /// Returns an error if either index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    /// // CNOT: Y₀ → Y₀X₁, so y_output_pauli(0, 0) = Y = 2, y_output_pauli(0, 1) = X = 1
    /// assert_eq!(cnot.y_output_pauli(0, 0).unwrap(), 2);
    /// assert_eq!(cnot.y_output_pauli(0, 1).unwrap(), 1);
    /// ```
    pub fn y_output_pauli(&self, input_index: usize, output_index: usize) -> crate::Result<u8> {
        self.inner
            .y_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns the Pauli code of a single entry in the Z output, in constant time.
    ///
    /// This is a constant-time O(1) version of looking up a specific Pauli within
    /// the full Z output string: equivalent to indexing into
    /// `self.z_output(input_index)` at position `output_index`, but without
    /// constructing the entire [`PauliString`](crate::PauliString).
    ///
    /// The returned integer encodes the Pauli as: 0 = I, 1 = X, 2 = Y, 3 = Z.
    ///
    /// # Errors
    ///
    /// Returns an error if either index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    /// // CNOT: Z₀ → Z₀, so z_output_pauli(0, 0) = Z = 3, z_output_pauli(0, 1) = I = 0
    /// assert_eq!(cnot.z_output_pauli(0, 0).unwrap(), 3);
    /// assert_eq!(cnot.z_output_pauli(0, 1).unwrap(), 0);
    /// // CNOT: Z₁ → Z₀Z₁
    /// assert_eq!(cnot.z_output_pauli(1, 0).unwrap(), 3);
    /// assert_eq!(cnot.z_output_pauli(1, 1).unwrap(), 3);
    /// ```
    pub fn z_output_pauli(&self, input_index: usize, output_index: usize) -> crate::Result<u8> {
        self.inner
            .z_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns a single Pauli code from the inverse tableau's X output, in constant
    /// time.
    ///
    /// This is equivalent to `self.inverse(false).x_output_pauli(input_index, output_index)`
    /// but avoids computing the full inverse tableau. The Pauli terms (not the sign)
    /// of the inverse can be read in O(1) time per entry.
    ///
    /// The returned integer encodes the Pauli as: 0 = I, 1 = X, 2 = Y, 3 = Z.
    ///
    /// # Errors
    ///
    /// Returns an error if either index is out of bounds.
    pub fn inverse_x_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> crate::Result<u8> {
        self.inner
            .inverse_x_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns a single Pauli code from the inverse tableau's Y output, in constant
    /// time.
    ///
    /// This is equivalent to `self.inverse(false).y_output_pauli(input_index, output_index)`
    /// but avoids computing the full inverse tableau. The Pauli terms (not the sign)
    /// of the inverse can be read in O(1) time per entry.
    ///
    /// The returned integer encodes the Pauli as: 0 = I, 1 = X, 2 = Y, 3 = Z.
    ///
    /// # Errors
    ///
    /// Returns an error if either index is out of bounds.
    pub fn inverse_y_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> crate::Result<u8> {
        self.inner
            .inverse_y_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns a single Pauli code from the inverse tableau's Z output, in constant
    /// time.
    ///
    /// This is equivalent to `self.inverse(false).z_output_pauli(input_index, output_index)`
    /// but avoids computing the full inverse tableau. The Pauli terms (not the sign)
    /// of the inverse can be read in O(1) time per entry.
    ///
    /// The returned integer encodes the Pauli as: 0 = I, 1 = X, 2 = Y, 3 = Z.
    ///
    /// # Errors
    ///
    /// Returns an error if either index is out of bounds.
    pub fn inverse_z_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> crate::Result<u8> {
        self.inner
            .inverse_z_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns the full Pauli string result of conjugating an X generator by the
    /// tableau's Clifford operation.
    ///
    /// For a tableau representing Clifford operation C, this returns the Pauli string
    /// `C · X_{target} · C†`. This is the `target`-th row of the "X half" of the
    /// tableau.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// // H conjugates X to Z:
    /// assert_eq!(h.x_output(0), stim::PauliString::from_text("+Z").unwrap());
    ///
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    /// // CNOT conjugates X₀ to X₀⊗X₁:
    /// assert_eq!(cnot.x_output(0), stim::PauliString::from_text("+XX").unwrap());
    /// // CNOT conjugates X₁ to I⊗X₁:
    /// assert_eq!(cnot.x_output(1), stim::PauliString::from_text("+_X").unwrap());
    /// ```
    #[must_use]
    pub fn x_output(&self, target: usize) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.x_output(target),
            imag: false,
        }
    }

    /// Returns the full Pauli string result of conjugating a Y generator by the
    /// tableau's Clifford operation.
    ///
    /// For a tableau representing Clifford operation C, this returns the Pauli string
    /// `C · Y_{target} · C†`. Since `Y_k = i · X_k · Z_k`, the Y output is derived
    /// from the X and Z outputs.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// // H conjugates Y to -Y:
    /// assert_eq!(h.y_output(0), stim::PauliString::from_text("-Y").unwrap());
    ///
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    /// assert_eq!(cnot.y_output(0), stim::PauliString::from_text("+YX").unwrap());
    /// assert_eq!(cnot.y_output(1), stim::PauliString::from_text("+ZY").unwrap());
    /// ```
    #[must_use]
    pub fn y_output(&self, target: usize) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.y_output(target),
            imag: false,
        }
    }

    /// Returns the full Pauli string result of conjugating a Z generator by the
    /// tableau's Clifford operation.
    ///
    /// For a tableau representing Clifford operation C, this returns the Pauli string
    /// `C · Z_{target} · C†`. This is the `target`-th row of the "Z half" of the
    /// tableau. When the tableau is interpreted as a stabilizer state preparation
    /// (applying C to |0...0⟩), the Z outputs are the stabilizers of that state.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// // H conjugates Z to X:
    /// assert_eq!(h.z_output(0), stim::PauliString::from_text("+X").unwrap());
    ///
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    /// assert_eq!(cnot.z_output(0), stim::PauliString::from_text("+Z_").unwrap());
    /// assert_eq!(cnot.z_output(1), stim::PauliString::from_text("+ZZ").unwrap());
    /// ```
    #[must_use]
    pub fn z_output(&self, target: usize) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.z_output(target),
            imag: false,
        }
    }

    /// Conjugates a single-qubit X Pauli generator by the inverse of the tableau,
    /// returning the full Pauli string.
    ///
    /// This is a faster alternative to `self.inverse(unsigned).x_output(target)`,
    /// because it avoids computing the full inverse tableau. The Pauli body (terms)
    /// of the result can be computed in O(n) time, and the sign in O(n²) time (or
    /// skipped if `unsigned` is `true`).
    ///
    /// When `unsigned` is `true`, the sign of the result is forced to positive,
    /// saving the O(n²) sign computation.
    #[must_use]
    pub fn inverse_x_output(&self, target: usize, unsigned: bool) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.inverse_x_output(target, unsigned),
            imag: false,
        }
    }

    /// Conjugates a single-qubit Y Pauli generator by the inverse of the tableau,
    /// returning the full Pauli string.
    ///
    /// This is a faster alternative to `self.inverse(unsigned).y_output(target)`,
    /// because it avoids computing the full inverse tableau. See
    /// [`inverse_x_output`](Self::inverse_x_output) for performance details.
    ///
    /// When `unsigned` is `true`, the sign of the result is forced to positive.
    #[must_use]
    pub fn inverse_y_output(&self, target: usize, unsigned: bool) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.inverse_y_output(target, unsigned),
            imag: false,
        }
    }

    /// Conjugates a single-qubit Z Pauli generator by the inverse of the tableau,
    /// returning the full Pauli string.
    ///
    /// This is a faster alternative to `self.inverse(unsigned).z_output(target)`,
    /// because it avoids computing the full inverse tableau. See
    /// [`inverse_x_output`](Self::inverse_x_output) for performance details.
    ///
    /// When `unsigned` is `true`, the sign of the result is forced to positive.
    #[must_use]
    pub fn inverse_z_output(&self, target: usize, unsigned: bool) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.inverse_z_output(target, unsigned),
            imag: false,
        }
    }

    /// Conjugates a Pauli string by the tableau's Clifford operation.
    ///
    /// If P is a Pauli product and C is the Clifford operation represented by this
    /// tableau, this method returns `Q = C · P · C†`, the equivalent Pauli product
    /// after the operation. This is the core operation that tableaux are designed
    /// to perform efficiently.
    ///
    /// The conjugation works because if you have a Pauli P before a Clifford C:
    ///
    /// ```text
    /// C · P = C · P · (C† · C) = (C · P · C†) · C = Q · C
    /// ```
    ///
    /// So measuring P before C is equivalent to measuring Q after C.
    ///
    /// The Pauli string may have fewer qubits than the tableau (it is implicitly
    /// padded with identity), or the same number.
    ///
    /// # Examples
    ///
    /// ```
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    ///
    /// // X on the control qubit spreads to both qubits:
    /// assert_eq!(
    ///     cnot.conjugate(&stim::PauliString::from_text("X_").unwrap()),
    ///     stim::PauliString::from_text("XX").unwrap(),
    /// );
    ///
    /// // Z on the target qubit spreads to both qubits:
    /// assert_eq!(
    ///     cnot.conjugate(&stim::PauliString::from_text("_Z").unwrap()),
    ///     stim::PauliString::from_text("ZZ").unwrap(),
    /// );
    ///
    /// // Conjugation preserves signs correctly:
    /// assert_eq!(
    ///     cnot.conjugate(&stim::PauliString::from_text("YY").unwrap()),
    ///     stim::PauliString::from_text("-XZ").unwrap(),
    /// );
    /// ```
    #[must_use]
    pub fn conjugate(&self, pauli_string: &crate::PauliString) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.conjugate_pauli_string(&pauli_string.inner),
            imag: pauli_string.imag,
        }
    }

    /// Alias of [`conjugate`](Self::conjugate).
    ///
    /// Returns the result of conjugating the given Pauli string by this tableau's
    /// Clifford operation. This mirrors Python's `__call__` protocol: in the Python
    /// API, `tableau(pauli_string)` conjugates the Pauli string. See
    /// [`conjugate`](Self::conjugate) for full documentation.
    #[must_use]
    pub fn call(&self, pauli_string: &crate::PauliString) -> crate::PauliString {
        self.conjugate(pauli_string)
    }

    /// Returns the stabilizer generators of the tableau, optionally canonicalized.
    ///
    /// The stabilizer generators of the tableau are its Z outputs. When the tableau is
    /// viewed as a state preparation operation (applying C to |0...0⟩), these Z outputs
    /// are exactly the stabilizers of the resulting state.
    ///
    /// When `canonicalize` is `true`, the generators are rewritten into a standard
    /// canonical form via Gaussian elimination. Two stabilizer states produce the same
    /// canonical generators if and only if they describe the same quantum state. This
    /// is useful for comparing states that may have different generator choices.
    ///
    /// The canonical form is computed by pivoting on standard generators in the order
    /// X₀, Z₀, X₁, Z₁, X₂, Z₂, etc., performing elimination to ensure each pivot
    /// generator appears in exactly one stabilizer.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal Pauli string conversion fails (should not
    /// occur for well-formed tableaux).
    ///
    /// # Examples
    ///
    /// ```
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    ///
    /// // Raw stabilizers (Z outputs directly):
    /// assert_eq!(
    ///     cnot.to_stabilizers(false).unwrap(),
    ///     vec![
    ///         stim::PauliString::from_text("+Z_").unwrap(),
    ///         stim::PauliString::from_text("+ZZ").unwrap(),
    ///     ]
    /// );
    ///
    /// // Canonicalized stabilizers:
    /// assert_eq!(
    ///     cnot.to_stabilizers(true).unwrap(),
    ///     vec![
    ///         stim::PauliString::from_text("+Z_").unwrap(),
    ///         stim::PauliString::from_text("+_Z").unwrap(),
    ///     ]
    /// );
    /// ```
    pub fn to_stabilizers(&self, canonicalize: bool) -> crate::Result<Vec<crate::PauliString>> {
        self.inner
            .to_stabilizer_texts(canonicalize)
            .into_iter()
            .map(|text| crate::PauliString::from_text(&text))
            .collect()
    }

    /// Converts the tableau into a circuit using the default decomposition method.
    pub fn to_circuit(&self) -> crate::Result<crate::Circuit> {
        self.to_circuit_with_method(TableauSynthesisMethod::Elimination)
    }

    /// Converts the tableau into a circuit with a specific decomposition method.
    pub fn to_circuit_with_method(
        &self,
        method: TableauSynthesisMethod,
    ) -> crate::Result<crate::Circuit> {
        self.inner
            .to_circuit(method.as_str())
            .map(crate::Circuit::from_inner)
            .map_err(crate::StimError::from)
    }

    /// Converts the tableau into a Pauli string when possible.
    pub fn to_pauli_string(&self) -> crate::Result<crate::PauliString> {
        self.inner
            .to_pauli_string()
            .map(|inner| crate::PauliString { inner, imag: false })
            .map_err(crate::StimError::from)
    }

    /// Converts the tableau into a unitary matrix.
    pub fn to_unitary_matrix(&self, endian: crate::Endian) -> Vec<Vec<crate::Complex32>> {
        let flat = self
            .inner
            .to_unitary_matrix_data(endian.as_str())
            .expect("typed endian should be accepted by stim-cxx");
        let n = 1usize << self.num_qubits();
        assert_eq!(
            flat.len(),
            n * n * 2,
            "unexpected flat unitary matrix size from stim-cxx"
        );
        let mut result = vec![vec![crate::Complex32::new(0.0, 0.0); n]; n];
        for (row_index, row) in result.iter_mut().enumerate() {
            for (col_index, cell) in row.iter_mut().enumerate() {
                let k = (row_index * n + col_index) * 2;
                *cell = crate::Complex32::new(flat[k], flat[k + 1]);
            }
        }
        result
    }

    /// Exports the tableau into unpacked boolean matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// let cnot = stim::Tableau::from_named_gate("CNOT").unwrap();
    /// let (x2x, x2z, z2x, z2z, x_signs, z_signs) = cnot.to_ndarray().unwrap();
    /// assert_eq!(x2x, ndarray::array![[true, true], [false, true]]);
    /// assert_eq!(x2z, ndarray::array![[false, false], [false, false]]);
    /// assert_eq!(z2x, ndarray::array![[false, false], [false, false]]);
    /// assert_eq!(z2z, ndarray::array![[true, false], [true, true]]);
    /// assert_eq!(x_signs, ndarray::array![false, false]);
    /// assert_eq!(z_signs, ndarray::array![false, false]);
    /// ```
    pub fn to_ndarray(&self) -> crate::Result<TableauNumpyBool> {
        let n = self.len();
        let mut x2x = Array2::from_elem((n, n), false);
        let mut x2z = Array2::from_elem((n, n), false);
        let mut z2x = Array2::from_elem((n, n), false);
        let mut z2z = Array2::from_elem((n, n), false);
        let mut x_signs = Array1::from_elem(n, false);
        let mut z_signs = Array1::from_elem(n, false);

        for i in 0..n {
            x_signs[i] = self.x_sign(i)? < 0;
            z_signs[i] = self.z_sign(i)? < 0;
            for j in 0..n {
                let xp = self.x_output_pauli(i, j)?;
                x2x[[i, j]] = matches!(xp, 1 | 2);
                x2z[[i, j]] = matches!(xp, 2 | 3);
                let zp = self.z_output_pauli(i, j)?;
                z2x[[i, j]] = matches!(zp, 1 | 2);
                z2z[[i, j]] = matches!(zp, 2 | 3);
            }
        }
        Ok((x2x, x2z, z2x, z2z, x_signs, z_signs))
    }

    /// Exports the tableau into bit-packed matrices.
    pub fn to_ndarray_bit_packed(&self) -> crate::Result<TableauNumpyPacked> {
        let (x2x, x2z, z2x, z2z, x_signs, z_signs) = self.to_ndarray()?;
        let pack_matrix = |table: &Array2<bool>| -> Array2<u8> {
            let row_bytes = table.ncols().div_ceil(8);
            let packed_rows = table
                .rows()
                .into_iter()
                .flat_map(|row| {
                    crate::pack_bits(row.as_slice().expect("tableau rows are contiguous"))
                })
                .collect::<Vec<_>>();
            Array2::from_shape_vec((table.nrows(), row_bytes), packed_rows)
                .expect("packed tableau arrays should be rectangular")
        };
        Ok((
            pack_matrix(&x2x),
            pack_matrix(&x2z),
            pack_matrix(&z2x),
            pack_matrix(&z2z),
            Array1::from_vec(crate::pack_bits(
                x_signs
                    .as_slice()
                    .expect("tableau sign arrays are contiguous"),
            )),
            Array1::from_vec(crate::pack_bits(
                z_signs
                    .as_slice()
                    .expect("tableau sign arrays are contiguous"),
            )),
        ))
    }

    /// Converts the tableau into a stabilizer state vector.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// assert_eq!(
    ///     h.to_state_vector(stim::Endian::Little),
    ///     vec![
    ///         stim::Complex32::new(0.5_f32.sqrt(), 0.0),
    ///         stim::Complex32::new(0.5_f32.sqrt(), 0.0),
    ///     ]
    /// );
    /// ```
    pub fn to_state_vector(&self, endian: crate::Endian) -> Vec<crate::Complex32> {
        let flat = self
            .inner
            .to_state_vector_data(endian.as_str())
            .expect("typed endian should be accepted by stim-cxx");
        assert_eq!(
            flat.len() % 2,
            0,
            "unexpected flat state vector size from stim-cxx"
        );
        flat.chunks_exact(2)
            .map(|pair| crate::Complex32::new(pair[0], pair[1]))
            .collect()
    }
}

impl Display for Tableau {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.to_text())
    }
}

impl fmt::Debug for Tableau {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.to_repr_text())
    }
}

impl Add for Tableau {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner.add(&rhs.inner),
        }
    }
}

impl AddAssign for Tableau {
    fn add_assign(&mut self, rhs: Self) {
        self.inner.add_assign(&rhs.inner);
    }
}

impl Mul for Tableau {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        rhs.then(&self)
            .expect("tableau multiplication requires equal qubit counts")
    }
}

#[cfg(test)]
mod tests {
    use super::Tableau;
    use crate::{Complex32, PauliString};

    #[test]
    fn tableau_to_unitary_matrix_matches_documented_example() {
        let cnot = Tableau::from_conjugated_generators(
            &[
                PauliString::from_text("XX").unwrap(),
                PauliString::from_text("_X").unwrap(),
            ],
            &[
                PauliString::from_text("Z_").unwrap(),
                PauliString::from_text("ZZ").unwrap(),
            ],
        )
        .unwrap();

        assert_eq!(
            cnot.to_unitary_matrix(crate::Endian::Big),
            vec![
                vec![
                    Complex32::new(1.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(1.0, 0.0),
                ],
                vec![
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(1.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
            ]
        );
    }

    #[test]
    fn tableau_to_unitary_matrix_reports_documented_error_shapes() {
        assert!("middle".parse::<crate::Endian>().is_err());
    }

    #[test]
    fn tableau_to_state_vector_matches_documented_examples() {
        let i2 = Tableau::from_named_gate("I").unwrap();
        let x = Tableau::from_named_gate("X").unwrap();
        let h = Tableau::from_named_gate("H").unwrap();

        assert_eq!(
            (x.clone() + i2.clone()).to_state_vector(crate::Endian::Little),
            vec![
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
            ]
        );
        assert_eq!(
            (i2.clone() + x.clone()).to_state_vector(crate::Endian::Little),
            vec![
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
            ]
        );
        assert_eq!(
            (i2 + x).to_state_vector(crate::Endian::Big),
            vec![
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
            ]
        );
        assert_eq!(
            (h.clone() + h).to_state_vector(crate::Endian::Little),
            vec![
                Complex32::new(0.5, 0.0),
                Complex32::new(0.5, 0.0),
                Complex32::new(0.5, 0.0),
                Complex32::new(0.5, 0.0),
            ]
        );
    }

    #[test]
    fn tableau_to_state_vector_reports_documented_error_shapes() {
        assert!("middle".parse::<crate::Endian>().is_err());
    }

    #[test]
    fn tableau_from_state_vector_matches_documented_examples() {
        assert_eq!(
            Tableau::from_state_vector(
                &[
                    Complex32::new(0.5f32.sqrt(), 0.0),
                    Complex32::new(0.0, 0.5f32.sqrt()),
                ],
                crate::Endian::Little,
            )
            .unwrap(),
            Tableau::from_conjugated_generators(
                &[PauliString::from_text("+Z").unwrap()],
                &[PauliString::from_text("+Y").unwrap()],
            )
            .unwrap()
        );

        assert_eq!(
            Tableau::from_state_vector(
                &[
                    Complex32::new(0.5f32.sqrt(), 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.5f32.sqrt(), 0.0),
                ],
                crate::Endian::Little,
            )
            .unwrap(),
            Tableau::from_conjugated_generators(
                &[
                    PauliString::from_text("+Z_").unwrap(),
                    PauliString::from_text("+_X").unwrap(),
                ],
                &[
                    PauliString::from_text("+XX").unwrap(),
                    PauliString::from_text("+ZZ").unwrap(),
                ],
            )
            .unwrap()
        );
    }

    #[test]
    fn tableau_from_state_vector_reports_documented_error_shapes() {
        assert!("middle".parse::<crate::Endian>().is_err());

        let not_stabilizer = Tableau::from_state_vector(
            &[
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
            ],
            crate::Endian::Little,
        )
        .unwrap_err();
        assert!(
            not_stabilizer.message().contains("power of 2")
                || not_stabilizer.message().contains("stabilizer state")
        );
    }

    #[test]
    fn tableau_from_unitary_matrix_matches_documented_examples() {
        assert_eq!(
            Tableau::from_unitary_matrix(
                &[
                    vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
                    vec![Complex32::new(0.0, 0.0), Complex32::new(0.0, 1.0)],
                ],
                crate::Endian::Little,
            )
            .unwrap(),
            Tableau::from_conjugated_generators(
                &[PauliString::from_text("+Y").unwrap()],
                &[PauliString::from_text("+Z").unwrap()],
            )
            .unwrap()
        );

        assert_eq!(
            Tableau::from_unitary_matrix(
                &[
                    vec![
                        Complex32::new(1.0, 0.0),
                        Complex32::new(0.0, 0.0),
                        Complex32::new(0.0, 0.0),
                        Complex32::new(0.0, 0.0),
                    ],
                    vec![
                        Complex32::new(0.0, 0.0),
                        Complex32::new(1.0, 0.0),
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
                ],
                crate::Endian::Little,
            )
            .unwrap(),
            Tableau::from_conjugated_generators(
                &[
                    PauliString::from_text("+XZ").unwrap(),
                    PauliString::from_text("+YX").unwrap(),
                ],
                &[
                    PauliString::from_text("+ZZ").unwrap(),
                    PauliString::from_text("+_Z").unwrap(),
                ],
            )
            .unwrap()
        );
    }

    #[test]
    fn tableau_unitary_export_import_round_trips_for_small_examples() {
        for tableau in [
            Tableau::from_named_gate("H").unwrap(),
            Tableau::from_named_gate("CNOT").unwrap(),
        ] {
            let little = tableau.to_unitary_matrix(crate::Endian::Little);
            assert_eq!(
                Tableau::from_unitary_matrix(&little, crate::Endian::Little).unwrap(),
                tableau
            );

            let big = tableau.to_unitary_matrix(crate::Endian::Big);
            assert_eq!(
                Tableau::from_unitary_matrix(&big, crate::Endian::Big).unwrap(),
                tableau
            );
        }
    }

    #[test]
    fn tableau_from_unitary_matrix_reports_documented_error_shapes() {
        let not_square = Tableau::from_unitary_matrix(
            &[
                vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
                vec![Complex32::new(0.0, 0.0)],
            ],
            crate::Endian::Little,
        )
        .unwrap_err();
        assert!(not_square.message().contains("square"));

        assert!("middle".parse::<crate::Endian>().is_err());

        let not_clifford = Tableau::from_unitary_matrix(
            &[
                vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
                vec![Complex32::new(0.0, 0.0), Complex32::new(0.0, 0.0)],
            ],
            crate::Endian::Little,
        )
        .unwrap_err();
        assert!(not_clifford.message().contains("Clifford"));
    }

    #[test]
    fn tableau_numpy_style_bit_conversions_match_documented_examples() {
        let cnot = Tableau::from_named_gate("CNOT").unwrap();
        let (x2x, x2z, z2x, z2z, x_signs, z_signs) = cnot.to_ndarray().unwrap();
        assert_eq!(x2x, ndarray::array![[true, true], [false, true]]);
        assert_eq!(x2z, ndarray::array![[false, false], [false, false]]);
        assert_eq!(z2x, ndarray::array![[false, false], [false, false]]);
        assert_eq!(z2z, ndarray::array![[true, false], [true, true]]);
        assert_eq!(x_signs, ndarray::array![false, false]);
        assert_eq!(z_signs, ndarray::array![false, false]);

        let t = Tableau::from_conjugated_generators(
            &[
                PauliString::from_text("-Y_ZY").unwrap(),
                PauliString::from_text("-Y_YZ").unwrap(),
                PauliString::from_text("-XXX_").unwrap(),
                PauliString::from_text("+ZYX_").unwrap(),
            ],
            &[
                PauliString::from_text("-_ZZX").unwrap(),
                PauliString::from_text("+YZXZ").unwrap(),
                PauliString::from_text("+XZ_X").unwrap(),
                PauliString::from_text("-YYXX").unwrap(),
            ],
        )
        .unwrap();

        let (x2x, x2z, z2x, z2z, x_signs, z_signs) = t.to_ndarray().unwrap();
        assert_eq!(x_signs, ndarray::array![true, true, true, false]);
        assert_eq!(z_signs, ndarray::array![true, false, false, true]);
        assert_eq!(
            x2x,
            ndarray::array![
                [true, false, false, true],
                [true, false, true, false],
                [true, true, true, false],
                [false, true, true, false],
            ]
        );
        assert_eq!(
            x2z,
            ndarray::array![
                [true, false, true, true],
                [true, false, true, true],
                [false, false, false, false],
                [true, true, false, false],
            ]
        );
        assert_eq!(
            z2x,
            ndarray::array![
                [false, false, false, true],
                [true, false, true, false],
                [true, false, false, true],
                [true, true, true, true],
            ]
        );
        assert_eq!(
            z2z,
            ndarray::array![
                [false, true, true, false],
                [true, true, false, true],
                [false, true, false, false],
                [true, true, false, false],
            ]
        );

        let (x2x_p, x2z_p, z2x_p, z2z_p, x_signs_p, z_signs_p) = t.to_ndarray_bit_packed().unwrap();
        assert_eq!(x_signs_p, ndarray::array![7]);
        assert_eq!(z_signs_p, ndarray::array![9]);
        assert_eq!(x2x_p, ndarray::array![[9], [5], [7], [6]]);
        assert_eq!(x2z_p, ndarray::array![[13], [13], [0], [3]]);
        assert_eq!(z2x_p, ndarray::array![[8], [5], [9], [15]]);
        assert_eq!(z2z_p, ndarray::array![[6], [11], [2], [3]]);

        assert_eq!(
            Tableau::from_ndarray(
                x2x.view(),
                x2z.view(),
                z2x.view(),
                z2z.view(),
                x_signs.view(),
                z_signs.view(),
            )
            .unwrap(),
            t
        );
        assert_eq!(
            Tableau::from_ndarray_bit_packed(
                x2x_p.view(),
                x2z_p.view(),
                z2x_p.view(),
                z2z_p.view(),
                x_signs_p.view(),
                z_signs_p.view(),
                4,
            )
            .unwrap(),
            t
        );
    }

    #[test]
    fn tableau_pow_and_call_aliases_match_existing_behavior() {
        let tableau = Tableau::from_named_gate("H").unwrap();
        let pauli = PauliString::from_text("+X").unwrap();

        assert_eq!(tableau.pow(3), tableau.raised_to(3));
        assert_eq!(tableau.call(&pauli), tableau.conjugate(&pauli));
    }

    #[test]
    fn tableau_numpy_style_conversions_validate_shapes() {
        let err = Tableau::from_ndarray(
            ndarray::array![[true]].view(),
            ndarray::array![[false]].view(),
            ndarray::array![[false]].view(),
            ndarray::array![[true]].view(),
            ndarray::array![false].view(),
            ndarray::Array1::<bool>::from_vec(vec![]).view(),
        )
        .unwrap_err();
        assert!(
            err.message()
                .contains("Inconsistent x_signs/z_signs lengths")
        );

        let err = Tableau::from_ndarray(
            ndarray::array![[true, false]].view(),
            ndarray::array![[false, false]].view(),
            ndarray::array![[false, false]].view(),
            ndarray::array![[true, false]].view(),
            ndarray::array![false].view(),
            ndarray::array![false].view(),
        )
        .unwrap_err();
        assert!(err.message().contains("square"));

        let err = Tableau::from_ndarray_bit_packed(
            ndarray::array![[1]].view(),
            ndarray::array![[0]].view(),
            ndarray::array![[0]].view(),
            ndarray::array![[1]].view(),
            ndarray::Array1::<u8>::from_vec(vec![]).view(),
            ndarray::array![0].view(),
            1,
        )
        .unwrap_err();
        assert!(err.message().contains("sign vectors"));
    }
}

#[cfg(test)]
mod circuit_interop_tests {
    use std::str::FromStr;

    use crate::{Circuit, PauliString, Tableau};

    fn documented_to_circuit_tableau() -> Tableau {
        Tableau::from_conjugated_generators(
            &[
                PauliString::from_text("+YZ__").unwrap(),
                PauliString::from_text("-Y_XY").unwrap(),
                PauliString::from_text("+___Y").unwrap(),
                PauliString::from_text("+YZX_").unwrap(),
            ],
            &[
                PauliString::from_text("+XZYY").unwrap(),
                PauliString::from_text("-XYX_").unwrap(),
                PauliString::from_text("-ZXXZ").unwrap(),
                PauliString::from_text("+XXZ_").unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn tableau_identity_and_named_gate_constructors_match_documented_examples() {
        let identity = Tableau::new(3);
        let h = Tableau::from_named_gate("H").unwrap();
        let cnot = Tableau::from_named_gate("CNOT").unwrap();
        let s = Tableau::from_named_gate("S").unwrap();

        assert_eq!(identity.num_qubits(), 3);
        assert_eq!(
            identity.to_string(),
            "+-xz-xz-xz-\n| ++ ++ ++\n| XZ __ __\n| __ XZ __\n| __ __ XZ"
        );
        assert_eq!(h.to_string(), "+-xz-\n| ++\n| ZX");
        assert_eq!(cnot.to_string(), "+-xz-xz-\n| ++ ++\n| XZ _Z\n| X_ XZ");
        assert_eq!(s.to_string(), "+-xz-\n| ++\n| YZ");
    }

    #[test]
    fn tableau_random_matches_documented_basic_contract() {
        let t = Tableau::random(10);
        assert_eq!(t.len(), 10);
        assert_ne!(t, Tableau::random(10));
    }

    #[test]
    fn tableau_iter_all_matches_documented_small_counts() {
        let mut zero_unsigned = Tableau::iter_all(0, true);
        assert_eq!(zero_unsigned.next(), Some(Tableau::new(0)));
        assert_eq!(zero_unsigned.next(), None);

        let mut zero_signed = Tableau::iter_all(0, false);
        assert_eq!(zero_signed.next(), Some(Tableau::new(0)));
        assert_eq!(zero_signed.next(), None);

        let one_unsigned = Tableau::iter_all(1, true);
        assert_eq!(one_unsigned.clone().count(), 6);
        assert_eq!(one_unsigned.count(), 6);
        assert_eq!(Tableau::iter_all(1, false).count(), 24);

        let two_unsigned = Tableau::iter_all(2, true);
        assert_eq!(two_unsigned.clone().count(), 720);
        assert_eq!(Tableau::iter_all(2, false).count(), 11520);
    }

    #[test]
    fn tableau_copy_matches_documented_behavior() {
        let t1 = Tableau::new(3);
        let t2 = t1.clone();

        assert_eq!(t1, t2);
        assert_ne!((&t1 as *const Tableau), (&t2 as *const Tableau));
    }

    #[test]
    fn tableau_from_stabilizers_matches_documented_examples_and_errors() {
        assert_eq!(
            Tableau::from_stabilizers(
                &[
                    PauliString::from_text("XX").unwrap(),
                    PauliString::from_text("ZZ").unwrap(),
                ],
                false,
                false,
            )
            .unwrap(),
            Tableau::from_conjugated_generators(
                &[
                    PauliString::from_text("+Z_").unwrap(),
                    PauliString::from_text("+_X").unwrap(),
                ],
                &[
                    PauliString::from_text("+XX").unwrap(),
                    PauliString::from_text("+ZZ").unwrap(),
                ],
            )
            .unwrap()
        );

        assert!(
            Tableau::from_stabilizers(
                &[
                    PauliString::from_text("XX_").unwrap(),
                    PauliString::from_text("ZZ_").unwrap(),
                    PauliString::from_text("-YY_").unwrap(),
                    PauliString::from_text("").unwrap(),
                ],
                true,
                true,
            )
            .is_ok()
        );

        assert!(
            Tableau::from_stabilizers(
                &[
                    PauliString::from_text("Z").unwrap(),
                    PauliString::from_text("X").unwrap(),
                ],
                false,
                false,
            )
            .unwrap_err()
            .to_string()
            .contains("anticommute")
        );

        assert!(
            Tableau::from_stabilizers(
                &[
                    PauliString::from_text("Z_").unwrap(),
                    PauliString::from_text("-_Z").unwrap(),
                    PauliString::from_text("ZZ").unwrap(),
                ],
                false,
                false,
            )
            .unwrap_err()
            .to_string()
            .contains("contradict")
        );
    }

    #[test]
    fn tableau_from_conjugated_generators_matches_documented_examples_and_validation() {
        assert_eq!(
            Tableau::new(3),
            Tableau::from_conjugated_generators(
                &[
                    PauliString::from_text("X__").unwrap(),
                    PauliString::from_text("_X_").unwrap(),
                    PauliString::from_text("__X").unwrap(),
                ],
                &[
                    PauliString::from_text("Z__").unwrap(),
                    PauliString::from_text("_Z_").unwrap(),
                    PauliString::from_text("__Z").unwrap(),
                ],
            )
            .unwrap()
        );
        assert_eq!(
            Tableau::from_named_gate("S").unwrap(),
            Tableau::from_conjugated_generators(
                &[PauliString::from_text("Y").unwrap()],
                &[PauliString::from_text("Z").unwrap()],
            )
            .unwrap()
        );
        assert_eq!(
            Tableau::from_named_gate("S_DAG").unwrap(),
            Tableau::from_conjugated_generators(
                &[PauliString::from_text("-Y").unwrap()],
                &[PauliString::from_text("Z").unwrap()],
            )
            .unwrap()
        );

        let err = Tableau::from_conjugated_generators(
            &[PauliString::from_text("X_").unwrap()],
            &[
                PauliString::from_text("Z_").unwrap(),
                PauliString::from_text("_Z").unwrap(),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("len(xs) != len(zs)"));

        let err = Tableau::from_conjugated_generators(
            &[
                PauliString::from_text("X_").unwrap(),
                PauliString::from_text("_X_").unwrap(),
            ],
            &[
                PauliString::from_text("Z_").unwrap(),
                PauliString::from_text("_Z").unwrap(),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("len(p) == len(xs)"));

        let err = Tableau::from_conjugated_generators(
            &[
                PauliString::from_text("X_").unwrap(),
                PauliString::from_text("_X").unwrap(),
            ],
            &[
                PauliString::from_text("Z_").unwrap(),
                PauliString::from_text("_Z_").unwrap(),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("len(p) == len(zs)"));

        let err = Tableau::from_conjugated_generators(
            &[
                PauliString::from_text("X_").unwrap(),
                PauliString::from_text("_Z").unwrap(),
            ],
            &[
                PauliString::from_text("Z_").unwrap(),
                PauliString::from_text("_Z").unwrap(),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("commutativity"));
    }

    #[test]
    fn tableau_direct_sum_matches_documented_examples() {
        let s = Tableau::from_named_gate("S").unwrap();
        let cz = Tableau::from_named_gate("CZ").unwrap();

        assert_eq!(
            (s.clone() + cz.clone()).to_string(),
            "+-xz-xz-xz-\n| ++ ++ ++\n| YZ __ __\n| __ XZ Z_\n| __ Z_ XZ"
        );

        let mut combined = Tableau::from_named_gate("S").unwrap();
        combined += cz.clone();
        assert_eq!(
            combined.to_string(),
            "+-xz-xz-xz-\n| ++ ++ ++\n| YZ __ __\n| __ XZ Z_\n| __ Z_ XZ"
        );

        assert_eq!(Tableau::new(0) + Tableau::new(0), Tableau::new(0));
        assert_eq!(Tableau::new(1) + Tableau::new(2), Tableau::new(3));
        assert_eq!(
            Tableau::new(0) + Tableau::from_named_gate("CNOT").unwrap() + Tableau::new(0),
            Tableau::from_named_gate("CNOT").unwrap()
        );
    }

    #[test]
    fn tableau_mul_matches_documented_composition_semantics() {
        assert_eq!(Tableau::new(0) * Tableau::new(0), Tableau::new(0));
        assert_eq!(Tableau::new(1) * Tableau::new(1), Tableau::new(1));

        let h = Tableau::from_named_gate("H").unwrap();
        let s = Tableau::from_named_gate("S").unwrap();
        let product = s.clone() * h.clone();
        let via_then = h.then(&s).unwrap();

        assert_eq!(product, via_then);
        let p = PauliString::from_text("X").unwrap();
        assert_eq!(product.conjugate(&p), s.conjugate(&h.conjugate(&p)));
    }

    #[test]
    fn tableau_conjugate_matches_documented_examples() {
        let t = Tableau::from_named_gate("CNOT").unwrap();

        assert_eq!(
            t.conjugate(&PauliString::from_text("__").unwrap()),
            PauliString::from_text("__").unwrap()
        );
        assert_eq!(
            t.conjugate(&PauliString::from_text("-__").unwrap()),
            PauliString::from_text("-__").unwrap()
        );
        assert_eq!(
            t.conjugate(&PauliString::from_text("X_").unwrap()),
            PauliString::from_text("XX").unwrap()
        );
        assert_eq!(
            t.conjugate(&PauliString::from_text("Y_").unwrap()),
            PauliString::from_text("YX").unwrap()
        );
        assert_eq!(
            t.conjugate(&PauliString::from_text("Z_").unwrap()),
            PauliString::from_text("Z_").unwrap()
        );
        assert_eq!(
            t.conjugate(&PauliString::from_text("_X").unwrap()),
            PauliString::from_text("_X").unwrap()
        );
        assert_eq!(
            t.conjugate(&PauliString::from_text("_Y").unwrap()),
            PauliString::from_text("ZY").unwrap()
        );
        assert_eq!(
            t.conjugate(&PauliString::from_text("_Z").unwrap()),
            PauliString::from_text("ZZ").unwrap()
        );
        assert_eq!(
            t.conjugate(&PauliString::from_text("YY").unwrap()),
            PauliString::from_text("-XZ").unwrap()
        );
        assert_eq!(
            t.conjugate(&PauliString::from_text("-YY").unwrap()),
            PauliString::from_text("XZ").unwrap()
        );
    }

    #[test]
    fn tableau_to_stabilizers_matches_documented_examples() {
        let t = Tableau::from_named_gate("CNOT").unwrap();

        assert_eq!(
            t.to_stabilizers(false).unwrap(),
            vec![
                PauliString::from_text("+Z_").unwrap(),
                PauliString::from_text("+ZZ").unwrap(),
            ]
        );
        assert_eq!(
            t.to_stabilizers(true).unwrap(),
            vec![
                PauliString::from_text("+Z_").unwrap(),
                PauliString::from_text("+_Z").unwrap(),
            ]
        );
    }

    #[test]
    fn tableau_to_circuit_matches_documented_examples_and_validation() {
        let tableau = documented_to_circuit_tableau();

        assert_eq!(
            tableau.to_circuit().unwrap(),
            tableau
                .to_circuit_with_method(crate::TableauSynthesisMethod::Elimination)
                .unwrap()
        );
        assert_eq!(
            tableau.to_circuit().unwrap().to_string(),
            "S 0\nH 0 1 3\nCX 0 1 0 2 0 3\nS 1 3\nH 1 3\nCX 1 0 3 0 3 1 1 3 3 1\nH 1\nS 1\nCX 1 3\nH 2 3\nCX 2 1 3 1 3 2 2 3 3 2\nH 3\nCX 2 3\nS 3\nH 3 0 1 2\nS 0 0 1 1 2 2\nH 0 1 2\nS 3 3"
        );
        assert_eq!(
            tableau
                .to_circuit_with_method(crate::TableauSynthesisMethod::GraphState)
                .unwrap()
                .to_string(),
            "RX 0 1 2 3\nTICK\nCZ 0 3 1 2 1 3\nTICK\nX 0 1\nZ 2\nS 2 3\nH 3\nS 3"
        );
        assert_eq!(
            tableau
                .to_circuit_with_method(crate::TableauSynthesisMethod::MppStateUnsigned)
                .unwrap()
                .to_string(),
            "MPP X0*Z1*Y2*Y3 !X0*Y1*X2 !Z0*X1*X2*Z3 X0*X1*Z2"
        );
        assert_eq!(
            tableau
                .to_circuit_with_method(crate::TableauSynthesisMethod::MppState)
                .unwrap()
                .to_string(),
            "MPP X0*Z1*Y2*Y3 !X0*Y1*X2 !Z0*X1*X2*Z3 X0*X1*Z2\nCX rec[-3] 2 rec[-1] 2\nCY rec[-4] 0 rec[-3] 0 rec[-3] 3 rec[-2] 3 rec[-1] 0\nCZ rec[-4] 1 rec[-1] 1"
        );

        assert!("nope".parse::<crate::TableauSynthesisMethod>().is_err());
    }

    #[test]
    fn to_tableau_matches_documented_example() {
        let tableau = Circuit::from_str("H 0\nCNOT 0 1")
            .unwrap()
            .to_tableau(false, false, false)
            .unwrap();

        assert_eq!(tableau.num_qubits(), 2);
        assert_eq!(
            format!("{tableau:?}"),
            "stim.Tableau.from_conjugated_generators(\n    xs=[\n        stim.PauliString(\"+Z_\"),\n        stim.PauliString(\"+_X\"),\n    ],\n    zs=[\n        stim.PauliString(\"+XX\"),\n        stim.PauliString(\"+ZZ\"),\n    ],\n)"
        );
    }

    #[test]
    fn from_circuit_delegates_to_the_same_conversion_path() {
        let circuit = Circuit::from_str("H 0\nCNOT 0 1").unwrap();
        let via_tableau = circuit.to_tableau(false, false, false).unwrap();
        let via_circuit = circuit.to_tableau(false, false, false).unwrap();

        assert_eq!(via_tableau, via_circuit);
        assert_eq!(via_tableau.len(), 2);
        assert!(!via_tableau.is_empty());
        assert_eq!(Tableau::new(0).len(), 0);
        assert!(Tableau::new(0).is_empty());
    }

    #[test]
    fn tableau_inverse_and_exponentiation_match_documented_examples() {
        let s = Tableau::from_named_gate("S").unwrap();
        let s_dag = Tableau::from_named_gate("S_DAG").unwrap();
        let z = Tableau::from_named_gate("Z").unwrap();

        assert_eq!(s.inverse(false), s_dag);
        assert_eq!(z.inverse(false), z);
        assert_eq!(s.raised_to(0), Tableau::new(1));
        assert_eq!(s.raised_to(1), s);
        assert_eq!(Tableau::from_named_gate("S").unwrap().raised_to(2), z);
        assert_eq!(Tableau::from_named_gate("S").unwrap().raised_to(-1), s_dag);
        assert_eq!(Tableau::from_named_gate("S").unwrap().raised_to(3), s_dag);
        assert_eq!(
            Tableau::from_named_gate("S").unwrap().raised_to(5),
            Tableau::from_named_gate("S").unwrap()
        );
        assert_eq!(
            Tableau::from_named_gate("S").unwrap().raised_to(400000001),
            Tableau::from_named_gate("S").unwrap()
        );
        assert_eq!(
            Tableau::from_named_gate("S").unwrap().raised_to(-399999999),
            Tableau::from_named_gate("S").unwrap()
        );
    }

    #[test]
    fn tableau_then_matches_documented_composition_semantics() {
        let h = Tableau::from_named_gate("H").unwrap();
        let s = Tableau::from_named_gate("S").unwrap();
        let hs = h.then(&s).unwrap();

        assert_eq!(
            Tableau::new(0).then(&Tableau::new(0)).unwrap(),
            Tableau::new(0)
        );
        assert_eq!(
            Tableau::new(1).then(&Tableau::new(1)).unwrap(),
            Tableau::new(1)
        );
        assert_eq!(hs.x_output(0), PauliString::from_text("+Z").unwrap());
        assert_eq!(hs.z_output(0), PauliString::from_text("+Y").unwrap());

        let err = Tableau::new(3).then(&Tableau::new(4)).unwrap_err();
        assert!(err.to_string().contains("len(self) != len(second)"));
    }

    #[test]
    fn tableau_append_matches_documented_examples_and_validation() {
        let cy = Tableau::from_named_gate("CY").unwrap();
        let sqrt_x = Tableau::from_named_gate("SQRT_X").unwrap();
        let sqrt_x_dag = Tableau::from_named_gate("SQRT_X_DAG").unwrap();
        let cz = Tableau::from_named_gate("CZ").unwrap();

        let mut t = Tableau::new(2);
        assert!(
            t.append(&cy, &[0])
                .unwrap_err()
                .to_string()
                .contains("len(targets) != len(gate)")
        );
        assert!(
            t.append(&cy, &[0, 0])
                .unwrap_err()
                .to_string()
                .contains("collision")
        );
        assert!(
            t.append(&cy, &[1, 2])
                .unwrap_err()
                .to_string()
                .contains("target >= len(tableau)")
        );

        t.append(&sqrt_x_dag, &[1]).unwrap();
        t.append(&cy, &[0, 1]).unwrap();
        t.append(&sqrt_x, &[1]).unwrap();
        assert_eq!(t, cz);

        let mut t = Tableau::new(2);
        t.append(&sqrt_x, &[1]).unwrap();
        t.append(&cy, &[0, 1]).unwrap();
        t.append(&sqrt_x_dag, &[1]).unwrap();
        assert_ne!(t, cz);
    }

    #[test]
    fn tableau_prepend_matches_documented_examples_and_validation() {
        let cy = Tableau::from_named_gate("CY").unwrap();
        let sqrt_x = Tableau::from_named_gate("SQRT_X").unwrap();
        let sqrt_x_dag = Tableau::from_named_gate("SQRT_X_DAG").unwrap();
        let cz = Tableau::from_named_gate("CZ").unwrap();

        let mut t = Tableau::new(2);
        assert!(
            t.prepend(&cy, &[0])
                .unwrap_err()
                .to_string()
                .contains("len(targets) != len(gate)")
        );
        assert!(
            t.prepend(&cy, &[0, 0])
                .unwrap_err()
                .to_string()
                .contains("collision")
        );
        assert!(
            t.prepend(&cy, &[1, 2])
                .unwrap_err()
                .to_string()
                .contains("target >= len(tableau)")
        );

        t.prepend(&sqrt_x_dag, &[1]).unwrap();
        t.prepend(&cy, &[0, 1]).unwrap();
        t.prepend(&sqrt_x, &[1]).unwrap();
        assert_ne!(t, cz);

        let mut t = Tableau::new(2);
        t.prepend(&sqrt_x, &[1]).unwrap();
        t.prepend(&cy, &[0, 1]).unwrap();
        t.prepend(&sqrt_x_dag, &[1]).unwrap();
        assert_eq!(t, cz);
    }

    #[test]
    fn tableau_signs_and_constant_time_pauli_queries_match_documented_examples() {
        let s = Tableau::from_named_gate("S").unwrap();
        let s_dag = Tableau::from_named_gate("S_DAG").unwrap();
        let sqrt_x = Tableau::from_named_gate("SQRT_X").unwrap();
        let sqrt_x_dag = Tableau::from_named_gate("SQRT_X_DAG").unwrap();
        let cnot = Tableau::from_named_gate("CNOT").unwrap();

        assert_eq!(s_dag.x_sign(0).unwrap(), -1);
        assert_eq!(s.x_sign(0).unwrap(), 1);
        assert_eq!(s_dag.y_sign(0).unwrap(), 1);
        assert_eq!(s.y_sign(0).unwrap(), -1);
        assert_eq!(sqrt_x_dag.z_sign(0).unwrap(), 1);
        assert_eq!(sqrt_x.z_sign(0).unwrap(), -1);

        assert_eq!(cnot.x_output_pauli(0, 0).unwrap(), 1);
        assert_eq!(cnot.x_output_pauli(0, 1).unwrap(), 1);
        assert_eq!(cnot.x_output_pauli(1, 0).unwrap(), 0);
        assert_eq!(cnot.x_output_pauli(1, 1).unwrap(), 1);
        assert_eq!(cnot.y_output_pauli(0, 0).unwrap(), 2);
        assert_eq!(cnot.y_output_pauli(0, 1).unwrap(), 1);
        assert_eq!(cnot.z_output_pauli(0, 0).unwrap(), 3);
        assert_eq!(cnot.z_output_pauli(1, 1).unwrap(), 3);

        assert_eq!(
            Tableau::from_named_gate("H").unwrap().x_output(0),
            PauliString::from_text("+Z").unwrap()
        );
        assert_eq!(
            Tableau::from_named_gate("H").unwrap().z_output(0),
            PauliString::from_text("+X").unwrap()
        );
        assert_eq!(cnot.x_output(0), PauliString::from_text("+XX").unwrap());
        assert_eq!(cnot.x_output(1), PauliString::from_text("+_X").unwrap());
        assert_eq!(cnot.y_output(0), PauliString::from_text("+YX").unwrap());
        assert_eq!(cnot.y_output(1), PauliString::from_text("+ZY").unwrap());
        assert_eq!(cnot.z_output(0), PauliString::from_text("+Z_").unwrap());
        assert_eq!(cnot.z_output(1), PauliString::from_text("+ZZ").unwrap());
    }

    #[test]
    fn tableau_inverse_constant_time_pauli_queries_match_documented_examples() {
        let t = Tableau::from_named_gate("CNOT").unwrap();
        let t_inv = t.inverse(false);

        assert_eq!(
            t.inverse_x_output_pauli(0, 0).unwrap(),
            t_inv.x_output_pauli(0, 0).unwrap()
        );
        assert_eq!(
            t.inverse_x_output_pauli(0, 1).unwrap(),
            t_inv.x_output_pauli(0, 1).unwrap()
        );
        assert_eq!(
            t.inverse_x_output_pauli(1, 0).unwrap(),
            t_inv.x_output_pauli(1, 0).unwrap()
        );
        assert_eq!(
            t.inverse_x_output_pauli(1, 1).unwrap(),
            t_inv.x_output_pauli(1, 1).unwrap()
        );
        assert_eq!(
            t.inverse_y_output_pauli(0, 0).unwrap(),
            t_inv.y_output_pauli(0, 0).unwrap()
        );
        assert_eq!(
            t.inverse_y_output_pauli(0, 1).unwrap(),
            t_inv.y_output_pauli(0, 1).unwrap()
        );
        assert_eq!(
            t.inverse_y_output_pauli(1, 0).unwrap(),
            t_inv.y_output_pauli(1, 0).unwrap()
        );
        assert_eq!(
            t.inverse_y_output_pauli(1, 1).unwrap(),
            t_inv.y_output_pauli(1, 1).unwrap()
        );
        assert_eq!(
            t.inverse_z_output_pauli(0, 0).unwrap(),
            t_inv.z_output_pauli(0, 0).unwrap()
        );
        assert_eq!(
            t.inverse_z_output_pauli(0, 1).unwrap(),
            t_inv.z_output_pauli(0, 1).unwrap()
        );
        assert_eq!(
            t.inverse_z_output_pauli(1, 0).unwrap(),
            t_inv.z_output_pauli(1, 0).unwrap()
        );
        assert_eq!(
            t.inverse_z_output_pauli(1, 1).unwrap(),
            t_inv.z_output_pauli(1, 1).unwrap()
        );
    }

    #[test]
    fn tableau_inverse_outputs_and_copy_match_documented_behavior() {
        let t = Tableau::from_named_gate("CNOT").unwrap();
        let t_inv = t.inverse(false);
        let copied = t.clone();
        let px = t.inverse_x_output(0, false);
        let py = t.inverse_y_output(0, false);
        let pz = t.inverse_z_output(0, false);

        assert_eq!(copied, t);
        assert_eq!(copied.to_string(), t.to_string());
        assert_eq!(px, t_inv.x_output(0));
        assert_eq!(py, t_inv.y_output(0));
        assert_eq!(pz, t_inv.z_output(0));
        assert_eq!(px.clone(), px);
    }

    #[test]
    fn pauli_string_sign_and_to_tableau_match_documented_examples() {
        assert_eq!(PauliString::from_text("X").unwrap().sign(), 1);
        assert_eq!(PauliString::from_text("-X").unwrap().sign(), -1);

        let zz = PauliString::from_text("ZZ").unwrap();
        let tableau = zz.to_tableau();
        assert_eq!(
            format!("{tableau:?}"),
            "stim.Tableau.from_conjugated_generators(\n    xs=[\n        stim.PauliString(\"-X_\"),\n        stim.PauliString(\"-_X\"),\n    ],\n    zs=[\n        stim.PauliString(\"+Z_\"),\n        stim.PauliString(\"+_Z\"),\n    ],\n)"
        );

        let p = PauliString::from_text("+YX_Z").unwrap();
        assert_eq!(p.to_tableau().to_pauli_string().unwrap(), p);

        let err = Tableau::from_named_gate("CNOT")
            .unwrap()
            .to_pauli_string()
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("isn't equivalent to a Pauli product")
        );
    }

    #[test]
    fn to_tableau_enforces_ignore_flags() {
        let noise = Circuit::from_str("X_ERROR(0.1) 0").unwrap();
        let measure = Circuit::from_str("M 0").unwrap();
        let reset = Circuit::from_str("R 0").unwrap();

        assert!(noise.to_tableau(false, false, false).is_err());
        assert!(measure.to_tableau(false, false, false).is_err());
        assert!(reset.to_tableau(false, false, false).is_err());

        assert!(noise.to_tableau(true, false, false).is_ok());
        assert!(measure.to_tableau(false, true, false).is_ok());
        assert!(reset.to_tableau(false, false, true).is_ok());
    }
}
