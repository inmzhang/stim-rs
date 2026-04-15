use std::fmt::{self, Display, Formatter};
use std::ops::{Add, AddAssign, Mul};
use std::pin::Pin;

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

/// A Clifford tableau describing how Paulis are conjugated by an operation.
#[derive(Clone, PartialEq, Eq)]
pub struct Tableau {
    pub(crate) inner: stim_cxx::Tableau,
}

impl Tableau {
    /// Creates the identity tableau over `num_qubits` qubits.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            inner: stim_cxx::Tableau::new(num_qubits),
        }
    }

    /// Returns a random tableau over `num_qubits` qubits.
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

    /// Iterates over all tableaux on `num_qubits` qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// let all: Vec<_> = stim::Tableau::iter_all(1, true).collect();
    /// assert_eq!(all.len(), 6);
    /// ```
    #[must_use]
    pub fn iter_all(num_qubits: usize, unsigned: bool) -> crate::TableauIterator {
        crate::TableauIterator {
            inner: stim_cxx::Tableau::iter_all(num_qubits, unsigned),
        }
    }

    /// Builds a tableau from a circuit.
    pub fn from_circuit(
        circuit: &crate::Circuit,
        ignore_noise: bool,
        ignore_measurement: bool,
        ignore_reset: bool,
    ) -> crate::Result<Self> {
        circuit.to_tableau(ignore_noise, ignore_measurement, ignore_reset)
    }

    /// Returns the tableau of a named Clifford gate.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// assert_eq!(h.x_output(0), stim::PauliString::from_text("+Z").unwrap());
    /// assert_eq!(h.z_output(0), stim::PauliString::from_text("+X").unwrap());
    /// ```
    pub fn from_named_gate(name: &str) -> crate::Result<Self> {
        stim_cxx::Tableau::from_named_gate(name)
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Infers the tableau of a stabilizer state vector.
    ///
    /// # Examples
    ///
    /// ```
    /// let t = stim::Tableau::from_state_vector(
    ///     &[
    ///         stim::Complex32::new(0.5_f32.sqrt(), 0.0),
    ///         stim::Complex32::new(0.0, 0.5_f32.sqrt()),
    ///     ],
    ///     "little",
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
        endian: &str,
    ) -> crate::Result<Self> {
        let mut flat = Vec::with_capacity(state_vector.len() * 2);
        for amp in state_vector {
            flat.push(amp.re);
            flat.push(amp.im);
        }
        stim_cxx::Tableau::from_state_vector_data(flat, endian)
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Infers a tableau from a unitary matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// let t = stim::Tableau::from_unitary_matrix(
    ///     &[
    ///         vec![stim::Complex32::new(1.0, 0.0), stim::Complex32::new(0.0, 0.0)],
    ///         vec![stim::Complex32::new(0.0, 0.0), stim::Complex32::new(0.0, 1.0)],
    ///     ],
    ///     "little",
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
        endian: &str,
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
        stim_cxx::Tableau::from_unitary_matrix_data(flat, endian)
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Builds a tableau from unpacked boolean matrices.
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

    /// Builds a tableau from bit-packed boolean matrices.
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

    /// Builds a tableau from conjugated X/Z generators.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_conjugated_generators(
    ///     &[stim::PauliString::from_text("+Z").unwrap()],
    ///     &[stim::PauliString::from_text("+X").unwrap()],
    /// )
    /// .unwrap();
    /// assert_eq!(h, stim::Tableau::from_named_gate("H").unwrap());
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

    /// Builds a tableau from stabilizer generators.
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

    /// Composes two tableaux as `self` followed by `second`.
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

    /// Returns the number of qubits acted on by the tableau.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Appends a gate tableau onto selected targets.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut t = stim::Tableau::new(2);
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// t.append(&h, &[1]).unwrap();
    /// assert_eq!(t.x_output(1), stim::PauliString::from_text("+_Z").unwrap());
    /// ```
    pub fn append(&mut self, gate: &Self, targets: &[usize]) -> crate::Result<()> {
        Pin::new(&mut self.inner)
            .append(&gate.inner, targets)
            .map_err(crate::StimError::from)
    }

    /// Prepends a gate tableau onto selected targets.
    pub fn prepend(&mut self, gate: &Self, targets: &[usize]) -> crate::Result<()> {
        Pin::new(&mut self.inner)
            .prepend(&gate.inner, targets)
            .map_err(crate::StimError::from)
    }

    /// Returns the number of qubits acted on by the tableau.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_qubits()
    }

    /// Returns whether the tableau acts on zero qubits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the inverse tableau.
    #[must_use]
    pub fn inverse(&self, unsigned: bool) -> Self {
        Self {
            inner: self.inner.inverse(unsigned),
        }
    }

    /// Raises the tableau to a power.
    #[must_use]
    pub fn raised_to(&self, exponent: i64) -> Self {
        Self {
            inner: self.inner.raised_to(exponent),
        }
    }

    /// Alias of [`Self::raised_to`].
    #[must_use]
    pub fn pow(&self, exponent: i64) -> Self {
        self.raised_to(exponent)
    }

    /// Returns the X sign of the target output.
    pub fn x_sign(&self, target: usize) -> crate::Result<i32> {
        self.inner.x_sign(target).map_err(crate::StimError::from)
    }

    /// Returns the Y sign of the target output.
    pub fn y_sign(&self, target: usize) -> crate::Result<i32> {
        self.inner.y_sign(target).map_err(crate::StimError::from)
    }

    /// Returns the Z sign of the target output.
    pub fn z_sign(&self, target: usize) -> crate::Result<i32> {
        self.inner.z_sign(target).map_err(crate::StimError::from)
    }

    /// Returns the X output Pauli code at a specific output coordinate.
    pub fn x_output_pauli(&self, input_index: usize, output_index: usize) -> crate::Result<u8> {
        self.inner
            .x_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns the Y output Pauli code at a specific output coordinate.
    pub fn y_output_pauli(&self, input_index: usize, output_index: usize) -> crate::Result<u8> {
        self.inner
            .y_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns the Z output Pauli code at a specific output coordinate.
    pub fn z_output_pauli(&self, input_index: usize, output_index: usize) -> crate::Result<u8> {
        self.inner
            .z_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns the inverse X output Pauli code at a specific output coordinate.
    pub fn inverse_x_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> crate::Result<u8> {
        self.inner
            .inverse_x_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns the inverse Y output Pauli code at a specific output coordinate.
    pub fn inverse_y_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> crate::Result<u8> {
        self.inner
            .inverse_y_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns the inverse Z output Pauli code at a specific output coordinate.
    pub fn inverse_z_output_pauli(
        &self,
        input_index: usize,
        output_index: usize,
    ) -> crate::Result<u8> {
        self.inner
            .inverse_z_output_pauli(input_index, output_index)
            .map_err(crate::StimError::from)
    }

    /// Returns the full Pauli string output of an input X generator.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// assert_eq!(h.x_output(0), stim::PauliString::from_text("+Z").unwrap());
    /// ```
    #[must_use]
    pub fn x_output(&self, target: usize) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.x_output(target),
            imag: false,
        }
    }

    /// Returns the full Pauli string output of an input Y generator.
    #[must_use]
    pub fn y_output(&self, target: usize) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.y_output(target),
            imag: false,
        }
    }

    /// Returns the full Pauli string output of an input Z generator.
    #[must_use]
    pub fn z_output(&self, target: usize) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.z_output(target),
            imag: false,
        }
    }

    /// Returns the full inverse X output of the tableau.
    #[must_use]
    pub fn inverse_x_output(&self, target: usize, unsigned: bool) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.inverse_x_output(target, unsigned),
            imag: false,
        }
    }

    /// Returns the full inverse Y output of the tableau.
    #[must_use]
    pub fn inverse_y_output(&self, target: usize, unsigned: bool) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.inverse_y_output(target, unsigned),
            imag: false,
        }
    }

    /// Returns the full inverse Z output of the tableau.
    #[must_use]
    pub fn inverse_z_output(&self, target: usize, unsigned: bool) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.inverse_z_output(target, unsigned),
            imag: false,
        }
    }

    /// Conjugates a Pauli string by the tableau.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// let x = stim::PauliString::from_text("+X").unwrap();
    /// assert_eq!(h.conjugate(&x), stim::PauliString::from_text("+Z").unwrap());
    /// ```
    #[must_use]
    pub fn conjugate(&self, pauli_string: &crate::PauliString) -> crate::PauliString {
        crate::PauliString {
            inner: self.inner.conjugate_pauli_string(&pauli_string.inner),
            imag: pauli_string.imag,
        }
    }

    /// Alias of [`Self::conjugate`].
    #[must_use]
    pub fn call(&self, pauli_string: &crate::PauliString) -> crate::PauliString {
        self.conjugate(pauli_string)
    }

    /// Converts the tableau into stabilizer generators.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = stim::Tableau::from_named_gate("H").unwrap();
    /// assert_eq!(
    ///     h.to_stabilizers(true).unwrap(),
    ///     vec![stim::PauliString::from_text("+X").unwrap()]
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
        self.to_circuit_with_method("elimination")
    }

    /// Converts the tableau into a circuit with a specific decomposition method.
    pub fn to_circuit_with_method(&self, method: &str) -> crate::Result<crate::Circuit> {
        self.inner
            .to_circuit(method)
            .map(|inner| crate::Circuit { inner })
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
    pub fn to_unitary_matrix(&self, endian: &str) -> crate::Result<Vec<Vec<crate::Complex32>>> {
        let flat = self
            .inner
            .to_unitary_matrix_data(endian)
            .map_err(crate::StimError::from)?;
        let n = 1usize << self.num_qubits();
        if flat.len() != n * n * 2 {
            return Err(crate::StimError::new(
                "unexpected flat unitary matrix size from stim-cxx",
            ));
        }
        let mut result = vec![vec![crate::Complex32::new(0.0, 0.0); n]; n];
        for (row_index, row) in result.iter_mut().enumerate() {
            for (col_index, cell) in row.iter_mut().enumerate() {
                let k = (row_index * n + col_index) * 2;
                *cell = crate::Complex32::new(flat[k], flat[k + 1]);
            }
        }
        Ok(result)
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
    ///     h.to_state_vector("little").unwrap(),
    ///     vec![
    ///         stim::Complex32::new(0.5_f32.sqrt(), 0.0),
    ///         stim::Complex32::new(0.5_f32.sqrt(), 0.0),
    ///     ]
    /// );
    /// ```
    pub fn to_state_vector(&self, endian: &str) -> crate::Result<Vec<crate::Complex32>> {
        let flat = self
            .inner
            .to_state_vector_data(endian)
            .map_err(crate::StimError::from)?;
        if flat.len() % 2 != 0 {
            return Err(crate::StimError::new(
                "unexpected flat state vector size from stim-cxx",
            ));
        }
        Ok(flat
            .chunks_exact(2)
            .map(|pair| crate::Complex32::new(pair[0], pair[1]))
            .collect())
    }

    /// Returns an owned copy of the tableau.
    #[must_use]
    pub fn copy(&self) -> Self {
        self.clone()
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
            cnot.to_unitary_matrix("big").unwrap(),
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
        let err = Tableau::from_named_gate("H")
            .unwrap()
            .to_unitary_matrix("middle")
            .unwrap_err();
        assert!(err.message().contains("endian"));
    }

    #[test]
    fn tableau_to_state_vector_matches_documented_examples() {
        let i2 = Tableau::from_named_gate("I").unwrap();
        let x = Tableau::from_named_gate("X").unwrap();
        let h = Tableau::from_named_gate("H").unwrap();

        assert_eq!(
            (x.clone() + i2.clone()).to_state_vector("little").unwrap(),
            vec![
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
            ]
        );
        assert_eq!(
            (i2.clone() + x.clone()).to_state_vector("little").unwrap(),
            vec![
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
            ]
        );
        assert_eq!(
            (i2 + x).to_state_vector("big").unwrap(),
            vec![
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
            ]
        );
        assert_eq!(
            (h.clone() + h).to_state_vector("little").unwrap(),
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
        let err = Tableau::from_named_gate("H")
            .unwrap()
            .to_state_vector("middle")
            .unwrap_err();
        assert!(err.message().contains("endian"));
    }

    #[test]
    fn tableau_from_state_vector_matches_documented_examples() {
        assert_eq!(
            Tableau::from_state_vector(
                &[
                    Complex32::new(0.5f32.sqrt(), 0.0),
                    Complex32::new(0.0, 0.5f32.sqrt()),
                ],
                "little",
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
                "little",
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
        let bad_endian =
            Tableau::from_state_vector(&[Complex32::new(1.0, 0.0)], "middle").unwrap_err();
        assert!(bad_endian.message().contains("endian"));

        let not_stabilizer = Tableau::from_state_vector(
            &[
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
            ],
            "little",
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
                "little",
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
                "little",
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
            let little = tableau.to_unitary_matrix("little").unwrap();
            assert_eq!(
                Tableau::from_unitary_matrix(&little, "little").unwrap(),
                tableau
            );

            let big = tableau.to_unitary_matrix("big").unwrap();
            assert_eq!(Tableau::from_unitary_matrix(&big, "big").unwrap(), tableau);
        }
    }

    #[test]
    fn tableau_from_unitary_matrix_reports_documented_error_shapes() {
        let not_square = Tableau::from_unitary_matrix(
            &[
                vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
                vec![Complex32::new(0.0, 0.0)],
            ],
            "little",
        )
        .unwrap_err();
        assert!(not_square.message().contains("square"));

        let bad_endian =
            Tableau::from_unitary_matrix(&[vec![Complex32::new(1.0, 0.0)]], "middle").unwrap_err();
        assert!(bad_endian.message().contains("endian"));

        let not_clifford = Tableau::from_unitary_matrix(
            &[
                vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
                vec![Complex32::new(0.0, 0.0), Complex32::new(0.0, 0.0)],
            ],
            "little",
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
        let t2 = t1.copy();

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
            tableau.to_circuit_with_method("elimination").unwrap()
        );
        assert_eq!(
            tableau.to_circuit().unwrap().to_string(),
            "S 0\nH 0 1 3\nCX 0 1 0 2 0 3\nS 1 3\nH 1 3\nCX 1 0 3 0 3 1 1 3 3 1\nH 1\nS 1\nCX 1 3\nH 2 3\nCX 2 1 3 1 3 2 2 3 3 2\nH 3\nCX 2 3\nS 3\nH 3 0 1 2\nS 0 0 1 1 2 2\nH 0 1 2\nS 3 3"
        );
        assert_eq!(
            tableau
                .to_circuit_with_method("graph_state")
                .unwrap()
                .to_string(),
            "RX 0 1 2 3\nTICK\nCZ 0 3 1 2 1 3\nTICK\nX 0 1\nZ 2\nS 2 3\nH 3\nS 3"
        );
        assert_eq!(
            tableau
                .to_circuit_with_method("mpp_state_unsigned")
                .unwrap()
                .to_string(),
            "MPP X0*Z1*Y2*Y3 !X0*Y1*X2 !Z0*X1*X2*Z3 X0*X1*Z2"
        );
        assert_eq!(
            tableau
                .to_circuit_with_method("mpp_state")
                .unwrap()
                .to_string(),
            "MPP X0*Z1*Y2*Y3 !X0*Y1*X2 !Z0*X1*X2*Z3 X0*X1*Z2\nCX rec[-3] 2 rec[-1] 2\nCY rec[-4] 0 rec[-3] 0 rec[-3] 3 rec[-2] 3 rec[-1] 0\nCZ rec[-4] 1 rec[-1] 1"
        );

        let err = tableau.to_circuit_with_method("nope").unwrap_err();
        assert!(err.to_string().contains("Unknown method"));
        assert!(err.to_string().contains("mpp_state_unsigned"));
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
        let via_tableau = Tableau::from_circuit(&circuit, false, false, false).unwrap();
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
        let copied = t.copy();
        let px = t.inverse_x_output(0, false);
        let py = t.inverse_y_output(0, false);
        let pz = t.inverse_z_output(0, false);

        assert_eq!(copied, t);
        assert_eq!(copied.to_string(), t.to_string());
        assert_eq!(px, t_inv.x_output(0));
        assert_eq!(py, t_inv.y_output(0));
        assert_eq!(pz, t_inv.z_output(0));
        assert_eq!(px.copy(), px);
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
