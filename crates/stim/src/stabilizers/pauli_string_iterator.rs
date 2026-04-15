/// An iterator that yields [`crate::PauliString`] values matching specified
/// weight and Pauli-type filters.
///
/// Created by [`crate::PauliString::iter_all`], this iterator lazily
/// enumerates Pauli strings over a given number of qubits, subject to
/// constraints on minimum/maximum Hamming weight and which Pauli operators
/// (`X`, `Y`, `Z`) are allowed at non-identity positions.
///
/// The iterator produces strings in a deterministic canonical order. It can
/// be cloned to create an independent copy that resumes from the same
/// position.
///
/// # Examples
///
/// ```
/// // Iterate over all weight-1 Pauli strings on 2 qubits using only X and Z.
/// let paulis: Vec<_> = stim::PauliString::iter_all(2, 1, 1, "XZ")
///     .map(|p| p.to_string())
///     .collect();
/// assert_eq!(paulis, ["+X_", "+Z_", "+_X", "+_Z"]);
/// ```
#[derive(Clone)]
pub struct PauliStringIterator {
    pub(crate) inner: stim_cxx::PauliStringIterator,
}

impl Iterator for PauliStringIterator {
    type Item = crate::PauliString;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|inner| crate::PauliString { inner, imag: false })
    }
}
