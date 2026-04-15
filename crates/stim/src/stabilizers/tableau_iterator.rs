/// An iterator that yields all Clifford [`crate::Tableau`] values of a given
/// qubit count.
///
/// Created by [`crate::Tableau::iter_all`], this iterator lazily enumerates
/// every stabilizer tableau (i.e. every Clifford operation) over the specified
/// number of qubits. The number of tableaux grows extremely fast with qubit
/// count: there are 24 single-qubit Cliffords, 11,520 two-qubit Cliffords,
/// and so on.
///
/// The iterator can be configured to produce only unsigned tableaux (those
/// with all-positive stabilizer signs), which reduces the count by a factor
/// of 2^(2n) for `n` qubits.
///
/// The iterator can be cloned to create an independent copy that resumes from
/// the same position.
///
/// # Examples
///
/// ```
/// // Count all single-qubit Clifford operations.
/// let count = stim::Tableau::iter_all(1, false).count();
/// assert_eq!(count, 24);
///
/// // Unsigned single-qubit Cliffords (ignoring sign): 24 / 2^(2*1) = 6.
/// let unsigned_count = stim::Tableau::iter_all(1, true).count();
/// assert_eq!(unsigned_count, 6);
/// ```
#[derive(Clone)]
pub struct TableauIterator {
    pub(crate) inner: stim_cxx::TableauIterator,
}

impl Iterator for TableauIterator {
    type Item = crate::Tableau;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|inner| crate::Tableau { inner })
    }
}
