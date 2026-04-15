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
