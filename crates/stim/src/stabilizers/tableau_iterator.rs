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
