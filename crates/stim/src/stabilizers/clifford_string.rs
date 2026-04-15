use std::fmt::{self, Display, Formatter};
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// A replacement value accepted by [`CliffordString::set`].
///
/// Either a gate name string or a [`crate::GateData`] reference.
pub enum CliffordGateValue<'a> {
    /// A gate name like `"H"` or `"S"`.
    Name(&'a str),
    /// A reference to gate metadata.
    GateData(&'a crate::GateData),
}

impl<'a> From<&'a str> for CliffordGateValue<'a> {
    fn from(value: &'a str) -> Self {
        Self::Name(value)
    }
}

impl<'a> From<&'a crate::GateData> for CliffordGateValue<'a> {
    fn from(value: &'a crate::GateData) -> Self {
        Self::GateData(value)
    }
}

/// A sequence of single-qubit Clifford gates.
#[derive(Clone, PartialEq, Eq)]
pub struct CliffordString {
    pub(crate) inner: stim_cxx::CliffordString,
}

impl CliffordString {
    /// Creates an identity Clifford string over `num_qubits` qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// let c = stim::CliffordString::new(3);
    /// assert_eq!(c.to_string(), "I,I,I");
    /// assert_eq!(c.len(), 3);
    /// ```
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            inner: stim_cxx::CliffordString::new(num_qubits),
        }
    }

    /// Parses a Clifford string from comma-separated gate names.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = stim::CliffordString::from_text("X,Y,Z,H,SQRT_X,C_XYZ").unwrap();
    /// assert_eq!(s.get(2).unwrap().name(), "Z");
    /// assert_eq!(s.get(-1).unwrap().name(), "C_XYZ");
    /// ```
    pub fn from_text(text: &str) -> crate::Result<Self> {
        stim_cxx::CliffordString::from_text(text)
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Converts a Pauli string into the corresponding Clifford string.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = stim::PauliString::from_text("-XYZ").unwrap();
    /// let c = stim::CliffordString::from_pauli_string(&p);
    /// assert_eq!(c.to_string(), "X,Y,Z");
    /// ```
    #[must_use]
    pub fn from_pauli_string(pauli_string: &crate::PauliString) -> Self {
        Self {
            inner: stim_cxx::CliffordString::from_pauli_string(&pauli_string.inner),
        }
    }

    /// Extracts the Clifford action of a circuit as a Clifford string.
    ///
    /// # Examples
    ///
    /// ```
    /// let circuit: stim::Circuit = "H 0 1\nS 1 2".parse().unwrap();
    /// let c = stim::CliffordString::from_circuit(&circuit).unwrap();
    /// assert_eq!(c.to_string(), "H,C_ZYX,S");
    /// ```
    pub fn from_circuit(circuit: &crate::Circuit) -> crate::Result<Self> {
        stim_cxx::CliffordString::from_circuit(&circuit.inner)
            .map(|inner| Self { inner })
            .map_err(crate::StimError::from)
    }

    /// Returns a random Clifford string of the requested length.
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

    /// Returns the 24 single-qubit Clifford gates in Stim's canonical order.
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

    /// Returns an owned copy of the Clifford string.
    #[must_use]
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Returns the number of single-qubit Clifford entries.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of single-qubit Clifford entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_qubits()
    }

    /// Returns whether the Clifford string is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a gate by index.
    ///
    /// Negative indices count from the end.
    ///
    /// # Examples
    ///
    /// ```
    /// let c = stim::CliffordString::from_text("I,X,H").unwrap();
    /// assert_eq!(c.get(1).unwrap().name(), "X");
    /// assert_eq!(c.get(-1).unwrap().name(), "H");
    /// ```
    pub fn get(&self, index: isize) -> crate::Result<crate::GateData> {
        self.inner
            .get_item_name(index as i64)
            .map_err(crate::StimError::from)
            .and_then(|name| crate::gate_data(&name))
    }

    /// Replaces a gate by index.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut c = stim::CliffordString::from_text("I,I,I").unwrap();
    /// c.set(1, "H").unwrap();
    /// c.set(-1, &stim::gate_data("S").unwrap()).unwrap();
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
            CliffordGateValue::Name(name) => crate::gate_data(name)?.name(),
            CliffordGateValue::GateData(gate) => gate.name(),
        };
        let mut parts: Vec<String> = if self.is_empty() {
            Vec::new()
        } else {
            self.to_string().split(',').map(str::to_owned).collect()
        };
        parts[normalized] = replacement;
        *self = Self::from_text(&parts.join(","))?;
        Ok(())
    }

    /// Returns a sliced Clifford string.
    ///
    /// # Examples
    ///
    /// ```
    /// let c = stim::CliffordString::from_text("I,X,H,Y,S").unwrap();
    /// assert_eq!(c.slice(None, Some(-1), 1).unwrap().to_string(), "I,X,H,Y");
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

    /// Returns the Clifford outputs of the X generators.
    ///
    /// The returned boolean vector contains the sign bits for each output.
    ///
    /// # Examples
    ///
    /// ```
    /// let (paulis, signs) = stim::CliffordString::from_text("I,Y,H,S").unwrap().x_outputs();
    /// assert_eq!(paulis, stim::PauliString::from_text("+XXZY").unwrap());
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

    /// Returns the X outputs and packed sign bits.
    ///
    /// # Examples
    ///
    /// ```
    /// let (paulis, signs) = stim::CliffordString::from_text("I,Y,H,S").unwrap().x_outputs_bit_packed();
    /// assert_eq!(paulis, stim::PauliString::from_text("+XXZY").unwrap());
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

    /// Returns the Y outputs and unpacked sign bits.
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

    /// Returns the Y outputs and packed sign bits.
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

    /// Returns the Z outputs and unpacked sign bits.
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

    /// Returns the Z outputs and packed sign bits.
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

    /// Raises each Clifford gate in the string to the given power.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut c = stim::CliffordString::from_text("I,X,H,S,C_XYZ").unwrap();
    /// c.ipow(3);
    /// assert_eq!(c, stim::CliffordString::from_text("I,X,H,S_DAG,I").unwrap());
    /// ```
    #[must_use]
    pub fn pow(&self, exponent: i64) -> Self {
        Self {
            inner: self.inner.pow(exponent),
        }
    }

    /// Mutates the Clifford string by raising each entry to a power in place.
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

impl Add for CliffordString {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner.add(&rhs.inner),
        }
    }
}

impl AddAssign for CliffordString {
    fn add_assign(&mut self, rhs: Self) {
        self.inner.add_assign(&rhs.inner);
    }
}

impl Mul<CliffordString> for CliffordString {
    type Output = Self;

    fn mul(self, rhs: CliffordString) -> Self::Output {
        Self {
            inner: self.inner.mul_clifford(&rhs.inner),
        }
    }
}

impl MulAssign<CliffordString> for CliffordString {
    fn mul_assign(&mut self, rhs: CliffordString) {
        self.inner.mul_assign_clifford(&rhs.inner);
    }
}

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

impl MulAssign<u64> for CliffordString {
    fn mul_assign(&mut self, rhs: u64) {
        self.inner
            .repeat_assign(rhs)
            .expect("CliffordString repetition should fit platform size");
    }
}

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

        let s = CliffordString::from_text("  X  ,   Y  ,  Z  , H_XZ , SQRT_X,C_XYZ,   ").unwrap();
        assert_eq!(s.to_string(), "X,Y,Z,H,SQRT_X,C_XYZ");
        assert_eq!(s.get(2).unwrap().name(), "Z");
        assert_eq!(s.get(-1).unwrap().name(), "C_XYZ");
        assert_eq!(
            s.slice(None, Some(-1), 1).unwrap().to_string(),
            "X,Y,Z,H,SQRT_X"
        );
        assert_eq!(s.slice(None, None, 2).unwrap().to_string(), "X,Z,SQRT_X");

        let from_pauli =
            CliffordString::from_pauli_string(&PauliString::from_text("-XYZ").unwrap());
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

        let (x_paulis, x_signs) = CliffordString::from_text("I,Y,H,S").unwrap().x_outputs();
        assert_eq!(x_paulis, PauliString::from_text("+XXZY").unwrap());
        assert_eq!(x_signs, vec![false, true, false, false]);
        assert_eq!(
            CliffordString::from_text("I,Y,H,S")
                .unwrap()
                .x_outputs_bit_packed()
                .1,
            vec![2]
        );

        let (y_paulis, y_signs) = CliffordString::from_text("I,X,H,S").unwrap().y_outputs();
        assert_eq!(y_paulis, PauliString::from_text("+YYYX").unwrap());
        assert_eq!(y_signs, vec![false, true, true, true]);
        assert_eq!(
            CliffordString::from_text("I,X,H,S")
                .unwrap()
                .y_outputs_bit_packed()
                .1,
            vec![14]
        );

        let (z_paulis, z_signs) = CliffordString::from_text("I,Y,H,S").unwrap().z_outputs();
        assert_eq!(z_paulis, PauliString::from_text("+ZZXZ").unwrap());
        assert_eq!(z_signs, vec![false, true, false, false]);
        assert_eq!(
            CliffordString::from_text("I,Y,H,S")
                .unwrap()
                .z_outputs_bit_packed()
                .1,
            vec![2]
        );
    }

    #[test]
    fn clifford_string_arithmetic_and_powers_match_documented_examples() {
        assert_eq!(
            CliffordString::from_text("I,X,H").unwrap() + CliffordString::from_text("Y,S").unwrap(),
            CliffordString::from_text("I,X,H,Y,S").unwrap()
        );

        let mut concat = CliffordString::from_text("I,X,H").unwrap();
        concat += CliffordString::from_text("Y,S").unwrap();
        assert_eq!(concat, CliffordString::from_text("I,X,H,Y,S").unwrap());

        assert_eq!(
            CliffordString::from_text("S,X,X").unwrap()
                * CliffordString::from_text("S,Z,H,Z").unwrap(),
            CliffordString::from_text("Z,Y,SQRT_Y,Z").unwrap()
        );

        let mut mul_assign = CliffordString::from_text("S,X,X").unwrap();
        mul_assign *= CliffordString::from_text("S,Z,H,Z").unwrap();
        assert_eq!(
            mul_assign,
            CliffordString::from_text("Z,Y,SQRT_Y,Z").unwrap()
        );

        assert_eq!(
            CliffordString::from_text("I,X,H").unwrap() * 3,
            CliffordString::from_text("I,X,H,I,X,H,I,X,H").unwrap()
        );
        assert_eq!(
            2 * CliffordString::from_text("I,X,H").unwrap(),
            CliffordString::from_text("I,X,H,I,X,H").unwrap()
        );

        let mut repeated = CliffordString::from_text("I,X,H").unwrap();
        repeated *= 2;
        assert_eq!(repeated, CliffordString::from_text("I,X,H,I,X,H").unwrap());

        let mut p = CliffordString::from_text("I,X,H,S,C_XYZ").unwrap();
        p.ipow(3);
        assert_eq!(p, CliffordString::from_text("I,X,H,S_DAG,I").unwrap());
        p.ipow(2);
        assert_eq!(p, CliffordString::from_text("I,I,I,Z,I").unwrap());
        assert_eq!(p.pow(2), CliffordString::from_text("I,I,I,I,I").unwrap());
        p.ipow(2);
        assert_eq!(p, CliffordString::from_text("I,I,I,I,I").unwrap());
    }

    #[test]
    fn clifford_string_set_updates_single_gate_entries() {
        let mut s = CliffordString::from_text("I,I,I,I,I").unwrap();
        s.set(1, "H").unwrap();
        assert_eq!(s, CliffordString::from_text("I,H,I,I,I").unwrap());
        s.set(-1, &crate::gate_data("S").unwrap()).unwrap();
        assert_eq!(s, CliffordString::from_text("I,H,I,I,S").unwrap());
    }
}
