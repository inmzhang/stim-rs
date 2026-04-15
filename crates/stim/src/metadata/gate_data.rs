use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};

use ndarray::Array2;

use crate::{Flow, Result, StimError, Tableau};

/// Metadata about a Stim gate.
pub struct GateData {
    pub(crate) inner: stim_cxx::GateData,
}

impl GateData {
    /// Looks up metadata for a gate by name or alias.
    ///
    /// # Examples
    ///
    /// ```
    /// let gate = stim::GateData::new("H").unwrap();
    /// assert_eq!(gate.name(), "H");
    /// assert!(gate.is_unitary());
    /// ```
    pub fn new(name: &str) -> Result<Self> {
        gate_data(name)
    }

    /// Returns the canonical gate name.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(stim::gate_data("cnot").unwrap().name(), "CX");
    /// ```
    #[must_use]
    pub fn name(&self) -> String {
        self.inner.name()
    }

    /// Returns the accepted aliases for the gate.
    ///
    /// # Examples
    ///
    /// ```
    /// let cx = stim::gate_data("CX").unwrap();
    /// assert!(cx.aliases().contains(&"CNOT".to_string()));
    /// assert!(cx.aliases().contains(&"CX".to_string()));
    /// ```
    #[must_use]
    pub fn aliases(&self) -> Vec<String> {
        self.inner.aliases()
    }

    /// Returns the allowed number of parenthesized numeric arguments.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(stim::gate_data("H").unwrap().num_parens_arguments_range(), vec![0]);
    /// assert_eq!(stim::gate_data("M").unwrap().num_parens_arguments_range(), vec![0, 1]);
    /// ```
    #[must_use]
    pub fn num_parens_arguments_range(&self) -> Vec<u8> {
        self.inner.num_parens_arguments_range()
    }

    /// Returns whether the gate is noisy.
    #[must_use]
    pub fn is_noisy_gate(&self) -> bool {
        self.inner.is_noisy_gate()
    }

    /// Returns whether the gate is a reset.
    #[must_use]
    pub fn is_reset(&self) -> bool {
        self.inner.is_reset()
    }

    /// Returns whether the gate acts on one qubit at a time.
    #[must_use]
    pub fn is_single_qubit_gate(&self) -> bool {
        self.inner.is_single_qubit_gate()
    }

    /// Returns whether the gate is symmetric in its targets.
    #[must_use]
    pub fn is_symmetric_gate(&self) -> bool {
        self.inner.is_symmetric_gate()
    }

    /// Returns whether the gate acts on two qubits at a time.
    #[must_use]
    pub fn is_two_qubit_gate(&self) -> bool {
        self.inner.is_two_qubit_gate()
    }

    /// Returns whether the gate is unitary.
    #[must_use]
    pub fn is_unitary(&self) -> bool {
        self.inner.is_unitary()
    }

    /// Returns whether the gate produces measurements.
    #[must_use]
    pub fn produces_measurements(&self) -> bool {
        self.inner.produces_measurements()
    }

    /// Returns whether the gate accepts measurement-record targets.
    #[must_use]
    pub fn takes_measurement_record_targets(&self) -> bool {
        self.inner.takes_measurement_record_targets()
    }

    /// Returns whether the gate accepts Pauli-product targets.
    #[must_use]
    pub fn takes_pauli_targets(&self) -> bool {
        self.inner.takes_pauli_targets()
    }

    /// Returns the stabilizer flows associated with the gate, when present.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(
    ///     stim::gate_data("H").unwrap().flows().unwrap(),
    ///     vec![
    ///         stim::Flow::from_text("X -> Z").unwrap(),
    ///         stim::Flow::from_text("Z -> X").unwrap(),
    ///     ]
    /// );
    /// ```
    #[must_use]
    pub fn flows(&self) -> Option<Vec<Flow>> {
        let flows = self
            .inner
            .flows()
            .into_iter()
            .map(|text| Flow::from_text(&text))
            .collect::<Result<Vec<_>>>()
            .expect("gate flow texts from stim-cxx should be valid canonical flow text");
        if flows.is_empty() { None } else { Some(flows) }
    }

    /// Returns the tableau of the gate, when it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("M").unwrap().tableau().is_none());
    /// assert_eq!(
    ///     stim::gate_data("H").unwrap().tableau().unwrap(),
    ///     stim::Tableau::from_named_gate("H").unwrap()
    /// );
    /// ```
    #[must_use]
    pub fn tableau(&self) -> Option<Tableau> {
        self.inner.tableau().map(|inner| Tableau { inner })
    }

    /// Returns the unitary matrix of the gate when it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("M").unwrap().unitary_matrix().is_none());
    /// assert_eq!(
    ///     stim::gate_data("X").unwrap().unitary_matrix().unwrap(),
    ///     ndarray::array![
    ///         [stim::Complex32::new(0.0, 0.0), stim::Complex32::new(1.0, 0.0)],
    ///         [stim::Complex32::new(1.0, 0.0), stim::Complex32::new(0.0, 0.0)],
    ///     ]
    /// );
    /// ```
    #[must_use]
    pub fn unitary_matrix(&self) -> Option<Array2<crate::Complex32>> {
        self.tableau().map(|tableau| {
            let matrix = tableau
                .to_unitary_matrix("big")
                .expect("unitary gate tableaux should convert into unitary matrices");
            let nrows = matrix.len();
            let ncols = matrix.first().map_or(0, Vec::len);
            Array2::from_shape_vec((nrows, ncols), matrix.into_iter().flatten().collect())
                .expect("unitary gate matrices should be rectangular")
        })
    }

    /// Returns the ordinary inverse of the gate when one exists.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(stim::gate_data("S").unwrap().inverse().unwrap().name(), "S_DAG");
    /// assert!(stim::gate_data("X_ERROR").unwrap().inverse().is_none());
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        self.inner.inverse().map(|inner| Self { inner })
    }

    /// Returns the generalized inverse of the gate.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(
    ///     stim::gate_data("R").unwrap().generalized_inverse().name(),
    ///     "M"
    /// );
    /// assert_eq!(
    ///     stim::gate_data("X_ERROR").unwrap().generalized_inverse().name(),
    ///     "X_ERROR"
    /// );
    /// ```
    #[must_use]
    pub fn generalized_inverse(&self) -> Self {
        Self {
            inner: self.inner.generalized_inverse(),
        }
    }

    /// Returns the Hadamard-conjugated gate when one exists.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(
    ///     stim::gate_data("X")
    ///         .unwrap()
    ///         .hadamard_conjugated(false)
    ///         .unwrap()
    ///         .name(),
    ///     "Z"
    /// );
    /// ```
    #[must_use]
    pub fn hadamard_conjugated(&self, unsigned_only: bool) -> Option<Self> {
        self.inner
            .hadamard_conjugated(unsigned_only)
            .map(|inner| Self { inner })
    }
}

/// Looks up metadata for a gate by name or alias.
///
/// # Examples
///
/// ```
/// let h = stim::gate_data("H").unwrap();
/// assert_eq!(h.name(), "H");
/// ```
pub fn gate_data(name: &str) -> Result<GateData> {
    stim_cxx::gate_data(name)
        .map(|inner| GateData { inner })
        .map_err(StimError::from)
}

/// Returns the canonical gate inventory keyed by canonical gate name.
///
/// # Examples
///
/// ```
/// let gates = stim::all_gate_data();
/// assert!(gates.contains_key("H"));
/// assert!(gates.contains_key("CX"));
/// ```
#[must_use]
pub fn all_gate_data() -> BTreeMap<String, GateData> {
    stim_cxx::all_gate_names()
        .into_iter()
        .map(|name| {
            let gate = gate_data(&name).expect("canonical Stim gate name should resolve");
            (name, gate)
        })
        .collect()
}

impl Clone for GateData {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl PartialEq for GateData {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for GateData {}

impl Display for GateData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

impl fmt::Debug for GateData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "stim::gate_data({:?})", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::GateData;
    use crate::{Complex32, Flow, all_gate_data, gate_data};

    #[test]
    fn gate_data_new_matches_lookup() {
        let gate = GateData::new("H").unwrap();
        assert_eq!(gate.name(), "H");
        assert!(gate.is_unitary());
    }

    #[test]
    fn gate_data_debug_uses_lookup_form() {
        let gate = gate_data("cnot").expect("gate should parse");

        assert_eq!(format!("{gate:?}"), "stim::gate_data(\"CX\")");
    }

    #[test]
    fn gate_data_resolves_aliases_to_canonical_entries() {
        let canonical = gate_data("CX").expect("canonical gate should resolve");
        let alias = gate_data("cnot").expect("alias should resolve");

        assert_eq!(canonical.name(), "CX");
        assert_eq!(alias.name(), "CX");
        assert_eq!(canonical, alias);
        assert!(canonical.aliases().contains(&"CNOT".to_string()));
        assert!(canonical.aliases().contains(&"CX".to_string()));
    }

    #[test]
    fn gate_data_exposes_representative_metadata_flags() {
        let h = gate_data("H").expect("gate should exist");
        let x_error = gate_data("X_ERROR").expect("gate should exist");
        let m = gate_data("M").expect("gate should exist");
        let r = gate_data("R").expect("gate should exist");
        let detector = gate_data("DETECTOR").expect("gate should exist");
        let cx = gate_data("CX").expect("gate should exist");
        let cz = gate_data("CZ").expect("gate should exist");

        assert!(h.is_single_qubit_gate());
        assert!(h.is_unitary());
        assert!(!h.is_noisy_gate());
        assert_eq!(h.num_parens_arguments_range(), vec![0]);

        assert!(x_error.is_noisy_gate());
        assert_eq!(x_error.num_parens_arguments_range(), vec![1]);

        assert!(cx.is_two_qubit_gate());
        assert!(!cx.is_symmetric_gate());
        assert!(cz.is_symmetric_gate());

        assert_eq!(m.num_parens_arguments_range(), vec![0, 1]);
        assert!(m.produces_measurements());

        assert!(r.is_reset());
        assert_eq!(r.num_parens_arguments_range(), vec![0]);

        assert!(detector.takes_measurement_record_targets());
        assert!(!detector.takes_pauli_targets());
        assert!(!detector.produces_measurements());
    }

    #[test]
    fn gate_data_exposes_inverse_and_hadamard_relationships() {
        let h = gate_data("H").expect("gate should exist");
        let s = gate_data("S").expect("gate should exist");
        let x_error = gate_data("X_ERROR").expect("gate should exist");
        let r = gate_data("R").expect("gate should exist");
        let x = gate_data("X").expect("gate should exist");
        let cx = gate_data("CX").expect("gate should exist");
        let ry = gate_data("RY").expect("gate should exist");

        assert_eq!(h.inverse().expect("H has inverse").name(), "H");
        assert_eq!(s.inverse().expect("S has inverse").name(), "S_DAG");
        assert!(x_error.inverse().is_none());

        assert_eq!(x_error.generalized_inverse().name(), "X_ERROR");
        assert_eq!(r.generalized_inverse().name(), "M");

        assert_eq!(
            x.hadamard_conjugated(false)
                .expect("X has H conjugate")
                .name(),
            "Z"
        );
        assert_eq!(
            cx.hadamard_conjugated(false)
                .expect("CX has H conjugate")
                .name(),
            "XCZ"
        );
        assert!(ry.hadamard_conjugated(false).is_none());
        assert_eq!(
            ry.hadamard_conjugated(true)
                .expect("unsigned H conjugate exists")
                .name(),
            "RY"
        );
    }

    #[test]
    fn gate_data_has_stable_identity_and_representation() {
        let canonical = gate_data("CX").expect("gate should exist");
        let alias = gate_data("cnot").expect("alias should resolve");
        let cloned = alias.clone();
        let other = gate_data("H").expect("other gate should exist");
        let mpp = gate_data("mpp").expect("gate should exist");

        assert_eq!(canonical, alias);
        assert_eq!(cloned, canonical);
        assert_ne!(canonical, other);

        assert_eq!(canonical.to_string(), "CX");
        assert_eq!(format!("{canonical:?}"), "stim::gate_data(\"CX\")");
        assert_eq!(alias.to_string(), "CX");
        assert_eq!(format!("{alias:?}"), "stim::gate_data(\"CX\")");
        assert_eq!(mpp.to_string(), "MPP");
        assert_eq!(format!("{mpp:?}"), "stim::gate_data(\"MPP\")");
    }

    #[test]
    fn gate_data_reports_unknown_gate_names() {
        let error = gate_data("definitely_not_a_gate").expect_err("unknown gate should fail");

        assert!(error.message().contains("definitely_not_a_gate"));
    }

    #[test]
    fn all_gate_data_enumerates_canonical_inventory_with_roundtrip_invariants() {
        let inventory = all_gate_data();

        assert!(inventory.contains_key("CX"));
        assert!(inventory.contains_key("DETECTOR"));
        assert!(inventory.contains_key("H"));
        assert!(inventory.contains_key("MPP"));
        assert!(!inventory.contains_key("CNOT"));
        assert!(!inventory.contains_key("NOT_A_GATE"));

        let cx = inventory.get("CX").expect("CX should be present");
        assert_eq!(cx, &gate_data("cnot").expect("alias lookup should resolve"));
        assert_eq!(cx.name(), "CX");
        assert!(cx.aliases().contains(&"CNOT".to_string()));

        for (name, gate) in &inventory {
            assert_eq!(gate.name(), *name);
            assert_eq!(
                gate,
                &gate_data(name).expect("inventory key should roundtrip")
            );
            assert_eq!(gate.to_string(), *name);
            assert_eq!(format!("{gate:?}"), format!("stim::gate_data({name:?})"));
        }
    }

    #[test]
    fn gate_data_flows_match_documented_examples() {
        assert_eq!(
            gate_data("H").unwrap().flows().unwrap(),
            vec![
                Flow::from_text("X -> Z").unwrap(),
                Flow::from_text("Z -> X").unwrap(),
            ]
        );

        let iswap_flows: Vec<String> = gate_data("ISWAP")
            .unwrap()
            .flows()
            .unwrap()
            .into_iter()
            .map(|flow| flow.to_string())
            .collect();
        assert_eq!(
            iswap_flows,
            vec!["X_ -> ZY", "Z_ -> _Z", "_X -> YZ", "_Z -> Z_"]
        );

        let mxx_flows: Vec<String> = gate_data("MXX")
            .unwrap()
            .flows()
            .unwrap()
            .into_iter()
            .map(|flow| flow.to_string())
            .collect();
        assert_eq!(
            mxx_flows,
            vec!["X_ -> X_", "_X -> _X", "ZZ -> ZZ", "XX -> rec[-1]"]
        );
    }

    #[test]
    fn gate_data_tableau_matches_documented_examples() {
        assert!(gate_data("M").unwrap().tableau().is_none());

        assert_eq!(
            format!("{:?}", gate_data("H").unwrap().tableau().unwrap()),
            "stim.Tableau.from_conjugated_generators(\n    xs=[\n        stim.PauliString(\"+Z\"),\n    ],\n    zs=[\n        stim.PauliString(\"+X\"),\n    ],\n)"
        );

        assert_eq!(
            format!("{:?}", gate_data("ISWAP").unwrap().tableau().unwrap()),
            "stim.Tableau.from_conjugated_generators(\n    xs=[\n        stim.PauliString(\"+ZY\"),\n        stim.PauliString(\"+YZ\"),\n    ],\n    zs=[\n        stim.PauliString(\"+_Z\"),\n        stim.PauliString(\"+Z_\"),\n    ],\n)"
        );
    }

    #[test]
    fn gate_data_unitary_matrix_matches_documented_examples() {
        assert!(gate_data("M").unwrap().unitary_matrix().is_none());

        assert_eq!(
            gate_data("X").unwrap().unitary_matrix().unwrap(),
            ndarray::array![
                [Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)],
                [Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
            ]
        );

        assert_eq!(
            gate_data("ISWAP").unwrap().unitary_matrix().unwrap(),
            ndarray::array![
                [
                    Complex32::new(1.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                [
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 1.0),
                    Complex32::new(0.0, 0.0),
                ],
                [
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 1.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
                [
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(1.0, 0.0),
                ],
            ]
        );
    }
}
