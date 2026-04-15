use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};

use ndarray::Array2;

use crate::{Flow, Result, StimError, Tableau};

/// Metadata about a gate supported by Stim.
///
/// Every gate that Stim recognises — unitaries such as `H` and `CX`, noise channels
/// such as `X_ERROR` and `DEPOLARIZE1`, measurements (`M`, `MXX`, `MPP`), resets (`R`,
/// `RX`), and annotations (`DETECTOR`, `TICK`) — has an associated `GateData` value
/// that exposes its canonical name, aliases, stabilizer flows, Clifford tableau,
/// unitary matrix, inverse relationships, and various Boolean classification flags.
///
/// Obtain a `GateData` through [`GateData::new`], the free function [`gate_data`], or
/// the bulk inventory [`all_gate_data`].
///
/// Two `GateData` values are equal when they refer to the same canonical gate,
/// regardless of which alias was used to look them up.
///
/// # Examples
///
/// ```
/// let h = stim::gate_data("h").unwrap();
/// assert_eq!(h.name(), "H");
/// assert!(h.is_unitary());
///
/// // The Clifford tableau for H swaps the X and Z bases.
/// let tableau = h.tableau().unwrap();
/// assert_eq!(
///     tableau,
///     stim::Tableau::from_named_gate("H").unwrap()
/// );
/// ```
pub struct GateData {
    pub(crate) inner: stim_cxx::GateData,
}

impl GateData {
    /// Looks up metadata for a gate by name or alias.
    ///
    /// Gate names are case-insensitive: `"h"`, `"H"`, and `"h_xz"` all resolve to the
    /// canonical `"H"` gate. Aliases such as `"CNOT"` are also accepted and will
    /// resolve to the corresponding canonical name (in that case, `"CX"`).
    ///
    /// This is equivalent to calling the free function [`gate_data`].
    ///
    /// # Errors
    ///
    /// Returns an error if `name` does not match any known gate or alias.
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

    /// Returns the canonical name of the gate.
    ///
    /// Each Stim gate has exactly one canonical (upper-case) name. When a gate is
    /// looked up by an alias or a differently-cased variant, the canonical name is
    /// still returned. For example, looking up `"cnot"` yields a `GateData` whose
    /// `name()` is `"CX"`.
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

    /// Returns all aliases that can be used to refer to this gate.
    ///
    /// Every gate has at least one alias — its canonical name. Many gates have
    /// additional historical or convenience aliases. For instance, the `CX` gate can
    /// also be referred to as `CNOT` or `ZCX`. Although gates can be looked up with
    /// lower- or mixed-case names, the returned list contains only the upper-cased
    /// aliases.
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

    /// Returns the range of allowed parenthesized numeric argument counts for this
    /// gate.
    ///
    /// In Stim circuit syntax, gates can take parenthesized arguments — for example,
    /// `X_ERROR(0.01) 0` has one argument (the error probability), while `H 0` has
    /// zero. This method returns the set of valid argument counts as a `Vec<u8>`.
    ///
    /// Common patterns:
    /// - `[0]` — no arguments allowed (e.g. `H`, `R`)
    /// - `[1]` — exactly one argument required (e.g. `X_ERROR`)
    /// - `[0, 1]` — zero or one argument (e.g. `M`, where the optional argument is
    ///   the measurement flip probability)
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

    /// Returns whether the gate can produce noise.
    ///
    /// Noise gates are those whose operation introduces probabilistic errors into the
    /// system. This includes explicit error channels like `X_ERROR`, `DEPOLARIZE1`, and
    /// `CORRELATED_ERROR`, but also measurement operations such as `M`, `MXX`, and
    /// `MPP`, because measurements in Stim can include a flip probability argument
    /// (e.g. `M(0.001) 2 3 5` flips its result 0.1% of the time).
    ///
    /// Unitary gates (`H`, `CX`, …), resets (`R`, `RX`, …), and annotations
    /// (`DETECTOR`, `TICK`, …) are *not* noisy.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("X_ERROR").unwrap().is_noisy_gate());
    /// assert!(stim::gate_data("M").unwrap().is_noisy_gate());
    /// assert!(!stim::gate_data("H").unwrap().is_noisy_gate());
    /// assert!(!stim::gate_data("R").unwrap().is_noisy_gate());
    /// ```
    #[must_use]
    pub fn is_noisy_gate(&self) -> bool {
        self.inner.is_noisy_gate()
    }

    /// Returns whether the gate resets qubits in any basis.
    ///
    /// Reset gates force qubits into a fixed state, destroying whatever state the
    /// qubit previously held. This includes `R` (reset to |0⟩), `RX` (reset to |+⟩),
    /// `RY`, and combined measure-reset gates like `MR` and `MRY`.
    ///
    /// Measurement-only gates (`M`, `MXX`, `MPP`), unitary gates, noise channels, and
    /// annotations do *not* count as resets.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("R").unwrap().is_reset());
    /// assert!(stim::gate_data("MR").unwrap().is_reset());
    /// assert!(!stim::gate_data("M").unwrap().is_reset());
    /// assert!(!stim::gate_data("H").unwrap().is_reset());
    /// ```
    #[must_use]
    pub fn is_reset(&self) -> bool {
        self.inner.is_reset()
    }

    /// Returns whether the gate acts on a single qubit at a time.
    ///
    /// Single-qubit gates apply their operation independently to each of their
    /// targets. For example, `H 0 1 2` applies three independent Hadamard operations.
    ///
    /// Variable-target-count gates like `CORRELATED_ERROR` and `MPP` are *not*
    /// considered single-qubit gates, even when they happen to target only one qubit.
    /// Annotations like `DETECTOR` and `TICK` are also not single-qubit gates.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("H").unwrap().is_single_qubit_gate());
    /// assert!(stim::gate_data("M").unwrap().is_single_qubit_gate());
    /// assert!(stim::gate_data("X_ERROR").unwrap().is_single_qubit_gate());
    /// assert!(!stim::gate_data("CX").unwrap().is_single_qubit_gate());
    /// assert!(!stim::gate_data("MPP").unwrap().is_single_qubit_gate());
    /// ```
    #[must_use]
    pub fn is_single_qubit_gate(&self) -> bool {
        self.inner.is_single_qubit_gate()
    }

    /// Returns whether the gate is unchanged when its targets are swapped.
    ///
    /// A two-qubit gate is symmetric if swapping its two targets has no observable
    /// effect — equivalently, if it is unaffected when conjugated by `SWAP`. For
    /// example, `CZ` is symmetric (control and target are interchangeable), while `CX`
    /// is not (the control and target roles differ).
    ///
    /// Single-qubit gates are vacuously symmetric. Multi-qubit gates are symmetric if
    /// swapping *any* pair of their targets has no effect.
    ///
    /// Note: symmetry is checked *without broadcasting*. `SWAP` is symmetric even
    /// though `SWAP 1 2 3 4` is not the same circuit as `SWAP 1 3 2 4`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("CZ").unwrap().is_symmetric_gate());
    /// assert!(stim::gate_data("ISWAP").unwrap().is_symmetric_gate());
    /// assert!(!stim::gate_data("CX").unwrap().is_symmetric_gate());
    /// assert!(!stim::gate_data("CXSWAP").unwrap().is_symmetric_gate());
    /// ```
    #[must_use]
    pub fn is_symmetric_gate(&self) -> bool {
        self.inner.is_symmetric_gate()
    }

    /// Returns whether the gate acts on exactly two qubits at a time.
    ///
    /// Two-qubit gates must be given an even number of targets in a Stim circuit,
    /// because the targets are consumed in pairs. For example, `CX 0 1 2 3` applies
    /// two CX operations: one to qubits (0, 1) and another to qubits (2, 3).
    ///
    /// Variable-target-count gates like `CORRELATED_ERROR` and `MPP` are *not*
    /// considered two-qubit gates, even when they happen to target exactly two qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("CX").unwrap().is_two_qubit_gate());
    /// assert!(stim::gate_data("MXX").unwrap().is_two_qubit_gate());
    /// assert!(!stim::gate_data("H").unwrap().is_two_qubit_gate());
    /// assert!(!stim::gate_data("MPP").unwrap().is_two_qubit_gate());
    /// ```
    #[must_use]
    pub fn is_two_qubit_gate(&self) -> bool {
        self.inner.is_two_qubit_gate()
    }

    /// Returns whether the gate is a unitary operation.
    ///
    /// Unitary gates are reversible quantum operations whose action can be described by
    /// a unitary matrix and a Clifford tableau. This includes single-qubit Cliffords
    /// (`H`, `S`, `X`, `Y`, `Z`, …) and multi-qubit Cliffords (`CX`, `CZ`, `SWAP`,
    /// `ISWAP`, …).
    ///
    /// Resets, measurements, noise channels, and annotations are *not* unitary.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("H").unwrap().is_unitary());
    /// assert!(stim::gate_data("CX").unwrap().is_unitary());
    /// assert!(!stim::gate_data("M").unwrap().is_unitary());
    /// assert!(!stim::gate_data("R").unwrap().is_unitary());
    /// assert!(!stim::gate_data("X_ERROR").unwrap().is_unitary());
    /// ```
    #[must_use]
    pub fn is_unitary(&self) -> bool {
        self.inner.is_unitary()
    }

    /// Returns whether the gate produces measurement results.
    ///
    /// Gates that produce measurements append one or more bits to the measurement
    /// record when they are executed. This includes single-qubit measurements (`M`,
    /// `MX`, `MY`), measure-and-reset gates (`MR`, `MRX`, `MRY`), two-qubit
    /// measurements (`MXX`, `MYY`, `MZZ`), multi-body measurements (`MPP`), and
    /// heralded erasure (`HERALDED_ERASE`).
    ///
    /// Unitary gates, resets, noise channels, and annotations do *not* produce
    /// measurements.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("M").unwrap().produces_measurements());
    /// assert!(stim::gate_data("MPP").unwrap().produces_measurements());
    /// assert!(!stim::gate_data("H").unwrap().produces_measurements());
    /// assert!(!stim::gate_data("R").unwrap().produces_measurements());
    /// assert!(!stim::gate_data("DETECTOR").unwrap().produces_measurements());
    /// ```
    #[must_use]
    pub fn produces_measurements(&self) -> bool {
        self.inner.produces_measurements()
    }

    /// Returns whether the gate can accept measurement-record (`rec`) targets.
    ///
    /// Some gates allow referencing previous measurement results as targets using
    /// `rec[-k]` syntax in Stim circuits. For example, `CX rec[-1] 1` applies a
    /// controlled-X conditioned on the most recent measurement result. `DETECTOR`
    /// uses record targets to declare which measurements to compare.
    ///
    /// Most gates (unitaries acting on qubits, measurements, resets, noise channels)
    /// do *not* accept record targets.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("CX").unwrap().takes_measurement_record_targets());
    /// assert!(stim::gate_data("DETECTOR").unwrap().takes_measurement_record_targets());
    /// assert!(!stim::gate_data("H").unwrap().takes_measurement_record_targets());
    /// assert!(!stim::gate_data("M").unwrap().takes_measurement_record_targets());
    /// ```
    #[must_use]
    pub fn takes_measurement_record_targets(&self) -> bool {
        self.inner.takes_measurement_record_targets()
    }

    /// Returns whether the gate expects Pauli-product targets.
    ///
    /// Some gates operate on Pauli-product targets rather than plain qubit indices.
    /// In Stim circuit syntax these look like `X0`, `Y1`, `Z2` rather than bare `0`,
    /// `1`, `2`. The two main examples are `CORRELATED_ERROR` (which applies a
    /// correlated Pauli error across specified qubits) and `MPP` (which measures
    /// multi-body Pauli products).
    ///
    /// Most gates (unitaries, single-qubit measurements, resets, single-qubit noise)
    /// take plain qubit targets, not Pauli targets.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(stim::gate_data("CORRELATED_ERROR").unwrap().takes_pauli_targets());
    /// assert!(stim::gate_data("MPP").unwrap().takes_pauli_targets());
    /// assert!(!stim::gate_data("H").unwrap().takes_pauli_targets());
    /// assert!(!stim::gate_data("CX").unwrap().takes_pauli_targets());
    /// assert!(!stim::gate_data("X_ERROR").unwrap().takes_pauli_targets());
    /// ```
    #[must_use]
    pub fn takes_pauli_targets(&self) -> bool {
        self.inner.takes_pauli_targets()
    }

    /// Returns the stabilizer flow generators for the gate, or `None` if the gate has
    /// no fixed set of flows.
    ///
    /// A stabilizer flow describes an input-output relationship that a gate satisfies:
    /// an input Pauli string is transformed into an output Pauli string, potentially
    /// mediated by certain measurement results. For unitary gates the flows correspond
    /// to the Clifford tableau conjugation rules; for measurement and reset gates the
    /// flows capture the measurement and reset semantics.
    ///
    /// Returns `None` for variable-target-count gates like `MPP`. This is *not*
    /// because `MPP` has no stabilizer flows, but because its flows depend on how many
    /// qubits it targets and in which bases.
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

    /// Returns the Clifford tableau of the gate, or `None` if the gate is not unitary.
    ///
    /// The Clifford tableau describes how a unitary gate conjugates each single-qubit
    /// Pauli operator (X and Z on each qubit) into a new Pauli string. This
    /// representation fully specifies any Clifford gate up to global phase.
    ///
    /// Non-unitary gates — measurements, resets, noise channels, and annotations —
    /// do not have tableaux, so this method returns `None` for them.
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

    /// Returns the unitary matrix representation of the gate, or `None` if the gate is
    /// not unitary.
    ///
    /// The matrix is computed from the gate's Clifford tableau and returned as an
    /// [`ndarray::Array2<Complex32>`]. The matrix uses big-endian qubit ordering (the
    /// first qubit is the most-significant bit of the row/column index).
    ///
    /// Non-unitary gates — measurements, resets, noise channels, and annotations —
    /// do not have unitary matrices, so this method returns `None` for them.
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

    /// Returns the inverse of the gate, or `None` if the gate has no inverse.
    ///
    /// The inverse `V` of a gate `U` satisfies the property that applying `U` followed
    /// by `V` (or vice versa) is equivalent to doing nothing. In circuit terms:
    ///
    /// ```text
    /// U 0 1
    /// V 0 1
    /// ```
    ///
    /// is a no-op.
    ///
    /// Only unitary gates have inverses. Noise channels (`X_ERROR`, `DEPOLARIZE1`, …),
    /// measurements (`M`, `MXX`, …), resets (`R`, …), and annotations (`DETECTOR`,
    /// `TICK`, …) return `None`.
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

    /// Returns the closest-thing-to-an-inverse for the gate, choosing *something* even
    /// when no true inverse exists.
    ///
    /// The generalized inverse applies different rules depending on gate category:
    ///
    /// - **Unitary gates**: the generalized inverse is the actual inverse `U⁻¹`.
    /// - **Reset / measurement gates**: the generalized inverse is a gate whose
    ///   stabilizer flows are the time-reverses of the original gate's flows (up to
    ///   Pauli feedback, potentially with additional flows). For example, `R` has the
    ///   flow `1 -> Z`, and its generalized inverse `M` has the time-reversed flow
    ///   `Z -> rec[-1]`.
    /// - **Noise channels** (e.g. `X_ERROR`): the generalized inverse is the same
    ///   noise channel.
    /// - **Annotations** (e.g. `TICK`, `DETECTOR`): the generalized inverse is the
    ///   same annotation.
    ///
    /// Unlike [`inverse`](GateData::inverse), this method always returns a value.
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

    /// Returns the Hadamard-conjugated form of the gate, or `None` if Stim does not
    /// define one.
    ///
    /// The Hadamard conjugate can be thought of as the XZ dual of the gate: the gate
    /// you get by exchanging the X and Z bases on every qubit. Concretely, it is the
    /// gate `H⊗ⁿ · U · H⊗ⁿ` where `n` is the number of qubits the gate acts on.
    ///
    /// For example, the Hadamard conjugate of `X` is `Z`, of `SQRT_X` is `SQRT_Z`,
    /// and of `CX` is `XCZ` (because swapping X↔Z flips which qubit is the control).
    ///
    /// When `unsigned_only` is `false`, the returned gate must be *exactly* the
    /// Hadamard conjugate. When `unsigned_only` is `true`, the returned gate only
    /// needs to match up to the signs of its stabilizer flows (i.e. it may differ by
    /// Pauli gates). This relaxation allows gates like `RY` — whose exact conjugate
    /// introduces a sign change that does not correspond to any named Stim gate — to
    /// return `Some` with the unsigned match.
    ///
    /// Returns `None` if Stim does not define a gate equal to the (possibly unsigned)
    /// Hadamard conjugate.
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
/// This is the primary entry point for inspecting Stim's built-in gates. The lookup
/// is case-insensitive: `"h"`, `"H"`, and `"cnot"` all resolve successfully. Aliases
/// are resolved to their canonical gate — for example, `"CNOT"` resolves to the
/// canonical `"CX"` gate.
///
/// To enumerate *all* canonical gates at once, see [`all_gate_data`].
///
/// # Errors
///
/// Returns an error if `name` does not match any known gate or alias.
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

/// Returns the full canonical gate inventory, keyed by canonical gate name.
///
/// The returned map contains one entry per canonical Stim gate. Aliases are *not*
/// included as separate keys — for example, the map contains `"CX"` but not `"CNOT"`.
/// To look up a gate by alias, use [`gate_data`] instead.
///
/// This is useful for enumerating all gates that Stim supports, for example to build
/// a gate reference or to iterate over gate properties programmatically.
///
/// # Examples
///
/// ```
/// let gates = stim::all_gate_data();
/// assert!(gates.contains_key("H"));
/// assert!(gates.contains_key("CX"));
///
/// // Aliases are not included as separate keys.
/// assert!(!gates.contains_key("CNOT"));
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
