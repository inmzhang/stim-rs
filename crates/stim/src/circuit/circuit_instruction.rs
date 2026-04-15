use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::{Circuit, GateTarget, Result, StimError, gate_data};

/// A single instruction from a stabilizer circuit, such as `H 0 1` or
/// `CNOT rec[-1] 5`.
///
/// Each instruction consists of a gate name, zero or more numeric gate
/// arguments (e.g. the error probability for `DEPOLARIZE1(0.01)`), an ordered
/// list of [`GateTarget`]s the gate acts on, and an optional string tag for
/// annotation.
///
/// Gate names are canonicalized on construction, so aliases like `"cnot"` are
/// stored as `"CX"`.
///
/// # Examples
///
/// ```
/// use stim::CircuitInstruction;
///
/// let h = CircuitInstruction::new(
///     "H", [0u32, 1u32], std::iter::empty::<f64>(), "",
/// ).expect("H is a valid gate");
/// assert_eq!(h.name(), "H");
/// assert_eq!(h.num_measurements(), 0);
///
/// // Aliases are canonicalized automatically.
/// let cx = CircuitInstruction::new(
///     "cnot", [0u32, 1u32], std::iter::empty::<f64>(), "",
/// ).expect("cnot is an alias for CX");
/// assert_eq!(cx.name(), "CX");
/// ```
#[derive(Clone, PartialEq)]
pub struct CircuitInstruction {
    name: String,
    tag: String,
    gate_args: Vec<f64>,
    targets: Vec<GateTarget>,
    num_measurements: u64,
}

impl CircuitInstruction {
    /// Creates a new circuit instruction with the given gate name, targets,
    /// gate arguments, and optional tag.
    ///
    /// The gate name is canonicalized so that aliases resolve to the standard
    /// name used by Stim (e.g. `"cnot"` becomes `"CX"`). The number of
    /// measurements produced by this instruction is computed eagerly and
    /// cached.
    ///
    /// # Errors
    ///
    /// Returns an error if the combination of gate name, targets, and
    /// arguments is not valid according to Stim's gate definitions.
    ///
    /// # Examples
    ///
    /// ```
    /// let inst = stim::CircuitInstruction::new(
    ///     "X_ERROR", [5u32, 7u32], [0.125], "",
    /// ).expect("valid instruction");
    /// assert_eq!(inst.to_string(), "X_ERROR(0.125) 5 7");
    /// ```
    pub fn new(
        name: impl Into<String>,
        targets: impl IntoIterator<Item = impl Into<GateTarget>>,
        gate_args: impl IntoIterator<Item = f64>,
        tag: impl Into<String>,
    ) -> Result<Self> {
        let raw_name = name.into();
        let instruction = Self {
            name: canonical_gate_name(&raw_name),
            tag: tag.into(),
            gate_args: gate_args.into_iter().collect(),
            targets: targets.into_iter().map(Into::into).collect(),
            num_measurements: 0,
        };
        let num_measurements = instruction.compute_num_measurements()?;
        Ok(Self {
            num_measurements,
            ..instruction
        })
    }

    /// Parses a single instruction from Stim program text.
    ///
    /// The text must contain exactly one instruction — not a `REPEAT` block
    /// and not multiple lines of operations.
    ///
    /// This is also available via the [`FromStr`] implementation.
    ///
    /// # Errors
    ///
    /// Returns an error if the text contains multiple operations, a `REPEAT`
    /// block, or is otherwise not a valid single instruction.
    ///
    /// # Examples
    ///
    /// ```
    /// use stim::CircuitInstruction;
    ///
    /// let inst = CircuitInstruction::from_stim_program_text("H 0 1")
    ///     .expect("valid instruction text");
    /// assert_eq!(inst.name(), "H");
    /// ```
    pub fn from_stim_program_text(text: &str) -> Result<Self> {
        let circuit = Circuit::from_str(text)?;
        let normalized = circuit.to_string();
        if normalized.contains('\n') {
            return Err(StimError::new(
                "expected a single circuit instruction, got multiple operations",
            ));
        }
        if normalized.starts_with("REPEAT") {
            return Err(StimError::new(
                "CircuitInstruction cannot represent REPEAT blocks",
            ));
        }

        let (head, tail) = normalized
            .split_once(' ')
            .map_or((normalized.as_str(), ""), |(head, tail)| (head, tail));
        let (name, tag, gate_args) = parse_head(head)?;
        let targets = parse_targets(tail)?;
        Self::new(name, targets, gate_args, tag)
    }

    /// Returns the canonical gate name (e.g. `"CX"`, `"H"`, `"M"`).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the instruction's tag string, or `""` if no tag was set.
    ///
    /// Tags are the bracketed annotations in Stim syntax, e.g. the `100ns`
    /// in `I[100ns] 2`.
    #[must_use]
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Returns a copy of the gate's numeric arguments.
    ///
    /// For example, `DEPOLARIZE1(0.01)` has a single gate argument `[0.01]`,
    /// while `H 0` has an empty argument list.
    #[must_use]
    pub fn gate_args_copy(&self) -> Vec<f64> {
        self.gate_args.clone()
    }

    /// Returns a copy of all targets the instruction acts on.
    #[must_use]
    pub fn targets_copy(&self) -> Vec<GateTarget> {
        self.targets.clone()
    }

    /// Returns the number of measurement results this instruction produces.
    ///
    /// Non-measuring gates like `H` return `0`, while `M 2 3 5 7 11` returns
    /// `5`. Two-qubit measurements like `MXX` count each pair as one
    /// measurement.
    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        self.num_measurements
    }

    /// Groups the instruction's targets into logical operation groups.
    ///
    /// The grouping strategy depends on the gate type:
    ///
    /// - **Single-qubit gates and measurements**: each target becomes its own
    ///   group of size 1.
    /// - **Two-qubit gates**: targets are paired into groups of size 2.
    /// - **Gates with combiners** (e.g. `MPP X0*Y1`): targets are split on
    ///   combiner boundaries.
    /// - **Other gates**: all targets form a single group.
    ///
    /// Returns an empty `Vec` if the instruction has no targets.
    #[must_use]
    pub fn target_groups(&self) -> Vec<Vec<GateTarget>> {
        if self.targets.is_empty() {
            return Vec::new();
        }
        if self.targets.iter().any(|target| target.is_combiner()) {
            return split_on_combiners(&self.targets);
        }

        match gate_data(&self.name) {
            Ok(gate) if gate.is_two_qubit_gate() => self
                .targets
                .chunks_exact(2)
                .map(|chunk| chunk.to_vec())
                .collect(),
            Ok(gate)
                if gate.is_single_qubit_gate()
                    || gate.produces_measurements()
                    || gate.takes_measurement_record_targets() =>
            {
                self.targets
                    .iter()
                    .copied()
                    .map(|target| vec![target])
                    .collect()
            }
            _ => vec![self.targets.clone()],
        }
    }

    fn compute_num_measurements(&self) -> Result<u64> {
        let mut circuit = Circuit::new();
        circuit.append_instruction(self)?;
        Ok(circuit.num_measurements())
    }

    fn fmt_repr(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "stim::CircuitInstruction({:?}, [", self.name)?;
        for (index, target) in self.targets.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{target:?}")?;
        }
        write!(f, "], {:?}", self.gate_args)?;
        if !self.tag.is_empty() {
            write!(f, ", tag={:?}", self.tag)?;
        }
        f.write_str(")")
    }
}

impl Eq for CircuitInstruction {}

impl PartialOrd for CircuitInstruction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CircuitInstruction {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.name
            .cmp(&other.name)
            .then_with(|| self.tag.cmp(&other.tag))
            .then_with(|| compare_f64_slices(&self.gate_args, &other.gate_args))
            .then_with(|| self.targets.cmp(&other.targets))
    }
}

impl std::hash::Hash for CircuitInstruction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.tag.hash(state);
        for arg in &self.gate_args {
            arg.to_bits().hash(state);
        }
        self.targets.hash(state);
    }
}

impl Display for CircuitInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)?;
        if !self.tag.is_empty() {
            write!(f, "[{}]", self.tag)?;
        }
        if !self.gate_args.is_empty() {
            f.write_str("(")?;
            for (index, arg) in self.gate_args.iter().enumerate() {
                if index > 0 {
                    f.write_str(",")?;
                }
                write!(f, "{arg}")?;
            }
            f.write_str(")")?;
        }
        for target in &self.targets {
            if target.is_combiner() {
                write!(f, "{target}")?;
            } else {
                write!(f, " {target}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for CircuitInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_repr(f)
    }
}

impl Circuit {
    /// Appends a [`CircuitInstruction`] to the end of this circuit.
    ///
    /// The instruction is serialized to Stim program text and re-parsed by
    /// the underlying C++ engine, so the result is always consistent with
    /// Stim's own representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the instruction's name is empty or if the
    /// serialized text is rejected by the Stim parser.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit = stim::Circuit::new();
    /// let h = stim::CircuitInstruction::new(
    ///     "H", [0u32], std::iter::empty::<f64>(), "",
    /// ).expect("valid instruction");
    /// circuit.append_instruction(&h).expect("append succeeds");
    /// assert_eq!(circuit.to_string(), "H 0");
    /// ```
    pub fn append_instruction(&mut self, instruction: &CircuitInstruction) -> Result<()> {
        if instruction.name().trim().is_empty() {
            return Err(StimError::new("instruction name must not be empty"));
        }
        self.append_from_stim_program_text(&instruction.to_string())
    }
}

fn split_on_combiners(targets: &[GateTarget]) -> Vec<Vec<GateTarget>> {
    let mut groups = Vec::new();
    let mut current = Vec::new();
    for (index, target) in targets.iter().enumerate() {
        if target.is_combiner() {
            continue;
        }
        current.push(*target);
        let at_group_end = targets
            .get(index + 1)
            .is_none_or(|next| !next.is_combiner());
        if at_group_end {
            groups.push(std::mem::take(&mut current));
        }
    }
    groups
}

fn compare_f64_slices(left: &[f64], right: &[f64]) -> std::cmp::Ordering {
    for (l, r) in left.iter().zip(right.iter()) {
        let cmp = l.to_bits().cmp(&r.to_bits());
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
    }
    left.len().cmp(&right.len())
}

impl FromStr for CircuitInstruction {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_stim_program_text(s)
    }
}

fn canonical_gate_name(name: &str) -> String {
    gate_data(name)
        .map(|gate| gate.name())
        .unwrap_or_else(|_| name.to_string())
}

fn parse_head(head: &str) -> Result<(String, String, Vec<f64>)> {
    let name_end = head.find(['[', '(']).unwrap_or(head.len());
    let name = &head[..name_end];
    let mut rest = &head[name_end..];
    let mut tag = String::new();
    let mut gate_args = Vec::new();

    if let Some(after_open) = rest.strip_prefix('[') {
        let close = after_open
            .find(']')
            .ok_or_else(|| StimError::new("unterminated instruction tag"))?;
        tag = after_open[..close].to_string();
        rest = &after_open[close + 1..];
    }

    if let Some(after_open) = rest.strip_prefix('(') {
        let close = after_open
            .find(')')
            .ok_or_else(|| StimError::new("unterminated instruction argument list"))?;
        let raw_args = &after_open[..close];
        if !raw_args.is_empty() {
            gate_args = raw_args
                .split(',')
                .map(|arg| {
                    arg.parse::<f64>()
                        .map_err(|_| StimError::new(format!("invalid gate arg: {arg}")))
                })
                .collect::<Result<Vec<_>>>()?;
        }
    }

    Ok((name.to_string(), tag, gate_args))
}

fn parse_targets(targets_text: &str) -> Result<Vec<GateTarget>> {
    let mut targets = Vec::new();
    for group in targets_text.split_whitespace() {
        let mut parts = group.split('*').peekable();
        while let Some(part) = parts.next() {
            if !part.is_empty() {
                targets.push(GateTarget::from_target_str(part)?);
            }
            if parts.peek().is_some() {
                targets.push(GateTarget::combiner());
            }
        }
    }
    Ok(targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};
    use std::str::FromStr;

    use crate::target_combiner;
    use crate::target_rec;
    use crate::target_x;
    use crate::target_y;

    #[test]
    fn constructor_and_accessors_preserve_values() {
        let instruction = CircuitInstruction::new("X_ERROR", [5u32, 7u32], [0.125], "").unwrap();

        assert_eq!(instruction.name(), "X_ERROR");
        assert_eq!(instruction.tag(), "");
        assert_eq!(instruction.gate_args_copy(), vec![0.125]);
        assert_eq!(
            instruction.targets_copy(),
            vec![GateTarget::new(5u32), GateTarget::new(7u32)]
        );
        assert_eq!(instruction.num_measurements(), 0);
        assert_eq!(instruction.to_string(), "X_ERROR(0.125) 5 7");
    }

    #[test]
    fn constructor_canonicalizes_aliases_and_parser_roundtrips_lines() {
        let aliased = CircuitInstruction::new(
            "cnot",
            [target_rec(-1).unwrap(), GateTarget::new(5u32)],
            std::iter::empty::<f64>(),
            "annotated",
        )
        .unwrap();
        let parsed = CircuitInstruction::from_str("CX[annotated] rec[-1] 5").unwrap();
        let mpp = CircuitInstruction::from_str("MPP X0*Y1 X5*X6").unwrap();

        assert_eq!(aliased.name(), "CX");
        assert_eq!(aliased, parsed);
        assert_eq!(
            mpp.targets_copy(),
            vec![
                target_x(0u32, false).unwrap(),
                target_combiner(),
                target_y(1u32, false).unwrap(),
                target_x(5u32, false).unwrap(),
                target_combiner(),
                target_x(6u32, false).unwrap(),
            ]
        );
    }

    #[test]
    fn display_debug_and_tagged_append_match_binding_conventions() {
        let instruction =
            CircuitInstruction::new("I", [2u32], std::iter::empty::<f64>(), "100ns").unwrap();

        assert_eq!(instruction.to_string(), "I[100ns] 2");
        assert_eq!(
            format!("{instruction:?}"),
            "stim::CircuitInstruction(\"I\", [stim::GateTarget(2)], [], tag=\"100ns\")"
        );

        let mut circuit = Circuit::new();
        circuit.append_instruction(&instruction).unwrap();
        assert_eq!(circuit.to_string(), "I[100ns] 2");
    }

    #[test]
    fn equality_hash_and_order_follow_name_tag_args_then_targets() {
        let first =
            CircuitInstruction::new("X_ERROR", [5u32], [0.125], "").expect("instruction builds");
        let same =
            CircuitInstruction::new("X_ERROR", [5u32], [0.125], "").expect("instruction builds");
        let different_tag =
            CircuitInstruction::new("X_ERROR", [5u32], [0.125], "tag").expect("instruction builds");
        let different_name =
            CircuitInstruction::new("Y_ERROR", [5u32], [0.125], "").expect("instruction builds");

        assert_eq!(first, same);
        assert_ne!(first, different_tag);
        assert_ne!(first, different_name);

        let ordered = [
            different_name.clone(),
            different_tag.clone(),
            same.clone(),
            first.clone(),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
        assert_eq!(ordered, vec![first.clone(), different_tag, different_name]);

        let hashed = [first.clone(), same, first.clone()]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 1);
        assert!(hashed.contains(&first));
    }

    #[test]
    fn target_groups_follow_documented_shapes() {
        let single =
            CircuitInstruction::new("H", [0u32, 1u32, 2u32], std::iter::empty::<f64>(), "")
                .unwrap();
        let two_qubit = CircuitInstruction::new(
            "CX",
            [0u32, 1u32, 2u32, 3u32],
            std::iter::empty::<f64>(),
            "",
        )
        .unwrap();
        let mpp = CircuitInstruction::new(
            "MPP",
            [
                target_x(0u32, false).unwrap(),
                target_combiner(),
                target_y(1u32, false).unwrap(),
                target_x(5u32, false).unwrap(),
                target_combiner(),
                target_x(6u32, false).unwrap(),
            ],
            std::iter::empty::<f64>(),
            "",
        )
        .unwrap();
        let detector = CircuitInstruction::new(
            "DETECTOR",
            [target_rec(-1).unwrap(), target_rec(-2).unwrap()],
            [2.0, 3.0],
            "",
        )
        .unwrap();
        let correlated = CircuitInstruction::new(
            "CORRELATED_ERROR",
            [
                target_x(0u32, false).unwrap(),
                target_y(1u32, false).unwrap(),
            ],
            [0.1],
            "",
        )
        .unwrap();

        assert_eq!(
            single.target_groups(),
            vec![
                vec![GateTarget::new(0u32)],
                vec![GateTarget::new(1u32)],
                vec![GateTarget::new(2u32)],
            ]
        );
        assert_eq!(
            two_qubit.target_groups(),
            vec![
                vec![GateTarget::new(0u32), GateTarget::new(1u32)],
                vec![GateTarget::new(2u32), GateTarget::new(3u32)],
            ]
        );
        assert_eq!(
            mpp.target_groups(),
            vec![
                vec![
                    target_x(0u32, false).unwrap(),
                    target_y(1u32, false).unwrap()
                ],
                vec![
                    target_x(5u32, false).unwrap(),
                    target_x(6u32, false).unwrap()
                ],
            ]
        );
        assert_eq!(
            detector.target_groups(),
            vec![vec![target_rec(-1).unwrap()], vec![target_rec(-2).unwrap()]]
        );
        assert_eq!(
            correlated.target_groups(),
            vec![vec![
                target_x(0u32, false).unwrap(),
                target_y(1u32, false).unwrap(),
            ]]
        );
    }

    #[test]
    fn num_measurements_matches_instruction_semantics() {
        assert_eq!(
            CircuitInstruction::new("H", [0u32], std::iter::empty::<f64>(), "")
                .unwrap()
                .num_measurements(),
            0
        );
        assert_eq!(
            CircuitInstruction::new("M", [0u32], std::iter::empty::<f64>(), "")
                .unwrap()
                .num_measurements(),
            1
        );
        assert_eq!(
            CircuitInstruction::new(
                "M",
                [2u32, 3u32, 5u32, 7u32, 11u32],
                std::iter::empty::<f64>(),
                "",
            )
            .unwrap()
            .num_measurements(),
            5
        );
        assert_eq!(
            CircuitInstruction::new(
                "MXX",
                [0u32, 1u32, 4u32, 5u32, 11u32, 13u32],
                std::iter::empty::<f64>(),
                "",
            )
            .unwrap()
            .num_measurements(),
            3
        );
        assert_eq!(
            CircuitInstruction::new(
                "MPP",
                [
                    target_x(0u32, false).unwrap(),
                    target_combiner(),
                    target_x(1u32, false).unwrap(),
                    target_x(0u32, false).unwrap(),
                    target_combiner(),
                    target_y(1u32, false).unwrap(),
                    target_combiner(),
                    target_y(2u32, false).unwrap(),
                ],
                std::iter::empty::<f64>(),
                "",
            )
            .unwrap()
            .num_measurements(),
            2
        );
        assert_eq!(
            CircuitInstruction::new("HERALDED_ERASE", [0u32], [0.25], "")
                .unwrap()
                .num_measurements(),
            1
        );
    }
}
