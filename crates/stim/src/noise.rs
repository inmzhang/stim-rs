use std::collections::{BTreeMap, BTreeSet};

use crate::{Circuit, CircuitInstruction, GateTarget, Result, StimError, gate_data};

/// A single noise-channel operation to insert before or after a circuit instruction.
#[derive(Clone, Debug, PartialEq)]
struct NoiseOperation {
    gate_name: String,
    args: Vec<f64>,
}

impl NoiseOperation {
    fn new(gate_name: &str, args: &[f64]) -> Result<Self> {
        let gate = gate_data(gate_name)?;
        if gate.produces_measurements() || !gate.is_noisy_gate() {
            return Err(StimError::new(format!(
                "'{}' is not a pure noise channel",
                gate.name()
            )));
        }
        validate_gate_args(&gate.name(), args, &gate.num_parens_arguments_range())?;
        Ok(Self {
            gate_name: gate.name(),
            args: args.to_vec(),
        })
    }

    #[must_use]
    pub fn gate_name(&self) -> &str {
        &self.gate_name
    }

    #[must_use]
    pub fn args(&self) -> &[f64] {
        &self.args
    }
}

/// Describes noise to insert around a matching circuit instruction.
#[derive(Clone, Debug, Default, PartialEq)]
struct NoiseRule {
    before: Vec<NoiseOperation>,
    after: Vec<NoiseOperation>,
    flip_result: f64,
}

impl NoiseRule {
    #[must_use]
    fn new() -> Self {
        Self::default()
    }

    fn with_after(mut self, gate_name: &str, args: &[f64]) -> Result<Self> {
        self.after.push(NoiseOperation::new(gate_name, args)?);
        Ok(self)
    }

    fn with_flip_result(mut self, probability: f64) -> Result<Self> {
        validate_probability(probability, "flip_result")?;
        self.flip_result = probability;
        Ok(self)
    }

    #[must_use]
    pub fn before(&self) -> &[NoiseOperation] {
        &self.before
    }

    #[must_use]
    pub fn after(&self) -> &[NoiseOperation] {
        &self.after
    }

    #[must_use]
    pub fn flip_result(&self) -> f64 {
        self.flip_result
    }
}

/// Applies a noise model to a circuit and returns a noisy circuit.
pub trait NoiseModel {
    /// Produces a noisy circuit derived from `circuit` according to the model.
    fn noisy_circuit(&self, circuit: &Circuit) -> Result<Circuit>;
}

#[derive(Clone, Copy, Debug)]
struct NoiseStrength {
    single_qubit_depolarization: f64,
    two_qubit_depolarization: f64,
    measurement_depolarization: f64,
    measurement_flip: f64,
    reset_z_bias_flip: f64,
    reset_x_bias_flip: f64,
    reset_y_bias_flip: f64,
    idle_depolarization: f64,
    additional_depolarization_waiting_for_measure_or_reset: f64,
}

impl NoiseStrength {
    fn uniform_depolarizing(probability: f64) -> Result<Self> {
        validate_probability(probability, "uniform_depolarizing probability")?;
        Ok(Self {
            single_qubit_depolarization: probability,
            two_qubit_depolarization: probability,
            measurement_depolarization: probability,
            measurement_flip: probability,
            reset_z_bias_flip: probability,
            reset_x_bias_flip: probability,
            reset_y_bias_flip: probability,
            idle_depolarization: probability,
            additional_depolarization_waiting_for_measure_or_reset: 0.0,
        })
    }

    fn si1000(probability: f64) -> Result<Self> {
        validate_probability(probability, "si1000 probability")?;
        Ok(Self {
            single_qubit_depolarization: probability / 10.0,
            two_qubit_depolarization: probability,
            measurement_depolarization: probability,
            measurement_flip: probability * 5.0,
            reset_z_bias_flip: probability * 2.0,
            reset_x_bias_flip: 0.0,
            reset_y_bias_flip: 0.0,
            idle_depolarization: probability / 10.0,
            additional_depolarization_waiting_for_measure_or_reset: probability * 2.0,
        })
    }
}

/// Internal pure-Rust circuit noise model layered on top of `stim::Circuit`.
#[derive(Clone, Debug, Default)]
struct NoiseModelConfig {
    idle_depolarization: f64,
    additional_depolarization_waiting_for_measure_or_reset: f64,
    gate_rules: BTreeMap<String, NoiseRule>,
    measure_rules: BTreeMap<String, NoiseRule>,
    any_measurement_rule: Option<NoiseRule>,
    any_clifford_1q_rule: Option<NoiseRule>,
    any_clifford_2q_rule: Option<NoiseRule>,
    allow_multiple_uses_of_a_qubit_in_one_tick: bool,
}

impl NoiseModelConfig {
    #[must_use]
    fn new() -> Self {
        Self::default()
    }

    fn with_idle_depolarization(mut self, probability: f64) -> Result<Self> {
        validate_probability(probability, "idle_depolarization")?;
        self.idle_depolarization = probability;
        Ok(self)
    }

    fn with_additional_depolarization_waiting_for_measure_or_reset(
        mut self,
        probability: f64,
    ) -> Result<Self> {
        validate_probability(
            probability,
            "additional_depolarization_waiting_for_measure_or_reset",
        )?;
        self.additional_depolarization_waiting_for_measure_or_reset = probability;
        Ok(self)
    }

    fn with_gate_rule(mut self, gate_name: &str, rule: NoiseRule) -> Result<Self> {
        let canonical = gate_data(gate_name)?.name();
        self.gate_rules.insert(canonical, rule);
        Ok(self)
    }

    #[must_use]
    fn with_measure_rule(mut self, basis: &str, rule: NoiseRule) -> Self {
        self.measure_rules.insert(basis.to_string(), rule);
        self
    }

    #[must_use]
    #[cfg(test)]
    fn with_any_measurement_rule(mut self, rule: NoiseRule) -> Self {
        self.any_measurement_rule = Some(rule);
        self
    }

    #[must_use]
    fn with_any_clifford_1q_rule(mut self, rule: NoiseRule) -> Self {
        self.any_clifford_1q_rule = Some(rule);
        self
    }

    #[must_use]
    fn with_any_clifford_2q_rule(mut self, rule: NoiseRule) -> Self {
        self.any_clifford_2q_rule = Some(rule);
        self
    }

    fn from_strength(strength: NoiseStrength) -> Result<Self> {
        let two_qubit_channel = "DEPOLARIZE2";
        let measurement_rule = NoiseRule::new()
            .with_after("DEPOLARIZE1", &[strength.measurement_depolarization])?
            .with_flip_result(strength.measurement_flip)?;
        let two_qubit_measurement_rule = NoiseRule::new()
            .with_after(two_qubit_channel, &[strength.measurement_depolarization])?
            .with_flip_result(strength.measurement_flip)?;

        let mut config = Self::new()
            .with_idle_depolarization(strength.idle_depolarization)?
            .with_additional_depolarization_waiting_for_measure_or_reset(
                strength.additional_depolarization_waiting_for_measure_or_reset,
            )?
            .with_any_clifford_1q_rule(
                NoiseRule::new()
                    .with_after("DEPOLARIZE1", &[strength.single_qubit_depolarization])?,
            )
            .with_any_clifford_2q_rule(
                NoiseRule::new()
                    .with_after(two_qubit_channel, &[strength.two_qubit_depolarization])?,
            )
            .with_measure_rule("X", measurement_rule.clone())
            .with_measure_rule("Y", measurement_rule.clone())
            .with_measure_rule("Z", measurement_rule)
            .with_measure_rule("XX", two_qubit_measurement_rule.clone())
            .with_measure_rule("YY", two_qubit_measurement_rule.clone())
            .with_measure_rule("ZZ", two_qubit_measurement_rule);

        if strength.reset_z_bias_flip > 0.0 {
            config = config
                .with_gate_rule(
                    "R",
                    NoiseRule::new().with_after("X_ERROR", &[strength.reset_z_bias_flip])?,
                )?
                .with_gate_rule(
                    "RZ",
                    NoiseRule::new().with_after("X_ERROR", &[strength.reset_z_bias_flip])?,
                )?;
        }

        if strength.reset_x_bias_flip > 0.0 {
            config = config.with_gate_rule(
                "RX",
                NoiseRule::new().with_after("Z_ERROR", &[strength.reset_x_bias_flip])?,
            )?;
        }

        if strength.reset_y_bias_flip > 0.0 {
            config = config.with_gate_rule(
                "RY",
                NoiseRule::new().with_after("X_ERROR", &[strength.reset_y_bias_flip])?,
            )?;
        }

        Ok(config)
    }
}

impl NoiseModel for NoiseModelConfig {
    fn noisy_circuit(&self, circuit: &Circuit) -> Result<Circuit> {
        let mut output = Circuit::new();
        let flattened = circuit.flattened();
        let flattened_text = flattened.to_string();
        let mut pending_moment = PendingMoment::default();
        let mut active_qubits = BTreeSet::new();
        let mut used_qubits = BTreeSet::new();
        let mut collapsed_qubits = BTreeSet::new();
        let mut unitary_1q_qubits = BTreeSet::new();
        let mut non_unitary_or_multi_use_qubits = BTreeSet::new();
        let mut generic_1q_noise_applied = BTreeSet::new();

        for line in flattened_text.lines() {
            if line.is_empty() {
                continue;
            }

            if line == "TICK" {
                flush_pending_moment(&mut output, &mut pending_moment)?;
                self.append_tick_noise(
                    &mut output,
                    &active_qubits,
                    &used_qubits,
                    &collapsed_qubits,
                )?;
                output.append_from_stim_program_text("TICK")?;
                used_qubits.clear();
                collapsed_qubits.clear();
                unitary_1q_qubits.clear();
                non_unitary_or_multi_use_qubits.clear();
                generic_1q_noise_applied.clear();
                continue;
            }

            if annotation_line_name(line).is_some() {
                pending_moment.body.append_from_stim_program_text(line)?;
                continue;
            }

            let instruction = CircuitInstruction::from_stim_program_text(line)?;

            for split_instruction in split_instruction_for_noise(&instruction)? {
                let gate = gate_data(split_instruction.name())?;
                let qubit_targets = qubit_targets(&split_instruction);

                self.record_qubit_usage(
                    &split_instruction,
                    &gate,
                    &qubit_targets,
                    &mut unitary_1q_qubits,
                    &mut non_unitary_or_multi_use_qubits,
                )?;
                used_qubits.extend(qubit_targets.iter().copied());

                let resolved_rule = self.resolve_noise_rule(&split_instruction, &gate)?;
                self.append_instruction_with_noise(
                    &mut pending_moment,
                    &split_instruction,
                    &qubit_targets,
                    resolved_rule.as_ref(),
                    &mut generic_1q_noise_applied,
                )?;

                if gate.is_reset() {
                    active_qubits.extend(qubit_targets.iter().copied());
                } else if gate.produces_measurements() && !gate.is_reset() {
                    for qubit in qubit_targets {
                        active_qubits.remove(&qubit);
                        collapsed_qubits.insert(qubit);
                    }
                } else if !is_annotation(split_instruction.name()) {
                    active_qubits.extend(qubit_targets.iter().copied());
                }
            }
        }

        flush_pending_moment(&mut output, &mut pending_moment)?;
        Ok(output)
    }
}

/// A ready-made near-uniform depolarizing noise model.
#[derive(Clone, Debug)]
pub struct UniformDepolarizing {
    strength: NoiseStrength,
}

impl UniformDepolarizing {
    /// Creates the standard near-uniform depolarizing noise model.
    pub fn new(probability: f64) -> Result<Self> {
        Ok(Self {
            strength: NoiseStrength::uniform_depolarizing(probability)?,
        })
    }
}

impl NoiseModel for UniformDepolarizing {
    fn noisy_circuit(&self, circuit: &Circuit) -> Result<Circuit> {
        NoiseModelConfig::from_strength(self.strength)?.noisy_circuit(circuit)
    }
}

/// A ready-made superconducting-inspired SI1000 noise model.
#[derive(Clone, Debug)]
pub struct Si1000 {
    strength: NoiseStrength,
}

impl Si1000 {
    /// Creates the superconducting-inspired SI1000 noise model.
    pub fn new(probability: f64) -> Result<Self> {
        Ok(Self {
            strength: NoiseStrength::si1000(probability)?,
        })
    }
}

impl NoiseModel for Si1000 {
    fn noisy_circuit(&self, circuit: &Circuit) -> Result<Circuit> {
        NoiseModelConfig::from_strength(self.strength)?.noisy_circuit(circuit)
    }
}

#[derive(Default)]
struct PendingMoment {
    before: BTreeMap<(String, Vec<u64>), Vec<u32>>,
    body: Circuit,
    after: BTreeMap<(String, Vec<u64>), Vec<u32>>,
}

#[derive(Clone, Debug)]
struct ResolvedNoiseRule {
    rule: NoiseRule,
    source: RuleSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuleSource {
    ExplicitGate,
    ExplicitMeasure,
    AnyMeasurement,
    AnyClifford1Q,
    AnyClifford2Q,
    CombinedMeasureReset,
}

impl NoiseModelConfig {
    fn append_tick_noise(
        &self,
        output: &mut Circuit,
        active_qubits: &BTreeSet<u32>,
        used_qubits: &BTreeSet<u32>,
        collapsed_qubits: &BTreeSet<u32>,
    ) -> Result<()> {
        if self.idle_depolarization > 0.0 {
            let idle_targets: Vec<u32> = active_qubits
                .iter()
                .filter(|qubit| !used_qubits.contains(qubit))
                .copied()
                .collect();
            if !idle_targets.is_empty() {
                output.append("DEPOLARIZE1", &idle_targets, &[self.idle_depolarization])?;
            }
        }

        if self.additional_depolarization_waiting_for_measure_or_reset > 0.0
            && !collapsed_qubits.is_empty()
        {
            let waiting_targets: Vec<u32> = active_qubits
                .iter()
                .filter(|qubit| !collapsed_qubits.contains(qubit))
                .copied()
                .collect();
            if !waiting_targets.is_empty() {
                output.append(
                    "DEPOLARIZE1",
                    &waiting_targets,
                    &[self.additional_depolarization_waiting_for_measure_or_reset],
                )?;
            }
        }

        Ok(())
    }

    fn record_qubit_usage(
        &self,
        instruction: &CircuitInstruction,
        gate: &crate::GateData,
        qubit_targets: &[u32],
        unitary_1q_qubits: &mut BTreeSet<u32>,
        non_unitary_or_multi_use_qubits: &mut BTreeSet<u32>,
    ) -> Result<()> {
        if self.allow_multiple_uses_of_a_qubit_in_one_tick
            || qubit_targets.is_empty()
            || is_annotation(instruction.name())
            || occurs_in_classical_control_system(instruction)?
        {
            return Ok(());
        }

        let is_single_qubit_unitary = gate.is_unitary() && gate.is_single_qubit_gate();
        if is_single_qubit_unitary {
            if qubit_targets
                .iter()
                .any(|qubit| non_unitary_or_multi_use_qubits.contains(qubit))
            {
                return Err(StimError::new(format!(
                    "qubit reused in the same TICK without enabling multiple uses: {}",
                    instruction
                )));
            }
            unitary_1q_qubits.extend(qubit_targets.iter().copied());
            return Ok(());
        }

        if qubit_targets.iter().any(|qubit| {
            unitary_1q_qubits.contains(qubit) || non_unitary_or_multi_use_qubits.contains(qubit)
        }) {
            return Err(StimError::new(format!(
                "qubit reused in the same TICK without enabling multiple uses: {}",
                instruction
            )));
        }
        non_unitary_or_multi_use_qubits.extend(qubit_targets.iter().copied());
        Ok(())
    }

    fn resolve_noise_rule(
        &self,
        instruction: &CircuitInstruction,
        gate: &crate::GateData,
    ) -> Result<Option<ResolvedNoiseRule>> {
        if is_annotation(instruction.name()) || occurs_in_classical_control_system(instruction)? {
            return Ok(None);
        }

        if let Some(rule) = self.gate_rules.get(gate.name().as_str()) {
            return Ok(Some(ResolvedNoiseRule {
                rule: rule.clone(),
                source: RuleSource::ExplicitGate,
            }));
        }

        if gate.is_unitary() && gate.is_single_qubit_gate() {
            if let Some(rule) = &self.any_clifford_1q_rule {
                return Ok(Some(ResolvedNoiseRule {
                    rule: rule.clone(),
                    source: RuleSource::AnyClifford1Q,
                }));
            }
        }

        if gate.is_unitary() && gate.is_two_qubit_gate() {
            if let Some(rule) = &self.any_clifford_2q_rule {
                return Ok(Some(ResolvedNoiseRule {
                    rule: rule.clone(),
                    source: RuleSource::AnyClifford2Q,
                }));
            }
        }

        if gate.is_reset() && gate.produces_measurements() {
            if let Some(rule) = self.resolve_combined_measure_reset_rule(instruction, gate)? {
                return Ok(Some(rule));
            }
        }

        if gate.produces_measurements() {
            if let Some(basis) = measurement_basis_for_instruction(instruction) {
                if let Some(rule) = self.measure_rules.get(&basis) {
                    return Ok(Some(ResolvedNoiseRule {
                        rule: rule.clone(),
                        source: RuleSource::ExplicitMeasure,
                    }));
                }
            }
            if let Some(rule) = &self.any_measurement_rule {
                return Ok(Some(ResolvedNoiseRule {
                    rule: rule.clone(),
                    source: RuleSource::AnyMeasurement,
                }));
            }
        }

        Ok(None)
    }

    fn resolve_combined_measure_reset_rule(
        &self,
        instruction: &CircuitInstruction,
        gate: &crate::GateData,
    ) -> Result<Option<ResolvedNoiseRule>> {
        let reset_gate_name = match gate.name().as_str() {
            "MR" | "MRZ" => "R",
            "MRX" => "RX",
            "MRY" => "RY",
            _ => return Ok(None),
        };

        let mut rule = NoiseRule::new();
        let mut has_noise = false;

        if let Some(reset_rule) = self.gate_rules.get(reset_gate_name) {
            rule.before.extend(reset_rule.before.clone());
            rule.after.extend(reset_rule.after.clone());
            has_noise |= !reset_rule.before.is_empty() || !reset_rule.after.is_empty();
        }

        if let Some(basis) = measurement_basis_for_instruction(instruction) {
            if let Some(measure_rule) = self.measure_rules.get(&basis) {
                rule.flip_result = measure_rule.flip_result;
                has_noise |= rule.flip_result > 0.0;
            } else if let Some(measure_rule) = &self.any_measurement_rule {
                rule.flip_result = measure_rule.flip_result;
                has_noise |= rule.flip_result > 0.0;
            }
        }

        if has_noise {
            Ok(Some(ResolvedNoiseRule {
                rule,
                source: RuleSource::CombinedMeasureReset,
            }))
        } else {
            Ok(None)
        }
    }

    fn append_instruction_with_noise(
        &self,
        pending_moment: &mut PendingMoment,
        instruction: &CircuitInstruction,
        qubit_targets: &[u32],
        resolved_rule: Option<&ResolvedNoiseRule>,
        generic_1q_noise_applied: &mut BTreeSet<u32>,
    ) -> Result<()> {
        let Some(resolved_rule) = resolved_rule else {
            append_instruction_verbatim(&mut pending_moment.body, instruction)?;
            return Ok(());
        };

        let noise_targets = if resolved_rule.source == RuleSource::AnyClifford1Q {
            let fresh_targets: Vec<u32> = qubit_targets
                .iter()
                .filter(|qubit| !generic_1q_noise_applied.contains(qubit))
                .copied()
                .collect();
            generic_1q_noise_applied.extend(fresh_targets.iter().copied());
            fresh_targets
        } else {
            qubit_targets.to_vec()
        };

        record_noise_operations(
            &mut pending_moment.before,
            &noise_targets,
            resolved_rule.rule.before(),
        );

        let mut gate_args = instruction.gate_args_copy();
        if resolved_rule.rule.flip_result() > 0.0 {
            if !gate_args.is_empty() {
                return Err(StimError::new(format!(
                    "cannot inject result flips into already-parameterized instruction '{}'",
                    instruction
                )));
            }
            gate_args.push(resolved_rule.rule.flip_result());
        }

        let rewritten = CircuitInstruction::new(
            instruction.name(),
            instruction.targets_copy(),
            gate_args,
            instruction.tag(),
        )?;
        append_instruction_verbatim(&mut pending_moment.body, &rewritten)?;

        record_noise_operations(
            &mut pending_moment.after,
            &noise_targets,
            resolved_rule.rule.after(),
        );
        Ok(())
    }
}

fn validate_probability(probability: f64, context: &str) -> Result<()> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(StimError::new(format!(
            "{context} must be a finite probability between 0 and 1, got {probability}"
        )));
    }
    Ok(())
}

fn validate_gate_args(gate_name: &str, args: &[f64], allowed_counts: &[u8]) -> Result<()> {
    if !allowed_counts.contains(&(args.len() as u8)) {
        return Err(StimError::new(format!(
            "wrong number of arguments for '{gate_name}': got {}, expected one of {:?}",
            args.len(),
            allowed_counts
        )));
    }

    if args.len() == 1 {
        validate_probability(args[0], gate_name)?;
    } else if !args.is_empty() {
        let mut total = 0.0;
        for probability in args {
            validate_probability(*probability, gate_name)?;
            total += probability;
        }
        if total > 1.0 + f64::EPSILON {
            return Err(StimError::new(format!(
                "noise-channel probabilities for '{gate_name}' must sum to at most 1, got {total}"
            )));
        }
    }

    Ok(())
}

fn is_annotation(gate_name: &str) -> bool {
    matches!(
        gate_name,
        "DETECTOR" | "OBSERVABLE_INCLUDE" | "QUBIT_COORDS" | "SHIFT_COORDS" | "MPAD"
    )
}

fn occurs_in_classical_control_system(instruction: &CircuitInstruction) -> Result<bool> {
    if is_annotation(instruction.name()) || instruction.name() == "TICK" {
        return Ok(true);
    }

    let gate = gate_data(instruction.name())?;
    if gate.is_unitary() && gate.is_two_qubit_gate() {
        let targets = instruction.targets_copy();
        for pair in targets.chunks_exact(2) {
            let classical_0 =
                pair[0].is_measurement_record_target() || pair[0].is_sweep_bit_target();
            let classical_1 =
                pair[1].is_measurement_record_target() || pair[1].is_sweep_bit_target();
            if !(classical_0 || classical_1) {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    Ok(false)
}

fn split_instruction_for_noise(
    instruction: &CircuitInstruction,
) -> Result<Vec<CircuitInstruction>> {
    if instruction.name() != "MPP" {
        return Ok(vec![instruction.clone()]);
    }

    let groups = instruction.target_groups();
    if groups.len() <= 1 {
        return Ok(vec![instruction.clone()]);
    }

    groups
        .into_iter()
        .map(|group| {
            let mut targets = Vec::with_capacity(group.len() * 2 - 1);
            for (index, target) in group.iter().enumerate() {
                if index > 0 {
                    targets.push(GateTarget::combiner());
                }
                targets.push(*target);
            }
            CircuitInstruction::new(
                "MPP",
                targets,
                instruction.gate_args_copy(),
                instruction.tag(),
            )
        })
        .collect()
}

fn qubit_targets(instruction: &CircuitInstruction) -> Vec<u32> {
    instruction
        .targets_copy()
        .into_iter()
        .filter_map(GateTarget::qubit_value)
        .collect()
}

fn measurement_basis_for_instruction(instruction: &CircuitInstruction) -> Option<String> {
    match instruction.name() {
        "M" | "MZ" | "MR" | "MRZ" => Some("Z".to_string()),
        "MX" | "MRX" => Some("X".to_string()),
        "MY" | "MRY" => Some("Y".to_string()),
        "MXX" => Some("XX".to_string()),
        "MYY" => Some("YY".to_string()),
        "MZZ" => Some("ZZ".to_string()),
        "MPP" => {
            let groups = instruction.target_groups();
            let group = groups.first()?;
            let basis: String = group.iter().map(|target| target.pauli_type()).collect();
            Some(basis)
        }
        _ => None,
    }
}

fn annotation_line_name(line: &str) -> Option<&str> {
    line.split_once([' ', '[', '('])
        .map(|(name, _)| name)
        .or(Some(line))
        .filter(|name| is_annotation(name))
}

fn append_instruction_verbatim(
    output: &mut Circuit,
    instruction: &CircuitInstruction,
) -> Result<()> {
    if instruction.tag().is_empty() {
        output.append_gate_targets(
            instruction.name(),
            &instruction.targets_copy(),
            &instruction.gate_args_copy(),
        )
    } else {
        output.append_from_stim_program_text(&instruction.to_string())
    }
}

fn record_noise_operations(
    grouped_operations: &mut BTreeMap<(String, Vec<u64>), Vec<u32>>,
    qubit_targets: &[u32],
    operations: &[NoiseOperation],
) {
    if qubit_targets.is_empty() {
        return;
    }

    for operation in operations {
        let key = (
            operation.gate_name().to_string(),
            operation.args().iter().map(|arg| arg.to_bits()).collect(),
        );
        grouped_operations
            .entry(key)
            .or_default()
            .extend(qubit_targets.iter().copied());
    }
}

fn flush_pending_moment(output: &mut Circuit, pending_moment: &mut PendingMoment) -> Result<()> {
    append_grouped_noise_ops(output, &pending_moment.before)?;
    *output += pending_moment.body.clone();
    append_grouped_noise_ops(output, &pending_moment.after)?;
    pending_moment.before.clear();
    pending_moment.body = Circuit::new();
    pending_moment.after.clear();
    Ok(())
}

fn append_grouped_noise_ops(
    output: &mut Circuit,
    grouped_operations: &BTreeMap<(String, Vec<u64>), Vec<u32>>,
) -> Result<()> {
    for ((gate_name, args), targets) in grouped_operations {
        if targets.is_empty() {
            continue;
        }
        let gate_args: Vec<f64> = args.iter().map(|arg| f64::from_bits(*arg)).collect();
        output.append(gate_name, targets, &gate_args)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        NoiseModel, NoiseModelConfig, NoiseOperation, NoiseRule, Si1000, UniformDepolarizing,
        annotation_line_name, append_grouped_noise_ops, append_instruction_verbatim,
        occurs_in_classical_control_system, record_noise_operations, split_instruction_for_noise,
        validate_gate_args, validate_probability,
    };
    use crate::Circuit;

    #[test]
    fn uniform_depolarizing_adds_gate_and_measurement_noise() {
        let circuit: Circuit = "H 0\nTICK\nM 0".parse().unwrap();

        let noisy = UniformDepolarizing::new(0.001)
            .unwrap()
            .noisy_circuit(&circuit)
            .unwrap();

        assert_eq!(
            noisy.to_string(),
            "H 0\nDEPOLARIZE1(0.001) 0\nTICK\nM(0.001) 0\nDEPOLARIZE1(0.001) 0"
        );
    }

    #[test]
    fn internal_config_splits_mpp_measurement_rules_by_basis() {
        let model = NoiseModelConfig::new()
            .with_any_clifford_1q_rule(NoiseRule::new())
            .with_any_clifford_2q_rule(NoiseRule::new())
            .with_any_measurement_rule(
                NoiseRule::new()
                    .with_after("DEPOLARIZE1", &[0.125])
                    .unwrap()
                    .with_flip_result(0.25)
                    .unwrap(),
            )
            .with_measure_rule("XX", NoiseRule::new().with_flip_result(0.375).unwrap());

        let circuit: Circuit = "MPP Z0*Z1 X2*X3 X4*X5*X6".parse().unwrap();
        let noisy = model.noisy_circuit(&circuit).unwrap();

        assert_eq!(
            noisy.to_string(),
            "MPP(0.25) Z0*Z1\nMPP(0.375) X2*X3\nMPP(0.25) X4*X5*X6\nDEPOLARIZE1(0.125) 0 1 4 5 6"
        );
    }

    #[test]
    fn idle_noise_skips_qubits_that_were_only_annotated() {
        let circuit: Circuit = "\
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
H 0
TICK
TICK"
            .parse()
            .unwrap();

        let noisy = UniformDepolarizing::new(0.001)
            .unwrap()
            .noisy_circuit(&circuit)
            .unwrap();

        assert_eq!(
            noisy.to_string(),
            "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(1, 0) 1\nH 0\nDEPOLARIZE1(0.001) 0\nTICK\nDEPOLARIZE1(0.001) 0\nTICK"
        );
        assert!(!noisy.to_string().contains("DEPOLARIZE1(0.001) 1"));
    }

    #[test]
    fn si1000_uses_asymmetric_reset_gate_and_measurement_rates() {
        let circuit: Circuit = "R 0\nTICK\nH 0\nTICK\nM 0".parse().unwrap();

        let noisy = Si1000::new(0.001).unwrap().noisy_circuit(&circuit).unwrap();

        assert_eq!(
            noisy.to_string(),
            "R 0\nX_ERROR(0.002) 0\nTICK\nH 0\nDEPOLARIZE1(0.0001) 0\nTICK\nM(0.005) 0\nDEPOLARIZE1(0.001) 0"
        );
    }

    #[test]
    fn rejects_non_noise_channels_inside_noise_rules() {
        let error = NoiseRule::new().with_after("H", &[]).unwrap_err();
        assert!(error.to_string().contains("pure noise channel"));
    }

    #[test]
    fn internal_config_covers_combined_reset_and_classical_control_paths() {
        let model = NoiseModelConfig::new()
            .with_gate_rule(
                "R",
                NoiseRule::new().with_after("X_ERROR", &[0.25]).unwrap(),
            )
            .unwrap()
            .with_measure_rule("Z", NoiseRule::new().with_flip_result(0.125).unwrap());

        let mr_circuit: Circuit = "MR 0".parse().unwrap();
        let noisy = model.noisy_circuit(&mr_circuit).unwrap();
        assert!(noisy.to_string().contains("MR(0.125) 0"));
        assert!(noisy.to_string().contains("X_ERROR(0.25) 0"));

        let control_model = NoiseModelConfig::new().with_any_clifford_2q_rule(
            NoiseRule::new().with_after("DEPOLARIZE2", &[0.01]).unwrap(),
        );
        let classically_controlled: Circuit = "M 0\nCX rec[-1] 1".parse().unwrap();
        let noisy = control_model
            .noisy_circuit(&classically_controlled)
            .unwrap();
        assert_eq!(noisy.to_string(), "M 0\nCX rec[-1] 1");
        assert!(
            occurs_in_classical_control_system(
                &crate::CircuitInstruction::from_stim_program_text("CX rec[-1] 1").unwrap()
            )
            .unwrap()
        );
    }

    #[test]
    fn internal_noise_validation_and_helper_paths_are_exercised() {
        let reuse_model = NoiseModelConfig::new().with_any_clifford_1q_rule(NoiseRule::new());
        let reused_qubit: Circuit = "H 0\nM 0".parse().unwrap();
        let error = reuse_model.noisy_circuit(&reused_qubit).unwrap_err();
        assert!(error.to_string().contains("qubit reused in the same TICK"));

        let measurement_flip_model = NoiseModelConfig::new()
            .with_any_measurement_rule(NoiseRule::new().with_flip_result(0.25).unwrap());
        let parameterized_measurement: Circuit = "MPP(0.125) X0*X1".parse().unwrap();
        let error = measurement_flip_model
            .noisy_circuit(&parameterized_measurement)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot inject result flips into already-parameterized instruction")
        );

        let mpp = crate::CircuitInstruction::from_stim_program_text("MPP X0*X1 Z2*Z3").unwrap();
        let split = split_instruction_for_noise(&mpp).unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!(
            split[0].target_groups(),
            vec![vec![
                crate::GateTarget::x(0, false).unwrap(),
                crate::GateTarget::x(1, false).unwrap(),
            ]]
        );
        assert_eq!(
            split[1].target_groups(),
            vec![vec![
                crate::GateTarget::z(2, false).unwrap(),
                crate::GateTarget::z(3, false).unwrap(),
            ]]
        );

        assert_eq!(
            split_instruction_for_noise(
                &crate::CircuitInstruction::from_stim_program_text("MPP X0*X1").unwrap()
            )
            .unwrap()
            .len(),
            1
        );

        assert!(annotation_line_name("DETECTOR(1, 2) rec[-1]").is_some());
        assert!(annotation_line_name("H 0").is_none());

        let mut with_tag = Circuit::new();
        let tagged = crate::CircuitInstruction::new("H", [0u32], [], "tag").unwrap();
        append_instruction_verbatim(&mut with_tag, &tagged).unwrap();
        assert_eq!(with_tag.to_string(), "H[tag] 0");

        let mut without_tag = Circuit::new();
        let plain = crate::CircuitInstruction::new("H", [1u32], [], "").unwrap();
        append_instruction_verbatim(&mut without_tag, &plain).unwrap();
        assert_eq!(without_tag.to_string(), "H 1");

        let mut grouped = std::collections::BTreeMap::new();
        record_noise_operations(
            &mut grouped,
            &[],
            &[NoiseOperation::new("X_ERROR", &[0.25]).unwrap()],
        );
        assert!(grouped.is_empty());
        grouped.insert(("X_ERROR".to_string(), vec![0.25f64.to_bits()]), Vec::new());
        let mut output = Circuit::new();
        append_grouped_noise_ops(&mut output, &grouped).unwrap();
        assert!(output.is_empty());

        let invalid_probability = validate_probability(f64::NAN, "noise").unwrap_err();
        assert!(
            invalid_probability
                .to_string()
                .contains("must be a finite probability")
        );
        let invalid_count = validate_gate_args("X_ERROR", &[], &[1]).unwrap_err();
        assert!(
            invalid_count
                .to_string()
                .contains("wrong number of arguments for 'X_ERROR'")
        );
        let invalid_sum =
            validate_gate_args("PAULI_CHANNEL_1", &[0.5, 0.5, 0.5], &[3]).unwrap_err();
        assert!(invalid_sum.to_string().contains("must sum to at most 1"));
    }
}
