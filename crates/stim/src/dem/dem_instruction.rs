use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::{DemTarget, DetectorErrorModel, Result, StimError};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DemInstructionTarget {
    DemTarget(DemTarget),
    RelativeOffset(u64),
}

impl DemInstructionTarget {
    #[must_use]
    pub fn is_separator(self) -> bool {
        matches!(self, Self::DemTarget(target) if target.is_separator())
    }
}

impl From<DemTarget> for DemInstructionTarget {
    fn from(value: DemTarget) -> Self {
        Self::DemTarget(value)
    }
}

impl From<u64> for DemInstructionTarget {
    fn from(value: u64) -> Self {
        Self::RelativeOffset(value)
    }
}

impl From<u32> for DemInstructionTarget {
    fn from(value: u32) -> Self {
        Self::RelativeOffset(value as u64)
    }
}

impl From<usize> for DemInstructionTarget {
    fn from(value: usize) -> Self {
        Self::RelativeOffset(value as u64)
    }
}

impl Display for DemInstructionTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::DemTarget(target) => write!(f, "{target}"),
            Self::RelativeOffset(value) => write!(f, "{value}"),
        }
    }
}

impl fmt::Debug for DemInstructionTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::DemTarget(target) => write!(f, "{target:?}"),
            Self::RelativeOffset(value) => write!(f, "{value}"),
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct DemInstruction {
    instruction_type: String,
    tag: String,
    args: Vec<f64>,
    targets: Vec<DemInstructionTarget>,
}

impl DemInstruction {
    pub fn new(
        instruction_type: impl Into<String>,
        args: impl IntoIterator<Item = f64>,
        targets: impl IntoIterator<Item = impl Into<DemInstructionTarget>>,
        tag: impl Into<String>,
    ) -> Result<Self> {
        let instruction = Self {
            instruction_type: instruction_type.into(),
            tag: tag.into(),
            args: args.into_iter().collect(),
            targets: targets.into_iter().map(Into::into).collect(),
        };
        instruction.validate()?;
        Ok(instruction)
    }

    pub fn from_dem_text(text: &str) -> Result<Self> {
        let normalized = DetectorErrorModel::from_str(text)?.to_string();
        if normalized.is_empty() {
            return Err(StimError::new(
                "expected a single detector error model instruction, got empty text",
            ));
        }
        if normalized.starts_with("repeat ") {
            return Err(StimError::new(
                "DemInstruction cannot represent DEM repeat blocks",
            ));
        }
        if normalized.lines().count() != 1 {
            return Err(StimError::new(
                "expected a single detector error model instruction",
            ));
        }

        let (head, tail) = split_head_and_tail(&normalized);
        let (instruction_type, tag, args) = parse_head(head)?;
        let targets = parse_targets(tail)?;
        Self::new(instruction_type, args, targets, tag)
    }

    #[must_use]
    pub fn r#type(&self) -> &str {
        &self.instruction_type
    }

    #[must_use]
    pub fn tag(&self) -> &str {
        &self.tag
    }

    #[must_use]
    pub fn args_copy(&self) -> Vec<f64> {
        self.args.clone()
    }

    #[must_use]
    pub fn targets_copy(&self) -> Vec<DemInstructionTarget> {
        self.targets.clone()
    }

    #[must_use]
    pub fn target_groups(&self) -> Vec<Vec<DemInstructionTarget>> {
        if self.targets.is_empty() {
            return vec![Vec::new()];
        }

        let mut groups = Vec::new();
        let mut current = Vec::new();
        for target in &self.targets {
            if target.is_separator() {
                groups.push(std::mem::take(&mut current));
            } else {
                current.push(*target);
            }
        }
        groups.push(current);
        groups
    }

    fn validate(&self) -> Result<()> {
        if self.instruction_type.trim().is_empty() {
            return Err(StimError::new("instruction type must not be empty"));
        }
        let _ = DetectorErrorModel::from_str(&self.to_string())?;
        Ok(())
    }

    fn fmt_repr(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stim::DemInstruction({:?}, {:?}",
            self.instruction_type, self.args
        )?;
        f.write_str(", [")?;
        for (index, target) in self.targets.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{target:?}")?;
        }
        f.write_str("]")?;
        if !self.tag.is_empty() {
            write!(f, ", tag={:?}", self.tag)?;
        }
        f.write_str(")")
    }
}

impl Eq for DemInstruction {}

impl PartialOrd for DemInstruction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DemInstruction {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.instruction_type
            .cmp(&other.instruction_type)
            .then_with(|| self.tag.cmp(&other.tag))
            .then_with(|| compare_f64_slices(&self.args, &other.args))
            .then_with(|| self.targets.cmp(&other.targets))
    }
}

impl std::hash::Hash for DemInstruction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.instruction_type.hash(state);
        self.tag.hash(state);
        for arg in &self.args {
            arg.to_bits().hash(state);
        }
        self.targets.hash(state);
    }
}

impl Display for DemInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.instruction_type)?;
        if !self.tag.is_empty() {
            write!(f, "[{}]", self.tag)?;
        }
        if !self.args.is_empty() {
            f.write_str("(")?;
            for (index, arg) in self.args.iter().enumerate() {
                if index > 0 {
                    f.write_str(",")?;
                }
                write!(f, "{arg}")?;
            }
            f.write_str(")")?;
        }
        for target in &self.targets {
            write!(f, " {target}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for DemInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_repr(f)
    }
}

impl FromStr for DemInstruction {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_dem_text(s)
    }
}

fn parse_head(head: &str) -> Result<(String, String, Vec<f64>)> {
    let name_end = head.find(['[', '(']).unwrap_or(head.len());
    let instruction_type = &head[..name_end];
    let mut rest = &head[name_end..];
    let mut tag = String::new();
    let mut args = Vec::new();

    if let Some(after_open) = rest.strip_prefix('[') {
        let close = after_open
            .find(']')
            .ok_or_else(|| StimError::new("unterminated DEM instruction tag"))?;
        tag = after_open[..close].to_string();
        rest = &after_open[close + 1..];
    }

    if let Some(after_open) = rest.strip_prefix('(') {
        let close = after_open
            .find(')')
            .ok_or_else(|| StimError::new("unterminated DEM instruction argument list"))?;
        let raw_args = &after_open[..close];
        if !raw_args.is_empty() {
            args = raw_args
                .split(',')
                .map(|arg| {
                    arg.trim()
                        .parse::<f64>()
                        .map_err(|_| StimError::new(format!("invalid DEM arg: {arg}")))
                })
                .collect::<Result<Vec<_>>>()?;
        }
    }

    Ok((instruction_type.to_string(), tag, args))
}

fn parse_targets(text: &str) -> Result<Vec<DemInstructionTarget>> {
    text.split_whitespace()
        .map(|token| {
            DemTarget::from_text(token)
                .map(DemInstructionTarget::from)
                .or_else(|_| {
                    token
                        .parse::<u64>()
                        .map(DemInstructionTarget::from)
                        .map_err(|_| {
                            StimError::new(format!(
                                "failed to parse DEM instruction target: {token}"
                            ))
                        })
                })
        })
        .collect()
}

fn split_head_and_tail(text: &str) -> (&str, &str) {
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            ' ' if paren_depth == 0 && bracket_depth == 0 => {
                return (&text[..index], &text[index + 1..]);
            }
            _ => {}
        }
    }
    (text, "")
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};
    use std::str::FromStr;

    use crate::target_logical_observable_id;
    use crate::target_relative_detector_id;
    use crate::target_separator;

    #[test]
    fn constructor_and_accessors_preserve_values() {
        let instruction = DemInstruction::new(
            "error",
            [0.125],
            [
                target_relative_detector_id(5).unwrap(),
                target_logical_observable_id(2).unwrap(),
            ],
            "test-tag",
        )
        .unwrap();

        assert_eq!(instruction.r#type(), "error");
        assert_eq!(instruction.tag(), "test-tag");
        assert_eq!(instruction.args_copy(), vec![0.125]);
        assert_eq!(
            instruction.targets_copy(),
            vec![
                DemInstructionTarget::from(target_relative_detector_id(5).unwrap()),
                DemInstructionTarget::from(target_logical_observable_id(2).unwrap()),
            ]
        );
        assert_eq!(instruction.to_string(), "error[test-tag](0.125) D5 L2");
    }

    #[test]
    fn parsed_line_roundtrips_and_strips_comments() {
        let instruction = DemInstruction::from_str("error(0.125) D5 L6 ^ D4  # comment").unwrap();

        assert_eq!(instruction.r#type(), "error");
        assert_eq!(instruction.tag(), "");
        assert_eq!(instruction.args_copy(), vec![0.125]);
        assert_eq!(
            instruction.targets_copy(),
            vec![
                DemInstructionTarget::from(target_relative_detector_id(5).unwrap()),
                DemInstructionTarget::from(target_logical_observable_id(6).unwrap()),
                DemInstructionTarget::from(target_separator()),
                DemInstructionTarget::from(target_relative_detector_id(4).unwrap()),
            ]
        );
        assert_eq!(instruction.to_string(), "error(0.125) D5 L6 ^ D4");
    }

    #[test]
    fn equality_hash_and_order_follow_all_fields() {
        let first = DemInstruction::new(
            "error",
            [0.125],
            [target_relative_detector_id(2).unwrap()],
            "",
        )
        .unwrap();
        let same = DemInstruction::new(
            "error",
            [0.125],
            [target_relative_detector_id(2).unwrap()],
            "",
        )
        .unwrap();
        let different = DemInstruction::new(
            "error",
            [0.125],
            [target_relative_detector_id(3).unwrap()],
            "",
        )
        .unwrap();

        assert_eq!(first, same);
        assert_ne!(first, different);

        let ordered = [different.clone(), same.clone(), first.clone()]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(ordered, vec![first.clone(), different.clone()]);

        let hashed = [first.clone(), same, different.clone()]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 2);
        assert!(hashed.contains(&first));
        assert!(hashed.contains(&different));
    }

    #[test]
    fn target_groups_split_on_separators_and_preserve_empty_groups() {
        let split = DemInstruction::new(
            "error",
            [0.01],
            [
                target_relative_detector_id(0).unwrap(),
                target_relative_detector_id(1).unwrap(),
                target_separator(),
                target_relative_detector_id(2).unwrap(),
            ],
            "",
        )
        .unwrap();
        let single = DemInstruction::new(
            "error",
            [0.01],
            [
                target_relative_detector_id(0).unwrap(),
                target_logical_observable_id(0).unwrap(),
            ],
            "",
        )
        .unwrap();
        let empty =
            DemInstruction::new("error", [0.01], Vec::<DemInstructionTarget>::new(), "").unwrap();

        assert_eq!(
            split.target_groups(),
            vec![
                vec![
                    DemInstructionTarget::from(target_relative_detector_id(0).unwrap()),
                    DemInstructionTarget::from(target_relative_detector_id(1).unwrap())
                ],
                vec![DemInstructionTarget::from(
                    target_relative_detector_id(2).unwrap()
                )],
            ]
        );
        assert_eq!(
            single.target_groups(),
            vec![vec![
                DemInstructionTarget::from(target_relative_detector_id(0).unwrap()),
                DemInstructionTarget::from(target_logical_observable_id(0).unwrap()),
            ]]
        );
        assert_eq!(empty.target_groups(), vec![Vec::new()]);
    }

    #[test]
    fn raw_integer_targets_are_supported_for_shift_detectors() {
        let instruction =
            DemInstruction::new("shift_detectors", [1.0, 2.0, 3.0], [5u64], "").unwrap();

        assert_eq!(
            instruction.targets_copy(),
            vec![DemInstructionTarget::from(5u64)]
        );
        assert_eq!(instruction.to_string(), "shift_detectors(1,2,3) 5");
        assert_eq!(
            DemInstruction::from_str("shift_detectors(1,2,3) 5").unwrap(),
            instruction
        );
    }

    #[test]
    fn debug_matches_binding_conventions() {
        let instruction = DemInstruction::new(
            "error",
            [0.125],
            [target_relative_detector_id(5).unwrap()],
            "test-tag",
        )
        .unwrap();

        assert_eq!(
            format!("{instruction:?}"),
            "stim::DemInstruction(\"error\", [0.125], [stim::DemTarget('D5')], tag=\"test-tag\")"
        );

        let shifted = DemInstruction::new("shift_detectors", [1.0], [5u64], "").unwrap();
        assert_eq!(
            format!("{shifted:?}"),
            "stim::DemInstruction(\"shift_detectors\", [1.0], [5])"
        );
    }
}
