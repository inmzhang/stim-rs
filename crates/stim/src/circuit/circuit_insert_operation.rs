use crate::{Circuit, CircuitInstruction, CircuitRepeatBlock};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CircuitInsertOperation {
    Instruction(CircuitInstruction),
    Circuit(Circuit),
    RepeatBlock(CircuitRepeatBlock),
}

impl From<CircuitInstruction> for CircuitInsertOperation {
    fn from(value: CircuitInstruction) -> Self {
        Self::Instruction(value)
    }
}

impl From<&CircuitInstruction> for CircuitInsertOperation {
    fn from(value: &CircuitInstruction) -> Self {
        Self::Instruction(value.clone())
    }
}

impl From<Circuit> for CircuitInsertOperation {
    fn from(value: Circuit) -> Self {
        Self::Circuit(value)
    }
}

impl From<&Circuit> for CircuitInsertOperation {
    fn from(value: &Circuit) -> Self {
        Self::Circuit(value.clone())
    }
}

impl From<CircuitRepeatBlock> for CircuitInsertOperation {
    fn from(value: CircuitRepeatBlock) -> Self {
        Self::RepeatBlock(value)
    }
}

impl From<&CircuitRepeatBlock> for CircuitInsertOperation {
    fn from(value: &CircuitRepeatBlock) -> Self {
        Self::RepeatBlock(value.clone())
    }
}
