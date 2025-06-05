use super::pattern::{TYPE_LORA, TYPE_VOCAB};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum Type {
    Default,
    LoRA,
    Vocab,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Default => Ok(()),
            Type::LoRA => write!(f, "{TYPE_LORA}"),
            Type::Vocab => write!(f, "{TYPE_VOCAB}"),
        }
    }
}
