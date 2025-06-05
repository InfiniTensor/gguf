use std::{fmt, num::NonZeroU32};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SizeLabel {
    e: NonZeroU32,
    a: u32,
    b: u32,
    l: char,
}

impl SizeLabel {
    pub fn new(e: u32, a: u32, b: u32, l: char) -> Self {
        Self {
            e: NonZeroU32::new(e).unwrap(),
            a,
            b,
            l,
        }
    }
}

impl fmt::Display for SizeLabel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &Self { e, a, b, l } = self;
        match e.get() {
            1 => {}
            _ => write!(f, "{e}x")?,
        }
        match b {
            0 => write!(f, "{a}{l}"),
            _ => write!(f, "{a}.{b}{l}"),
        }
    }
}
