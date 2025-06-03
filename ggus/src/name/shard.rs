use std::{fmt, num::NonZero};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shard {
    pub index: NonZero<u32>,
    pub count: NonZero<u32>,
}

impl Default for Shard {
    fn default() -> Self {
        Self {
            index: NonZero::new(1).unwrap(),
            count: NonZero::new(1).unwrap(),
        }
    }
}

impl Shard {
    pub fn new(index: u32, count: u32) -> Self {
        Self {
            index: NonZero::new(index).unwrap(),
            count: NonZero::new(count).unwrap(),
        }
    }
}

impl fmt::Display for Shard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &Self { index, count } = self;
        if count.get() == 1 {
            Ok(())
        } else {
            write!(f, "-{index:05}-of-{count:05}")
        }
    }
}
