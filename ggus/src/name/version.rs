use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Version {
    major: u32,
    minor: u32,
}

impl Version {
    pub const DEFAULT: Self = Self { major: 1, minor: 0 };

    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &Self { major, minor } = self;
        write!(f, "v{major}.{minor}")
    }
}
