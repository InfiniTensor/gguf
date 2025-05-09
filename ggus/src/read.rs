use crate::metadata::GGufMetaDataValueType;
use std::{
    alloc::Layout,
    str::{Utf8Error, from_utf8, from_utf8_unchecked},
};

#[derive(Clone)]
#[repr(transparent)]
pub struct GGufReader<'a>(&'a [u8]);

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GGufReadError {
    Eos,
    Utf8(Utf8Error),
    Bool(u8),
}

impl<'a> GGufReader<'a> {
    #[inline]
    pub const fn new(data: &'a [u8]) -> Self {
        Self(data)
    }

    #[inline]
    pub const fn remaining(&self) -> &'a [u8] {
        self.0
    }

    pub(crate) fn skip<T>(&mut self, len: usize) -> Result<&mut Self, GGufReadError> {
        let len = Layout::array::<T>(len).unwrap().size();
        let (_, tail) = self.0.split_at_checked(len).ok_or(GGufReadError::Eos)?;
        self.0 = tail;
        Ok(self)
    }

    pub(crate) fn skip_str(&mut self) -> Result<&mut Self, GGufReadError> {
        let len = self.read::<u64>()?;
        self.skip::<u8>(len as _)
    }

    pub fn read<T: Copy>(&mut self) -> Result<T, GGufReadError> {
        let ptr = self.0.as_ptr().cast::<T>();
        self.skip::<T>(1)?;
        Ok(unsafe { ptr.read_unaligned() })
    }

    pub fn read_bool(&mut self) -> Result<bool, GGufReadError> {
        match self.read::<u8>()? {
            0 => Ok(false),
            1 => Ok(true),
            e => Err(GGufReadError::Bool(e)),
        }
    }

    pub fn read_str(&mut self) -> Result<&'a str, GGufReadError> {
        let len = self.read::<u64>()? as _;
        let (s, tail) = self.0.split_at_checked(len).ok_or(GGufReadError::Eos)?;
        let ans = from_utf8(s).map_err(GGufReadError::Utf8)?;
        self.0 = tail;
        Ok(ans)
    }

    /// Read a string without checking if it is valid utf8.
    ///
    /// # Safety
    ///
    /// This function does not check if the data is valid utf8.
    pub unsafe fn read_str_unchecked(&mut self) -> &'a str {
        let len = self.read::<u64>().unwrap() as _;
        let (s, tail) = self.0.split_at(len);
        self.0 = tail;
        unsafe { from_utf8_unchecked(s) }
    }

    pub fn read_arr_header(&mut self) -> Result<(GGufMetaDataValueType, usize), GGufReadError> {
        Ok((self.read()?, self.read::<u64>()? as _))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read() {
        let data: &[u8] = &[1, 2, 3, 4, 5];
        let mut reader = GGufReader::new(data);
        assert_eq!(reader.read::<u8>().unwrap(), 1);
        assert_eq!(reader.read::<u8>().unwrap(), 2);
        assert_eq!(reader.read::<u8>().unwrap(), 3);
        assert_eq!(reader.read::<u8>().unwrap(), 4);
        assert_eq!(reader.read::<u8>().unwrap(), 5);
    }

    #[test]
    fn test_read_bool() {
        let data: &[u8] = &[0, 1, 2];
        let mut reader = GGufReader::new(data);
        assert!(!reader.read_bool().unwrap());
        assert!(reader.read_bool().unwrap());
        assert!(matches!(reader.read_bool(), Err(GGufReadError::Bool(2))));
    }
}
