use crate::metadata::GGufMetaDataValueType;
use std::{
    alloc::Layout,
    str::{Utf8Error, from_utf8, from_utf8_unchecked},
};

/// [`GGufReader`] 定义读取 GGUF 文件的读取器。
#[derive(Clone)]
#[repr(transparent)]
pub struct GGufReader<'a>(&'a [u8]);

/// [`GGufReadError`] 定义 GGUF 读取器可能遇到的错误。
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GGufReadError {
    /// 读取过程中遇到的错误。
    Eos,
    /// 读取的字符串不是有效的 UTF-8 编码。
    Utf8(Utf8Error),
    /// 读取布尔值时遇到的错误，表示读取到的字节不是 0 或 1。
    Bool(u8),
}

impl<'a> GGufReader<'a> {
    /// 创建一个新的 [`GGufReader`] 实例。
    #[inline]
    pub const fn new(data: &'a [u8]) -> Self {
        Self(data)
    }

    /// 获取当前读取器的剩余数据。
    #[inline]
    pub const fn remaining(&self) -> &'a [u8] {
        self.0
    }

    /// 跳过指定长度的字节。
    pub(crate) fn skip<T>(&mut self, len: usize) -> Result<&mut Self, GGufReadError> {
        let len = Layout::array::<T>(len).unwrap().size();
        let (_, tail) = self.0.split_at_checked(len).ok_or(GGufReadError::Eos)?;
        self.0 = tail;
        Ok(self)
    }

    /// 跳过一个字符串，读取其长度但不返回内容。
    pub(crate) fn skip_str(&mut self) -> Result<&mut Self, GGufReadError> {
        let len = self.read::<u64>()?;
        self.skip::<u8>(len as _)
    }

    /// 读取指定类型的值。
    pub fn read<T: Copy>(&mut self) -> Result<T, GGufReadError> {
        let ptr = self.0.as_ptr().cast::<T>();
        self.skip::<T>(1)?;
        Ok(unsafe { ptr.read_unaligned() })
    }

    /// 读取 bool 值。
    pub fn read_bool(&mut self) -> Result<bool, GGufReadError> {
        match self.read::<u8>()? {
            0 => Ok(false),
            1 => Ok(true),
            e => Err(GGufReadError::Bool(e)),
        }
    }

    /// 读取字符串。
    pub fn read_str(&mut self) -> Result<&'a str, GGufReadError> {
        let len = self.read::<u64>()? as _;
        let (s, tail) = self.0.split_at_checked(len).ok_or(GGufReadError::Eos)?;
        let ans = from_utf8(s).map_err(GGufReadError::Utf8)?;
        self.0 = tail;
        Ok(ans)
    }

    /// 读取字符串，不检查 UTF-8 编码。
    ///
    /// # Safety
    ///
    /// 调用此函数时，必须确保读取的字节是有效的 UTF-8 编码，否则会导致未定义行为。
    pub unsafe fn read_str_unchecked(&mut self) -> &'a str {
        let len = self.read::<u64>().unwrap() as _;
        let (s, tail) = self.0.split_at(len);
        self.0 = tail;
        unsafe { from_utf8_unchecked(s) }
    }

    /// 读取一个数组头部，返回元数据类型和数组长度。
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
