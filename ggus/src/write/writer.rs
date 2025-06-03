use crate::{GENERAL_ALIGNMENT, GGmlType, GGufFileHeader, GGufMetaDataValueType, pad};
use internal::Internal;
use std::{
    io::{Result, Write},
    slice::from_raw_parts,
};

/// [`GGufWriter`] 用于写入 GGUF 文件格式的数据。
#[repr(transparent)]
pub struct GGufWriter<T: Write>(Internal<T>);

impl<T: Write> GGufWriter<T> {
    /// 创建一个新的 [`GGufWriter`] 实例。
    #[inline]
    pub fn new(writer: T) -> Self {
        Self(Internal::new(writer))
    }

    /// 获取已写入的字节数。
    #[inline]
    pub const fn written_bytes(&self) -> usize {
        self.0.written_bytes()
    }

    /// 写入 GGUF 文件头。
    pub fn write_header(&mut self, header: GGufFileHeader) -> Result<()> {
        self.write(unsafe {
            from_raw_parts(
                &header as *const _ as *const u8,
                size_of::<GGufFileHeader>(),
            )
        })
    }

    /// 写入指定值。
    pub fn write<U: Copy + 'static>(&mut self, val: &[U]) -> Result<()> {
        self.0
            .write_bytes(unsafe { from_raw_parts(val.as_ptr().cast(), size_of_val(val)) })
    }

    /// 写入字符串。
    pub fn write_str(&mut self, val: impl AsRef<str>) -> Result<()> {
        let val = val.as_ref().as_bytes();
        self.write(&[val.len() as u64])?;
        self.write(val)
    }

    /// 写入对齐方式。
    pub fn write_alignment(&mut self, alignment: usize) -> Result<()> {
        self.write_meta_kv(
            GENERAL_ALIGNMENT,
            GGufMetaDataValueType::U32,
            &(alignment as u32).to_le_bytes(),
        )?;
        Ok(())
    }

    /// 写入元数据键值对。
    pub fn write_meta_kv(
        &mut self,
        key: &str,
        ty: GGufMetaDataValueType,
        val: &[u8],
    ) -> Result<Option<usize>> {
        self.write_str(key)?;
        self.write(&[ty])?;
        self.write(val)?;

        Ok(if key == GENERAL_ALIGNMENT {
            let &[a, b, c, d] = val else {
                panic!("general.alignment must be an u32")
            };
            Some(u32::from_le_bytes([a, b, c, d]) as _)
        } else {
            None
        })
    }

    /// 写入张量信息。
    pub fn write_tensor_info(
        &mut self,
        name: &str,
        shape: &[u64],
        ty: GGmlType,
        offset: u64,
    ) -> Result<()> {
        self.write_str(name)?;
        self.write(&[shape.len() as u32])?;
        self.write(shape)?;
        self.write(&[ty])?;
        self.write(&[offset])
    }

    /// 写入填充值。
    pub fn write_padding(&mut self, alignment: usize) -> Result<()> {
        for _ in 0..pad(self.written_bytes(), alignment) {
            self.write(&[0u8])?;
        }
        Ok(())
    }

    /// 写入数据。
    pub fn write_data(&mut self, data: &[u8]) -> Result<()> {
        self.write(data)
    }
}

mod internal {
    use std::io::{BufWriter, Result, Write};

    pub(super) struct Internal<T: Write>(BufWriter<T>, usize);

    impl<T: Write> Internal<T> {
        #[inline]
        pub fn new(writer: T) -> Self {
            Self(BufWriter::new(writer), 0)
        }

        #[inline]
        pub const fn written_bytes(&self) -> usize {
            self.1
        }

        #[inline]
        pub fn write_bytes(&mut self, val: &[u8]) -> Result<()> {
            self.1 += val.len();
            self.0.write_all(val.as_ref())
        }
    }
}
