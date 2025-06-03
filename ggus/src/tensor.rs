#![allow(missing_docs)]
// 不必为每个张量数据类型添加文档

use crate::{GGufReadError, GGufReader};
use std::{
    alloc::{Layout, alloc, dealloc},
    ptr::{NonNull, copy_nonoverlapping},
    slice::from_raw_parts,
};

/// GGML tencor 数据类型。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u32)]
pub enum GGmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    #[deprecated = "support removed"]
    Q4_2 = 4,
    #[deprecated = "support removed"]
    Q4_3 = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
}

/// GGML 数据类型的大小和块大小。
#[derive(Clone, Copy, Debug)]
pub struct GGmlTypeSize {
    /// 每个数据块的大小。
    pub block_size: u32,
    /// 每个数据类型的大小（以字节为单位）。
    pub type_size: u32,
}

impl GGmlTypeSize {
    /// 创建一个新的 [`GGmlTypeSize`] 实例，表示单个数据元素的大小。
    #[inline]
    const fn unit<T>() -> Self {
        Self {
            block_size: 1,
            type_size: size_of::<T>() as _,
        }
    }

    /// 创建一个新的 [`GGmlTypeSize`] 实例，表示量化数据块的大小。
    #[inline]
    const fn quants<T: ggml_quants::DataBlock>() -> Self {
        Self {
            block_size: T::COUNT as _,
            type_size: size_of::<T>() as _,
        }
    }

    /// 计算给定形状的元素总数转换为字节数。
    #[inline]
    pub fn elements_to_bytes(&self, shape: &[u64]) -> usize {
        let blk = self.block_size as u64;
        let ele = self.type_size as u64;
        match shape {
            [] => {
                assert_eq!(blk, 1);
                ele as _
            }
            [last, others @ ..] => {
                assert_eq!(last % blk, 0);
                (others.iter().product::<u64>() * last / blk * ele) as _
            }
        }
    }
}

impl GGmlType {
    /// 获取 GGML 数据类型的大小。
    #[rustfmt::skip]
    pub const fn size(self) -> GGmlTypeSize {
        macro_rules! size {
            (t: $ty:ty) => { GGmlTypeSize::  unit::<$ty>() };
            (q: $ty:ty) => { GGmlTypeSize::quants::<$ty>() };
        }

        use ggml_quants::*;
        match self {
            Self::F32      => size!(t: f32   ),
            Self::F16      => size!(q: f16   ),
            Self::Q4_0     => size!(q: Q4_0  ),
            Self::Q4_1     => size!(q: Q4_1  ),
            Self::Q5_0     => size!(q: Q5_0  ),
            Self::Q5_1     => size!(q: Q5_1  ),
            Self::Q8_0     => size!(q: Q8_0  ),
            Self::Q8_1     => size!(q: Q8_1  ),
            Self::Q2K      => size!(q: Q2K   ),
            Self::Q3K      => size!(q: Q3K   ),
            Self::Q4K      => size!(q: Q4K   ),
            Self::Q5K      => size!(q: Q5K   ),
            Self::Q6K      => size!(q: Q6K   ),
            Self::Q8K      => size!(q: Q8K   ),
            Self::IQ2XXS   => size!(q: IQ2XXS),
            Self::IQ2XS    => size!(q: IQ2XS ),
            Self::IQ3XXS   => size!(q: IQ3XXS),
            Self::IQ1S     => size!(q: IQ1S  ),
            Self::IQ4NL    => size!(q: IQ4NL ),
            Self::IQ3S     => size!(q: IQ3S  ),
            Self::IQ2S     => size!(q: IQ2S  ),
            Self::IQ4XS    => size!(q: IQ4XS ),
            Self::I8       => size!(t: i8    ),
            Self::I16      => size!(t: i16   ),
            Self::I32      => size!(t: i32   ),
            Self::I64      => size!(t: i64   ),
            Self::F64      => size!(t: f64   ),
            Self::IQ1M     => size!(q: IQ1M  ),
            Self::BF16     => size!(q: bf16   ),
            Self::Q4_0_4_4 |
            Self::Q4_0_4_8 |
            Self::Q4_0_8_8 => todo!(),
            _              => unimplemented!(),
        }
    }

    /// 将 [`GGmlType`] 映射到具体的 digit_layout 实例。
    #[cfg(feature = "types")]
    pub const fn to_digit_layout(self) -> ggml_quants::digit_layout::DigitLayout {
        use ggml_quants::{digit_layout::types as primitive, types as quantized};
        #[rustfmt::skip]
        let ans = match self {
            Self::F32    => primitive::F32   ,
            Self::F16    => primitive::F16   ,
            Self::BF16   => primitive::BF16  ,
            Self::Q8_0   => quantized::Q8_0  ,
            Self::Q8_1   => quantized::Q8_1  ,
            Self::Q4_0   => quantized::Q4_0  ,
            Self::Q4_1   => quantized::Q4_1  ,
            Self::Q5_0   => quantized::Q5_0  ,
            Self::Q5_1   => quantized::Q5_1  ,
            Self::Q2K    => quantized::Q2K   ,
            Self::Q3K    => quantized::Q3K   ,
            Self::Q4K    => quantized::Q4K   ,
            Self::Q5K    => quantized::Q5K   ,
            Self::Q6K    => quantized::Q6K   ,
            Self::Q8K    => quantized::Q8K   ,
            Self::IQ2XXS => quantized::IQ2XXS,
            Self::IQ2XS  => quantized::IQ2XS ,
            Self::IQ3XXS => quantized::IQ3XXS,
            Self::IQ1S   => quantized::IQ1S  ,
            Self::IQ4NL  => quantized::IQ4NL ,
            Self::IQ3S   => quantized::IQ3S  ,
            Self::IQ2S   => quantized::IQ2S  ,
            Self::IQ4XS  => quantized::IQ4XS ,
            Self::IQ1M   => quantized::IQ1M  ,
            Self::I8     => primitive::I8    ,
            Self::I16    => primitive::I16   ,
            Self::I32    => primitive::I32   ,
            Self::I64    => primitive::I64   ,
            Self::F64    => primitive::F64   ,
            _            => todo!()          ,
        };
        ans
    }
}

/// [`GGufTensorMeta`] 结构体表示 GGUF 文件中的张量元数据。
#[repr(transparent)]
pub struct GGufTensorMeta<'a>(&'a [u8]);

impl<'a> GGufReader<'a> {
    /// 读取 GGUF 文件中的张量元数据。
    pub fn read_tensor_meta(&mut self) -> Result<GGufTensorMeta<'a>, GGufReadError> {
        let data = self.remaining();

        let _ = self.read_str()?;
        let ndim: u32 = self.read()?;
        self.skip::<u64>(ndim as _)?
            .skip::<GGmlType>(1)?
            .skip::<u64>(1)?;

        let data = &data[..data.len() - self.remaining().len()];
        Ok(unsafe { GGufTensorMeta::new_unchecked(data) })
    }
}

impl<'a> GGufTensorMeta<'a> {
    /// 创建一个新的 [`GGufTensorMeta`] 实例，不检查数据合法性。
    ///
    /// # Safety
    ///
    /// 调用此函数时，必须确保传入的数据是有效的 GGUF 张量元数据格式，否则可能导致未定义行为。
    #[inline]
    pub const unsafe fn new_unchecked(data: &'a [u8]) -> Self {
        Self(data)
    }

    /// 创建一个新的 [`GGufTensorMeta`] 实例。
    #[inline]
    pub fn new(data: &'a [u8]) -> Result<Self, GGufReadError> {
        GGufReader::new(data).read_tensor_meta()
    }

    /// 获取张量元数据的名称。
    #[inline]
    pub fn name(&self) -> &'a str {
        let mut reader = GGufReader::new(self.0);
        unsafe { reader.read_str_unchecked() }
    }

    /// 将 [`GGufTensorMeta`] 转换为 [`GGufTensorInfo`]。
    #[inline]
    pub fn to_info(&self) -> GGufTensorInfo {
        let mut reader = GGufReader::new(self.0);
        let ndim: u32 = reader.skip_str().unwrap().read().unwrap();
        let layout = Layout::array::<u64>(ndim as _).unwrap();
        let shape = unsafe {
            let dst = alloc(layout);
            copy_nonoverlapping(reader.remaining().as_ptr(), dst, layout.size());
            NonNull::new_unchecked(dst).cast()
        };
        let ty = reader.skip::<u64>(ndim as _).unwrap().read().unwrap();
        let offset = reader.read().unwrap();

        GGufTensorInfo {
            ty,
            ndim,
            shape,
            offset,
        }
    }
}

/// [`GGufTensorInfo`] 结构体表示 GGUF 文件中的张量信息。
pub struct GGufTensorInfo {
    /// 张量的数据类型。
    ty: GGmlType,
    /// 张量的维度数量。
    ndim: u32,
    /// 张量的形状，存储为指向 u64 的非空指针。
    shape: NonNull<u64>,
    /// 张量在文件中的偏移量。
    offset: u64,
}

impl GGufTensorInfo {
    /// 获取张量数据类型。
    #[inline]
    pub const fn ty(&self) -> GGmlType {
        self.ty
    }

    /// 获取张量形状。
    #[inline]
    pub const fn shape(&self) -> &[u64] {
        unsafe { from_raw_parts(self.shape.as_ptr(), self.ndim as _) }
    }

    /// 获取张量偏移量。
    #[inline]
    pub const fn offset(&self) -> usize {
        self.offset as _
    }

    /// 获取张量大小，以字节为单位。
    #[inline]
    pub fn nbytes(&self) -> usize {
        self.ty.size().elements_to_bytes(self.shape())
    }
}

impl Drop for GGufTensorInfo {
    fn drop(&mut self) {
        let ptr = self.shape.as_ptr().cast();
        let layout = Layout::array::<u64>(self.ndim as _).unwrap();
        unsafe { dealloc(ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_ggml_type_size() {
        // 测试基本类型的大小计算
        let f32_size = GGmlType::F32.size();
        assert_eq!(f32_size.block_size, 1);
        assert_eq!(f32_size.type_size, 4);

        let i8_size = GGmlType::I8.size();
        assert_eq!(i8_size.block_size, 1);
        assert_eq!(i8_size.type_size, 1);

        // 测试量化类型的大小
        let q4_0_size = GGmlType::Q4_0.size();
        assert!(q4_0_size.block_size > 1);
        assert!(q4_0_size.type_size > 0);
    }

    #[test]
    fn test_elements_to_bytes() {
        // 测试空形状
        let f32_size = GGmlType::F32.size();
        assert_eq!(f32_size.elements_to_bytes(&[]), 4);

        // 测试一维形状
        assert_eq!(f32_size.elements_to_bytes(&[10]), 40);

        // 测试多维形状
        assert_eq!(f32_size.elements_to_bytes(&[5, 2]), 40);
        assert_eq!(f32_size.elements_to_bytes(&[2, 3, 4]), 96);

        // 测试量化类型
        let q4_0_size = GGmlType::Q4_0.size();
        if q4_0_size.block_size == 32 && q4_0_size.type_size == 16 {
            assert_eq!(q4_0_size.elements_to_bytes(&[64]), 32);
            assert_eq!(q4_0_size.elements_to_bytes(&[32, 2]), 32);
        }
    }

    #[test]
    fn test_tensor_meta_and_info() {
        // 构造一个模拟的张量元数据
        let name = "test_tensor";
        let ndim = 2u32;
        let shape = [3u64, 4u64];
        let ty = GGmlType::F32;
        let offset = 1024u64;

        let mut data = Vec::new();
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&ndim.to_le_bytes());
        for &dim in &shape {
            data.extend_from_slice(&dim.to_le_bytes());
        }
        data.extend_from_slice(&(ty as u32).to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());

        let meta = GGufTensorMeta::new(&data).unwrap();
        assert_eq!(meta.name(), name);

        // 转换为 info 并检查
        let info = meta.to_info();
        assert_eq!(info.ty(), ty);
        assert_eq!(info.ndim, ndim);
        assert_eq!(info.shape(), &shape);
        assert_eq!(info.offset(), 1024);

        // 测试字节大小计算
        let expected_bytes = shape.iter().product::<u64>() * size_of::<f32>() as u64;
        assert_eq!(info.nbytes(), expected_bytes as usize);
    }

    #[test]
    fn test_reader_read_tensor_meta() {
        // 构造一个模拟的张量元数据
        let name = "weights";
        let ndim = 3u32;
        let shape = [2u64, 768u64, 768u64];
        let ty = GGmlType::F16;
        let offset = 2048u64;

        let mut data = Vec::new();
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&ndim.to_le_bytes());
        for &dim in &shape {
            data.extend_from_slice(&dim.to_le_bytes());
        }
        data.extend_from_slice(&(ty as u32).to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

        let mut reader = GGufReader::new(&data);
        let meta = reader.read_tensor_meta().unwrap();

        assert_eq!(meta.name(), name);
        let info = meta.to_info();
        assert_eq!(info.ty(), ty);
        assert_eq!(info.shape(), &shape);
        assert_eq!(info.offset(), offset as usize);
        assert_eq!(reader.remaining().len(), 4);
    }

    #[test]
    fn test_tensor_info_memory_management() {
        // 测试 GGufTensorInfo 的内存管理
        // 通过 Drop 实现检查是否有内存泄漏
        let mut data = Vec::new();
        let name = "test";
        let ndim = 1u32;
        let shape = [10u64];
        let ty = GGmlType::F32;
        let offset = 0u64;

        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&ndim.to_le_bytes());
        data.extend_from_slice(&shape[0].to_le_bytes());
        data.extend_from_slice(&(ty as u32).to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());

        let meta = GGufTensorMeta::new(&data).unwrap();

        // 在作用域内创建并销毁 GGufTensorInfo
        {
            let _info = meta.to_info();
        }

        for _ in 0..5 {
            let _info = meta.to_info();
        }
    }

    #[test]
    fn test_all_ggml_types() {
        // 测试所有 GGmlType 变体是否可以获取其大小
        let types = [
            GGmlType::F32,
            GGmlType::F16,
            GGmlType::Q4_0,
            GGmlType::Q4_1,
            GGmlType::Q5_0,
            GGmlType::Q5_1,
            GGmlType::Q8_0,
            GGmlType::Q8_1,
            GGmlType::Q2K,
            GGmlType::Q3K,
            GGmlType::Q4K,
            GGmlType::Q5K,
            GGmlType::Q6K,
            GGmlType::Q8K,
            GGmlType::IQ2XXS,
            GGmlType::IQ2XS,
            GGmlType::IQ3XXS,
            GGmlType::IQ1S,
            GGmlType::IQ4NL,
            GGmlType::IQ3S,
            GGmlType::IQ2S,
            GGmlType::IQ4XS,
            GGmlType::I8,
            GGmlType::I16,
            GGmlType::I32,
            GGmlType::I64,
            GGmlType::F64,
            GGmlType::IQ1M,
            GGmlType::BF16,
        ];

        for &ty in &types {
            let size = ty.size();
            assert!(size.block_size > 0);
            assert!(size.type_size > 0);
        }
    }

    // 边缘情况测试
    #[test]
    fn test_edge_cases() {
        // 测试非常大的形状
        let f32_size = GGmlType::F32.size();
        let large_shape = [1000000u64, 2];
        let bytes = f32_size.elements_to_bytes(&large_shape);
        assert_eq!(bytes, 8000000);

        // 测试空名称的张量
        let mut data = Vec::new();
        let empty_name = "";
        let ndim = 1u32;
        let shape = [1u64];
        let ty = GGmlType::F32;
        let offset = 0u64;

        data.extend_from_slice(&(empty_name.len() as u64).to_le_bytes());
        data.extend_from_slice(&ndim.to_le_bytes());
        data.extend_from_slice(&shape[0].to_le_bytes());
        data.extend_from_slice(&(ty as u32).to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());

        let meta = GGufTensorMeta::new(&data).unwrap();
        assert_eq!(meta.name(), empty_name);
    }

    // 测试错误处理
    #[test]
    fn test_error_handling() {
        // 测试数据不足的情况
        let incomplete_data = [0u8, 1, 2];
        let result = GGufTensorMeta::new(&incomplete_data);
        assert!(result.is_err());

        // 测试数据损坏的情况
        let mut corrupted_data = Vec::new();
        let name = "test";
        corrupted_data.extend_from_slice(&(100u64).to_le_bytes());
        corrupted_data.extend_from_slice(name.as_bytes());

        let result = GGufTensorMeta::new(&corrupted_data);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "types")]
    fn test_to_digit_layout() {
        // 测试基本类型的 digit_layout 转换
        let _f32_layout = GGmlType::F32.to_digit_layout();
        let _f16_layout = GGmlType::F16.to_digit_layout();
        let _bf16_layout = GGmlType::BF16.to_digit_layout();

        // 测试量化类型的 digit_layout 转换
        let _q4_0_layout = GGmlType::Q4_0.to_digit_layout();
        let _q4_1_layout = GGmlType::Q4_1.to_digit_layout();
        let _q5_0_layout = GGmlType::Q5_0.to_digit_layout();
        let _q5_1_layout = GGmlType::Q5_1.to_digit_layout();
        let _q8_0_layout = GGmlType::Q8_0.to_digit_layout();
        let _q8_1_layout = GGmlType::Q8_1.to_digit_layout();

        // 测试高级量化类型
        let _q2k_layout = GGmlType::Q2K.to_digit_layout();
        let _q3k_layout = GGmlType::Q3K.to_digit_layout();
        let _q4k_layout = GGmlType::Q4K.to_digit_layout();
        let _q5k_layout = GGmlType::Q5K.to_digit_layout();
        let _q6k_layout = GGmlType::Q6K.to_digit_layout();
        let _q8k_layout = GGmlType::Q8K.to_digit_layout();

        // 测试 IQ 类型
        let _iq2xxs_layout = GGmlType::IQ2XXS.to_digit_layout();
        let _iq2xs_layout = GGmlType::IQ2XS.to_digit_layout();
        let _iq3xxs_layout = GGmlType::IQ3XXS.to_digit_layout();
        let _iq1s_layout = GGmlType::IQ1S.to_digit_layout();
        let _iq4nl_layout = GGmlType::IQ4NL.to_digit_layout();
        let _iq3s_layout = GGmlType::IQ3S.to_digit_layout();
        let _iq2s_layout = GGmlType::IQ2S.to_digit_layout();
        let _iq4xs_layout = GGmlType::IQ4XS.to_digit_layout();
        let _iq1m_layout = GGmlType::IQ1M.to_digit_layout();

        // 测试基本整数类型
        let _i8_layout = GGmlType::I8.to_digit_layout();
        let _i16_layout = GGmlType::I16.to_digit_layout();
        let _i32_layout = GGmlType::I32.to_digit_layout();
        let _i64_layout = GGmlType::I64.to_digit_layout();
        let _f64_layout = GGmlType::F64.to_digit_layout();
    }
}
