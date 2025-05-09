#![doc = include_str!("../README.md")]
#![deny(warnings, missing_docs)]

//! 本模块提供了数据块的定义和量化/反量化的实现。
//! 包括对不同数据类型的量化支持，以及并行处理的扩展。

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::slice::{from_raw_parts, from_raw_parts_mut};

/// 数据块定义
pub trait DataBlock: Sized + 'static {
    /// 数据块的唯一标识符（仅在启用 `types` 功能时可用）
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout;

    /// 数据块的元素数量
    const COUNT: usize;

    /// 数据块的全零值
    const ZEROS: Self;
}

/// 宏用于实现 `DataBlock` 特性
macro_rules! impl_data_block {
    ($ty:ty = $id:expr; $zero:expr ) => {
        impl DataBlock for $ty {
            #[cfg(feature = "types")]
            const ID: digit_layout::DigitLayout = $id;
            const COUNT: usize = Self::ID.group_size();
            const ZEROS: Self = $zero;
        }
    };
}

// 为常见数据类型实现 `DataBlock`
use digit_layout::types as ty;
impl_data_block!(u8  = ty::U8 ; 0 );
impl_data_block!(i8  = ty::I8 ; 0 );
impl_data_block!(u16 = ty::U16; 0 );
impl_data_block!(i16 = ty::I16; 0 );
impl_data_block!(u32 = ty::U32; 0 );
impl_data_block!(i32 = ty::I32; 0 );
impl_data_block!(f32 = ty::F32; 0.);
impl_data_block!(u64 = ty::U64; 0 );
impl_data_block!(i64 = ty::I64; 0 );
impl_data_block!(f64 = ty::F64; 0.);

/// 量化和反量化的特性
///
/// # 类型参数
/// - `T`: 数据类型
/// - `N`: 数据块大小
pub trait Quantize<T, const N: usize>: DataBlock {
    /// 将数据量化为当前类型
    fn quantize(data: &[T; N]) -> Self;

    /// 将当前类型的数据反量化为原始数据
    fn dequantize(&self) -> [T; N];
}

/// 为支持 `f16` 的数据块实现量化和反量化
impl<Blk, const N: usize> Quantize<f16, N> for Blk
where
    Blk: Quantize<f32, N>,
{
    #[inline]
    fn quantize(data: &[f16; N]) -> Self {
        Self::quantize(&data.map(f16::to_f32))
    }
    #[inline]
    fn dequantize(&self) -> [f16; N] {
        self.dequantize().map(f16::from_f32)
    }
}

/// 为支持 `bf16` 的数据块实现量化和反量化
impl<Blk, const N: usize> Quantize<bf16, N> for Blk
where
    Blk: Quantize<f32, N>,
{
    #[inline]
    fn quantize(data: &[bf16; N]) -> Self {
        Self::quantize(&data.map(bf16::to_f32))
    }

    #[inline]
    fn dequantize(&self) -> [bf16; N] {
        self.dequantize().map(bf16::from_f32)
    }
}

/// 并行量化和反量化的扩展特性
///
/// # 类型参数
///
/// - `T`: 数据类型
/// - `N`: 数据块大小
pub trait QuantExt<T, const N: usize>: Sized {
    /// 将数据切片量化为目标类型
    fn quantize_slice(dst: &mut [Self], src: &[T]) -> Result<(), QuantizeError>;

    /// 将目标类型的数据切片反量化为原始数据
    fn dequantize_slice(dst: &mut [T], src: &[Self]) -> Result<(), QuantizeError>;
}

/// 量化错误类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizeError {
    /// 数据长度不可整除
    Indivisible,
    /// 数据长度不匹配
    LengthMismatch,
}

/// 为实现 `Quantize` 的数据块提供并行量化和反量化支持
impl<Blk, T, const N: usize> QuantExt<T, N> for Blk
where
    Blk: Quantize<T, N> + Send + Sync,
    T: Send + Sync,
{
    fn quantize_slice(dst: &mut [Self], src: &[T]) -> Result<(), QuantizeError> {
        if src.len() % N != 0 {
            return Err(QuantizeError::Indivisible);
        }
        if dst.len() != src.len() / N {
            return Err(QuantizeError::LengthMismatch);
        }
        let src = unsafe { from_raw_parts(src.as_ptr().cast::<[T; N]>(), dst.len()) };
        dst.into_par_iter()
            .zip(src)
            .for_each(|(dst, src)| *dst = Blk::quantize(src));
        Ok(())
    }

    fn dequantize_slice(dst: &mut [T], src: &[Self]) -> Result<(), QuantizeError> {
        if dst.len() % N != 0 {
            return Err(QuantizeError::Indivisible);
        }
        if src.len() != dst.len() / N {
            return Err(QuantizeError::LengthMismatch);
        }
        let dst = unsafe { from_raw_parts_mut(dst.as_mut_ptr().cast::<[T; N]>(), src.len()) };
        src.into_par_iter()
            .zip(dst)
            .for_each(|(src, dst)| *dst = Blk::dequantize(src));
        Ok(())
    }
}

mod structs;
pub use structs::*;

#[cfg(feature = "types")]
pub extern crate digit_layout;

/// 类型定义模块
#[cfg(feature = "types")]
pub mod types;

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::Quantize;
    use std::fmt;

    /// 测试量化和反量化的工具函数
    ///
    /// # 参数
    ///
    /// - `N`: 数据块大小
    /// - `T`: 数据类型
    /// - `abs`: 绝对误差阈值
    /// - `rel`: 相对误差阈值
    pub fn test<const N: usize, T: Quantize<f32, N>>(abs: f32, rel: f32) {
        use rand::Rng;
        use std::iter::zip;

        let mut data = [0.0f32; N];
        rand::rng().fill(&mut data[..]);

        let quant = T::quantize(&data);
        let dequant = T::dequantize(&quant);

        let mut ec = ErrorCollector::new(abs, rel);
        for (a, b) in zip(data, dequant) {
            ec.push(Diff::new(a, b))
        }
        println!("{ec}");

        for &i in ec.outliers() {
            println!("{} vs {}", data[i], dequant[i]);
        }

        assert!(ec.outliers().is_empty());
    }

    /// 差异计算
    struct Diff {
        /// 绝对误差
        pub abs: f32,
        /// 相对误差
        pub rel: f32,
    }

    impl Diff {
        /// 创建新的差异
        fn new(a: f32, b: f32) -> Self {
            let abs = (a - b).abs();
            let rel = abs / (a.abs() + b.abs() + f32::EPSILON);
            Self { abs, rel }
        }
    }

    /// 误差收集器
    struct ErrorCollector {
        threshold: Diff,
        max_diff: Diff,
        outliers: Vec<usize>,
        count: usize,
    }

    impl ErrorCollector {
        /// 创建新的误差收集器
        fn new(abs: f32, rel: f32) -> Self {
            Self {
                threshold: Diff { abs, rel },
                max_diff: Diff { abs: 0.0, rel: 0.0 },
                outliers: vec![],
                count: 0,
            }
        }

        /// 添加新的差异
        fn push(&mut self, diff: Diff) {
            self.max_diff.abs = f32::max(self.max_diff.abs, diff.abs);
            self.max_diff.rel = f32::max(self.max_diff.rel, diff.rel);

            if diff.abs > self.threshold.abs && diff.rel > self.threshold.rel {
                self.outliers.push(self.count);
            }

            self.count += 1;
        }

        /// 获取异常值索引
        fn outliers(&self) -> &[usize] {
            &self.outliers
        }
    }

    impl fmt::Display for ErrorCollector {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "abs: {:.3e}, rel: {:.3e}, outliers: {}/{}",
                self.max_diff.abs,
                self.max_diff.rel,
                self.outliers.len(),
                self.count,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuantExt, Quantize};
    use half::f16;
    use rand::Rng;

    const N: usize = 32;

    /// 生成指定量化类型的测试代码
    macro_rules! generate_tests {
        ($blk:ty, $tolerance:expr) => {
            #[test]
            fn test_quant_dequant_slice_ok() {
                let mut rng = rand::rng();
                let input: Vec<f32> = (0..(2 * N)).map(|_| rng.random_range(-1.0..1.0)).collect();
                let mut quantized: Vec<$blk> = (0..(input.len() / N))
                    .map(|_| <$blk>::quantize(&[0.0; N]))
                    .collect();
                let mut output = vec![0f32; input.len()];

                <$blk>::quantize_slice(&mut quantized, &input).unwrap();
                <$blk>::dequantize_slice(&mut output, &quantized).unwrap();

                for (a, b) in input.iter().zip(&output) {
                    assert!((a - b).abs() < $tolerance, "{a} vs {b}");
                }
            }

            #[test]
            fn test_quant_slice_indivisible() {
                let mut rng = rand::rng();
                let input: Vec<f32> = (0..(N - 1)).map(|_| rng.random_range(-1.0..1.0)).collect();
                let mut quantized: Vec<$blk> =
                    (0..1).map(|_| <$blk>::quantize(&[0.0; N])).collect();
                let err = <$blk>::quantize_slice(&mut quantized, &input).unwrap_err();
                assert_eq!(err, crate::QuantizeError::Indivisible);
            }

            #[test]
            fn test_quant_slice_length_mismatch() {
                let mut rng = rand::rng();
                let input: Vec<f32> = (0..(2 * N)).map(|_| rng.random_range(-1.0..1.0)).collect();
                let mut quantized: Vec<$blk> =
                    (0..3).map(|_| <$blk>::quantize(&[0.0; N])).collect(); // 应为 2
                let err = <$blk>::quantize_slice(&mut quantized, &input).unwrap_err();
                assert_eq!(err, crate::QuantizeError::LengthMismatch);
            }

            #[test]
            fn test_dequant_slice_indivisible() {
                let mut rng = rand::rng();
                let mut output: Vec<f32> =
                    (0..(N - 1)).map(|_| rng.random_range(-1.0..1.0)).collect();
                let src: Vec<$blk> = (0..1).map(|_| <$blk>::quantize(&[0.0; N])).collect();
                let err = <$blk>::dequantize_slice(&mut output, &src).unwrap_err();
                assert_eq!(err, crate::QuantizeError::Indivisible);
            }

            #[test]
            fn test_dequant_slice_length_mismatch() {
                let mut rng = rand::rng();
                let mut output: Vec<f32> =
                    (0..(2 * N)).map(|_| rng.random_range(-1.0..1.0)).collect();
                let src: Vec<$blk> = (0..3).map(|_| <$blk>::quantize(&[0.0; N])).collect(); // 应为 2
                let err = <$blk>::dequantize_slice(&mut output, &src).unwrap_err();
                assert_eq!(err, crate::QuantizeError::LengthMismatch);
            }

            #[test]
            fn test_block_zeros_and_count() {
                assert_eq!(<$blk>::COUNT, N);
                let dequant: [f32; N] = <$blk>::quantize(&[0.0; N]).dequantize();
                assert!(dequant.iter().all(|&x| x == 0.0));
            }

            #[test]
            fn test_quantize_f16() {
                let mut rng = rand::rng();
                let input: [f16; N] =
                    core::array::from_fn(|_| f16::from_f32(rng.random_range(-1.0..1.0)));
                let blk = <$blk as Quantize<f16, N>>::quantize(&input);
                let dequant = <$blk as Quantize<f16, N>>::dequantize(&blk);
                for (a, b) in input.iter().zip(&dequant) {
                    let diff = (a.to_f32() - b.to_f32()).abs();
                    assert!(diff < $tolerance, "diff = {diff}");
                }
            }

            #[test]
            fn test_quantize_bf16() {
                let mut rng = rand::rng();
                let input: [bf16; N] =
                    core::array::from_fn(|_| bf16::from_f32(rng.random_range(-1.0..1.0)));
                let blk = <$blk as Quantize<bf16, N>>::quantize(&input);
                let dequant = <$blk as Quantize<bf16, N>>::dequantize(&blk);
                for (a, b) in input.iter().zip(&dequant) {
                    let diff = (a.to_f32() - b.to_f32()).abs();
                    assert!(diff < $tolerance, "diff = {diff}");
                }
            }
        };
    }

    // 以 Q8_0 为例生成测试
    generate_tests!(Q8_0, 4.5e-3);
}
