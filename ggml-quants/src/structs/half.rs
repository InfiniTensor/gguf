use super::{_1, bf16, f16};
use crate::{DataBlock, Quantize};

use digit_layout::types as ty;
impl_data_block!( f16 = ty:: F16;  f16::ZERO);
impl_data_block!(bf16 = ty::BF16; bf16::ZERO);

impl Quantize<f32, _1> for f16 {
    #[inline]
    fn quantize(&[data]: &[f32; _1]) -> Self {
        // 验证块大小是否正确，需要对常量进行断言
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _1)
        }
        f16::from_f32(data)
    }
    #[inline]
    fn dequantize(&self) -> [f32; _1] {
        [self.to_f32()]
    }
}

impl Quantize<f32, _1> for bf16 {
    #[inline]
    fn quantize(&[data]: &[f32; _1]) -> Self {
        // 验证块大小是否正确，需要对常量进行断言
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _1)
        }
        bf16::from_f32(data)
    }
    #[inline]
    fn dequantize(&self) -> [f32; _1] {
        [self.to_f32()]
    }
}

#[test]
fn test_f16() {
    crate::test_utils::test::<1, f16>(4e-3, 0.);
}

#[test]
fn test_bf16() {
    crate::test_utils::test::<1, bf16>(8e-3, 0.);
}
