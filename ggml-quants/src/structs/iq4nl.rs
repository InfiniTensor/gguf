use super::{_32, f16};
use crate::{DataBlock, Quantize};

/// IQ4NL 量化结构体
#[repr(C)]
pub struct IQ4NL {
    /// 缩放因子
    pub delta: f16,
    /// 量化值
    pub qs: [u16; _32 / 2],
}

impl_data_block! {
    IQ4NL = crate::types::IQ4NL;
    Self {
        delta: f16::ZERO,
        qs: [0; _32 / 2],
    }
}

impl Quantize<f32, _32> for IQ4NL {
    fn quantize(_data: &[f32; _32]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _32] {
        todo!()
    }
}
