use super::{_256, f16};
use crate::{DataBlock, Quantize};

/// IQ2XS 量化结构体
#[repr(C)]
pub struct IQ2XS {
    /// 缩放因子
    pub delta: f16,
    /// 低位量化值
    pub qs: [u16; _256 / 8],
    /// 高位量化值
    pub qh: [u8; _256 / 32],
}

impl_data_block! {
    IQ2XS = crate::types::IQ2XS;
    Self {
        delta: f16::ZERO,
        qs: [0; _256 / 8],
        qh: [0; _256 / 32],
    }
}

impl Quantize<f32, _256> for IQ2XS {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
