use super::{_256, f16};
use crate::{DataBlock, Quantize};

/// IQ4XS 量化结构体
#[repr(C)]
pub struct IQ4XS {
    /// 全局缩放因子
    pub delta: f16,
    /// 高位缩放因子
    pub scales_h: u16,
    /// 低位缩放因子
    pub scales_l: [u8; _256 / 64],
    /// 量化值
    pub qs: [u16; _256 / 2],
}

impl_data_block! {
    IQ4XS = crate::types::IQ4XS;
    Self {
        delta: f16::ZERO,
        scales_h: 0,
        scales_l: [0; _256 / 64],
        qs: [0; _256 / 2],
    }
}

impl Quantize<f32, _256> for IQ4XS {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
