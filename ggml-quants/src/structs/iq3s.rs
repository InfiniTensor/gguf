use super::{_256, f16};
use crate::{DataBlock, Quantize};

/// IQ3S 量化结构体
#[repr(C)]
pub struct IQ3S {
    /// 全局缩放因子
    pub delta: f16,
    /// 低位量化值
    pub qs: [u8; _256 / 4],
    /// 高位量化值
    pub qh: [u8; _256 / 32],
    /// 符号位
    pub signs: [u8; _256 / 8],
    /// 局部缩放因子
    pub scales: [u8; _256 / 64],
}

impl_data_block! {
    IQ3S = crate::types::IQ3S;
    Self {
        delta: f16::ZERO,
        qs: [0; _256 / 4],
        qh: [0; _256 / 32],
        signs: [0; _256 / 8],
        scales: [0; _256 / 64],
    }
}

impl Quantize<f32, _256> for IQ3S {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
