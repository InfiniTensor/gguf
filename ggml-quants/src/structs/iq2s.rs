use super::{_256, f16};
use crate::{DataBlock, Quantize};

/// IQ2S 量化结构体
#[repr(C)]
pub struct IQ2S {
    /// 全局缩放因子
    pub delta: f16,
    /// 低位量化值
    pub qs: [u8; _256 / 4],
    /// 高位量化值
    pub qh: [u8; _256 / 32],
    /// 局部缩放因子
    pub scales: [u8; _256 / 32],
}

impl_data_block! {
    IQ2S = crate::types::IQ2S;
    Self {
        delta: f16::ZERO,
        qs: [0; _256 / 4],
        qh: [0; _256 / 32],
        scales: [0; _256 / 32],
    }
}

impl Quantize<f32, _256> for IQ2S {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
