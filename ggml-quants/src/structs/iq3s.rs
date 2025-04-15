use super::{_256, f16};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ3S {
    pub delta: f16,
    pub qs: [u8; _256 / 4],
    pub qh: [u8; _256 / 32],
    pub signs: [u8; _256 / 8],
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
