use super::{_256, DeltaMin};
use crate::{DataBlock, Quantize};

/// Q2K 量化结构体
#[repr(C)]
pub struct Q2K {
    /// 局部缩放因子
    pub scales: [u8; _256 / 16],
    /// 量化值
    pub qs: [u8; _256 / 4],
    /// 全局缩放因子和最小值
    pub delta_min: DeltaMin,
}

impl_data_block! {
    Q2K = crate::types::Q2K;
    Self {
        scales: [0; _256 / 16],
        qs: [0; _256 / 4],
        delta_min: DeltaMin::ZERO,
    }
}

impl Quantize<f32, _256> for Q2K {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
