use super::{_256, DeltaMin};
use crate::{DataBlock, Quantize};

/// Q4K 量化结构体
#[repr(C)]
pub struct Q4K {
    /// 全局缩放因子和最小值
    pub delta_min: DeltaMin,
    /// 局部缩放因子
    pub scales: [u8; 12],
    /// 量化值
    pub qs: [u8; _256 / 2],
}

impl_data_block! {
    Q4K = crate::types::Q4K;
    Self {
        delta_min: DeltaMin::ZERO,
        scales: [0; 12],
        qs: [0; _256 / 2],
    }
}

impl Quantize<f32, _256> for Q4K {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
