use super::{_256, max_by_abs};
use crate::{DataBlock, Quantize};
use half::f16;
use std::iter::zip;

/// Q8K 量化结构体
#[repr(C)]
pub struct Q8K {
    /// 缩放因子
    pub delta: f16,
    /// 量化值
    pub quants: [i8; _256],
    /// 量化和
    pub sums: [i16; _256 / 16],
}

impl_data_block! {
    Q8K = crate::types::Q8K;
    Self {
        delta: f16::ZERO,
        quants: [0; _256],
        sums: [0; _256 / 16],
    }
}

impl Quantize<f32, _256> for Q8K {
    fn quantize(data: &[f32; _256]) -> Self {
        // 验证块大小是否正确，需要对常量进行断言
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _256)
        }

        let max = max_by_abs(data);
        if max == 0. {
            return Self::ZEROS;
        }

        let delta = max / -127.;
        let recip = delta.recip();

        let mut quants = [0; _256];
        let mut sums = [0; _256 / 16];
        for (i, (y, &x)) in zip(&mut quants, data).enumerate() {
            *y = (x * recip).round().min(127.) as i8;
            sums[i / 16] += *y as i16;
        }

        Self {
            delta: f16::from_f32(delta),
            quants,
            sums,
        }
    }

    #[inline]
    fn dequantize(&self) -> [f32; _256] {
        let delta = self.delta.to_f32();
        self.quants.map(|x| x as f32 * delta)
    }
}

#[test]
fn test_q8_k() {
    crate::test_utils::test::<256, Q8K>(4.5e-3, 0.);
}
