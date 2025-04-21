use super::{_32, DeltaMin, min_max};
use crate::{DataBlock, Quantize};
use std::iter::zip;

#[repr(C)]
pub struct Q5_1 {
    pub delta_min: DeltaMin,
    pub qh: [u8; _32 / 8],
    pub ql: [u8; _32 / 2],
}

impl_data_block! {
    Q5_1 = crate::types::Q5_1;
    Self {
        delta_min: DeltaMin::ZERO,
        qh: [0; _32 / 8],
        ql: [0; _32 / 2],
    }
}

impl Quantize<f32, _32> for Q5_1 {
    fn quantize(data: &[f32; _32]) -> Self {
        // 验证块大小是否正确，需要对常量进行断言
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _32)
        }

        let (min, max) = min_max(data);
        if min == max {
            return Self {
                delta_min: DeltaMin::no_delta(min),
                qh: [0; _32 / 8],
                ql: [0; _32 / 2],
            };
        }

        let delta = (max - min) / ((1 << 5) - 1) as f32;
        let recip = delta.recip();
        let f = |x| (((x - min) * recip + 0.5) as u8).min(31);

        let (l, h) = data.split_at(_32 / 2);
        let mut qh = 0;
        let mut ql = [0u8; _32 / 2];
        for (i, (&l, &h)) in zip(l, h).enumerate() {
            let l = f(l);
            let h = f(h);
            qh |= ((l as u32 >> 4) & 1) << i;
            qh |= ((h as u32 >> 4) & 1) << (i + _32 / 2);
            ql[i] = ((h & 0xf) << 4) | (l & 0xf);
        }

        Self {
            delta_min: DeltaMin::new(delta, min),
            qh: qh.to_le_bytes(),
            ql,
        }
    }

    fn dequantize(&self) -> [f32; _32] {
        let (delta, min) = self.delta_min.to_f32();
        let qh = u32::from_le_bytes(self.qh);
        let f = |l, h| (l | (h as u8 & 0x10)) as f32 * delta + min;

        let mut ans = [0.; _32];
        let (l, h) = ans.split_at_mut(_32 / 2);
        #[rustfmt::skip]
        for (i, x) in self.ql.iter().enumerate() {
            l[i] = f(x & 0xf, (qh >>  i               ) << 4);
            h[i] = f(x >>  4,  qh >> (i + _32 / 2 - 4)      );
        };
        ans
    }
}

#[test]
fn test_q5_1() {
    crate::test_utils::test::<32, Q5_1>(2e-2, 0.);
}
