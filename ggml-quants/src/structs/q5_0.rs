use super::{_32, f16, max_by_abs};
use crate::{DataBlock, Quantize};
use std::iter::zip;

#[repr(C)]
pub struct Q5_0 {
    pub delta: f16,
    pub qh: [u8; _32 / 8],
    pub ql: [u8; _32 / 2],
}

impl_data_block! {
    Q5_0 = crate::types::Q5_0;
    Self {
        delta: f16::ZERO,
        qh: [0; _32 / 8],
        ql: [0; _32 / 2],
    }
}

impl Quantize<f32, _32> for Q5_0 {
    fn quantize(data: &[f32; _32]) -> Self {
        // 验证块大小是否正确，需要对常量进行断言
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _32)
        }

        let max = max_by_abs(data);
        if max == 0. {
            return Self::ZEROS;
        }

        let delta = max / -16.;
        let recip = delta.recip();
        let f = |x: f32| ((x * recip + 16.5) as u8).min(31);

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
            delta: f16::from_f32(delta),
            qh: qh.to_le_bytes(),
            ql,
        }
    }

    fn dequantize(&self) -> [f32; _32] {
        let delta = self.delta.to_f32();
        let qh = u32::from_le_bytes(self.qh);
        let f = |l, h| ((l | (h as u8 & 0x10)) as i8 - 16) as f32 * delta;

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
fn test_q5_0() {
    crate::test_utils::test::<32, Q5_0>(4e-2, 0.);
}
