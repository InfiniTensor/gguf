use super::{_32, max_abs};
use crate::{DataBlock, Quantize};
use half::f16;

#[repr(C)]
pub struct Q8_0 {
    delta: f16,
    quants: [i8; _32],
}

impl_data_block! {
    Q8_0 = crate::types::Q8_0;
    Self {
        delta: f16::ZERO,
        quants: [0; _32],
    }
}

impl Quantize<f32, _32> for Q8_0 {
    fn quantize(data: &[f32; _32]) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _32)
        }

        let amax = max_abs(data);
        if amax == 0. {
            return Self::ZEROS;
        }

        let delta = amax / i8::MAX as f32;
        let recip = delta.recip();
        Self {
            delta: f16::from_f32(delta),
            quants: data.map(|x| (x * recip).round() as _),
        }
    }

    #[inline]
    fn dequantize(&self) -> [f32; _32] {
        let delta = self.delta.to_f32();
        self.quants.map(|x| x as f32 * delta)
    }
}

#[test]
fn test_q8_0() {
    crate::test_utils::test::<32, Q8_0>(4.2e-3, 0.);
}
