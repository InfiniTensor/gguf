use super::{_256, DeltaMin};
use crate::{DataBlock, Quantize};
use half::f16;

const R_MIN: f32 = -1.0;
const R_DELTA: f32 = 0.1;
const N_STEPS: i32 = 20;

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
    fn quantize(data: &[f32; _256]) -> Self {
        const QK_K: usize = 256;
        const N_MAX: i32 = 15;

        fn nearest_int(val: f32) -> i32 {
            val.round() as i32
        }

        fn make_qkx2_quants(
            x: &[f32],
            weights: &[f32],
            l: &mut [u8],
            the_min: &mut f32,
            l_aux: &mut [u8],
        ) -> f32 {
            let n = x.len();

            let (mut min, max) = x
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, &v| {
                    (acc.0.min(v), acc.1.max(v))
                });

            let mut sum_w = 0.0;
            let mut sum_x = 0.0;
            for i in 0..n {
                let w = weights[i];
                sum_w += w;
                sum_x += w * x[i];
            }

            if min > 0.0 {
                min = 0.0;
            }

            if max == min {
                l.fill(0);
                *the_min = -min;
                return 0.0;
            }

            let mut scale = (max - min) / N_MAX as f32;
            let mut best_mad = f32::MAX;

            for i in 0..n {
                let d = x[i] - min;
                let li = nearest_int(d / scale).clamp(0, N_MAX) as u8;
                l[i] = li;
            }

            for is in 0..=N_STEPS {
                let iscale = (R_MIN + R_DELTA * (is as f32) + N_MAX as f32) / (max - min);
                let mut sum_l = 0.0;
                let mut sum_l2 = 0.0;
                let mut sum_xl = 0.0;

                for i in 0..n {
                    let li = nearest_int(iscale * (x[i] - min)).clamp(0, N_MAX);
                    l_aux[i] = li as u8;
                    let w = weights[i];
                    sum_l += w * li as f32;
                    sum_l2 += w * (li * li) as f32;
                    sum_xl += w * li as f32 * x[i];
                }

                let d_val = sum_w * sum_l2 - sum_l * sum_l;
                if d_val > 0.0 {
                    let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d_val;
                    let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d_val;

                    if this_min > 0.0 {
                        this_min = 0.0;
                        this_scale = sum_xl / sum_l2;
                    }

                    let mut mad = 0.0;
                    for i in 0..n {
                        let diff = this_scale * l_aux[i] as f32 + this_min - x[i];
                        mad += weights[i] * diff * diff;
                    }

                    if mad < best_mad {
                        l[..n].copy_from_slice(&l_aux[..n]);
                        best_mad = mad;
                        scale = this_scale;
                        min = this_min;
                    }
                }
            }
            *the_min = -min;
            scale
        }

        let mut l = [0u8; QK_K];
        let mut l_aux = [0u8; 32];
        let mut weights = [0.0f32; 32];
        let mut mins = [0.0f32; QK_K / 32];
        let mut scales = [0.0f32; QK_K / 32];

        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;

        let mut y = Self::ZEROS;

        for j in 0..(QK_K / 32) {
            let x_chunk = &data[32 * j..32 * (j + 1)];
            let l_chunk = &mut l[32 * j..32 * (j + 1)];

            let mut sum_x2 = 0.0;
            for &val in x_chunk {
                sum_x2 += val * val;
            }
            let av_x = (sum_x2 / 32.0).sqrt();

            for l_w in 0..32 {
                weights[l_w] = av_x + x_chunk[l_w].abs();
            }

            let scale = make_qkx2_quants(x_chunk, &weights, l_chunk, &mut mins[j], &mut l_aux);
            scales[j] = scale;

            if scale > max_scale {
                max_scale = scale;
            }
            if mins[j] > max_min {
                max_min = mins[j];
            }
        }

        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        for j in 0..(QK_K / 32) {
            let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).min(63) as u8;

            if j < 4 {
                y.scales[j] = ls;
                y.scales[j + 4] = lm;
            } else {
                y.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y.scales[j - 4] |= (ls >> 4) << 6;
                y.scales[j] |= (lm >> 4) << 6;
            }
        }

        y.delta_min.delta = f16::from_f32(max_scale / 63.0);
        y.delta_min.min = f16::from_f32(max_min / 63.0);

        for j in 0..(QK_K / 32) {
            let (sc, m) = Self::get_scale_min_k4(j, &y.scales);
            let d = y.delta_min.delta.to_f32() * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = y.delta_min.min.to_f32() * m as f32;

            for ii in 0..32 {
                let val = (data[32 * j + ii] + dm) / d;
                let li = nearest_int(val).clamp(0, 15) as u8;
                l[32 * j + ii] = li;
            }
        }

        let mut q_ptr = 0;
        for j in (0..QK_K).step_by(64) {
            for li in 0..32 {
                y.qs[q_ptr] = l[j + li] | (l[j + li + 32] << 4);
                q_ptr += 1;
            }
        }
        y
    }

    fn dequantize(&self) -> [f32; _256] {
        const QK_K: usize = 256;
        let mut y = [0.0f32; QK_K];
        let d = self.delta_min.delta.to_f32();
        let min = self.delta_min.min.to_f32();

        let mut is = 0;
        for (j_chunk, y_chunk) in y.chunks_mut(64).enumerate() {
            let q_chunk = &self.qs[j_chunk * 32..(j_chunk + 1) * 32];

            let (sc1, m1) = Self::get_scale_min_k4(is, &self.scales);
            let d1 = d * sc1 as f32;
            let m1 = min * m1 as f32;
            is += 1;

            let (sc2, m2) = Self::get_scale_min_k4(is, &self.scales);
            let d2 = d * sc2 as f32;
            let m2 = min * m2 as f32;
            is += 1;

            let (y1, y2) = y_chunk.split_at_mut(32);
            for i in 0..32 {
                y1[i] = d1 * (q_chunk[i] & 0x0F) as f32 - m1;
                y2[i] = d2 * (q_chunk[i] >> 4) as f32 - m2;
            }
        }

        y
    }
}

impl Q4K {
    fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
        if j < 4 {
            (scales[j] & 63, scales[j + 4] & 63)
        } else {
            let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            (d, m)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32_DATA: [f32; 256] = [
        -0.007232666,
        -0.012878418,
        -0.018554688,
        -0.026733398,
        0.02722168,
        0.013061523,
        0.0039978027,
        -0.014831543,
        -0.0045776367,
        0.016723633,
        -0.0031585693,
        -0.017944336,
        -0.026611328,
        0.015136719,
        0.00024414063,
        0.00074768066,
        -0.024658203,
        -0.026000977,
        -0.024536133,
        0.0390625,
        0.03112793,
        0.020019531,
        -0.04736328,
        -0.00033187866,
        0.0077819824,
        0.01586914,
        -0.013793945,
        -0.012329102,
        -0.006652832,
        0.035888672,
        0.025146484,
        0.0053710938,
        0.0027770996,
        0.0023498535,
        -0.061523438,
        0.025146484,
        -0.0154418945,
        0.01928711,
        0.002609253,
        -0.021362305,
        -0.012573242,
        -0.032470703,
        0.018310547,
        0.00090408325,
        -0.03100586,
        0.03149414,
        0.025390625,
        0.011474609,
        0.0095825195,
        0.014770508,
        0.030151367,
        -0.013427734,
        0.021972656,
        0.028198242,
        0.011169434,
        0.004699707,
        -0.048828125,
        -0.0073242188,
        0.00021076202,
        0.020507813,
        -0.023925781,
        0.033691406,
        -0.045898438,
        0.0146484375,
        -0.007537842,
        -0.017456055,
        -0.008117676,
        -0.008911133,
        -0.025756836,
        -0.004119873,
        0.008544922,
        0.007751465,
        -0.021728516,
        0.01550293,
        0.004180908,
        -0.01965332,
        0.0050354004,
        0.007537842,
        0.029296875,
        -0.010192871,
        0.032714844,
        -0.009094238,
        0.0063476563,
        0.0023651123,
        -0.017089844,
        0.024047852,
        0.044433594,
        0.009216309,
        0.024536133,
        -0.0056152344,
        -0.0076904297,
        0.030761719,
        0.02709961,
        -0.007873535,
        0.031982422,
        -0.002090454,
        0.014770508,
        0.014099121,
        -0.022460938,
        -0.0056152344,
        -0.021362305,
        -0.023925781,
        -0.025390625,
        -0.05078125,
        0.020751953,
        -0.060302734,
        0.017089844,
        0.03125,
        0.017578125,
        -0.0011825562,
        0.05859375,
        -0.016113281,
        -0.011108398,
        0.03125,
        0.003829956,
        -0.004486084,
        -0.025512695,
        0.0078125,
        0.05078125,
        0.005218506,
        0.007293701,
        0.010131836,
        0.028076172,
        -0.0077819824,
        0.016601563,
        0.026245117,
        0.008422852,
        -0.0057678223,
        -0.005554199,
        0.0087890625,
        0.03100586,
        -0.0011749268,
        0.002243042,
        0.01953125,
        -0.009216309,
        -0.011291504,
        -0.04321289,
        0.0028839111,
        0.010192871,
        0.009643555,
        -0.008178711,
        -0.045166016,
        -0.014038086,
        -0.014160156,
        0.0072631836,
        0.02734375,
        0.01373291,
        -0.018554688,
        0.00030899048,
        0.0138549805,
        0.013366699,
        -0.012634277,
        0.0079956055,
        -0.00047683716,
        -0.041015625,
        -0.045898438,
        0.014343262,
        0.0029144287,
        -0.0053710938,
        -0.019042969,
        -0.019897461,
        -0.0065307617,
        0.018432617,
        0.002319336,
        -0.012939453,
        0.018188477,
        -0.030761719,
        0.0059509277,
        0.0058288574,
        -0.0046691895,
        -0.007232666,
        -0.0044555664,
        -0.037109375,
        -0.00024032593,
        0.04248047,
        0.028442383,
        0.00491333,
        0.0072631836,
        0.021728516,
        -0.0059814453,
        -0.007507324,
        -0.003112793,
        -0.000541687,
        -0.015258789,
        0.022216797,
        0.01928711,
        0.028076172,
        -0.037109375,
        -0.0004234314,
        0.0079956055,
        0.003250122,
        0.010864258,
        -0.03955078,
        0.0010070801,
        0.010864258,
        0.037109375,
        0.01574707,
        0.020996094,
        0.004211426,
        -0.011108398,
        0.025390625,
        0.0063171387,
        0.007873535,
        -0.007171631,
        0.03173828,
        -0.014343262,
        0.0018539429,
        0.00982666,
        0.012573242,
        -0.010131836,
        -0.030029297,
        0.013916016,
        0.011352539,
        0.010559082,
        0.006011963,
        0.00982666,
        -0.0026245117,
        0.012451172,
        0.012390137,
        0.020751953,
        0.0016860962,
        -0.012817383,
        0.0048828125,
        -0.018920898,
        0.0043945313,
        -0.015991211,
        0.018310547,
        0.006072998,
        -0.025390625,
        -0.0074157715,
        0.020996094,
        -0.002319336,
        -0.010192871,
        -0.010620117,
        0.0039978027,
        0.00680542,
        -0.0034332275,
        0.0034484863,
        0.0087890625,
        0.019042969,
        -0.040039063,
        0.008178711,
        0.0073242188,
        -0.01928711,
        0.0017089844,
        0.0029754639,
        0.01361084,
        0.005706787,
        -0.024047852,
        -0.029052734,
        0.012084961,
        0.010253906,
        0.0068359375,
        0.019042969,
        -0.001953125,
        -0.014831543,
    ];

    const Q4_K_BLOCK: [u8; 144] = [
        92, 8, 236, 19, 171, 175, 162, 191, 177, 191, 155, 189, 249, 88, 150, 160, 167, 166, 5,
        228, 125, 219, 169, 102, 135, 91, 216, 165, 84, 251, 232, 200, 180, 196, 244, 143, 222,
        236, 192, 168, 42, 155, 166, 214, 103, 255, 45, 201, 148, 146, 68, 100, 64, 69, 72, 23,
        161, 9, 151, 177, 151, 119, 236, 84, 109, 180, 119, 118, 66, 139, 223, 136, 139, 133, 164,
        109, 156, 164, 141, 101, 55, 90, 174, 120, 73, 172, 23, 134, 128, 105, 90, 106, 7, 112,
        246, 198, 138, 141, 187, 101, 88, 107, 123, 70, 186, 168, 193, 0, 123, 137, 119, 149, 176,
        104, 234, 191, 75, 140, 233, 150, 125, 121, 169, 182, 158, 165, 200, 234, 10, 182, 178, 91,
        170, 170, 217, 186, 71, 58, 202, 204, 184, 229, 153, 100,
    ];

    #[test]
    fn test_const_data_dequantize() {
        let const_q =
            &unsafe { std::slice::from_raw_parts(Q4_K_BLOCK.as_ptr() as *const Q4K, 1) }[0];
        let dq = const_q.dequantize();
        for (i, (a, b)) in dq.iter().zip(F32_DATA.iter()).enumerate() {
            assert!((a - b).abs() < 1e-2, "mismatch at {}: {} vs {}", i, a, b);
        }
    }

    #[test]
    fn test_const_data_quantize() {
        let const_q =
            &unsafe { std::slice::from_raw_parts(Q4_K_BLOCK.as_ptr() as *const Q4K, 1) }[0];
        let q = Q4K::quantize(&F32_DATA);
        assert!(
            (q.delta_min.delta.to_f32() - const_q.delta_min.delta.to_f32()).abs() < 1e-5,
            "delta_min mismatch, {} vs {}",
            q.delta_min.delta.to_f32(),
            const_q.delta_min.delta.to_f32()
        );
        assert_eq!(q.scales, const_q.scales, "scales mismatch");
        assert_eq!(q.qs, const_q.qs, "qs mismatch");
    }
}
