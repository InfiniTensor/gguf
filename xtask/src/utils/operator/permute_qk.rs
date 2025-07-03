use super::{
    super::Tensor,
    Content, DataPromise,
    merge::{merge_qkv, split_qkv},
};
use ggus::{DataFuture, GGufMetaError::NotExist, GGufMetaMapExt};
use mem_rearrange::{Rearranging, ndarray_layout::Endian::LittleEndian};
use memmap2::MmapMut;
use regex::Regex;
use std::sync::LazyLock;

impl Content<'_> {
    pub(super) fn permute_qk(&mut self, direction: bool) {
        let nh = self.llm_attention_head_count().unwrap();
        let nkvh = match self.llm_attention_head_count_kv() {
            Ok(val) => val,
            Err(NotExist) => nh,
            Err(e) => panic!("{e:?}"),
        };

        let tensors = std::mem::take(&mut self.tensors);
        for (name, tensor) in tensors {
            static QK_REGEX: LazyLock<Regex> =
                LazyLock::new(|| Regex::new(r"(attn_q|attn_k|attn_qkv)\.(weight|bias)$").unwrap());

            let tensor = if let Some(captures) = QK_REGEX.captures(&name) {
                match &captures[1] {
                    "attn_q" => permute_qk(tensor, nh, !direction),
                    "attn_k" => permute_qk(tensor, nkvh, !direction),
                    "attn_qkv" => {
                        let [q, k, v] = split_qkv(tensor, nh, nkvh);
                        let q = permute_qk(q, nh, !direction);
                        let k = permute_qk(k, nkvh, !direction);
                        merge_qkv([Some(q), Some(k), Some(v)]).1
                    }
                    _ => unreachable!(),
                }
            } else {
                tensor
            };
            self.tensors.insert(name, tensor);
        }
    }
}

fn permute_qk(tensor: Tensor, nh: usize, rev: bool) -> Tensor {
    let Tensor { ty, shape, data } = tensor;
    let [c, r] = match &*shape {
        &[r] => [1, r],
        &[c, r] => [c, r],
        [..] => todo!(),
    };
    let c = ty.size().elements_to_bytes(&[c]);
    let r = r as usize;

    let tiles = if rev {
        [2, r / nh / 2, nh]
    } else {
        [r / nh / 2, 2, nh]
    };

    type Layout = mem_rearrange::ndarray_layout::ArrayLayout<4>;
    let src = Layout::new_contiguous(&[c, r], LittleEndian, 1)
        .tile_le(1, &tiles)
        .transpose(&[2, 1]);
    let dst = Layout::new_contiguous(src.shape(), LittleEndian, 1);
    let rearrange = Rearranging::new(&dst, &src, 1).unwrap();

    let data = DataPromise::lazy(move || {
        let mut ans = MmapMut::map_anon(c * r).unwrap();
        unsafe { rearrange.launch(ans.as_mut_ptr(), data.get().as_ptr()) };
        ans
    });
    Tensor { ty, shape, data }
}
