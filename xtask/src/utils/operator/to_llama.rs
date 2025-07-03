use super::{
    super::{DataPromise, MetaValue, Tensor},
    Content, Operator,
};
use ggus::{
    DataFuture, GGmlType, GGufMetaError, GGufMetaMapExt,
    ggml_quants::{bf16, f16},
};
use log::warn;
use memmap2::MmapMut;
use regex::Regex;
use std::{alloc::Layout, collections::HashMap, ops::MulAssign, sync::LazyLock};

impl Operator {
    #[inline]
    pub fn to_llama(extra: &str) -> Self {
        static REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\w+)=(\w+)").unwrap());
        Self::ToLlama(
            REGEX
                .captures_iter(extra)
                .map(|captures| (captures[1].into(), captures[2].into()))
                .collect(),
        )
    }
}

impl Content<'_> {
    pub(super) fn convert_to_llama(&mut self, extra: HashMap<String, String>) {
        match self.general_architecture() {
            Ok("llama") => {}
            Ok("minicpm") => from_minicpm(self, extra),
            Ok(arch) => todo!("Unsupported architecture: {arch:?}"),
            Err(e) => panic!("Faild to read general architecture: {e:?}"),
        }
    }
}

fn from_minicpm(content: &mut Content, extra: HashMap<String, String>) {
    const ERR_MSG: &str =
        "convert from minicpm requires extra args \"embd_scale=<num> res_scale=<num>\"";
    const EMBD_SCALE: &str = "minicpm.embedding_scale";
    const RES_SCALE: &str = "minicpm.residual_scale";
    const LOGIT_SCALE: &str = "minicpm.logit_scale";

    let nblk = content.llm_block_count().expect("Missing llm_block_count");
    let mut embd_scale = match content.get_f32(EMBD_SCALE) {
        Ok(v) => Some(v as f64),
        Err(GGufMetaError::NotExist) => None,
        Err(e) => panic!("Failed to read {EMBD_SCALE}: {e:?}"),
    };
    let mut res_scale = match content.get_f32(RES_SCALE) {
        Ok(v) => Some(v as f64),
        Err(GGufMetaError::NotExist) => None,
        Err(e) => panic!("Failed to read {RES_SCALE}: {e:?}"),
    };

    for (k, v) in extra {
        match k.as_str() {
            "embd_scale" => embd_scale = Some(v.parse().expect(ERR_MSG)),
            "res_scale" => res_scale = Some(v.parse().expect(ERR_MSG)),
            _ => {}
        }
    }
    let embd_scale = embd_scale.expect(ERR_MSG);
    let mut res_scale = res_scale.expect(ERR_MSG);
    let common_res_scale = 1.4 / (nblk as f64).sqrt();
    if (res_scale - 1.4).abs() < 10. * f64::EPSILON {
        res_scale = common_res_scale;
        warn!("residual scale in meta is 1.4, use {res_scale:.3} = 1.4 / √{nblk}")
    } else if (res_scale - common_res_scale).abs() >= 10. * f64::EPSILON {
        warn!("residual scale in meta is neither 1.4 nor 1.4 / √nblk")
    }

    for (name, tensor) in content.tensors.iter_mut() {
        static BLK_TENSOR_REGEX: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"^blk\.(\d+)\.(\w+)\.weight$").unwrap());
        if name == "token_embd.weight" {
            scale_tensor(tensor, embd_scale);
        } else if let Some(captures) = BLK_TENSOR_REGEX.captures(name) {
            match &captures[2] {
                "attn_output" | "ffn_down" => scale_tensor(tensor, res_scale),
                _ => {}
            }
        }
    }

    set_arch(content, "minicpm", "llama", |k| {
        k != EMBD_SCALE && k != RES_SCALE && k != LOGIT_SCALE
    })
}

fn set_arch(content: &mut Content, old: &str, new: &str, mut f: impl FnMut(&str) -> bool) {
    let old = format!("{old}.");
    for (k, v) in std::mem::take(&mut content.meta_kvs) {
        if k == "general.architecture" {
            content.meta_kvs.insert(k, MetaValue::string(new));
        } else if f(&k) {
            let k = match k.strip_prefix(&old) {
                Some(body) => format!("{new}.{body}").into(),
                None => k,
            };
            content.meta_kvs.insert(k, v);
        }
    }
}

fn scale_tensor(tensor: &mut Tensor, scale: f64) {
    let data = tensor.data.clone();
    tensor.data = match tensor.ty {
        GGmlType::F64 => DataPromise::lazy(move || scale_data(data.get(), scale)),
        GGmlType::F32 => DataPromise::lazy(move || scale_data(data.get(), scale as f32)),
        GGmlType::F16 => DataPromise::lazy(move || scale_data(data.get(), f16::from_f64(scale))),
        GGmlType::BF16 => DataPromise::lazy(move || scale_data(data.get(), bf16::from_f64(scale))),
        ty => todo!("unsupported tensor type: {ty:?}"),
    };
}

fn scale_data<T: MulAssign + Clone>(data: &[u8], scale: T) -> MmapMut {
    assert_eq!(data.len() % size_of::<T>(), 0);
    let len = data.len() / size_of::<T>();

    let mut ans = MmapMut::map_anon(Layout::array::<T>(len).unwrap().size()).unwrap();
    ans.copy_from_slice(data);

    let (&mut [], data, &mut []) = (unsafe { ans.align_to_mut::<T>() }) else {
        panic!("data not aligned")
    };
    for x in data {
        *x *= scale.clone();
    }

    ans
}
