mod cast;
mod merge;
mod permute_qk;
mod set_meta;
mod sort;
mod to_llama;

use super::{Content, DataPromise, compile_patterns};
use ggus::{GGmlType, GGufMetaDataValueType};
use regex::Regex;
use std::{collections::HashMap, fmt};

#[allow(unused)]
pub(crate) enum Operator {
    FilterMetaKey(Regex),
    FilterTensorName(Regex),
    Cast(HashMap<String, GGmlType>),
    ToLlama(HashMap<String, String>),
    MergeLinear(bool),
    PermuteQK(bool),
    SortTensors,
    SetMeta(HashMap<String, (GGufMetaDataValueType, Vec<u8>)>),
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::FilterMetaKey(regex) => write!(f, "filter-meta: {}", regex.as_str()),
            Self::FilterTensorName(regex) => write!(f, "filter-tensor: {}", regex.as_str()),
            Self::Cast(types) => write!(f, "cast:{types:?}"),
            Self::ToLlama(extra) => write!(f, "to-llama:{extra:?}"),
            &Self::MergeLinear(val) => {
                if val {
                    write!(f, "merge-linear")
                } else {
                    write!(f, "split-linear")
                }
            }
            Self::PermuteQK(true) => write!(f, "permute-qk"),
            Self::PermuteQK(false) => write!(f, "permute-qk-rev"),
            Self::SortTensors => write!(f, "sort-tensors"),
            Self::SetMeta(map) => write!(f, "set-meta: {} items", map.len()),
        }
    }
}

impl Operator {
    #[inline]
    pub fn filter_meta_key(p: impl AsRef<str>) -> Self {
        Self::FilterMetaKey(compile_patterns(p.as_ref()))
    }

    #[inline]
    pub fn filter_tensor_name(p: impl AsRef<str>) -> Self {
        Self::FilterTensorName(compile_patterns(p.as_ref()))
    }
}

impl Content<'_> {
    pub fn apply(&mut self, op: Operator) {
        use Operator::*;
        match op {
            ToLlama(extra) => self.convert_to_llama(extra),
            FilterMetaKey(r) => self.meta_kvs.retain(|k, _| r.is_match(k)),
            FilterTensorName(r) => self.tensors.retain(|k, _| r.is_match(k)),
            Cast(types) => self.cast(types),
            MergeLinear(ty) => self.merge_linear(ty),
            PermuteQK(ty) => self.permute_qk(ty),
            SortTensors => self.sort_tensors(),
            SetMeta(map) => self.set_meta(map),
        }
    }
}
