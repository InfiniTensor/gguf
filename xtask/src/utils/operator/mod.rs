﻿mod cast;
mod merge;
mod set_meta;
mod sort;
mod to_llama;

use super::{compile_patterns, Content, DataPromise};
use ggus::{GGmlType, GGufMetaDataValueType};
use regex::Regex;
use std::{collections::HashMap, fmt};

#[allow(unused)]
pub(crate) enum Operator {
    ToLlama(Option<String>),
    FilterMetaKey(Regex),
    FilterTensorName(Regex),
    Cast(HashMap<String, GGmlType>),
    MergeLinear(bool),
    SetMeta(HashMap<String, (GGufMetaDataValueType, Vec<u8>)>),
    SortTensors,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::ToLlama(_extra) => write!(f, "to-llama"),
            Self::FilterMetaKey(regex) => write!(f, "filter-meta: {}", regex.as_str()),
            Self::FilterTensorName(regex) => write!(f, "filter-tensor: {}", regex.as_str()),
            &Self::Cast { .. } => write!(f, "cast"),
            &Self::MergeLinear(val) => {
                if val {
                    write!(f, "merge-linear")
                } else {
                    write!(f, "split-linear")
                }
            }
            Self::SetMeta(map) => {
                write!(f, "set-meta: {} items", map.len())
            }
            Self::SortTensors => write!(f, "sort-tensors"),
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
            SetMeta(map) => self.set_meta(map),
            SortTensors => self.sort_tensors(),
        }
    }
}
