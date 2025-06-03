#![doc = include_str!("../README.md")]
#![deny(warnings, missing_docs)]

pub extern crate ggml_quants;

mod file;
mod header;
mod metadata;
mod name;
mod read;
mod tensor;
mod write;

pub use file::{GGuf, GGufError};
pub use header::GGufFileHeader;
pub use metadata::{
    DEFAULT_ALIGNMENT, GENERAL_ALIGNMENT, GGmlTokenType, GGufFileType, GGufMetaDataValueType,
    GGufMetaError, GGufMetaKV, GGufMetaMap, GGufMetaMapExt, GGufMetaValueArray,
};
pub use name::{GGufExtNotMatch, GGufFileName};
pub use read::{GGufReadError, GGufReader};
pub use tensor::{GGmlType, GGmlTypeSize, GGufTensorInfo, GGufTensorMeta};
pub use write::{
    DataFuture, GGufFileSimulator, GGufFileWriter, GGufTensorSimulator, GGufTensorWriter,
    GGufWriter,
};

#[inline(always)]
const fn pad(pos: usize, align: usize) -> usize {
    (align - pos % align) % align
}
