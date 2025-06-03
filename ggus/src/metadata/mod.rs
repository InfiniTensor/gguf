//! See <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#standardized-key-value-pairs>.

#![allow(missing_docs)]
// ! This module provides types and constants for GGUF metadata handling.

mod collection;
mod meta_kv;

pub use collection::{GGufMetaError, GGufMetaMap, GGufMetaMapExt};
pub use meta_kv::{GGufMetaKV, GGufMetaValueArray};

/// 默认对齐方式。
pub const DEFAULT_ALIGNMENT: usize = 32;
/// 表示对齐方式的键。
pub const GENERAL_ALIGNMENT: &str = "general.alignment";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u32)]
pub enum GGufMetaDataValueType {
    /// 8 位无符号整数。
    U8 = 0,
    /// 8 位有符号整数。
    I8 = 1,
    /// 16 位无符号小端整数。
    U16 = 2,
    /// 16 位有符号小端整数。
    I16 = 3,
    /// 32 位无符号小端整数。
    U32 = 4,
    /// 32 位有符号小端整数。
    I32 = 5,
    /// 32 位 IEEE754 浮点数。
    F32 = 6,
    /// bool 值。
    ///
    /// 必须是 0 或 1，其他值未定义。
    Bool = 7,
    /// 非空结尾的 UTF-8 字符串，长度预先指定。
    String = 8,
    /// 非字符类型的数组，长度和类型预先指定。
    ///
    /// 数组可以嵌套；数组长度表示元素个数，而非字节数。
    Array = 9,
    /// 64 位无符号小端整数。
    U64 = 10,
    /// 64 位有符号小端整数。
    I64 = 11,
    /// 64 位 IEEE754 浮点数。
    F64 = 12,
}

/// [`GGufMetaDataValueType`] 获取和处理 GGUF 元数据值类型的相关信息。
impl GGufMetaDataValueType {
    /// 获取 GGUF 元数据值类型的名称。
    pub fn name(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "str",
            Self::Array => "arr",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::U64 => "u64",
        }
    }
}

/// GGUF 文件类型枚举。
///
/// 表示 GGUF 文件中使用的主要数据类型。
#[derive(num_enum::TryFromPrimitive, Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u32)]
pub enum GGufFileType {
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    MostlyQ4_1SomeF16 = 4,
    #[deprecated = "support removed"]
    MostlyQ4_2 = 5,
    #[deprecated = "support removed"]
    MostlyQ4_3 = 6,
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ51 = 9,
    MostlyQ2K = 10,
    MostlyQ3KS = 11,
    MostlyQ3KM = 12,
    MostlyQ3KL = 13,
    MostlyQ4KS = 14,
    MostlyQ4KM = 15,
    MostlyQ5KS = 16,
    MostlyQ5KM = 17,
    MostlyQ6K = 18,
    MostlyIQ2XXS = 19,
    MostlyIQ2XS = 20,
    MostlyQ2KS = 21,
    MostlyIQ3XS = 22,
    MostlyIQ3XXS = 23,
    MostlyIQ1S = 24,
    MostlyIQ4NL = 25,
    MostlyIQ3S = 26,
    MostlyIQ3M = 27,
    MostlyIQ2S = 28,
    MostlyIQ2M = 29,
    MostlyIQ4XS = 30,
    MostlyIQ1M = 31,
    MostlyBF16 = 32,
    MostlyQ4_0_4_4 = 33,
    MostlyQ4_0_4_8 = 34,
    MostlyQ4_0_8_8 = 35,
    // GUESSED = 1024  # not specified in the model file
}

/// 枚举 GGML 中的不同 token 类型。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(i32)]
pub enum GGmlTokenType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    User = 4,
    Unused = 5,
    Byte = 6,
}

#[test]
fn test_gguf_meta_data_value_type() {
    assert_eq!(GGufMetaDataValueType::U8.name(), "u8");
    assert_eq!(GGufMetaDataValueType::I8.name(), "i8");
    assert_eq!(GGufMetaDataValueType::U16.name(), "u16");
    assert_eq!(GGufMetaDataValueType::I16.name(), "i16");
    assert_eq!(GGufMetaDataValueType::U32.name(), "u32");
    assert_eq!(GGufMetaDataValueType::I32.name(), "i32");
    assert_eq!(GGufMetaDataValueType::F32.name(), "f32");
    assert_eq!(GGufMetaDataValueType::Bool.name(), "bool");
    assert_eq!(GGufMetaDataValueType::String.name(), "str");
    assert_eq!(GGufMetaDataValueType::Array.name(), "arr");
    assert_eq!(GGufMetaDataValueType::I64.name(), "i64");
    assert_eq!(GGufMetaDataValueType::F64.name(), "f64");
    assert_eq!(GGufMetaDataValueType::U64.name(), "u64");
}

#[test]
fn test_gguf_file_type() {
    assert_eq!(GGufFileType::AllF32 as u32, 0);
    assert_eq!(GGufFileType::MostlyF16 as u32, 1);
    assert_eq!(GGufFileType::MostlyQ4_0 as u32, 2);
    assert_eq!(GGufFileType::MostlyQ4_1 as u32, 3);
    assert_eq!(GGufFileType::MostlyQ4_1SomeF16 as u32, 4);
    assert_eq!(GGufFileType::MostlyQ8_0 as u32, 7);
}

#[test]
fn test_ggml_token_type() {
    assert_eq!(GGmlTokenType::Normal as i32, 1);
    assert_eq!(GGmlTokenType::Unknown as i32, 2);
    assert_eq!(GGmlTokenType::Control as i32, 3);
    assert_eq!(GGmlTokenType::User as i32, 4);
    assert_eq!(GGmlTokenType::Unused as i32, 5);
    assert_eq!(GGmlTokenType::Byte as i32, 6);
}
