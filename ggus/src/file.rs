use crate::{
    DEFAULT_ALIGNMENT, GENERAL_ALIGNMENT, GGufFileHeader, GGufMetaDataValueType, GGufMetaKV,
    GGufMetaMap, GGufReadError, GGufReader, GGufTensorMeta, pad,
};
use indexmap::IndexMap;
use log::{info, warn};
use std::{error::Error, fmt};

/// GGUF 文件的主要结构体，包含文件头、元数据键值对、张量元数据和实际数据。
pub struct GGuf<'a> {
    /// GGUF 文件头，包含版本、张量数量、元数据键值对数量等信息。
    pub header: GGufFileHeader,
    /// 对齐方式，通常为 32 或 64 字节。
    pub alignment: usize,
    /// 元数据键值对，使用 [`IndexMap`] 在提供高效查找的同时保持键值对的逻辑顺序。
    pub meta_kvs: IndexMap<&'a str, GGufMetaKV<'a>>,
    /// 张量元数据，存储的 [`GGufTensorMeta`] 类型是 GGuf 文件中张量元信息原始数据的直接映射，以避免解析不需要的张量带来的开销。
    pub tensors: IndexMap<&'a str, GGufTensorMeta<'a>>,
    /// 实际数据部分，包含所有张量的数据。
    pub data: &'a [u8],
}

/// GGUF 文件解析时可能遇到的错误类型。
#[derive(Debug)]
pub enum GGufError {
    /// 读取 GGUF 文件时发生的错误。
    Reading(GGufReadError),
    /// GGUF 文件的魔术值不匹配，表示文件格式不正确。
    MagicMismatch,
    /// GGUF 文件的字节序不支持，当前实现仅支持本机字节序。
    EndianNotSupport,
    /// GGUF 文件的版本不支持，当前实现仅支持版本 3。
    VersionNotSupport,
    /// 元数据键值对中的对齐类型与预期不匹配。
    AlignmentTypeMismatch(GGufMetaDataValueType),
    /// 元数据键重复，GGUF 文件中不允许有重复的元数据键。
    DuplicateMetaKey(String),
    /// 张量名称重复，GGUF 文件中不允许有重复的张量名称。
    DuplicateTensorName(String),
}

impl fmt::Display for GGufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reading(e) => write!(f, "reading error: {e:?}"),
            Self::MagicMismatch => f.write_str("magic mismatch"),
            Self::EndianNotSupport => f.write_str("endian not support"),
            Self::VersionNotSupport => f.write_str("version not support"),
            Self::AlignmentTypeMismatch(ty) => write!(f, "alignment type mismatch: {ty:?}"),
            Self::DuplicateMetaKey(key) => write!(f, "duplicate meta key: {key}"),
            Self::DuplicateTensorName(name) => write!(f, "duplicate tensor name: {name}"),
        }
    }
}

impl Error for GGufError {}

impl GGufMetaMap for GGuf<'_> {
    fn get(&self, key: &str) -> Option<(GGufMetaDataValueType, &[u8])> {
        self.meta_kvs.get(key).map(|kv| (kv.ty(), kv.value_bytes()))
    }
}

impl<'a> GGuf<'a> {
    /// 创建一个新的 [`GGuf`] 实例，解析给定的 GGUF 数据。
    pub fn new(data: &'a [u8]) -> Result<Self, GGufError> {
        use GGufError::*;

        let mut reader = GGufReader::new(data);

        let header = reader.read_header().map_err(Reading)?;
        if !header.is_magic_correct() {
            return Err(MagicMismatch);
        }
        if !header.is_native_endian() {
            return Err(EndianNotSupport);
        }
        if header.version != 3 {
            return Err(VersionNotSupport);
        }

        let mut alignment = DEFAULT_ALIGNMENT;
        let mut meta_kvs = IndexMap::with_capacity(header.metadata_kv_count as _);
        for _ in 0..header.metadata_kv_count {
            let kv = reader.read_meta_kv().map_err(Reading)?;
            let k = kv.key();
            if k == GENERAL_ALIGNMENT {
                type Ty = GGufMetaDataValueType;
                alignment = match kv.ty() {
                    Ty::U32 => kv.value_reader().read::<u32>().map_err(Reading)? as _,
                    Ty::U64 => kv.value_reader().read::<u64>().map_err(Reading)? as _,
                    ty => return Err(AlignmentTypeMismatch(ty)),
                }
            }
            if meta_kvs.insert(k, kv).is_some() {
                return Err(DuplicateMetaKey(k.into()));
            }
        }

        let mut data_len = 0;
        let mut tensors = IndexMap::with_capacity(header.tensor_count as _);
        for _ in 0..header.tensor_count {
            let tensor = reader.read_tensor_meta().map_err(Reading)?;
            let name = tensor.name();
            let info = tensor.to_info();
            let end = info.offset() + info.nbytes();
            if end > data_len {
                data_len = end;
            }
            if tensors.insert(name, tensor).is_some() {
                return Err(DuplicateTensorName(name.into()));
            }
        }

        let cursor = data.len() - reader.remaining().len();
        let padding = if tensors.is_empty() {
            0
        } else {
            pad(cursor, alignment)
        };
        reader.skip::<u8>(padding).map_err(Reading)?;
        let data = reader.remaining();
        let data = if data.len() == data_len {
            data
        } else {
            let padding = pad(data_len, alignment);
            if data.len() == data_len + padding {
                info!("unnecessary padding detected")
            } else {
                warn!(
                    "extra {} bytes detected after tensor data",
                    data.len() - data_len
                )
            }
            &data[..data_len]
        };

        Ok(Self {
            header,
            alignment,
            meta_kvs,
            tensors,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GGmlType, GGufMetaDataValueType};
    use std::fmt::Write as _;

    // 创建一个最小的有效 GGUF 文件数据
    fn create_minimal_gguf_data() -> Vec<u8> {
        let mut data = Vec::new();

        // 添加文件头
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes());
        data.extend_from_slice(&3u64.to_le_bytes());

        // 添加元数据

        // 元数据 1: general.architecture = "llama"
        let key = "general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&(GGufMetaDataValueType::String as u32).to_le_bytes());
        let value = "llama\0";
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value.as_bytes());

        // 元数据 2: general.alignment = 32
        let key = "general.alignment";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&(GGufMetaDataValueType::U32 as u32).to_le_bytes());
        data.extend_from_slice(&32u32.to_le_bytes());

        // 元数据 3: llm.context_length = 4096
        let key = "llm.context_length";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&(GGufMetaDataValueType::U32 as u32).to_le_bytes());
        data.extend_from_slice(&4096u32.to_le_bytes());

        // 添加张量元数据

        // 张量 1: tensor1
        let tensor_name = "tensor1";
        data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor_name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(&4u64.to_le_bytes());
        data.extend_from_slice(&(GGmlType::F32 as u32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        // 张量 2: tensor2
        let tensor_name = "tensor2";
        data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor_name.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(&(GGmlType::F16 as u32).to_le_bytes());
        data.extend_from_slice(&48u64.to_le_bytes());

        // 添加填充以对齐到 32 字节边界
        let current_size = data.len();
        let padding_size = pad(current_size, 32);
        data.extend(vec![0; padding_size]);

        // 添加张量数据

        // tensor1 数据: 3x4 F32 矩阵 (48 字节)
        for i in 0..12 {
            data.extend_from_slice(&(i as f32).to_le_bytes());
        }

        // tensor2 数据: 5 个 F16 值 (10 字节)
        data.extend([0u8; 10]);

        data
    }

    // 创建具有不同错误的 GGUF 数据
    fn create_invalid_magic_data() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"XXXX");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data
    }

    fn create_invalid_version_data() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&99u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data
    }

    fn create_duplicate_meta_data() -> Vec<u8> {
        let mut data = Vec::new();

        // 添加文件头
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes());

        // 两个相同的键
        for _ in 0..2 {
            let key = "duplicate.key";
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            data.extend_from_slice(&(GGufMetaDataValueType::U32 as u32).to_le_bytes());
            data.extend_from_slice(&1u32.to_le_bytes());
        }

        data
    }

    fn create_duplicate_tensor_data() -> Vec<u8> {
        let mut data = Vec::new();

        // 添加文件头
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        // 两个相同名称的张量
        for _ in 0..2 {
            let tensor_name = "duplicate_tensor";
            data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
            data.extend_from_slice(tensor_name.as_bytes());
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&1u64.to_le_bytes());
            data.extend_from_slice(&(GGmlType::F32 as u32).to_le_bytes());
            data.extend_from_slice(&0u64.to_le_bytes());
        }

        data
    }

    fn create_invalid_alignment_type_data() -> Vec<u8> {
        let mut data = Vec::new();

        // 添加文件头
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        // 使用不支持的类型作为 alignment
        let key = "general.alignment";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&(GGufMetaDataValueType::String as u32).to_le_bytes());
        let value = "not_a_number\0";
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value.as_bytes());

        data
    }

    #[test]
    fn test_valid_gguf_parsing() {
        let data = create_minimal_gguf_data();
        let gguf = GGuf::new(&data).expect("Error parsing valid GGUF data");

        // 验证基本属性
        assert_eq!(gguf.header.version, 3);
        assert_eq!(gguf.header.tensor_count, 2);
        assert_eq!(gguf.header.metadata_kv_count, 3);
        assert_eq!(gguf.alignment, 32);

        // 验证元数据
        assert_eq!(gguf.meta_kvs.len(), 3);

        let arch_kv = gguf.meta_kvs.get("general.architecture").unwrap();
        assert_eq!(arch_kv.ty(), GGufMetaDataValueType::String);

        let ctx_kv = gguf.meta_kvs.get("llm.context_length").unwrap();
        assert_eq!(ctx_kv.ty(), GGufMetaDataValueType::U32);
        assert_eq!(ctx_kv.value_reader().read::<u32>().unwrap(), 4096);

        // 验证张量
        assert_eq!(gguf.tensors.len(), 2);

        let tensor1 = gguf.tensors.get("tensor1").unwrap();
        let tensor1_info = tensor1.to_info();
        assert_eq!(tensor1_info.ty(), GGmlType::F32);
        assert_eq!(tensor1_info.shape(), &[3, 4]);
        assert_eq!(tensor1_info.offset(), 0);

        let tensor2 = gguf.tensors.get("tensor2").unwrap();
        let tensor2_info = tensor2.to_info();
        assert_eq!(tensor2_info.ty(), GGmlType::F16);
        assert_eq!(tensor2_info.shape(), &[5]);
        assert_eq!(tensor2_info.offset(), 48);

        // 验证 GGufMetaMap 实现
        let (ty, _bytes) = gguf.get("general.architecture").unwrap();
        assert_eq!(ty, GGufMetaDataValueType::String);
    }

    #[test]
    fn test_invalid_magic() {
        let data = create_invalid_magic_data();
        let result = GGuf::new(&data);
        assert!(matches!(result, Err(GGufError::MagicMismatch)));
    }

    #[test]
    fn test_invalid_version() {
        let data = create_invalid_version_data();
        let result = GGuf::new(&data);
        assert!(matches!(result, Err(GGufError::VersionNotSupport)));
    }

    #[test]
    fn test_duplicate_meta_key() {
        let data = create_duplicate_meta_data();
        let result = GGuf::new(&data);
        assert!(matches!(result, Err(GGufError::DuplicateMetaKey(_))));
        if let Err(GGufError::DuplicateMetaKey(key)) = result {
            assert_eq!(key, "duplicate.key");
        }
    }

    #[test]
    fn test_duplicate_tensor_name() {
        let data = create_duplicate_tensor_data();
        let result = GGuf::new(&data);
        assert!(matches!(result, Err(GGufError::DuplicateTensorName(_))));
        if let Err(GGufError::DuplicateTensorName(name)) = result {
            assert_eq!(name, "duplicate_tensor");
        }
    }

    #[test]
    fn test_invalid_alignment_type() {
        let data = create_invalid_alignment_type_data();
        let result = GGuf::new(&data);
        assert!(matches!(result, Err(GGufError::AlignmentTypeMismatch(_))));
        if let Err(GGufError::AlignmentTypeMismatch(ty)) = result {
            assert_eq!(ty, GGufMetaDataValueType::String);
        }
    }

    #[test]
    fn test_gguf_error_display() {
        // 测试 GGufError 的 Display 实现
        let errors = [
            (GGufError::MagicMismatch, "magic mismatch"),
            (GGufError::EndianNotSupport, "endian not support"),
            (GGufError::VersionNotSupport, "version not support"),
            (
                GGufError::DuplicateMetaKey("test.key".into()),
                "duplicate meta key: test.key",
            ),
            (
                GGufError::DuplicateTensorName("test".into()),
                "duplicate tensor name: test",
            ),
            (
                GGufError::AlignmentTypeMismatch(GGufMetaDataValueType::String),
                "alignment type mismatch: String",
            ),
            (GGufError::Reading(GGufReadError::Eos), "reading error: Eos"),
        ];

        for (error, expected) in &errors {
            let mut s = String::new();
            write!(s, "{}", error).unwrap();
            assert_eq!(&s, expected);
        }
    }

    #[test]
    fn test_reading_truncated_data() {
        // 测试处理不完整数据的情况
        let mut data = create_minimal_gguf_data();
        // 截断数据
        data.truncate(data.len() / 2);

        let result = GGuf::new(&data);
        assert!(matches!(result, Err(GGufError::Reading(_))));
    }

    #[test]
    fn test_extra_data_handling() {
        let mut data = create_minimal_gguf_data();
        data.extend_from_slice(&[0xAA; 100]);
        let gguf = GGuf::new(&data).expect("Error parsing valid GGUF data");
        assert_eq!(gguf.tensors.len(), 2);
    }
}
