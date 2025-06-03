use super::{DEFAULT_ALIGNMENT, GGufFileType, GGufMetaDataValueType as Ty, GGufMetaValueArray};
use crate::{GGufReadError, GGufReader};

/// [`GGufMetaMap`] trait 定义了获取 GGUF 元数据键值对的接口。
pub trait GGufMetaMap {
    /// 获取元数据键值对。
    fn get(&self, key: &str) -> Option<(Ty, &[u8])>;
}

/// [`GGufMetaError`] 枚举定义了 GGUF 元数据键值对获取时可能出现的错误。
#[derive(Debug)]
pub enum GGufMetaError {
    /// 元数据键值对不存在。
    NotExist,
    /// 元数据类型不匹配。
    TypeMismatch(Ty),
    /// 元数据数组类型不匹配。
    ArrTypeMismatch(Ty),
    /// 元数据超出范围。
    OutOfRange,
    /// 读取元数据时发生错误。
    Read(GGufReadError),
}

/// [`GGufMetaMapExt`] trait 扩展了 [`GGufMetaMap`]，提供了各种类型的元数据键值对获取函数。
pub trait GGufMetaMapExt: GGufMetaMap {
    /// 各类元数据键值对获取函数。
    ///
    /// # 不同数据类型元数据键值对获取函数
    /// 获取字符串类型的元数据键值对。
    fn get_str(&self, key: &str) -> Result<&str, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        match ty {
            Ty::String => GGufReader::new(val).read_str().map_err(GGufMetaError::Read),
            _ => Err(GGufMetaError::TypeMismatch(ty)),
        }
    }

    /// 获取 usize 类型的元数据键值对。
    fn get_usize(&self, key: &str) -> Result<usize, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;

        macro_rules! read {
            ($ty:ty) => {
                GGufReader::new(val)
                    .read::<$ty>()
                    .map_err(GGufMetaError::Read)?
            };
        }
        macro_rules! convert {
            ($val:expr) => {
                $val.try_into().map_err(|_| GGufMetaError::OutOfRange)?
            };
        }

        #[rustfmt::skip]
        let ans = match ty {
            Ty::U8   =>          read!(u8 ).into(),
            Ty::U16  =>          read!(u16).into(),
            Ty::U32  => convert!(read!(u32)      ),
            Ty::U64  => convert!(read!(u64)      ),
            Ty::I8   => convert!(read!(i8 )      ),
            Ty::I16  => convert!(read!(i16)      ),
            Ty::I32  => convert!(read!(i32)      ),
            Ty::I64  => convert!(read!(i64)      ),
            Ty::Bool => if read!(bool) { 1 } else { 0 },
            _        => return Err(GGufMetaError::TypeMismatch(ty)),
        };

        Ok(ans)
    }

    /// 获取 f32 类型的元数据键值对。
    fn get_f32(&self, key: &str) -> Result<f32, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        if ty == Ty::F32 {
            GGufReader::new(val).read().map_err(GGufMetaError::Read)
        } else {
            Err(GGufMetaError::TypeMismatch(ty))
        }
    }

    /// 获取 u32 类型的元数据键值对。
    fn get_u32(&self, key: &str) -> Result<u32, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        if ty == Ty::U32 {
            GGufReader::new(val).read().map_err(GGufMetaError::Read)
        } else {
            Err(GGufMetaError::TypeMismatch(ty))
        }
    }

    /// 获取 bool 类型的元数据键值对。
    fn get_bool(&self, key: &str) -> Result<bool, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        if ty == Ty::Bool {
            GGufReader::new(val)
                .read_bool()
                .map_err(GGufMetaError::Read)
        } else {
            Err(GGufMetaError::TypeMismatch(ty))
        }
    }

    /// 获取字符串数组类型的元数据键值对。
    fn get_str_arr(&self, key: &str) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        let mut reader = GGufReader::new(val);
        let (ty, len) = match ty {
            Ty::Array => reader.read_arr_header().map_err(GGufMetaError::Read)?,
            ty => return Err(GGufMetaError::TypeMismatch(ty)),
        };
        if ty == Ty::String {
            Ok(GGufMetaValueArray::new(reader, len))
        } else {
            Err(GGufMetaError::ArrTypeMismatch(ty))
        }
    }

    /// 获取 i32 数组类型的元数据键值对。
    fn get_i32_arr(&self, key: &str) -> Result<GGufMetaValueArray<i32>, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        let mut reader = GGufReader::new(val);
        let (ty, len) = match ty {
            Ty::Array => reader.read_arr_header().map_err(GGufMetaError::Read)?,
            ty => return Err(GGufMetaError::TypeMismatch(ty)),
        };
        if ty == Ty::I32 {
            Ok(GGufMetaValueArray::new(reader, len))
        } else {
            Err(GGufMetaError::ArrTypeMismatch(ty))
        }
    }

    /// 获取 f32 数组类型的元数据键值对。
    fn get_f32_arr(&self, key: &str) -> Result<GGufMetaValueArray<f32>, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        let mut reader = GGufReader::new(val);
        let (ty, len) = match ty {
            Ty::Array => reader.read_arr_header().map_err(GGufMetaError::Read)?,
            ty => return Err(GGufMetaError::TypeMismatch(ty)),
        };
        if ty == Ty::F32 {
            Ok(GGufMetaValueArray::new(reader, len))
        } else {
            Err(GGufMetaError::ArrTypeMismatch(ty))
        }
    }

    /// # general 字段元数据键值对获取函数
    /// 获取架构。
    #[inline]
    fn general_architecture(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.architecture")
    }

    /// 获取量化版本。
    #[inline]
    fn general_quantization_version(&self) -> Result<usize, GGufMetaError> {
        self.get_usize("general.quantization_version")
    }

    /// 获取对齐方式。
    #[inline]
    fn general_alignment(&self) -> Result<usize, GGufMetaError> {
        match self.get_usize("general.alignment") {
            Ok(n) => Ok(n),
            Err(GGufMetaError::NotExist) => Ok(DEFAULT_ALIGNMENT),
            Err(e) => Err(e),
        }
    }

    /// 获取名称。
    #[inline]
    fn general_name(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.name")
    }

    /// 获取作者。
    #[inline]
    fn general_author(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.author")
    }

    /// 获取版本。
    #[inline]
    fn general_version(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.version")
    }

    /// 获取组织。
    #[inline]
    fn general_organization(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.organization")
    }

    /// 获取基准名称。
    #[inline]
    fn general_basename(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.basename")
    }

    /// 获取微调标识。
    #[inline]
    fn general_finetune(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.finetune")
    }

    /// 获取描述。
    #[inline]
    fn general_description(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.description")
    }

    /// 获取量化作者。
    #[inline]
    fn general_quantized_by(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.quantized_by")
    }

    /// 获取量化版本。
    #[inline]
    fn general_size_label(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.size_label")
    }

    /// 获取许可。
    #[inline]
    fn general_license(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.license")
    }

    /// 获取许可名称。
    #[inline]
    fn general_license_name(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.license.name")
    }

    /// 获取许可链接。
    #[inline]
    fn general_license_link(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.license.link")
    }

    /// 获取链接。
    #[inline]
    fn general_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.url")
    }

    /// 获取 DOI。
    #[inline]
    fn general_doi(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.doi")
    }

    /// 获取 UUID。
    #[inline]
    fn general_uuid(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.uuid")
    }

    /// 获取仓库链接。
    #[inline]
    fn general_repo_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.repo_url")
    }

    /// 获取标签。
    #[inline]
    fn general_tags(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("general.tags")
    }

    /// 获取语言。
    #[inline]
    fn general_languages(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("general.languages")
    }

    /// 获取数据集。
    #[inline]
    fn general_datasets(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("general.datasets")
    }

    /// 获取文件类型。
    #[inline]
    fn general_filetype(&self) -> Result<GGufFileType, GGufMetaError> {
        (self.get_usize("general.filetype")? as u32)
            .try_into()
            .map_err(|_| GGufMetaError::OutOfRange)
    }

    /// 获取源信息。
    #[inline]
    fn general_source_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.url")
    }

    /// 获取源 DOI。
    #[inline]
    fn general_source_doi(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.doi")
    }

    /// 获取源 UUID。
    #[inline]
    fn general_source_uuid(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.uuid")
    }

    /// 获取源仓库链接。
    #[inline]
    fn general_source_repo_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.repo_url")
    }

    /// 获取基准模型数量。
    #[inline]
    fn general_base_model_count(&self) -> Result<usize, GGufMetaError> {
        self.get_usize("general.base_model.count")
    }

    /// 获取基准模型名称。
    #[inline]
    fn general_base_model_name(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.name"))
    }

    /// 获取基准模型作者。
    #[inline]
    fn general_base_model_author(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.author"))
    }

    /// 获取基准模型版本。
    #[inline]
    fn general_base_model_version(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.version"))
    }

    /// 获取基准模型组织。
    #[inline]
    fn general_base_model_organization(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.organization"))
    }

    /// 获取基准模型 URL。
    #[inline]
    fn general_base_model_url(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.url"))
    }

    /// 获取基准模型 DOI。
    #[inline]
    fn general_base_model_doi(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.doi"))
    }

    /// 获取基准模型 UUID。
    #[inline]
    fn general_base_model_uuid(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.uuid"))
    }

    /// 获取基准模型仓库链接。
    #[inline]
    fn general_base_model_repo_url(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.repo_url"))
    }

    /// # llm 字段元数据键值对获取函数
    /// 获取上下文长度。
    #[inline]
    fn llm_context_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.context_length"))
    }

    /// 获取嵌入层长度。
    #[inline]
    fn llm_embedding_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.embedding_length"))
    }

    /// 获取块数量。
    #[inline]
    fn llm_block_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.block_count"))
    }

    /// 获取前馈层长度。
    #[inline]
    fn llm_feed_forward_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.feed_forward_length"))
    }

    /// 获取是否使用并行残差。
    #[inline]
    fn llm_use_parallel_residual(&self) -> Result<bool, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_bool(&format!("{llm}.use_parallel_residual"))
    }

    /// 获取张量数据布局。
    #[inline]
    fn llm_tensor_data_layout(&self) -> Result<&str, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_str(&format!("{llm}.tensor_data_layout"))
    }

    /// 获取专家数量。
    #[inline]
    fn llm_expert_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.expert_count"))
    }

    /// 获取已使用的专家数量。
    #[inline]
    fn llm_expert_used_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.expert_used_count"))
    }

    /// 获取注意力头数量。
    #[inline]
    fn llm_attention_head_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.attention.head_count"))
    }

    /// 获取注意力 KV 头数量。
    #[inline]
    fn llm_attention_head_count_kv(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        match self.get_usize(&format!("{llm}.attention.head_count_kv")) {
            Ok(n) => Ok(n),
            Err(GGufMetaError::NotExist) => self.llm_attention_head_count(),
            Err(e) => Err(e),
        }
    }

    /// 获取最大 Alibi 偏置值。
    #[inline]
    fn llm_attention_max_alibi_bias(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.max_alibi_bias"))
    }

    /// 获取 K/Q/V 限幅阈值。
    #[inline]
    fn llm_attention_clamp_kqv(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.clamp_kqv"))
    }

    /// 获取归一化 ε 参数。
    #[inline]
    fn llm_attention_layer_norm_epsilon(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.layer_norm_epsilon"))
    }

    /// 获取 RMS 归一化 ε 参数。
    #[inline]
    fn llm_attention_layer_norm_rms_epsilon(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.layer_norm_rms_epsilon"))
    }

    /// 获取注意力 Key 长度。
    #[inline]
    fn llm_attention_key_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        match self.get_usize(&format!("{llm}.attention.key_length")) {
            Ok(n) => Ok(n),
            Err(GGufMetaError::NotExist) => {
                let n_embed = self.llm_embedding_length()?;
                let n_head = self.llm_attention_head_count()?;
                Ok(n_embed / n_head)
            }
            Err(e) => Err(e),
        }
    }

    /// 获取注意力 Value 长度。
    #[inline]
    fn llm_attention_value_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        match self.get_usize(&format!("{llm}.attention.value_length")) {
            Ok(n) => Ok(n),
            Err(GGufMetaError::NotExist) => {
                let n_embed = self.llm_embedding_length()?;
                let n_head = self.llm_attention_head_count()?;
                Ok(n_embed / n_head)
            }
            Err(e) => Err(e),
        }
    }

    /// 获取 RoPE 维度数量。
    #[inline]
    fn llm_rope_dimension_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.rope.dimension_count"))
    }

    /// 获取 RoPE 频率基数。
    #[inline]
    fn llm_rope_freq_base(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.rope.freq_base"))
    }

    /// 获取 RoPE 缩放类型。
    #[inline]
    fn llm_rope_scaling_type(&self) -> Result<&str, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_str(&format!("{llm}.rope.scaling.type"))
    }

    /// 获取 RoPE 缩放因子。
    #[inline]
    fn llm_rope_scaling_factor(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.rope.scaling.factor"))
    }

    /// 获取 RoPE 缩放的原始上下文长度。
    #[inline]
    fn llm_rope_scaling_original_context_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.rope.scaling.original_context_length"))
    }

    /// 获取 RoPE 缩放是否经过微调。
    #[inline]
    fn llm_rope_scaling_finetuned(&self) -> Result<bool, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_bool(&format!("{llm}.rope.scaling.finetuned"))
    }

    /// 获取 RoPE 缩放线性因子。
    #[inline]
    fn llm_rope_scale_linear(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.rope.scale_linear"))
    }

    /// 获取状态空间模型（SSM）卷积核大小。
    #[inline]
    fn llm_ssm_conv_kernel(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.conv_kernel"))
    }

    /// 获取状态空间模型（SSM）内部大小。
    #[inline]
    fn llm_ssm_inner_size(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.inner_size"))
    }

    /// 获取状态空间模型（SSM）状态大小。
    #[inline]
    fn llm_ssm_state_size(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.state_size"))
    }

    /// 获取状态空间模型（SSM）时间步长秩。
    #[inline]
    fn llm_ssm_time_step_rank(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.time_step_rank"))
    }

    /// # tokenizer 字段元数据键值对获取函数
    /// 获取用于 ggml 的分词器模型。
    #[inline]
    fn tokenizer_ggml_model(&self) -> Result<&str, GGufMetaError> {
        self.get_str("tokenizer.ggml.model")
    }

    /// 获取 ggml 分词器的词汇表。
    #[inline]
    fn tokenizer_ggml_tokens(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("tokenizer.ggml.tokens")
    }

    /// 获取 ggml 分词器的分数。
    #[inline]
    fn tokenizer_ggml_scores(&self) -> Result<GGufMetaValueArray<f32>, GGufMetaError> {
        self.get_f32_arr("tokenizer.ggml.scores")
    }

    /// 获取 ggml 分词器的 token 类型。
    #[inline]
    fn tokenizer_ggml_token_type(&self) -> Result<GGufMetaValueArray<i32>, GGufMetaError> {
        self.get_i32_arr("tokenizer.ggml.token_type")
    }

    /// 获取 ggml 分词器的合并规则。
    #[inline]
    fn tokenizer_ggml_merges(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("tokenizer.ggml.merges")
    }

    /// 获取 ggml 分词器的添加的 token。
    #[inline]
    fn tokenizer_ggml_added_tokens(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("tokenizer.ggml.added_tokens")
    }

    /// 获取 ggml 分词器的起始标记 ID。
    #[inline]
    fn tokenizer_ggml_bos_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.bos_token_id")
    }

    /// 获取 ggml 分词器的结束标记 ID。
    #[inline]
    fn tokenizer_ggml_eos_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.eos_token_id")
    }

    /// 获取 ggml 分词器的未知标记 ID。
    #[inline]
    fn tokenizer_ggml_unknown_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.unknown_token_id")
    }

    /// 获取 ggml 分词器的分隔符标记 ID。
    #[inline]
    fn tokenizer_ggml_separator_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.separator_token_id")
    }

    /// 获取 ggml 分词器的填充标记 ID。
    #[inline]
    fn tokenizer_ggml_padding_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.padding_token_id")
    }

    /// 获取 RWKV 分词器的词表。
    #[inline]
    fn tokenizer_rwkv_world(&self) -> Result<&str, GGufMetaError> {
        self.get_str("tokenizer.rwkv.world")
    }

    /// 获取聊天模板。
    #[inline]
    fn tokenizer_chat_template(&self) -> Result<&str, GGufMetaError> {
        self.get_str("tokenizer.chat_template")
    }
}

impl<T: GGufMetaMap> GGufMetaMapExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // 实现一个简单的 MetaMap 用于测试
    struct TestMetaMap {
        data: HashMap<String, (Ty, Vec<u8>)>,
    }

    impl TestMetaMap {
        fn new() -> Self {
            let mut map = HashMap::new();

            // 添加各种类型的测试数据

            // 通用字段
            map.insert(
                "general.architecture".to_string(),
                (Ty::String, encode_string("llama")),
            );
            map.insert(
                "general.name".to_string(),
                (Ty::String, encode_string("TestModel")),
            );
            map.insert(
                "general.author".to_string(),
                (Ty::String, encode_string("Test Author")),
            );
            map.insert(
                "general.version".to_string(),
                (Ty::String, encode_string("1.0")),
            );
            map.insert(
                "general.description".to_string(),
                (Ty::String, encode_string("Test Description")),
            );
            map.insert(
                "general.quantization_version".to_string(),
                (Ty::U32, encode_u32(2)),
            );
            map.insert("general.alignment".to_string(), (Ty::U32, encode_u32(32)));
            map.insert(
                "general.size_label".to_string(),
                (Ty::String, encode_string("7B")),
            );
            map.insert(
                "general.finetune".to_string(),
                (Ty::String, encode_string("chat")),
            );
            map.insert(
                "general.tags".to_string(),
                (Ty::Array, encode_string_array(&["tag1", "tag2"])),
            );
            map.insert(
                "general.languages".to_string(),
                (Ty::Array, encode_string_array(&["en", "zh"])),
            );

            // LLM 特定字段
            map.insert(
                "llama.context_length".to_string(),
                (Ty::U32, encode_u32(4096)),
            );
            map.insert(
                "llama.embedding_length".to_string(),
                (Ty::U32, encode_u32(4096)),
            );
            map.insert("llama.block_count".to_string(), (Ty::U32, encode_u32(32)));
            map.insert(
                "llama.feed_forward_length".to_string(),
                (Ty::U32, encode_u32(11008)),
            );
            map.insert(
                "llama.attention.head_count".to_string(),
                (Ty::U32, encode_u32(32)),
            );
            map.insert(
                "llama.rope.dimension_count".to_string(),
                (Ty::U32, encode_u32(128)),
            );
            map.insert(
                "llama.rope.freq_base".to_string(),
                (Ty::F32, encode_f32(10000.0)),
            );
            map.insert(
                "llama.use_parallel_residual".to_string(),
                (Ty::Bool, encode_bool(true)),
            );
            map.insert(
                "llama.tensor_data_layout".to_string(),
                (Ty::String, encode_string("row-major")),
            );
            map.insert("llama.expert_count".to_string(), (Ty::U32, encode_u32(0)));
            map.insert(
                "llama.expert_used_count".to_string(),
                (Ty::U32, encode_u32(0)),
            );

            // 分词器相关字段
            map.insert(
                "tokenizer.ggml.model".to_string(),
                (Ty::String, encode_string("llama")),
            );
            map.insert(
                "tokenizer.ggml.bos_token_id".to_string(),
                (Ty::U32, encode_u32(1)),
            );
            map.insert(
                "tokenizer.ggml.eos_token_id".to_string(),
                (Ty::U32, encode_u32(2)),
            );
            map.insert(
                "tokenizer.chat_template".to_string(),
                (
                    Ty::String,
                    encode_string("<|im_start|>user\n{prompt}<|im_end|>"),
                ),
            );
            map.insert(
                "tokenizer.ggml.unknown_token_id".to_string(),
                (Ty::U32, encode_u32(0)),
            );
            map.insert(
                "tokenizer.ggml.separator_token_id".to_string(),
                (Ty::U32, encode_u32(3)),
            );
            map.insert(
                "tokenizer.ggml.padding_token_id".to_string(),
                (Ty::U32, encode_u32(4)),
            );
            map.insert(
                "tokenizer.ggml.tokens".to_string(),
                (Ty::Array, encode_string_array(&["<bos>", "<eos>", "hello"])),
            );
            map.insert(
                "tokenizer.ggml.scores".to_string(),
                (Ty::Array, {
                    let mut data = Vec::new();
                    // 写入数组头部
                    data.extend_from_slice(&(Ty::F32 as u32).to_le_bytes());
                    data.extend_from_slice(&(3u64).to_le_bytes());
                    // 写入数据
                    data.extend_from_slice(&(0.0f32).to_le_bytes());
                    data.extend_from_slice(&(-1.0f32).to_le_bytes());
                    data.extend_from_slice(&(-2.0f32).to_le_bytes());
                    data
                }),
            );
            map.insert(
                "tokenizer.ggml.token_type".to_string(),
                (Ty::Array, {
                    let mut data = Vec::new();
                    // 写入数组头部
                    data.extend_from_slice(&(Ty::I32 as u32).to_le_bytes());
                    data.extend_from_slice(&(3u64).to_le_bytes());
                    // 写入数据
                    data.extend_from_slice(&(1i32).to_le_bytes());
                    data.extend_from_slice(&(1i32).to_le_bytes());
                    data.extend_from_slice(&(0i32).to_le_bytes());
                    data
                }),
            );
            map.insert(
                "tokenizer.ggml.merges".to_string(),
                (Ty::Array, encode_string_array(&["h e", "he llo"])),
            );
            map.insert(
                "tokenizer.ggml.added_tokens".to_string(),
                (Ty::Array, encode_string_array(&["<custom_token>"])),
            );
            map.insert(
                "tokenizer.rwkv.world".to_string(),
                (Ty::String, encode_string("world")),
            );

            // attention 相关字段
            map.insert(
                "llama.attention.layer_norm_epsilon".to_string(),
                (Ty::F32, encode_f32(1e-6)),
            );
            map.insert(
                "llama.attention.layer_norm_rms_epsilon".to_string(),
                (Ty::F32, encode_f32(1e-5)),
            );
            map.insert(
                "llama.attention.max_alibi_bias".to_string(),
                (Ty::F32, encode_f32(8.0)),
            );
            map.insert(
                "llama.attention.clamp_kqv".to_string(),
                (Ty::F32, encode_f32(0.0)),
            );

            // rope 相关字段
            map.insert(
                "llama.rope.scaling.type".to_string(),
                (Ty::String, encode_string("linear")),
            );
            map.insert(
                "llama.rope.scaling.factor".to_string(),
                (Ty::F32, encode_f32(1.0)),
            );
            map.insert(
                "llama.rope.scaling.original_context_length".to_string(),
                (Ty::U32, encode_u32(2048)),
            );
            map.insert(
                "llama.rope.scaling.finetuned".to_string(),
                (Ty::Bool, encode_bool(false)),
            );
            map.insert(
                "llama.rope.scale_linear".to_string(),
                (Ty::F32, encode_f32(1.0)),
            );

            // ssm 相关字段
            map.insert(
                "llama.ssm.conv_kernel".to_string(),
                (Ty::U32, encode_u32(4)),
            );
            map.insert(
                "llama.ssm.inner_size".to_string(),
                (Ty::U32, encode_u32(256)),
            );
            map.insert(
                "llama.ssm.state_size".to_string(),
                (Ty::U32, encode_u32(16)),
            );
            map.insert(
                "llama.ssm.time_step_rank".to_string(),
                (Ty::U32, encode_u32(8)),
            );

            Self { data: map }
        }
    }

    impl GGufMetaMap for TestMetaMap {
        fn get(&self, key: &str) -> Option<(Ty, &[u8])> {
            self.data.get(key).map(|(ty, data)| (*ty, data.as_slice()))
        }
    }

    // 辅助函数，用于编码各种数据类型
    fn encode_string(s: &str) -> Vec<u8> {
        let mut result = Vec::new();
        let len = s.len() as u64;
        result.extend_from_slice(&len.to_le_bytes());
        result.extend_from_slice(s.as_bytes());
        result
    }

    fn encode_u32(val: u32) -> Vec<u8> {
        val.to_le_bytes().to_vec()
    }

    fn encode_f32(val: f32) -> Vec<u8> {
        val.to_le_bytes().to_vec()
    }

    fn encode_bool(val: bool) -> Vec<u8> {
        vec![if val { 1u8 } else { 0u8 }]
    }

    fn encode_string_array(strings: &[&str]) -> Vec<u8> {
        let mut result = Vec::new();

        // 写入数组头部
        let element_type = Ty::String as u32;
        result.extend_from_slice(&element_type.to_le_bytes());

        let len = strings.len() as u64;
        result.extend_from_slice(&len.to_le_bytes());

        // 写入每个字符串
        for s in strings {
            let str_len = s.len() as u64;
            result.extend_from_slice(&str_len.to_le_bytes());
            result.extend_from_slice(s.as_bytes());
        }

        result
    }

    #[test]
    fn test_get_usize_all_types() {
        // 创建包含所有支持类型的测试数据
        let mut map = HashMap::new();

        // 添加所有可以转换为 usize 的类型
        map.insert("u8_val".to_string(), (Ty::U8, vec![42]));
        map.insert(
            "u16_val".to_string(),
            (Ty::U16, (1000u16).to_le_bytes().to_vec()),
        );
        map.insert(
            "u32_val".to_string(),
            (Ty::U32, (100000u32).to_le_bytes().to_vec()),
        );
        map.insert(
            "u64_val".to_string(),
            (Ty::U64, (5000000000u64).to_le_bytes().to_vec()),
        );
        map.insert("i8_val".to_string(), (Ty::I8, vec![42]));
        map.insert(
            "i16_val".to_string(),
            (Ty::I16, (1000i16).to_le_bytes().to_vec()),
        );
        map.insert(
            "i32_val".to_string(),
            (Ty::I32, (100000i32).to_le_bytes().to_vec()),
        );
        map.insert(
            "i64_val".to_string(),
            (Ty::I64, (5000000000i64).to_le_bytes().to_vec()),
        );
        map.insert("bool_true".to_string(), (Ty::Bool, vec![1]));
        map.insert("bool_false".to_string(), (Ty::Bool, vec![0]));

        // 添加无法转换为 usize 的类型
        map.insert(
            "string_val".to_string(),
            (Ty::String, encode_string("test")),
        );
        map.insert("f32_val".to_string(), (Ty::F32, encode_f32(1.23)));
        map.insert(
            "array_val".to_string(),
            (Ty::Array, encode_string_array(&["test"])),
        );

        // 添加会导致溢出或负数的值
        map.insert("negative_i8".to_string(), (Ty::I8, vec![0xFF])); // -1
        map.insert(
            "negative_i16".to_string(),
            (Ty::I16, (-1000i16).to_le_bytes().to_vec()),
        );
        map.insert(
            "negative_i32".to_string(),
            (Ty::I32, (-100000i32).to_le_bytes().to_vec()),
        );
        map.insert(
            "negative_i64".to_string(),
            (Ty::I64, (-5000000000i64).to_le_bytes().to_vec()),
        );

        // 添加损坏的值
        map.insert("corrupt_u32".to_string(), (Ty::U32, vec![1, 2])); // 长度不足的数据

        let meta_map = TestMetaMap { data: map };

        // 测试正常转换情况
        assert_eq!(meta_map.get_usize("u8_val").unwrap(), 42);
        assert_eq!(meta_map.get_usize("u16_val").unwrap(), 1000);
        assert_eq!(meta_map.get_usize("u32_val").unwrap(), 100000);
        assert_eq!(meta_map.get_usize("u64_val").unwrap(), 5000000000);
        assert_eq!(meta_map.get_usize("i8_val").unwrap(), 42);
        assert_eq!(meta_map.get_usize("i16_val").unwrap(), 1000);
        assert_eq!(meta_map.get_usize("i32_val").unwrap(), 100000);
        assert_eq!(meta_map.get_usize("i64_val").unwrap(), 5000000000);
        assert_eq!(meta_map.get_usize("bool_true").unwrap(), 1);
        assert_eq!(meta_map.get_usize("bool_false").unwrap(), 0);

        // 测试类型不匹配
        assert!(matches!(
            meta_map.get_usize("string_val").unwrap_err(),
            GGufMetaError::TypeMismatch(Ty::String)
        ));
        assert!(matches!(
            meta_map.get_usize("f32_val").unwrap_err(),
            GGufMetaError::TypeMismatch(Ty::F32)
        ));
        assert!(matches!(
            meta_map.get_usize("array_val").unwrap_err(),
            GGufMetaError::TypeMismatch(Ty::Array)
        ));

        // 测试负数值
        assert!(matches!(
            meta_map.get_usize("negative_i8").unwrap_err(),
            GGufMetaError::OutOfRange
        ));
        assert!(matches!(
            meta_map.get_usize("negative_i16").unwrap_err(),
            GGufMetaError::OutOfRange
        ));
        assert!(matches!(
            meta_map.get_usize("negative_i32").unwrap_err(),
            GGufMetaError::OutOfRange
        ));
        assert!(matches!(
            meta_map.get_usize("negative_i64").unwrap_err(),
            GGufMetaError::OutOfRange
        ));

        // 测试损坏的值
        assert!(matches!(
            meta_map.get_usize("corrupt_u32").unwrap_err(),
            GGufMetaError::Read(_)
        ));

        // 测试不存在的值
        assert!(matches!(
            meta_map.get_usize("non_existent").unwrap_err(),
            GGufMetaError::NotExist
        ));
    }

    #[test]
    fn test_general_fields() {
        let meta_map = TestMetaMap::new();

        // 测试基础 general 字段
        assert_eq!(meta_map.general_architecture().unwrap(), "llama");
        assert_eq!(meta_map.general_name().unwrap(), "TestModel");
        assert_eq!(meta_map.general_author().unwrap(), "Test Author");
        assert_eq!(meta_map.general_version().unwrap(), "1.0");
        assert_eq!(meta_map.general_description().unwrap(), "Test Description");
        assert_eq!(meta_map.general_quantization_version().unwrap(), 2);
        assert_eq!(meta_map.general_alignment().unwrap(), 32);
        assert_eq!(meta_map.general_size_label().unwrap(), "7B");
        assert_eq!(meta_map.general_finetune().unwrap(), "chat");

        // 测试默认 alignment 回退处理
        let map = HashMap::new();
        let meta_map_without_alignment = TestMetaMap { data: map };
        assert_eq!(
            meta_map_without_alignment.general_alignment().unwrap(),
            DEFAULT_ALIGNMENT
        );

        // 测试所有可选的 general 字段缺失情况
        assert!(meta_map.general_organization().is_err());
        assert!(meta_map.general_basename().is_err());
        assert!(meta_map.general_quantized_by().is_err());
        assert!(meta_map.general_license().is_err());
        assert!(meta_map.general_license_name().is_err());
        assert!(meta_map.general_license_link().is_err());
        assert!(meta_map.general_url().is_err());
        assert!(meta_map.general_doi().is_err());
        assert!(meta_map.general_uuid().is_err());
        assert!(meta_map.general_repo_url().is_err());

        // 测试数组类型字段
        let tags: Vec<_> = meta_map
            .general_tags()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(tags, vec!["tag1", "tag2"]);

        let languages: Vec<_> = meta_map
            .general_languages()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(languages, vec!["en", "zh"]);

        // datasets 字段缺失的情况
        assert!(meta_map.general_datasets().is_err());

        // 测试 source 相关字段缺失情况
        assert!(meta_map.general_source_url().is_err());
        assert!(meta_map.general_source_doi().is_err());
        assert!(meta_map.general_source_uuid().is_err());
        assert!(meta_map.general_source_repo_url().is_err());

        // 测试 base_model 相关字段缺失情况
        assert!(meta_map.general_base_model_count().is_err());
        assert!(meta_map.general_base_model_name(0).is_err());
        assert!(meta_map.general_base_model_author(0).is_err());
        assert!(meta_map.general_base_model_version(0).is_err());
        assert!(meta_map.general_base_model_organization(0).is_err());
        assert!(meta_map.general_base_model_url(0).is_err());
        assert!(meta_map.general_base_model_doi(0).is_err());
        assert!(meta_map.general_base_model_uuid(0).is_err());
        assert!(meta_map.general_base_model_repo_url(0).is_err());
    }

    #[test]
    fn test_general_additional_fields() {
        // 创建一个包含所有 general 字段的测试数据
        let mut map = HashMap::new();

        // 基本 general 字段
        map.insert(
            "general.architecture".to_string(),
            (Ty::String, encode_string("llama")),
        );
        map.insert(
            "general.name".to_string(),
            (Ty::String, encode_string("TestModel")),
        );
        map.insert(
            "general.author".to_string(),
            (Ty::String, encode_string("Test Author")),
        );

        // 添加额外的 general 字段
        map.insert(
            "general.organization".to_string(),
            (Ty::String, encode_string("TestOrg")),
        );
        map.insert(
            "general.basename".to_string(),
            (Ty::String, encode_string("base_model")),
        );
        map.insert(
            "general.quantized_by".to_string(),
            (Ty::String, encode_string("quantizer")),
        );
        map.insert(
            "general.license".to_string(),
            (Ty::String, encode_string("MIT")),
        );
        map.insert(
            "general.license.name".to_string(),
            (Ty::String, encode_string("MIT License")),
        );
        map.insert(
            "general.license.link".to_string(),
            (Ty::String, encode_string("https://mit-license.org")),
        );
        map.insert(
            "general.url".to_string(),
            (Ty::String, encode_string("https://example.com/model")),
        );
        map.insert(
            "general.doi".to_string(),
            (Ty::String, encode_string("10.1234/test")),
        );
        map.insert(
            "general.uuid".to_string(),
            (
                Ty::String,
                encode_string("123e4567-e89b-12d3-a456-426614174000"),
            ),
        );
        map.insert(
            "general.repo_url".to_string(),
            (Ty::String, encode_string("https://github.com/example/repo")),
        );
        map.insert("general.filetype".to_string(), (Ty::U32, encode_u32(0)));

        // source 相关字段
        map.insert(
            "general.source.url".to_string(),
            (Ty::String, encode_string("https://example.com/source")),
        );
        map.insert(
            "general.source.doi".to_string(),
            (Ty::String, encode_string("10.1234/source")),
        );
        map.insert(
            "general.source.uuid".to_string(),
            (
                Ty::String,
                encode_string("123e4567-e89b-12d3-a456-426614174001"),
            ),
        );
        map.insert(
            "general.source.repo_url".to_string(),
            (
                Ty::String,
                encode_string("https://github.com/example/source"),
            ),
        );

        // base_model 相关字段
        map.insert(
            "general.base_model.count".to_string(),
            (Ty::U32, encode_u32(2)),
        );
        map.insert(
            "general.base_model.0.name".to_string(),
            (Ty::String, encode_string("Base Model 1")),
        );
        map.insert(
            "general.base_model.0.author".to_string(),
            (Ty::String, encode_string("Base Author 1")),
        );
        map.insert(
            "general.base_model.0.version".to_string(),
            (Ty::String, encode_string("1.0")),
        );
        map.insert(
            "general.base_model.0.organization".to_string(),
            (Ty::String, encode_string("Base Org 1")),
        );
        map.insert(
            "general.base_model.0.url".to_string(),
            (Ty::String, encode_string("https://example.com/base1")),
        );
        map.insert(
            "general.base_model.0.doi".to_string(),
            (Ty::String, encode_string("10.1234/base1")),
        );
        map.insert(
            "general.base_model.0.uuid".to_string(),
            (Ty::String, encode_string("uuid-base-1")),
        );
        map.insert(
            "general.base_model.0.repo_url".to_string(),
            (
                Ty::String,
                encode_string("https://github.com/example/base1"),
            ),
        );

        map.insert(
            "general.base_model.1.name".to_string(),
            (Ty::String, encode_string("Base Model 2")),
        );
        map.insert(
            "general.base_model.1.author".to_string(),
            (Ty::String, encode_string("Base Author 2")),
        );

        // 数组类型字段
        map.insert(
            "general.datasets".to_string(),
            (Ty::Array, encode_string_array(&["dataset1", "dataset2"])),
        );

        let meta_map = TestMetaMap { data: map };

        // 测试附加 general 字段
        assert_eq!(meta_map.general_organization().unwrap(), "TestOrg");
        assert_eq!(meta_map.general_basename().unwrap(), "base_model");
        assert_eq!(meta_map.general_quantized_by().unwrap(), "quantizer");
        assert_eq!(meta_map.general_license().unwrap(), "MIT");
        assert_eq!(meta_map.general_license_name().unwrap(), "MIT License");
        assert_eq!(
            meta_map.general_license_link().unwrap(),
            "https://mit-license.org"
        );
        assert_eq!(meta_map.general_url().unwrap(), "https://example.com/model");
        assert_eq!(meta_map.general_doi().unwrap(), "10.1234/test");
        assert_eq!(
            meta_map.general_uuid().unwrap(),
            "123e4567-e89b-12d3-a456-426614174000"
        );
        assert_eq!(
            meta_map.general_repo_url().unwrap(),
            "https://github.com/example/repo"
        );
        assert_eq!(meta_map.general_filetype().unwrap(), GGufFileType::AllF32);

        // 测试 source 相关字段
        assert_eq!(
            meta_map.general_source_url().unwrap(),
            "https://example.com/source"
        );
        assert_eq!(meta_map.general_source_doi().unwrap(), "10.1234/source");
        assert_eq!(
            meta_map.general_source_uuid().unwrap(),
            "123e4567-e89b-12d3-a456-426614174001"
        );
        assert_eq!(
            meta_map.general_source_repo_url().unwrap(),
            "https://github.com/example/source"
        );

        // 测试 base_model 相关字段
        assert_eq!(meta_map.general_base_model_count().unwrap(), 2);
        assert_eq!(meta_map.general_base_model_name(0).unwrap(), "Base Model 1");
        assert_eq!(
            meta_map.general_base_model_author(0).unwrap(),
            "Base Author 1"
        );
        assert_eq!(meta_map.general_base_model_version(0).unwrap(), "1.0");
        assert_eq!(
            meta_map.general_base_model_organization(0).unwrap(),
            "Base Org 1"
        );
        assert_eq!(
            meta_map.general_base_model_url(0).unwrap(),
            "https://example.com/base1"
        );
        assert_eq!(meta_map.general_base_model_doi(0).unwrap(), "10.1234/base1");
        assert_eq!(meta_map.general_base_model_uuid(0).unwrap(), "uuid-base-1");
        assert_eq!(
            meta_map.general_base_model_repo_url(0).unwrap(),
            "https://github.com/example/base1"
        );

        assert_eq!(meta_map.general_base_model_name(1).unwrap(), "Base Model 2");
        assert_eq!(
            meta_map.general_base_model_author(1).unwrap(),
            "Base Author 2"
        );
        assert!(meta_map.general_base_model_version(1).is_err()); // 未设置的字段

        // 测试 datasets 数组字段
        let datasets: Vec<_> = meta_map
            .general_datasets()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(datasets, vec!["dataset1", "dataset2"]);
    }

    #[test]
    fn test_filetype_errors() {
        // 测试无效的 filetype 值
        let mut map = HashMap::new();
        map.insert("general.filetype".to_string(), (Ty::U32, encode_u32(999))); // 无效的值

        let meta_map = TestMetaMap { data: map };

        let result = meta_map.general_filetype();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GGufMetaError::OutOfRange));
    }

    #[test]
    fn test_llm_fields() {
        let meta_map = TestMetaMap::new();

        // 基本字段测试
        assert_eq!(meta_map.llm_context_length().unwrap(), 4096);
        assert_eq!(meta_map.llm_embedding_length().unwrap(), 4096);
        assert_eq!(meta_map.llm_block_count().unwrap(), 32);
        assert_eq!(meta_map.llm_feed_forward_length().unwrap(), 11008);
        assert!(meta_map.llm_use_parallel_residual().unwrap());
        assert_eq!(meta_map.llm_tensor_data_layout().unwrap(), "row-major");
        assert_eq!(meta_map.llm_expert_count().unwrap(), 0);
        assert_eq!(meta_map.llm_expert_used_count().unwrap(), 0);

        // attention 相关字段测试
        assert_eq!(meta_map.llm_attention_head_count().unwrap(), 32);
        assert_eq!(meta_map.llm_attention_head_count_kv().unwrap(), 32);
        assert_eq!(meta_map.llm_attention_layer_norm_epsilon().unwrap(), 1e-6);
        assert_eq!(
            meta_map.llm_attention_layer_norm_rms_epsilon().unwrap(),
            1e-5
        );
        assert_eq!(meta_map.llm_attention_max_alibi_bias().unwrap(), 8.0);
        assert_eq!(meta_map.llm_attention_clamp_kqv().unwrap(), 0.0);

        // rope 相关字段测试
        assert_eq!(meta_map.llm_rope_dimension_count().unwrap(), 128);
        assert_eq!(meta_map.llm_rope_freq_base().unwrap(), 10000.0);
        assert_eq!(meta_map.llm_rope_scaling_type().unwrap(), "linear");
        assert_eq!(meta_map.llm_rope_scaling_factor().unwrap(), 1.0);
        assert_eq!(
            meta_map.llm_rope_scaling_original_context_length().unwrap(),
            2048
        );
        assert!(!meta_map.llm_rope_scaling_finetuned().unwrap());
        assert_eq!(meta_map.llm_rope_scale_linear().unwrap(), 1.0);

        // ssm 相关字段测试
        assert_eq!(meta_map.llm_ssm_conv_kernel().unwrap(), 4);
        assert_eq!(meta_map.llm_ssm_inner_size().unwrap(), 256);
        assert_eq!(meta_map.llm_ssm_state_size().unwrap(), 16);
        assert_eq!(meta_map.llm_ssm_time_step_rank().unwrap(), 8);
    }

    #[test]
    fn test_tokenizer_fields() {
        let meta_map = TestMetaMap::new();

        // 测试基本属性
        assert_eq!(meta_map.tokenizer_ggml_model().unwrap(), "llama");
        assert_eq!(meta_map.tokenizer_ggml_bos_token_id().unwrap(), 1);
        assert_eq!(meta_map.tokenizer_ggml_eos_token_id().unwrap(), 2);
        assert_eq!(
            meta_map.tokenizer_chat_template().unwrap(),
            "<|im_start|>user\n{prompt}<|im_end|>"
        );

        // 测试可选的令牌 ID
        assert_eq!(meta_map.tokenizer_ggml_unknown_token_id().unwrap(), 0);
        assert_eq!(meta_map.tokenizer_ggml_separator_token_id().unwrap(), 3);
        assert_eq!(meta_map.tokenizer_ggml_padding_token_id().unwrap(), 4);

        // 测试 tokens 数组
        let tokens = meta_map
            .tokenizer_ggml_tokens()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], "<bos>");
        assert_eq!(tokens[1], "<eos>");
        assert_eq!(tokens[2], "hello");

        // 测试 scores 数组
        let scores = meta_map
            .tokenizer_ggml_scores()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 0.0).abs() < 1e-6);
        assert!((scores[1] - (-1.0)).abs() < 1e-6);
        assert!((scores[2] - (-2.0)).abs() < 1e-6);

        // 测试 token_type 数组
        let token_types = meta_map
            .tokenizer_ggml_token_type()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(token_types.len(), 3);
        assert_eq!(token_types[0], 1);
        assert_eq!(token_types[1], 1);
        assert_eq!(token_types[2], 0);

        // 测试 merges 数组
        let merges = meta_map
            .tokenizer_ggml_merges()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(merges.len(), 2);
        assert_eq!(merges[0], "h e");
        assert_eq!(merges[1], "he llo");

        // 测试 added_tokens 数组
        let added_tokens = meta_map
            .tokenizer_ggml_added_tokens()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(added_tokens.len(), 1);
        assert_eq!(added_tokens[0], "<custom_token>");

        // 测试 RWKV 特定字段
        assert_eq!(meta_map.tokenizer_rwkv_world().unwrap(), "world");

        // 测试错误处理（试图访问一个不存在的字段）
        assert!(matches!(
            meta_map.get_str("tokenizer.non_existent").unwrap_err(),
            GGufMetaError::NotExist
        ));
    }

    #[test]
    fn test_array_fields() {
        let meta_map = TestMetaMap::new();

        let tags: Vec<_> = meta_map
            .general_tags()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(tags, vec!["tag1", "tag2"]);

        let languages: Vec<_> = meta_map
            .general_languages()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(languages, vec!["en", "zh"]);
    }

    #[test]
    fn test_missing_fields() {
        let meta_map = TestMetaMap::new();

        // 对于未定义的字段，我们期望得到 NotExist 错误
        assert!(matches!(
            meta_map.general_license().unwrap_err(),
            GGufMetaError::NotExist
        ));
        assert!(matches!(
            meta_map.general_url().unwrap_err(),
            GGufMetaError::NotExist
        ));
        assert!(meta_map.llm_attention_head_count_kv().is_ok());
    }

    #[test]
    fn test_calculated_fields() {
        let meta_map = TestMetaMap::new();

        // 测试那些有默认计算逻辑的字段
        assert_eq!(meta_map.llm_attention_key_length().unwrap(), 128); // 4096 / 32
        assert_eq!(meta_map.llm_attention_value_length().unwrap(), 128); // 4096 / 32

        // 确保回退机制正常工作
        assert_eq!(meta_map.llm_attention_head_count_kv().unwrap(), 32); // 回退到 head_count
    }

    #[test]
    fn test_error_handling() {
        // 创建一个特殊的测试 map，添加一些错误的类型
        let mut map = HashMap::new();
        map.insert("wrong_type_field".to_string(), (Ty::String, encode_u32(42)));
        map.insert(
            "array_wrong_type".to_string(),
            (Ty::Array, {
                let mut data = Vec::new();
                // 写入类型为 U16 的数组头部
                data.extend_from_slice(&(Ty::U16 as u32).to_le_bytes());
                data.extend_from_slice(&(2u64).to_le_bytes());
                data
            }),
        );

        let meta_map = TestMetaMap { data: map };

        // 对于错误的数据编码
        let result = meta_map.get_str("wrong_type_field");
        assert!(result.is_err());
        assert!(matches!(result, Err(GGufMetaError::Read(_))));

        // 对于数组类型不匹配的情况
        let result = meta_map.get_str_arr("array_wrong_type");
        assert!(result.is_err());
        if let Err(err) = result {
            match err {
                GGufMetaError::ArrTypeMismatch(ty) => {
                    assert_eq!(ty, Ty::U16);
                }
                _ => panic!("Expected ArrTypeMismatch, got {:?}", err),
            }
        }

        // 测试字段不存在的情况
        let result = meta_map.get_str("non_existent_field");
        assert!(result.is_err());
        assert!(matches!(result, Err(GGufMetaError::NotExist)));
    }

    #[test]
    fn test_get_functions_errors() {
        // 创建一个包含多种类型的测试数据的 map
        let mut map = HashMap::new();

        // 添加不同类型的测试数据
        map.insert(
            "string_val".to_string(),
            (Ty::String, encode_string("test")),
        );
        map.insert("u32_val".to_string(), (Ty::U32, encode_u32(42)));
        map.insert("f32_val".to_string(), (Ty::F32, encode_f32(1.23)));
        map.insert("bool_val".to_string(), (Ty::Bool, encode_bool(true)));
        map.insert("corrupt_string".to_string(), (Ty::String, vec![1, 2, 3])); // 损坏的数据
        map.insert("wrong_size_u32".to_string(), (Ty::U32, vec![1, 2])); // 长度不足的数据

        // 创建数组类型测试数据
        map.insert(
            "string_array".to_string(),
            (Ty::Array, encode_string_array(&["test1", "test2"])),
        );
        map.insert(
            "f32_array".to_string(),
            (Ty::Array, {
                let mut data = Vec::new();
                data.extend_from_slice(&(Ty::F32 as u32).to_le_bytes());
                data.extend_from_slice(&(2u64).to_le_bytes());
                data.extend_from_slice(&(1.0f32).to_le_bytes());
                data.extend_from_slice(&(2.0f32).to_le_bytes());
                data
            }),
        );
        map.insert(
            "i32_array".to_string(),
            (Ty::Array, {
                let mut data = Vec::new();
                data.extend_from_slice(&(Ty::I32 as u32).to_le_bytes());
                data.extend_from_slice(&(2u64).to_le_bytes());
                data.extend_from_slice(&(10i32).to_le_bytes());
                data.extend_from_slice(&(20i32).to_le_bytes());
                data
            }),
        );
        map.insert("corrupt_array".to_string(), (Ty::Array, vec![1, 2, 3])); // 损坏的数组

        let meta_map = TestMetaMap { data: map };

        // 测试类型不匹配错误
        let result = meta_map.get_str("u32_val");
        assert!(matches!(result, Err(GGufMetaError::TypeMismatch(Ty::U32))));

        let result = meta_map.get_usize("string_val");
        assert!(matches!(
            result,
            Err(GGufMetaError::TypeMismatch(Ty::String))
        ));

        let result = meta_map.get_f32("bool_val");
        assert!(matches!(result, Err(GGufMetaError::TypeMismatch(Ty::Bool))));

        let result = meta_map.get_bool("f32_val");
        assert!(matches!(result, Err(GGufMetaError::TypeMismatch(Ty::F32))));

        let result = meta_map.get_u32("string_val");
        assert!(matches!(
            result,
            Err(GGufMetaError::TypeMismatch(Ty::String))
        ));

        // 测试数据损坏错误
        let result = meta_map.get_str("corrupt_string");
        assert!(matches!(result, Err(GGufMetaError::Read(_))));

        let result = meta_map.get_u32("wrong_size_u32");
        assert!(matches!(result, Err(GGufMetaError::Read(_))));

        // 测试数组相关错误
        let result = meta_map.get_str_arr("string_val");
        assert!(matches!(
            result,
            Err(GGufMetaError::TypeMismatch(Ty::String))
        ));

        let result = meta_map.get_f32_arr("i32_array");
        assert!(matches!(
            result,
            Err(GGufMetaError::ArrTypeMismatch(Ty::I32))
        ));

        let result = meta_map.get_i32_arr("f32_array");
        assert!(matches!(
            result,
            Err(GGufMetaError::ArrTypeMismatch(Ty::F32))
        ));

        let result = meta_map.get_str_arr("corrupt_array");
        assert!(matches!(result, Err(GGufMetaError::Read(_))));

        // 测试字段不存在的情况
        let result = meta_map.get_str("non_existent");
        assert!(matches!(result, Err(GGufMetaError::NotExist)));

        let result = meta_map.get_usize("non_existent");
        assert!(matches!(result, Err(GGufMetaError::NotExist)));

        let result = meta_map.get_f32("non_existent");
        assert!(matches!(result, Err(GGufMetaError::NotExist)));

        let result = meta_map.get_bool("non_existent");
        assert!(matches!(result, Err(GGufMetaError::NotExist)));

        let result = meta_map.get_u32("non_existent");
        assert!(matches!(result, Err(GGufMetaError::NotExist)));

        let result = meta_map.get_str_arr("non_existent");
        assert!(matches!(result, Err(GGufMetaError::NotExist)));

        let result = meta_map.get_f32_arr("non_existent");
        assert!(matches!(result, Err(GGufMetaError::NotExist)));

        let result = meta_map.get_i32_arr("non_existent");
        assert!(matches!(result, Err(GGufMetaError::NotExist)));
    }
}
