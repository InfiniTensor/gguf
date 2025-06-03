mod shard;
mod size_label;
mod r#type;
mod version;

use shard::Shard;
use size_label::SizeLabel;
use std::{borrow::Cow, fmt, num::NonZero, path::Path};
use r#type::Type;
use version::Version;

/// [`GGufFileName`] 定义 GGUF 文件名的结构。
#[derive(Clone, Debug)]
pub struct GGufFileName<'a> {
    /// 基础名称，通常是模型的名称。
    pub base_name: Cow<'a, str>,
    /// 可选的大小标签，表示模型的大小。
    pub size_label: Option<SizeLabel>,
    /// 可选的微调信息，表示模型的微调版本。
    pub fine_tune: Cow<'a, str>,
    /// 版本信息，表示 GGUF 文件的版本。
    pub version: Version,
    /// 可选的编码信息，表示模型的编码格式。
    pub encoding: Option<Cow<'a, str>>,
    /// 模型类型，表示 GGUF 文件的类型（如 LoRA、Vocab 等）。
    pub type_: Type,
    /// 分片信息，表示 GGUF 文件的分片索引和总数。
    pub shard: Shard,
}

mod pattern {
    use regex::Regex;
    use std::sync::LazyLock;

    pub const NAME_: &str = r"-(\d+x)?(\d+)(\.\d+)?([QTBMK])(-\w+)?$";
    pub const VERSION_: &str = r"-v(\d+)\.(\d+)$";
    pub const TYPE_LORA: &str = r"-LoRA";
    pub const TYPE_VOCAB: &str = r"-vocab";
    pub const SHARD_: &str = r"-(\d{5})-of-(\d{5})$";
    pub const EXT: &str = ".gguf";

    pub static NAME: LazyLock<Regex> = LazyLock::new(|| Regex::new(NAME_).unwrap());
    pub static VERSION: LazyLock<Regex> = LazyLock::new(|| Regex::new(VERSION_).unwrap());
    pub static SHARD: LazyLock<Regex> = LazyLock::new(|| Regex::new(SHARD_).unwrap());
}

/// 错误类型，表示 GGUF 文件名不符合预期的扩展名。
#[derive(Debug)]
pub struct GGufExtNotMatch;

impl<'a> TryFrom<&'a str> for GGufFileName<'a> {
    type Error = GGufExtNotMatch;

    fn try_from(name: &'a str) -> Result<Self, Self::Error> {
        let Some(mut name) = name.strip_suffix(pattern::EXT) else {
            return Err(GGufExtNotMatch);
        };

        let shard = pattern::SHARD
            .captures(name)
            .map_or_else(Shard::default, |capture| {
                let (full, [index, count]) = capture.extract();
                name = &name[..name.len() - full.len()];
                Shard::new(index.parse().unwrap(), count.parse().unwrap())
            });

        let name_no_shard = name;
        let type_ = if name.ends_with(pattern::TYPE_VOCAB) {
            name = &name[..name.len() - pattern::TYPE_VOCAB.len()];
            Type::Vocab
        } else if name.ends_with(pattern::TYPE_LORA) {
            name = &name[..name.len() - pattern::TYPE_LORA.len()];
            Type::LoRA
        } else {
            Type::Default
        };

        let Some((head, encoding)) = name.rsplit_once('-') else {
            return Ok(Self {
                base_name: name_no_shard.into(),
                size_label: None,
                fine_tune: "".into(),
                version: Version::DEFAULT,
                encoding: None,
                type_,
                shard,
            });
        };
        name = head;

        let version = pattern::VERSION
            .captures(name)
            .map_or(Version::DEFAULT, |capture| {
                let (full, [major, minor]) = capture.extract();
                name = &name[..name.len() - full.len()];
                Version::new(major.parse().unwrap(), minor.parse().unwrap())
            });

        if let Some(capture) = pattern::NAME.captures(name) {
            let base_name = &name[..name.len() - capture.get(0).unwrap().len()];
            let e = capture.get(1).map_or(1, |m| {
                m.as_str().strip_suffix('x').unwrap().parse().unwrap()
            });
            let a = capture.get(2).unwrap().as_str().parse().unwrap();
            let b = capture.get(3).map_or(0, |m| {
                m.as_str().strip_prefix('.').unwrap().parse().unwrap()
            });
            let l = capture.get(4).unwrap().as_str().chars().next().unwrap();
            let fine_tune = capture
                .get(5)
                .map_or("", |m| m.as_str().strip_prefix('-').unwrap());

            Ok(Self {
                base_name: base_name.into(),
                size_label: Some(SizeLabel::new(e, a, b, l)),
                fine_tune: fine_tune.into(),
                version,
                encoding: Some(encoding.into()),
                type_,
                shard,
            })
        } else {
            Ok(Self {
                base_name: name_no_shard.into(),
                size_label: None,
                fine_tune: "".into(),
                version: Version::DEFAULT,
                encoding: None,
                type_,
                shard,
            })
        }
    }
}

impl<'a> TryFrom<&'a Path> for GGufFileName<'a> {
    type Error = GGufExtNotMatch;
    #[inline]
    fn try_from(value: &'a Path) -> Result<Self, Self::Error> {
        Self::try_from(value.file_name().unwrap().to_str().unwrap())
    }
}

impl GGufFileName<'_> {
    /// 计算 GGUF 文件名的分片数量。
    #[inline]
    pub fn shard_count(&self) -> usize {
        self.shard.count.get() as _
    }

    /// 将 GGUF 文件名转换为单个分片的名称。
    #[inline]
    pub fn into_single(self) -> Self {
        Self {
            shard: Default::default(),
            ..self
        }
    }

    /// 将 GGUF 文件名转换为迭代器，迭代所有分片。
    #[inline]
    pub fn iter_all(self) -> Self {
        Self {
            shard: Shard {
                index: NonZero::new(1).unwrap(),
                ..self.shard
            },
            ..self
        }
    }

    /// 将 GGUF 文件名分割为 n 个分片。
    #[inline]
    pub fn split_n(self, n: usize) -> Self {
        Self {
            shard: Shard {
                index: NonZero::new(1).unwrap(),
                count: NonZero::new(n as _).unwrap(),
            },
            ..self
        }
    }
}

impl Iterator for GGufFileName<'_> {
    type Item = Self;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.shard.index.get() <= self.shard.count.get() {
            let ans = self.clone();
            self.shard.index = self.shard.index.checked_add(1).unwrap();
            Some(ans)
        } else {
            None
        }
    }
}

impl fmt::Display for GGufFileName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.base_name)?;
        if let Some(size_label) = &self.size_label {
            write!(f, "-{size_label}")?
        }
        if !self.fine_tune.is_empty() {
            write!(f, "-{}", self.fine_tune)?
        }
        write!(f, "-{}", self.version)?;
        if let Some(encoding) = &self.encoding {
            write!(f, "-{}", encoding)?
        }
        write!(f, "{}", self.type_)?;
        write!(f, "{}", self.shard)?;
        write!(f, ".gguf")
    }
}

#[test]
fn test_name() {
    fn check(name: &str) {
        println!("{name} -> {}", GGufFileName::try_from(name).unwrap())
    }

    check("mmproj.gguf");
    check("FM9G-71B-F16.gguf");
    check("test-cases-00002-of-00005.gguf");
    check("Gpt-163M-v2.0-F32.gguf");
    check("TinyLlama-1.1B-Chat-v1.0-Q8_0.gguf");
    check("MiniCPM3-1B-sft-v0.0-F16.gguf");
    check("MiniCPM-V-Clip-1B-v2.6-F16.gguf");
}

#[test]
fn test_name_types() {
    let vocab_name = GGufFileName::try_from("tokenizer-vocab.gguf").unwrap();
    assert!(matches!(vocab_name.type_, Type::Vocab));
    assert_eq!(vocab_name.base_name, "tokenizer-vocab");
    assert_eq!(vocab_name.to_string(), "tokenizer-vocab-v1.0-Vocab.gguf");

    let lora_name = GGufFileName::try_from("adapter-LoRA.gguf").unwrap();
    assert!(matches!(lora_name.type_, Type::LoRA));
    assert_eq!(lora_name.base_name, "adapter-LoRA");
    assert_eq!(lora_name.to_string(), "adapter-LoRA-v1.0-LoRA.gguf");
}

#[test]
fn test_name_shard() {
    let name = GGufFileName::try_from("test-cases-00002-of-00005.gguf").unwrap();
    let expected = Shard::new(2, 5);
    assert_eq!(name.shard, expected);
    assert_eq!(name.shard.index, NonZero::new(2).unwrap());
    assert_eq!(name.shard.count, NonZero::new(5).unwrap());
    assert_eq!(name.shard_count(), 5);
    assert_eq!(name.iter_all().shard.index, NonZero::new(1).unwrap());
}

#[test]
fn test_name_errors() {
    assert!(GGufFileName::try_from("test-cases-00002-of-00005").is_err());
    assert!(GGufFileName::try_from("test-cases-00002-of-00005.ggufx").is_err());
    assert!(GGufFileName::try_from("test-cases-00002-of-00005.gguf.").is_err());
    assert!(GGufFileName::try_from("test-cases-00002-of-00005.gguf.abc").is_err());
}

#[test]
fn test_name_into_single() {
    let name = GGufFileName::try_from("test-cases-00002-of-00005.gguf").unwrap();
    assert_eq!(name.shard.index, NonZero::new(2).unwrap());
    assert_eq!(name.shard.count, NonZero::new(5).unwrap());
    let name = name.into_single();
    assert_eq!(name.shard.index, NonZero::new(1).unwrap());
    assert_eq!(name.shard.count, NonZero::new(1).unwrap());
}

#[test]
fn test_from_path() {
    use std::path::PathBuf;

    let path = PathBuf::from("/some/path/model-2x7.5B-F16.gguf");
    let name = GGufFileName::try_from(path.as_path()).unwrap();
    assert_eq!(name.base_name, "model");
    assert!(name.size_label.is_some());
    assert_eq!(name.size_label.as_ref().unwrap().to_string(), "2x7.5B");
    assert_eq!(name.encoding, Some("F16".into()));

    // 测试无效路径
    let invalid_path = PathBuf::from("/some/path/model.bin");
    assert!(GGufFileName::try_from(invalid_path.as_path()).is_err());
}

#[test]
fn test_iterator_implementation() {
    let name = GGufFileName::try_from("model-00001-of-00003.gguf").unwrap();

    // 测试迭代整个分片序列
    let mut iter = name.clone();
    let first = iter.next().unwrap();
    assert_eq!(first.shard.index, NonZero::new(1).unwrap());

    let second = iter.next().unwrap();
    assert_eq!(second.shard.index, NonZero::new(2).unwrap());

    let third = iter.next().unwrap();
    assert_eq!(third.shard.index, NonZero::new(3).unwrap());

    assert!(iter.next().is_none());

    // 测试 split_n 方法
    let original = GGufFileName::try_from("model-v1.0-F16-00002-of-00003.gguf").unwrap();
    let split = original.clone().split_n(5);

    assert_eq!(split.shard.index, NonZero::new(1).unwrap());
    assert_eq!(split.shard.count, NonZero::new(5).unwrap());

    assert_eq!(split.base_name, original.base_name);
    assert_eq!(split.version, original.version);
    assert_eq!(split.encoding, original.encoding);
    assert_eq!(split.type_, original.type_);

    let all_shards: Vec<_> = split.collect();
    assert_eq!(all_shards.len(), 5);
    assert_eq!(all_shards[0].shard.index, NonZero::new(1).unwrap());
    assert_eq!(all_shards[4].shard.index, NonZero::new(5).unwrap());
}
