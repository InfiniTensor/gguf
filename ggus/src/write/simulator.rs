use super::GGufWriter;
use crate::{DEFAULT_ALIGNMENT, GGmlType, GGufMetaDataValueType, pad};
use std::io::{Result, Write};

/// 简化的 GGUF 文件模拟器。
pub struct GGufFileSimulator {
    writer: GGufWriter<NWrite>,
    alignment: usize,
}

/// 完整的 GGUF 文件模拟器。
pub struct GGufTensorSimulator {
    writer: GGufWriter<NWrite>,
    alignment: usize,
    data: Vec<usize>,
    offset: usize,
}

impl Default for GGufFileSimulator {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl GGufFileSimulator {
    /// 创建一个新的 GGUF 文件模拟器。
    #[inline]
    pub fn new() -> Self {
        let mut writer = GGufWriter::new(NWrite);
        writer.write_header(Default::default()).unwrap();
        Self {
            writer,
            alignment: DEFAULT_ALIGNMENT,
        }
    }

    /// 使用指定的对齐值创建一个新的 GGUF 文件模拟器。
    #[inline]
    pub fn with_alignment(alignment: usize) -> Self {
        let mut ans = Self::new();
        ans.write_alignment(alignment);
        ans
    }

    /// 写入新的对齐值，并更新内部状态。
    #[inline]
    pub fn write_alignment(&mut self, alignment: usize) {
        self.writer.write_alignment(alignment).unwrap();
        self.alignment = alignment;
    }

    /// 写入元数据键值对，如果键为 `general.alignment`，则更新对齐值。
    #[inline]
    pub fn write_meta_kv(&mut self, key: &str, ty: GGufMetaDataValueType, val: &[u8]) {
        if let Some(alignment) = self.writer.write_meta_kv(key, ty, val).unwrap() {
            self.alignment = alignment;
        }
    }

    /// 完成模拟器的构建，返回一个 [`GGufTensorSimulator`]。
    #[inline]
    pub fn finish(self) -> GGufTensorSimulator {
        GGufTensorSimulator {
            writer: self.writer,
            alignment: self.alignment,
            data: Vec::new(),
            offset: 0,
        }
    }
}

impl GGufTensorSimulator {
    /// 写入张量数据。
    pub fn write_tensor(&mut self, name: &str, ty: GGmlType, shape: &[u64]) {
        self.offset += pad(self.offset, self.alignment);
        self.writer
            .write_tensor_info(name, shape, ty, self.offset as _)
            .unwrap();

        let len = ty.size().elements_to_bytes(shape);
        self.offset += len;
        self.data.push(len);
    }

    /// 获取已写入的字节数。
    pub fn written_bytes(&self) -> usize {
        let mut total = self.writer.written_bytes();
        for len in &self.data {
            total += pad(total, self.alignment);
            total += len;
        }
        total
    }
}

struct NWrite;

impl Write for NWrite {
    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        Ok(buf.len())
    }
    #[inline(always)]
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn test_simulator() {
    // 测试默认构造
    let mut sim = GGufFileSimulator::new();
    sim.write_meta_kv("test_key", GGufMetaDataValueType::String, b"test_value");
    sim.write_alignment(16);
    let mut tensor_sim = sim.finish();
    tensor_sim.write_tensor("test_tensor", GGmlType::F32, &[2, 2]);
    assert_eq!(tensor_sim.written_bytes(), 160);

    // 测试 default 构造函数
    let sim_default = GGufFileSimulator::default();
    assert_eq!(sim_default.alignment, DEFAULT_ALIGNMENT);

    // 测试 with_alignment 构造函数
    let sim2 = GGufFileSimulator::with_alignment(32);
    assert_eq!(sim2.alignment, 32);

    // 测试 flush 方法不会 panic
    let mut writer = NWrite;
    writer.flush().unwrap();
}
