use super::GGufWriter;
use crate::{DEFAULT_ALIGNMENT, GGmlType, GGufFileHeader, GGufMetaDataValueType, pad};
use log::trace;
use std::{
    borrow::Borrow,
    io::{Result, Write},
    time::Instant,
};

/// 简化的 GGUF 文件写入器，负责将 GGUF 文件头和元数据写入到文件中。
pub struct GGufFileWriter<T: Write> {
    writer: GGufWriter<T>,
    alignment: usize,
}

/// 完整的 GGUF 张量写入器，负责将张量数据写入到 GGUF 文件中。
pub struct GGufTensorWriter<T: Write, U> {
    writer: GGufWriter<T>,
    alignment: usize,
    data: Vec<U>,
    offset: usize,
    write_data: bool,
}

/// Trait 用于获取数据的未来值，允许从不同类型中获取字节切片。
pub trait DataFuture {
    /// 获取数据的字节切片。
    fn get(&self) -> &[u8];
}

impl<T: Borrow<[u8]>> DataFuture for T {
    /// 将借用的字节切片转换为 &[u8]。
    #[inline]
    fn get(&self) -> &[u8] {
        self.borrow()
    }
}

impl<T: Write> GGufFileWriter<T> {
    /// 创建一个新的 [`GGufFileWriter`] 实例，初始化 GGUF 文件头。
    #[inline]
    pub fn new(writer: T, header: GGufFileHeader) -> Result<Self> {
        let mut writer = GGufWriter::new(writer);
        writer.write_header(header)?;
        Ok(Self {
            writer,
            alignment: DEFAULT_ALIGNMENT,
        })
    }

    /// 使用指定的对齐值创建 [`GGufFileWriter`] 实例。
    #[inline]
    pub fn with_alignment(writer: T, header: GGufFileHeader, alignment: usize) -> Result<Self> {
        let mut ans = Self::new(writer, header)?;
        ans.write_alignment(alignment)?;
        Ok(ans)
    }

    /// 写入新的对齐值，并更新内部状态。
    #[inline]
    pub fn write_alignment(&mut self, alignment: usize) -> Result<()> {
        self.writer.write_alignment(alignment)?;
        self.alignment = alignment;
        Ok(())
    }

    /// 写入元数据键值对，如果键为 `general.alignment`，则更新对齐值。
    #[inline]
    pub fn write_meta_kv(
        &mut self,
        key: &str,
        ty: GGufMetaDataValueType,
        val: &[u8],
    ) -> Result<()> {
        if let Some(alignment) = self.writer.write_meta_kv(key, ty, val)? {
            self.alignment = alignment;
        }
        Ok(())
    }

    /// 完成元数据写入，并返回一个 [`GGufTensorWriter`] 实例。
    #[inline]
    pub fn finish<U>(self, write_data: bool) -> GGufTensorWriter<T, U> {
        GGufTensorWriter {
            writer: self.writer,
            alignment: self.alignment,
            data: Vec::new(),
            offset: 0,
            write_data,
        }
    }
}

impl<T: Write, U: DataFuture> GGufTensorWriter<T, U> {
    /// 写入张量数据。
    pub fn write_tensor(&mut self, name: &str, ty: GGmlType, shape: &[u64], data: U) -> Result<()> {
        self.offset += pad(self.offset, self.alignment);
        self.writer
            .write_tensor_info(name, shape, ty, self.offset as _)
            .unwrap();

        let len = ty.size().elements_to_bytes(shape);
        self.offset += len;
        if self.write_data {
            self.data.push(data)
        }
        Ok(())
    }

    /// 完成张量写入，返回已写入的字节数。
    pub fn finish(self) -> Result<usize> {
        let Self {
            mut writer,
            alignment,
            data,
            ..
        } = self;

        let total = data.len().to_string();
        let width = total.len();
        for (i, data) in data.into_iter().enumerate() {
            let t0 = Instant::now();
            let data = data.get();
            let t1 = Instant::now();
            writer.write_padding(alignment)?;
            writer.write_data(data)?;
            let t2 = Instant::now();
            trace!(
                "data {i:>width$}/{total} size = {} bytes, calculate in {:?}, write in {:?}",
                data.len(),
                t1 - t0,
                t2 - t1,
            )
        }
        Ok(writer.written_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GGufFileHeader;
    use std::io::Cursor;

    // 测试辅助函数，创建一个标准的文件头
    fn create_test_header() -> GGufFileHeader {
        GGufFileHeader::new(3, 0, 0)
    }

    #[test]
    fn test_file_writer_new() {
        // 测试基本创建功能
        let cursor = Cursor::new(Vec::new());
        let writer = GGufFileWriter::new(cursor, create_test_header()).unwrap();

        // 确认默认对齐值
        assert_eq!(writer.alignment, DEFAULT_ALIGNMENT);
    }

    #[test]
    fn test_file_writer_with_alignment() {
        // 测试使用自定义对齐值创建
        let cursor = Cursor::new(Vec::new());
        let alignment = 64;
        let writer =
            GGufFileWriter::with_alignment(cursor, create_test_header(), alignment).unwrap();

        // 确认对齐值正确设置
        assert_eq!(writer.alignment, alignment);
    }

    #[test]
    fn test_write_alignment() {
        // 测试更改对齐值
        let cursor = Cursor::new(Vec::new());
        let mut writer = GGufFileWriter::new(cursor, create_test_header()).unwrap();

        // 设置新的对齐值
        let new_alignment = 128;
        writer.write_alignment(new_alignment).unwrap();

        // 确认对齐值已更新
        assert_eq!(writer.alignment, new_alignment);
    }

    #[test]
    fn test_write_meta_kv() {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        // 测试写入元数据键值对
        let cursor = Cursor::new(Vec::new());
        let mut writer = GGufFileWriter::new(cursor, create_test_header()).unwrap();

        // 写入常规键值对
        writer
            .write_meta_kv("test.key", GGufMetaDataValueType::U32, &[1, 0, 0, 0])
            .unwrap();

        // 写入可能更改对齐值的键值对
        writer
            .write_meta_kv(
                "general.alignment",
                GGufMetaDataValueType::U32,
                &[64, 0, 0, 0],
            )
            .unwrap();

        // 对齐值应该被更新
        assert_eq!(writer.alignment, 64);

        // 写入非 u32 类型的键值对，触发 panic
        let result = catch_unwind(AssertUnwindSafe(|| {
            writer
                .write_meta_kv(
                    "general.alignment",
                    GGufMetaDataValueType::String,
                    b"test\0",
                )
                .unwrap();
        }));
        assert!(result.is_err(), "Expected panic for non-u32 value type");
    }

    #[test]
    fn test_finish_and_tensor_writer() {
        // 测试完成元数据写入并转换为张量写入器
        let cursor = Cursor::new(Vec::new());
        let writer = GGufFileWriter::new(cursor, create_test_header()).unwrap();

        // 创建张量写入器，不实际写入数据
        let tensor_writer = writer.finish::<Vec<u8>>(false);

        // 验证对齐值被正确传递
        assert_eq!(tensor_writer.alignment, DEFAULT_ALIGNMENT);
        assert_eq!(tensor_writer.offset, 0);
        assert!(tensor_writer.data.is_empty());
        assert!(!tensor_writer.write_data);
    }

    #[test]
    fn test_tensor_writer_write_tensor() {
        // 测试张量写入
        let cursor = Cursor::new(Vec::new());
        let writer = GGufFileWriter::new(cursor, create_test_header()).unwrap();

        // 创建张量写入器，保存数据
        let mut tensor_writer = writer.finish::<Vec<u8>>(true);

        // 写入张量
        let shape = [2, 3];
        let data = vec![0u8; 24]; // 假设是2x3的f32数据
        tensor_writer
            .write_tensor("test_tensor", GGmlType::F32, &shape, data.clone())
            .unwrap();

        // 验证偏移量和数据缓存
        assert_eq!(tensor_writer.offset, 24); // 2*3*4=24字节
        assert_eq!(tensor_writer.data.len(), 1);
        assert_eq!(tensor_writer.data[0].get(), data.as_slice());
    }

    #[test]
    fn test_tensor_writer_multiple_tensors() {
        // 测试写入多个张量
        let cursor = Cursor::new(Vec::new());
        let writer = GGufFileWriter::new(cursor, create_test_header()).unwrap();

        // 创建张量写入器
        let mut tensor_writer = writer.finish::<Vec<u8>>(true);

        // 写入第一个张量
        let shape1 = [2, 3];
        let data1 = vec![0u8; 24];
        tensor_writer
            .write_tensor("tensor1", GGmlType::F32, &shape1, data1)
            .unwrap();

        // 写入第二个张量，考虑对齐
        let shape2 = [4, 4];
        let data2 = vec![0u8; 64]; // f16数据，每个元素2字节
        tensor_writer
            .write_tensor("tensor2", GGmlType::F16, &shape2, data2)
            .unwrap();

        // 验证张量数据和偏移量
        assert_eq!(tensor_writer.data.len(), 2);

        // 考虑对齐后的偏移量
        let expected_offset = 24 + pad(24, DEFAULT_ALIGNMENT) + 32;
        assert_eq!(tensor_writer.offset, expected_offset);
    }

    #[test]
    fn test_tensor_writer_finish() {
        // 测试完成张量写入
        let cursor = Cursor::new(Vec::new());
        let writer = GGufFileWriter::new(cursor, create_test_header()).unwrap();

        // 创建张量写入器
        let mut tensor_writer = writer.finish::<Vec<u8>>(true);

        // 写入张量
        let shape = [2, 2];
        let data = vec![0u8; 16]; // 2x2的f32数据
        tensor_writer
            .write_tensor("test_tensor", GGmlType::F32, &shape, data)
            .unwrap();

        // 完成写入
        let bytes_written = tensor_writer.finish().unwrap();

        // 验证写入的字节数
        assert!(bytes_written > 16); // 应该包括头部、元数据和张量数据
    }

    #[test]
    fn test_end_to_end_write_process() {
        // 端到端测试，完整的写入流程
        let cursor = Cursor::new(Vec::new());
        let header = GGufFileHeader::new(3, 0, 0);

        // 创建文件写入器
        let mut writer = GGufFileWriter::new(cursor, header).unwrap();

        // 写入元数据
        writer.write_alignment(64).unwrap();
        writer
            .write_meta_kv(
                "general.architecture",
                GGufMetaDataValueType::String,
                b"llama\0",
            )
            .unwrap();
        writer
            .write_meta_kv(
                "general.name",
                GGufMetaDataValueType::String,
                b"test_model\0",
            )
            .unwrap();
        writer
            .write_meta_kv(
                "llm.context_length",
                GGufMetaDataValueType::U32,
                &4096u32.to_le_bytes(),
            )
            .unwrap();

        // 完成元数据写入并创建张量写入器
        let mut tensor_writer = writer.finish::<Vec<u8>>(true);

        // 写入第一个张量
        let shape1 = [5, 5];
        let data1 = vec![0u8; 100]; // f32数据，每个元素4字节
        tensor_writer
            .write_tensor("embeddings", GGmlType::F32, &shape1, data1)
            .unwrap();

        // 写入第二个张量
        let shape2 = [10, 20];
        let data2 = vec![0u8; 400]; // f32数据
        tensor_writer
            .write_tensor("weights", GGmlType::F32, &shape2, data2)
            .unwrap();

        // 完成写入
        let total_bytes = tensor_writer.finish().unwrap();

        // 验证总写入字节数
        assert!(total_bytes > 500); // 头部+元数据+两个张量+对齐填充
    }
}
