# ggus

[![CI](https://github.com/InfiniTensor/gguf/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/InfiniTensor/gguf/actions)
[![Latest version](https://img.shields.io/crates/v/ggus.svg)](https://crates.io/crates/ggus)
[![Documentation](https://docs.rs/ggus/badge.svg)](https://docs.rs/ggus)
[![license](https://img.shields.io/github/license/InfiniTensor/gguf)](https://mit-license.org/)

[![GitHub Issues](https://img.shields.io/github/issues/InfiniTensor/gguf)](https://github.com/InfiniTensor/gguf/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/InfiniTensor/gguf)](https://github.com/InfiniTensor/gguf/pulls)
![GitHub repo size](https://img.shields.io/github/repo-size/InfiniTensor/gguf)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/InfiniTensor/gguf)
![GitHub contributors](https://img.shields.io/github/contributors/InfiniTensor/gguf)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/InfiniTensor/gguf)

`ggus` 是一个高性能的 Rust 实现的 GGUF (GGML Unified Format) 文件格式处理库。

## 项目简介

GGUF 是一种用于存储大型语言模型权重和元数据的文件格式，提出于 `llama.cpp` 项目。

该库提供了全面的 API 用于读取、解析、修改和创建 GGUF 文件，支持所有 GGUF 规范中定义的数据类型和元数据。`ggus` 以 Rust 的安全性、性能和并发能力为基础，为 LLM 模型的部署和管理提供了可靠的工具。

更多细节请参考 GGUF 的[官方规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)。

## 主要特性

该库的核心功能和特性包括：
- 完整支持 GGUF 文件格式的读取和写入；
- 高效的内存管理和数据访问；
- 丰富的元数据处理功能；
- 张量数据的便捷访问；
- 严格的类型检查和错误处理；
- 零拷贝设计，最小化内存占用；
- 完全兼容 GGML 生态系统；

## 使用示例

下面的示例展示了创建并写入 GGUF 文件、读取 GGUF 文件头的过程：

```rust
use ggus::{
    DataFuture, GGufFileHeader, GGufFileWriter, GGufMetaDataValueType, GGufTensorWriter, GGmlType,
    GGufReader, GGufReadError,
};
use std::fs::File;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ========== 写入部分 ==========
    let file = File::create("new_model.gguf")?;
    let header = GGufFileHeader::new(3, 0, 0);

    let mut writer = GGufFileWriter::new(file, header)?;
    
    // 写入元数据
    writer.write_alignment(32)?;
    writer.write_meta_kv("general.architecture", GGufMetaDataValueType::String, b"llama\0")?;
    writer.write_meta_kv("general.name", GGufMetaDataValueType::String, b"My Model\0")?;
    writer.write_meta_kv("llm.context_length", GGufMetaDataValueType::U32, &2048u32.to_le_bytes())?;

    // 写入张量
    let mut tensor_writer = writer.finish::<Vec<u8>>(true);
    let shape = [4, 4];
    let data_bytes = vec![
        1u8, 0, 0, 0,   // f32 = 1.0 little endian
        0, 0, 128, 63,  // f32 = 1.0
        0, 0, 0, 64,    // f32 = 2.0
        0, 0, 64, 64,   // f32 = 3.0
        0, 0, 128, 64,  // f32 = 4.0
        0, 0, 160, 64,  // f32 = 5.0
        0, 0, 176, 64,  // f32 = 6.0
        0, 0, 192, 64,  // f32 = 8.0
        0, 0, 208, 64,  // f32 = 6.5 
        0, 0, 224, 64,  // f32 = 7.0
        0, 0, 240, 64,  // f32 = 7.5
        0, 0, 0, 65,    // f32 = 8.0
        0, 0, 16, 65,   // f32 = 9.0
        0, 0, 32, 65,   // f32 = 10.0
        0, 0, 48, 65,   // f32 = 11.0
        0, 0, 64, 65,   // f32 = 12.0
    ];

    tensor_writer.write_tensor("weight", GGmlType::F32, &shape, data_bytes)?;
    tensor_writer.finish()?;

    // ========== 读取部分 ==========
    let data = std::fs::read("new_model.gguf")?;
    let file_name = std::path::Path::new("new_model.gguf").file_name().unwrap().to_str().unwrap();
    println!("文件名: {file_name}\n");

    let mut reader = GGufReader::new(&data);
    let header = match reader.read_header() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("读取 GGUF 头失败: {:?}", e);
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "读取 GGUF 头失败",
            )));
        }
    };

    println!(
        "字节序: {}",
        if header.is_native_endian() {
            "Native"
        } else {
            "Swapped"
        }
    );
    println!("版本号: {}", header.version);
    println!("元数据键值对数量: {}", header.metadata_kv_count);
    println!("张量数量: {}", header.tensor_count);

    Ok(())
}
```

更详细的示例可以参考[示例代码](https://github.com/InfiniTensor/gguf/blob/main/xtask/src/show.rs)，它展示了如何打印 GGUF 文件的内容。

## 应用场景

`ggus` 库适用于以下场景：

1. LLM 模型部署工具：用于构建加载、转换和优化 LLM 模型的工具；
2. 模型格式转换：在不同的模型格式之间进行转换，如 PyTorch/Safetensors 到 GGUF；
3. 量化工具：为模型参数实现不同的量化策略；
4. 模型检查和分析：检查模型结构和权重；
5. 自定义 LLM 推理引擎：为 Rust 编写的推理引擎提供模型加载功能；
6. 模型合并和裁剪：合并多个模型或移除不必要的层；
