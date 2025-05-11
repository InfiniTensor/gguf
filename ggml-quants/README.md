# ggml-quants

[![CI](https://github.com/InfiniTensor/gguf/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/InfiniTensor/gguf/actions)
[![Latest version](https://img.shields.io/crates/v/ggml-quants.svg)](https://crates.io/crates/ggml-quants)
[![Documentation](https://docs.rs/ggml-quants/badge.svg)](https://docs.rs/ggml-quants)
[![license](https://img.shields.io/github/license/InfiniTensor/gguf)](https://mit-license.org/)

[![GitHub Issues](https://img.shields.io/github/issues/InfiniTensor/gguf)](https://github.com/InfiniTensor/gguf/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/InfiniTensor/gguf)](https://github.com/InfiniTensor/gguf/pulls)
![GitHub repo size](https://img.shields.io/github/repo-size/InfiniTensor/gguf)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/InfiniTensor/gguf)
![GitHub contributors](https://img.shields.io/github/contributors/InfiniTensor/gguf)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/InfiniTensor/gguf)

`ggml-quants` 是一个 Rust 库，用于实现 `ggml` 定义的量化数据类型及其对应的量化和反量化算法。

## 项目简介

`ggml-quants` 提供了一组高效的量化工具，用于将浮点数数据压缩为更小的量化格式（如 `Q4_0`, `Q8_1` 等），并支持从量化数据还原为浮点数。  

该库的核心功能包括：
- 支持多种量化格式（如 `Q4_0`, `Q8_0`, `Q8_1` 等）；
- 提供通用的量化和反量化接口；
- 使用并行化技术（基于 `rayon`）提升大规模数据处理性能；

## 使用示例

```rust
use ggml_quants::{Quantize, Q8_1};

// 原始浮点数数据
let data: [f32; 32] = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
    1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
    2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
];

// 量化数据
let quantized = Q8_1::quantize(&data);

// 反量化数据
let dequantized: [f32; 32] = quantized.dequantize();
```

## 应用场景

在 `gguf` 项目中，模型权重通常以浮点数形式存储（如 `f32` 或 `f16`），这会占用大量内存，成为限制性能的主要因素。通过使用 `ggml-quants` 提供的量化工具，可以在能够容忍的精度损失下，将权重从 `f32` 压缩为更小的格式（如 `Q4_0` 或 `Q8_1`），从而：

- 减少存储空间：降低模型部署对硬件内存的需求；
- 加快加载速度：量化后的权重文件更小，加载时间显著减少；
- 提升推理效率：在支持量化计算的硬件（如 GPU 或专用加速器）上，推理速度可以显著提升；
