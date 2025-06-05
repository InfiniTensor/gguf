# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1] - 2025-06-05

### Changed

- Improve documentation and testing;
- Make `GGufFileName::Version` optional;

## [0.5.0] - 2025-02-24

### Changed

- Upgrade Rust to 2024 edition;
- Upgrade dependency `ggml-quants` 0.0 to 0.1;
- Replace dependency `fancy-regex` 0.14 with `regex` 1.11;
- Refactor GGufFileName parsing to support shard for non-standard filenames;
- Format every file;

[Unreleased]: https://github.com/InfiniTensor/gguf/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/InfiniTensor/gguf/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/InfiniTensor/gguf/releases/tag/v0.5.0
