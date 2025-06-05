# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-06-05

### Changed

- Introduce [*glob*](https://crates.io/crates/glob) to select multiple shard files using glob wildcards;

## [0.2.0] - 2025-02-24

### Added

- Add subcommand `diff` to diff two gguf files;
- Add q8 to f32 dequantize cast;

### Changed

- Upgrade Rust to 2024 edition;
- Upgrade dependency `ggus` 0.4 to 0.5;
- Format every file;

[Unreleased]: https://github.com/InfiniTensor/gguf/compare/v0.5.1...HEAD
[0.3.0]: https://github.com/InfiniTensor/gguf/compare/v0.5.0...v0.5.1
[0.2.0]: https://github.com/InfiniTensor/gguf/releases/tag/v0.5.0
