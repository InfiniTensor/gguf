# gguf 实用工具

[![CI](https://github.com/InfiniTensor/gguf/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/InfiniTensor/gguf/actions)
[![Latest version](https://img.shields.io/crates/v/gguf-utils.svg)](https://crates.io/crates/gguf-utils)
[![license](https://img.shields.io/github/license/InfiniTensor/gguf)](https://mit-license.org/)

[![GitHub Issues](https://img.shields.io/github/issues/InfiniTensor/gguf)](https://github.com/InfiniTensor/gguf/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/InfiniTensor/gguf)](https://github.com/InfiniTensor/gguf/pulls)
![GitHub repo size](https://img.shields.io/github/repo-size/InfiniTensor/gguf)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/InfiniTensor/gguf)
![GitHub contributors](https://img.shields.io/github/contributors/InfiniTensor/gguf)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/InfiniTensor/gguf)

## 帮助信息

```shell
gguf-utils --help
```

或

```shell
# in project dir
cargo xtask --help
```

```plaintext
gguf-utils is a command-line tool for working with gguf files

Usage: gguf-utils <COMMAND>

Commands:
  show      Show the contents of gguf files
  split     Split gguf files into shards
  merge     Merge shards into a single gguf file
  convert   Convert gguf files to different format
  set-meta  Set metadata of gguf files
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

## 参数说明

下述命令的描述中：

- 所有 `<FILE>` 表示指定单个文件，可以是相对路径也可以是绝对路径；
- 所有 `<FILE_PATTERN>` 表示 GLOB 通配符，即可以包含 `.`、`..`、`*` 和 `**` 等符号的文件通配符。
  > **NOTICE** 部分 shell 会直接将通配符展开为多个路径再传递给进程，将导致参数数量错误。在这种 shell 中需要用引号包围通配符传入。

## 阅读内容

```shell
gguf-utils show --help
```

或

```shell
# in project dir
cargo show --help
```

```plaintext
Show the contents of gguf files

Usage: gguf-utils show [OPTIONS] <FILE_PATTERN>

Arguments:
  <FILE_PATTERN>  The file to show

Options:
  -n, --array-detail <ARRAY_DETAIL>    How many elements to show in arrays, `all` for all elements [default: 8]
  -m, --filter-meta <FILTER_META>      Meta to show [default: *]
  -t, --filter-tensor <FILTER_TENSOR>  Tensors to show [default: *]
      --log <LOG>                      Log level, may be "off", "trace", "debug", "info" or "error"
  -h, --help                           Print help
```

## 分片

```shell
gguf-utils split --help
```

或

```shell
# in project dir
cargo split --help
```

```plaintext
Split gguf files into shards

Usage: gguf-utils split [OPTIONS] <FILE>

Arguments:
  <FILE>  File to split

Options:
  -o, --output-dir <OUTPUT_DIR>    Output directory for converted files
  -t, --max-tensors <MAX_TENSORS>  Max count of tensors per shard
  -s, --max-bytes <MAX_BYTES>      Max size in bytes per shard
      --no-tensor-first            If set, the first shard will not contain any tensor
      --no-data                    If set, tensor data will not be written to output files
      --log <LOG>                  Log level, may be "off", "trace", "debug", "info" or "error"
  -h, --help                       Print help
```

## 合并

```shell
gguf-utils merge --help
```

或

```shell
# in project dir
cargo merge --help
```

```plaintext
Merge shards into a single gguf file

Usage: gguf-utils merge [OPTIONS] <FILE_PATTERN>

Arguments:
  <FILE_PATTERN>  Glob pattern to match shards

Options:
  -o, --output-dir <OUTPUT_DIR>  Output directory for merged file
      --no-data                  If set, tensor data will not be written to output files
      --log <LOG>                Log level, may be "off", "trace", "debug", "info" or "error"
  -h, --help                     Print help
```

## 转换格式

```shell
gguf-utils convert --help
```

或

```shell
# in project dir
cargo convert --help
```

```plaintext
Convert gguf files to different format

Usage: gguf-utils convert [OPTIONS] --steps <STEPS> <FILE_PATTERN>

Arguments:
  <FILE_PATTERN>  File to convert

Options:
  -x, --steps <STEPS>              Steps to apply, separated by "->", maybe "sort", "permute-qk", "merge-linear", "split-linear", "to-llama:<extra>", "cast:<types>", "filter-meta:<key>" or "filter-tensor:<name>"
  -o, --output-dir <OUTPUT_DIR>    Output directory for converted files
  -t, --max-tensors <MAX_TENSORS>  Max count of tensors per shard
  -s, --max-bytes <MAX_BYTES>      Max size in bytes per shard
      --no-tensor-first            If set, the first shard will not contain any tensor
      --no-data                    If set, tensor data will not be written to output files
      --log <LOG>                  Log level, may be "off", "trace", "debug", "info" or "error"
  -h, --help                       Print help
```

## 修改元信息

```shell
gguf-utils set-meta --help
```

或

```shell
# in project dir
cargo set-meta --help
```

```plaintext
Set metadata of gguf files

Usage: gguf-utils set-meta [OPTIONS] <FILE> <META_KVS>

Arguments:
  <FILE>      File to set metadata
  <META_KVS>  Meta data to set for the file

Options:
  -o, --output-dir <OUTPUT_DIR>    Output directory for converted files
  -t, --max-tensors <MAX_TENSORS>  Max count of tensors per shard
  -s, --max-bytes <MAX_BYTES>      Max size in bytes per shard
      --no-tensor-first            If set, the first shard will not contain any tensor
      --no-data                    If set, tensor data will not be written to output files
      --log <LOG>                  Log level, may be "off", "trace", "debug", "info" or "error"
  -h, --help                       Print help
```

`<META_KVS>` 是具有特定格式的字符串或文本文件路径。工具将先检查文件是否为路径，如果是则从文件读取，否则视作字符串字面量。

格式要求如下：

1. 配置代数类型元信息

   > 代数类型包括整型、无符号整型、浮点型和布尔。

   ```plaintext
   '<KEY>'<Ty> <VAL>
   ```

2. 配置字符串元信息

   单行字符串：

   ```plaintext
   '<KEY>'str "<VAL>"
   ```

   多行字符串：

   ```plaintext
   '<KEY>'str<Sep>
   <Sep> [Content]
   <Sep> [Content]
   <Sep> [Content]

   ```

   其中 `Sep` 是表示字符串继续的分隔符。必须紧邻 `str`，之间不能包含空白字符，且分隔符中也不能包含空白字符。
   连续的多行字符串，每行必须以分隔符+空格起始，此行后续所有字符（包括换行符）都被视作多行字符串的内容，不转义。
   任何不以分隔符开始的行（包括空行）都将结束多行字符串。

3. 配置数组元信息

   TODO: 当前此功能未实现。

这是一个配置元信息的示例文件内容：

```plaintext
'llama.block_count'             u64 22
'llama.context_length'          u64 2048
'llama.embedding_length'        u64 2048
'llama.feed_forward_length'     u64 5632
'llama.attention.head_count'    u64 32
'llama.attention.head_count_kv' u64 4
'llama.rope.dimension_count'    u64 64

'tokenizer.chat_template' str|
| {%- for message in messages -%}
| {%- if message['role'] == 'user' -%}
| {{ '<|user|>
| ' + message['content'] + eos_token }}
| {%- elif message['role'] == 'system' -%}
| {{ '<|system|>
| ' + message['content'] + eos_token }}
| {%- elif message['role'] == 'assistant' -%}
| {{ '<|assistant|>
| ' + message['content'] + eos_token }}
| {%- endif -%}
| {%- if loop.last and add_generation_prompt -%}
| {{ '<|assistant|>
| ' }}
| {%- endif -%}
| {%- endfor -%}
```
