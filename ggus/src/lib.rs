#![doc = include_str!("../README.md")]
#![deny(warnings)]

mod header;
mod metadata;
mod name;
mod reader;
mod tensor;
mod writer;

pub use header::GGufFileHeader;
pub use metadata::{utok, GGufArray, GGufFileType, GGufMetaDataValueType, GGufMetaKVPairs};
pub use name::GGufFileName;
pub use reader::{GGmlReadError, GGmlReader};
pub use tensor::GGufTensors;
pub use writer::GGufWriter;

#[test]
fn test_read() {
    use std::fs::File;

    let Some(args) = std::option_env!("ARGS") else {
        return;
    };

    let file = File::open(args).unwrap();
    let file = unsafe { memmap2::Mmap::map(&file) }.unwrap();

    let header = unsafe { file.as_ptr().cast::<GGufFileHeader>().read() };
    assert!(header.is_magic_correct());
    assert!(header.is_native_endian());
    assert!(header.version == 3);
    println!("{header:?}");
    println!();

    let cursor = sizeof!(GGufFileHeader);
    let pairs = GGufMetaKVPairs::scan(header.metadata_kv_count, &file[cursor..]).unwrap();
    for key in pairs.keys() {
        println!("{key}");
    }
    println!();

    let cursor = cursor + pairs.nbytes();
    let tensors = GGufTensors::scan(header.tensor_count, &file[cursor..]).unwrap();
    for name in tensors.names() {
        let tensor = tensors.get(name).unwrap();
        println!(
            "{}:\t{:?}\t+{:#x}\t{:?}",
            tensor.name(),
            tensor.ggml_type(),
            tensor.offset(),
            tensor.shape()
        );
    }
}

macro_rules! sizeof {
    ($ty:ty) => {
        std::mem::size_of::<$ty>()
    };
}

use sizeof;