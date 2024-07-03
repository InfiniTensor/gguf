﻿use crate::loose_shards::LooseShards;
use ggus::{
    GGufFileHeader, GGufMetaDataValueType, GGufMetaKVPairs, GGufReadError, GGufTensors, GGufWriter,
};
use indexmap::{IndexMap, IndexSet};
use std::{fs::File, iter::zip, path::PathBuf};

#[derive(Args, Default)]
pub struct MergeArgs {
    #[clap(long, short)]
    file: PathBuf,
}

impl MergeArgs {
    pub fn merge(self) {
        let shards = LooseShards::from(&*self.file);
        if shards.count() < 2 {
            println!("Model does not need to be merged.");
            return;
        }

        let mut files = Vec::new();
        for path in &shards {
            match File::open(&path) {
                Ok(file) => files.push(unsafe { memmap2::Mmap::map(&file).unwrap() }),
                Err(e) => {
                    eprintln!("Failed to open");
                    eprintln!("  file: {}", path.display());
                    eprintln!("  cause: {e}");
                    return;
                }
            }
        }
        let files = files;

        let files = files
            .iter()
            .map(|data| GGufFile::new(data).unwrap())
            .collect::<Vec<_>>();

        let kvs = files
            .iter()
            .flat_map(|file| file.meta_kvs.kvs())
            .filter(|kv| {
                let key = kv.key();
                !key.starts_with("split.") && key != "general.alignment"
            })
            .collect::<IndexSet<_>>();
        let tensors = files
            .iter()
            .flat_map(|file| file.tensors.iter().map(move |t| (t, file.data)))
            .collect::<IndexMap<_, _>>();

        let out = File::create(shards.single_file()).unwrap();
        let header = GGufFileHeader::new(3, tensors.len() as _, (kvs.len() + 1) as _);
        let mut writer: GGufWriter<File> = GGufWriter::new(out, header).unwrap();

        let align = files
            .iter()
            .map(|file| file.meta_kvs.alignment())
            .max()
            .unwrap();

        writer
            .write_meta_kv(
                "general.alignment",
                GGufMetaDataValueType::U64,
                (align as u64).to_le_bytes(),
            )
            .unwrap();

        for kv in kvs {
            writer
                .write_meta_kv(kv.key(), kv.ty(), kv.value_bytes())
                .unwrap();
        }

        let mut cursor = 0;
        let mut paddings = Vec::with_capacity(tensors.len());
        for t in tensors.keys() {
            writer
                .write_tensor_info(t.name(), t.shape(), t.ggml_type(), cursor)
                .unwrap();
            let nbytes = t.nbytes();
            let length = (nbytes + align - 1) / align * align;
            cursor += length;
            paddings.push(length - nbytes);
        }

        let padding = (writer.written_bytes() + align - 1) / align * align - writer.written_bytes();
        for _ in 0..padding {
            writer.write(0u8).unwrap();
        }

        for ((t, data), padding) in zip(tensors, paddings) {
            writer
                .write_bytes(&data[t.offset()..][..t.nbytes()])
                .unwrap();
            for _ in 0..padding {
                writer.write(0u8).unwrap();
            }
        }
    }
}

struct GGufFile<'a> {
    meta_kvs: GGufMetaKVPairs<'a>,
    tensors: GGufTensors<'a>,
    data: &'a [u8],
}

#[derive(Debug)]
enum GGufError<'a> {
    MagicMismatch,
    EndianNotSupport,
    VersionNotSupport,
    #[allow(dead_code)]
    Reading(GGufReadError<'a>),
}

impl<'a> GGufFile<'a> {
    fn new(data: &'a [u8]) -> Result<Self, GGufError<'a>> {
        let header = unsafe { data.as_ptr().cast::<GGufFileHeader>().read() };
        if !header.is_magic_correct() {
            return Err(GGufError::MagicMismatch);
        }
        if !header.is_native_endian() {
            return Err(GGufError::EndianNotSupport);
        }
        if header.version != 3 {
            return Err(GGufError::VersionNotSupport);
        }

        let cursor = header.nbytes();
        let meta_kvs = GGufMetaKVPairs::scan(header.metadata_kv_count, &data[cursor..])
            .map_err(GGufError::Reading)?;

        let cursor = cursor + meta_kvs.nbytes();
        let tensors =
            GGufTensors::scan(header.tensor_count, &data[cursor..]).map_err(GGufError::Reading)?;

        let align = meta_kvs.alignment();
        let cursor = (cursor + tensors.nbytes() + align - 1) / align * align;
        Ok(Self {
            meta_kvs,
            tensors,
            data: &data[cursor..],
        })
    }
}
