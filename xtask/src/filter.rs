﻿use crate::utils::{operate, show_file_info, Operator, OutputConfig};
use ggus::GGufFileName;
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct FilterArgs {
    /// The file to filter
    file: PathBuf,
    /// Output directory for filtered file
    #[clap(long, short)]
    output_dir: Option<PathBuf>,
    /// Meta to keep
    #[clap(long, short = 'm', default_value = "*")]
    filter_meta: String,
    /// Tensors to keep
    #[clap(long, short = 't', default_value = "*")]
    filter_tensor: String,
}

impl FilterArgs {
    pub fn filter(self) {
        let Self {
            file,
            output_dir,
            filter_meta,
            filter_tensor,
        } = self;

        let files = operate(
            GGufFileName::try_from(&*file).unwrap(),
            [&file],
            [
                Operator::filter_meta_key(filter_meta),
                Operator::filter_tensor_name(filter_tensor),
            ],
            OutputConfig {
                dir: output_dir,
                shard_max_tensor_count: usize::MAX,
                shard_max_file_size: Default::default(),
                shard_no_tensor_first: false,
            },
        )
        .unwrap();

        show_file_info(&files);
    }
}
