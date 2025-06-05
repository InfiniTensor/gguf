use crate::{
    LogArgs, list_files, merge_shards,
    utils::{OutputConfig, operate, show_file_info},
};
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct MergeArgs {
    /// Glob pattern to match shards
    file_pattern: String,
    /// Output directory for merged file
    #[clap(long, short)]
    output_dir: Option<PathBuf>,
    /// If set, tensor data will not be written to output files
    #[clap(long)]
    no_data: bool,

    #[clap(flatten)]
    log: LogArgs,
}

impl MergeArgs {
    pub fn merge(self) {
        let Self {
            file_pattern,
            output_dir,
            no_data,
            log,
        } = self;
        log.init();

        let files = list_files(&file_pattern).collect::<Vec<_>>();
        match files.len() {
            0 => {
                println!("No such file");
                return;
            }
            1 => {
                println!("Files does not need to merge.");
                return;
            }
            _ => {}
        }

        let files = operate(
            merge_shards(&files).to_owned(),
            files,
            [],
            OutputConfig {
                dir: output_dir,
                shard_max_tensor_count: usize::MAX,
                shard_max_file_size: Default::default(),
                shard_no_tensor_first: false,
                write_data: !no_data,
            },
        )
        .unwrap();

        show_file_info(&files)
    }
}
