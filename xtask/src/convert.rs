use crate::{
    LogArgs, list_files, merge_shards,
    utils::{Operator, OutputArgs, operate, show_file_info},
};
use std::collections::HashMap;

#[derive(Args, Default)]
pub struct ConvertArgs {
    /// File to convert
    file_pattern: String,
    /// Steps to apply, separated by "->", maybe "sort", "permute-qk", "merge-linear", "split-linear", "to-llama:<extra>", "cast:<types>", "filter-meta:<key>" or "filter-tensor:<name>"
    #[clap(long, short = 'x')]
    steps: String,

    #[clap(flatten)]
    output: OutputArgs,
    #[clap(flatten)]
    log: LogArgs,
}

impl ConvertArgs {
    pub fn convert(self) {
        let Self {
            file_pattern,
            steps,
            output,
            log,
        } = self;
        log.init();

        let files = list_files(&file_pattern).collect::<Vec<_>>();
        if files.is_empty() {
            println!("No such file.");
            return;
        }

        let files = operate(
            merge_shards(&files).to_owned(),
            files,
            steps.split("->").map(|op| match op.trim() {
                "sort" => Operator::SortTensors,
                "permute-qk" => Operator::PermuteQK(true),
                "permute-qk-rev" | "!permute-qk" => Operator::PermuteQK(false),
                "merge-linear" => Operator::MergeLinear(true),
                "split-linear" | "!merge-linear" => Operator::MergeLinear(false),
                "to-llama" => Operator::ToLlama(HashMap::new()),
                op => match op.split_once(':') {
                    Some(("cast", types)) => Operator::cast(types),
                    Some(("to-llama", extra)) => Operator::to_llama(extra),
                    Some(("filter-meta", key)) => Operator::filter_meta_key(key),
                    Some(("filter-tensor", name)) => Operator::filter_tensor_name(name),
                    _ => panic!("Unsupported operation: {op}"),
                },
            }),
            output.into(),
        )
        .unwrap();

        show_file_info(&files);
    }
}
