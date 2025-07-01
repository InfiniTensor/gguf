#![deny(warnings)]

mod convert;
mod diff;
mod merge;
mod set_meta;
mod show;
mod split;
mod utils;

#[macro_use]
extern crate clap;

use clap::Parser;
use ggus::GGufFileName;
use log::warn;
use std::path::Path;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Show(args) => args.show(),
        Split(args) => args.split(),
        Merge(args) => args.merge(),
        Convert(args) => args.convert(),
        Diff(args) => args.diff(),
        SetMeta(args) => args.set_meta(),
    }
}

/// gguf-utils is a command-line tool for working with gguf files.
#[derive(Parser)]
#[clap(version)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show the contents of gguf files
    Show(show::ShowArgs),
    /// Split gguf files into shards
    Split(split::SplitArgs),
    /// Merge shards into a single gguf file
    Merge(merge::MergeArgs),
    /// Convert gguf files to different format
    Convert(convert::ConvertArgs),
    /// Diff two gguf files
    Diff(diff::DiffArgs),
    /// Set metadata of gguf files
    SetMeta(set_meta::SetMetaArgs),
}

#[derive(Args, Default)]
struct LogArgs {
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,
}

impl LogArgs {
    fn init(self) {
        use colored::{Color, Colorize};
        use flexi_logger::DeferredNow;
        use log::{Level, Record};

        let level = self
            .log
            .map_or("info", |level| match level.to_lowercase().as_str() {
                "all" | "trace" => "trace",
                "debug" => "debug",
                "info" => "info",
                "error" | "off" | "none" => "error",
                level => panic!("Unknown log level `{level}`"),
            });
        let config = format!("error, xtask={level}, ggus={level}");

        // <https://docs.rs/flexi_logger/0.30.1/flexi_logger/struct.LogSpecification.html>
        flexi_logger::Logger::try_with_env_or_str(config)
            .unwrap()
            .format(log_format)
            .start()
            .unwrap();

        fn log_format(
            w: &mut dyn std::io::Write,
            now: &mut DeferredNow,
            record: &Record,
        ) -> Result<(), std::io::Error> {
            let color = match record.level() {
                Level::Error => Color::Red,
                Level::Warn => Color::Yellow,
                Level::Info => Color::Green,
                Level::Debug => Color::BrightCyan,
                Level::Trace => Color::BrightBlack,
            };
            write!(
                w,
                "{} {:<5} [{}] {}",
                now.format_rfc3339().color(Color::BrightBlack),
                record.level().to_string().color(color),
                record.module_path().unwrap_or("<unnamed>"),
                record.args().to_string().color(color),
            )
        }
    }
}

fn list_files(pattern: &str) -> impl Iterator<Item = std::path::PathBuf> {
    glob::glob(pattern)
        .unwrap()
        .filter_map(|res| res.ok())
        .filter(|p| {
            log::trace!("glob match {}", p.display());
            p.is_file() || p.is_symlink()
        })
}

fn merge_shards<T: AsRef<Path>>(files: &[T]) -> GGufFileName {
    files
        .iter()
        .map(|name| GGufFileName::try_from(name.as_ref().file_name().unwrap().to_str().unwrap()))
        .collect::<Result<Vec<_>, _>>()
        .ok()
        .and_then(|names| GGufFileName::merge_shards(&names))
        .unwrap_or_else(|| {
            warn!("file names mismatch, use default name as output");
            GGufFileName::default()
        })
}
