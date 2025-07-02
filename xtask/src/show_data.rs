use ggus::{
    GGuf,
    ggml_quants::{
        bf16,
        digit_layout::{DigitLayout, types},
        f16,
    },
};
use mem_rearrange::ndarray_layout::ArrayLayout;
use memmap2::Mmap;
use std::{fmt, fs::File, path::PathBuf};

#[derive(Args, Default)]
pub struct ShowDataArgs {
    /// Name of file to show
    file: PathBuf,
    /// Name of tensor to show
    tensor: String,
}

impl ShowDataArgs {
    pub fn show(self) {
        let Self { file, tensor } = self;
        let file = File::open(&file).unwrap();
        let file = unsafe { Mmap::map(&file) }.unwrap();
        let gguf = GGuf::new(&file).unwrap();
        println!("{}", Fmt::new(gguf, &tensor))
    }
}

struct Fmt<'a> {
    ty: DigitLayout,
    layout: ArrayLayout<3>,
    data: &'a [u8],
}

impl<'a> Fmt<'a> {
    fn new(gguf: GGuf<'a>, name: &str) -> Self {
        let tensor = gguf
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("tensor `{name}` not exist in this file"))
            .to_info();
        let ty = tensor.ty().to_digit_layout();
        Self {
            ty,
            layout: ArrayLayout::<3>::new_contiguous(
                &tensor
                    .shape()
                    .iter()
                    .rev()
                    .map(|d| *d as usize)
                    .collect::<Vec<_>>(),
                mem_rearrange::ndarray_layout::Endian::BigEndian,
                tensor.ty().size().type_size as _,
            ),
            data: &gguf.data[tensor.offset()..][..tensor.nbytes()],
        }
    }
}

impl fmt::Display for Fmt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        macro_rules! write_arr {
            ($ty:ty) => {
                self.layout
                    .write_array(f, self.data.as_ptr().cast::<DataFmt<$ty>>())
            };
        }

        match self.ty {
            types::F16 => unsafe { write_arr!(f16) },
            types::BF16 => unsafe { write_arr!(bf16) },
            types::F32 => unsafe { write_arr!(f32) },
            types::U32 => unsafe { write_arr!(u32) },
            types::U64 => unsafe { write_arr!(u64) },
            others => panic!("Unsupported data type {others}"),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
struct DataFmt<T>(T);

impl fmt::Display for DataFmt<f16> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == f16::ZERO {
            write_zero(f)
        } else {
            write_f32(f, self.0.to_f32())
        }
    }
}

impl fmt::Display for DataFmt<bf16> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == bf16::ZERO {
            write_zero(f)
        } else {
            write_f32(f, self.0.to_f32())
        }
    }
}

impl fmt::Display for DataFmt<f32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0. {
            write_zero(f)
        } else {
            write_f32(f, self.0)
        }
    }
}

impl fmt::Display for DataFmt<u32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write_zero(f)
        } else {
            write_int(f, self.0)
        }
    }
}

impl fmt::Display for DataFmt<u64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write_zero(f)
        } else {
            write_int(f, self.0)
        }
    }
}

#[inline(always)]
fn write_zero(f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, " ________")
}

#[inline(always)]
fn write_f32(f: &mut fmt::Formatter, data: f32) -> fmt::Result {
    write!(f, "{data:>9.3e}")
}

#[inline(always)]
fn write_int(f: &mut fmt::Formatter, data: impl fmt::Display) -> fmt::Result {
    write!(f, "{data:>6}")
}
