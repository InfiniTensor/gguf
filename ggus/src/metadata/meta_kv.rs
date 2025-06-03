use super::GGufMetaDataValueType as Ty;
use crate::{GGufReadError, GGufReader};
use std::marker::PhantomData;

/// [`GGufMetaKV`] 结构体表示 GGUF 文件中的元数据键值对。
#[derive(Clone)]
#[repr(transparent)]
pub struct GGufMetaKV<'a>(&'a [u8]);

impl<'a> GGufReader<'a> {
    /// 读取元数据键值对。
    pub fn read_meta_kv(&mut self) -> Result<GGufMetaKV<'a>, GGufReadError> {
        let data = self.remaining();

        let _k = self.read_str()?;
        let ty = self.read()?;
        self.read_meta_value(ty, 1)?;

        let data = &data[..data.len() - self.remaining().len()];
        Ok(unsafe { GGufMetaKV::new_unchecked(data) })
    }

    /// 读取元数据值。
    fn read_meta_value(&mut self, ty: Ty, len: usize) -> Result<&mut Self, GGufReadError> {
        match ty {
            Ty::U8 => self.skip::<u8>(len),
            Ty::I8 => self.skip::<i8>(len),
            Ty::U16 => self.skip::<u16>(len),
            Ty::I16 => self.skip::<i16>(len),
            Ty::U32 => self.skip::<u32>(len),
            Ty::I32 => self.skip::<i32>(len),
            Ty::F32 => self.skip::<f32>(len),
            Ty::U64 => self.skip::<u64>(len),
            Ty::I64 => self.skip::<i64>(len),
            Ty::F64 => self.skip::<f64>(len),
            Ty::Bool => {
                for _ in 0..len {
                    self.read_bool()?;
                }
                Ok(self)
            }
            Ty::String => {
                for _ in 0..len {
                    self.read_str()?;
                }
                Ok(self)
            }
            Ty::Array => {
                let (ty, len) = self.read_arr_header()?;
                self.read_meta_value(ty, len)
            }
        }
    }
}

impl<'a> GGufMetaKV<'a> {
    /// 创建一个新的 [`GGufMetaKV`] 实例，不检查数据合法性。
    ///
    /// # Safety
    ///
    /// 调用此方法时必须确保 `data` 是有效的 GGUF 元数据键值对格式。
    #[inline]
    pub const unsafe fn new_unchecked(data: &'a [u8]) -> Self {
        Self(data)
    }

    /// 创建一个新的 [`GGufMetaKV`] 实例。
    #[inline]
    pub fn new(data: &'a [u8]) -> Result<Self, GGufReadError> {
        GGufReader::new(data).read_meta_kv()
    }

    /// 获取元数据键值对的键。
    #[inline]
    pub fn key(&self) -> &'a str {
        let mut reader = self.reader();
        unsafe { reader.read_str_unchecked() }
    }

    /// 获取元数据键值对的类型。
    #[inline]
    pub fn ty(&self) -> Ty {
        self.reader().skip_str().unwrap().read().unwrap()
    }

    /// 获取元数据键值对的值字节。
    pub fn value_bytes(&self) -> &'a [u8] {
        self.value_reader().remaining()
    }

    /// 获取元数据键值对的值读取器。
    pub fn value_reader(&self) -> GGufReader<'a> {
        let mut reader = self.reader();
        reader.skip_str().unwrap().skip::<Ty>(1).unwrap();
        reader
    }

    /// 读取整数类型的值。
    pub fn read_integer(&self) -> isize {
        let mut reader = self.reader();
        let ty = reader.skip_str().unwrap().read::<Ty>().unwrap();
        match ty {
            Ty::Bool | Ty::U8 => reader.read::<u8>().unwrap().into(),
            Ty::I8 => reader.read::<i8>().unwrap().into(),
            Ty::U16 => reader.read::<u16>().unwrap().try_into().unwrap(),
            Ty::I16 => reader.read::<i16>().unwrap().into(),
            Ty::U32 => reader.read::<u32>().unwrap().try_into().unwrap(),
            Ty::I32 => reader.read::<i32>().unwrap().try_into().unwrap(),
            Ty::U64 => reader.read::<u64>().unwrap().try_into().unwrap(),
            Ty::I64 => reader.read::<i64>().unwrap().try_into().unwrap(),
            Ty::Array | Ty::String | Ty::F32 | Ty::F64 => panic!("not an integer type"),
        }
    }

    /// 读取无符号整数类型的值。
    pub fn read_unsigned(&self) -> usize {
        let mut reader = self.reader();
        let ty = reader.skip_str().unwrap().read::<Ty>().unwrap();
        match ty {
            Ty::Bool | Ty::U8 => reader.read::<u8>().unwrap().into(),
            Ty::U16 => reader.read::<u16>().unwrap().into(),
            Ty::U32 => reader.read::<u32>().unwrap().try_into().unwrap(),
            Ty::U64 => reader.read::<u64>().unwrap().try_into().unwrap(),
            Ty::I8 => reader.read::<i8>().unwrap().try_into().unwrap(),
            Ty::I16 => reader.read::<i16>().unwrap().try_into().unwrap(),
            Ty::I32 => reader.read::<i32>().unwrap().try_into().unwrap(),
            Ty::I64 => reader.read::<i64>().unwrap().try_into().unwrap(),
            Ty::Array | Ty::String | Ty::F32 | Ty::F64 => panic!("not an integer type"),
        }
    }

    #[inline]
    fn reader(&self) -> GGufReader<'a> {
        GGufReader::new(self.0)
    }
}

/// [`GGufMetaValueArray`] 结构体表示 GGUF 文件中元数据值的数组。
pub struct GGufMetaValueArray<'a, T: ?Sized> {
    reader: GGufReader<'a>,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: ?Sized> GGufMetaValueArray<'a, T> {
    /// 创建一个新的 [`GGufMetaValueArray`] 实例。
    pub const fn new(reader: GGufReader<'a>, len: usize) -> Self {
        Self {
            reader,
            len,
            _phantom: PhantomData,
        }
    }

    /// 检查数组是否为空。
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 获取数组的长度。
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }
}

impl<'a> Iterator for GGufMetaValueArray<'a, str> {
    type Item = Result<&'a str, GGufReadError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            self.len -= 1;
            Some(self.reader.read_str())
        } else {
            None
        }
    }
}

impl<T: Copy> Iterator for GGufMetaValueArray<'_, T> {
    type Item = Result<T, GGufReadError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            self.len -= 1;
            Some(self.reader.read())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 测试辅助函数：构建元数据键值对二进制数据
    fn build_kv_data(key: &str, ty: Ty, value: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();

        // 写入键
        let key_len = key.len() as u64;
        data.extend_from_slice(&key_len.to_le_bytes());
        data.extend_from_slice(key.as_bytes());

        // 写入类型
        let ty_val = ty as u32;
        data.extend_from_slice(&ty_val.to_le_bytes());

        // 写入值
        data.extend_from_slice(value);

        data
    }

    // 编码各种基础类型数据
    fn encode_u64(value: u64) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn encode_i64(value: i64) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn encode_u32(value: u32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn encode_i32(value: i32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn encode_u16(value: u16) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn encode_i16(value: i16) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn encode_f32(value: f32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn encode_f64(value: f64) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn encode_string(value: &str) -> Vec<u8> {
        let mut data = Vec::new();
        let str_len = value.len() as u64;
        data.extend_from_slice(&str_len.to_le_bytes());
        data.extend_from_slice(value.as_bytes());
        data
    }

    fn encode_bool(value: bool) -> Vec<u8> {
        vec![if value { 1 } else { 0 }]
    }

    fn encode_array_header(ty: Ty, len: usize) -> Vec<u8> {
        let mut data = Vec::new();
        let ty_val = ty as u32;
        data.extend_from_slice(&ty_val.to_le_bytes());
        data.extend_from_slice(&(len as u64).to_le_bytes());
        data
    }

    #[test]
    fn test_meta_kv_creation() {
        // 测试基本的键值对创建
        let value_bytes = encode_u32(42);
        let data = build_kv_data("test_key", Ty::U32, &value_bytes);

        // 使用 new 方法创建
        let kv = GGufMetaKV::new(&data).unwrap();

        // 验证基本属性
        assert_eq!(kv.key(), "test_key");
        assert_eq!(kv.ty(), Ty::U32);

        // 测试 value_bytes 和 value_reader
        assert_eq!(kv.value_bytes(), &value_bytes);

        // 创建一个 reader 并验证是否能正确读取值
        let mut reader = kv.value_reader();
        assert_eq!(reader.read::<u32>().unwrap(), 42);
    }

    #[test]
    fn test_read_integer_types() {
        // 测试不同整数类型的读取
        let test_cases = [
            (Ty::U8, &[42u8][..], 42isize),
            (Ty::I8, &[-42i8 as u8][..], -42isize),
            (Ty::U16, &encode_u16(1000)[..], 1000isize),
            (Ty::I16, &encode_i16(-1000)[..], -1000isize),
            (Ty::U32, &encode_u32(100000)[..], 100000isize),
            (Ty::I32, &encode_i32(-100000)[..], -100000isize),
            (Ty::U64, &encode_u64(10000000000)[..], 10000000000isize),
            (Ty::I64, &encode_i64(-10000000000)[..], -10000000000isize),
            (Ty::Bool, &[1][..], 1isize),
        ];

        for (ty, value_bytes, expected) in test_cases {
            let data = build_kv_data("int_key", ty, value_bytes);
            let kv = GGufMetaKV::new(&data).unwrap();
            assert_eq!(kv.read_integer(), expected);
        }
    }

    #[test]
    fn test_read_unsigned() {
        // 测试不同无符号整数类型的读取
        let test_cases = [
            (Ty::U8, &[42u8][..], 42usize),
            (Ty::I8, &[42i8 as u8][..], 42usize),
            (Ty::U16, &encode_u16(1000)[..], 1000usize),
            (Ty::I16, &encode_i16(1000)[..], 1000usize),
            (Ty::U32, &encode_u32(100000)[..], 100000usize),
            (Ty::I32, &encode_i32(100000)[..], 100000usize),
            (Ty::U64, &encode_u64(10000000000)[..], 10000000000usize),
            (Ty::I64, &encode_i64(10000000000)[..], 10000000000usize),
            (Ty::Bool, &[1][..], 1usize),
        ];

        for (ty, value_bytes, expected) in test_cases {
            let data = build_kv_data("uint_key", ty, value_bytes);
            let kv = GGufMetaKV::new(&data).unwrap();
            assert_eq!(kv.read_unsigned(), expected);
        }
    }

    #[test]
    fn test_float_values() {
        // 测试浮点数类型的键值对
        let f32_value = 42.5f32;
        let f64_value = 123.456f64;

        // 构建测试数据并测试
        let f32_data = build_kv_data("f32_key", Ty::F32, &encode_f32(f32_value));
        let kv = GGufMetaKV::new(&f32_data).unwrap();
        assert_eq!(kv.ty(), Ty::F32);
        let mut reader = kv.value_reader();
        assert_eq!(reader.read::<f32>().unwrap(), f32_value);

        let f64_data = build_kv_data("f64_key", Ty::F64, &encode_f64(f64_value));
        let kv = GGufMetaKV::new(&f64_data).unwrap();
        assert_eq!(kv.ty(), Ty::F64);
        let mut reader = kv.value_reader();
        assert_eq!(reader.read::<f64>().unwrap(), f64_value);
    }

    #[test]
    fn test_string_values() {
        // 测试字符串类型的键值对
        let test_str = "Hello, GGUF!";
        let value_bytes = encode_string(test_str);
        let data = build_kv_data("string_key", Ty::String, &value_bytes);

        let kv = GGufMetaKV::new(&data).unwrap();
        assert_eq!(kv.ty(), Ty::String);

        let mut reader = kv.value_reader();
        assert_eq!(reader.read_str().unwrap(), test_str);
    }

    #[test]
    fn test_bool_values() {
        // 测试布尔类型的键值对
        let test_cases = [(true, 1usize), (false, 0usize)];

        for (bool_val, int_val) in test_cases {
            let value_bytes = encode_bool(bool_val);
            let data = build_kv_data("bool_key", Ty::Bool, &value_bytes);

            let kv = GGufMetaKV::new(&data).unwrap();
            assert_eq!(kv.ty(), Ty::Bool);

            // 使用 read_unsigned 读取布尔值
            assert_eq!(kv.read_unsigned(), int_val);

            let mut reader = kv.value_reader();
            assert_eq!(reader.read_bool().unwrap(), bool_val);
        }
    }

    #[test]
    fn test_array_values() {
        // 测试数组类型的键值对 - 字符串数组
        let strings = ["first", "second", "third"];

        // 构建字符串数组的值
        let mut value_bytes = encode_array_header(Ty::String, strings.len());
        for s in &strings {
            value_bytes.extend_from_slice(&encode_string(s));
        }

        let data = build_kv_data("array_key", Ty::Array, &value_bytes);
        let kv = GGufMetaKV::new(&data).unwrap();
        assert_eq!(kv.ty(), Ty::Array);

        // 使用 GGufMetaValueArray 读取数组
        let mut reader = kv.value_reader();
        let (ty, len) = reader.read_arr_header().unwrap();
        assert_eq!(ty, Ty::String);
        assert_eq!(len, strings.len());

        let array = GGufMetaValueArray::<str>::new(reader, len);
        let result: Result<Vec<&str>, _> = array.collect();
        assert_eq!(result.unwrap(), strings);

        // 测试整数数组
        let numbers = [10, 20, 30, 40];

        // 构建整数数组的值
        let mut value_bytes = encode_array_header(Ty::U32, numbers.len());
        for &n in &numbers {
            value_bytes.extend_from_slice(&encode_u32(n));
        }

        let data = build_kv_data("number_array", Ty::Array, &value_bytes);
        let kv = GGufMetaKV::new(&data).unwrap();

        let mut reader = kv.value_reader();
        let (ty, len) = reader.read_arr_header().unwrap();
        assert_eq!(ty, Ty::U32);
        assert_eq!(len, numbers.len());

        let array = GGufMetaValueArray::<u32>::new(reader, len);
        let result: Result<Vec<u32>, _> = array.collect();
        assert_eq!(result.unwrap(), numbers);
    }

    #[test]
    fn test_meta_value_array_helpers() {
        // 测试 GGufMetaValueArray 辅助方法
        let reader = GGufReader::new(&[]);
        let empty_array = GGufMetaValueArray::<str>::new(reader, 0);
        assert!(empty_array.is_empty());
        assert_eq!(empty_array.len(), 0);

        let reader = GGufReader::new(&[]);
        let non_empty_array = GGufMetaValueArray::<u32>::new(reader, 5);
        assert!(!non_empty_array.is_empty());
        assert_eq!(non_empty_array.len(), 5);
    }

    #[test]
    fn test_read_integer_wrong_type() {
        use std::panic::catch_unwind;

        // 测试尝试从非整数类型读取整数时的行为
        let string_value = encode_string("not an integer");
        let data = build_kv_data("wrong_type", Ty::String, &string_value);

        let kv = GGufMetaKV::new(&data).unwrap();

        // 使用 catch_unwind 捕获 panic
        let result_int = catch_unwind(|| {
            kv.read_integer();
        });
        let result_uint = catch_unwind(|| {
            kv.read_unsigned();
        });

        // 验证确实发生了 panic
        assert!(result_int.is_err());
        assert!(result_uint.is_err());
    }
}
