//! Header-only shard index with bounded, on-demand tensor reads.
//! Supports the dtypes needed by the standalone OLMoE converter.
use crate::backend::invalid;
use serde::{
    de::{self, MapAccess, Visitor},
    Deserialize,
};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    path::Path,
    sync::Mutex,
};

const MAX_HEADER_BYTES: u64 = 100_000_000;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub enum DType {
    I8,
    F16,
    BF16,
    F32,
}
impl DType {
    fn size(self) -> usize {
        match self {
            Self::I8 => 1,
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    dtype: DType,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub byte_len: usize,
    shard: usize,
    offset: u64,
}

// Reject duplicate JSON keys rather than silently replacing tensor descriptors.
struct Header(BTreeMap<String, serde_json::Value>);
impl<'de> Deserialize<'de> for Header {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = Header;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("unique tensor names")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Header, M::Error> {
                let mut entries = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                    if entries.insert(key.clone(), value).is_some() {
                        return Err(de::Error::custom(format!("duplicate tensor: {key}")));
                    }
                }
                Ok(Header(entries))
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

pub struct TensorIndex {
    tensors: BTreeMap<String, TensorInfo>,
    shards: Vec<Mutex<File>>,
}
impl TensorIndex {
    /// Scans shard headers only. Checkpoint files must remain unchanged after indexing.
    pub fn open(directory: impl AsRef<Path>) -> io::Result<Self> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                paths.push(path);
            }
        }
        paths.sort();
        if paths.is_empty() {
            return Err(invalid("no .safetensors shards found"));
        }
        let mut index = Self {
            tensors: BTreeMap::new(),
            shards: Vec::new(),
        };
        for path in paths {
            index.add_shard(&path)?;
        }
        Ok(index)
    }

    fn add_shard(&mut self, path: &Path) -> io::Result<()> {
        let mut file = File::open(path)?;
        let shard = self.shards.len();
        let file_len = file.metadata()?.len();
        let mut prefix = [0; 8];
        file.read_exact(&mut prefix)?;
        let header_len = u64::from_le_bytes(prefix);
        let start = header_len
            .checked_add(8)
            .ok_or_else(|| invalid("header overflow"))?;
        if header_len == 0 || header_len > MAX_HEADER_BYTES || start > file_len {
            return Err(invalid("invalid Safetensors header length"));
        }
        let mut header = vec![0; header_len as usize];
        file.read_exact(&mut header)?;
        if header.first() != Some(&b'{') {
            return Err(invalid("header must start with '{'"));
        }
        let Header(entries) =
            serde_json::from_slice(&header).map_err(|e| invalid(e.to_string()))?;
        let mut ranges = Vec::new();
        for (name, value) in entries {
            if name == "__metadata__" {
                serde_json::from_value::<BTreeMap<String, String>>(value)
                    .map_err(|e| invalid(e.to_string()))?;
                continue;
            }
            let descriptor: Descriptor =
                serde_json::from_value(value).map_err(|e| invalid(format!("{name}: {e}")))?;
            let elements = descriptor
                .shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or_else(|| invalid("tensor shape overflow"))?;
            let bytes = elements
                .checked_mul(descriptor.dtype.size())
                .ok_or_else(|| invalid("tensor byte length overflow"))?;
            let [begin, end] = descriptor.data_offsets;
            if end < begin || end > file_len - start || end - begin != bytes as u64 {
                return Err(invalid(format!("{name}: invalid tensor offsets or shape")));
            }
            ranges.push((begin, end));
            let info = TensorInfo {
                dtype: descriptor.dtype,
                shape: descriptor.shape,
                byte_len: bytes,
                shard,
                offset: start + begin,
            };
            if self.tensors.insert(name.clone(), info).is_some() {
                return Err(invalid(format!("duplicate tensor across shards: {name}")));
            }
        }
        ranges.sort_unstable();
        let mut cursor = 0;
        for (begin, end) in ranges {
            if begin != cursor {
                return Err(invalid("tensor data has gaps or overlaps"));
            }
            cursor = end;
        }
        if cursor != file_len - start {
            return Err(invalid("unindexed tensor data"));
        }
        self.shards.push(Mutex::new(file));
        Ok(())
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }
    pub fn tensor(&self, name: &str) -> io::Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor: {name}")))
    }

    /// Caller supplies a limit before any payload allocation (raw bytes only).
    pub fn read_raw(&self, name: &str, max_bytes: usize) -> io::Result<Vec<u8>> {
        let info = self.tensor(name)?;
        if info.byte_len > max_bytes {
            return Err(invalid(format!("{name}: tensor exceeds read budget")));
        }
        let mut data = vec![0; info.byte_len];
        // Keep seek and read under one lock: readers share this file's cursor.
        // Allocate before locking so other readers can keep using the shard.
        let mut file = self.shards[info.shard]
            .lock()
            .map_err(|_| io::Error::other("Safetensors shard lock poisoned"))?;
        file.seek(SeekFrom::Start(info.offset))?;
        file.read_exact(&mut data)?;
        Ok(data)
    }

    /// Limit applies to decoded f32 bytes. Peak memory also includes raw bytes.
    pub fn read_f32(&self, name: &str, max_decoded_bytes: usize) -> io::Result<Vec<f32>> {
        let info = self.tensor(name)?;
        if info.dtype == DType::I8 {
            return Err(invalid("I8 requires explicit quantization scales"));
        }
        let count = info.byte_len / info.dtype.size();
        if count > max_decoded_bytes / 4 {
            return Err(invalid("decoded tensor exceeds read budget"));
        }
        let raw = self.read_raw(name, info.byte_len)?;
        let values = match info.dtype {
            DType::F32 => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            DType::BF16 => raw
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
            DType::F16 => raw
                .chunks_exact(2)
                .map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            DType::I8 => unreachable!(),
        };
        Ok(values)
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    match exponent {
        0 => sign * (fraction as f32) * 2f32.powi(-24),
        31 if fraction == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + fraction as f32 / 1024.0) * 2f32.powi(exponent as i32 - 15),
    }
}
