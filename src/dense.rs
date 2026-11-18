//! Dense projection weights kept as decoded F32 or as the checkpoint's original BF16 bits.
//!
//! BF16 -> F32 is exact (a 16-bit shift), so `Bf16` storage computes the same products as
//! the F32 copy of a BF16 checkpoint while keeping half the resident bytes. Products are
//! accumulated in F32 in the same order as `dot_product`, so results are bitwise identical.
use crate::backend::{finite, invalid, Matrix};
use rayon::prelude::*;
use std::io;

/// Row-major dense weight matrix used for attention projections, router and LM head.
pub enum DenseMatrix {
    F32(Matrix),
    Bf16 {
        rows: usize,
        columns: usize,
        values: Vec<u16>,
    },
}

/// MOE_DENSE_STORAGE: `f32` decodes every dense matrix (default, historical behaviour);
/// `bf16` keeps BF16 checkpoint matrices as BF16 in RAM (other dtypes still decode to F32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseStorage {
    F32,
    Bf16,
}

impl DenseStorage {
    pub fn from_environment() -> io::Result<Self> {
        match std::env::var("MOE_DENSE_STORAGE").as_deref() {
            Ok("f32") | Err(std::env::VarError::NotPresent) => Ok(Self::F32),
            Ok("bf16") => Ok(Self::Bf16),
            _ => Err(invalid("MOE_DENSE_STORAGE must be f32 or bf16")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }
}

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// F32 rounding to BF16 (round-to-nearest-even); used by tests and fixtures.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

/// Same independent-accumulator shape as `dot_product`, decoding BF16 on the fly.
fn dot_bf16(left: &[u16], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let mut sums = [0.0f32; 8];
    let mut left_chunks = left.chunks_exact(8);
    let mut right_chunks = right.chunks_exact(8);
    for (l, r) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
        for i in 0..8 {
            sums[i] += bf16_to_f32(l[i]) * r[i];
        }
    }
    let mut sum = sums.into_iter().sum::<f32>();
    for (&l, &r) in left_chunks.remainder().iter().zip(right_chunks.remainder()) {
        sum += bf16_to_f32(l) * r;
    }
    sum
}

impl From<Matrix> for DenseMatrix {
    fn from(matrix: Matrix) -> Self {
        Self::F32(matrix)
    }
}

impl DenseMatrix {
    /// BF16 matrix from raw little-endian checkpoint bytes; rejects non-finite weights.
    pub fn from_bf16_bytes(rows: usize, columns: usize, raw: &[u8]) -> io::Result<Self> {
        if rows.checked_mul(columns).and_then(|n| n.checked_mul(2)) != Some(raw.len()) {
            return Err(invalid("BF16 matrix size mismatch"));
        }
        let values: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        // Exponent all ones = Inf/NaN.
        if values.iter().any(|&v| v & 0x7f80 == 0x7f80) {
            return Err(invalid("non-finite BF16 weight"));
        }
        Ok(Self::Bf16 {
            rows,
            columns,
            values,
        })
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::F32(m) => m.rows,
            Self::Bf16 { rows, .. } => *rows,
        }
    }

    pub fn columns(&self) -> usize {
        match self {
            Self::F32(m) => m.columns,
            Self::Bf16 { columns, .. } => *columns,
        }
    }

    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::F32(m) => m.values.len() * 4,
            Self::Bf16 { values, .. } => values.len() * 2,
        }
    }

    /// Mutable F32 view (tests that overwrite weights); `None` for BF16 storage.
    pub fn as_f32_mut(&mut self) -> Option<&mut Matrix> {
        match self {
            Self::F32(m) => Some(m),
            Self::Bf16 { .. } => None,
        }
    }

    pub fn multiply_loaded_into(&self, input: &[f32], output: &mut [f32]) -> io::Result<()> {
        match self {
            Self::F32(m) => m.multiply_loaded_into(input, output),
            Self::Bf16 {
                rows,
                columns,
                values,
            } => {
                if input.len() != *columns {
                    return Err(invalid("matrix input dimension mismatch"));
                }
                if output.len() != *rows {
                    return Err(invalid("matrix output dimension mismatch"));
                }
                finite(input)?;
                if values.len() >= 262_144 {
                    output
                        .par_iter_mut()
                        .zip(values.par_chunks_exact(*columns))
                        .for_each(|(target, row)| *target = dot_bf16(row, input));
                } else {
                    output
                        .iter_mut()
                        .zip(values.chunks_exact(*columns))
                        .for_each(|(target, row)| *target = dot_bf16(row, input));
                }
                finite(output)
            }
        }
    }

    /// Upload to VRAM as F32 (cuBLAS SGEMV). BF16 is widened in a temporary host buffer.
    pub fn to_gpu(&self, engine: &crate::gpu::GpuComputeEngine) -> Option<crate::gpu::GpuMatrix> {
        match self {
            Self::F32(m) => crate::gpu::GpuMatrix::from_matrix(m, engine),
            Self::Bf16 {
                rows,
                columns,
                values,
            } => {
                let widened = Matrix {
                    rows: *rows,
                    columns: *columns,
                    values: values.iter().map(|&v| bf16_to_f32(v)).collect(),
                };
                crate::gpu::GpuMatrix::from_matrix(&widened, engine)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_storage_matches_widened_f32_matrix() {
        let (rows, columns) = (37, 1029);
        let bits: Vec<u16> = (0..rows * columns)
            .map(|i| f32_to_bf16(((i * 7919 % 2003) as f32 - 1001.0) / 997.0))
            .collect();
        let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let bf16 = DenseMatrix::from_bf16_bytes(rows, columns, &raw).unwrap();
        let f32m = DenseMatrix::F32(Matrix {
            rows,
            columns,
            values: bits.iter().map(|&b| bf16_to_f32(b)).collect(),
        });
        assert_eq!(bf16.resident_bytes() * 2, f32m.resident_bytes());
        let input: Vec<f32> = (0..columns).map(|i| (i as f32 * 0.37).sin()).collect();
        let (mut a, mut b) = (vec![0.0; rows], vec![0.0; rows]);
        bf16.multiply_loaded_into(&input, &mut a).unwrap();
        f32m.multiply_loaded_into(&input, &mut b).unwrap();
        // Same products and accumulator layout: bitwise identical.
        assert_eq!(a, b);
    }

    #[test]
    fn bf16_rejects_non_finite_and_bad_shapes() {
        let inf = f32_to_bf16(f32::INFINITY).to_le_bytes();
        assert!(DenseMatrix::from_bf16_bytes(1, 1, &inf).is_err());
        assert!(DenseMatrix::from_bf16_bytes(2, 1, &[0, 0]).is_err());
        let ok = DenseMatrix::from_bf16_bytes(1, 2, &[0, 0x3f, 0, 0x40]).unwrap();
        let mut out = [0.0];
        assert!(ok.multiply_loaded_into(&[1.0], &mut out).is_err());
        ok.multiply_loaded_into(&[1.0, 1.0], &mut out).unwrap();
        assert_eq!(out[0], 0.5 + 2.0);
    }
}
