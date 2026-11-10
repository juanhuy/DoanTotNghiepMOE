//! On-demand OLMoE merged INT8 & INT4 experts from Safetensors shards.
//! Does not implement the OLMoE decoder or tokenizer.
use crate::{
    backend::{finite, invalid},
    safetensors::{DType, TensorIndex},
};
use rayon::prelude::*;
use std::io;

/// Định dạng lượng tử hóa của trọng số Expert
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantType {
    /// 1 byte trên mỗi tham số (INT8, ~6.015 MB / expert)
    Int8,
    /// 2 tham số 4-bit đóng gói trong 1 byte (INT4 nibble packing, ~3.008 MB / expert)
    Int4,
}

/// Selected once when loading a model; both variants preserve expert precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateUpMode {
    Separate,
    Paired,
}
impl GateUpMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Separate => "separate",
            Self::Paired => "paired",
        }
    }
    pub(crate) fn from_environment() -> io::Result<Self> {
        match std::env::var("MOE_GATE_UP_MODE") {
            Ok(value) => value.parse(),
            Err(std::env::VarError::NotPresent) => Ok(Self::Paired),
            Err(_) => Err(invalid("invalid MOE_GATE_UP_MODE")),
        }
    }
}
impl std::str::FromStr for GateUpMode {
    type Err = io::Error;
    fn from_str(value: &str) -> io::Result<Self> {
        match value {
            "separate" => Ok(Self::Separate),
            "paired" => Ok(Self::Paired),
            _ => Err(invalid("MOE_GATE_UP_MODE must be separate or paired")),
        }
    }
}

pub struct Int8Expert {
    hidden: usize,
    intermediate: usize,
    weights: Vec<u8>,
    scales: Vec<f32>,
    quant_type: QuantType,
}

pub(crate) struct ExpertScratch {
    pub(crate) gate_up_mode: GateUpMode,
    pub(crate) activated: Vec<f32>,
    pub(crate) output: Vec<f32>,
}

impl ExpertScratch {
    pub(crate) fn new(hidden: usize, intermediate: usize) -> Self {
        Self {
            gate_up_mode: GateUpMode::Paired,
            activated: vec![0.0; intermediate],
            output: vec![0.0; hidden],
        }
    }
}

impl Int8Expert {
    /// Nạp một Expert INT8 tái sử dụng bộ đệm `weights` và `scales` đã cấp phát trước.
    /// Nếu capacity của buffers đã đủ, hàm đảm bảo ZERO HEAP ALLOCATIONS.
    pub fn load_reusing(
        index: &TensorIndex,
        layer: usize,
        expert: usize,
        hidden: usize,
        intermediate: usize,
        mut weights: Vec<u8>,
        mut scales: Vec<f32>,
    ) -> io::Result<Self> {
        if hidden == 0 || intermediate == 0 {
            return Err(invalid("expert dimensions must be positive"));
        }
        let weight_count = hidden
            .checked_mul(intermediate)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| invalid("expert shape overflow"))?;
        let scale_count = intermediate
            .checked_mul(2)
            .and_then(|n| n.checked_add(hidden))
            .ok_or_else(|| invalid("scale shape overflow"))?;

        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        let weight_name = format!("{prefix}.merged_weight");
        let scale_name = format!("{prefix}.qs");

        let weight_info = index.tensor(&weight_name)?;
        let scale_info = index.tensor(&scale_name)?;

        // Tự động nhận diện định dạng lượng tử hóa:
        // - INT8: shape == [weight_count] (1 byte/tham số)
        // - INT4: shape == [weight_count / 2] (2 nibbles 4-bit/byte)
        let (quant_type, expected_weight_bytes) = if weight_info.shape == [weight_count] {
            (QuantType::Int8, weight_count)
        } else if weight_info.shape == [weight_count / 2] {
            if hidden % 2 != 0 || intermediate % 2 != 0 {
                return Err(invalid("hidden and intermediate must be even for INT4"));
            }
            (QuantType::Int4, weight_count / 2)
        } else {
            return Err(invalid(format!(
                "invalid expert weight shape: expected [{weight_count}] (INT8) or [{}] (INT4), got {:?}",
                weight_count / 2,
                weight_info.shape
            )));
        };

        if (weight_info.dtype != DType::I8 && weight_info.dtype != DType::U8)
            || scale_info.dtype != DType::F32
            || scale_info.shape != [scale_count]
        {
            return Err(invalid(
                "expected flat I8/U8 merged_weight and flat F32 qs with exact expert dimensions",
            ));
        }

        // Tái sử dụng vùng nhớ: resize không cấp phát lại nếu capacity đã đủ
        if weights.capacity() < expected_weight_bytes {
            weights.reserve_exact(expected_weight_bytes - weights.len());
        }
        weights.resize(expected_weight_bytes, 0);

        if scales.capacity() < scale_count {
            scales.reserve_exact(scale_count - scales.len());
        }
        scales.resize(scale_count, 0.0);

        // Tận dụng đầu mảng weights làm scratch buffer đọc scales (nếu weights đủ lớn)
        let scale_raw_bytes = scale_count * 4;
        if weights.len() >= scale_raw_bytes {
            index.read_f32_into(&scale_name, &mut weights[..scale_raw_bytes], &mut scales)?;
        } else {
            let mut scale_scratch = vec![0u8; scale_raw_bytes];
            index.read_f32_into(&scale_name, &mut scale_scratch, &mut scales)?;
        }

        if scales.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(invalid("expert scales must be finite and positive"));
        }

        // Đọc weights (INT8 6.0MB hoặc INT4 3.0MB) qua POSIX DMA pread
        index.read_raw_into(&weight_name, &mut weights)?;

        Ok(Self {
            hidden,
            intermediate,
            weights,
            scales,
            quant_type,
        })
    }

    pub fn load(
        index: &TensorIndex,
        layer: usize,
        expert: usize,
        hidden: usize,
        intermediate: usize,
        max_bytes: usize,
    ) -> io::Result<Self> {
        let weight_count = hidden
            .checked_mul(intermediate)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| invalid("expert shape overflow"))?;
        let scale_count = intermediate
            .checked_mul(2)
            .and_then(|n| n.checked_add(hidden))
            .ok_or_else(|| invalid("scale shape overflow"))?;

        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        let weight_name = format!("{prefix}.merged_weight");
        let weight_info = index.tensor(&weight_name)?;

        let actual_weight_bytes = if weight_info.shape == [weight_count / 2] {
            weight_count / 2
        } else {
            weight_count
        };

        let total = scale_count
            .checked_mul(4)
            .and_then(|n| n.checked_add(actual_weight_bytes))
            .ok_or_else(|| invalid("expert byte count overflow"))?;
        if total > max_bytes {
            return Err(invalid("expert exceeds read budget"));
        }
        let weights = Vec::with_capacity(actual_weight_bytes);
        let scales = Vec::with_capacity(scale_count);
        Self::load_reusing(index, layer, expert, hidden, intermediate, weights, scales)
    }

    pub fn quant_type(&self) -> QuantType {
        self.quant_type
    }

    pub fn resident_bytes(&self) -> usize {
        self.weights.len() + self.scales.len() * 4
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.weights.capacity() + self.scales.capacity() * std::mem::size_of::<f32>()
    }

    pub fn from_raw(
        hidden: usize,
        intermediate: usize,
        weights: Vec<u8>,
        scales: Vec<f32>,
    ) -> Self {
        let weight_count = hidden * intermediate * 3;
        let quant_type = if weights.len() == weight_count / 2 {
            QuantType::Int4
        } else {
            QuantType::Int8
        };
        Self {
            hidden,
            intermediate,
            weights,
            scales,
            quant_type,
        }
    }

    /// Thu hồi lại buffers (weights, scales) từ expert để tái sử dụng, triệt tiêu cấp phát heap
    pub fn into_raw(self) -> (Vec<u8>, Vec<f32>) {
        (self.weights, self.scales)
    }

    pub fn weights(&self) -> &[u8] {
        &self.weights
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Uses row scales during matvec; never expands the expert to an f32 matrix.
    pub fn forward(&self, input: &[f32]) -> io::Result<Vec<f32>> {
        let mut scratch = ExpertScratch::new(self.hidden, self.intermediate);
        Ok(self.forward_reuse(input, &mut scratch)?.to_vec())
    }

    pub(crate) fn forward_reuse<'a>(
        &self,
        input: &[f32],
        scratch: &'a mut ExpertScratch,
    ) -> io::Result<&'a [f32]> {
        if input.len() != self.hidden {
            return Err(invalid("expert input dimension mismatch"));
        }
        if scratch.activated.len() != self.intermediate || scratch.output.len() != self.hidden {
            return Err(invalid("expert scratch dimension mismatch"));
        }
        finite(input)?;
        let hidden_stride = match self.quant_type {
            QuantType::Int8 => self.hidden,
            QuantType::Int4 => self.hidden / 2,
        };
        let n = hidden_stride * self.intermediate;
        fused_gate_up_swiglu(
            &self.weights[..n],
            &self.weights[n..2 * n],
            &self.scales[..self.intermediate],
            &self.scales[self.intermediate..2 * self.intermediate],
            input,
            &mut scratch.activated,
            self.hidden,
            self.quant_type,
            scratch.gate_up_mode,
        );
        matvec_into(
            &self.weights[2 * n..],
            &self.scales[2 * self.intermediate..],
            &scratch.activated,
            &mut scratch.output,
            self.quant_type,
        );
        finite(&scratch.output)?;
        Ok(&scratch.output)
    }

    /// Tính toán tuần tự trên 1 CPU core duy nhất, tận dụng 100% đường ống SIMD 32-lane 4-bank.
    /// Dành cho mô hình Coarse-Grained Parallelism (nhiều experts chạy song song trên các workers độc lập của Rayon).
    pub(crate) fn forward_single_threaded<'a>(
        &self,
        input: &[f32],
        scratch: &'a mut ExpertScratch,
    ) -> io::Result<&'a [f32]> {
        if input.len() != self.hidden {
            return Err(invalid("expert input dimension mismatch"));
        }
        if scratch.activated.len() != self.intermediate || scratch.output.len() != self.hidden {
            return Err(invalid("expert scratch dimension mismatch"));
        }
        finite(input)?;
        let hidden_stride = match self.quant_type {
            QuantType::Int8 => self.hidden,
            QuantType::Int4 => self.hidden / 2,
        };
        let n = hidden_stride * self.intermediate;
        fused_gate_up_swiglu_seq(
            &self.weights[..n],
            &self.weights[n..2 * n],
            &self.scales[..self.intermediate],
            &self.scales[self.intermediate..2 * self.intermediate],
            input,
            &mut scratch.activated,
            self.hidden,
            self.quant_type,
            scratch.gate_up_mode,
        );
        matvec_into_seq(
            &self.weights[2 * n..],
            &self.scales[2 * self.intermediate..],
            &scratch.activated,
            &mut scratch.output,
            self.quant_type,
        );
        finite(&scratch.output)?;
        Ok(&scratch.output)
    }
}

/// Tích vô hướng INT8 x FP32 chuẩn tham chiếu (scalar fallback) với 4 bộ cộng độc lập
#[inline(always)]
fn dot_i8_f32_scalar(row: &[u8], input: &[f32]) -> f32 {
    let mut s0 = [0.0f32; 8];
    let mut s1 = [0.0f32; 8];
    let mut s2 = [0.0f32; 8];
    let mut s3 = [0.0f32; 8];
    let mut qs32 = row.chunks_exact(32);
    let mut xs32 = input.chunks_exact(32);
    for (q, x) in qs32.by_ref().zip(xs32.by_ref()) {
        for lane in 0..8 {
            s0[lane] += (q[lane] as i8 as f32) * x[lane];
            s1[lane] += (q[lane + 8] as i8 as f32) * x[lane + 8];
            s2[lane] += (q[lane + 16] as i8 as f32) * x[lane + 16];
            s3[lane] += (q[lane + 24] as i8 as f32) * x[lane + 24];
        }
    }
    let rem_q = qs32.remainder();
    let rem_x = xs32.remainder();
    let mut s_rem = [0.0f32; 8];
    let mut qs8 = rem_q.chunks_exact(8);
    let mut xs8 = rem_x.chunks_exact(8);
    for (q, x) in qs8.by_ref().zip(xs8.by_ref()) {
        for lane in 0..8 {
            s_rem[lane] += (q[lane] as i8 as f32) * x[lane];
        }
    }
    let mut sum = 0.0f32;
    for lane in 0..8 {
        sum += s0[lane] + s1[lane] + s2[lane] + s3[lane] + s_rem[lane];
    }
    for (&q, &x) in qs8.remainder().iter().zip(xs8.remainder()) {
        sum += (q as i8 as f32) * x;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i8_f32_avx2(row: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let mut qs32 = row.chunks_exact(32);
    let mut xs32 = input.chunks_exact(32);

    for (q, x) in qs32.by_ref().zip(xs32.by_ref()) {
        let q_ptr = q.as_ptr();
        let x_ptr = x.as_ptr();

        // Lane 0: elements 0..8
        let val0 = std::ptr::read_unaligned(q_ptr as *const i64);
        let q0_i8 = _mm_cvtsi64_si128(val0);
        let q0_i32 = _mm256_cvtepi8_epi32(q0_i8);
        let q0_f32 = _mm256_cvtepi32_ps(q0_i32);
        let x0 = _mm256_loadu_ps(x_ptr);
        sum0 = _mm256_fmadd_ps(q0_f32, x0, sum0);

        // Lane 1: elements 8..16
        let val1 = std::ptr::read_unaligned(q_ptr.add(8) as *const i64);
        let q1_i8 = _mm_cvtsi64_si128(val1);
        let q1_i32 = _mm256_cvtepi8_epi32(q1_i8);
        let q1_f32 = _mm256_cvtepi32_ps(q1_i32);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));
        sum1 = _mm256_fmadd_ps(q1_f32, x1, sum1);

        // Lane 2: elements 16..24
        let val2 = std::ptr::read_unaligned(q_ptr.add(16) as *const i64);
        let q2_i8 = _mm_cvtsi64_si128(val2);
        let q2_i32 = _mm256_cvtepi8_epi32(q2_i8);
        let q2_f32 = _mm256_cvtepi32_ps(q2_i32);
        let x2 = _mm256_loadu_ps(x_ptr.add(16));
        sum2 = _mm256_fmadd_ps(q2_f32, x2, sum2);

        // Lane 3: elements 24..32
        let val3 = std::ptr::read_unaligned(q_ptr.add(24) as *const i64);
        let q3_i8 = _mm_cvtsi64_si128(val3);
        let q3_i32 = _mm256_cvtepi8_epi32(q3_i8);
        let q3_f32 = _mm256_cvtepi32_ps(q3_i32);
        let x3 = _mm256_loadu_ps(x_ptr.add(24));
        sum3 = _mm256_fmadd_ps(q3_f32, x3, sum3);
    }

    let mut total_vec = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    let rem_q = qs32.remainder();
    let rem_x = xs32.remainder();
    let mut qs8 = rem_q.chunks_exact(8);
    let mut xs8 = rem_x.chunks_exact(8);

    for (q, x) in qs8.by_ref().zip(xs8.by_ref()) {
        let val = std::ptr::read_unaligned(q.as_ptr() as *const i64);
        let q_i8 = _mm_cvtsi64_si128(val);
        let q_i32 = _mm256_cvtepi8_epi32(q_i8);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let x_vec = _mm256_loadu_ps(x.as_ptr());
        total_vec = _mm256_fmadd_ps(q_f32, x_vec, total_vec);
    }

    // Horizontal reduction of 8-wide float vector
    let low = _mm256_castps256_ps128(total_vec);
    let high = _mm256_extractf128_ps(total_vec, 1);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, shuf2);
    let mut sum = _mm_cvtss_f32(sum32);

    for (&q, &x) in qs8.remainder().iter().zip(xs8.remainder()) {
        sum += (q as i8 as f32) * x;
    }
    sum
}

/// Tích vô hướng INT8 x FP32 tăng tốc phần cứng AVX2/FMA với graceful scalar fallback
#[inline(always)]
fn dot_i8_f32(row: &[u8], input: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Hardware support for AVX2 and FMA was verified at runtime.
            return unsafe { dot_i8_f32_avx2(row, input) };
        }
    }
    dot_i8_f32_scalar(row, input)
}

/// Tích vô hướng INT4 x FP32 chuẩn tham chiếu (scalar fallback)
/// Giải mã hai số nguyên có dấu 4-bit [-8, 7] trên mỗi byte (nibble packing)
#[inline(always)]
fn dot_i4_f32_scalar(row: &[u8], input: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    let num_chunks = input.len() / 8;
    for c in 0..num_chunks {
        let q4 = &row[c * 4..(c + 1) * 4];
        let x8 = &input[c * 8..(c + 1) * 8];

        let q0 = ((q4[0] as i8) << 4) >> 4;
        let q1 = (q4[0] as i8) >> 4;
        let q2 = ((q4[1] as i8) << 4) >> 4;
        let q3 = (q4[1] as i8) >> 4;
        let q4_val = ((q4[2] as i8) << 4) >> 4;
        let q5 = (q4[2] as i8) >> 4;
        let q6 = ((q4[3] as i8) << 4) >> 4;
        let q7 = (q4[3] as i8) >> 4;

        s0 += (q0 as f32) * x8[0] + (q1 as f32) * x8[1];
        s1 += (q2 as f32) * x8[2] + (q3 as f32) * x8[3];
        s2 += (q4_val as f32) * x8[4] + (q5 as f32) * x8[5];
        s3 += (q6 as f32) * x8[6] + (q7 as f32) * x8[7];
    }

    let mut sum = s0 + s1 + s2 + s3;
    let rem_q = &row[num_chunks * 4..];
    let rem_x = &input[num_chunks * 8..];
    for (i, &b) in rem_q.iter().enumerate() {
        let q0 = ((b as i8) << 4) >> 4;
        let q1 = (b as i8) >> 4;
        if 2 * i < rem_x.len() {
            sum += (q0 as f32) * rem_x[2 * i];
        }
        if 2 * i + 1 < rem_x.len() {
            sum += (q1 as f32) * rem_x[2 * i + 1];
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i4_f32_avx2(row: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let num_chunks_32 = input.len() / 32;
    let mask_0f = _mm_set1_epi8(0x0F);
    let c8 = _mm_set1_epi8(8);

    for c in 0..num_chunks_32 {
        let q_ptr = row.as_ptr().add(c * 16);
        let x_ptr = input.as_ptr().add(c * 32);

        let v = _mm_loadu_si128(q_ptr as *const __m128i);
        let v_lo = _mm_and_si128(v, mask_0f);
        let v_hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask_0f);

        let unp_lo = _mm_unpacklo_epi8(v_lo, v_hi);
        let unp_hi = _mm_unpackhi_epi8(v_lo, v_hi);

        let s_lo = _mm_sub_epi8(_mm_xor_si128(unp_lo, c8), c8);
        let s_hi = _mm_sub_epi8(_mm_xor_si128(unp_hi, c8), c8);

        // Lane 0: elements 0..8
        let q0_i32 = _mm256_cvtepi8_epi32(s_lo);
        let q0_f32 = _mm256_cvtepi32_ps(q0_i32);
        let x0 = _mm256_loadu_ps(x_ptr);
        sum0 = _mm256_fmadd_ps(q0_f32, x0, sum0);

        // Lane 1: elements 8..16
        let s_lo_hi = _mm_srli_si128(s_lo, 8);
        let q1_i32 = _mm256_cvtepi8_epi32(s_lo_hi);
        let q1_f32 = _mm256_cvtepi32_ps(q1_i32);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));
        sum1 = _mm256_fmadd_ps(q1_f32, x1, sum1);

        // Lane 2: elements 16..24
        let q2_i32 = _mm256_cvtepi8_epi32(s_hi);
        let q2_f32 = _mm256_cvtepi32_ps(q2_i32);
        let x2 = _mm256_loadu_ps(x_ptr.add(16));
        sum2 = _mm256_fmadd_ps(q2_f32, x2, sum2);

        // Lane 3: elements 24..32
        let s_hi_hi = _mm_srli_si128(s_hi, 8);
        let q3_i32 = _mm256_cvtepi8_epi32(s_hi_hi);
        let q3_f32 = _mm256_cvtepi32_ps(q3_i32);
        let x3 = _mm256_loadu_ps(x_ptr.add(24));
        sum3 = _mm256_fmadd_ps(q3_f32, x3, sum3);
    }

    let mut total_vec = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    let rem_x = &input[num_chunks_32 * 32..];
    let rem_q = &row[num_chunks_32 * 16..];

    let num_chunks_8 = rem_x.len() / 8;
    for c in 0..num_chunks_8 {
        let q = &rem_q[c * 4..(c + 1) * 4];
        let x = &rem_x[c * 8..(c + 1) * 8];

        let b0 = q[0];
        let b1 = q[1];
        let b2 = q[2];
        let b3 = q[3];

        let q0 = ((b0 as i8) << 4) >> 4;
        let q1 = (b0 as i8) >> 4;
        let q2 = ((b1 as i8) << 4) >> 4;
        let q3 = (b1 as i8) >> 4;
        let q4_val = ((b2 as i8) << 4) >> 4;
        let q5 = (b2 as i8) >> 4;
        let q6 = ((b3 as i8) << 4) >> 4;
        let q7 = (b3 as i8) >> 4;

        let q_packed: i64 = (q0 as u8 as i64)
            | ((q1 as u8 as i64) << 8)
            | ((q2 as u8 as i64) << 16)
            | ((q3 as u8 as i64) << 24)
            | ((q4_val as u8 as i64) << 32)
            | ((q5 as u8 as i64) << 40)
            | ((q6 as u8 as i64) << 48)
            | ((q7 as u8 as i64) << 56);

        let q_i8 = _mm_cvtsi64_si128(q_packed);
        let q_i32 = _mm256_cvtepi8_epi32(q_i8);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let x_vec = _mm256_loadu_ps(x.as_ptr());
        total_vec = _mm256_fmadd_ps(q_f32, x_vec, total_vec);
    }

    // Horizontal reduction of 8-wide float vector
    let low = _mm256_castps256_ps128(total_vec);
    let high = _mm256_extractf128_ps(total_vec, 1);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, shuf2);
    let mut sum = _mm_cvtss_f32(sum32);

    let tail_x = &rem_x[num_chunks_8 * 8..];
    let tail_q = &rem_q[num_chunks_8 * 4..];
    for (i, &b) in tail_q.iter().enumerate() {
        let q0 = ((b as i8) << 4) >> 4;
        let q1 = (b as i8) >> 4;
        if 2 * i < tail_x.len() {
            sum += (q0 as f32) * tail_x[2 * i];
        }
        if 2 * i + 1 < tail_x.len() {
            sum += (q1 as f32) * tail_x[2 * i + 1];
        }
    }
    sum
}

/// Tích vô hướng INT4 x FP32 tăng tốc phần cứng AVX2/FMA với graceful scalar fallback
#[inline(always)]
fn dot_i4_f32(row: &[u8], input: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Hardware support for AVX2 and FMA was verified at runtime.
            return unsafe { dot_i4_f32_avx2(row, input) };
        }
    }
    dot_i4_f32_scalar(row, input)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i8_f32_pair_avx2(g_row: &[u8], u_row: &[u8], input: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;

    let mut sum_g0 = _mm256_setzero_ps();
    let mut sum_g1 = _mm256_setzero_ps();
    let mut sum_g2 = _mm256_setzero_ps();
    let mut sum_g3 = _mm256_setzero_ps();

    let mut sum_u0 = _mm256_setzero_ps();
    let mut sum_u1 = _mm256_setzero_ps();
    let mut sum_u2 = _mm256_setzero_ps();
    let mut sum_u3 = _mm256_setzero_ps();

    let mut g_chunks = g_row.chunks_exact(32);
    let mut u_chunks = u_row.chunks_exact(32);
    let mut x_chunks = input.chunks_exact(32);

    for ((g, u), x) in g_chunks
        .by_ref()
        .zip(u_chunks.by_ref())
        .zip(x_chunks.by_ref())
    {
        let g_ptr = g.as_ptr();
        let u_ptr = u.as_ptr();
        let x_ptr = x.as_ptr();

        // Nạp vector input một lần duy nhất vào 4 thanh ghi YMM (256-bit)
        let x0 = _mm256_loadu_ps(x_ptr);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));
        let x2 = _mm256_loadu_ps(x_ptr.add(16));
        let x3 = _mm256_loadu_ps(x_ptr.add(24));

        // Nhánh Gate: unpack i8 -> f32 và FMA với x
        let g0_i8 = _mm_cvtsi64_si128(std::ptr::read_unaligned(g_ptr as *const i64));
        let g0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g0_i8));
        sum_g0 = _mm256_fmadd_ps(g0_f, x0, sum_g0);

        let g1_i8 = _mm_cvtsi64_si128(std::ptr::read_unaligned(g_ptr.add(8) as *const i64));
        let g1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g1_i8));
        sum_g1 = _mm256_fmadd_ps(g1_f, x1, sum_g1);

        let g2_i8 = _mm_cvtsi64_si128(std::ptr::read_unaligned(g_ptr.add(16) as *const i64));
        let g2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g2_i8));
        sum_g2 = _mm256_fmadd_ps(g2_f, x2, sum_g2);

        let g3_i8 = _mm_cvtsi64_si128(std::ptr::read_unaligned(g_ptr.add(24) as *const i64));
        let g3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g3_i8));
        sum_g3 = _mm256_fmadd_ps(g3_f, x3, sum_g3);

        // Nhánh Up: tái sử dụng x0..x3 trực tiếp từ thanh ghi, không đọc lại L1 Cache
        let u0_i8 = _mm_cvtsi64_si128(std::ptr::read_unaligned(u_ptr as *const i64));
        let u0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u0_i8));
        sum_u0 = _mm256_fmadd_ps(u0_f, x0, sum_u0);

        let u1_i8 = _mm_cvtsi64_si128(std::ptr::read_unaligned(u_ptr.add(8) as *const i64));
        let u1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u1_i8));
        sum_u1 = _mm256_fmadd_ps(u1_f, x1, sum_u1);

        let u2_i8 = _mm_cvtsi64_si128(std::ptr::read_unaligned(u_ptr.add(16) as *const i64));
        let u2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u2_i8));
        sum_u2 = _mm256_fmadd_ps(u2_f, x2, sum_u2);

        let u3_i8 = _mm_cvtsi64_si128(std::ptr::read_unaligned(u_ptr.add(24) as *const i64));
        let u3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u3_i8));
        sum_u3 = _mm256_fmadd_ps(u3_f, x3, sum_u3);
    }

    let mut tot_g = _mm256_add_ps(_mm256_add_ps(sum_g0, sum_g1), _mm256_add_ps(sum_g2, sum_g3));
    let mut tot_u = _mm256_add_ps(_mm256_add_ps(sum_u0, sum_u1), _mm256_add_ps(sum_u2, sum_u3));

    let rem_g = g_chunks.remainder();
    let rem_u = u_chunks.remainder();
    let rem_x = x_chunks.remainder();

    let mut g8 = rem_g.chunks_exact(8);
    let mut u8 = rem_u.chunks_exact(8);
    let mut x8 = rem_x.chunks_exact(8);

    for ((g, u), x) in g8.by_ref().zip(u8.by_ref()).zip(x8.by_ref()) {
        let x_vec = _mm256_loadu_ps(x.as_ptr());

        let g_val = std::ptr::read_unaligned(g.as_ptr() as *const i64);
        let g_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(g_val)));
        tot_g = _mm256_fmadd_ps(g_f, x_vec, tot_g);

        let u_val = std::ptr::read_unaligned(u.as_ptr() as *const i64);
        let u_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(u_val)));
        tot_u = _mm256_fmadd_ps(u_f, x_vec, tot_u);
    }

    #[inline(always)]
    unsafe fn hreduce(v: __m256) -> f32 {
        let low = _mm256_castps256_ps128(v);
        let high = _mm256_extractf128_ps(v, 1);
        let sum128 = _mm_add_ps(low, high);
        let shuf = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, shuf2);
        _mm_cvtss_f32(sum32)
    }

    let mut final_g = hreduce(tot_g);
    let mut final_u = hreduce(tot_u);

    for ((&g, &u), &x) in g8
        .remainder()
        .iter()
        .zip(u8.remainder())
        .zip(x8.remainder())
    {
        final_g += (g as i8 as f32) * x;
        final_u += (u as i8 as f32) * x;
    }

    (final_g, final_u)
}

#[inline(always)]
fn dot_i8_f32_pair(gate_row: &[u8], up_row: &[u8], input: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Hardware support verified at runtime
            return unsafe { dot_i8_f32_pair_avx2(gate_row, up_row, input) };
        }
    }
    (dot_i8_f32(gate_row, input), dot_i8_f32(up_row, input))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i4_f32_pair_avx2(g_row: &[u8], u_row: &[u8], input: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;

    let mut sum_g0 = _mm256_setzero_ps();
    let mut sum_g1 = _mm256_setzero_ps();
    let mut sum_g2 = _mm256_setzero_ps();
    let mut sum_g3 = _mm256_setzero_ps();

    let mut sum_u0 = _mm256_setzero_ps();
    let mut sum_u1 = _mm256_setzero_ps();
    let mut sum_u2 = _mm256_setzero_ps();
    let mut sum_u3 = _mm256_setzero_ps();

    let num_chunks_32 = input.len() / 32;
    let mask_0f = _mm_set1_epi8(0x0F);
    let c8 = _mm_set1_epi8(8);

    for c in 0..num_chunks_32 {
        let x_ptr = input.as_ptr().add(c * 32);
        let g_ptr = g_row.as_ptr().add(c * 16);
        let u_ptr = u_row.as_ptr().add(c * 16);

        // Nạp 32 floats của vector input một lần duy nhất
        let x0 = _mm256_loadu_ps(x_ptr);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));
        let x2 = _mm256_loadu_ps(x_ptr.add(16));
        let x3 = _mm256_loadu_ps(x_ptr.add(24));

        // Nhánh Gate: giải mã nibbles và FMA với x0..x3
        let vg = _mm_loadu_si128(g_ptr as *const __m128i);
        let vg_lo = _mm_and_si128(vg, mask_0f);
        let vg_hi = _mm_and_si128(_mm_srli_epi16(vg, 4), mask_0f);
        let unp_glo = _mm_unpacklo_epi8(vg_lo, vg_hi);
        let unp_ghi = _mm_unpackhi_epi8(vg_lo, vg_hi);
        let sg_lo = _mm_sub_epi8(_mm_xor_si128(unp_glo, c8), c8);
        let sg_hi = _mm_sub_epi8(_mm_xor_si128(unp_ghi, c8), c8);

        sum_g0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sg_lo)), x0, sum_g0);
        sum_g1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(sg_lo, 8))),
            x1,
            sum_g1,
        );
        sum_g2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sg_hi)), x2, sum_g2);
        sum_g3 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(sg_hi, 8))),
            x3,
            sum_g3,
        );

        // Nhánh Up: giải mã nibbles và FMA trực tiếp với x0..x3 có sẵn trên thanh ghi
        let vu = _mm_loadu_si128(u_ptr as *const __m128i);
        let vu_lo = _mm_and_si128(vu, mask_0f);
        let vu_hi = _mm_and_si128(_mm_srli_epi16(vu, 4), mask_0f);
        let unp_ulo = _mm_unpacklo_epi8(vu_lo, vu_hi);
        let unp_uhi = _mm_unpackhi_epi8(vu_lo, vu_hi);
        let su_lo = _mm_sub_epi8(_mm_xor_si128(unp_ulo, c8), c8);
        let su_hi = _mm_sub_epi8(_mm_xor_si128(unp_uhi, c8), c8);

        sum_u0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(su_lo)), x0, sum_u0);
        sum_u1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(su_lo, 8))),
            x1,
            sum_u1,
        );
        sum_u2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(su_hi)), x2, sum_u2);
        sum_u3 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(su_hi, 8))),
            x3,
            sum_u3,
        );
    }

    let mut tot_g = _mm256_add_ps(_mm256_add_ps(sum_g0, sum_g1), _mm256_add_ps(sum_g2, sum_g3));
    let mut tot_u = _mm256_add_ps(_mm256_add_ps(sum_u0, sum_u1), _mm256_add_ps(sum_u2, sum_u3));

    let rem_x = &input[num_chunks_32 * 32..];
    let rem_g = &g_row[num_chunks_32 * 16..];
    let rem_u = &u_row[num_chunks_32 * 16..];

    let num_chunks_8 = rem_x.len() / 8;
    for c in 0..num_chunks_8 {
        let x = &rem_x[c * 8..(c + 1) * 8];
        let g = &rem_g[c * 4..(c + 1) * 4];
        let u = &rem_u[c * 4..(c + 1) * 4];

        let x_vec = _mm256_loadu_ps(x.as_ptr());

        // Decode 4 bytes -> 8 ints for Gate
        let g0 = ((g[0] as i8) << 4) >> 4;
        let g1 = (g[0] as i8) >> 4;
        let g2 = ((g[1] as i8) << 4) >> 4;
        let g3 = (g[1] as i8) >> 4;
        let g4 = ((g[2] as i8) << 4) >> 4;
        let g5 = (g[2] as i8) >> 4;
        let g6 = ((g[3] as i8) << 4) >> 4;
        let g7 = (g[3] as i8) >> 4;
        let g_packed: i64 = (g0 as u8 as i64)
            | ((g1 as u8 as i64) << 8)
            | ((g2 as u8 as i64) << 16)
            | ((g3 as u8 as i64) << 24)
            | ((g4 as u8 as i64) << 32)
            | ((g5 as u8 as i64) << 40)
            | ((g6 as u8 as i64) << 48)
            | ((g7 as u8 as i64) << 56);
        let g_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(g_packed)));
        tot_g = _mm256_fmadd_ps(g_f, x_vec, tot_g);

        // Decode 4 bytes -> 8 ints for Up
        let u0 = ((u[0] as i8) << 4) >> 4;
        let u1 = (u[0] as i8) >> 4;
        let u2 = ((u[1] as i8) << 4) >> 4;
        let u3 = (u[1] as i8) >> 4;
        let u4 = ((u[2] as i8) << 4) >> 4;
        let u5 = (u[2] as i8) >> 4;
        let u6 = ((u[3] as i8) << 4) >> 4;
        let u7 = (u[3] as i8) >> 4;
        let u_packed: i64 = (u0 as u8 as i64)
            | ((u1 as u8 as i64) << 8)
            | ((u2 as u8 as i64) << 16)
            | ((u3 as u8 as i64) << 24)
            | ((u4 as u8 as i64) << 32)
            | ((u5 as u8 as i64) << 40)
            | ((u6 as u8 as i64) << 48)
            | ((u7 as u8 as i64) << 56);
        let u_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(u_packed)));
        tot_u = _mm256_fmadd_ps(u_f, x_vec, tot_u);
    }

    #[inline(always)]
    unsafe fn hreduce(v: __m256) -> f32 {
        let low = _mm256_castps256_ps128(v);
        let high = _mm256_extractf128_ps(v, 1);
        let sum128 = _mm_add_ps(low, high);
        let shuf = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, shuf2);
        _mm_cvtss_f32(sum32)
    }

    let mut final_g = hreduce(tot_g);
    let mut final_u = hreduce(tot_u);

    let tail_x = &rem_x[num_chunks_8 * 8..];
    let tail_g = &rem_g[num_chunks_8 * 4..];
    let tail_u = &rem_u[num_chunks_8 * 4..];

    for (i, (&bg, &bu)) in tail_g.iter().zip(tail_u).enumerate() {
        let g0 = ((bg as i8) << 4) >> 4;
        let g1 = (bg as i8) >> 4;
        let u0 = ((bu as i8) << 4) >> 4;
        let u1 = (bu as i8) >> 4;
        if 2 * i < tail_x.len() {
            let xv = tail_x[2 * i];
            final_g += (g0 as f32) * xv;
            final_u += (u0 as f32) * xv;
        }
        if 2 * i + 1 < tail_x.len() {
            let xv = tail_x[2 * i + 1];
            final_g += (g1 as f32) * xv;
            final_u += (u1 as f32) * xv;
        }
    }

    (final_g, final_u)
}

#[inline(always)]
fn dot_i4_f32_pair(gate_row: &[u8], up_row: &[u8], input: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Hardware support verified at runtime
            return unsafe { dot_i4_f32_pair_avx2(gate_row, up_row, input) };
        }
    }
    (dot_i4_f32(gate_row, input), dot_i4_f32(up_row, input))
}

/// Kernel Fused Gate-Up Projection & SwiGLU:
/// Gộp tính toán Gate và Up trong cùng 1 lần duyệt đa luồng Rayon, tái sử dụng vector `input` trong CPU L1 Cache
/// và tính ngay kích hoạt SiLU mà không cần ghi/đọc qua bộ đệm trung gian.
#[allow(clippy::too_many_arguments)]
fn fused_gate_up_swiglu(
    gate_weights: &[u8],
    up_weights: &[u8],
    gate_scales: &[f32],
    up_scales: &[f32],
    input: &[f32],
    activated: &mut [f32],
    hidden: usize,
    quant_type: QuantType,
    mode: GateUpMode,
) {
    debug_assert_eq!(activated.len(), gate_scales.len());
    debug_assert_eq!(activated.len(), up_scales.len());

    let stride = match quant_type {
        QuantType::Int8 => hidden,
        QuantType::Int4 => hidden / 2,
    };

    let compute_lane = |i: usize, target: &mut f32| {
        let gate_row = &gate_weights[i * stride..(i + 1) * stride];
        let up_row = &up_weights[i * stride..(i + 1) * stride];
        let (dot_g, dot_u) = match (quant_type, mode) {
            (QuantType::Int8, GateUpMode::Paired) => dot_i8_f32_pair(gate_row, up_row, input),
            (QuantType::Int8, GateUpMode::Separate) => {
                (dot_i8_f32(gate_row, input), dot_i8_f32(up_row, input))
            }
            (QuantType::Int4, GateUpMode::Paired) => dot_i4_f32_pair(gate_row, up_row, input),
            (QuantType::Int4, GateUpMode::Separate) => {
                (dot_i4_f32(gate_row, input), dot_i4_f32(up_row, input))
            }
        };
        let g = dot_g * gate_scales[i];
        let u = dot_u * up_scales[i];
        let silu = g / (1.0 + (-g).exp());
        *target = silu * u;
    };

    if gate_weights.len() >= 262_144 {
        activated
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, target)| compute_lane(i, target));
    } else {
        activated
            .iter_mut()
            .enumerate()
            .for_each(|(i, target)| compute_lane(i, target));
    }
}

// Separate accumulators shorten the dependency chain and let LLVM vectorize
// without target-specific instructions or expanding INT8/INT4 weights to a matrix.
fn matvec_into(
    weights: &[u8],
    scales: &[f32],
    input: &[f32],
    output: &mut [f32],
    quant_type: QuantType,
) {
    let row_len = match quant_type {
        QuantType::Int8 => input.len(),
        QuantType::Int4 => input.len() / 2,
    };
    let compute = |target: &mut f32, row: &[u8], scale: &f32| {
        let dot = match quant_type {
            QuantType::Int8 => dot_i8_f32(row, input),
            QuantType::Int4 => dot_i4_f32(row, input),
        };
        *target = dot * scale;
    };
    if weights.len() >= 262_144 {
        output
            .par_iter_mut()
            .zip(weights.par_chunks_exact(row_len))
            .zip(scales.par_iter())
            .for_each(|((target, row), scale)| compute(target, row, scale));
    } else {
        output
            .iter_mut()
            .zip(weights.chunks_exact(row_len))
            .zip(scales)
            .for_each(|((target, row), scale)| compute(target, row, scale));
    }
}

/// Phiên bản tuần tự 100% của Fused Gate-Up SwiGLU: không gọi par_iter_mut()
#[inline]
#[allow(clippy::too_many_arguments)]
fn fused_gate_up_swiglu_seq(
    gate_weights: &[u8],
    up_weights: &[u8],
    gate_scales: &[f32],
    up_scales: &[f32],
    input: &[f32],
    activated: &mut [f32],
    hidden: usize,
    quant_type: QuantType,
    mode: GateUpMode,
) {
    debug_assert_eq!(activated.len(), gate_scales.len());
    debug_assert_eq!(activated.len(), up_scales.len());

    let stride = match quant_type {
        QuantType::Int8 => hidden,
        QuantType::Int4 => hidden / 2,
    };

    let compute_lane = |i: usize, target: &mut f32| {
        let gate_row = &gate_weights[i * stride..(i + 1) * stride];
        let up_row = &up_weights[i * stride..(i + 1) * stride];
        let (dot_g, dot_u) = match (quant_type, mode) {
            (QuantType::Int8, GateUpMode::Paired) => dot_i8_f32_pair(gate_row, up_row, input),
            (QuantType::Int8, GateUpMode::Separate) => {
                (dot_i8_f32(gate_row, input), dot_i8_f32(up_row, input))
            }
            (QuantType::Int4, GateUpMode::Paired) => dot_i4_f32_pair(gate_row, up_row, input),
            (QuantType::Int4, GateUpMode::Separate) => {
                (dot_i4_f32(gate_row, input), dot_i4_f32(up_row, input))
            }
        };
        let g = dot_g * gate_scales[i];
        let u = dot_u * up_scales[i];
        let silu = g / (1.0 + (-g).exp());
        *target = silu * u;
    };

    activated
        .iter_mut()
        .enumerate()
        .for_each(|(i, target)| compute_lane(i, target));
}

/// Phiên bản tuần tự 100% của Down projection: không gọi par_iter_mut()
#[inline]
fn matvec_into_seq(
    weights: &[u8],
    scales: &[f32],
    input: &[f32],
    output: &mut [f32],
    quant_type: QuantType,
) {
    let row_len = match quant_type {
        QuantType::Int8 => input.len(),
        QuantType::Int4 => input.len() / 2,
    };
    let compute = |target: &mut f32, row: &[u8], scale: &f32| {
        let dot = match quant_type {
            QuantType::Int8 => dot_i8_f32(row, input),
            QuantType::Int4 => dot_i4_f32(row, input),
        };
        *target = dot * scale;
    };
    output
        .iter_mut()
        .zip(weights.chunks_exact(row_len))
        .zip(scales)
        .for_each(|((target, row), scale)| compute(target, row, scale));
}

#[cfg(test)]
mod kernel_tests {
    use super::*;
    use std::{hint::black_box, time::Instant};

    fn scalar(weights: &[u8], scales: &[f32], input: &[f32], output: &mut [f32]) {
        output
            .iter_mut()
            .zip(weights.chunks_exact(input.len()))
            .zip(scales)
            .for_each(|((target, row), scale)| {
                *target = row
                    .iter()
                    .zip(input)
                    .map(|(q, x)| (*q as i8 as f32) * x)
                    .sum::<f32>()
                    * scale;
            });
    }

    fn data(rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let weights = (0..rows * cols)
            .map(|i| ((i * 73 + 19) % 256) as u8)
            .collect();
        let scales = (0..rows).map(|i| 0.001 * (1 + i % 7) as f32).collect();
        let input = (0..cols)
            .map(|i| ((i * 31 % 101) as f32 - 50.0) / 51.0)
            .collect();
        (weights, scales, input)
    }

    #[test]
    fn blocked_kernel_matches_f64_reference_with_tails_and_signed_weights() {
        for cols in [1, 2, 7, 8, 9, 15, 16, 17, 1024, 2048, 2051] {
            let (weights, scales, input) = data(11, cols);
            let mut actual = vec![0.0; 11];
            matvec_into(&weights, &scales, &input, &mut actual, QuantType::Int8);
            for ((row, &scale), &got) in weights.chunks_exact(cols).zip(&scales).zip(&actual) {
                let expected = row
                    .iter()
                    .zip(&input)
                    .map(|(&q, &x)| q as i8 as f64 * x as f64)
                    .sum::<f64>()
                    * scale as f64;
                assert!(
                    (got as f64 - expected).abs() <= 2e-5 * (1.0 + expected.abs()),
                    "cols={cols}, actual={got}, expected={expected}"
                );
            }
        }
    }

    struct GateUpBenchCase {
        expert: Int8Expert,
        input: Vec<f32>,
        scratch: ExpertScratch,
    }

    impl GateUpBenchCase {
        fn new(hidden: usize, intermediate: usize, seed: u64) -> Self {
            let mut state = seed;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let weights = (0..3 * hidden * intermediate)
                .map(|_| next() as u8)
                .collect();
            let scales = (0..2 * intermediate + hidden)
                .map(|row| {
                    let cols = if row < 2 * intermediate {
                        hidden
                    } else {
                        intermediate
                    };
                    (0.5 + (next() % 1000) as f32 / 1000.0) / (127.0 * (cols as f32).sqrt())
                })
                .collect();
            let input = (0..hidden)
                .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            Self {
                expert: Int8Expert {
                    hidden,
                    intermediate,
                    weights,
                    scales,
                    quant_type: QuantType::Int8,
                },
                input,
                scratch: ExpertScratch {
                    gate_up_mode: GateUpMode::Paired,
                    activated: vec![0.0; intermediate],
                    output: vec![0.0; hidden],
                },
            }
        }

        // Both variants use SIMD-capable dot products, identical scale/SiLU and
        // identical down projection. Buffers are allocated before timing.
        fn run(&mut self, paired: bool, full_expert: bool) -> &[f32] {
            if paired && full_expert {
                return self
                    .expert
                    .forward_single_threaded(&self.input, &mut self.scratch)
                    .unwrap();
            }
            let e = &self.expert;
            let n = e.hidden * e.intermediate;
            if full_expert {
                assert_eq!(self.input.len(), e.hidden);
                assert_eq!(self.scratch.activated.len(), e.intermediate);
                assert_eq!(self.scratch.output.len(), e.hidden);
                finite(&self.input).unwrap();
            }
            if paired {
                fused_gate_up_swiglu_seq(
                    &e.weights[..n],
                    &e.weights[n..2 * n],
                    &e.scales[..e.intermediate],
                    &e.scales[e.intermediate..2 * e.intermediate],
                    &self.input,
                    &mut self.scratch.activated,
                    e.hidden,
                    QuantType::Int8,
                    GateUpMode::Paired,
                );
            } else {
                for (row, activated) in self.scratch.activated.iter_mut().enumerate() {
                    let start = row * e.hidden;
                    let gate = dot_i8_f32(&e.weights[start..start + e.hidden], &self.input)
                        * e.scales[row];
                    let up = dot_i8_f32(&e.weights[n + start..n + start + e.hidden], &self.input)
                        * e.scales[e.intermediate + row];
                    *activated = (gate / (1.0 + (-gate).exp())) * up;
                }
            }
            if full_expert {
                matvec_into_seq(
                    &e.weights[2 * n..],
                    &e.scales[2 * e.intermediate..],
                    &self.scratch.activated,
                    &mut self.scratch.output,
                    QuantType::Int8,
                );
                finite(&self.scratch.output).unwrap();
                &self.scratch.output
            } else {
                &self.scratch.activated
            }
        }
    }

    #[test]
    fn gate_up_separate_and_paired_match_through_swiglu_and_down_projection() {
        for (hidden, intermediate) in [(7, 5), (32, 9), (33, 17), (2048, 1024)] {
            for seed in [17, 73, 991] {
                let mut case = GateUpBenchCase::new(hidden, intermediate, seed);
                for full in [false, true] {
                    let reference = case.run(false, full).to_vec();
                    let actual = case.run(true, full);
                    assert_eq!(actual.len(), reference.len());
                    for (a, b) in actual.iter().zip(&reference) {
                        assert!(a.is_finite() && b.is_finite());
                        assert!((a-b).abs() < 2e-5, "hidden={hidden} intermediate={intermediate} seed={seed} full={full}: {a} vs {b}");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "release-only paired INT8 Gate-Up benchmark; use run_benchmark.py --gate-up-kernel"]
    fn benchmark_gate_up_paired() {
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(!cfg!(debug_assertions), "requires --release");
        }
        assert_eq!(std::env::var("RAYON_NUM_THREADS").unwrap(), "1");
        let testbed = std::env::var("MOE_GATE_UP_TESTBED").unwrap();
        assert!(matches!(testbed.as_str(), "i5" | "i7"));
        let avx2_fma = {
            #[cfg(target_arch = "x86_64")]
            {
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };
        assert!(
            avx2_fma,
            "this protocol requires both AVX2 and FMA; refuse scalar fallback"
        );
        let mut measurements = Vec::new();
        let mut parity = Vec::new();
        for seed in [17, 73, 991] {
            let mut case = GateUpBenchCase::new(2048, 1024, seed);
            for full in [false, true] {
                let scope = if full { "full_expert" } else { "gate_up_silu" };
                let expected = case.run(false, full).to_vec();
                assert!(expected.iter().all(|x| x.is_finite()));
                let actual = case.run(true, full);
                let max_error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(actual.iter().all(|x| x.is_finite()));
                assert!(max_error < 2e-5, "parity before timing failed: {max_error}");
                parity
                    .push(serde_json::json!({"seed":seed,"scope":scope,"max_abs_error":max_error}));
                for _ in 0..10 {
                    black_box(case.run(false, full));
                    black_box(case.run(true, full));
                }
                for pair in 0..9 {
                    for position in 0..2 {
                        let paired = (pair + position) % 2 == 1;
                        let started = Instant::now();
                        for _ in 0..100 {
                            black_box(black_box(&mut case).run(paired, full));
                        }
                        let elapsed_ns = started.elapsed().as_nanos() as u64;
                        measurements.push(serde_json::json!({
                            "seed":seed,"scope":scope,"pair":pair,"position":position,
                            "variant":if paired {"paired_simd"} else {"separate_simd"},
                            "iterations":100,"elapsed_ns":elapsed_ns
                        }));
                    }
                }
            }
        }
        let report = serde_json::json!({
            "schema_version":1,"kind":"gate_up_kernel","protocol":"gate-up-int8-v1",
            "testbed":testbed,"hidden":2048,"intermediate":1024,
            "seeds":[17,73,991],"pairs_per_seed_scope":9,"iterations":100,
            "warmup_per_variant_scope":10,"threads":1,"avx2_fma":avx2_fma,
            "dataset":"synthetic xorshift64 v1; fixed source recipe; no model weights",
            "scopes":["gate_up_silu","full_expert"],"parity":parity,"samples":measurements,
            "notes":["resident preallocated buffers; no file I/O; sequential expert task",
                "full expert candidate uses production forward_single_threaded",
                "no CPU affinity/frequency isolation; no inference speedup or L1 traffic claim"]
        });
        let path = std::env::var("MOE_GATE_UP_REPORT").unwrap();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        serde_json::to_writer_pretty(file, &report).unwrap();
    }

    fn resident_gate_up_batch(cases: &mut [GateUpBenchCase], row_parallel: bool) {
        use std::hint::black_box;
        if row_parallel {
            let case = &mut cases[0];
            black_box(
                case.expert
                    .forward_reuse(black_box(&case.input), &mut case.scratch)
                    .unwrap(),
            );
        } else {
            cases.par_iter_mut().for_each(|case| {
                black_box(
                    case.expert
                        .forward_single_threaded(black_box(&case.input), &mut case.scratch)
                        .unwrap(),
                );
            });
        }
    }

    #[test]
    fn resident_batch_uses_both_production_paths_and_modes() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            for row_parallel in [false, true] {
                let mut cases: Vec<_> = (17..25)
                    .map(|seed| GateUpBenchCase::new(33, 17, seed))
                    .collect();
                for case in &mut cases {
                    case.scratch.gate_up_mode = GateUpMode::Separate;
                }
                resident_gate_up_batch(&mut cases, row_parallel);
                let expected: Vec<_> = cases.iter().map(|c| c.scratch.output.clone()).collect();
                for case in &mut cases {
                    case.scratch.gate_up_mode = GateUpMode::Paired;
                }
                resident_gate_up_batch(&mut cases, row_parallel);
                for (case, reference) in
                    cases
                        .iter()
                        .zip(expected)
                        .take(if row_parallel { 1 } else { 8 })
                {
                    for (a, b) in case.scratch.output.iter().zip(reference) {
                        assert!(a.is_finite() && b.is_finite() && (a - b).abs() < 2e-5);
                    }
                }
            }
        });
    }

    #[test]
    #[ignore = "bounded real-weight resident benchmark; use --gate-up-resident"]
    fn benchmark_gate_up_resident() {
        use std::hint::black_box;
        use std::time::Instant;
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(!cfg!(debug_assertions), "requires release");
        }
        assert_eq!(std::env::var("RAYON_NUM_THREADS").unwrap(), "4");
        assert_eq!(rayon::current_num_threads(), 4);
        let testbed = std::env::var("MOE_GATE_UP_TESTBED").unwrap();
        assert!(matches!(testbed.as_str(), "i5" | "i7"));
        #[cfg(target_arch = "x86_64")]
        let avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2_fma = false;
        assert!(avx2_fma);
        let directory = std::env::var("MOE_RESIDENT_MODEL").unwrap();
        let config: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(std::path::Path::new(&directory).join("config.json")).unwrap(),
        )
        .unwrap();
        for (key, value) in [
            ("hidden_size", 2048),
            ("intermediate_size", 1024),
            ("num_hidden_layers", 16),
            ("num_experts", 64),
            ("num_experts_per_tok", 8),
        ] {
            assert_eq!(config[key].as_u64(), Some(value));
        }
        let index = TensorIndex::open(&directory).unwrap();
        // Fixed representative layers/IDs; no routing or post-measurement selection.
        let mut groups: Vec<Vec<GateUpBenchCase>> = [0, 7, 15]
            .into_iter()
            .map(|layer| {
                (0..8)
                    .map(|expert| {
                        let loaded =
                            Int8Expert::load(&index, layer, expert, 2048, 1024, 8 * 1024 * 1024)
                                .unwrap();
                        assert_eq!(loaded.quant_type, QuantType::Int8);
                        GateUpBenchCase {
                            expert: loaded,
                            input: vec![0.0; 2048],
                            scratch: ExpertScratch::new(2048, 1024),
                        }
                    })
                    .collect()
            })
            .collect();
        let payload_bytes: usize = groups
            .iter()
            .flatten()
            .map(|c| c.expert.resident_bytes())
            .sum();
        assert!(payload_bytes <= 256 * 1024 * 1024);
        let process_before = std::fs::read_to_string("/proc/self/status").unwrap();
        let io_before = std::fs::read_to_string("/proc/self/io").unwrap();
        let mut parity = Vec::new();
        let mut samples = Vec::new();
        for (layer, cases) in [0, 7, 15].into_iter().zip(&mut groups) {
            for seed in [17u64, 73, 991] {
                let mut state = seed;
                // Synthetic activations, real checkpoint weights/scales. Same for all variants.
                for case in cases.iter_mut() {
                    for x in &mut case.input {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        *x = (state % 2001) as f32 / 1000.0 - 1.0;
                    }
                }
                for (scope, row_parallel) in [("expert_tasks8", false), ("row_parallel", true)] {
                    for c in cases.iter_mut() {
                        c.scratch.gate_up_mode = GateUpMode::Separate;
                    }
                    resident_gate_up_batch(cases, row_parallel);
                    let references: Vec<Vec<f32>> =
                        cases.iter().map(|c| c.scratch.output.clone()).collect();
                    for c in cases.iter_mut() {
                        c.scratch.gate_up_mode = GateUpMode::Paired;
                    }
                    resident_gate_up_batch(cases, row_parallel);
                    let used = if row_parallel { 1 } else { 8 };
                    let mut max_abs_error = 0.0f32;
                    for (c, reference) in cases.iter().zip(&references).take(used) {
                        for (a, b) in c.scratch.output.iter().zip(reference) {
                            assert!(a.is_finite() && b.is_finite());
                            max_abs_error = max_abs_error.max((a - b).abs());
                        }
                    }
                    assert!(
                        max_abs_error < 2e-5,
                        "layer={layer} seed={seed} scope={scope} error={max_abs_error}"
                    );
                    parity.push(serde_json::json!({"layer":layer,"seed":seed,"scope":scope,"max_abs_error":max_abs_error}));
                    for _ in 0..10 {
                        for mode in [GateUpMode::Separate, GateUpMode::Paired] {
                            for c in cases.iter_mut() {
                                c.scratch.gate_up_mode = mode;
                            }
                            resident_gate_up_batch(cases, row_parallel);
                        }
                    }
                    for pair in 0..9 {
                        for position in 0..2 {
                            let paired = (pair + position) % 2 == 1;
                            for c in cases.iter_mut() {
                                c.scratch.gate_up_mode = if paired {
                                    GateUpMode::Paired
                                } else {
                                    GateUpMode::Separate
                                };
                            }
                            let started = Instant::now();
                            for _ in 0..40 {
                                resident_gate_up_batch(black_box(cases), row_parallel);
                            }
                            let elapsed_ns = started.elapsed().as_nanos() as u64;
                            samples.push(serde_json::json!({
                                "layer":layer,"seed":seed,"scope":scope,"pair":pair,"position":position,
                                "variant":if paired {"paired_simd"} else {"separate_simd"},
                                "iterations":40,"elapsed_ns":elapsed_ns
                            }));
                        }
                    }
                }
            }
        }
        let process_after = std::fs::read_to_string("/proc/self/status").unwrap();
        let io_after = std::fs::read_to_string("/proc/self/io").unwrap();
        let report = serde_json::json!({
            "schema_version":1,"kind":"gate_up_resident","protocol":"gate-up-resident-v1",
            "testbed":testbed,"hidden":2048,"intermediate":1024,"threads":4,
            "avx2_fma":avx2_fma,"layers":[0,7,15],"expert_ids":[0,1,2,3,4,5,6,7],
            "seeds":[17,73,991],"scopes":["expert_tasks8","row_parallel"],
            "pairs_per_case":9,"iterations":40,"warmup_per_variant_scope":10,
            "payload_bytes":payload_bytes,"payload_budget_bytes":256*1024*1024,
            "dataset":"real INT8 weights/scales; synthetic xorshift64 activations; no model forward",
            "process_before":process_before,"process_after":process_after,
            "io_before":io_before,"io_after":io_after,
            "parity":parity,"samples":samples
        });
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(std::env::var("MOE_GATE_UP_REPORT").unwrap())
            .unwrap();
        serde_json::to_writer_pretty(file, &report).unwrap();
    }

    #[test]
    #[ignore = "release-only microbenchmark; writes JSON to MOE_KERNEL_REPORT"]
    fn benchmark_int8_kernel() {
        // Runtime guard: debug builds must still compile the ignored benchmark.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(!cfg!(debug_assertions), "run with --release");
        }
        let mut results = Vec::new();
        for (rows, cols) in [(1024, 2048), (2048, 1024)] {
            let (weights, scales, input) = data(rows, cols);
            let mut output = vec![0.0; rows];
            let mut expected = vec![0.0; rows];
            scalar(&weights, &scales, &input, &mut expected);
            matvec_into(&weights, &scales, &input, &mut output, QuantType::Int8);
            let max_error = output
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            for _ in 0..10 {
                scalar(black_box(&weights), &scales, &input, black_box(&mut output));
                matvec_into(
                    black_box(&weights),
                    &scales,
                    &input,
                    black_box(&mut output),
                    QuantType::Int8,
                );
            }
            let mut scalar_ms = Vec::new();
            let mut blocked_ms = Vec::new();
            for sample in 0..9 {
                for variant in 0..2 {
                    let use_scalar = (sample + variant) % 2 == 0;
                    let started = Instant::now();
                    for _ in 0..30 {
                        if use_scalar {
                            scalar(
                                black_box(&weights),
                                black_box(&scales),
                                black_box(&input),
                                black_box(&mut output),
                            );
                        } else {
                            matvec_into(
                                black_box(&weights),
                                black_box(&scales),
                                black_box(&input),
                                black_box(&mut output),
                                QuantType::Int8,
                            );
                        }
                        black_box(&output);
                    }
                    let ms = started.elapsed().as_secs_f64() * 1000.0 / 30.0;
                    if use_scalar {
                        scalar_ms.push(ms);
                    } else {
                        blocked_ms.push(ms);
                    }
                }
            }
            results.push(
                serde_json::json!({"rows": rows, "cols": cols, "scalar_ms": scalar_ms,
                "blocked_ms": blocked_ms, "max_abs_error_vs_scalar": max_error}),
            );
        }
        let report = serde_json::json!({"schema_version": 1, "samples": 9, "iterations_per_sample": 30,
            "warmup_per_variant": 10, "threads": 1, "results": results,
            "notes": ["synthetic deterministic weights and input; resident buffers; no I/O or allocation in timing",
                "alternating measurement order; default release target; CPU frequency and background load uncontrolled"]});
        let path = std::env::var("MOE_KERNEL_REPORT").expect("set MOE_KERNEL_REPORT output path");
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }

    #[test]
    fn test_forward_single_threaded_matches_forward_reuse() {
        let hidden = 64;
        let intermediate = 32;
        let n = hidden * intermediate;
        let weights: Vec<u8> = (0..3 * n).map(|i| ((i * 17 + 5) % 256) as u8).collect();
        let scales: Vec<f32> = (0..2 * intermediate + hidden)
            .map(|i| 0.005 * (1 + i % 13) as f32)
            .collect();
        let expert = Int8Expert::from_raw(hidden, intermediate, weights, scales);
        assert_eq!(expert.quant_type(), QuantType::Int8);
        let input: Vec<f32> = (0..hidden)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 12.0)
            .collect();

        let mut scratch1 = ExpertScratch::new(hidden, intermediate);
        let mut scratch2 = ExpertScratch::new(hidden, intermediate);

        let out_reuse = expert
            .forward_reuse(&input, &mut scratch1)
            .unwrap()
            .to_vec();
        let out_seq = expert
            .forward_single_threaded(&input, &mut scratch2)
            .unwrap()
            .to_vec();

        assert_eq!(out_reuse.len(), out_seq.len());
        for (i, (a, b)) in out_reuse.iter().zip(&out_seq).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "Bit mismatch at lane {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_dot_i8_f32_simd_parity_with_scalar() {
        // Kiểm tra tính nhất quán giữa kernel SIMD AVX2/FMA và hàm tham chiếu scalar cho INT8
        for len in [1, 2, 7, 8, 9, 15, 16, 31, 32, 33, 64, 127, 128, 1024, 2048] {
            let row: Vec<u8> = (0..len).map(|i| ((i * 37 + 13) % 256) as u8).collect();
            let input: Vec<f32> = (0..len)
                .map(|i| ((i * 19 % 47) as f32 - 23.0) / 24.0)
                .collect();

            let scalar_res = dot_i8_f32_scalar(&row, &input);
            let dispatch_res = dot_i8_f32(&row, &input);

            let diff = (scalar_res - dispatch_res).abs();
            let tolerance = 1e-4 * (1.0 + scalar_res.abs());
            assert!(
                diff <= tolerance,
                "Length {len}: scalar={scalar_res}, dispatch={dispatch_res}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_dot_i4_f32_simd_parity_with_scalar() {
        // Kiểm tra tính nhất quán giữa kernel SIMD AVX2/FMA và hàm tham chiếu scalar cho INT4
        for len in [
            1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 33, 64, 127, 128, 1024, 2048,
        ] {
            let byte_count = (len + 1) / 2;
            let row: Vec<u8> = (0..byte_count)
                .map(|i| ((i * 41 + 17) % 256) as u8)
                .collect();
            let input: Vec<f32> = (0..len)
                .map(|i| ((i * 29 % 53) as f32 - 26.0) / 27.0)
                .collect();

            let scalar_res = dot_i4_f32_scalar(&row, &input);
            let dispatch_res = dot_i4_f32(&row, &input);

            let diff = (scalar_res - dispatch_res).abs();
            let tolerance = 1e-4 * (1.0 + scalar_res.abs());
            assert!(
                diff <= tolerance,
                "Length {len}: scalar={scalar_res}, dispatch={dispatch_res}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_dot_i8_f32_pair_parity() {
        // Kiểm tra tính nhất quán của kernel cặp Gate-Up INT8 AVX2 SIMD với các hàm dot đơn lẻ
        for len in [
            1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 33, 64, 127, 128, 1024, 2048,
        ] {
            let gate_row: Vec<u8> = (0..len).map(|i| ((i * 31 + 7) % 256) as u8).collect();
            let up_row: Vec<u8> = (0..len).map(|i| ((i * 47 + 13) % 256) as u8).collect();
            let input: Vec<f32> = (0..len)
                .map(|i| ((i * 19 % 43) as f32 - 21.0) / 22.0)
                .collect();

            let ref_g = dot_i8_f32(&gate_row, &input);
            let ref_u = dot_i8_f32(&up_row, &input);
            let (pair_g, pair_u) = dot_i8_f32_pair(&gate_row, &up_row, &input);

            let diff_g = (ref_g - pair_g).abs();
            let tol_g = 1e-4 * (1.0 + ref_g.abs());
            assert!(
                diff_g <= tol_g,
                "Length {len} Gate mismatch: ref={ref_g}, pair={pair_g}, diff={diff_g}"
            );

            let diff_u = (ref_u - pair_u).abs();
            let tol_u = 1e-4 * (1.0 + ref_u.abs());
            assert!(
                diff_u <= tol_u,
                "Length {len} Up mismatch: ref={ref_u}, pair={pair_u}, diff={diff_u}"
            );
        }
    }

    #[test]
    fn test_dot_i4_f32_pair_parity() {
        // Kiểm tra tính nhất quán của kernel cặp Gate-Up INT4 AVX2 SIMD với các hàm dot đơn lẻ
        for len in [
            1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 33, 64, 127, 128, 1024, 2048,
        ] {
            let byte_count = (len + 1) / 2;
            let gate_row: Vec<u8> = (0..byte_count)
                .map(|i| ((i * 37 + 11) % 256) as u8)
                .collect();
            let up_row: Vec<u8> = (0..byte_count)
                .map(|i| ((i * 53 + 23) % 256) as u8)
                .collect();
            let input: Vec<f32> = (0..len)
                .map(|i| ((i * 29 % 53) as f32 - 26.0) / 27.0)
                .collect();

            let ref_g = dot_i4_f32(&gate_row, &input);
            let ref_u = dot_i4_f32(&up_row, &input);
            let (pair_g, pair_u) = dot_i4_f32_pair(&gate_row, &up_row, &input);

            let diff_g = (ref_g - pair_g).abs();
            let tol_g = 1e-4 * (1.0 + ref_g.abs());
            assert!(
                diff_g <= tol_g,
                "Length {len} INT4 Gate mismatch: ref={ref_g}, pair={pair_g}, diff={diff_g}"
            );

            let diff_u = (ref_u - pair_u).abs();
            let tol_u = 1e-4 * (1.0 + ref_u.abs());
            assert!(
                diff_u <= tol_u,
                "Length {len} INT4 Up mismatch: ref={ref_u}, pair={pair_u}, diff={diff_u}"
            );
        }
    }

    #[test]
    fn test_forward_single_threaded_matches_forward_reuse_int4() {
        let hidden = 64;
        let intermediate = 32;
        let n = (hidden / 2) * intermediate;
        let weights: Vec<u8> = (0..3 * n).map(|i| ((i * 23 + 7) % 256) as u8).collect();
        let scales: Vec<f32> = (0..2 * intermediate + hidden)
            .map(|i| 0.003 * (1 + i % 11) as f32)
            .collect();
        let expert = Int8Expert::from_raw(hidden, intermediate, weights, scales);
        assert_eq!(expert.quant_type(), QuantType::Int4);
        let input: Vec<f32> = (0..hidden)
            .map(|i| ((i * 11 % 29) as f32 - 14.0) / 15.0)
            .collect();

        let mut scratch1 = ExpertScratch::new(hidden, intermediate);
        let mut scratch2 = ExpertScratch::new(hidden, intermediate);

        let out_reuse = expert
            .forward_reuse(&input, &mut scratch1)
            .unwrap()
            .to_vec();
        let out_seq = expert
            .forward_single_threaded(&input, &mut scratch2)
            .unwrap()
            .to_vec();

        assert_eq!(out_reuse.len(), out_seq.len());
        for (i, (a, b)) in out_reuse.iter().zip(&out_seq).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "INT4 Bit mismatch at lane {i}: {a} vs {b}"
            );
        }
    }
}

pub(crate) mod batch_prototype;
