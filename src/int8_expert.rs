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

pub struct Int8Expert {
    hidden: usize,
    intermediate: usize,
    weights: Vec<u8>,
    scales: Vec<f32>,
    quant_type: QuantType,
}

pub(crate) struct ExpertScratch {
    pub(crate) activated: Vec<f32>,
    pub(crate) output: Vec<f32>,
}

impl ExpertScratch {
    pub(crate) fn new(hidden: usize, intermediate: usize) -> Self {
        Self {
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
        let (g, u) = match quant_type {
            QuantType::Int8 => (
                dot_i8_f32(gate_row, input) * gate_scales[i],
                dot_i8_f32(up_row, input) * up_scales[i],
            ),
            QuantType::Int4 => (
                dot_i4_f32(gate_row, input) * gate_scales[i],
                dot_i4_f32(up_row, input) * up_scales[i],
            ),
        };
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
        let (g, u) = match quant_type {
            QuantType::Int8 => (
                dot_i8_f32(gate_row, input) * gate_scales[i],
                dot_i8_f32(up_row, input) * up_scales[i],
            ),
            QuantType::Int4 => (
                dot_i4_f32(gate_row, input) * gate_scales[i],
                dot_i4_f32(up_row, input) * up_scales[i],
            ),
        };
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

    #[test]
    fn test_dot_i4_f32_simd_parity_with_scalar() {
        // Kiểm tra tính nhất quán giữa kernel SIMD AVX2/FMA và hàm tham chiếu scalar cho INT4
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 33, 64, 127, 128, 1024, 2048] {
            let byte_count = (len + 1) / 2;
            let row: Vec<u8> = (0..byte_count).map(|i| ((i * 41 + 17) % 256) as u8).collect();
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
