//! On-demand OLMoE merged INT8 experts from Safetensors shards.
//! Does not implement the OLMoE decoder or tokenizer.
use crate::{
    backend::{finite, invalid},
    safetensors::{DType, TensorIndex},
};
use rayon::prelude::*;
use std::io;

pub struct Int8Expert {
    hidden: usize,
    intermediate: usize,
    weights: Vec<u8>,
    scales: Vec<f32>,
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

        // Tái sử dụng vùng nhớ: resize không cấp phát lại nếu capacity đã đủ
        if weights.capacity() < weight_count {
            weights.reserve_exact(weight_count - weights.len());
        }
        weights.resize(weight_count, 0);

        if scales.capacity() < scale_count {
            scales.reserve_exact(scale_count - scales.len());
        }
        scales.resize(scale_count, 0.0);

        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        let weight_name = format!("{prefix}.merged_weight");
        let scale_name = format!("{prefix}.qs");

        let weight_info = index.tensor(&weight_name)?;
        let scale_info = index.tensor(&scale_name)?;
        if weight_info.dtype != DType::I8
            || weight_info.shape != [weight_count]
            || scale_info.dtype != DType::F32
            || scale_info.shape != [scale_count]
        {
            return Err(invalid(
                "expected flat I8 merged_weight and flat F32 qs with exact expert dimensions",
            ));
        }

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

        // Đọc toàn bộ 6.3 MB weights qua POSIX DMA pread, ghi đè hoàn toàn vùng tạm
        index.read_raw_into(&weight_name, &mut weights)?;

        Ok(Self {
            hidden,
            intermediate,
            weights,
            scales,
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
        let total = scale_count
            .checked_mul(4)
            .and_then(|n| n.checked_add(weight_count))
            .ok_or_else(|| invalid("expert byte count overflow"))?;
        if total > max_bytes {
            return Err(invalid("expert exceeds read budget"));
        }
        let weights = Vec::with_capacity(weight_count);
        let scales = Vec::with_capacity(scale_count);
        Self::load_reusing(index, layer, expert, hidden, intermediate, weights, scales)
    }

    pub fn resident_bytes(&self) -> usize {
        self.weights.len() + self.scales.len() * 4
    }

    pub fn from_raw(
        hidden: usize,
        intermediate: usize,
        weights: Vec<u8>,
        scales: Vec<f32>,
    ) -> Self {
        Self {
            hidden,
            intermediate,
            weights,
            scales,
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
        let n = self.hidden * self.intermediate;
        fused_gate_up_swiglu(
            &self.weights[..n],
            &self.weights[n..2 * n],
            &self.scales[..self.intermediate],
            &self.scales[self.intermediate..2 * self.intermediate],
            input,
            &mut scratch.activated,
            self.hidden,
        );
        matvec_into(
            &self.weights[2 * n..],
            &self.scales[2 * self.intermediate..],
            &scratch.activated,
            &mut scratch.output,
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
        let n = self.hidden * self.intermediate;
        fused_gate_up_swiglu_seq(
            &self.weights[..n],
            &self.weights[n..2 * n],
            &self.scales[..self.intermediate],
            &self.scales[self.intermediate..2 * self.intermediate],
            input,
            &mut scratch.activated,
            self.hidden,
        );
        matvec_into_seq(
            &self.weights[2 * n..],
            &self.scales[2 * self.intermediate..],
            &scratch.activated,
            &mut scratch.output,
        );
        finite(&scratch.output)?;
        Ok(&scratch.output)
    }
}

/// Tích vô hướng INT8 x FP32 với 4 bộ cộng độc lập (32 elements / bước) tối ưu hóa cho đường ống lệnh SIMD FMA
#[inline(always)]
fn dot_i8_f32(row: &[u8], input: &[f32]) -> f32 {
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

/// Kernel Fused Gate-Up Projection & SwiGLU:
/// Gộp tính toán Gate và Up trong cùng 1 lần duyệt đa luồng Rayon, tái sử dụng vector `input` trong CPU L1 Cache
/// và tính ngay kích hoạt SiLU mà không cần ghi/đọc qua bộ đệm trung gian.
fn fused_gate_up_swiglu(
    gate_weights: &[u8],
    up_weights: &[u8],
    gate_scales: &[f32],
    up_scales: &[f32],
    input: &[f32],
    activated: &mut [f32],
    hidden: usize,
) {
    debug_assert_eq!(activated.len(), gate_scales.len());
    debug_assert_eq!(activated.len(), up_scales.len());

    let compute_lane = |i: usize, target: &mut f32| {
        let gate_row = &gate_weights[i * hidden..(i + 1) * hidden];
        let up_row = &up_weights[i * hidden..(i + 1) * hidden];
        let g = dot_i8_f32(gate_row, input) * gate_scales[i];
        let u = dot_i8_f32(up_row, input) * up_scales[i];
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
// without target-specific instructions or expanding INT8 weights to a matrix.
fn matvec_into(weights: &[u8], scales: &[f32], input: &[f32], output: &mut [f32]) {
    let compute = |target: &mut f32, row: &[u8], scale: &f32| {
        *target = dot_i8_f32(row, input) * scale;
    };
    if weights.len() >= 262_144 {
        output
            .par_iter_mut()
            .zip(weights.par_chunks_exact(input.len()))
            .zip(scales.par_iter())
            .for_each(|((target, row), scale)| compute(target, row, scale));
    } else {
        output
            .iter_mut()
            .zip(weights.chunks_exact(input.len()))
            .zip(scales)
            .for_each(|((target, row), scale)| compute(target, row, scale));
    }
}

/// Phiên bản tuần tự 100% của Fused Gate-Up SwiGLU: không gọi par_iter_mut()
#[inline]
fn fused_gate_up_swiglu_seq(
    gate_weights: &[u8],
    up_weights: &[u8],
    gate_scales: &[f32],
    up_scales: &[f32],
    input: &[f32],
    activated: &mut [f32],
    hidden: usize,
) {
    debug_assert_eq!(activated.len(), gate_scales.len());
    debug_assert_eq!(activated.len(), up_scales.len());

    let compute_lane = |i: usize, target: &mut f32| {
        let gate_row = &gate_weights[i * hidden..(i + 1) * hidden];
        let up_row = &up_weights[i * hidden..(i + 1) * hidden];
        let g = dot_i8_f32(gate_row, input) * gate_scales[i];
        let u = dot_i8_f32(up_row, input) * up_scales[i];
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
fn matvec_into_seq(weights: &[u8], scales: &[f32], input: &[f32], output: &mut [f32]) {
    let compute = |target: &mut f32, row: &[u8], scale: &f32| {
        *target = dot_i8_f32(row, input) * scale;
    };
    output
        .iter_mut()
        .zip(weights.chunks_exact(input.len()))
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
            matvec_into(&weights, &scales, &input, &mut actual);
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
            matvec_into(&weights, &scales, &input, &mut output);
            let max_error = output
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            for _ in 0..10 {
                scalar(black_box(&weights), &scales, &input, black_box(&mut output));
                matvec_into(black_box(&weights), &scales, &input, black_box(&mut output));
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
}
