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
    gate: Vec<f32>,
    up: Vec<f32>,
    activated: Vec<f32>,
    output: Vec<f32>,
}

impl ExpertScratch {
    pub(crate) fn new(hidden: usize, intermediate: usize) -> Self {
        Self {
            gate: vec![0.0; intermediate],
            up: vec![0.0; intermediate],
            activated: vec![0.0; intermediate],
            output: vec![0.0; hidden],
        }
    }
}

impl Int8Expert {
    pub fn load(
        index: &TensorIndex,
        layer: usize,
        expert: usize,
        hidden: usize,
        intermediate: usize,
        max_bytes: usize,
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
        let total = scale_count
            .checked_mul(4)
            .and_then(|n| n.checked_add(weight_count))
            .ok_or_else(|| invalid("expert byte count overflow"))?;
        if total > max_bytes {
            return Err(invalid("expert exceeds read budget"));
        }
        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        let weight_name = format!("{prefix}.merged_weight");
        let scale_name = format!("{prefix}.qs");
        let weights = index.tensor(&weight_name)?;
        let scales = index.tensor(&scale_name)?;
        if weights.dtype != DType::I8
            || weights.shape != [weight_count]
            || scales.dtype != DType::F32
            || scales.shape != [scale_count]
        {
            return Err(invalid(
                "expected flat I8 merged_weight and flat F32 qs with exact expert dimensions",
            ));
        }
        let scales = index.read_f32(&scale_name, scale_count * 4)?;
        if scales.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(invalid("expert scales must be finite and positive"));
        }
        let weights = index.read_raw(&weight_name, weight_count)?;
        Ok(Self {
            hidden,
            intermediate,
            weights,
            scales,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.weights.len() + self.scales.len() * 4
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
        if scratch.gate.len() != self.intermediate || scratch.output.len() != self.hidden {
            return Err(invalid("expert scratch dimension mismatch"));
        }
        finite(input)?;
        let n = self.hidden * self.intermediate;
        matvec_into(
            &self.weights[..n],
            &self.scales[..self.intermediate],
            input,
            &mut scratch.gate,
        );
        matvec_into(
            &self.weights[n..2 * n],
            &self.scales[self.intermediate..2 * self.intermediate],
            input,
            &mut scratch.up,
        );
        finite(&scratch.gate)?;
        finite(&scratch.up)?;
        scratch
            .activated
            .iter_mut()
            .zip(&scratch.gate)
            .zip(&scratch.up)
            .for_each(|((target, g), u)| *target = g / (1.0 + (-g).exp()) * u);
        finite(&scratch.activated)?;
        matvec_into(
            &self.weights[2 * n..],
            &self.scales[2 * self.intermediate..],
            &scratch.activated,
            &mut scratch.output,
        );
        finite(&scratch.output)?;
        Ok(&scratch.output)
    }
}

// Separate accumulators shorten the dependency chain and let LLVM vectorize
// without target-specific instructions or expanding INT8 weights to a matrix.
fn matvec_into(weights: &[u8], scales: &[f32], input: &[f32], output: &mut [f32]) {
    let compute = |target: &mut f32, row: &[u8], scale: &f32| {
        let mut sums = [0.0f32; 8];
        let mut qs = row.chunks_exact(8);
        let mut xs = input.chunks_exact(8);
        for (q, x) in qs.by_ref().zip(xs.by_ref()) {
            for lane in 0..8 {
                sums[lane] += (q[lane] as i8 as f32) * x[lane];
            }
        }
        let mut sum = sums.into_iter().sum::<f32>();
        for (&q, &x) in qs.remainder().iter().zip(xs.remainder()) {
            sum += (q as i8 as f32) * x;
        }
        *target = sum * scale;
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
}
