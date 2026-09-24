//! Small, portable f32 CPU reference kernels. No GPU allocation or kernels.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io, sync::OnceLock};

// A tile is small enough to keep one head's K/V window cache-friendly while
// avoiding per-token iterator setup in the long-context path.
const ATTENTION_BLOCK_POSITIONS: usize = 32;

pub(crate) fn configure_parallelism() -> io::Result<()> {
    static CONFIGURED: OnceLock<()> = OnceLock::new();
    if CONFIGURED.get().is_some() {
        return Ok(());
    }
    let threads = match std::env::var("RAYON_NUM_THREADS") {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|threads| *threads > 0)
            .ok_or_else(|| invalid("RAYON_NUM_THREADS must be a positive integer"))?,
        Err(std::env::VarError::NotPresent) => std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(8),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(invalid("RAYON_NUM_THREADS must be valid UTF-8"))
        }
    };
    // Another library may already own the global pool. In that case Rayon keeps
    // that valid pool and the engine uses it instead of replacing it.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
    let _ = CONFIGURED.set(());
    Ok(())
}

pub(crate) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f32>,
}

impl Matrix {
    pub fn validate(&self, rows: usize, columns: usize) -> io::Result<()> {
        if rows == 0
            || columns == 0
            || self.rows != rows
            || self.columns != columns
            || rows.checked_mul(columns) != Some(self.values.len())
            || self.values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid matrix shape or non-finite weights"));
        }
        Ok(())
    }

    pub fn multiply(&self, input: &[f32]) -> io::Result<Vec<f32>> {
        self.validate(self.rows, input.len())?;
        self.multiply_validated_input(input)
    }

    /// Fast path for weights already validated by a checkpoint loader.
    pub(crate) fn multiply_loaded_into(&self, input: &[f32], output: &mut [f32]) -> io::Result<()> {
        if input.len() != self.columns {
            return Err(invalid("matrix input dimension mismatch"));
        }
        if output.len() != self.rows {
            return Err(invalid("matrix output dimension mismatch"));
        }
        finite(input)?;
        if self.values.len() >= 262_144 {
            output
                .par_iter_mut()
                .zip(self.values.par_chunks_exact(self.columns))
                .for_each(|(target, row)| *target = dot_product(row, input));
        } else {
            output
                .iter_mut()
                .zip(self.values.chunks_exact(self.columns))
                .for_each(|(target, row)| *target = dot_product(row, input));
        }
        finite(output)
    }

    fn multiply_validated_input(&self, input: &[f32]) -> io::Result<Vec<f32>> {
        finite(input)?;
        let result: Vec<_> = self
            .values
            .chunks_exact(self.columns)
            .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
            .collect();
        finite(&result)?;
        Ok(result)
    }
}

// Independent accumulators shorten the dependency chain and allow LLVM to
// vectorize portable release builds without requiring a CPU-specific target.
pub(crate) fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let mut sums = [0.0f32; 8];
    let mut left_chunks = left.chunks_exact(8);
    let mut right_chunks = right.chunks_exact(8);
    for (left, right) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
        for lane in 0..8 {
            sums[lane] += left[lane] * right[lane];
        }
    }
    let mut sum = sums.into_iter().sum::<f32>();
    for (&left, &right) in left_chunks.remainder().iter().zip(right_chunks.remainder()) {
        sum += left * right;
    }
    sum
}

/// Causal attention over a contiguous KV cache using online softmax. Heads
/// write disjoint context slices, so long contexts use Rayon without a score
/// buffer or a second pass over values.
pub(crate) fn causal_attention_into(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    num_attention_heads: usize,
    num_key_value_heads: usize,
    context: &mut [f32],
) -> io::Result<()> {
    if num_attention_heads == 0
        || num_key_value_heads == 0
        || num_attention_heads % num_key_value_heads != 0
        || query.len() != context.len()
        || query.len() % num_attention_heads != 0
        || keys.len() != values.len()
    {
        return Err(invalid("attention dimension mismatch"));
    }
    let head_width = query.len() / num_attention_heads;
    let key_value_width = head_width
        .checked_mul(num_key_value_heads)
        .ok_or_else(|| invalid("attention dimension overflow"))?;
    if key_value_width == 0 || keys.len() % key_value_width != 0 {
        return Err(invalid("attention KV dimension mismatch"));
    }
    let positions = keys.len() / key_value_width;
    if positions == 0 {
        return Err(invalid("attention requires at least one KV position"));
    }
    finite(query)?;
    finite(keys)?;
    finite(values)?;
    let group = num_attention_heads / num_key_value_heads;
    let scale = (head_width as f32).sqrt();
    let run_head = |head: usize, output: &mut [f32]| -> io::Result<()> {
        let key_start = (head / group) * head_width;
        let query = &query[head * head_width..(head + 1) * head_width];
        output.fill(0.0);
        let mut maximum = f32::NEG_INFINITY;
        let mut normalizer = 0.0f32;
        for (key_block, value_block) in keys
            .chunks(ATTENTION_BLOCK_POSITIONS * key_value_width)
            .zip(values.chunks(ATTENTION_BLOCK_POSITIONS * key_value_width))
        {
            for (key, value) in key_block
                .chunks_exact(key_value_width)
                .zip(value_block.chunks_exact(key_value_width))
            {
                let score = dot_product(query, &key[key_start..key_start + head_width]) / scale;
                if !score.is_finite() {
                    return Err(invalid("non-finite attention score"));
                }
                let weight = if score > maximum {
                    let rescale = (maximum - score).exp();
                    for target in output.iter_mut() {
                        *target *= rescale;
                    }
                    normalizer *= rescale;
                    maximum = score;
                    1.0
                } else {
                    (score - maximum).exp()
                };
                normalizer += weight;
                for (target, value) in output
                    .iter_mut()
                    .zip(&value[key_start..key_start + head_width])
                {
                    *target += weight * value;
                }
            }
        }
        if !normalizer.is_finite() || normalizer <= 0.0 {
            return Err(invalid("invalid attention softmax normalizer"));
        }
        for value in output.iter_mut() {
            *value /= normalizer;
        }
        finite(output)
    };
    if num_attention_heads * positions >= 4096 {
        context
            .par_chunks_exact_mut(head_width)
            .enumerate()
            .try_for_each(|(head, output)| run_head(head, output))?;
    } else {
        for head in 0..num_attention_heads {
            run_head(
                head,
                &mut context[head * head_width..(head + 1) * head_width],
            )?;
        }
    }
    Ok(())
}

pub(crate) fn finite(values: &[f32]) -> io::Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite computation result"));
    }
    Ok(())
}

pub(crate) fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let scale = (input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32 + eps).sqrt();
    input
        .iter()
        .zip(weight)
        .map(|(x, w)| x / scale * w)
        .collect()
}

pub(crate) fn rms_norm_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) -> io::Result<()> {
    if input.len() != weight.len() || input.len() != output.len() || input.is_empty() {
        return Err(invalid("RMSNorm dimension mismatch"));
    }
    finite(input)?;
    let scale =
        (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32 + eps).sqrt();
    output
        .iter_mut()
        .zip(input)
        .zip(weight)
        .for_each(|((target, value), gain)| *target = value / scale * gain);
    finite(output)
}

pub(crate) fn rms_norm_in_place(values: &mut [f32], weight: &[f32], eps: f32) -> io::Result<()> {
    if values.len() != weight.len() || values.is_empty() {
        return Err(invalid("RMSNorm dimension mismatch"));
    }
    finite(values)?;
    let scale =
        (values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32 + eps).sqrt();
    values
        .iter_mut()
        .zip(weight)
        .for_each(|(value, gain)| *value = *value / scale * gain);
    finite(values)
}

pub(crate) fn softmax(values: &mut [f32]) -> io::Result<()> {
    finite(values)?;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = values
        .iter_mut()
        .map(|x| {
            *x = (*x - max).exp();
            *x
        })
        .sum();
    for x in values {
        *x /= sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{hint::black_box, time::Instant};

    fn scalar_dot(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum()
    }

    fn data(rows: usize, columns: usize) -> (Matrix, Vec<f32>) {
        let values = (0..rows * columns)
            .map(|index| ((index * 73 % 257) as f32 - 128.0) / 129.0)
            .collect();
        let input = (0..columns)
            .map(|index| ((index * 31 % 101) as f32 - 50.0) / 51.0)
            .collect();
        (
            Matrix {
                rows,
                columns,
                values,
            },
            input,
        )
    }

    #[test]
    fn loaded_matrix_path_checks_input_and_matches_checked_path() {
        let matrix = Matrix {
            rows: 2,
            columns: 2,
            values: vec![1.0, 2.0, 3.0, 4.0],
        };
        let mut output = vec![0.0; 2];
        matrix
            .multiply_loaded_into(&[2.0, 1.0], &mut output)
            .unwrap();
        assert_eq!(output, matrix.multiply(&[2.0, 1.0]).unwrap());
        assert!(matrix.multiply_loaded_into(&[1.0], &mut output).is_err());
        assert!(matrix
            .multiply_loaded_into(&[f32::NAN, 1.0], &mut output)
            .is_err());
        assert!(matrix
            .multiply_loaded_into(&[2.0, 1.0], &mut output[..1])
            .is_err());

        let mut normalized = vec![0.0; 2];
        rms_norm_into(&[3.0, 4.0], &[1.0, 1.0], 1e-5, &mut normalized).unwrap();
        let mut in_place = vec![3.0, 4.0];
        rms_norm_in_place(&mut in_place, &[1.0, 1.0], 1e-5).unwrap();
        assert_eq!(normalized, in_place);
    }

    #[test]
    fn blocked_dot_matches_f64_reference_including_tails() {
        for columns in [1, 2, 7, 8, 9, 15, 16, 17, 1024, 2048, 2051] {
            let (matrix, input) = data(11, columns);
            for row in matrix.values.chunks_exact(columns) {
                let actual = dot_product(row, &input) as f64;
                let expected = row
                    .iter()
                    .zip(&input)
                    .map(|(&left, &right)| left as f64 * right as f64)
                    .sum::<f64>();
                assert!(
                    (actual - expected).abs() <= 2e-5 * (1.0 + expected.abs()),
                    "columns={columns}, actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn causal_attention_matches_scalar_for_mha_and_gqa() {
        for (heads, kv_heads) in [(2, 2), (4, 2)] {
            let width = 8;
            let positions = 17;
            let query: Vec<_> = (0..heads * width)
                .map(|i| (i as f32 * 0.17).sin())
                .collect();
            let keys: Vec<_> = (0..positions * kv_heads * width)
                .map(|i| (i as f32 * 0.11).cos())
                .collect();
            let values: Vec<_> = (0..positions * kv_heads * width)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let mut actual = vec![0.0; heads * width];
            causal_attention_into(&query, &keys, &values, heads, kv_heads, &mut actual).unwrap();

            let mut expected = vec![0.0; heads * width];
            for head in 0..heads {
                let key_start = (head / (heads / kv_heads)) * width;
                let mut score: Vec<_> = keys
                    .chunks_exact(kv_heads * width)
                    .map(|key| {
                        scalar_dot(
                            &query[head * width..(head + 1) * width],
                            &key[key_start..key_start + width],
                        ) / (width as f32).sqrt()
                    })
                    .collect();
                softmax(&mut score).unwrap();
                for (weight, value) in score.iter().zip(values.chunks_exact(kv_heads * width)) {
                    for offset in 0..width {
                        expected[head * width + offset] += weight * value[key_start + offset];
                    }
                }
            }
            for (observed, expected) in actual.iter().zip(expected) {
                assert!((observed - expected).abs() < 2e-6);
            }
        }
    }

    fn materialized_attention(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        heads: usize,
        width: usize,
        output: &mut [f32],
    ) {
        for head in 0..heads {
            let mut score: Vec<_> = keys
                .chunks_exact(heads * width)
                .map(|key| {
                    scalar_dot(
                        &query[head * width..(head + 1) * width],
                        &key[head * width..(head + 1) * width],
                    ) / (width as f32).sqrt()
                })
                .collect();
            softmax(&mut score).unwrap();
            let target = &mut output[head * width..(head + 1) * width];
            target.fill(0.0);
            for (weight, value) in score.iter().zip(values.chunks_exact(heads * width)) {
                for (target, value) in target
                    .iter_mut()
                    .zip(&value[head * width..(head + 1) * width])
                {
                    *target += weight * value;
                }
            }
        }
    }

    #[test]
    #[ignore = "release-only A/B benchmark; writes JSON to MOE_ATTENTION_REPORT"]
    fn benchmark_attention_online_vs_materialized() {
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(!cfg!(debug_assertions), "run with --release");
        }
        let (heads, width, positions) = (16, 128, 469);
        let query: Vec<_> = (0..heads * width)
            .map(|i| (i as f32 * 0.17).sin())
            .collect();
        let keys: Vec<_> = (0..positions * heads * width)
            .map(|i| (i as f32 * 0.11).cos())
            .collect();
        let values: Vec<_> = (0..positions * heads * width)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let mut online = vec![0.0; heads * width];
        let mut materialized = vec![0.0; heads * width];
        for _ in 0..5 {
            causal_attention_into(&query, &keys, &values, heads, heads, &mut online).unwrap();
            materialized_attention(&query, &keys, &values, heads, width, &mut materialized);
        }
        let mut online_ms = Vec::new();
        let mut materialized_ms = Vec::new();
        for sample in 0..7 {
            for variant in 0..2 {
                let is_online = (sample + variant) % 2 == 0;
                let started = Instant::now();
                for _ in 0..10 {
                    if is_online {
                        causal_attention_into(
                            black_box(&query),
                            black_box(&keys),
                            black_box(&values),
                            heads,
                            heads,
                            black_box(&mut online),
                        )
                        .unwrap();
                    } else {
                        materialized_attention(
                            black_box(&query),
                            black_box(&keys),
                            black_box(&values),
                            heads,
                            width,
                            black_box(&mut materialized),
                        );
                    }
                }
                let elapsed = started.elapsed().as_secs_f64() * 1000.0 / 10.0;
                if is_online {
                    online_ms.push(elapsed);
                } else {
                    materialized_ms.push(elapsed);
                }
            }
        }
        let max_abs_error = online
            .iter()
            .zip(&materialized)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let report = serde_json::json!({"schema_version":1,"heads":heads,"head_width":width,"positions":positions,"samples":7,"iterations_per_sample":10,"online_ms":online_ms,"materialized_ms":materialized_ms,"max_abs_error":max_abs_error,"notes":["synthetic resident f32 KV; no allocation in online timed region","materialized baseline allocates score vectors as the pre-online production path did","alternating order; one Rayon configuration; CPU frequency/background load uncontrolled"]});
        let path = std::env::var("MOE_ATTENTION_REPORT").expect("set MOE_ATTENTION_REPORT");
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }

    #[test]
    #[ignore = "release-only microbenchmark; writes JSON to MOE_DENSE_REPORT"]
    fn benchmark_dense_kernel() {
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(!cfg!(debug_assertions), "run with --release");
        }
        let mut results = Vec::new();
        for (rows, columns) in [(2048, 2048), (8192, 2048)] {
            let (matrix, input) = data(rows, columns);
            let mut output = vec![0.0; rows];
            for _ in 0..5 {
                for (target, row) in output.iter_mut().zip(matrix.values.chunks_exact(columns)) {
                    *target = scalar_dot(black_box(row), black_box(&input));
                }
                matrix
                    .multiply_loaded_into(black_box(&input), black_box(&mut output))
                    .unwrap();
            }
            let mut scalar_ms = Vec::new();
            let mut blocked_ms = Vec::new();
            for sample in 0..7 {
                for variant in 0..2 {
                    let use_scalar = (sample + variant) % 2 == 0;
                    let started = Instant::now();
                    for _ in 0..10 {
                        if use_scalar {
                            for (target, row) in
                                output.iter_mut().zip(matrix.values.chunks_exact(columns))
                            {
                                *target = scalar_dot(black_box(row), black_box(&input));
                            }
                        } else {
                            matrix
                                .multiply_loaded_into(black_box(&input), black_box(&mut output))
                                .unwrap();
                        }
                        black_box(&output);
                    }
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0 / 10.0;
                    if use_scalar {
                        scalar_ms.push(elapsed);
                    } else {
                        blocked_ms.push(elapsed);
                    }
                }
            }
            results.push(serde_json::json!({"rows":rows,"columns":columns,
                "scalar_ms":scalar_ms,"blocked_ms":blocked_ms}));
        }
        let report = serde_json::json!({"schema_version":1,"samples":7,
            "iterations_per_sample":10,"warmup_per_variant":5,"threads":1,
            "results":results,"notes":["synthetic deterministic resident f32 buffers; no allocation in timing",
                "alternating measurement order; default release target; CPU frequency and background load uncontrolled"]});
        let path = std::env::var("MOE_DENSE_REPORT").expect("set MOE_DENSE_REPORT");
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}
