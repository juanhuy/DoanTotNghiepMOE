//! Opt-in INT8 multi-token prefill kernel; benchmark and fixtures remain test-only.
use super::*;

pub(crate) struct BatchScratch<const B: usize> {
    activated_rows: Vec<[f32; B]>,
    activated: [Vec<f32>; B],
    output_rows: Vec<[f32; B]>,
    pub(crate) outputs: [Vec<f32>; B],
}
impl<const B: usize> BatchScratch<B> {
    pub(crate) fn new(hidden: usize, intermediate: usize) -> Self {
        Self {
            activated_rows: vec![[0.0; B]; intermediate],
            activated: std::array::from_fn(|_| vec![0.0; intermediate]),
            output_rows: vec![[0.0; B]; hidden],
            outputs: std::array::from_fn(|_| vec![0.0; hidden]),
        }
    }
    pub(crate) fn allocated_bytes(&self) -> usize {
        (self.activated_rows.capacity() + self.output_rows.capacity()) * B * 4
            + self
                .activated
                .iter()
                .chain(&self.outputs)
                .map(|v| v.capacity() * 4)
                .sum::<usize>()
    }
}

fn dot_batch<const B: usize>(row: &[u8], inputs: &[&[f32]; B]) -> [f32; B] {
    assert!(inputs.iter().all(|x| x.len() == row.len()));
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: CPU features checked, every input has row.len() elements.
        // The helper reads only complete 32/8-element chunks, then scalar tails.
        // Inputs/weights are immutable for the call; accumulators are thread-local.
        return unsafe { dot_batch_avx2(row, inputs) };
    }
    std::array::from_fn(|t| dot_i8_f32(row, inputs[t]))
}

#[cfg(target_arch = "x86_64")]
// Explicit token/bank indices preserve the existing SIMD reduction order.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_batch_avx2<const B: usize>(row: &[u8], inputs: &[&[f32]; B]) -> [f32; B] {
    use std::arch::x86_64::*;
    let mut sums = [[_mm256_setzero_ps(); 4]; B];
    let full = row.len() / 32 * 32;
    for offset in (0..full).step_by(32) {
        for bank in 0..4 {
            let at = offset + bank * 8;
            let q8 =
                _mm_cvtsi64_si128(std::ptr::read_unaligned(row.as_ptr().add(at) as *const i64));
            let q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            // One INT8->FP32 conversion shared by all tokens in this tile.
            for t in 0..B {
                let x = _mm256_loadu_ps(inputs[t].as_ptr().add(at));
                sums[t][bank] = _mm256_fmadd_ps(q, x, sums[t][bank]);
            }
        }
    }
    std::array::from_fn(|t| {
        let s = &sums[t];
        let mut total = _mm256_add_ps(_mm256_add_ps(s[0], s[1]), _mm256_add_ps(s[2], s[3]));
        let tail8 = full + (row.len() - full) / 8 * 8;
        for at in (full..tail8).step_by(8) {
            let q8 =
                _mm_cvtsi64_si128(std::ptr::read_unaligned(row.as_ptr().add(at) as *const i64));
            let q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            total = _mm256_fmadd_ps(q, _mm256_loadu_ps(inputs[t].as_ptr().add(at)), total);
        }
        let low = _mm256_castps256_ps128(total);
        let high = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(low, high);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);
        for at in tail8..row.len() {
            result += (row[at] as i8 as f32) * inputs[t][at];
        }
        result
    })
}

impl Int8Expert {
    pub(crate) fn forward_batch_prototype<const B: usize>(
        &self,
        inputs: [&[f32]; B],
        scratch: &mut BatchScratch<B>,
    ) -> io::Result<()> {
        if !matches!(B, 2 | 4) || self.quant_type != QuantType::Int8 {
            return Err(invalid("batch prototype requires INT8 and 2 or 4 tokens"));
        }
        if inputs.iter().any(|x| x.len() != self.hidden)
            || scratch.activated_rows.len() != self.intermediate
            || scratch.output_rows.len() != self.hidden
            || scratch
                .activated
                .iter()
                .any(|v| v.len() != self.intermediate)
            || scratch.outputs.iter().any(|v| v.len() != self.hidden)
        {
            return Err(invalid("batch prototype dimension mismatch"));
        }
        for input in inputs {
            finite(input)?;
        }
        let n = self.hidden * self.intermediate;
        scratch
            .activated_rows
            .par_iter_mut()
            .enumerate()
            .for_each(|(r, out)| {
                let a = r * self.hidden;
                let gates = dot_batch(&self.weights[a..a + self.hidden], &inputs);
                let ups = dot_batch(&self.weights[n + a..n + a + self.hidden], &inputs);
                for t in 0..B {
                    let g = gates[t] * self.scales[r];
                    let u = ups[t] * self.scales[self.intermediate + r];
                    out[t] = (g / (1.0 + (-g).exp())) * u;
                }
            });
        // Include this layout conversion in candidate timing; no hidden setup advantage.
        for t in 0..B {
            for r in 0..self.intermediate {
                scratch.activated[t][r] = scratch.activated_rows[r][t];
            }
        }
        let activated: [&[f32]; B] = std::array::from_fn(|t| scratch.activated[t].as_slice());
        scratch
            .output_rows
            .par_iter_mut()
            .enumerate()
            .for_each(|(r, out)| {
                let a = 2 * n + r * self.intermediate;
                let dots = dot_batch(&self.weights[a..a + self.intermediate], &activated);
                for t in 0..B {
                    out[t] = dots[t] * self.scales[2 * self.intermediate + r];
                }
            });
        for t in 0..B {
            for r in 0..self.hidden {
                scratch.outputs[t][r] = scratch.output_rows[r][t];
            }
            finite(&scratch.outputs[t])?;
        }
        Ok(())
    }
}

#[cfg(test)]
fn inputs<const B: usize>(hidden: usize, seed: u64) -> [Vec<f32>; B] {
    let mut state = seed;
    std::array::from_fn(|_| {
        (0..hidden)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    })
}

#[cfg(test)]
fn synthetic(hidden: usize, intermediate: usize) -> Int8Expert {
    Int8Expert {
        hidden,
        intermediate,
        weights: (0..3 * hidden * intermediate)
            .map(|i| (i * 73 + 129) as u8)
            .collect(),
        scales: vec![0.0001; 2 * intermediate + hidden],
        quant_type: QuantType::Int8,
    }
}

#[cfg(test)]
fn separate<const B: usize>(e: &Int8Expert, x: &[Vec<f32>; B], scratch: &mut [ExpertScratch; B]) {
    use std::hint::black_box;
    for (s, x) in scratch.iter_mut().zip(x) {
        black_box(e.forward_reuse(black_box(x), s).unwrap());
    }
}

#[cfg(test)]
fn parity<const B: usize>(
    e: &Int8Expert,
    x: &[Vec<f32>; B],
    base: &mut [ExpertScratch; B],
    batch: &mut BatchScratch<B>,
) -> f32 {
    separate(e, x, base);
    e.forward_batch_prototype(std::array::from_fn(|t| x[t].as_slice()), batch)
        .unwrap();
    let mut error = 0.0f32;
    for (reference, output) in base.iter().zip(&batch.outputs) {
        for (a, b) in reference.output.iter().zip(output) {
            assert!(a.is_finite() && b.is_finite());
            error = error.max((a - b).abs());
        }
    }
    assert!(error < 2e-5, "batch{B}: max_abs_error={error}");
    error
}

#[cfg(test)]
#[test]
fn batch_dot_matches_current_simd_for_signed_weights_and_tails() {
    for len in [1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 1024, 2048, 2051] {
        let row: Vec<_> = (0..len).map(|i| (i * 73 + 129) as u8).collect();
        let x = inputs::<4>(len, 73);
        let actual = dot_batch::<4>(&row, &std::array::from_fn(|t| x[t].as_slice()));
        for t in 0..4 {
            assert_eq!(actual[t].to_bits(), dot_i8_f32(&row, &x[t]).to_bits());
        }
    }
}

#[cfg(test)]
#[test]
fn full_batch2_batch4_match_production_expert_with_tail_dimensions() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        for (h, i) in [(7, 5), (33, 17), (64, 32), (2048, 1024)] {
            let e = synthetic(h, i);
            parity(
                &e,
                &inputs::<2>(h, 17),
                &mut std::array::from_fn(|_| ExpertScratch::new(h, i)),
                &mut BatchScratch::new(h, i),
            );
            parity(
                &e,
                &inputs::<4>(h, 991),
                &mut std::array::from_fn(|_| ExpertScratch::new(h, i)),
                &mut BatchScratch::new(h, i),
            );
        }
    });
}

#[cfg(test)]
#[test]
fn batch_prototype_rejects_invalid_shape_nonfinite_and_precision() {
    let mut e = synthetic(8, 4);
    let x = inputs::<2>(8, 17);
    let mut s = BatchScratch::new(8, 4);
    assert!(e
        .forward_batch_prototype([&x[0][..7], &x[1]], &mut s)
        .is_err());
    assert!(e
        .forward_batch_prototype([&[f32::NAN; 8], &x[1]], &mut s)
        .is_err());
    s.outputs[0].pop();
    assert!(e.forward_batch_prototype([&x[0], &x[1]], &mut s).is_err());
    e.quant_type = QuantType::Int4;
    assert!(e
        .forward_batch_prototype([&x[0], &x[1]], &mut BatchScratch::new(8, 4))
        .is_err());
}

#[cfg(test)]
fn measure<const B: usize>(
    e: &Int8Expert,
    layer: usize,
    seed: u64,
    parity_rows: &mut Vec<serde_json::Value>,
    samples: &mut Vec<serde_json::Value>,
) -> usize {
    use std::{hint::black_box, time::Instant};
    let x = inputs::<B>(e.hidden, seed);
    let mut base: [ExpertScratch; B] =
        std::array::from_fn(|_| ExpertScratch::new(e.hidden, e.intermediate));
    let mut batch = BatchScratch::new(e.hidden, e.intermediate);
    let bytes = batch.allocated_bytes()
        + base
            .iter()
            .map(|s| (s.activated.capacity() + s.output.capacity()) * 4)
            .sum::<usize>()
        + x.iter().map(|v| v.capacity() * 4).sum::<usize>();
    assert!(bytes <= 256 * 1024);
    let scope = format!("batch{B}");
    let error = parity(e, &x, &mut base, &mut batch);
    parity_rows
        .push(serde_json::json!({"layer":layer,"seed":seed,"scope":scope,"max_abs_error":error}));
    let mut run = |shared: bool| {
        if shared {
            e.forward_batch_prototype(std::array::from_fn(|t| x[t].as_slice()), &mut batch)
                .unwrap();
            black_box(&batch.outputs);
        } else {
            separate(e, &x, &mut base);
        }
    };
    for _ in 0..10 {
        run(false);
        run(true);
    }
    for pair in 0..9 {
        for position in 0..2 {
            let shared = (pair + position) % 2 == 1;
            let started = Instant::now();
            for _ in 0..40 {
                run(shared);
            }
            samples.push(serde_json::json!({
                "layer":layer,"seed":seed,"scope":scope,"pair":pair,"position":position,
                "variant":if shared {"shared_weights"} else {"separate_tokens"},
                "iterations":40,"elapsed_ns":started.elapsed().as_nanos() as u64
            }));
        }
    }
    bytes
}

#[cfg(test)]
#[test]
#[ignore = "bounded real-weight batched expert prototype; use --expert-batch-kernel"]
fn benchmark_expert_batch() {
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
    assert_eq!(config["hidden_size"].as_u64(), Some(2048));
    assert_eq!(config["intermediate_size"].as_u64(), Some(1024));
    let index = TensorIndex::open(&directory).unwrap();
    let experts: Vec<_> = [0, 7, 15]
        .into_iter()
        .map(|layer| {
            let e = Int8Expert::load(&index, layer, 0, 2048, 1024, 8 * 1024 * 1024).unwrap();
            assert_eq!(e.quant_type, QuantType::Int8);
            e
        })
        .collect();
    let payload_bytes: usize = experts.iter().map(Int8Expert::resident_bytes).sum();
    assert!(payload_bytes <= 256 * 1024 * 1024);
    let process_before = std::fs::read_to_string("/proc/self/status").unwrap();
    let io_before = std::fs::read_to_string("/proc/self/io").unwrap();
    let mut parity_rows = Vec::new();
    let mut samples = Vec::new();
    let mut scratch_bytes = 0;
    for (layer, e) in [0, 7, 15].into_iter().zip(&experts) {
        for seed in [17, 73, 991] {
            scratch_bytes =
                scratch_bytes.max(measure::<2>(e, layer, seed, &mut parity_rows, &mut samples));
            scratch_bytes =
                scratch_bytes.max(measure::<4>(e, layer, seed, &mut parity_rows, &mut samples));
        }
    }
    let process_after = std::fs::read_to_string("/proc/self/status").unwrap();
    let io_after = std::fs::read_to_string("/proc/self/io").unwrap();
    let report = serde_json::json!({
        "schema_version":1,"kind":"expert_batch_resident","protocol":"expert-batch-int8-v1",
        "testbed":testbed,"hidden":2048,"intermediate":1024,"threads":4,"avx2_fma":avx2_fma,
        "layers":[0,7,15],"expert_ids":[0],"seeds":[17,73,991],"scopes":["batch2","batch4"],
        "pairs_per_case":9,"iterations":40,"warmup_per_variant_scope":10,
        "payload_bytes":payload_bytes,"payload_budget_bytes":256*1024*1024,
        "scratch_peak_capacity_bytes":scratch_bytes,"scratch_budget_bytes":256*1024,
        "baseline_gate_up":"paired","baseline_dispatch":"serial_tokens_row_parallel","parity":parity_rows,"samples":samples,
        "process_before":process_before,"process_after":process_after,"io_before":io_before,"io_after":io_after,
        "notes":["real INT8 weights/scales; synthetic activations; no model forward",
                 "baseline serial token dispatch, production forward_reuse row-parallel Rayon4",
                 "candidate row-parallel4, weight conversion shared; layout copies timed",
                 "scheduling differs; speedup cannot be attributed solely to weight reuse"]
    });
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var("MOE_GATE_UP_REPORT").unwrap())
        .unwrap();
    serde_json::to_writer_pretty(file, &report).unwrap();
}
