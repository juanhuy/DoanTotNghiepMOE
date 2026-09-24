use moe_tier_engine::{engine::ChatMessage, InferenceEngine};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct Run {
    index: usize,
    cache_state: &'static str,
    response: moe_tier_engine::InferenceResponse,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    model_directory: String,
    prompt: String,
    max_tokens: usize,
    cache_bytes_per_layer: usize,
    context_limit: usize,
    dense_budget_bytes: usize,
    logical_cpus: usize,
    load_ms: f64,
    runs: Vec<Run>,
    notes: Vec<&'static str>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=5).contains(&args.len()) {
        return Err(
            "usage: benchmark_olmoe MODEL_DIRECTORY PROMPT [MAX_TOKENS] [RUNS] [CACHE_MIB_PER_LAYER]"
                .into(),
        );
    }
    let max_tokens = args
        .get(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(16);
    let run_count: usize = args
        .get(3)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(2);
    if run_count == 0 {
        return Err("RUNS must be positive".into());
    }
    let cache_mib: usize = args
        .get(4)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(128);
    let cache_bytes_per_layer = cache_mib
        .checked_mul(1024 * 1024)
        .ok_or("cache byte size overflow")?;
    let context_limit = 512;
    let dense_budget_bytes = 3 * 1024 * 1024 * 1024;
    let load_started = Instant::now();
    let engine = InferenceEngine::load_with_limits(
        &args[0],
        cache_bytes_per_layer,
        context_limit,
        dense_budget_bytes,
    )
    .await?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    if engine.model_name() != "olmoe" {
        return Err("benchmark_olmoe requires an OLMoE checkpoint".into());
    }
    let mut runs = Vec::with_capacity(run_count);
    for index in 0..run_count {
        let response = engine
            .chat(
                vec![ChatMessage {
                    role: "user".into(),
                    content: args[1].clone(),
                }],
                max_tokens,
            )
            .await?;
        runs.push(Run {
            index,
            cache_state: if index == 0 {
                "engine-cold"
            } else {
                "engine-warm"
            },
            response,
        });
    }
    let report = Report {
        schema_version: 1,
        model_directory: args[0].clone(),
        prompt: args[1].clone(),
        max_tokens,
        cache_bytes_per_layer,
        context_limit,
        dense_budget_bytes,
        logical_cpus: std::thread::available_parallelism()?.get(),
        load_ms,
        runs,
        notes: vec![
            "engine-cold means the in-process expert cache starts empty",
            "the operating-system page cache is not cleared or measured",
            "decode_tokens_per_second counts model steps after the first sampled token",
        ],
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
