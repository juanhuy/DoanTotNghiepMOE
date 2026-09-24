use moe_tier_engine::{engine::ChatMessage, InferenceEngine};
use serde::Serialize;
use std::time::Instant;

const PROMPTS: [&str; 3] = [
    "What is 2 + 2?",
    "Write a short Python function that adds two integers.",
    "Hãy trả lời ngắn gọn: thủ đô của Việt Nam là gì?",
];

#[derive(Serialize)]
struct WorkloadRun {
    round: usize,
    prompt_index: usize,
    prompt: &'static str,
    response: moe_tier_engine::InferenceResponse,
}

#[derive(Serialize)]
struct WorkloadReport {
    schema_version: u32,
    model_directory: String,
    max_tokens: usize,
    rounds: usize,
    cache_bytes_per_layer: usize,
    load_ms: f64,
    runs: Vec<WorkloadRun>,
    notes: Vec<&'static str>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=4).contains(&args.len()) {
        return Err(
            "usage: benchmark_workload MODEL_DIRECTORY [MAX_TOKENS] [ROUNDS] [CACHE_MIB_PER_LAYER]"
                .into(),
        );
    }
    let max_tokens = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(8);
    let rounds: usize = args
        .get(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(2);
    let cache_mib: usize = args
        .get(3)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(128);
    if rounds == 0 || max_tokens == 0 {
        return Err("MAX_TOKENS and ROUNDS must be positive".into());
    }
    let cache_bytes_per_layer = cache_mib
        .checked_mul(1024 * 1024)
        .ok_or("cache byte size overflow")?;
    let load_started = Instant::now();
    let engine = InferenceEngine::load_with_limits(
        &args[0],
        cache_bytes_per_layer,
        512,
        3 * 1024 * 1024 * 1024,
    )
    .await?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    if engine.model_name() != "olmoe" {
        return Err("benchmark_workload requires an OLMoE checkpoint".into());
    }
    let mut runs = Vec::with_capacity(rounds * PROMPTS.len());
    for round in 0..rounds {
        for (prompt_index, prompt) in PROMPTS.iter().enumerate() {
            let response = engine
                .chat(
                    vec![ChatMessage {
                        role: "user".into(),
                        content: (*prompt).into(),
                    }],
                    max_tokens,
                )
                .await?;
            runs.push(WorkloadRun {
                round,
                prompt_index,
                prompt,
                response,
            });
        }
    }
    let report = WorkloadReport {
        schema_version: 1,
        model_directory: args[0].clone(),
        max_tokens,
        rounds,
        cache_bytes_per_layer,
        load_ms,
        runs,
        notes: vec![
            "prompts run in fixed order in every round",
            "heat history and expert cache persist across every run",
            "the operating-system page cache is not cleared or measured",
        ],
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
