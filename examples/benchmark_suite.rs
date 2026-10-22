//! Versioned multi-turn workload. Run sequentially; do not compare different suites.
use moe_tier_engine::{engine::ChatMessage, InferenceEngine};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{path::Path, time::Instant};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    version: u32,
    cases: Vec<Case>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    max_tokens: usize,
    messages: Vec<ChatMessage>,
}

fn peak_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")
                .and_then(|s| s.split_whitespace().next()?.parse().ok())
        })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=7).contains(&args.len()) {
        return Err(
            "usage: benchmark_suite MODEL_DIRECTORY [ROUNDS=3] [SUITE=benchmarks/suites/suite-v1.json] [CASE_ID_OR_*] [CACHE_MIB=128] [CONTEXT=512] [GPU=auto|cpu]"
                .into(),
        );
    }
    let rounds = args
        .get(1)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(3);
    if rounds == 0 {
        return Err("ROUNDS must be positive".into());
    }
    let suite_file = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("benchmarks/suites/suite-v1.json");
    let suite: Suite = serde_json::from_slice(&std::fs::read(suite_file)?)?;
    let selected_case = args.get(3).map(String::as_str).filter(|s| *s != "*");
    let cache_mib: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(128);
    let cache_bytes = cache_mib
        .checked_mul(1024 * 1024)
        .ok_or("cache byte size overflow")?;
    let context_limit: usize = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(512);
    let gpu_mode = args.get(6).map(String::as_str).unwrap_or("auto");
    if !matches!(gpu_mode, "auto" | "cpu") {
        return Err("GPU must be auto or cpu".into());
    }
    if suite.version != 1 || suite.cases.is_empty() {
        return Err("expected nonempty version 1 suite".into());
    }
    let mut ids = std::collections::HashSet::new();
    if suite.cases.iter().any(|case| {
        case.id.is_empty()
            || !ids.insert(&case.id)
            || case.max_tokens == 0
            || case.messages.is_empty()
    }) {
        return Err("invalid or duplicate case".into());
    }
    if selected_case.is_some_and(|selected| !suite.cases.iter().any(|case| case.id == selected)) {
        return Err("CASE_ID is not present in the suite".into());
    }
    let started = Instant::now();
    let engine = InferenceEngine::load_with_limits(
        &args[0],
        cache_bytes,
        context_limit,
        3 * 1024 * 1024 * 1024,
    )
    .await?;
    if engine.model_name() != "olmoe" {
        return Err("suite requires OLMoE".into());
    }
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    if gpu_mode == "cpu" {
        engine.set_gpu_enabled(false)?;
    }
    let gpu_status_start = engine.expert_status().map(|s| s.gpu);
    let mut runs = Vec::new();
    for round in 0..rounds {
        for case in suite
            .cases
            .iter()
            .filter(|case| selected_case.is_none_or(|selected| case.id == selected))
        {
            eprintln!("round {} / {}, case {}", round + 1, rounds, case.id);
            let response = engine.chat(case.messages.clone(), case.max_tokens).await?;
            runs.push(json!({"round":round,"case_id":case.id,"response":response,"process_peak_rss_kib":peak_rss_kib()}));
        }
    }
    let rustc = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(Path::new(&args[0]).join("config.json"))?)?;
    let report = json!({"schema_version":1,"suite":suite,"selected_case":selected_case,"rounds":rounds,"runs":runs,
        "model_directory":args[0],"model_config":config,"load_ms":load_ms,
        "gpu_mode":gpu_mode,"gpu_status_start":gpu_status_start,"gpu_status_end":engine.expert_status().map(|s|s.gpu),
        "settings":{"context_limit":context_limit,"cache_bytes_per_layer":cache_bytes,"dense_budget_bytes":3221225472u64},
        "environment":{"rustc":rustc,"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"logical_cpus":std::thread::available_parallelism()?.get(),"debug_assertions":cfg!(debug_assertions)},
        "notes":["cpu mode disables GPU computation after load; load may initialize GPU and VRAM allocations may remain", "GPU status is logical engine telemetry, not physical VRAM measurement", "fixed case order; only first request has empty expert cache; fresh KV per request", "OS page cache and CPU frequency uncontrolled", "VmHWM is cumulative process peak RSS on Linux, not per-request allocation or system page cache", "evaluate answers manually; length finish may truncate; no quality score is inferred"]});
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
