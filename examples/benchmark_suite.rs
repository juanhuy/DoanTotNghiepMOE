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
    if !(1..=4).contains(&args.len()) {
        return Err(
            "usage: benchmark_suite MODEL_DIRECTORY [ROUNDS=3] [SUITE=benchmarks/suite-v1.json] [CASE_ID]"
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
        .unwrap_or("benchmarks/suite-v1.json");
    let suite: Suite = serde_json::from_slice(&std::fs::read(suite_file)?)?;
    let selected_case = args.get(3).map(String::as_str);
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
    let engine =
        InferenceEngine::load_with_limits(&args[0], 128 * 1024 * 1024, 512, 3 * 1024 * 1024 * 1024)
            .await?;
    if engine.model_name() != "olmoe" {
        return Err("suite requires OLMoE".into());
    }
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
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
        "settings":{"context_limit":512,"cache_bytes_per_layer":134217728,"dense_budget_bytes":3221225472u64},
        "environment":{"rustc":rustc,"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"logical_cpus":std::thread::available_parallelism()?.get(),"debug_assertions":cfg!(debug_assertions)},
        "notes":["fixed case order; only first request has empty expert cache; fresh KV per request", "OS page cache and CPU frequency uncontrolled", "VmHWM is cumulative process peak RSS on Linux, not per-request allocation or system page cache", "evaluate answers manually; length finish may truncate; no quality score is inferred"]});
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
