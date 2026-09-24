//! Small deterministic acceptance suite. Substring checks are a smoke signal,
//! not a standardized model-quality score.
use moe_tier_engine::{engine::ChatMessage, InferenceEngine};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Instant;

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
    category: String,
    max_tokens: usize,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    contains_all: Vec<String>,
    #[serde(default)]
    exact: Option<String>,
}

fn normalized(value: &str) -> String {
    value.trim().to_lowercase()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=2).contains(&args.len()) {
        return Err(
            "usage: evaluate_quality MODEL_DIRECTORY [SUITE=benchmarks/quality-v1.json]".into(),
        );
    }
    let suite_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("benchmarks/quality-v1.json");
    let suite: Suite = serde_json::from_slice(&std::fs::read(suite_path)?)?;
    if suite.version != 1 || suite.cases.is_empty() {
        return Err("expected nonempty quality suite version 1".into());
    }
    let mut ids = std::collections::HashSet::new();
    if suite.cases.iter().any(|case| {
        case.id.is_empty()
            || case.category.is_empty()
            || case.max_tokens == 0
            || case.messages.is_empty()
            || (case.exact.is_none() && case.contains_all.is_empty())
            || !ids.insert(&case.id)
    }) {
        return Err("invalid or duplicate quality case".into());
    }
    let load_started = Instant::now();
    let engine =
        InferenceEngine::load_with_limits(&args[0], 128 * 1024 * 1024, 512, 3 * 1024 * 1024 * 1024)
            .await?;
    if engine.model_name() != "olmoe" {
        return Err("quality suite requires OLMoE".into());
    }
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let mut results = Vec::new();
    let mut passed = 0usize;
    for case in &suite.cases {
        eprintln!("quality case {}", case.id);
        let response = engine.chat(case.messages.clone(), case.max_tokens).await?;
        let actual = normalized(&response.text);
        let exact_pass = case
            .exact
            .as_ref()
            .is_none_or(|expected| actual == normalized(expected));
        let contains_pass = case
            .contains_all
            .iter()
            .all(|expected| actual.contains(&normalized(expected)));
        let pass = exact_pass && contains_pass;
        passed += usize::from(pass);
        results.push(json!({"id":case.id,"category":case.category,"pass":pass,
            "expected_exact":case.exact,"expected_contains_all":case.contains_all,
            "response":response}));
    }
    let total = suite.cases.len();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version":1,"suite_path":suite_path,"model_directory":args[0],
            "load_ms":load_ms,"passed":passed,"total":total,
            "pass_rate":passed as f64 / total as f64,"results":results,
            "notes":["heuristic exact/substring acceptance checks; not a standardized quality benchmark",
                "greedy decoding; cases run once in fixed order with persistent expert cache and fresh KV"]
        }))?
    );
    Ok(())
}
