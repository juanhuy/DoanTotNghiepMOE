//! Qwen3-MoE parity: tiny random checkpoints from `tools/make_qwen3moe_reference.py`
//! (Transformers eager forward with INT8-dequantized experts) against the engine.
use moe_tier_engine::{engine::ChatMessage, olmoe::OlmoeModel};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Reference {
    ids: Vec<usize>,
    logits: Vec<Vec<f32>>,
}

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen3moe")
        .join(name)
}

#[test]
fn qwen3moe_quantized_logits_match_transformers() {
    for name in ["gqa-headdim", "many-experts"] {
        let dir = path(name);
        let reference: Reference =
            serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
        for cache in [0, 900, 100_000] {
            let mut model = OlmoeModel::load(&dir, cache, 32, 10_000_000).unwrap();
            for _ in 0..2 {
                let actual = model.logits(&reference.ids).unwrap();
                assert_eq!(actual.len(), reference.logits.len());
                let error = actual
                    .iter()
                    .flatten()
                    .zip(reference.logits.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    error < 2e-5,
                    "{name}, cache={cache}, max logit error={error}"
                );
            }
            // Generation path (prefill + decode) stays deterministic.
            let a = model.generate("token3 token4 token5", 3).unwrap();
            let b = model.generate("token3 token4 token5", 3).unwrap();
            assert_eq!(a.token_ids, b.token_ids);
            let status = model.expert_status();
            let gpu_computes = b.metrics.as_ref().map_or(0, |m| m.expert_gpu_computes);
            eprintln!(
                "{name} cache={cache}: gpu_dense={} tier={} gpu_compute={} expert_gpu_computes={gpu_computes}",
                status.gpu.dense_offloaded,
                status.gpu_expert_tier_enabled,
                status.gpu_expert_compute_enabled
            );
            if status.gpu_expert_compute_enabled {
                assert!(gpu_computes > 0, "GPU expert compute enabled but unused");
            }
        }
    }
}

/// Qwen3 chat template (namespace, reversed loop, str methods, `<think>` handling) rendered by
/// minijinja must equal `transformers` `apply_chat_template(..., enable_thinking=False)`.
#[test]
fn qwen3_chat_template_matches_transformers_rendering() {
    let dir = path("gqa-headdim");
    let model = OlmoeModel::load(&dir, 0, 32, 10_000_000).unwrap();
    let messages: Vec<ChatMessage> =
        serde_json::from_slice(&std::fs::read(dir.join("chat_messages.json")).unwrap()).unwrap();
    let expected = std::fs::read_to_string(dir.join("chat_expected_thinking_off.txt")).unwrap();
    assert_eq!(model.render_chat(&messages).unwrap(), expected);
}
