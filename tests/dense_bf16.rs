//! MOE_DENSE_STORAGE=bf16 on a checkpoint whose dense tensors are BF16 on disk: same logits as
//! F32 storage and as Transformers, with half the resident dense-matrix bytes. Own test binary
//! because it sets process environment variables.
use moe_tier_engine::olmoe::OlmoeModel;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Reference {
    ids: Vec<usize>,
    logits: Vec<Vec<f32>>,
}

#[test]
fn bf16_dense_storage_matches_f32_storage_and_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3moe/bf16-dense");
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
    let mut results = Vec::new();
    for storage in ["f32", "bf16"] {
        std::env::set_var("MOE_DENSE_STORAGE", storage);
        let mut model = OlmoeModel::load(&dir, 100_000, 32, 10_000_000).unwrap();
        let status = model.expert_status();
        assert_eq!(status.dense_storage, storage);
        let logits = model.logits(&reference.ids).unwrap();
        let error = logits
            .iter()
            .flatten()
            .zip(reference.logits.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(error < 2e-5, "{storage}: max logit error={error}");
        results.push((status.dense_matrix_bytes, logits));
    }
    std::env::remove_var("MOE_DENSE_STORAGE");
    assert_eq!(results[0].0, 2 * results[1].0);
    // CPU paths are bitwise identical; GPU dense uploads the same widened F32 values.
    assert_eq!(results[0].1, results[1].1);
}
