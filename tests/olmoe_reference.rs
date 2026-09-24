use moe_tier_engine::{engine::ChatMessage, olmoe::OlmoeModel, InferenceEngine};
use serde::Deserialize;
use std::path::PathBuf;
#[derive(Deserialize)]
struct Reference {
    ids: Vec<usize>,
    logits: Vec<Vec<f32>>,
}
fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/olmoe")
        .join(name)
}

#[test]
fn incremental_quantized_logits_match_transformers_full_causal_forward() {
    for name in ["unnormalized-mha", "normalized-gqa"] {
        let dir = path(name);
        let reference: Reference =
            serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
        for cache in [0, 900, 100_000] {
            let mut model = OlmoeModel::load(&dir, cache, 32, 10_000_000).unwrap();
            for _ in 0..2 {
                let actual = model.logits(&reference.ids).unwrap();
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
            let prompt = model
                .render_chat(&[ChatMessage {
                    role: "user".into(),
                    content: "token3 token4".into(),
                }])
                .unwrap();
            assert_eq!(prompt, "[BOS]token3 token4 ");
            let a = model.generate(&prompt, 3).unwrap();
            let b = model.generate(&prompt, 3).unwrap();
            assert_eq!(a.token_ids, b.token_ids);
            assert_eq!(a.prompt_tokens, 3);
            let metrics = a.metrics.as_ref().unwrap();
            let warm_metrics = b.metrics.as_ref().unwrap();
            assert!(metrics.expert_cache_hits + metrics.expert_cache_misses > 0);
            assert!(warm_metrics.expert_cache_misses <= metrics.expert_cache_misses);
            assert!(metrics.total_ms >= metrics.prefill_ms);
            assert!(metrics.expert_cache_bytes <= cache * 2);
            assert!(metrics.kv_cache_bytes > 0);
            assert!(model.generate(&prompt, 100).is_err());
            assert!(model.generate(&prompt, 0).is_err());
        }
        assert!(OlmoeModel::load(&dir, 0, 32, 1).is_err());
        assert!(OlmoeModel::load(&dir, 0, 65, 10_000_000).is_err());
    }
}

#[tokio::test]
async fn engine_selects_olmoe_and_rejects_invalid_messages() {
    let engine =
        InferenceEngine::load_with_limits(path("unnormalized-mha"), 10_000, 32, 10_000_000)
            .await
            .unwrap();
    assert_eq!(engine.model_name(), "olmoe");
    let output = engine
        .chat(
            vec![ChatMessage {
                role: "user".into(),
                content: "token3".into(),
            }],
            2,
        )
        .await
        .unwrap();
    assert!(!output.token_ids.is_empty());
    assert!(output.metrics.is_some());
    assert!(engine.chat(vec![], 2).await.is_err());
    assert!(engine
        .chat(
            vec![ChatMessage {
                role: "invalid".into(),
                content: "a".into()
            }],
            2
        )
        .await
        .is_err());
    let too_many = (0..129)
        .map(|_| ChatMessage {
            role: "user".into(),
            content: "x".into(),
        })
        .collect();
    let error = engine.chat(too_many, 2).await.unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}

#[tokio::test]
async fn stream_matches_complete_generation_and_validates_before_start() {
    use moe_tier_engine::engine::ChatEvent;
    let engine =
        InferenceEngine::load_with_limits(path("unnormalized-mha"), 10_000, 64, 10_000_000)
            .await
            .unwrap();
    let messages = || {
        vec![ChatMessage {
            role: "user".into(),
            content: "token3 token4".into(),
        }]
    };
    for _ in 0..3 {
        let expected = engine.chat(messages(), 8).await.unwrap();
        let mut stream = engine.chat_stream(messages(), 8).await.unwrap();
        let mut text = String::new();
        let mut finished = None;
        while let Some(event) = stream.recv().await {
            match event.unwrap() {
                ChatEvent::Delta(delta) => text.push_str(&delta),
                ChatEvent::Finished(result) => finished = Some(result),
            }
        }
        let result = finished.unwrap();
        assert_eq!(text, expected.text);
        assert_eq!(result.token_ids, expected.token_ids);
        assert_eq!(result.finish_reason, expected.finish_reason);
    }
    for (messages, limit) in [(messages(), 0), (messages(), 100), (vec![], 2)] {
        let error = engine.chat_stream(messages, limit).await.err().unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }
    // Disconnect immediately, including during prefill, and wait for permit release.
    for _ in 0..3 {
        let stream = engine.chat_stream(messages(), 32).await.unwrap();
        drop(stream);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                match engine.chat(messages(), 2).await {
                    Ok(_) => break,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        tokio::task::yield_now().await
                    }
                    Err(e) => panic!("{e}"),
                }
            }
        })
        .await
        .unwrap();
    }
}
