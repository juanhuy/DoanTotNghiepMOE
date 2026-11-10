pub mod config;
pub mod loader;
pub use config::ModelConfig;

use crate::{
    attention::{AttentionWeights, KvCache},
    backend::{finite, invalid, rms_norm, Matrix},
    expert::ExpertWeights,
    memory::TieredMemoryManager,
    router::Router,
};
use serde::{Deserialize, Serialize};
use std::io;

#[derive(Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    pub attention_norm: Vec<f32>,
    pub attention: AttentionWeights,
    pub ffn_norm: Vec<f32>,
    pub router: Matrix,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub format: String,
    pub config: ModelConfig,
    pub embedding: Matrix,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Matrix,
}

pub struct DecoderModel {
    checkpoint: Checkpoint,
    routers: Vec<Router>,
    // One namespace and byte budget per layer: expert IDs cannot collide across layers.
    memories: Vec<TieredMemoryManager>,
}

pub(crate) struct DecodeState {
    caches: Vec<KvCache>,
    position: usize,
}

impl DecoderModel {
    pub fn config(&self) -> &ModelConfig {
        &self.checkpoint.config
    }

    pub(crate) fn new_state(&self) -> DecodeState {
        DecodeState {
            caches: (0..self.config().num_layers)
                .map(|_| KvCache::default())
                .collect(),
            position: 0,
        }
    }

    pub(crate) async fn forward_token(
        &self,
        token: usize,
        state: &mut DecodeState,
    ) -> io::Result<Vec<f32>> {
        let config = self.config();
        if token >= config.vocab_size || state.position >= config.max_seq_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "token or context length out of range",
            ));
        }
        let dim = config.hidden_size;
        let mut hidden = self.checkpoint.embedding.values[token * dim..(token + 1) * dim].to_vec();
        for (layer_id, layer) in self.checkpoint.layers.iter().enumerate() {
            let normalized = rms_norm(&hidden, &layer.attention_norm, config.rms_norm_eps);
            let attention =
                layer
                    .attention
                    .forward(&normalized, &mut state.caches[layer_id], config)?;
            for (h, a) in hidden.iter_mut().zip(attention) {
                *h += a;
            }
            let normalized = rms_norm(&hidden, &layer.ffn_norm, config.rms_norm_eps);
            let decision = self.routers[layer_id]
                .route_hidden(&normalized)
                .map_err(invalid)?;
            for (id, weight) in decision.experts {
                let (_, bytes) = self.memories[layer_id].get(id).await?;
                let expert: ExpertWeights = serde_json::from_slice(&bytes).map_err(invalid_json)?;
                expert.validate(config)?;
                let output = expert.forward(&normalized)?;
                for (h, x) in hidden.iter_mut().zip(output) {
                    *h += weight * x;
                }
            }
            finite(&hidden)?;
        }
        let normalized = rms_norm(&hidden, &self.checkpoint.final_norm, config.rms_norm_eps);
        let logits = self.checkpoint.lm_head.multiply(&normalized)?;
        state.position += 1;
        Ok(logits)
    }
}

fn invalid_json(error: serde_json::Error) -> io::Error {
    invalid(error.to_string())
}

impl Checkpoint {
    fn validate(&self) -> io::Result<()> {
        self.config.validate()?;
        if self.format != "tiny-moe-f32-v1" || self.layers.len() != self.config.num_layers {
            return Err(invalid(
                "unsupported checkpoint format or incorrect layer count",
            ));
        }
        let dim = self.config.hidden_size;
        self.embedding.validate(self.config.vocab_size, dim)?;
        self.lm_head.validate(self.config.vocab_size, dim)?;
        let norm = |weights: &[f32]| -> io::Result<()> {
            if weights.len() != dim {
                return Err(invalid("invalid normalization weights"));
            }
            finite(weights)
        };
        norm(&self.final_norm)?;
        for layer in &self.layers {
            norm(&layer.attention_norm)?;
            norm(&layer.ffn_norm)?;
            layer.attention.validate(dim)?;
            layer
                .router
                .validate(self.config.num_experts as usize, dim)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer;

    #[tokio::test]
    async fn cached_and_uncached_generation_match_and_requests_are_isolated() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model");
        loader::write_demo(&path).await.unwrap();
        let cached = DecoderModel::load(&path, 1_000_000).await.unwrap();
        let uncached = DecoderModel::load(&path, 0).await.unwrap();
        let a = cached.generate("Xin chào", 8).await.unwrap();
        let b = uncached.generate("Xin chào", 8).await.unwrap();
        assert_eq!(a.token_ids, b.token_ids);
        assert!(!a.token_ids.is_empty());
        let (first, second) =
            tokio::join!(cached.generate("Xin chào", 8), cached.generate("Other", 8));
        assert_eq!(first.unwrap().token_ids, a.token_ids);
        assert_eq!(
            second.unwrap().token_ids,
            uncached.generate("Other", 8).await.unwrap().token_ids
        );
        for memory in &cached.memories {
            assert!(memory.stats().await.ram_hits > 0);
            assert!(memory.cached_ram_bytes().await <= 1_000_000);
        }
        assert!(cached.generate("x", 0).await.is_err());
        assert!(cached.generate(&"x".repeat(256), 1).await.is_err());
        assert!(cached.generate("x", usize::MAX).await.is_err());
        assert!(loader::write_demo(&path).await.is_err());
    }

    #[tokio::test]
    async fn logits_match_manual_residual_only_model_and_eos_stops() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model");
        loader::write_demo(&path).await.unwrap();
        let mut model = DecoderModel::load(&path, 0).await.unwrap();
        // Zero both residual branches; hidden remains the all-ones embedding.
        model.checkpoint.embedding.values.fill(1.0);
        for layer in &mut model.checkpoint.layers {
            layer.attention.output.values.fill(0.0);
        }
        for id in 0..model.config().num_layers {
            let store = crate::ExpertStore::new(path.join(format!("layer-{id}")));
            for expert in 0..model.config().num_experts {
                let mut weights: ExpertWeights =
                    serde_json::from_slice(&store.read(expert).await.unwrap()).unwrap();
                weights.down.values.fill(0.0);
                store
                    .write(expert, &serde_json::to_vec(&weights).unwrap())
                    .await
                    .unwrap();
            }
        }
        model.checkpoint.lm_head.values.fill(0.0);
        let dim = model.config().hidden_size;
        model.checkpoint.lm_head.values[tokenizer::EOS * dim..(tokenizer::EOS + 1) * dim].fill(1.0);
        let logits = model
            .forward_token(tokenizer::BOS, &mut model.new_state())
            .await
            .unwrap();
        let expected = dim as f32 / (1.0 + model.config().rms_norm_eps).sqrt();
        assert!((logits[tokenizer::EOS] - expected).abs() < 1e-4);
        assert_eq!(logits[0], 0.0);
        let output = model.generate("hello", 8).await.unwrap();
        assert_eq!(output.token_ids, vec![tokenizer::EOS]);
        assert_eq!(output.finish_reason, "stop");
        assert_eq!(output.text, "");
        model.checkpoint.lm_head.values.fill(0.0);
        let output = model.generate("hello", 3).await.unwrap();
        assert_eq!(output.token_ids, vec![0, 0, 0]);
        assert_eq!(output.finish_reason, "length");
    }

    #[tokio::test]
    async fn missing_experts_and_invalid_checkpoint_fail_at_load() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model");
        loader::write_demo(&path).await.unwrap();
        let file = path.join("model.json");
        let original = tokio::fs::read(&file).await.unwrap();
        let mut checkpoint: Checkpoint = serde_json::from_slice(&original).unwrap();
        checkpoint.config.num_heads = 0;
        tokio::fs::write(&file, serde_json::to_vec(&checkpoint).unwrap())
            .await
            .unwrap();
        assert!(DecoderModel::load(&path, 1024).await.is_err());
        tokio::fs::write(&file, &original).await.unwrap();
        tokio::fs::remove_file(path.join("layer-1/expert-0.bin"))
            .await
            .unwrap();
        assert!(DecoderModel::load(&path, 1024).await.is_err());
    }
}
