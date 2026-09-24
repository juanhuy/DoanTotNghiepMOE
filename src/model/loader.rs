//! Loader for our tiny reference format. This is not an OLMoE checkpoint adapter.
use super::*;
use crate::memory::ExpertStore;
use std::{io, path::Path};
use tokio::fs;

impl DecoderModel {
    pub async fn load(
        directory: impl AsRef<Path>,
        cache_bytes_per_layer: usize,
    ) -> io::Result<Self> {
        let directory = directory.as_ref();
        let checkpoint: Checkpoint =
            serde_json::from_slice(&fs::read(directory.join("model.json")).await?)
                .map_err(invalid_json)?;
        checkpoint.validate()?;
        let mut memories = Vec::new();
        let mut routers = Vec::new();
        for (id, layer) in checkpoint.layers.iter().enumerate() {
            let store = ExpertStore::new(directory.join(format!("layer-{id}")));
            // Validate every expert at startup, without retaining all experts in memory.
            for expert_id in 0..checkpoint.config.num_experts {
                let bytes = store.read(expert_id).await?;
                let weights: ExpertWeights =
                    serde_json::from_slice(&bytes).map_err(invalid_json)?;
                weights.validate(&checkpoint.config)?;
            }
            memories.push(TieredMemoryManager::with_ram_byte_capacity(
                store,
                cache_bytes_per_layer,
            ));
            routers.push(
                Router::from_weights(
                    checkpoint.config.num_experts,
                    checkpoint.config.top_k,
                    checkpoint.config.hidden_size,
                    layer.router.values.clone(),
                )
                .map_err(invalid)?,
            );
        }
        Ok(Self {
            checkpoint,
            memories,
            routers,
        })
    }
}

/// Creates a new directory only. Never overwrites an existing model.
/// Weights are deterministic pseudo-random numbers, not trained weights.
pub async fn write_demo(directory: impl AsRef<Path>) -> io::Result<()> {
    let directory = directory.as_ref();
    fs::create_dir(directory).await?;
    let config = ModelConfig::default();
    let mut seed = 42u64;
    let mut matrix = |rows: usize, columns: usize| {
        let values = (0..rows * columns)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let unit = (seed >> 32) as u32 as f32 / u32::MAX as f32;
                (2.0 * unit - 1.0) / (columns as f32).sqrt()
            })
            .collect();
        Matrix {
            rows,
            columns,
            values,
        }
    };
    let dim = config.hidden_size;
    let embedding = matrix(config.vocab_size, dim);
    let mut layers = Vec::new();
    for id in 0..config.num_layers {
        layers.push(LayerWeights {
            attention_norm: vec![1.; dim],
            ffn_norm: vec![1.; dim],
            attention: AttentionWeights {
                query: matrix(dim, dim),
                key: matrix(dim, dim),
                value: matrix(dim, dim),
                output: matrix(dim, dim),
            },
            router: matrix(config.num_experts as usize, dim),
        });
        let store = ExpertStore::new(directory.join(format!("layer-{id}")));
        for expert_id in 0..config.num_experts {
            let expert = ExpertWeights {
                gate: matrix(config.intermediate_size, dim),
                up: matrix(config.intermediate_size, dim),
                down: matrix(dim, config.intermediate_size),
            };
            store
                .write(
                    expert_id,
                    &serde_json::to_vec(&expert).map_err(invalid_json)?,
                )
                .await?;
        }
    }
    let checkpoint = Checkpoint {
        format: "tiny-moe-f32-v1".into(),
        embedding,
        layers,
        final_norm: vec![1.; dim],
        lm_head: matrix(config.vocab_size, dim),
        config,
    };
    fs::write(
        directory.join("model.json"),
        serde_json::to_vec_pretty(&checkpoint).map_err(invalid_json)?,
    )
    .await
}
