//! Incremental causal multi-head attention with adjacent-pair RoPE.
use crate::{
    backend::{softmax, Matrix},
    model::ModelConfig,
};
use serde::{Deserialize, Serialize};
use std::io;

#[derive(Clone, Serialize, Deserialize)]
pub struct AttentionWeights {
    pub query: Matrix,
    pub key: Matrix,
    pub value: Matrix,
    pub output: Matrix,
}

#[derive(Default)]
pub(crate) struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

fn rope(vector: &mut [f32], head_dim: usize, position: usize, theta: f32) {
    for head in vector.chunks_exact_mut(head_dim) {
        for (i, pair) in head.chunks_exact_mut(2).enumerate() {
            let angle = position as f32 / theta.powf((2 * i) as f32 / head_dim as f32);
            let (sin, cos) = angle.sin_cos();
            let (x, y) = (pair[0], pair[1]);
            pair[0] = x * cos - y * sin;
            pair[1] = x * sin + y * cos;
        }
    }
}

impl AttentionWeights {
    pub(crate) fn validate(&self, dim: usize) -> io::Result<()> {
        for matrix in [&self.query, &self.key, &self.value, &self.output] {
            matrix.validate(dim, dim)?;
        }
        Ok(())
    }

    pub(crate) fn forward(
        &self,
        hidden: &[f32],
        cache: &mut KvCache,
        config: &ModelConfig,
    ) -> io::Result<Vec<f32>> {
        let head_dim = config.hidden_size / config.num_heads;
        let mut q = self.query.multiply(hidden)?;
        let mut k = self.key.multiply(hidden)?;
        let v = self.value.multiply(hidden)?;
        rope(&mut q, head_dim, cache.keys.len(), config.rope_theta);
        rope(&mut k, head_dim, cache.keys.len(), config.rope_theta);
        cache.keys.push(k);
        cache.values.push(v);
        let mut merged = vec![0.0; config.hidden_size];
        for head in 0..config.num_heads {
            let start = head * head_dim;
            let end = start + head_dim;
            let mut scores: Vec<f32> = cache
                .keys
                .iter()
                .map(|key| {
                    q[start..end]
                        .iter()
                        .zip(&key[start..end])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / (head_dim as f32).sqrt()
                })
                .collect();
            softmax(&mut scores)?;
            for (score, value) in scores.iter().zip(&cache.values) {
                for i in start..end {
                    merged[i] += score * value[i];
                }
            }
        }
        self.output.multiply(&merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn two_token_attention_matches_scalar_reference() {
        let identity = Matrix {
            rows: 2,
            columns: 2,
            values: vec![1., 0., 0., 1.],
        };
        let weights = AttentionWeights {
            query: identity.clone(),
            key: identity.clone(),
            value: identity.clone(),
            output: identity,
        };
        let config = ModelConfig {
            hidden_size: 2,
            num_heads: 1,
            ..Default::default()
        };
        let mut cache = KvCache::default();
        assert_eq!(
            weights.forward(&[1., 0.], &mut cache, &config).unwrap(),
            vec![1., 0.]
        );
        let actual = weights.forward(&[0., 1.], &mut cache, &config).unwrap();
        // q1=(-sin(1), cos(1)), k0=(1,0), k1=q1; values stay unrotated.
        let a = (-1f32.sin() / 2f32.sqrt()).exp();
        let b = (1. / 2f32.sqrt()).exp();
        assert!((actual[0] - a / (a + b)).abs() < 1e-6);
        assert!((actual[1] - b / (a + b)).abs() < 1e-6);
    }
}
