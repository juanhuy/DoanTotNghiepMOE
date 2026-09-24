use crate::{backend::invalid, tokenizer::VOCAB_SIZE};
use serde::{Deserialize, Serialize};
use std::io;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_experts: u32,
    pub top_k: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl ModelConfig {
    pub fn validate(&self) -> io::Result<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_layers == 0
            || self.num_heads == 0
            || self.hidden_size % self.num_heads != 0
            || (self.hidden_size / self.num_heads) % 2 != 0
            || self.num_experts == 0
            || self.top_k == 0
            || self.top_k > self.num_experts as usize
            || self.max_seq_len == 0
            || self.vocab_size != VOCAB_SIZE
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
        {
            return Err(invalid(
                "invalid tiny-moe config (byte vocabulary and even head dimension required)",
            ));
        }
        Ok(())
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 16,
            intermediate_size: 32,
            num_layers: 2,
            num_heads: 2,
            num_experts: 4,
            top_k: 2,
            max_seq_len: 256,
            vocab_size: VOCAB_SIZE,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }
}
