use crate::{model::DecoderModel, tokenizer};
use serde::Serialize;
use std::io;

/// Request-local decode lookahead accounting. Bytes are payload/capacity, not SSD I/O or RSS.
#[derive(Debug, Default, Serialize)]
pub struct LookaheadMetrics {
    pub eligible_layers: u64,
    pub proposed: u64,
    pub loaded: u64,
    pub load_errors: u64,
    pub used: u64,
    pub unused: u64,
    pub abandoned: u64,
    pub pending: u64,
    pub file_bytes: u64,
    pub buffer_peak_bytes: usize,
}

#[derive(Debug, Default, Serialize)]
pub struct GenerationMetrics {
    pub lookahead_enabled: bool,
    pub lookahead: LookaheadMetrics,
    /// Decode expert payload capacity only; excludes dense/KV/scratch/allocator/driver.
    pub decode_expert_active_peak_bytes: usize,
    pub decode_expert_host_peak_bytes: usize,
    pub tokenizer_ms: f64,
    pub prefill_ms: f64,
    pub time_to_first_token_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub decode_tokens_per_second: f64,
    pub attention_ms: f64,
    pub expert_compute_ms: f64,
    pub lm_head_ms: f64,
    pub expert_io_ms: f64,
    pub prefill_expert_io_ms: f64,
    pub decode_expert_io_ms: f64,
    pub expert_file_read_ms: f64,
    pub expert_vram_download_ms: f64,
    pub expert_vram_upload_ms: f64,
    pub expert_file_bytes_read: u64,
    pub expert_vram_download_bytes: u64,
    pub expert_vram_upload_bytes: u64,
    pub expert_file_loads: u64,
    pub expert_vram_loads: u64,
    pub expert_cache_hits: u64,
    pub expert_cache_misses: u64,
    pub expert_cache_evictions: u64,
    pub expert_bytes_read: u64,
    pub expert_cache_bytes: usize,
    pub kv_cache_bytes: usize,
    pub cached_prompt_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct Generation {
    pub text: String,
    pub token_ids: Vec<usize>,
    pub prompt_tokens: usize,
    pub finish_reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<GenerationMetrics>,
}

impl DecoderModel {
    /// Greedy decoding, with a fresh KV cache for each request.
    pub async fn generate(&self, prompt: &str, max_tokens: usize) -> io::Result<Generation> {
        // Check before encoding to avoid an unnecessary allocation for large input.
        if max_tokens == 0 || prompt.len() >= self.config().max_seq_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "max_tokens must be positive and prompt plus BOS must fit context",
            ));
        }
        let tokens = tokenizer::encode(prompt);
        let available = self.config().max_seq_len - tokens.len();
        if max_tokens > available {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "prompt plus max_tokens exceeds context capacity",
            ));
        }
        let mut state = self.new_state();
        let mut logits = Vec::new();
        for &token in &tokens {
            logits = self.forward_token(token, &mut state).await?;
        }
        let mut output = Vec::new();
        let mut finish_reason = "length";
        for step in 0..max_tokens {
            // BOS is input-only. Ties choose the smaller token ID.
            let next = (0..logits.len())
                .filter(|&id| id != tokenizer::BOS)
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            output.push(next);
            if next == tokenizer::EOS {
                finish_reason = "stop";
                break;
            }
            if step + 1 < max_tokens {
                logits = self.forward_token(next, &mut state).await?;
            }
        }
        Ok(Generation {
            text: tokenizer::decode(&output),
            token_ids: output,
            prompt_tokens: tokens.len(),
            finish_reason,
            metrics: None,
        })
    }
}
