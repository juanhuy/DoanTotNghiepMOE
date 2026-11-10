//! CPU OLMoE decoder for merged INT8 checkpoints.
use crate::{
    backend::{
        causal_attention_fp8_into, causal_attention_into, configure_parallelism, finite, invalid,
        rms_norm_in_place, rms_norm_into, softmax, Matrix,
    },
    generation::{
        Generation, GenerationMetrics, KvBufferReuseMetrics, LookaheadMetrics, MemoryStageSample,
        RequestMemoryTrace,
    },
    gpu::GpuMode,
    int8_expert::{batch_prototype::BatchScratch, ExpertScratch, GateUpMode, Int8Expert},
    quantization::quantize_row_fp8,
    safetensors::TensorIndex,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tracing::{info, warn};

type DeltaSink<'a> = dyn FnMut(&str) -> io::Result<()> + 'a;

#[derive(Deserialize)]
pub struct Config {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    rope_theta: f32,
    #[serde(default)]
    norm_topk_prob: bool,
    eos_token_id: usize,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    clip_qkv: Option<f32>,
    hidden_act: String,
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

impl Config {
    fn validate(&self) -> io::Result<()> {
        let d = self.hidden_size;
        if self.model_type != "olmoe"
            || d == 0
            || d > 16384
            || self.intermediate_size == 0
            || self.intermediate_size > 65536
            || self.num_hidden_layers == 0
            || self.num_hidden_layers > 128
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || d % self.num_attention_heads != 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || (d / self.num_attention_heads) % 2 != 0
            || self.num_experts == 0
            || self.num_experts > 64
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_experts
            || self.vocab_size == 0
            || self.vocab_size > 300_000
            || self.eos_token_id >= self.vocab_size
            || self.max_position_embeddings == 0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.
            || self.attention_bias
            || self.tie_word_embeddings
            || self.rope_scaling.is_some()
            || self.clip_qkv.is_some()
            || self.hidden_act != "silu"
        {
            return Err(invalid("unsupported OLMoE config (requires unscaled RoPE, no biases/clipping, untied embeddings, SiLU)"));
        }
        Ok(())
    }
}
struct CachedExpert {
    id: usize,
    expert: Int8Expert,
    last_used: u64,
    hits_in_cache: u32,
    is_protected: bool,
    is_pinned: bool,
}

#[derive(Serialize, Clone, Debug, Default)]
pub struct SystemTelemetry {
    pub vm_rss_kib: Option<u64>,
    pub vm_hwm_kib: Option<u64>,
    pub vm_swap_kib: Option<u64>,
    pub mem_available_kib: Option<u64>,
    pub io_read_bytes: Option<u64>,
    pub io_rchar: Option<u64>,
    pub gpu_vram_used_mib: Option<u64>,
}

#[derive(Serialize, Clone, Debug)]
pub struct LayerExpertStatus {
    pub resident_experts: Vec<usize>,
    pub heat: Vec<u32>,
    pub cache_bytes: usize,
    pub recycled_bytes: usize,
}

#[derive(Serialize, Clone, Debug)]
pub struct ExpertStatusReport {
    pub expert_batch_prefill_enabled: bool,
    pub gate_up_mode: &'static str,
    pub expert_quantization: &'static str,
    pub kv_buffer_reuse_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_kv_buffer_reuse: Option<KvBufferReuseMetrics>,
    pub memory_trace_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_memory_trace: Option<RequestMemoryTrace>,
    pub gpu_mode_at_load: &'static str,
    pub lookahead_enabled: bool,
    pub expert_cache_policy: &'static str,
    pub num_layers: usize,
    pub num_experts: usize,
    pub cache_bytes_per_layer: usize,
    /// Separate upper bound for active expert payloads/prefetch, not total RSS.
    pub expert_request_payload_limit_bytes: usize,
    pub dense_bytes: usize,
    pub ram_rss_limit_bytes: usize,
    pub ram_system_headroom_bytes: usize,
    pub session_cache_allocated_bytes: usize,
    pub session_cache_limit_bytes: usize,
    pub layers: Vec<LayerExpertStatus>,
    pub system_telemetry: SystemTelemetry,
    pub gpu: crate::gpu::GpuTierStatus,
    pub kv_fp8: bool,
    pub session_dir: Option<String>,
}

/// Thước đo độ phức tạp ngôn ngữ (Perplexity & Cross-Entropy) phục vụ đánh giá học thuật
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PerplexityMetrics {
    pub token_count: usize,
    pub evaluated_tokens: usize,
    pub cross_entropy_loss: f64,
    pub perplexity: f64,
    pub bits_per_character: f64,
    pub eval_time_ms: f64,
    pub tokens_per_second: f64,
}

struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    router: Matrix,
    cache: VecDeque<CachedExpert>,
    bytes: usize,
    heat: Vec<u32>,
    clock: u64,
    recycled_buffers: Vec<(Vec<u8>, Vec<f32>)>,
    hot_pinned: usize,
    router_ema: Vec<f32>,
}

impl Layer {
    fn recycled_bytes(&self) -> usize {
        self.recycled_buffers
            .iter()
            .map(|(w, s)| w.capacity() + s.capacity() * 4)
            .sum()
    }

    fn trim_recycled(&mut self, budget: usize) {
        let mut bytes = self.recycled_bytes();
        while self.bytes.saturating_add(bytes) > budget {
            let Some((w, s)) = self.recycled_buffers.pop() else {
                break;
            };
            bytes = bytes.saturating_sub(w.capacity() + s.capacity() * 4);
        }
    }

    fn recycle(&mut self, buffers: (Vec<u8>, Vec<f32>), budget: usize) {
        let bytes = buffers.0.capacity() + buffers.1.capacity() * 4;
        if self.recycled_buffers.len() < 16
            && self
                .bytes
                .saturating_add(self.recycled_bytes())
                .saturating_add(bytes)
                <= budget
        {
            self.recycled_buffers.push(buffers);
        }
    }

    /// Cập nhật router EMA theo đà định tuyến (Routing Momentum từ Colibrì olmoe.c).
    /// `ema = (1.0 - smooth) * logits + smooth * ema`
    fn update_router_ema(&mut self, logits: &[f32], smooth: f32) {
        if smooth <= 0.0 || logits.is_empty() {
            return;
        }
        let is_uninit = self.router_ema.iter().all(|&v| v == 0.0);
        let n = self.router_ema.len().min(logits.len());
        if is_uninit {
            self.router_ema[..n].copy_from_slice(&logits[..n]);
        } else {
            for (ema, &logit) in self.router_ema[..n].iter_mut().zip(&logits[..n]) {
                *ema = (1.0 - smooth) * logit + smooth * *ema;
            }
        }
    }

    /// Trộn tín hiệu router logits dự phóng PILOT với lịch sử EMA của tầng kế tiếp.
    fn blend_pilot_router(&self, pilot_logits: &mut [f32], smooth: f32) {
        if smooth <= 0.0 || self.router_ema.iter().all(|&v| v == 0.0) {
            return;
        }
        let n = pilot_logits.len().min(self.router_ema.len());
        for (pilot, &ema) in pilot_logits[..n].iter_mut().zip(&self.router_ema[..n]) {
            *pilot = (1.0 - smooth) * *pilot + smooth * ema;
        }
    }

    /// Cập nhật trạng thái Ghim Chuyên Gia Nóng (HOT Expert Pinning lấy cảm hứng từ Colibrì).
    /// Các chuyên gia có tần suất kích hoạt cao nhất trong tầng sẽ được đánh dấu `is_pinned = true`,
    /// ngăn không bị cơ chế LRU loại bỏ ra đĩa khi bộ nhớ cache bị áp lực cao.
    pub fn update_hot_pins(&mut self) {
        if self.hot_pinned == 0 {
            return;
        }
        let mut heat_pairs: Vec<(usize, u32)> = self.heat.iter().copied().enumerate().collect();
        heat_pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let hot_set: std::collections::HashSet<usize> = heat_pairs
            .into_iter()
            .take(self.hot_pinned)
            .filter(|(_, h)| *h > 0)
            .map(|(eid, _)| eid)
            .collect();
        for cached in &mut self.cache {
            cached.is_pinned = hot_set.contains(&cached.id);
        }
    }
}
#[derive(Default, Clone)]
pub(crate) struct Kv {
    keys: Vec<f32>,
    values: Vec<f32>,
    fp8_keys: Vec<u8>,
    fp8_key_scales: Vec<f32>,
    fp8_values: Vec<u8>,
    fp8_value_scales: Vec<f32>,
    is_fp8: bool,
}

impl Kv {
    #[allow(dead_code)]
    fn new(is_fp8: bool) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            fp8_keys: Vec::new(),
            fp8_key_scales: Vec::new(),
            fp8_values: Vec::new(),
            fp8_value_scales: Vec::new(),
            is_fp8,
        }
    }

    #[allow(dead_code)]
    fn with_capacity(capacity_tokens: usize, kd: usize) -> Self {
        Self::with_capacity_precision(capacity_tokens, kd, false)
    }

    fn with_capacity_precision(capacity_tokens: usize, kd: usize, is_fp8: bool) -> Self {
        let total = capacity_tokens * kd;
        if is_fp8 {
            Self {
                keys: Vec::new(),
                values: Vec::new(),
                fp8_keys: Vec::with_capacity(total),
                fp8_key_scales: Vec::with_capacity(capacity_tokens),
                fp8_values: Vec::with_capacity(total),
                fp8_value_scales: Vec::with_capacity(capacity_tokens),
                is_fp8: true,
            }
        } else {
            Self {
                keys: Vec::with_capacity(total),
                values: Vec::with_capacity(total),
                fp8_keys: Vec::new(),
                fp8_key_scales: Vec::new(),
                fp8_values: Vec::new(),
                fp8_value_scales: Vec::new(),
                is_fp8: false,
            }
        }
    }

    fn positions(&self, kd: usize) -> usize {
        if self.is_fp8 {
            self.fp8_key_scales.len()
        } else {
            self.keys.len().checked_div(kd).unwrap_or(0)
        }
    }

    fn truncate_positions(&mut self, prefix: usize, kd: usize) {
        if self.is_fp8 {
            self.fp8_keys.truncate(prefix * kd);
            self.fp8_values.truncate(prefix * kd);
            self.fp8_key_scales.truncate(prefix);
            self.fp8_value_scales.truncate(prefix);
        } else {
            self.keys.truncate(prefix * kd);
            self.values.truncate(prefix * kd);
        }
    }

    fn memory_bytes(&self) -> usize {
        if self.is_fp8 {
            self.fp8_keys.len()
                + self.fp8_values.len()
                + (self.fp8_key_scales.len() + self.fp8_value_scales.len()) * 4
        } else {
            (self.keys.len() + self.values.len()) * 4
        }
    }

    fn allocated_bytes(&self) -> usize {
        if self.is_fp8 {
            self.fp8_keys.capacity()
                + self.fp8_values.capacity()
                + (self.fp8_key_scales.capacity() + self.fp8_value_scales.capacity()) * 4
        } else {
            (self.keys.capacity() + self.values.capacity()) * 4
        }
    }
}

struct SessionState {
    tokens: Vec<usize>,
    kv: Vec<Kv>,
    last_used: u64,
}

struct PreparedSession {
    id: String,
    tokens: Vec<usize>,
}

#[derive(Default)]
struct CandidateDiagnostics {
    positive: u64,
    in_ram: u64,
    in_vram: u64,
    available: u64,
}

#[derive(Default)]
struct RequestProfile {
    kv_buffer_reuse: KvBufferReuseMetrics,
    memory_trace: Option<RequestMemoryTrace>,
    memory_started: Option<Instant>,
    active_kv_container_bytes: usize,
    lookahead: LookaheadMetrics,
    decode_expert_active_peak_bytes: usize,
    decode_expert_host_peak_bytes: usize,
    attention: Duration,
    expert_compute: Duration,
    lm_head: Duration,
    expert_io: Duration,
    file_io: Duration,
    vram_io: Duration,
    upload_io: Duration,
    file_bytes: u64,
    vram_bytes: u64,
    upload_bytes: u64,
    file_loads: u64,
    vram_loads: u64,
    cache_hits: u64,
    cache_misses: u64,
    cache_evictions: u64,
    prefill_expert_batch2_calls: u64,
    prefill_expert_batch4_calls: u64,
    bytes_read: u64,
}

#[derive(Default)]
struct ActiveMemory {
    kv: usize,
    scratch: usize,
    prefill: usize,
}

struct StepScratch {
    expert: ExpertScratch,
    expert_scratches: Vec<ExpertScratch>,
    x: Vec<f32>,
    hidden: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    qkv_buffer: Vec<f32>,
    context: Vec<f32>,
    projected: Vec<f32>,
    router: Vec<f32>,
    pilot_hidden: Vec<f32>,
    pilot_router: Vec<f32>,
    logits: Vec<f32>,
    rope_bases: Vec<f32>,
    sincos: Vec<(f32, f32)>,
    fp8_key: Vec<u8>,
    fp8_value: Vec<u8>,
}

impl StepScratch {
    fn capacity_bytes(&self) -> usize {
        let floats = [
            &self.x,
            &self.hidden,
            &self.query,
            &self.key,
            &self.value,
            &self.qkv_buffer,
            &self.context,
            &self.projected,
            &self.router,
            &self.logits,
            &self.rope_bases,
            &self.pilot_hidden,
            &self.pilot_router,
        ];
        let expert_bytes = |s: &ExpertScratch| (s.activated.capacity() + s.output.capacity()) * 4;
        floats.iter().map(|v| v.capacity() * 4).sum::<usize>()
            + self.sincos.capacity() * std::mem::size_of::<(f32, f32)>()
            + expert_bytes(&self.expert)
            + self.expert_scratches.capacity() * std::mem::size_of::<ExpertScratch>()
            + self
                .expert_scratches
                .iter()
                .map(expert_bytes)
                .sum::<usize>()
    }

    fn new(config: &Config) -> Self {
        let hidden = config.hidden_size;
        let hd = hidden / config.num_attention_heads;
        let half_hd = hd / 2;
        let key_value = hd * config.num_key_value_heads;
        let total_qkv = hidden + key_value + key_value;
        let rope_bases: Vec<f32> = (0..half_hd)
            .map(|i| config.rope_theta.powf(2.0 * i as f32 / hd as f32))
            .collect();
        let sincos = vec![(0.0f32, 0.0f32); half_hd];
        let num_expert_slots = config.num_experts_per_tok.max(1);
        let expert_scratches = (0..num_expert_slots)
            .map(|_| ExpertScratch::new(hidden, config.intermediate_size))
            .collect();
        Self {
            expert: ExpertScratch::new(hidden, config.intermediate_size),
            expert_scratches,
            x: vec![0.0; hidden],
            hidden: vec![0.0; hidden],
            query: vec![0.0; hidden],
            key: vec![0.0; key_value],
            value: vec![0.0; key_value],
            qkv_buffer: vec![0.0; total_qkv],
            context: vec![0.0; hidden],
            projected: vec![0.0; hidden],
            router: vec![0.0; config.num_experts],
            pilot_hidden: vec![0.0; hidden],
            pilot_router: vec![0.0; config.num_experts],
            logits: vec![0.0; config.vocab_size],
            rope_bases,
            sincos,
            fp8_key: vec![0u8; key_value],
            fp8_value: vec![0u8; key_value],
        }
    }
}

/// Bộ ghi nhật ký định tuyến routing trực tiếp lấy cảm hứng từ engine Colibrì (route_trace.h).
/// Ghi vết định tuyến ra luồng văn bản: `<call_id> <row> <layer> <expert_0>:<gate_0> ...`
/// phục vụ phân tích ma trận tương quan kích hoạt (co-activation heatmap) trong luận văn.
#[derive(Debug)]
pub(crate) struct RouteTraceLogger {
    file: fs::File,
    call_id: usize,
}

impl RouteTraceLogger {
    pub(crate) fn new(path: &Path) -> Option<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| warn!("Không thể mở ROUTE_TRACE tại {}: {e}", path.display()))
            .ok()?;
        Some(Self { file, call_id: 0 })
    }

    #[inline]
    pub(crate) fn trace(&mut self, row: usize, layer: usize, experts: &[usize], gates: &[f32]) {
        use std::io::Write;
        let mut line = format!("{} {} {}", self.call_id, row, layer);
        for (&e, &g) in experts.iter().zip(gates) {
            line.push_str(&format!(" {}:{:.4}", e, g));
        }
        line.push('\n');
        let _ = self.file.write_all(line.as_bytes());
    }

    #[inline]
    pub(crate) fn advance_call(&mut self) {
        self.call_id += 1;
    }
}

pub struct OlmoeModel {
    kv_buffer_reuse_enabled: bool,
    last_kv_buffer_reuse: Option<KvBufferReuseMetrics>,
    memory_trace_enabled: bool,
    last_memory_trace: Option<RequestMemoryTrace>,
    // Retain the serial reference only in tests for paired in-process measurements.
    #[cfg(test)]
    serial_prefill_reference: bool,
    #[cfg(test)]
    strict_session_prefix_reference: bool,
    config: Config,
    ram_guard: crate::ram::RamGuard,
    dense_bytes: usize,
    index: TensorIndex,
    embedding: Matrix,
    head: Matrix,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    cache_bytes_per_layer: usize,
    context_limit: usize,
    tokenizer: Option<tokenizers::Tokenizer>,
    tokenizer_config: serde_json::Value,
    sessions: HashMap<String, SessionState>,
    session_clock: u64,
    session_cache_bytes: usize,
    victim_expert_cache: bool,
    expert_batch_prefill_enabled: bool,
    gate_up_mode: GateUpMode,
    expert_quantization: &'static str,
    lookahead_enabled: bool,
    gpu_mode_at_load: GpuMode,
    gpu_tier: crate::gpu::GpuTierManager,
    gpu_dense: Option<crate::gpu::GpuDenseStorage>,
    inter_layer_transitions: Vec<Vec<u16>>,
    usage_path: Option<PathBuf>,
    route_trace: Option<RouteTraceLogger>,
    kv_fp8: bool,
    session_dir: Option<PathBuf>,
}

pub const MOE_KV_MAGIC_FP32: &[u8; 8] = b"MOEKV1\0\0";
pub const MOE_KV_MAGIC_FP8: &[u8; 8] = b"MOEKV2\0\0";
pub const MOE_KV_VERSION: u32 = 1;

fn mat(
    index: &TensorIndex,
    name: &str,
    rows: usize,
    columns: usize,
    budget: &mut usize,
) -> io::Result<Matrix> {
    let values = tensor(index, name, &[rows, columns], budget)?;
    Ok(Matrix {
        rows,
        columns,
        values,
    })
}
fn tensor(
    index: &TensorIndex,
    name: &str,
    shape: &[usize],
    budget: &mut usize,
) -> io::Result<Vec<f32>> {
    if index.tensor(name)?.shape != shape {
        return Err(invalid(format!("{name}: incorrect shape")));
    }
    let values = index.read_f32(name, *budget)?;
    finite(&values)?;
    *budget -= values.len() * 4;
    Ok(values)
}

impl OlmoeModel {
    /// Dense budget excludes expert cache, read scratch buffers and per-request KV state.
    pub fn load(
        path: impl AsRef<Path>,
        cache_bytes_per_layer: usize,
        context_limit: usize,
        dense_budget: usize,
    ) -> io::Result<Self> {
        Self::load_with_gpu_mode(
            path,
            cache_bytes_per_layer,
            context_limit,
            dense_budget,
            GpuMode::from_environment()?,
        )
    }

    pub fn load_with_gpu_mode(
        path: impl AsRef<Path>,
        cache_bytes_per_layer: usize,
        context_limit: usize,
        dense_budget: usize,
        gpu_mode: GpuMode,
    ) -> io::Result<Self> {
        configure_parallelism()?;
        let victim_expert_cache = match std::env::var("MOE_EXPERT_CACHE_POLICY").as_deref() {
            Ok("victim") | Err(std::env::VarError::NotPresent) => true,
            Ok("inclusive") => false,
            _ => {
                return Err(invalid(
                    "MOE_EXPERT_CACHE_POLICY must be inclusive or victim",
                ))
            }
        };

        let lookahead_enabled =
            parse_lookahead_mode(std::env::var("MOE_LOOKAHEAD_PREFETCH").ok().as_deref())?;
        let memory_trace_enabled = match std::env::var("MOE_MEMORY_TRACE") {
            Ok(value) => parse_memory_trace_mode(Some(&value))?,
            Err(std::env::VarError::NotPresent) => false,
            Err(_) => return Err(invalid("MOE_MEMORY_TRACE must be on or off")),
        };
        let kv_buffer_reuse_enabled = match std::env::var("MOE_KV_BUFFER_REUSE") {
            Ok(value) => parse_kv_buffer_reuse_mode(Some(&value))?,
            Err(std::env::VarError::NotPresent) => false,
            Err(_) => return Err(invalid("MOE_KV_BUFFER_REUSE must be on or off")),
        };
        let path = path.as_ref();
        if path.join("INCOMPLETE").exists() {
            return Err(invalid("checkpoint conversion is incomplete"));
        }
        let config: Config = serde_json::from_slice(&fs::read(path.join("config.json"))?)
            .map_err(|e| invalid(e.to_string()))?;
        config.validate()?;
        if context_limit == 0 || context_limit > config.max_position_embeddings {
            return Err(invalid("context limit exceeds model configuration"));
        }
        let index = TensorIndex::open(path)?;
        let d = config.hidden_size;
        let kd = d / config.num_attention_heads * config.num_key_value_heads;
        let raw_weight_count = d
            .checked_mul(config.intermediate_size)
            .and_then(|count| count.checked_mul(3))
            .ok_or_else(|| invalid("expert size overflow"))?;
        let sample_w = index.tensor("model.layers.0.mlp.experts.0.merged_weight")?;
        let is_int4 = sample_w.dtype == crate::safetensors::DType::U8
            || sample_w.shape == [raw_weight_count / 2];
        let weight_slot_bytes = if is_int4 {
            raw_weight_count / 2
        } else {
            raw_weight_count
        };
        let expert_bytes = (config.intermediate_size * 2 + d)
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|scales| weight_slot_bytes.checked_add(scales))
            .ok_or_else(|| invalid("expert size overflow"))?;
        let active_set_bytes = expert_bytes
            .checked_mul(config.num_experts_per_tok)
            .ok_or_else(|| invalid("active expert set size overflow"))?;
        let gate_up_mode = GateUpMode::from_environment()?;
        let expert_dtypes: Vec<_> = index
            .tensors()
            .iter()
            .filter(|(name, _)| name.ends_with(".merged_weight"))
            .map(|(_, tensor)| tensor.dtype)
            .collect();
        let expert_quantization = if expert_dtypes
            .iter()
            .all(|d| *d == crate::safetensors::DType::I8)
        {
            "int8"
        } else if expert_dtypes
            .iter()
            .all(|d| *d == crate::safetensors::DType::U8)
        {
            "int4"
        } else {
            "mixed"
        };
        let expert_batch_prefill_enabled = match std::env::var("MOE_EXPERT_BATCH_PREFILL") {
            Ok(value) => {
                parse_expert_batch_prefill_mode(Some(&value), gpu_mode, expert_quantization)?
            }
            Err(std::env::VarError::NotPresent) => false,
            Err(_) => return Err(invalid("invalid MOE_EXPERT_BATCH_PREFILL")),
        };
        let ram_guard = crate::ram::RamGuard::from_environment()?;
        let dense_planned = index
            .tensors()
            .iter()
            .filter(|(name, _)| name.ends_with(".weight"))
            .try_fold(0usize, |total, (_, tensor)| {
                let count = tensor
                    .shape
                    .iter()
                    .try_fold(1usize, |n, &dim| n.checked_mul(dim))?;
                total.checked_add(count.checked_mul(4)?)
            })
            .ok_or_else(|| invalid("dense RAM estimate overflow"))?;
        // Reserve full configured caches and conservative transient/tokenizer/
        // allocator headroom before reading large weights. This is a plan,
        // not an exact RSS prediction or a replacement for runtime checks.
        let request_reserve = context_limit
            .saturating_mul(d)
            .saturating_mul(4)
            .saturating_mul(config.num_experts_per_tok + 12)
            .saturating_add(
                context_limit
                    .saturating_mul(kd)
                    .saturating_mul(8)
                    .saturating_mul(config.num_hidden_layers),
            )
            .saturating_add(expert_bytes.saturating_mul(config.num_experts_per_tok + 2));
        let planned = dense_planned
            .saturating_add(cache_bytes_per_layer.saturating_mul(config.num_hidden_layers))
            .saturating_add(256 * 1024 * 1024)
            .saturating_add(512 * 1024 * 1024)
            .saturating_add(request_reserve);
        ram_guard.check(planned)?;
        if cache_bytes_per_layer < active_set_bytes {
            tracing::warn!(
                cache_bytes_per_layer,
                active_set_bytes,
                "expert cache is smaller than one token's active expert set; severe thrashing is likely"
            );
        }
        let mut budget = dense_budget;
        let embedding = mat(
            &index,
            "model.embed_tokens.weight",
            config.vocab_size,
            d,
            &mut budget,
        )?;
        let head = mat(&index, "lm_head.weight", config.vocab_size, d, &mut budget)?;
        let norm = tensor(&index, "model.norm.weight", &[d], &mut budget)?;
        let mut layers = Vec::new();
        for id in 0..config.num_hidden_layers {
            let p = format!("model.layers.{id}");
            let mut load = |suffix: &str, shape: &[usize]| {
                tensor(&index, &format!("{p}.{suffix}"), shape, &mut budget)
            };
            let input_norm = load("input_layernorm.weight", &[d])?;
            let post_norm = load("post_attention_layernorm.weight", &[d])?;
            let q_norm = load("self_attn.q_norm.weight", &[d])?;
            let k_norm = load("self_attn.k_norm.weight", &[kd])?;
            let mut matrix =
                |suffix: &str, rows| mat(&index, &format!("{p}.{suffix}"), rows, d, &mut budget);
            let q = matrix("self_attn.q_proj.weight", d)?;
            let k = matrix("self_attn.k_proj.weight", kd)?;
            let v = matrix("self_attn.v_proj.weight", kd)?;
            let o = matrix("self_attn.o_proj.weight", d)?;
            let router = matrix("mlp.gate.weight", config.num_experts)?;
            // Validate expert descriptors without reading all expert payloads.
            for expert in 0..config.num_experts {
                let ep = format!("{p}.mlp.experts.{expert}");
                let w = index.tensor(&format!("{ep}.merged_weight"))?;
                let s = index.tensor(&format!("{ep}.qs"))?;
                let valid_w = if is_int4 {
                    w.dtype == crate::safetensors::DType::U8 && w.shape == [raw_weight_count / 2]
                } else {
                    w.dtype == crate::safetensors::DType::I8 && w.shape == [raw_weight_count]
                };
                if !valid_w
                    || s.dtype != crate::safetensors::DType::F32
                    || s.shape != [2 * config.intermediate_size + d]
                {
                    return Err(invalid(format!("{ep}: invalid merged expert layout")));
                }
            }
            let hot_pinned = std::env::var("MOE_HOT_PIN_COUNT")
                .or_else(|_| std::env::var("HOT"))
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            layers.push(Layer {
                input_norm,
                post_norm,
                q_norm,
                k_norm,
                q,
                k,
                v,
                o,
                router,
                cache: VecDeque::new(),
                bytes: 0,
                heat: vec![0; config.num_experts],
                clock: 0,
                recycled_buffers: Vec::new(),
                hot_pinned,
                router_ema: vec![0.0; config.num_experts],
            });
        }
        let tokenizer = if path.join("tokenizer.json").exists() {
            let mut tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
                .map_err(|e| invalid(e.to_string()))?;
            tokenizer
                .with_truncation(None)
                .map_err(|e| invalid(e.to_string()))?;
            tokenizer.with_padding(None);
            Some(tokenizer)
        } else {
            None
        };
        let tokenizer_config = if path.join("tokenizer_config.json").exists() {
            serde_json::from_slice(&fs::read(path.join("tokenizer_config.json"))?)
                .map_err(|e| invalid(e.to_string()))?
        } else {
            serde_json::Value::Null
        };
        let mut gpu_tier = match gpu_mode {
            GpuMode::Auto => crate::gpu::GpuTierManager::probe(),
            GpuMode::Cpu => {
                info!("CPU selected at load; skipping CUDA probe and GPU allocation");
                crate::gpu::GpuTierManager::cpu_only()
            }
        };
        let gpu_dense = if gpu_tier.is_enabled() {
            if let Some(engine) = crate::gpu::GpuComputeEngine::init() {
                let mut gpu_layers = Vec::with_capacity(layers.len());
                let mut total_bytes = 0;
                let mut failed = false;
                for l in &layers {
                    if let (Some(q), Some(k), Some(v), Some(o)) = (
                        crate::gpu::GpuMatrix::from_matrix(&l.q, &engine),
                        crate::gpu::GpuMatrix::from_matrix(&l.k, &engine),
                        crate::gpu::GpuMatrix::from_matrix(&l.v, &engine),
                        crate::gpu::GpuMatrix::from_matrix(&l.o, &engine),
                    ) {
                        total_bytes += q.bytes + k.bytes + v.bytes + o.bytes;
                        gpu_layers.push(crate::gpu::LayerDenseGpu {
                            q_proj: q,
                            k_proj: k,
                            v_proj: v,
                            o_proj: o,
                        });
                    } else {
                        failed = true;
                        break;
                    }
                }
                if !failed {
                    if let Some(lm_head_gpu) = crate::gpu::GpuMatrix::from_matrix(&head, &engine) {
                        total_bytes += lm_head_gpu.bytes;
                        gpu_tier.set_dense_allocated_bytes(total_bytes);
                        info!(
                            allocated_mb = total_bytes / 1048576,
                            "Đã nạp toàn bộ Dense weights lên NVIDIA GPU VRAM (cuBLAS SGEMV kích hoạt)"
                        );
                        let expert_cache_budget = gpu_tier
                            .budget()
                            .map(|b| (b.usable_vram_bytes as usize).saturating_sub(total_bytes))
                            .unwrap_or(0);
                        let weight_slot_bytes = if is_int4 {
                            (3 * config.hidden_size * config.intermediate_size) / 2
                        } else {
                            3 * config.hidden_size * config.intermediate_size
                        };
                        let scale_slot_bytes = (config.hidden_size + 2 * config.intermediate_size)
                            * std::mem::size_of::<f32>();
                        let single_expert_bytes = weight_slot_bytes + scale_slot_bytes;
                        let max_pool_slots = expert_cache_budget / single_expert_bytes;
                        let expert_pool = if max_pool_slots > 0 {
                            crate::gpu::GpuExpertPool::new(
                                &engine,
                                max_pool_slots,
                                weight_slot_bytes,
                                scale_slot_bytes,
                            )
                        } else {
                            None
                        };
                        let pinned_upload_buf =
                            crate::gpu::PinnedBuffer::alloc(&engine, single_expert_bytes);
                        info!(
                            expert_cache_mb = expert_cache_budget / 1048576,
                            pool_slots = max_pool_slots,
                            has_pinned = pinned_upload_buf.is_some(),
                            "Khởi tạo L1 VRAM Expert Cache (Pre-allocated VRAM Pool & Pinned Host Memory)"
                        );
                        Some(crate::gpu::GpuDenseStorage {
                            layers: gpu_layers,
                            lm_head: lm_head_gpu,
                            total_allocated_bytes: total_bytes,
                            expert_cache: crate::gpu::GpuExpertCache::new(expert_cache_budget),
                            expert_pool,
                            pinned_upload_buf,
                            engine,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if gpu_dense.is_none() && gpu_tier.is_enabled() {
            gpu_tier.set_dense_allocated_bytes(0);
            gpu_tier.set_enabled(false);
            warn!("Không thể offload Dense weights lên GPU VRAM (OOM hoặc lỗi driver). Tự động chuyển 100% về CPU.");
        }

        let num_layers = layers.len();
        let usage_path = std::env::var("MOE_USAGE_PATH")
            .map(PathBuf::from)
            .ok()
            .or_else(|| {
                let default_p = path.join(".moe_usage");
                if default_p.exists() {
                    Some(default_p)
                } else {
                    None
                }
            });

        if let Some(up) = &usage_path {
            Self::load_usage_from_path(up, &mut layers, config.num_experts);
        }

        let route_trace = std::env::var("ROUTE_TRACE")
            .or_else(|_| std::env::var("MOE_ROUTE_TRACE"))
            .ok()
            .and_then(|p| RouteTraceLogger::new(Path::new(&p)));

        let kv_fp8 = std::env::var("MOE_KV_FP8")
            .or_else(|_| std::env::var("FP8_KV"))
            .map(|v| v != "0" && v != "false")
            .unwrap_or(false);

        let session_dir = std::env::var("MOE_SESSION_DIR").ok().map(PathBuf::from);
        if let Some(dir) = &session_dir {
            let _ = fs::create_dir_all(dir);
        }

        Ok(Self {
            #[cfg(test)]
            serial_prefill_reference: true,
            #[cfg(test)]
            strict_session_prefix_reference: false,
            config,
            ram_guard,
            expert_batch_prefill_enabled,
            gate_up_mode,
            expert_quantization,
            dense_bytes: dense_budget - budget,
            index,
            embedding,
            head,
            norm,
            layers,
            cache_bytes_per_layer,
            context_limit,
            tokenizer,
            tokenizer_config,
            sessions: HashMap::new(),
            session_clock: 0,
            session_cache_bytes: 256 * 1024 * 1024,
            victim_expert_cache,
            lookahead_enabled,
            memory_trace_enabled,
            last_memory_trace: None,
            kv_buffer_reuse_enabled,
            last_kv_buffer_reuse: None,
            gpu_mode_at_load: gpu_mode,
            gpu_tier,
            gpu_dense,
            inter_layer_transitions: vec![vec![0u16; 64 * 64]; num_layers.saturating_sub(1)],
            usage_path,
            route_trace,
            kv_fp8,
            session_dir,
        })
    }

    fn record_memory(
        &self,
        profile: &mut RequestProfile,
        stage: &'static str,
        active: ActiveMemory,
    ) {
        let Some(trace) = profile.memory_trace.as_mut() else {
            return;
        };
        let expert_cache = self.layers.iter().map(|l| l.bytes).sum::<usize>();
        let recycled = self.layers.iter().map(Layer::recycled_bytes).sum::<usize>();
        let retained = self.sessions.iter().fold(0usize, |total, (id, state)| {
            total
                .saturating_add(id.capacity())
                .saturating_add(session_bytes(state))
        });
        let retained_kv = self
            .sessions
            .values()
            .map(|s| kv_capacity_bytes(&s.kv, s.kv.capacity() * std::mem::size_of::<Kv>()))
            .sum();
        let tracked = [
            self.dense_bytes,
            expert_cache,
            recycled,
            retained,
            active.kv,
            active.scratch,
            active.prefill,
        ]
        .into_iter()
        .fold(0usize, usize::saturating_add);
        let (rss_bytes, swap_bytes) = crate::ram::process_memory_bytes();
        trace.samples.push(MemoryStageSample {
            stage,
            elapsed_ms: profile.memory_started.map_or(0.0, |t| ms(t.elapsed())),
            rss_bytes,
            swap_bytes,
            thread_id: crate::ram::current_thread_id(),
            allocator: crate::ram::allocator_memory(),
            dense_weight_payload_bytes: self.dense_bytes,
            expert_cache_capacity_bytes: expert_cache,
            recycled_capacity_bytes: recycled,
            retained_session_capacity_bytes: retained,
            retained_kv_capacity_bytes: retained_kv,
            active_kv_capacity_bytes: active.kv,
            scratch_capacity_bytes: active.scratch,
            prefill_capacity_bytes: active.prefill,
            tracked_host_bytes: tracked,
        });
    }

    /// Nạp lịch sử tần suất kích hoạt routing từ file `.moe_usage` (hoặc `.coli_usage`)
    /// theo chuẩn tương thích ngược của Colibrì.
    fn load_usage_from_path(path: &Path, layers: &mut [Layer], num_experts: usize) {
        let Ok(content) = fs::read_to_string(path) else {
            return;
        };
        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            if parts[0] == "-1" {
                if parts.len() >= 3 {
                    let nl = parts[1].parse::<usize>().unwrap_or(0);
                    let ne = parts[2].parse::<usize>().unwrap_or(0);
                    if nl != layers.len() || ne != num_experts {
                        warn!(
                            "Lịch sử routing {} có cấu hình ({} layers, {} experts) không khớp mô hình ({} layers, {} experts). Bỏ qua.",
                            path.display(), nl, ne, layers.len(), num_experts
                        );
                        return;
                    }
                }
                continue;
            }
            if parts[0] == "-2" {
                continue;
            }
            if parts.len() >= 3 {
                if let (Ok(l), Ok(e), Ok(c)) = (
                    parts[0].parse::<usize>(),
                    parts[1].parse::<usize>(),
                    parts[2].parse::<u32>(),
                ) {
                    if l < layers.len() && e < num_experts {
                        layers[l].heat[e] = layers[l].heat[e].saturating_add(c);
                    }
                }
            }
        }
        for layer in layers.iter_mut() {
            layer.update_hot_pins();
        }
        info!(
            "Đã nạp lịch sử routing chuyên gia từ {}, tự động ghim hot experts",
            path.display()
        );
    }

    /// Lưu lịch sử tần suất kích hoạt routing ra file `.moe_usage` với đảm bảo tính nguyên tử (atomic file replace).
    fn save_usage_to_path(path: &Path, layers: &[Layer], num_experts: usize) -> io::Result<()> {
        let mut content = String::with_capacity(layers.len() * num_experts * 16);
        content.push_str(&format!("-1 {} {}\n", layers.len(), num_experts));
        content.push_str("-2 1 3192021591\n");
        for (l, layer) in layers.iter().enumerate() {
            for (e, &count) in layer.heat.iter().enumerate() {
                if count > 0 {
                    content.push_str(&format!("{l} {e} {count}\n"));
                }
            }
        }
        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, content)?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    pub fn expert_status(&self) -> ExpertStatusReport {
        let layers = self
            .layers
            .iter()
            .map(|layer| LayerExpertStatus {
                resident_experts: layer.cache.iter().map(|entry| entry.id).collect(),
                heat: layer.heat.clone(),
                cache_bytes: layer.bytes,
                recycled_bytes: layer.recycled_bytes(),
            })
            .collect();
        let mut gpu_status = self.gpu_status();
        if let Some(storage) = &self.gpu_dense {
            gpu_status.vram_experts_resident = storage.expert_cache.len();
            let total_vram = storage.total_allocated_bytes + storage.expert_cache.used_bytes;
            gpu_status.allocated_vram_bytes = total_vram as u64;
            if gpu_status.total_vram_bytes > total_vram as u64 {
                gpu_status.free_vram_bytes = gpu_status.total_vram_bytes - total_vram as u64;
            }
            gpu_status.message = format!(
                "Hoạt động ({:.2} GB / {:.2} GB VRAM · {} L1 Experts · cuBLAS SGEMV Active)",
                total_vram as f64 / 1073741824.0,
                gpu_status.total_vram_bytes as f64 / 1073741824.0,
                storage.expert_cache.len()
            );
        }
        let mut system_telemetry = Self::get_system_telemetry();
        if gpu_status.enabled {
            system_telemetry.gpu_vram_used_mib = Some(gpu_status.allocated_vram_bytes / 1048576);
        }
        ExpertStatusReport {
            expert_batch_prefill_enabled: self.expert_batch_prefill_enabled,
            gate_up_mode: self.gate_up_mode.as_str(),
            expert_quantization: self.expert_quantization,
            kv_buffer_reuse_enabled: self.kv_buffer_reuse_enabled,
            last_kv_buffer_reuse: self.last_kv_buffer_reuse.clone(),
            memory_trace_enabled: self.memory_trace_enabled,
            last_memory_trace: self.last_memory_trace.clone(),
            gpu_mode_at_load: self.gpu_mode_at_load.as_str(),
            lookahead_enabled: self.lookahead_enabled,
            expert_cache_policy: if self.victim_expert_cache {
                "victim"
            } else {
                "inclusive"
            },
            num_layers: self.layers.len(),
            num_experts: self.config.num_experts,
            cache_bytes_per_layer: self.cache_bytes_per_layer,
            expert_request_payload_limit_bytes: (self.config.num_experts_per_tok + 2)
                * (self.config.hidden_size * self.config.intermediate_size * 3
                    + (self.config.intermediate_size * 2 + self.config.hidden_size) * 4),
            dense_bytes: self.dense_bytes,
            ram_rss_limit_bytes: self.ram_guard.rss_limit,
            ram_system_headroom_bytes: self.ram_guard.system_headroom,
            session_cache_allocated_bytes: self.sessions.iter().fold(
                0usize,
                |total, (id, state)| {
                    total
                        .saturating_add(id.capacity())
                        .saturating_add(session_bytes(state))
                },
            ),
            session_cache_limit_bytes: self.session_cache_bytes,
            layers,
            system_telemetry,
            gpu: gpu_status,
            kv_fp8: self.kv_fp8,
            session_dir: self
                .session_dir
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
        }
    }

    fn gpu_status(&self) -> crate::gpu::GpuTierStatus {
        let mut status = self.gpu_tier.status();
        if self.gpu_mode_at_load == GpuMode::Cpu {
            status.message =
                "GPU tắt từ lúc nạp; khởi động lại với MOE_GPU_MODE=auto để sử dụng GPU".into();
        }
        status
    }

    /// Đảo trạng thái Bật / Tắt GPU Tier khi chạy ứng dụng
    pub fn toggle_gpu(&mut self) -> crate::gpu::GpuTierStatus {
        if self.gpu_mode_at_load == GpuMode::Cpu {
            return self.gpu_status();
        }
        let new_state = self.gpu_tier.toggle_enabled();
        if new_state {
            if let Some(storage) = self.gpu_dense.as_mut() {
                storage.engine.reset_circuit_breaker();
            }
        }
        self.gpu_status()
    }

    /// Thiết lập chế độ nén KV Cache bằng FP8 E4M3 (tiết kiệm 75% RAM KV Cache)
    pub fn set_kv_fp8(&mut self, enabled: bool) {
        if self.kv_fp8 != enabled {
            self.sessions.clear();
            self.kv_fp8 = enabled;
        }
    }

    /// Đảo trạng thái Bật / Tắt nén KV Cache FP8 E4M3
    pub fn toggle_kv_fp8(&mut self) -> bool {
        self.set_kv_fp8(!self.kv_fp8);
        self.kv_fp8
    }

    /// Trạng thái kích hoạt nén KV Cache FP8
    pub fn is_kv_fp8(&self) -> bool {
        self.kv_fp8
    }

    /// Thiết lập thư mục lưu trữ trạng thái KV cache bền vững (.moekv)
    pub fn set_session_dir(&mut self, dir: Option<PathBuf>) {
        if let Some(d) = &dir {
            let _ = fs::create_dir_all(d);
        }
        self.session_dir = dir;
    }

    /// Lấy đường dẫn thư mục lưu trữ trạng thái session
    pub fn session_dir(&self) -> Option<&Path> {
        self.session_dir.as_deref()
    }

    /// Thiết lập trạng thái Bật / Tắt GPU Tier
    pub fn set_gpu_enabled(&mut self, enabled: bool) -> crate::gpu::GpuTierStatus {
        if self.gpu_mode_at_load == GpuMode::Cpu {
            return self.gpu_status();
        }
        self.gpu_tier.set_enabled(enabled);
        if enabled {
            if let Some(storage) = self.gpu_dense.as_mut() {
                storage.engine.reset_circuit_breaker();
            }
        }
        self.gpu_status()
    }

    /// Xóa toàn bộ bộ đệm (KV session cache, Host RAM L2 expert cache, GPU VRAM L1 expert cache)
    /// Đưa hệ thống về trạng thái Khởi động lạnh (Cold Start / 0 cache) hoàn toàn
    pub fn reset_cache(&mut self) {
        self.last_kv_buffer_reuse = None;
        self.last_memory_trace = None;
        // 1. Xóa toàn bộ KV session cache
        self.sessions.clear();
        // Keep the configured admission budget so subsequent sessions can cache KV again.
        self.session_clock = 0;

        // 2. Xóa toàn bộ Host RAM L2 expert cache và đặt lại heatmap
        for layer in &mut self.layers {
            layer.cache.clear();
            layer.recycled_buffers.clear();
            layer.bytes = 0;
            layer.heat.fill(0);
            layer.clock = 0;
        }

        // 3. Xóa GPU VRAM L1 expert cache và hoàn trả các slot về VRAM Pool
        if let Some(storage) = self.gpu_dense.as_mut() {
            storage.clear_expert_cache();
            storage.engine.reset_circuit_breaker();
        }

        // 4. Đặt lại ma trận chuyển tiếp liên tầng (Inter-layer transition matrix)
        for trans in &mut self.inter_layer_transitions {
            trans.fill(0);
        }
    }

    pub(crate) fn reset_cache_with_policy(&mut self, policy: Option<&str>) -> io::Result<()> {
        if let Some(policy) = policy {
            self.victim_expert_cache = match policy {
                "inclusive" => false,
                "victim" => true,
                _ => return Err(invalid("invalid expert cache policy")),
            };
        }
        self.reset_cache();
        Ok(())
    }

    fn get_system_telemetry() -> SystemTelemetry {
        let mut telem = SystemTelemetry {
            gpu_vram_used_mib: Some(0),
            ..Default::default()
        };
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(s) = line.strip_prefix("VmRSS:") {
                    telem.vm_rss_kib = s.split_whitespace().next().and_then(|v| v.parse().ok());
                } else if let Some(s) = line.strip_prefix("VmHWM:") {
                    telem.vm_hwm_kib = s.split_whitespace().next().and_then(|v| v.parse().ok());
                } else if let Some(s) = line.strip_prefix("VmSwap:") {
                    telem.vm_swap_kib = s.split_whitespace().next().and_then(|v| v.parse().ok());
                }
            }
        }
        if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
            telem.mem_available_kib = info.lines().find_map(|line| {
                line.strip_prefix("MemAvailable:")
                    .and_then(|value| value.split_whitespace().next()?.parse().ok())
            });
        }
        if let Ok(io) = std::fs::read_to_string("/proc/self/io") {
            for line in io.lines() {
                if let Some(s) = line.strip_prefix("read_bytes:") {
                    telem.io_read_bytes = s.trim().parse().ok();
                } else if let Some(s) = line.strip_prefix("rchar:") {
                    telem.io_rchar = s.trim().parse().ok();
                }
            }
        }
        telem
    }

    pub fn require_chat(&self) -> io::Result<()> {
        if self.tokenizer.is_none() || !self.tokenizer_config["chat_template"].is_string() {
            return Err(invalid(
                "chat requires tokenizer.json and tokenizer_config.json with chat_template",
            ));
        }
        Ok(())
    }

    pub fn render_chat(&self, messages: &[crate::engine::ChatMessage]) -> io::Result<String> {
        self.require_chat()?;
        let template = self.tokenizer_config["chat_template"].as_str().unwrap();
        let special = |key: &str| {
            let v = &self.tokenizer_config[key];
            v.as_str()
                .or_else(|| v["content"].as_str())
                .unwrap_or("")
                .to_owned()
        };
        let mut env = minijinja::Environment::new();
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
        env.add_template("chat", template)
            .map_err(|e| invalid(e.to_string()))?;
        env.get_template("chat").unwrap().render(minijinja::context! {
            messages => messages, bos_token => special("bos_token"), eos_token => special("eos_token"), add_generation_prompt => true
        }).map_err(|e| invalid(e.to_string()))
    }

    fn dense_gemv(
        gpu_dense: &Option<crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        matrix_cpu: &Matrix,
        matrix_gpu: impl FnOnce(&crate::gpu::GpuDenseStorage) -> &crate::gpu::GpuMatrix,
        input: &[f32],
        output: &mut [f32],
    ) -> io::Result<()> {
        if gpu_enabled {
            if let Some(storage) = gpu_dense {
                if storage.engine.is_healthy() {
                    let m_gpu = matrix_gpu(storage);
                    match storage.engine.gemv(m_gpu, input, output) {
                        Ok(()) => return Ok(()),
                        Err(err) => {
                            warn!(
                                error = %err,
                                "cuBLAS GEMV gặp lỗi! Kích hoạt Circuit Breaker, tự động vô hiệu hóa GPU và chuyển về CPU."
                            );
                            storage.engine.trip_circuit_breaker();
                        }
                    }
                }
            }
        }
        matrix_cpu.multiply_loaded_into(input, output)
    }

    fn dense_qkv(
        gpu_dense: &Option<crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        layer_id: usize,
        layer: &Layer,
        scratch: &mut StepScratch,
    ) -> io::Result<()> {
        if gpu_enabled {
            if let Some(storage) = gpu_dense {
                if storage.engine.is_healthy() {
                    let g_layer = &storage.layers[layer_id];
                    match storage.engine.gemv_qkv(
                        &g_layer.q_proj,
                        &g_layer.k_proj,
                        &g_layer.v_proj,
                        &scratch.hidden,
                        &mut scratch.qkv_buffer,
                        &mut scratch.query,
                        &mut scratch.key,
                        &mut scratch.value,
                    ) {
                        Ok(()) => return Ok(()),
                        Err(err) => {
                            warn!(
                                error = %err,
                                "cuBLAS GEMV QKV gặp lỗi! Kích hoạt Circuit Breaker, tự động vô hiệu hóa GPU và chuyển về CPU."
                            );
                            storage.engine.trip_circuit_breaker();
                        }
                    }
                }
            }
        }
        layer
            .q
            .multiply_loaded_into(&scratch.hidden, &mut scratch.query)?;
        layer
            .k
            .multiply_loaded_into(&scratch.hidden, &mut scratch.key)?;
        layer
            .v
            .multiply_loaded_into(&scratch.hidden, &mut scratch.value)
    }

    fn dense_gemm<F>(
        gpu_dense: &Option<crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        matrix_cpu: &crate::backend::Matrix,
        matrix_gpu: F,
        batch_input: &[f32],
        batch_output: &mut [f32],
        batch_size: usize,
    ) -> io::Result<()>
    where
        F: FnOnce(&crate::gpu::GpuDenseStorage) -> &crate::gpu::GpuMatrix,
    {
        if gpu_enabled {
            if let Some(storage) = gpu_dense {
                if storage.engine.is_healthy() {
                    let m_gpu = matrix_gpu(storage);
                    match storage
                        .engine
                        .gemm(m_gpu, batch_input, batch_output, batch_size)
                    {
                        Ok(()) => return Ok(()),
                        Err(err) => {
                            warn!(
                                error = %err,
                                "cuBLAS GEMM gặp lỗi! Kích hoạt Circuit Breaker, tự động vô hiệu hóa GPU và chuyển về CPU."
                            );
                            storage.engine.trip_circuit_breaker();
                        }
                    }
                }
            }
        }
        let in_dim = matrix_cpu.columns;
        let out_dim = matrix_cpu.rows;
        for (x_chunk, out_chunk) in batch_input
            .chunks_exact(in_dim)
            .zip(batch_output.chunks_exact_mut(out_dim))
        {
            matrix_cpu.multiply_loaded_into(x_chunk, out_chunk)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dense_gemm_qkv(
        gpu_dense: &Option<crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        layer_id: usize,
        layer: &Layer,
        batch_hidden: &[f32],
        batch_size: usize,
        all_q: &mut [f32],
        all_k: &mut [f32],
        all_v: &mut [f32],
    ) -> io::Result<()> {
        if gpu_enabled {
            if let Some(storage) = gpu_dense {
                if storage.engine.is_healthy() {
                    let g_layer = &storage.layers[layer_id];
                    match storage.engine.gemm_qkv(
                        &g_layer.q_proj,
                        &g_layer.k_proj,
                        &g_layer.v_proj,
                        batch_hidden,
                        batch_size,
                        all_q,
                        all_k,
                        all_v,
                    ) {
                        Ok(()) => return Ok(()),
                        Err(err) => {
                            warn!(
                                error = %err,
                                "cuBLAS GEMM QKV gặp lỗi! Kích hoạt Circuit Breaker, tự động vô hiệu hóa GPU và chuyển về CPU."
                            );
                            storage.engine.trip_circuit_breaker();
                        }
                    }
                }
            }
        }
        let d = layer.q.columns;
        let q_dim = layer.q.rows;
        let k_dim = layer.k.rows;
        let v_dim = layer.v.rows;
        for (i, h_chunk) in batch_hidden.chunks_exact(d).enumerate() {
            layer
                .q
                .multiply_loaded_into(h_chunk, &mut all_q[i * q_dim..(i + 1) * q_dim])?;
            layer
                .k
                .multiply_loaded_into(h_chunk, &mut all_k[i * k_dim..(i + 1) * k_dim])?;
            layer
                .v
                .multiply_loaded_into(h_chunk, &mut all_v[i * v_dim..(i + 1) * v_dim])?;
        }
        Ok(())
    }

    /// Thăng hạng một Expert INT8 lên L1 VRAM Cache nếu còn ngân sách và chưa có trong cache
    fn promote_to_vram(
        gpu_dense: &mut Option<crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        layer_id: usize,
        expert_id: usize,
        weights: &[u8],
        scales: &[f32],
        profile: &mut RequestProfile,
    ) {
        if let Some(storage) = gpu_dense.as_mut() {
            if gpu_enabled && storage.expert_cache.max_bytes > 0 {
                let started = Instant::now();
                if storage.upload_and_cache(layer_id, expert_id, weights, scales) {
                    profile.upload_io += started.elapsed();
                    profile.upload_bytes += (weights.len() + std::mem::size_of_val(scales)) as u64;
                }
            }
        }
    }

    /// Nạp trước (prefetch) song song các expert bị miss bằng Rayon — Sub-step 16-B
    #[allow(clippy::too_many_arguments)]
    fn prefetch_missing_experts(
        layer_id: usize,
        expert_ids: &[usize],
        d: usize,
        intermediate_size: usize,
        index: &TensorIndex,
        layer: &mut Layer,
        gpu_dense: Option<&crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        max_prefetch_experts: usize,
    ) -> (HashMap<usize, Int8Expert>, Duration) {
        use rayon::prelude::*;

        let missing_ids: Vec<usize> = expert_ids
            .iter()
            .copied()
            .filter(|&eid| {
                if layer.cache.iter().any(|entry| entry.id == eid) {
                    return false;
                }
                if let Some(storage) = gpu_dense {
                    if gpu_enabled && storage.expert_cache.contains(layer_id, eid) {
                        return false;
                    }
                }
                true
            })
            .take(max_prefetch_experts)
            .collect();

        if missing_ids.len() < 2 {
            return (HashMap::new(), Duration::ZERO);
        }

        // Colibrì-inspired Sequential SSD locality: sắp xếp tăng dần theo EID để chuyển đổi
        // các lượt đọc ngẫu nhiên thành đọc tuần tự trên file shard đĩa.
        let mut missing_ids = missing_ids;
        missing_ids.sort_unstable();

        let weight_count = d * intermediate_size * 3;
        let scale_count = intermediate_size * 2 + d;
        let mut buffers = Vec::with_capacity(missing_ids.len());
        for _ in 0..missing_ids.len() {
            let buf = layer
                .recycled_buffers
                .pop()
                .unwrap_or_else(|| (vec![0u8; weight_count], vec![0.0f32; scale_count]));
            buffers.push(buf);
        }

        let started = Instant::now();
        let loaded: HashMap<usize, Int8Expert> = missing_ids
            .into_par_iter()
            .zip(buffers.into_par_iter())
            .filter_map(|(eid, (w, s))| {
                Int8Expert::load_reusing(index, layer_id, eid, d, intermediate_size, w, s)
                    .ok()
                    .map(|exp| (eid, exp))
            })
            .collect();
        let elapsed = started.elapsed();
        (loaded, elapsed)
    }

    /// Dự đoán danh sách expert của tầng tiếp theo (Lookahead Speculative Routing)
    /// Kết hợp giữa:
    /// 1. Tín hiệu dự phóng động PILOT Router từ Colibrì (đánh giá router tầng sau trên residual stream x hiện tại)
    /// 2. Ngưỡng độ tin cậy tích lũy Cumulative Confidence Cutoff (từ Colibrì olmoe.c pilot_conf_limit)
    /// 3. Ma trận xác suất chuyển tiếp giữa 2 tầng liên tiếp (Inter-Layer Markov Transition)
    /// 4. Tín hiệu nhiệt (Heat/Recency) của tầng tiếp theo
    /// 5. Lọc bỏ các expert đã có sẵn trong L2 Host RAM Cache hoặc L1 GPU VRAM Cache
    /// 6. Sắp xếp thứ tự EID tăng dần để tối ưu SSD read locality (Sequential Seek)
    #[allow(clippy::too_many_arguments)]
    fn predict_next_layer_experts(
        pilot_router: Option<&[f32]>,
        transitions: &[Vec<u16>],
        curr_layer_id: usize,
        curr_top_experts: &[usize],
        next_layer_heat: &[u32],
        next_layer_cache: &VecDeque<CachedExpert>,
        gpu_dense: Option<&crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        max_candidates: usize,
        conf_limit: Option<f32>,
    ) -> (Vec<usize>, CandidateDiagnostics) {
        if curr_layer_id >= transitions.len() || max_candidates == 0 {
            return (Vec::new(), CandidateDiagnostics::default());
        }
        let trans = &transitions[curr_layer_id];
        let mut scores = [0u32; 64];

        // 1. Tín hiệu dự phóng động PILOT Router: xếp hạng Top-K logits của router tầng kế tiếp
        // Áp dụng Cumulative Confidence Cutoff (học hỏi từ Colibrì olmoe.c pilot_conf_limit)
        let mut pilot_selected_count = max_candidates;
        if let Some(logits) = pilot_router {
            if logits.len() >= 64 {
                let max_l = logits[..64]
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut exps = [0.0f32; 64];
                let mut sum_exp = 0.0f32;
                for i in 0..64 {
                    exps[i] = (logits[i] - max_l).exp();
                    sum_exp += exps[i];
                }

                let mut pilot_order: Vec<usize> = (0..64).collect();
                pilot_order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

                let mut cum_prob = 0.0f32;
                let limit = conf_limit.unwrap_or(1.0);
                let mut taken = 0;
                for (rank, &eid) in pilot_order.iter().take(16).enumerate() {
                    let prob = if sum_exp > 0.0 {
                        exps[eid] / sum_exp
                    } else {
                        0.0
                    };
                    cum_prob += prob;
                    let pilot_score = ((16 - rank) * 100) as u32;
                    scores[eid] = scores[eid].saturating_add(pilot_score);
                    taken += 1;
                    if conf_limit.is_some() && cum_prob >= limit && taken >= 1 {
                        break;
                    }
                }
                if conf_limit.is_some() {
                    pilot_selected_count = taken;
                }
            }
        }

        // 2. Tích lũy xác suất chuyển tiếp từ các expert đang hoạt động ở tầng hiện tại (Markov)
        for &src in curr_top_experts {
            if src < 64 {
                let row_offset = src * 64;
                for (dst, score) in scores.iter_mut().enumerate() {
                    *score = score.saturating_add(trans[row_offset + dst] as u32);
                }
            }
        }

        // 3. Kết hợp với tín hiệu nhiệt (Heat/Recency) của tầng tiếp theo
        for (dst, score) in scores.iter_mut().enumerate() {
            if dst < next_layer_heat.len() {
                *score = score.saturating_add(next_layer_heat[dst] / 4);
            }
        }

        // 4. Lọc các expert CHƯA CÓ trong cache của tầng tiếp theo (và chỉ lấy expert có điểm tin cậy > 0)
        let next_layer_id = curr_layer_id + 1;
        let mut diagnostics = CandidateDiagnostics::default();
        let mut candidates = Vec::new();
        for (eid, score) in scores.into_iter().enumerate() {
            if score == 0 {
                continue;
            }
            diagnostics.positive += 1;
            if next_layer_cache.iter().any(|entry| entry.id == eid) {
                diagnostics.in_ram += 1;
            } else if gpu_enabled
                && gpu_dense.is_some_and(|s| s.expert_cache.contains(next_layer_id, eid))
            {
                diagnostics.in_vram += 1;
            } else {
                diagnostics.available += 1;
                candidates.push((eid, score));
            }
        }

        // 5. Sắp xếp giảm dần theo điểm số dự đoán tổng hợp
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // 6. Lấy top ứng viên có xác suất cao nhất (giới hạn bởi max_candidates và confidence threshold)
        let effective_limit = if conf_limit.is_some() && pilot_router.is_some() {
            max_candidates.min(pilot_selected_count)
        } else {
            max_candidates
        };
        let mut selected: Vec<usize> = candidates
            .into_iter()
            .take(effective_limit)
            .map(|(eid, _)| eid)
            .collect();

        // 7. Sắp xếp tăng dần theo EID để tối ưu hóa SSD read locality (Sequential Seek)
        selected.sort_unstable();
        (selected, diagnostics)
    }

    /// Tra cứu Expert theo thứ tự phân tầng 3 cấp: L2 Host RAM -> L1 GPU VRAM -> SSD NVMe Cold Storage
    #[allow(clippy::too_many_arguments)]
    fn acquire_expert_tiered(
        layer_id: usize,
        expert_id: usize,
        d: usize,
        intermediate_size: usize,
        weight_count: usize,
        scale_count: usize,
        index: &TensorIndex,
        layer: &mut Layer,
        gpu_dense: &mut Option<crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        victim_cache: bool,
        profile: &mut RequestProfile,
        uses_count: u64,
        preloaded: &mut HashMap<usize, Int8Expert>,
    ) -> io::Result<(Int8Expert, bool, u32, bool)> {
        let (expert, l2_hit, hits_in_cache, is_protected) =
            if let Some(pos) = layer.cache.iter().position(|entry| entry.id == expert_id) {
                profile.cache_hits += uses_count;
                let entry = layer.cache.remove(pos).unwrap();
                layer.bytes = layer.bytes.saturating_sub(entry.expert.allocated_bytes());
                (
                    entry.expert,
                    true,
                    entry.hits_in_cache.saturating_add(1),
                    entry.is_protected,
                )
            } else {
                // Cấp 1: Kiểm tra L1 GPU VRAM Cache trước khi đọc đĩa SSD NVMe
                let mut from_vram = None;
                if let Some(storage) = gpu_dense.as_mut() {
                    if gpu_enabled {
                        let crate::gpu::GpuDenseStorage {
                            engine,
                            expert_cache,
                            ..
                        } = storage;
                        if let Some(gpu_weights) = expert_cache.get(layer_id, expert_id) {
                            let io_started = Instant::now();
                            let (mut weights, mut scales) =
                                layer.recycled_buffers.pop().unwrap_or_else(|| {
                                    (vec![0u8; weight_count], vec![0.0f32; scale_count])
                                });
                            weights.resize(weight_count, 0);
                            scales.resize(scale_count, 0.0);
                            if engine
                                .download_expert(gpu_weights, &mut weights, &mut scales)
                                .is_ok()
                            {
                                let exp =
                                    Int8Expert::from_raw(d, intermediate_size, weights, scales);
                                let elapsed = io_started.elapsed();
                                profile.expert_io += elapsed;
                                profile.vram_io += elapsed;
                                profile.vram_loads += 1;
                                profile.vram_bytes += exp.resident_bytes() as u64;
                                profile.bytes_read += exp.resident_bytes() as u64;
                                profile.cache_hits += uses_count;
                                from_vram = Some(exp);
                            }
                        }
                    }
                }

                if let Some(exp) = from_vram {
                    (exp, false, 1, false)
                } else {
                    // Cấp 2: Miss cả L1 và L2 -> Lấy từ preloaded (16-B) hoặc nạp từ SSD NVMe Cold Storage
                    profile.cache_misses += 1;
                    profile.cache_hits += uses_count.saturating_sub(1);
                    let (exp, io_elapsed) = if let Some(p) = preloaded.remove(&expert_id) {
                        (p, Duration::ZERO)
                    } else {
                        let io_started = Instant::now();
                        let (w, s) = layer.recycled_buffers.pop().unwrap_or_else(|| {
                            (vec![0u8; weight_count], vec![0.0f32; scale_count])
                        });
                        let loaded = Int8Expert::load_reusing(
                            index,
                            layer_id,
                            expert_id,
                            d,
                            intermediate_size,
                            w,
                            s,
                        )?;
                        let elapsed = io_started.elapsed();
                        profile.file_io += elapsed;
                        profile.file_loads += 1;
                        profile.file_bytes += loaded.resident_bytes() as u64;
                        (loaded, elapsed)
                    };
                    profile.expert_io += io_elapsed;
                    profile.bytes_read += exp.resident_bytes() as u64;

                    if !victim_cache {
                        // Tự động thăng hạng (promote) expert vừa đọc lên L1 VRAM Cache
                        Self::promote_to_vram(
                            gpu_dense,
                            gpu_enabled,
                            layer_id,
                            expert_id,
                            exp.weights(),
                            exp.scales(),
                            profile,
                        );
                    }
                    (exp, false, 1, false)
                }
            };

        if l2_hit && !victim_cache {
            Self::promote_to_vram(
                gpu_dense,
                gpu_enabled,
                layer_id,
                expert_id,
                expert.weights(),
                expert.scales(),
                profile,
            );
        }

        Ok((expert, l2_hit, hits_in_cache, is_protected))
    }

    /// Quản lý nạp và trục xuất (admission & eviction) cho L2 Host RAM Cache
    /// Áp dụng thuật toán Segmented Frequency-Recency (SF-Cache / 2Q Partition):
    /// - 70% ngân sách cho Protected Segment (các Hot/Generalist Experts thường trực)
    /// - 30% ngân sách cho Probationary Segment (cửa sổ thử thách cho Specialist Experts mới)
    /// - Trục xuất (eviction) luôn ưu tiên loại bỏ các phần tử trong Probationary Segment,
    ///   bảo vệ tuyệt đối các Generalist Experts khỏi hiện tượng Cache Thrashing do Routing Drift.
    #[allow(clippy::too_many_arguments)]
    fn admit_expert_l2(
        layer: &mut Layer,
        expert_id: usize,
        expert: Int8Expert,
        l2_hit: bool,
        hits_in_cache: u32,
        mut is_protected: bool,
        cache_bytes_per_layer: usize,
        profile: &mut RequestProfile,
        gpu_dense: &mut Option<crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        victim_cache: bool,
        layer_id: usize,
    ) {
        let bytes = expert.allocated_bytes();
        if bytes == 0 || bytes > cache_bytes_per_layer {
            if bytes > cache_bytes_per_layer && victim_cache {
                // With no room in L2, retain the expert in L1 rather than losing
                // both caches. Existing L1 entries make this upload a no-op.
                Self::promote_to_vram(
                    gpu_dense,
                    gpu_enabled,
                    layer_id,
                    expert_id,
                    expert.weights(),
                    expert.scales(),
                    profile,
                );
            }
            return;
        }

        // 70% ngân sách cho Protected partition
        let protected_budget = (cache_bytes_per_layer * 7) / 10;

        if l2_hit {
            // Đã từng có trong cache -> Thăng hạng lên Protected
            is_protected = true;

            // Nếu vùng Protected vượt quá ngân sách 70%, giáng hạng (demote) phần tử nguội nhất về Probationary
            let current_protected_bytes: usize = layer
                .cache
                .iter()
                .filter(|e| e.is_protected)
                .map(|e| e.expert.resident_bytes())
                .sum();

            if current_protected_bytes + bytes > protected_budget {
                if let Some(demote_idx) = layer
                    .cache
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.is_protected)
                    .min_by_key(|(_, e)| (layer.heat[e.id], e.last_used))
                    .map(|(idx, _)| idx)
                {
                    layer.cache[demote_idx].is_protected = false;
                }
            }
        }

        // Đảm bảo tổng dung lượng không vượt quá cache_bytes_per_layer
        // Cơ chế HOT Expert Pinning (Colibrì-inspired): Bỏ qua các expert đã được ghim (`is_pinned = true`).
        // 1. Ưu tiên trục xuất các expert Probationary chưa ghim (!is_pinned && !is_protected).
        // 2. Nếu không còn, trục xuất các expert Protected chưa ghim (!is_pinned).
        // 3. Trường hợp áp lực cực đoan (toàn bộ cache đều bị ghim), buộc phải trục xuất phần tử nguội nhất.
        while layer.bytes > cache_bytes_per_layer.saturating_sub(bytes) {
            let victim_idx = layer
                .cache
                .iter()
                .enumerate()
                .filter(|(_, e)| !e.is_pinned && !e.is_protected)
                .min_by_key(|(_, e)| (layer.heat[e.id], e.last_used))
                .map(|(idx, _)| idx)
                .or_else(|| {
                    layer
                        .cache
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| !e.is_pinned)
                        .min_by_key(|(_, e)| (layer.heat[e.id], e.last_used))
                        .map(|(idx, _)| idx)
                })
                .or_else(|| {
                    layer
                        .cache
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, e)| (layer.heat[e.id], e.last_used))
                        .map(|(idx, _)| idx)
                });

            let Some(idx) = victim_idx else {
                break;
            };

            if let Some(removed) = layer.cache.remove(idx) {
                layer.bytes = layer.bytes.saturating_sub(removed.expert.allocated_bytes());
                if victim_cache {
                    Self::promote_to_vram(
                        gpu_dense,
                        gpu_enabled,
                        layer_id,
                        removed.id,
                        removed.expert.weights(),
                        removed.expert.scales(),
                        profile,
                    );
                }
                layer.recycle(
                    removed.expert.into_raw(),
                    cache_bytes_per_layer.saturating_sub(bytes),
                );
                profile.cache_evictions += 1;
            }
        }

        let is_pinned = if layer.hot_pinned > 0 {
            let mut heat_pairs: Vec<(usize, u32)> =
                layer.heat.iter().copied().enumerate().collect();
            heat_pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            heat_pairs
                .iter()
                .take(layer.hot_pinned)
                .any(|&(eid, h)| eid == expert_id && h > 0)
        } else {
            false
        };

        if layer.bytes <= cache_bytes_per_layer - bytes {
            layer.trim_recycled(cache_bytes_per_layer - bytes);
            if victim_cache && gpu_enabled {
                if let Some(storage) = gpu_dense.as_mut() {
                    // Synchronous download is complete; L2 now owns the host payload.
                    storage.expert_cache.remove_with_pool(
                        layer_id,
                        expert_id,
                        storage.expert_pool.as_mut(),
                    );
                }
            }
            layer.bytes += bytes;
            layer.cache.push_back(CachedExpert {
                id: expert_id,
                expert,
                last_used: layer.clock,
                hits_in_cache,
                is_protected,
                is_pinned,
            });
        }
    }

    fn step(
        &mut self,
        token: usize,
        kv: &mut [Kv],
        profile: &mut RequestProfile,
        scratch: &mut StepScratch,
        check: &dyn Fn() -> io::Result<()>,
    ) -> io::Result<()> {
        let result = self.step_inner(token, kv, profile, scratch, check);
        if result.is_err() {
            if profile.lookahead.pending > 0 || profile.lookahead.load_errors > 0 {
                warn!(lookahead = ?profile.lookahead, "decode failed; dropping pending speculative payloads");
            }
            profile.lookahead.abandoned += profile.lookahead.pending;
            profile.lookahead.pending = 0;
        }
        result
    }

    fn step_inner(
        &mut self,
        token: usize,
        kv: &mut [Kv],
        profile: &mut RequestProfile,
        scratch: &mut StepScratch,
        check: &dyn Fn() -> io::Result<()>,
    ) -> io::Result<()> {
        let victim_cache = self.victim_expert_cache;
        let c = &self.config;
        if token >= c.vocab_size {
            return Err(invalid("token ID exceeds model vocabulary"));
        }
        let d = c.hidden_size;
        let hd = d / c.num_attention_heads;
        let kd = hd * c.num_key_value_heads;
        let intermediate_size = c.intermediate_size;
        let weight_count = d * intermediate_size * 3;
        let scale_count = intermediate_size * 2 + d;
        scratch
            .x
            .copy_from_slice(&self.embedding.values[token * d..(token + 1) * d]);

        let (layers, gpu_dense, gpu_tier, index, cache_bytes_per_layer, transitions, route_trace) = (
            &mut self.layers,
            &mut self.gpu_dense,
            &self.gpu_tier,
            &self.index,
            self.cache_bytes_per_layer,
            &mut self.inter_layer_transitions,
            &mut self.route_trace,
        );

        let lookahead_enabled = self.lookahead_enabled;
        let num_layers = layers.len();
        let mut lookahead_prefetched: HashMap<usize, Int8Expert> = HashMap::new();
        let mut prev_top_experts: Vec<usize> = Vec::with_capacity(8);

        let pilot_smooth = std::env::var("MOE_PILOT_SMOOTH")
            .or_else(|_| std::env::var("SMOOTH"))
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.35);

        let pilot_conf_limit = std::env::var("MOE_PILOT_CONF_LIMIT")
            .or_else(|_| std::env::var("CONF_LIMIT"))
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .or(Some(0.90));

        for id in 0..num_layers {
            check()?;
            let (previous, current_and_next) = layers.split_at_mut(id);
            let (layer, remaining) = current_and_next.split_first_mut().unwrap();

            let attention_started = Instant::now();
            let position = kv[id].positions(kd);
            if position >= self.context_limit {
                return Err(invalid("context exhausted"));
            }
            rms_norm_into(
                &scratch.x,
                &layer.input_norm,
                c.rms_norm_eps,
                &mut scratch.hidden,
            )?;
            Self::dense_qkv(&*gpu_dense, gpu_tier.is_enabled(), id, layer, scratch)?;
            rms_norm_in_place(&mut scratch.query, &layer.q_norm, c.rms_norm_eps)?;
            rms_norm_in_place(&mut scratch.key, &layer.k_norm, c.rms_norm_eps)?;
            prepare_rope_sincos(&mut scratch.sincos, &scratch.rope_bases, position);
            apply_rope_cached(&mut scratch.query, hd, &scratch.sincos);
            apply_rope_cached(&mut scratch.key, hd, &scratch.sincos);
            if kv[id].is_fp8 {
                let k_scale = quantize_row_fp8(&scratch.key, &mut scratch.fp8_key);
                let v_scale = quantize_row_fp8(&scratch.value, &mut scratch.fp8_value);
                kv[id].fp8_keys.extend_from_slice(&scratch.fp8_key);
                kv[id].fp8_key_scales.push(k_scale);
                kv[id].fp8_values.extend_from_slice(&scratch.fp8_value);
                kv[id].fp8_value_scales.push(v_scale);
                causal_attention_fp8_into(
                    &scratch.query,
                    &kv[id].fp8_keys,
                    &kv[id].fp8_key_scales,
                    &kv[id].fp8_values,
                    &kv[id].fp8_value_scales,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    &mut scratch.context,
                )?;
            } else {
                kv[id].keys.extend_from_slice(&scratch.key);
                kv[id].values.extend_from_slice(&scratch.value);
                causal_attention_into(
                    &scratch.query,
                    &kv[id].keys,
                    &kv[id].values,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    &mut scratch.context,
                )?;
            }
            Self::dense_gemv(
                &*gpu_dense,
                gpu_tier.is_enabled(),
                &layer.o,
                |g| &g.layers[id].o_proj,
                &scratch.context,
                &mut scratch.projected,
            )?;
            for (a, b) in scratch.x.iter_mut().zip(&scratch.projected) {
                *a += b;
            }
            profile.attention += attention_started.elapsed();
            rms_norm_into(
                &scratch.x,
                &layer.post_norm,
                c.rms_norm_eps,
                &mut scratch.hidden,
            )?;
            layer
                .router
                .multiply_loaded_into(&scratch.hidden, &mut scratch.router)?;
            layer.update_router_ema(&scratch.router, pilot_smooth);
            softmax(&mut scratch.router)?;

            let mut expert_indices = [0usize; 64];
            for (i, idx) in expert_indices.iter_mut().enumerate().take(c.num_experts) {
                *idx = i;
            }
            let valid_indices = &mut expert_indices[..c.num_experts];
            let top_k = c.num_experts_per_tok;
            if top_k < valid_indices.len() {
                valid_indices.select_nth_unstable_by(top_k, |&a, &b| {
                    scratch.router[b]
                        .total_cmp(&scratch.router[a])
                        .then_with(|| a.cmp(&b))
                });
            }
            let top_experts = &mut valid_indices[..top_k];
            top_experts.sort_by(|&a, &b| {
                scratch.router[b]
                    .total_cmp(&scratch.router[a])
                    .then_with(|| a.cmp(&b))
            });
            let normalizer = if c.norm_topk_prob {
                top_experts
                    .iter()
                    .map(|&expert| scratch.router[expert])
                    .sum()
            } else {
                1.0
            };

            if let Some(tracer) = route_trace.as_mut() {
                let pos = if !kv.is_empty() && kd > 0 {
                    kv[0].positions(kd)
                } else {
                    0
                };
                let gates: Vec<f32> = top_experts
                    .iter()
                    .map(|&expert| scratch.router[expert] / normalizer)
                    .collect();
                tracer.trace(pos, id, top_experts, &gates);
            }

            // 1. Cập nhật Inter-Layer Markov Transition Matrix:
            // Tích lũy tần suất chuyển tiếp giữa top_experts của tầng trước (id - 1) và tầng hiện tại (id)
            if id > 0 && !prev_top_experts.is_empty() && id - 1 < transitions.len() {
                let trans = &mut transitions[id - 1];
                for &src in &prev_top_experts {
                    if src < 64 {
                        let row_offset = src * 64;
                        for &dst in top_experts.iter() {
                            if dst < 64 {
                                let idx = row_offset + dst;
                                trans[idx] = trans[idx].saturating_add(1);
                            }
                        }
                    }
                }
            }
            prev_top_experts.clear();
            prev_top_experts.extend_from_slice(top_experts);

            // 2. Phân loại Expert Hit/Miss kết hợp trúng Lookahead Prefetch từ tầng trước:
            let mut preloaded = HashMap::new();
            let mut hit_slots = Vec::with_capacity(top_k);
            let mut miss_slots = Vec::with_capacity(top_k);
            let mut miss_ids = Vec::with_capacity(top_k);

            for (slot, &eid) in top_experts.iter().enumerate() {
                let is_hit = layer.cache.iter().any(|entry| entry.id == eid)
                    || gpu_dense
                        .as_ref()
                        .is_some_and(|s| gpu_tier.is_enabled() && s.expert_cache.contains(id, eid));
                if is_hit {
                    hit_slots.push(slot);
                } else if let Some(exp) = lookahead_prefetched.remove(&eid) {
                    // TRÚNG LOOKAHEAD PREFETCH TỪ TẦNG TRƯỚC!
                    // Biến 1 Miss I/O từ SSD thành 1 Hit tức thời trong RAM!
                    profile.lookahead.used += 1;
                    profile.lookahead.pending -= 1;
                    preloaded.insert(eid, exp);
                    hit_slots.push(slot);
                } else {
                    miss_slots.push(slot);
                    miss_ids.push(eid);
                }
            }

            // Hoàn trả các expert còn lại trong lookahead_prefetched (nếu không dùng đến) vào recycled_buffers
            for (_, unused_exp) in lookahead_prefetched.drain() {
                profile.lookahead.unused += 1;
                profile.lookahead.pending -= 1;
                layer.recycle(unused_exp.into_raw(), cache_bytes_per_layer);
            }

            // 3. Dự đoán Speculative Lookahead Candidates cho tầng tiếp theo (id + 1):
            // Opportunistic Lookahead: Chỉ kích hoạt nạp trước khi tầng hiện tại KHÔNG CÓ MISS (toàn bộ Hit).
            // Nếu tầng hiện tại có miss, ưu tiên 100% băng thông I/O nạp miss để tránh hiện tượng Prefetch Contention.
            let diagnostics = &mut profile.lookahead;
            diagnostics.evaluated_layers += 1;
            if remaining.is_empty() {
                diagnostics.no_next_layer += 1;
            } else {
                diagnostics.miss_histogram.resize(top_k + 1, 0);
                diagnostics.miss_histogram[miss_slots.len()] += 1;
                if miss_slots.is_empty() {
                    diagnostics.eligible_layers += 1;
                    if !lookahead_enabled {
                        diagnostics.disabled_eligible_layers += 1;
                    }
                } else {
                    diagnostics.blocked_by_misses += 1;
                }
            }
            let next_candidates = if lookahead_enabled && miss_slots.is_empty() {
                if let Some(next_l) = remaining.first() {
                    // PILOT Router Lookahead (từ Colibrì):
                    // Đánh giá trực tiếp router của tầng kế tiếp (next_l) trên residual activation scratch.x
                    // Đạt độ chính xác dự đoán cao hơn nhiều so với chỉ dùng bảng tần suất Markov tĩnh.
                    let pilot_router_opt = if rms_norm_into(
                        &scratch.x,
                        &next_l.post_norm,
                        c.rms_norm_eps,
                        &mut scratch.pilot_hidden,
                    )
                    .is_ok()
                        && next_l
                            .router
                            .multiply_loaded_into(&scratch.pilot_hidden, &mut scratch.pilot_router)
                            .is_ok()
                    {
                        next_l.blend_pilot_router(&mut scratch.pilot_router, pilot_smooth);
                        Some(&scratch.pilot_router[..])
                    } else {
                        None
                    };

                    let (candidates, stats) = Self::predict_next_layer_experts(
                        pilot_router_opt,
                        transitions,
                        id,
                        top_experts,
                        &next_l.heat,
                        &next_l.cache,
                        gpu_dense.as_ref(),
                        gpu_tier.is_enabled(),
                        2, // At most two positive-score uncached candidates.
                        pilot_conf_limit,
                    );
                    profile.lookahead.prediction_calls += 1;
                    profile.lookahead.positive_candidates += stats.positive;
                    profile.lookahead.candidates_in_ram += stats.in_ram;
                    profile.lookahead.candidates_in_vram += stats.in_vram;
                    profile.lookahead.candidates_available += stats.available;
                    if stats.positive == 0 {
                        profile.lookahead.no_positive_score_calls += 1;
                    } else if candidates.is_empty() {
                        profile.lookahead.all_candidates_cached_calls += 1;
                    }
                    candidates
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            profile.lookahead.proposed += next_candidates.len() as u64;
            let speculative_attempts = next_candidates.len();

            // 4. Chuẩn bị danh sách Hit Experts sẵn sàng tính toán
            struct ReadyExpert {
                slot: usize,
                expert_id: usize,
                expert: Int8Expert,
                l2_hit: bool,
                hits_in_cache: u32,
                is_protected: bool,
            }

            let mut hit_ready = Vec::with_capacity(hit_slots.len());
            for &slot in &hit_slots {
                let expert_id = top_experts[slot];
                layer.clock = layer.clock.wrapping_add(1);
                if layer.clock % 65_536 == 0 {
                    layer.heat.iter_mut().for_each(|heat| *heat /= 2);
                }
                layer.heat[expert_id] = layer.heat[expert_id].saturating_add(1);
                layer.update_hot_pins();
                let mut empty_map = HashMap::new();
                let is_preloaded = preloaded.contains_key(&expert_id);
                let (expert, l2_hit, hits_in_cache, is_protected) = Self::acquire_expert_tiered(
                    id,
                    expert_id,
                    d,
                    intermediate_size,
                    weight_count,
                    scale_count,
                    index,
                    layer,
                    gpu_dense,
                    gpu_tier.is_enabled(),
                    victim_cache,
                    profile,
                    1,
                    if is_preloaded {
                        &mut preloaded
                    } else {
                        &mut empty_map
                    },
                )?;
                hit_ready.push(ReadyExpert {
                    slot,
                    expert_id,
                    expert,
                    l2_hit,
                    hits_in_cache,
                    is_protected,
                });
            }

            // 5. Coarse-Grained Parallel Multi-Expert Dispatching kết hợp Compute-I/O Scheduling:
            // - Rút sẵn các bộ đệm tái sử dụng (Zero-Allocation Recycled Buffers) từ pool
            // - Nhánh I/O Worker (Rayon): nạp miss_ids và next_candidates trực tiếp vào recycled buffers
            // - Nhánh Compute: Tính toán song song các Hit experts bằng forward_single_threaded (0 nested Rayon barriers!)
            let has_io_work = !miss_slots.is_empty() || !next_candidates.is_empty();

            let (curr_buffers, next_buffers) = if has_io_work {
                let mut cb = Vec::with_capacity(miss_ids.len());
                for _ in 0..miss_ids.len() {
                    let buf = layer
                        .recycled_buffers
                        .pop()
                        .unwrap_or_else(|| (vec![0u8; weight_count], vec![0.0f32; scale_count]));
                    cb.push(buf);
                }
                let mut nb = Vec::with_capacity(next_candidates.len());
                if let Some(next_l) = remaining.first_mut() {
                    for _ in 0..next_candidates.len() {
                        let buf = next_l.recycled_buffers.pop().unwrap_or_else(|| {
                            (vec![0u8; weight_count], vec![0.0f32; scale_count])
                        });
                        nb.push(buf);
                    }
                }
                (cb, nb)
            } else {
                (Vec::new(), Vec::new())
            };

            let next_buffer_bytes: usize = next_buffers.iter().map(buffer_bytes).sum();
            profile.lookahead.buffer_peak_bytes =
                profile.lookahead.buffer_peak_bytes.max(next_buffer_bytes);
            observe_decode_payload(
                profile,
                previous
                    .iter()
                    .chain(std::iter::once(&*layer))
                    .chain(remaining.iter()),
                hit_ready
                    .iter()
                    .map(|r| r.expert.allocated_bytes())
                    .sum::<usize>()
                    + preloaded
                        .values()
                        .map(Int8Expert::allocated_bytes)
                        .sum::<usize>()
                    + curr_buffers.iter().map(buffer_bytes).sum::<usize>()
                    + next_buffer_bytes,
            );

            if !hit_ready.is_empty() && has_io_work {
                let num_hits = hit_ready.len();
                let (hit_scratches, _) = scratch.expert_scratches[..top_k].split_at_mut(num_hits);
                let hidden = &scratch.hidden[..];

                let (io_res, compute_res) = rayon::join(
                    || {
                        let started = Instant::now();
                        let mut sorted_miss_ids = miss_ids.clone();
                        sorted_miss_ids.sort_unstable();
                        let curr_loaded: HashMap<usize, Int8Expert> = sorted_miss_ids
                            .into_par_iter()
                            .zip(curr_buffers.into_par_iter())
                            .filter_map(|(eid, (w, s))| {
                                Int8Expert::load_reusing(index, id, eid, d, intermediate_size, w, s)
                                    .ok()
                                    .map(|exp| (eid, exp))
                            })
                            .collect();

                        let next_loaded: HashMap<usize, Int8Expert> = if !next_candidates.is_empty()
                        {
                            next_candidates
                                .into_par_iter()
                                .zip(next_buffers.into_par_iter())
                                .filter_map(|(eid, (w, s))| {
                                    Int8Expert::load_reusing(
                                        index,
                                        id + 1,
                                        eid,
                                        d,
                                        intermediate_size,
                                        w,
                                        s,
                                    )
                                    .ok()
                                    .map(|exp| (eid, exp))
                                })
                                .collect()
                        } else {
                            HashMap::new()
                        };

                        (curr_loaded, next_loaded, started.elapsed())
                    },
                    || {
                        let started = Instant::now();
                        let res = hit_ready
                            .par_iter()
                            .zip(hit_scratches.par_iter_mut())
                            .try_for_each(|(ready, scratch_buf)| {
                                ready.expert.forward_single_threaded(hidden, scratch_buf)?;
                                Ok::<(), io::Error>(())
                            });
                        (res, started.elapsed())
                    },
                );

                record_lookahead_loads(profile, speculative_attempts, &io_res.1);
                compute_res.0?;
                profile.expert_compute += compute_res.1;
                profile.expert_io += io_res.2;
                profile.file_io += io_res.2;
                profile.file_loads += (io_res.0.len() + io_res.1.len()) as u64;
                profile.file_bytes += io_res
                    .0
                    .values()
                    .chain(io_res.1.values())
                    .map(|e| e.resident_bytes() as u64)
                    .sum::<u64>();
                preloaded.extend(io_res.0);
                lookahead_prefetched = io_res.1;
            } else if has_io_work {
                // hit_ready rỗng (toàn bộ là Miss)
                let started = Instant::now();
                let mut sorted_miss_ids = miss_ids.clone();
                sorted_miss_ids.sort_unstable();
                let curr_loaded: HashMap<usize, Int8Expert> = sorted_miss_ids
                    .into_par_iter()
                    .zip(curr_buffers.into_par_iter())
                    .filter_map(|(eid, (w, s))| {
                        Int8Expert::load_reusing(index, id, eid, d, intermediate_size, w, s)
                            .ok()
                            .map(|exp| (eid, exp))
                    })
                    .collect();

                let next_loaded: HashMap<usize, Int8Expert> = if !next_candidates.is_empty() {
                    next_candidates
                        .into_par_iter()
                        .zip(next_buffers.into_par_iter())
                        .filter_map(|(eid, (w, s))| {
                            Int8Expert::load_reusing(index, id + 1, eid, d, intermediate_size, w, s)
                                .ok()
                                .map(|exp| (eid, exp))
                        })
                        .collect()
                } else {
                    HashMap::new()
                };

                let elapsed = started.elapsed();
                record_lookahead_loads(profile, speculative_attempts, &next_loaded);
                profile.expert_io += elapsed;
                profile.file_io += elapsed;
                profile.file_loads += (curr_loaded.len() + next_loaded.len()) as u64;
                profile.file_bytes += curr_loaded
                    .values()
                    .chain(next_loaded.values())
                    .map(|e| e.resident_bytes() as u64)
                    .sum::<u64>();
                preloaded.extend(curr_loaded);
                lookahead_prefetched = next_loaded;
            } else {
                // has_io_work là false: 100% Cache Hit, tính song song toàn bộ hit_ready
                let num_hits = hit_ready.len();
                let (hit_scratches, _) = scratch.expert_scratches[..top_k].split_at_mut(num_hits);
                let hidden = &scratch.hidden[..];
                let started = Instant::now();
                hit_ready
                    .par_iter()
                    .zip(hit_scratches.par_iter_mut())
                    .try_for_each(|(ready, scratch_buf)| {
                        ready.expert.forward_single_threaded(hidden, scratch_buf)?;
                        Ok::<(), io::Error>(())
                    })?;
                profile.expert_compute += started.elapsed();
            }

            // 6. Tính toán song song các Miss Experts vừa được nạp từ SSD vào preloaded
            let mut miss_ready = Vec::with_capacity(miss_slots.len());
            for &slot in &miss_slots {
                let expert_id = top_experts[slot];
                layer.clock = layer.clock.wrapping_add(1);
                if layer.clock % 65_536 == 0 {
                    layer.heat.iter_mut().for_each(|heat| *heat /= 2);
                }
                layer.heat[expert_id] = layer.heat[expert_id].saturating_add(1);
                layer.update_hot_pins();
                let (expert, l2_hit, hits_in_cache, is_protected) = Self::acquire_expert_tiered(
                    id,
                    expert_id,
                    d,
                    intermediate_size,
                    weight_count,
                    scale_count,
                    index,
                    layer,
                    gpu_dense,
                    gpu_tier.is_enabled(),
                    victim_cache,
                    profile,
                    1,
                    &mut preloaded,
                )?;
                miss_ready.push(ReadyExpert {
                    slot,
                    expert_id,
                    expert,
                    l2_hit,
                    hits_in_cache,
                    is_protected,
                });
            }

            observe_decode_payload(
                profile,
                previous
                    .iter()
                    .chain(std::iter::once(&*layer))
                    .chain(remaining.iter()),
                hit_ready
                    .iter()
                    .chain(miss_ready.iter())
                    .map(|r| r.expert.allocated_bytes())
                    .sum::<usize>()
                    + preloaded
                        .values()
                        .map(Int8Expert::allocated_bytes)
                        .sum::<usize>()
                    + lookahead_prefetched
                        .values()
                        .map(Int8Expert::allocated_bytes)
                        .sum::<usize>(),
            );

            if !miss_ready.is_empty() {
                let num_hits = hit_ready.len();
                let (_, miss_scratches) = scratch.expert_scratches[..top_k].split_at_mut(num_hits);
                let hidden = &scratch.hidden[..];
                let started = Instant::now();
                miss_ready
                    .par_iter()
                    .zip(miss_scratches.par_iter_mut())
                    .try_for_each(|(ready, scratch_buf)| {
                        ready.expert.forward_single_threaded(hidden, scratch_buf)?;
                        Ok::<(), io::Error>(())
                    })?;
                profile.expert_compute += started.elapsed();
            }

            // 7. Deterministic Fused Reduction vào scratch.x theo đúng thứ tự Top-K ban đầu
            let mut slot_to_scratch = [0usize; 64];
            for (i, item) in hit_ready.iter().enumerate() {
                if item.slot < 64 {
                    slot_to_scratch[item.slot] = i;
                }
            }
            let num_hits = hit_ready.len();
            for (j, item) in miss_ready.iter().enumerate() {
                if item.slot < 64 {
                    slot_to_scratch[item.slot] = num_hits + j;
                }
            }

            for (slot, &expert_id) in top_experts.iter().enumerate() {
                let scratch_idx = slot_to_scratch[slot];
                let weight = scratch.router[expert_id] / normalizer;
                let out = &scratch.expert_scratches[scratch_idx].output;
                for (a, b) in scratch.x.iter_mut().zip(out.iter().copied()) {
                    *a += weight * b;
                }
            }

            // 8. Hoàn trả và nạp lại Cache L2: Ưu tiên trả Hit entries trước, sau đó mới nạp Miss entries
            for item in hit_ready {
                Self::admit_expert_l2(
                    layer,
                    item.expert_id,
                    item.expert,
                    item.l2_hit,
                    item.hits_in_cache,
                    item.is_protected,
                    cache_bytes_per_layer,
                    profile,
                    gpu_dense,
                    gpu_tier.is_enabled(),
                    victim_cache,
                    id,
                );
            }

            for item in miss_ready {
                Self::admit_expert_l2(
                    layer,
                    item.expert_id,
                    item.expert,
                    item.l2_hit,
                    item.hits_in_cache,
                    item.is_protected,
                    cache_bytes_per_layer,
                    profile,
                    gpu_dense,
                    gpu_tier.is_enabled(),
                    victim_cache,
                    id,
                );
            }
            finite(&scratch.x)?;
        }
        let head_started = Instant::now();
        rms_norm_into(&scratch.x, &self.norm, c.rms_norm_eps, &mut scratch.hidden)?;
        Self::dense_gemv(
            gpu_dense,
            gpu_tier.is_enabled(),
            &self.head,
            |g| &g.lm_head,
            &scratch.hidden,
            &mut scratch.logits,
        )?;
        profile.lm_head += head_started.elapsed();
        Ok(())
    }

    /// Layer-major causal prefill. Expert work is grouped by expert within each
    /// layer, so one load serves every routed prompt token in that layer.
    fn prefill(
        &mut self,
        tokens: &[usize],
        kv: &mut [Kv],
        profile: &mut RequestProfile,
        scratch: &mut StepScratch,
        check: &dyn Fn() -> io::Result<()>,
    ) -> io::Result<()> {
        let victim_cache = self.victim_expert_cache;
        let c = &self.config;
        let d = c.hidden_size;
        let hd = d / c.num_attention_heads;
        let kd = hd * c.num_key_value_heads;
        let batch_enabled = self.expert_batch_prefill_enabled && tokens.len() > 1;
        let mut batch2 = batch_enabled.then(|| BatchScratch::<2>::new(d, c.intermediate_size));
        let mut batch4 = batch_enabled.then(|| BatchScratch::<4>::new(d, c.intermediate_size));
        let batch_scratch_bytes = batch2.as_ref().map_or(0, BatchScratch::allocated_bytes)
            + batch4.as_ref().map_or(0, BatchScratch::allocated_bytes);
        let mut batch = vec![0.0; tokens.len() * d];
        for (&token, output) in tokens.iter().zip(batch.chunks_exact_mut(d)) {
            if token >= c.vocab_size {
                return Err(invalid("token ID exceeds model vocabulary"));
            }
            output.copy_from_slice(&self.embedding.values[token * d..(token + 1) * d]);
        }
        let mut hidden_batch = vec![0.0; batch.len()];
        let batch_size = tokens.len();
        let mut all_q = vec![0.0f32; batch_size * d];
        let mut all_k = vec![0.0f32; batch_size * kd];
        let mut all_v = vec![0.0f32; batch_size * kd];
        let mut norm_batch = vec![0.0f32; batch_size * d];
        let mut context_batch = vec![0.0f32; batch_size * d];
        let mut projected_batch = vec![0.0f32; batch_size * d];

        let route_count = tokens
            .len()
            .checked_mul(c.num_experts_per_tok)
            .ok_or_else(|| invalid("prefill route count overflow"))?;
        let contribution_count = route_count
            .checked_mul(d)
            .ok_or_else(|| invalid("prefill contribution size overflow"))?;
        let mut contributions = vec![0.0; contribution_count];
        // The grouped candidate remains test-only: paired measurements on i5
        // found it slower than the serial token dispatch used in production.
        #[cfg(not(test))]
        let dispatch_width = 1;
        #[cfg(test)]
        let dispatch_width = if self.serial_prefill_reference {
            1
        } else {
            scratch.expert_scratches.len()
        };
        let mut expert_uses: Vec<Vec<(usize, usize, f32)>> = (0..c.num_experts)
            .map(|_| Vec::with_capacity(tokens.len()))
            .collect();

        if profile.memory_trace.is_some() {
            let prefill = [
                &batch,
                &hidden_batch,
                &all_q,
                &all_k,
                &all_v,
                &norm_batch,
                &context_batch,
                &projected_batch,
                &contributions,
            ]
            .iter()
            .map(|v| v.capacity() * 4)
            .sum::<usize>()
                + expert_uses.capacity() * std::mem::size_of::<Vec<(usize, usize, f32)>>()
                + expert_uses
                    .iter()
                    .map(|v| v.capacity() * std::mem::size_of::<(usize, usize, f32)>())
                    .sum::<usize>()
                + batch_scratch_bytes;
            self.record_memory(
                profile,
                "prefill_buffers_live",
                ActiveMemory {
                    kv: kv_capacity_bytes(kv, profile.active_kv_container_bytes),
                    scratch: scratch.capacity_bytes(),
                    prefill,
                },
            );
        }

        let intermediate_size = c.intermediate_size;
        let weight_count = d * intermediate_size * 3;
        let scale_count = intermediate_size * 2 + d;

        let (layers, gpu_dense, gpu_tier, index, cache_bytes_per_layer, route_trace) = (
            &mut self.layers,
            &mut self.gpu_dense,
            &self.gpu_tier,
            &self.index,
            self.cache_bytes_per_layer,
            &mut self.route_trace,
        );

        for (layer_id, layer) in layers.iter_mut().enumerate() {
            check()?;
            let base_position = kv[layer_id].positions(kd);
            if base_position
                .checked_add(tokens.len())
                .is_none_or(|length| length > self.context_limit)
            {
                return Err(invalid("context exhausted"));
            }
            for list in &mut expert_uses {
                list.clear();
            }

            let attention_started = Instant::now();

            // 1. Chuẩn hóa RMSNorm cho toàn bộ batch đầu vào
            for (x_chunk, norm_chunk) in batch.chunks_exact(d).zip(norm_batch.chunks_exact_mut(d)) {
                rms_norm_into(x_chunk, &layer.input_norm, c.rms_norm_eps, norm_chunk)?;
            }

            // 2. Batched QKV cuBLAS SGEMM trên GPU (hoặc tuần tự trên CPU fallback)
            Self::dense_gemm_qkv(
                &*gpu_dense,
                gpu_tier.is_enabled(),
                layer_id,
                layer,
                &norm_batch,
                batch_size,
                &mut all_q,
                &mut all_k,
                &mut all_v,
            )?;

            // 3. RoPE, KV cache append & Causal Attention cho từng token
            for local_position in 0..batch_size {
                check()?;
                let position = base_position + local_position;
                let q_slice = &mut all_q[local_position * d..(local_position + 1) * d];
                let k_slice = &mut all_k[local_position * kd..(local_position + 1) * kd];
                let v_slice = &all_v[local_position * kd..(local_position + 1) * kd];

                rms_norm_in_place(q_slice, &layer.q_norm, c.rms_norm_eps)?;
                rms_norm_in_place(k_slice, &layer.k_norm, c.rms_norm_eps)?;
                prepare_rope_sincos(&mut scratch.sincos, &scratch.rope_bases, position);
                apply_rope_cached(q_slice, hd, &scratch.sincos);
                apply_rope_cached(k_slice, hd, &scratch.sincos);
                let ctx_slice = &mut context_batch[local_position * d..(local_position + 1) * d];
                if kv[layer_id].is_fp8 {
                    let k_scale = quantize_row_fp8(k_slice, &mut scratch.fp8_key);
                    let v_scale = quantize_row_fp8(v_slice, &mut scratch.fp8_value);
                    kv[layer_id].fp8_keys.extend_from_slice(&scratch.fp8_key);
                    kv[layer_id].fp8_key_scales.push(k_scale);
                    kv[layer_id]
                        .fp8_values
                        .extend_from_slice(&scratch.fp8_value);
                    kv[layer_id].fp8_value_scales.push(v_scale);
                    causal_attention_fp8_into(
                        q_slice,
                        &kv[layer_id].fp8_keys,
                        &kv[layer_id].fp8_key_scales,
                        &kv[layer_id].fp8_values,
                        &kv[layer_id].fp8_value_scales,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        ctx_slice,
                    )?;
                } else {
                    kv[layer_id].keys.extend_from_slice(k_slice);
                    kv[layer_id].values.extend_from_slice(v_slice);
                    causal_attention_into(
                        q_slice,
                        &kv[layer_id].keys,
                        &kv[layer_id].values,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        ctx_slice,
                    )?;
                }
            }

            // 4. Batched Output Projection cuBLAS SGEMM trên GPU
            Self::dense_gemm(
                &*gpu_dense,
                gpu_tier.is_enabled(),
                &layer.o,
                |g| &g.layers[layer_id].o_proj,
                &context_batch,
                &mut projected_batch,
                batch_size,
            )?;

            // 5. Cộng Residual Attention vào batch
            for (value, projected) in batch.iter_mut().zip(&projected_batch) {
                *value += projected;
            }
            profile.attention += attention_started.elapsed();

            // 6. Router và phân bổ MoE experts
            for local_position in 0..batch_size {
                check()?;
                scratch
                    .x
                    .copy_from_slice(&batch[local_position * d..(local_position + 1) * d]);
                rms_norm_into(
                    &scratch.x,
                    &layer.post_norm,
                    c.rms_norm_eps,
                    &mut scratch.hidden,
                )?;
                hidden_batch[local_position * d..(local_position + 1) * d]
                    .copy_from_slice(&scratch.hidden);
                layer
                    .router
                    .multiply_loaded_into(&scratch.hidden, &mut scratch.router)?;
                softmax(&mut scratch.router)?;

                let mut expert_indices = [0usize; 64];
                for (i, idx) in expert_indices.iter_mut().enumerate().take(c.num_experts) {
                    *idx = i;
                }
                let valid_indices = &mut expert_indices[..c.num_experts];
                let top_k = c.num_experts_per_tok;
                if top_k < valid_indices.len() {
                    valid_indices.select_nth_unstable_by(top_k, |&left, &right| {
                        scratch.router[right]
                            .total_cmp(&scratch.router[left])
                            .then_with(|| left.cmp(&right))
                    });
                }
                let top_experts = &mut valid_indices[..top_k];
                top_experts.sort_by(|&left, &right| {
                    scratch.router[right]
                        .total_cmp(&scratch.router[left])
                        .then_with(|| left.cmp(&right))
                });
                let normalizer = if c.norm_topk_prob {
                    top_experts
                        .iter()
                        .map(|&expert| scratch.router[expert])
                        .sum()
                } else {
                    1.0
                };
                for (slot, &expert) in top_experts.iter().enumerate() {
                    expert_uses[expert].push((
                        local_position,
                        slot,
                        scratch.router[expert] / normalizer,
                    ));
                }
                if let Some(tracer) = route_trace.as_mut() {
                    let gates: Vec<f32> = top_experts
                        .iter()
                        .map(|&expert| scratch.router[expert] / normalizer)
                        .collect();
                    tracer.trace(local_position, layer_id, top_experts, &gates);
                }
            }

            contributions.fill(0.0);
            let active_expert_ids: Vec<usize> = expert_uses
                .iter()
                .enumerate()
                .take(c.num_experts)
                .filter_map(|(eid, uses)| if !uses.is_empty() { Some(eid) } else { None })
                .collect();

            let (mut preloaded, prefetch_io) = Self::prefetch_missing_experts(
                layer_id,
                &active_expert_ids,
                d,
                intermediate_size,
                index,
                layer,
                gpu_dense.as_ref(),
                gpu_tier.is_enabled(),
                c.num_experts_per_tok,
            );
            profile.expert_io += prefetch_io;
            profile.file_io += prefetch_io;
            profile.file_loads += preloaded.len() as u64;
            profile.file_bytes += preloaded
                .values()
                .map(|e| e.resident_bytes() as u64)
                .sum::<u64>();

            for (expert_id, uses) in expert_uses.iter().enumerate().take(c.num_experts) {
                if uses.is_empty() {
                    continue;
                }
                check()?;
                let uses_u32 = u32::try_from(uses.len()).unwrap_or(u32::MAX);
                layer.clock = layer.clock.wrapping_add(uses.len() as u64);
                if layer.clock / 65_536 != layer.clock.saturating_sub(uses.len() as u64) / 65_536 {
                    layer.heat.iter_mut().for_each(|heat| *heat /= 2);
                }
                layer.heat[expert_id] = layer.heat[expert_id].saturating_add(uses_u32);
                layer.update_hot_pins();
                let (expert, l2_hit, hits_in_cache, is_protected) = Self::acquire_expert_tiered(
                    layer_id,
                    expert_id,
                    d,
                    intermediate_size,
                    weight_count,
                    scale_count,
                    index,
                    layer,
                    gpu_dense,
                    gpu_tier.is_enabled(),
                    victim_cache,
                    profile,
                    uses.len() as u64,
                    &mut preloaded,
                )?;

                // Dispatch tokens sharing this expert together. Each task owns a scratch
                // slot and uses the serial kernel, avoiding nested row-level fork/joins.
                // Copy results back in route order before the deterministic Top-K reduction.
                let mut use_offset = 0;
                while use_offset < uses.len() {
                    let remaining = uses.len() - use_offset;
                    let width = if batch_enabled {
                        if remaining >= 4 {
                            4
                        } else if remaining >= 2 {
                            2
                        } else {
                            1
                        }
                    } else {
                        dispatch_width.min(remaining)
                    };
                    let routes = &uses[use_offset..use_offset + width];
                    use_offset += width;
                    check()?;
                    let expert_started = Instant::now();
                    if routes.len() == 1 {
                        let (token, slot, weight) = routes[0];
                        let result = expert.forward_reuse(
                            &hidden_batch[token * d..(token + 1) * d],
                            &mut scratch.expert,
                        )?;
                        profile.expert_compute += expert_started.elapsed();
                        let contribution = (token * c.num_experts_per_tok + slot) * d;
                        for (target, value) in contributions[contribution..contribution + d]
                            .iter_mut()
                            .zip(result)
                        {
                            *target = weight * value;
                        }
                    } else if batch_enabled {
                        let outputs: &[Vec<f32>] = if routes.len() == 2 {
                            let buffer = batch2.as_mut().expect("batch2 prepared");
                            let inputs = std::array::from_fn(|t| {
                                let token = routes[t].0;
                                &hidden_batch[token * d..(token + 1) * d]
                            });
                            expert.forward_batch_prototype(inputs, buffer)?;
                            profile.prefill_expert_batch2_calls += 1;
                            &buffer.outputs
                        } else {
                            let buffer = batch4.as_mut().expect("batch4 prepared");
                            let inputs = std::array::from_fn(|t| {
                                let token = routes[t].0;
                                &hidden_batch[token * d..(token + 1) * d]
                            });
                            expert.forward_batch_prototype(inputs, buffer)?;
                            profile.prefill_expert_batch4_calls += 1;
                            &buffer.outputs
                        };
                        profile.expert_compute += expert_started.elapsed();
                        check()?;
                        for (&(token, slot, weight), output) in routes.iter().zip(outputs) {
                            let contribution = (token * c.num_experts_per_tok + slot) * d;
                            for (target, value) in contributions[contribution..contribution + d]
                                .iter_mut()
                                .zip(output)
                            {
                                *target = weight * value;
                            }
                        }
                    } else {
                        scratch.expert_scratches[..routes.len()]
                            .par_iter_mut()
                            .zip(routes.par_iter())
                            .try_for_each(|(buffer, &(token, _, _))| {
                                expert.forward_single_threaded(
                                    &hidden_batch[token * d..(token + 1) * d],
                                    buffer,
                                )?;
                                Ok::<(), io::Error>(())
                            })?;
                        profile.expert_compute += expert_started.elapsed();
                        check()?;
                        for (&(token, slot, weight), buffer) in
                            routes.iter().zip(&scratch.expert_scratches)
                        {
                            let contribution = (token * c.num_experts_per_tok + slot) * d;
                            for (target, value) in contributions[contribution..contribution + d]
                                .iter_mut()
                                .zip(&buffer.output)
                            {
                                *target = weight * value;
                            }
                        }
                    }
                }

                Self::admit_expert_l2(
                    layer,
                    expert_id,
                    expert,
                    l2_hit,
                    hits_in_cache,
                    is_protected,
                    cache_bytes_per_layer,
                    profile,
                    gpu_dense,
                    gpu_tier.is_enabled(),
                    victim_cache,
                    layer_id,
                );
            }
            for token in 0..tokens.len() {
                for slot in 0..c.num_experts_per_tok {
                    let contribution = (token * c.num_experts_per_tok + slot) * d;
                    for (value, addition) in batch[token * d..(token + 1) * d]
                        .iter_mut()
                        .zip(&contributions[contribution..contribution + d])
                    {
                        *value += addition;
                    }
                }
                finite(&batch[token * d..(token + 1) * d])?;
            }
        }

        scratch
            .x
            .copy_from_slice(&batch[(tokens.len() - 1) * d..tokens.len() * d]);
        let head_started = Instant::now();
        rms_norm_into(&scratch.x, &self.norm, c.rms_norm_eps, &mut scratch.hidden)?;
        Self::dense_gemv(
            gpu_dense,
            gpu_tier.is_enabled(),
            &self.head,
            |g| &g.lm_head,
            &scratch.hidden,
            &mut scratch.logits,
        )?;
        profile.lm_head += head_started.elapsed();
        Ok(())
    }

    /// Logits at every prefix, for numerical comparison with a reference decoder.
    pub fn logits(&mut self, tokens: &[usize]) -> io::Result<Vec<Vec<f32>>> {
        if tokens.is_empty() || tokens.len() > self.context_limit {
            return Err(invalid("invalid token count"));
        }
        let d = self.config.hidden_size;
        let kd = d / self.config.num_attention_heads * self.config.num_key_value_heads;
        let mut kv: Vec<_> = (0..self.layers.len())
            .map(|_| Kv::with_capacity_precision(self.context_limit, kd, self.kv_fp8))
            .collect();
        let mut profile = RequestProfile::default();
        let mut scratch = StepScratch::new(&self.config);
        scratch.expert.gate_up_mode = self.gate_up_mode;
        for expert_scratch in &mut scratch.expert_scratches {
            expert_scratch.gate_up_mode = self.gate_up_mode;
        }
        let mut logits = Vec::with_capacity(tokens.len());
        for &token in tokens {
            self.step(token, &mut kv, &mut profile, &mut scratch, &|| Ok(()))?;
            logits.push(scratch.logits.clone());
        }
        if let Some(tracer) = self.route_trace.as_mut() {
            tracer.advance_call();
        }
        Ok(logits)
    }

    /// Mã hóa chuỗi văn bản thành danh sách token ID sử dụng tokenizer của mô hình.
    pub fn encode(&self, text: &str) -> io::Result<Vec<usize>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| invalid("missing tokenizer.json"))?;
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&t| t as usize).collect())
    }

    /// Giải mã danh sách token ID thành chuỗi văn bản.
    pub fn decode(&self, tokens: &[usize]) -> io::Result<String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| invalid("missing tokenizer.json"))?;
        let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        tokenizer
            .decode(&ids, true)
            .map_err(|e| invalid(e.to_string()))
    }

    /// Tính toán độ phức tạp ngôn ngữ (Perplexity & Cross-Entropy Loss) trên một chuỗi văn bản.
    /// Áp dụng công thức học thuật chuẩn:
    /// NLL = ln(sum(exp(z_i))) - z_target
    /// Loss = mean(NLL)
    /// Perplexity = exp(Loss)
    pub fn compute_perplexity(&mut self, text: &str) -> io::Result<PerplexityMetrics> {
        let tokens = self.encode(text)?;
        if tokens.len() < 2 {
            return Err(invalid("perplexity requires at least 2 tokens"));
        }
        if tokens.len() > self.context_limit {
            return Err(invalid(format!(
                "token count {} exceeds context limit {}",
                tokens.len(),
                self.context_limit
            )));
        }

        let start_time = Instant::now();
        let all_logits = self.logits(&tokens)?;
        let eval_duration = start_time.elapsed();

        // Ở mỗi vị trí t từ 0 đến N-2: all_logits[t] là phân phối dự đoán cho token mục tiêu tokens[t+1]
        let num_eval = tokens.len() - 1;
        let mut total_nll = 0.0f64;

        for (t, &target_token) in tokens[1..].iter().enumerate() {
            let logits_t = &all_logits[t];
            if target_token >= logits_t.len() {
                return Err(invalid("target token index exceeds vocabulary size"));
            }

            // Log-Sum-Exp ổn định số học: max + ln(sum(exp(x - max)))
            let max_logit = logits_t.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum_exp: f64 = logits_t
                .iter()
                .map(|&v| ((v as f64) - max_logit).exp())
                .sum();
            let log_sum_exp = max_logit + sum_exp.ln();
            let target_logit = logits_t[target_token] as f64;

            let nll = log_sum_exp - target_logit;
            total_nll += nll;
        }

        let cross_entropy_loss = total_nll / (num_eval as f64);
        let perplexity = cross_entropy_loss.exp();
        let char_count = text.chars().count().max(1);
        let bits_per_character = (total_nll / std::f64::consts::LN_2) / (char_count as f64);
        let eval_time_ms = eval_duration.as_secs_f64() * 1000.0;
        let tokens_per_second = if eval_duration.as_secs_f64() > 0.0 {
            (tokens.len() as f64) / eval_duration.as_secs_f64()
        } else {
            0.0
        };

        Ok(PerplexityMetrics {
            token_count: tokens.len(),
            evaluated_tokens: num_eval,
            cross_entropy_loss,
            perplexity,
            bits_per_character,
            eval_time_ms,
            tokens_per_second,
        })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> io::Result<Generation> {
        self.generate_controlled(prompt, max_tokens, None, &|| Ok(()), || Ok(()), None)
    }

    /// Lưu trạng thái KV cache của một session xuống tệp nhị phân .moekv
    /// Áp dụng cơ chế ghi nguyên tử (Atomic file write qua tệp tạm .tmp và rename)
    pub(crate) fn save_session_to_file(
        &self,
        path: &Path,
        tokens: &[usize],
        kv: &[Kv],
    ) -> io::Result<()> {
        if kv.is_empty() {
            return Ok(());
        }
        let is_fp8 = kv[0].is_fp8;
        let c = &self.config;
        let hd = c.hidden_size / c.num_attention_heads;
        let kd = hd * c.num_key_value_heads;
        let num_layers = kv.len();
        let num_tokens = kv[0].positions(kd).min(tokens.len());
        if num_tokens == 0 {
            return Ok(());
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp_path = path.with_extension("tmp");
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);

        let magic = if is_fp8 {
            MOE_KV_MAGIC_FP8
        } else {
            MOE_KV_MAGIC_FP32
        };
        writer.write_all(magic)?;
        writer.write_all(&MOE_KV_VERSION.to_le_bytes())?;
        writer.write_all(&(num_layers as u32).to_le_bytes())?;
        writer.write_all(&(kd as u32).to_le_bytes())?;
        writer.write_all(&(if is_fp8 { 1u32 } else { 0u32 }).to_le_bytes())?;
        writer.write_all(&(num_tokens as u32).to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?; // reserved

        for (p, &tok) in tokens.iter().enumerate().take(num_tokens) {
            let tok = tok as u32;
            writer.write_all(&tok.to_le_bytes())?;
            for layer in kv {
                if is_fp8 {
                    let k_slice = &layer.fp8_keys[p * kd..(p + 1) * kd];
                    let k_scale = layer.fp8_key_scales[p];
                    let v_slice = &layer.fp8_values[p * kd..(p + 1) * kd];
                    let v_scale = layer.fp8_value_scales[p];
                    writer.write_all(k_slice)?;
                    writer.write_all(&k_scale.to_le_bytes())?;
                    writer.write_all(v_slice)?;
                    writer.write_all(&v_scale.to_le_bytes())?;
                } else {
                    let k_slice = &layer.keys[p * kd..(p + 1) * kd];
                    let v_slice = &layer.values[p * kd..(p + 1) * kd];
                    for &k in k_slice {
                        writer.write_all(&k.to_le_bytes())?;
                    }
                    for &v in v_slice {
                        writer.write_all(&v.to_le_bytes())?;
                    }
                }
            }
        }
        writer.flush()?;
        let underlying = writer
            .into_inner()
            .map_err(|e| io::Error::other(e.to_string()))?;
        underlying.sync_all()?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Nạp trạng thái KV cache đã lưu từ tệp .moekv trên đĩa
    /// Tự động so khớp tiền tố token IDs, khôi phục trực tiếp KV Cache trong vài mili-giây
    pub(crate) fn load_session_from_file(
        &self,
        path: &Path,
        tokens: &[usize],
        kd: usize,
    ) -> io::Result<Option<(usize, Vec<Kv>)>> {
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut header = [0u8; 32];
        if reader.read_exact(&mut header).is_err() {
            return Ok(None);
        }

        let magic = &header[0..8];
        let is_fp8 = if magic == MOE_KV_MAGIC_FP8 {
            true
        } else if magic == MOE_KV_MAGIC_FP32 {
            false
        } else {
            return Ok(None);
        };

        let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
        let file_layers = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let file_kd = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
        let file_tokens = u32::from_le_bytes(header[24..28].try_into().unwrap()) as usize;

        if version != MOE_KV_VERSION
            || file_layers != self.layers.len()
            || file_kd != kd
            || file_tokens == 0
        {
            return Ok(None);
        }

        let rec_bytes = if is_fp8 {
            4 + file_layers * (kd * 2 + 8)
        } else {
            4 + file_layers * (kd * 2 * 4)
        };

        let mut kv: Vec<Kv> = (0..file_layers)
            .map(|_| Kv::with_capacity_precision(self.context_limit, kd, is_fp8))
            .collect();

        let mut matched = 0usize;
        let mut scratch_record = vec![0u8; rec_bytes];

        for p in 0..file_tokens {
            if reader.read_exact(&mut scratch_record).is_err() {
                break;
            }
            let tok_id = u32::from_le_bytes(scratch_record[0..4].try_into().unwrap()) as usize;
            let expected_tok = tokens.get(p).copied();
            if Some(tok_id) == expected_tok && matched == p {
                matched += 1;
                let mut offset = 4;
                for layer in kv.iter_mut() {
                    if is_fp8 {
                        let k_slice = &scratch_record[offset..offset + kd];
                        offset += kd;
                        let k_scale = f32::from_le_bytes(
                            scratch_record[offset..offset + 4].try_into().unwrap(),
                        );
                        offset += 4;
                        let v_slice = &scratch_record[offset..offset + kd];
                        offset += kd;
                        let v_scale = f32::from_le_bytes(
                            scratch_record[offset..offset + 4].try_into().unwrap(),
                        );
                        offset += 4;

                        layer.fp8_keys.extend_from_slice(k_slice);
                        layer.fp8_key_scales.push(k_scale);
                        layer.fp8_values.extend_from_slice(v_slice);
                        layer.fp8_value_scales.push(v_scale);
                    } else {
                        for _ in 0..kd {
                            let val = f32::from_le_bytes(
                                scratch_record[offset..offset + 4].try_into().unwrap(),
                            );
                            offset += 4;
                            layer.keys.push(val);
                        }
                        for _ in 0..kd {
                            let val = f32::from_le_bytes(
                                scratch_record[offset..offset + 4].try_into().unwrap(),
                            );
                            offset += 4;
                            layer.values.push(val);
                        }
                    }
                }
            } else {
                break;
            }
        }

        // Causal KV remains valid up to the first changed token. Recompute at
        // least the final prompt token because the cache does not retain logits.
        let prefix = matched.min(tokens.len().saturating_sub(1));
        if prefix > 0 {
            for cache in &mut kv {
                cache.truncate_positions(prefix, kd);
            }
            Ok(Some((prefix, kv)))
        } else {
            Ok(None)
        }
    }

    fn take_prefix_kv(
        &mut self,
        tokens: &[usize],
        session_id: Option<&str>,
        kd: usize,
        max_tokens: usize,
        reuse: &mut KvBufferReuseMetrics,
    ) -> (usize, Vec<Kv>, Option<PreparedSession>) {
        *reuse = KvBufferReuseMetrics {
            enabled: self.kv_buffer_reuse_enabled,
            ..Default::default()
        };
        if let Some(mut state) = session_id.and_then(|id| self.sessions.remove(id)) {
            // Causal KV remains valid up to the first changed token. Recompute at
            // least the final prompt token because the cache does not retain logits.
            let prefix = tokens
                .iter()
                .zip(&state.tokens)
                .take_while(|(a, b)| a == b)
                .count()
                .min(tokens.len() - 1);
            #[cfg(test)]
            let prefix = if self.strict_session_prefix_reference {
                if state.tokens.len() < tokens.len() && tokens.starts_with(&state.tokens) {
                    state.tokens.len()
                } else {
                    0
                }
            } else {
                prefix
            };
            if prefix > 0 {
                for cache in &mut state.kv {
                    cache.truncate_positions(prefix, kd);
                }
                return (prefix, state.kv, None);
            }
            if self.kv_buffer_reuse_enabled {
                for cache in &mut state.kv {
                    cache.truncate_positions(0, kd);
                }
                reuse.reused_capacity_bytes =
                    kv_capacity_bytes(&state.kv, state.kv.capacity() * std::mem::size_of::<Kv>());
                reuse.source = Some("same_session_reset");
                return (0, state.kv, None);
            }
        }
        if self.kv_buffer_reuse_enabled {
            if let Some(id) = session_id {
                if let Some((kv, prepared)) =
                    self.take_evicted_kv(tokens, id, kd, max_tokens, reuse)
                {
                    return (0, kv, Some(prepared));
                }
            }
        }

        // Nếu session không có trên RAM, tự động tra cứu và khôi phục từ tệp .moekv trên NVMe disk
        if let Some(id) = session_id {
            if let Some(dir) = &self.session_dir {
                let file_path = dir.join(format!("{id}.moekv"));
                if let Ok(Some((prefix, kv))) = self.load_session_from_file(&file_path, tokens, kd)
                {
                    if prefix > 0 && kv.iter().all(|cache| cache.is_fp8 == self.kv_fp8) {
                        return (prefix, kv, None);
                    }
                }
            }
        }

        // Allocate only after lookup; a reusable session already owns its buffers.
        (
            0,
            (0..self.layers.len())
                .map(|_| Kv::with_capacity_precision(self.context_limit, kd, self.kv_fp8))
                .collect(),
            None,
        )
    }

    fn take_evicted_kv(
        &mut self,
        tokens: &[usize],
        id: &str,
        kd: usize,
        max_tokens: usize,
        reuse: &mut KvBufferReuseMetrics,
    ) -> Option<(Vec<Kv>, PreparedSession)> {
        if self.kv_fp8 {
            return None;
        }
        let slots = self.context_limit.checked_mul(kd)?;
        let minimum_kv = self.layers.len().checked_mul(
            slots
                .checked_mul(8)?
                .checked_add(std::mem::size_of::<Kv>())?,
        )?;
        // Even a minimal final state would force eviction. Do not evict for a
        // sessionless request, an oversized state, or merely speculative growth.
        let minimum_state = minimum_kv
            .checked_add(std::mem::size_of::<SessionState>())?
            .checked_add(tokens.len().checked_mul(std::mem::size_of::<usize>())?)?
            .checked_add(id.len())?;
        let retained = self.sessions.iter().fold(0usize, |total, (key, state)| {
            total
                .saturating_add(key.capacity())
                .saturating_add(session_bytes(state))
        });
        if minimum_state > self.session_cache_bytes
            || retained.saturating_add(minimum_state) <= self.session_cache_bytes
        {
            return None;
        }
        let (old_id, donor) = self
            .sessions
            .iter()
            .min_by_key(|(_, state)| state.last_used)?;
        if donor.kv.len() != self.layers.len()
            || donor
                .kv
                .iter()
                .any(|kv| kv.keys.capacity() < slots || kv.values.capacity() < slots)
        {
            return None;
        }
        let donor_bytes =
            kv_capacity_bytes(&donor.kv, donor.kv.capacity() * std::mem::size_of::<Kv>());
        let old_bytes = old_id.capacity().saturating_add(session_bytes(donor));
        let old_id = old_id.clone();
        // Reserve the token buffer once so admission size is exact before the
        // destructive handoff, even if decoding reaches its token limit.
        let mut prepared = PreparedSession {
            id: id.to_owned(),
            tokens: tokens.to_vec(),
        };
        prepared.tokens.reserve_exact(max_tokens.saturating_sub(1));
        let incoming = donor_bytes
            .saturating_add(std::mem::size_of::<SessionState>())
            .saturating_add(prepared.id.capacity())
            .saturating_add(
                prepared
                    .tokens
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>()),
            );
        if retained.saturating_sub(old_bytes).saturating_add(incoming) > self.session_cache_bytes {
            return None;
        }
        // Exclusive model ownership; the entry cannot disappear between lookup
        // and removal. Remove the old ID before any buffer can be overwritten.
        let mut donor = self.sessions.remove(&old_id)?;
        for kv in &mut donor.kv {
            kv.keys.clear();
            kv.values.clear();
        }
        reuse.reused_capacity_bytes = donor_bytes;
        reuse.early_evicted_sessions = 1;
        reuse.source = Some("evicted_session");
        Some((donor.kv, prepared))
    }

    pub(crate) fn generate_controlled(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        session_id: Option<&str>,
        check: &dyn Fn() -> io::Result<()>,
        ready: impl FnOnce() -> io::Result<()>,
        emit: Option<&mut DeltaSink<'_>>,
    ) -> io::Result<Generation> {
        self.generate_controlled_with_sampling(
            prompt, max_tokens, session_id, check, ready, emit, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_controlled_with_sampling(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        session_id: Option<&str>,
        check: &dyn Fn() -> io::Result<()>,
        ready: impl FnOnce() -> io::Result<()>,
        emit: Option<&mut DeltaSink<'_>>,
        sampling: Option<crate::sample::SamplingParams>,
    ) -> io::Result<Generation> {
        let mut profile = RequestProfile::default();
        profile.kv_buffer_reuse.enabled = self.kv_buffer_reuse_enabled;
        if self.memory_trace_enabled {
            profile.memory_trace = Some(RequestMemoryTrace {
                schema_version: 1,
                outcome: "running",
                samples: Vec::with_capacity(8),
            });
            profile.memory_started = Some(Instant::now());
        }
        self.record_memory(&mut profile, "before_request", ActiveMemory::default());
        let mut result = self.generate_inner(
            prompt,
            max_tokens,
            session_id,
            check,
            ready,
            emit,
            &mut profile,
            sampling,
        );
        // All generate_inner locals have now left scope, even on cancellation/error.
        self.record_memory(
            &mut profile,
            "after_generate_inner",
            ActiveMemory::default(),
        );
        if result
            .as_ref()
            .is_err_and(|error| error.kind() == io::ErrorKind::OutOfMemory)
        {
            // Drop host-owned cache payloads after the worker has stopped using
            // them. Allocators may retain freed pages; this cannot force RSS down.
            self.sessions.clear();
            for layer in &mut self.layers {
                layer.cache.clear();
                layer.recycled_buffers.clear();
                layer.bytes = 0;
            }
            self.record_memory(&mut profile, "after_error_cleanup", ActiveMemory::default());
        }
        if let Some(trace) = profile.memory_trace.as_mut() {
            trace.outcome = match &result {
                Ok(_) => "completed",
                Err(e) if e.kind() == io::ErrorKind::OutOfMemory => "out_of_memory",
                Err(e) if e.kind() == io::ErrorKind::Interrupted => "interrupted",
                Err(_) => "error",
            };
        }
        if let Ok(generation) = &mut result {
            if let Some(metrics) = &mut generation.metrics {
                metrics.memory_trace = profile.memory_trace.clone();
            }
        }
        self.last_memory_trace = profile.memory_trace;
        self.last_kv_buffer_reuse = Some(profile.kv_buffer_reuse);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_inner(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        session_id: Option<&str>,
        check: &dyn Fn() -> io::Result<()>,
        ready: impl FnOnce() -> io::Result<()>,
        mut emit: Option<&mut DeltaSink<'_>>,
        profile: &mut RequestProfile,
        sampling: Option<crate::sample::SamplingParams>,
    ) -> io::Result<Generation> {
        let guard = self.ram_guard;
        // Check at existing cancellation boundaries, including inside layers.
        // A small forward reserve leaves room for one bounded expert load group.
        let reserve = (self.config.hidden_size * self.config.intermediate_size * 3
            + (self.config.intermediate_size * 2 + self.config.hidden_size) * 4)
            .saturating_mul(self.config.num_experts_per_tok + 2);
        let request_reserve = self
            .context_limit
            .saturating_mul(self.config.hidden_size)
            .saturating_mul(4)
            .saturating_mul(self.config.num_experts_per_tok + 12)
            .saturating_add(
                self.context_limit
                    .saturating_mul(self.config.hidden_size)
                    .saturating_mul(8)
                    .saturating_mul(self.layers.len()),
            )
            .saturating_add(reserve)
            .saturating_add(if self.expert_batch_prefill_enabled {
                (self.config.hidden_size + self.config.intermediate_size) * 48
            } else {
                0
            })
            .saturating_add(64 * 1024 * 1024);
        check()?;
        guard.check(request_reserve)?;
        let last_check = std::cell::Cell::new(Instant::now());
        let guarded_check = || {
            check()?;
            // Cancellation stays cheap and immediate. Sampling avoids reading
            // /proc for every routed token while retaining layer/load boundaries.
            if last_check.get().elapsed() >= Duration::from_millis(25) {
                guard.check(reserve)?;
                last_check.set(Instant::now());
            }
            Ok(())
        };
        let check: &dyn Fn() -> io::Result<()> = &guarded_check;
        check()?;
        let total_started = Instant::now();
        let bad = |s| io::Error::new(io::ErrorKind::InvalidInput, s);
        if max_tokens == 0 {
            return Err(bad("max_tokens must be positive"));
        }
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| invalid("missing tokenizer.json"))?;
        let tokenizer_started = Instant::now();
        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| invalid(e.to_string()))?;
        let tokens: Vec<_> = encoding.get_ids().iter().map(|&t| t as usize).collect();
        let tokenizer_elapsed = tokenizer_started.elapsed();
        if tokens.is_empty()
            || tokens
                .len()
                .checked_add(max_tokens)
                .is_none_or(|n| n > self.context_limit)
        {
            return Err(bad(
                "prompt plus max_tokens exceeds context limit or prompt is empty",
            ));
        }
        // Clone only for streaming: the decoder must not borrow self while step mutates it.
        let stream_tokenizer = emit.as_ref().map(|_| tokenizer.clone());
        let mut decoder = stream_tokenizer.as_ref().map(|t| t.decode_stream(true));
        let mut emitted = String::new();
        ready()?;
        check()?;
        self.session_clock = self.session_clock.wrapping_add(1);
        let d = self.config.hidden_size;
        let kd = d / self.config.num_attention_heads * self.config.num_key_value_heads;
        let (cached_prompt_tokens, mut kv, mut prepared_session) = self.take_prefix_kv(
            &tokens,
            session_id,
            kd,
            max_tokens,
            &mut profile.kv_buffer_reuse,
        );
        if profile.memory_trace.is_some() {
            profile.active_kv_container_bytes = kv.capacity() * std::mem::size_of::<Kv>();
            self.record_memory(
                profile,
                "after_kv_lookup",
                ActiveMemory {
                    kv: kv_capacity_bytes(&kv, profile.active_kv_container_bytes),
                    scratch: 0,
                    prefill: 0,
                },
            );
        }
        let mut scratch = StepScratch::new(&self.config);
        scratch.expert.gate_up_mode = self.gate_up_mode;
        for expert_scratch in &mut scratch.expert_scratches {
            expert_scratch.gate_up_mode = self.gate_up_mode;
        }
        let prefill_started = Instant::now();
        self.prefill(
            &tokens[cached_prompt_tokens..],
            &mut kv,
            profile,
            &mut scratch,
            check,
        )?;
        let prefill_elapsed = prefill_started.elapsed();
        if profile.memory_trace.is_some() {
            self.record_memory(
                profile,
                "after_prefill",
                ActiveMemory {
                    kv: kv_capacity_bytes(&kv, profile.active_kv_container_bytes),
                    scratch: scratch.capacity_bytes(),
                    prefill: 0,
                },
            );
        }
        let prefill_expert_io = profile.expert_io;
        let mut output = Vec::new();
        let mut finish_reason = "length";
        let first_token_started = total_started.elapsed();
        let decode_started = Instant::now();
        let mut decode_steps = 0usize;
        let mut processed_tokens = prepared_session
            .as_mut()
            .map(|prepared| std::mem::take(&mut prepared.tokens))
            .unwrap_or_else(|| tokens.clone());
        let sampling_params = sampling.unwrap_or_else(crate::sample::SamplingParams::from_env);
        let mut rng = crate::sample::XorShift64::default();
        for step in 0..max_tokens {
            check()?;
            let token = if sampling_params.is_greedy() {
                argmax_greedy(&scratch.logits)
            } else {
                crate::sample::sample_token(
                    &scratch.logits,
                    &sampling_params,
                    &processed_tokens,
                    &mut rng,
                )
            };
            output.push(token);
            if let (Some(decoder), Some(emit)) = (decoder.as_mut(), emit.as_mut()) {
                if let Some(delta) = decoder
                    .step(token as u32)
                    .map_err(|e| invalid(e.to_string()))?
                {
                    if !delta.is_empty() {
                        emit(&delta)?;
                        emitted.push_str(&delta);
                    }
                }
            }
            if token == self.config.eos_token_id {
                finish_reason = "stop";
                break;
            }
            if step + 1 < max_tokens {
                self.step(token, &mut kv, profile, &mut scratch, check)?;
                processed_tokens.push(token);
                decode_steps += 1;
            }
        }
        let decode_elapsed = decode_started.elapsed();
        let ids: Vec<_> = output.iter().map(|&id| id as u32).collect();
        let text = self
            .tokenizer
            .as_ref()
            .unwrap()
            .decode(&ids, true)
            .map_err(|e| invalid(e.to_string()))?;
        if let Some(emit) = emit.as_mut() {
            // Flush any incomplete byte sequence held by the incremental decoder at EOS/limit.
            let tail = text
                .strip_prefix(&emitted)
                .ok_or_else(|| invalid("stream decoder output differs from final text"))?;
            if !tail.is_empty() {
                emit(tail)?;
            }
        }
        check()?;
        if let Some(tracer) = self.route_trace.as_mut() {
            tracer.advance_call();
        }
        if let Some(path) = &self.usage_path {
            let _ = Self::save_usage_to_path(path, &self.layers, self.config.num_experts);
        }
        let total_elapsed = total_started.elapsed();
        let cache_bytes = self.layers.iter().map(|layer| layer.bytes).sum();
        let kv_bytes = kv.iter().map(Kv::memory_bytes).sum();
        let mut active_kv_after = 0;
        if profile.memory_trace.is_some() {
            active_kv_after = kv_capacity_bytes(&kv, profile.active_kv_container_bytes);
            self.record_memory(
                profile,
                "before_session_admission",
                ActiveMemory {
                    kv: active_kv_after,
                    scratch: scratch.capacity_bytes(),
                    prefill: 0,
                },
            );
        }
        if let Some(id) = session_id {
            let id = prepared_session
                .take()
                .map(|prepared| prepared.id)
                .unwrap_or_else(|| id.to_owned());
            // Lưu bền vững trạng thái KV cache của session xuống NVMe disk (.moekv) nếu session_dir được cấu hình
            if let Some(dir) = &self.session_dir {
                let file_path = dir.join(format!("{id}.moekv"));
                let _ = self.save_session_to_file(&file_path, &processed_tokens, &kv);
            }
            let state = SessionState {
                tokens: processed_tokens,
                kv,
                last_used: self.session_clock,
            };
            let retained_bytes = session_bytes(&state).saturating_add(id.capacity());
            if retained_bytes <= self.session_cache_bytes {
                while self
                    .sessions
                    .iter()
                    .fold(0usize, |total, (key, state)| {
                        total
                            .saturating_add(session_bytes(state))
                            .saturating_add(key.capacity())
                    })
                    .saturating_add(retained_bytes)
                    > self.session_cache_bytes
                {
                    let Some(oldest) = self
                        .sessions
                        .iter()
                        .min_by_key(|(_, state)| state.last_used)
                        .map(|(id, _)| id.clone())
                    else {
                        break;
                    };
                    self.sessions.remove(&oldest);
                }
                self.sessions.insert(id, state);
            }
            active_kv_after = 0;
        }
        if profile.memory_trace.is_some() {
            self.record_memory(
                profile,
                "after_session_admission",
                ActiveMemory {
                    kv: active_kv_after,
                    scratch: scratch.capacity_bytes(),
                    prefill: 0,
                },
            );
        }
        let seconds = decode_elapsed.as_secs_f64();
        Ok(Generation {
            text,
            token_ids: output,
            prompt_tokens: tokens.len(),
            finish_reason,
            metrics: Some(GenerationMetrics {
                expert_batch_prefill_enabled: self.expert_batch_prefill_enabled,
                prefill_expert_batch2_calls: profile.prefill_expert_batch2_calls,
                prefill_expert_batch4_calls: profile.prefill_expert_batch4_calls,
                gate_up_mode: self.gate_up_mode.as_str(),
                kv_buffer_reuse: profile.kv_buffer_reuse.clone(),
                memory_trace: None, // Final boundary is attached by generate_controlled.
                lookahead_enabled: self.lookahead_enabled,
                lookahead: std::mem::take(&mut profile.lookahead),
                decode_expert_active_peak_bytes: profile.decode_expert_active_peak_bytes,
                decode_expert_host_peak_bytes: profile.decode_expert_host_peak_bytes,
                tokenizer_ms: ms(tokenizer_elapsed),
                prefill_ms: ms(prefill_elapsed),
                time_to_first_token_ms: first_token_started.as_secs_f64() * 1000.0,
                decode_ms: ms(decode_elapsed),
                total_ms: ms(total_elapsed),
                decode_tokens_per_second: if seconds > 0.0 {
                    decode_steps as f64 / seconds
                } else {
                    0.0
                },
                attention_ms: ms(profile.attention),
                expert_compute_ms: ms(profile.expert_compute),
                lm_head_ms: ms(profile.lm_head),
                expert_io_ms: ms(profile.expert_io),
                prefill_expert_io_ms: ms(prefill_expert_io),
                decode_expert_io_ms: ms(profile.expert_io.saturating_sub(prefill_expert_io)),
                expert_file_read_ms: ms(profile.file_io),
                expert_vram_download_ms: ms(profile.vram_io),
                expert_vram_upload_ms: ms(profile.upload_io),
                expert_file_bytes_read: profile.file_bytes,
                expert_vram_download_bytes: profile.vram_bytes,
                expert_vram_upload_bytes: profile.upload_bytes,
                expert_file_loads: profile.file_loads,
                expert_vram_loads: profile.vram_loads,
                expert_cache_hits: profile.cache_hits,
                expert_cache_misses: profile.cache_misses,
                expert_cache_evictions: profile.cache_evictions,
                expert_bytes_read: profile.bytes_read,
                expert_cache_bytes: cache_bytes,
                kv_cache_bytes: kv_bytes,
                cached_prompt_tokens,
            }),
        })
    }
}

fn kv_capacity_bytes(kv: &[Kv], container_bytes: usize) -> usize {
    kv.iter().fold(container_bytes, |total, cache| {
        total.saturating_add(cache.allocated_bytes())
    })
}

fn parse_expert_batch_prefill_mode(
    value: Option<&str>,
    gpu: GpuMode,
    precision: &str,
) -> io::Result<bool> {
    match value {
        None | Some("off") => Ok(false),
        Some("on") if gpu == GpuMode::Cpu && precision == "int8" => Ok(true),
        Some("on") => Err(invalid(
            "expert batch prefill requires explicit CPU mode and INT8",
        )),
        _ => Err(invalid("MOE_EXPERT_BATCH_PREFILL must be on or off")),
    }
}

fn parse_memory_trace_mode(value: Option<&str>) -> io::Result<bool> {
    match value {
        None | Some("off") => Ok(false),
        Some("on") => Ok(true),
        _ => Err(invalid("MOE_MEMORY_TRACE must be on or off")),
    }
}

fn parse_kv_buffer_reuse_mode(value: Option<&str>) -> io::Result<bool> {
    match value {
        None | Some("off") => Ok(false),
        Some("on") => Ok(true),
        _ => Err(invalid("MOE_KV_BUFFER_REUSE must be on or off")),
    }
}

fn parse_lookahead_mode(value: Option<&str>) -> io::Result<bool> {
    match value {
        None | Some("on") => Ok(true),
        Some("off") => Ok(false),
        _ => Err(invalid("MOE_LOOKAHEAD_PREFETCH must be on or off")),
    }
}

fn buffer_bytes(buffer: &(Vec<u8>, Vec<f32>)) -> usize {
    buffer.0.capacity() + buffer.1.capacity() * std::mem::size_of::<f32>()
}

fn observe_decode_payload<'a>(
    profile: &mut RequestProfile,
    layers: impl Iterator<Item = &'a Layer>,
    active: usize,
) {
    let retained: usize = layers
        .map(|l| {
            l.cache
                .iter()
                .map(|e| e.expert.allocated_bytes())
                .sum::<usize>()
                + l.recycled_bytes()
        })
        .sum();
    profile.decode_expert_active_peak_bytes = profile.decode_expert_active_peak_bytes.max(active);
    profile.decode_expert_host_peak_bytes =
        profile.decode_expert_host_peak_bytes.max(retained + active);
}

fn record_lookahead_loads(
    profile: &mut RequestProfile,
    attempts: usize,
    loaded: &HashMap<usize, Int8Expert>,
) {
    profile.lookahead.loaded += loaded.len() as u64;
    profile.lookahead.load_errors += (attempts - loaded.len()) as u64;
    profile.lookahead.pending += loaded.len() as u64;
    profile.lookahead.file_bytes += loaded
        .values()
        .map(|e| e.resident_bytes() as u64)
        .sum::<u64>();
}

impl Drop for OlmoeModel {
    fn drop(&mut self) {
        if let Some(path) = &self.usage_path {
            let _ = Self::save_usage_to_path(path, &self.layers, self.config.num_experts);
        }
    }
}

fn session_bytes(state: &SessionState) -> usize {
    // Retained vector allocations, including unused capacity after prefix truncation.
    // HashMap buckets/allocator bookkeeping and the active request are separate.
    state.kv.iter().fold(
        std::mem::size_of::<SessionState>()
            .saturating_add(
                state
                    .tokens
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
            .saturating_add(
                state
                    .kv
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Kv>()),
            ),
        |total, cache| total.saturating_add(cache.allocated_bytes()),
    )
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
#[inline]
fn prepare_rope_sincos(sincos: &mut [(f32, f32)], rope_bases: &[f32], position: usize) {
    for (entry, &base) in sincos.iter_mut().zip(rope_bases) {
        let angle = position as f32 / base;
        *entry = angle.sin_cos();
    }
}

#[inline]
fn apply_rope_cached(x: &mut [f32], hd: usize, sincos: &[(f32, f32)]) {
    let half = hd / 2;
    for head in x.chunks_exact_mut(hd) {
        for (i, &(sin, cos)) in sincos.iter().enumerate().take(half) {
            let a = head[i];
            let b = head[i + half];
            head[i] = a * cos - b * sin;
            head[i + half] = b * cos + a * sin;
        }
    }
}

#[inline]
pub(crate) fn argmax_greedy(logits: &[f32]) -> usize {
    if logits.is_empty() {
        return 0;
    }
    let mut best_idx = 0;
    let mut best_val = logits[0];
    for (i, &val) in logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
fn rotate(x: &mut [f32], hd: usize, position: usize, theta: f32) {
    let half = hd / 2;
    let mut sincos = vec![(0.0f32, 0.0f32); half];
    let bases: Vec<f32> = (0..half)
        .map(|i| theta.powf(2.0 * i as f32 / hd as f32))
        .collect();
    prepare_rope_sincos(&mut sincos, &bases, position);
    apply_rope_cached(x, hd, &sincos);
}

#[cfg(test)]
mod memory_trace_tests {
    use super::*;

    #[test]
    fn kv_buffer_reuse_recycles_one_slot_without_stale_prefix_or_overlap() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let mut reused = model(fixture);
            reused.kv_buffer_reuse_enabled = true;
            reused
                .generate_controlled("[BOS]token3 ", 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            reused.session_cache_bytes =
                reused.expert_status().session_cache_allocated_bytes + 1024;
            let pointers: Vec<_> = reused.sessions["a"]
                .kv
                .iter()
                .map(|kv| (kv.keys.as_ptr(), kv.values.as_ptr()))
                .collect();
            let output = reused
                .generate_controlled("[BOS]token8 ", 3, Some("b"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            let metrics = output.metrics.as_ref().unwrap();
            assert_eq!(metrics.kv_buffer_reuse.early_evicted_sessions, 1);
            assert!(metrics.kv_buffer_reuse.reused_capacity_bytes > 0);
            assert_eq!(metrics.kv_buffer_reuse.source, Some("evicted_session"));
            assert_eq!(metrics.cached_prompt_tokens, 0);
            assert!(!reused.sessions.contains_key("a"));
            assert!(reused.sessions.contains_key("b"));
            let after: Vec<_> = reused.sessions["b"]
                .kv
                .iter()
                .map(|kv| (kv.keys.as_ptr(), kv.values.as_ptr()))
                .collect();
            assert_eq!(pointers, after);
            let lookup = stage(metrics.memory_trace.as_ref().unwrap(), "after_kv_lookup");
            assert_eq!(lookup.retained_session_capacity_bytes, 0);
            assert_eq!(
                lookup.active_kv_capacity_bytes,
                metrics.kv_buffer_reuse.reused_capacity_bytes
            );
            let fresh = model(fixture).generate("[BOS]token8 ", 3).unwrap();
            assert_eq!(output.token_ids, fresh.token_ids);
        }
    }

    #[test]
    fn kv_buffer_reuse_preserves_prefix_and_resets_same_id_without_prefix() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let mut reused = model(fixture);
            reused.kv_buffer_reuse_enabled = true;
            reused
                .generate_controlled("[BOS]token3 ", 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            let pointer = reused.sessions["a"].kv[0].keys.as_ptr();
            let prefix = reused
                .generate_controlled(
                    "[BOS]token3 token4 ",
                    3,
                    Some("a"),
                    &|| Ok(()),
                    || Ok(()),
                    None,
                )
                .unwrap();
            assert!(prefix.metrics.as_ref().unwrap().cached_prompt_tokens > 0);
            assert_eq!(
                prefix
                    .metrics
                    .as_ref()
                    .unwrap()
                    .kv_buffer_reuse
                    .reused_capacity_bytes,
                0
            );
            assert_eq!(pointer, reused.sessions["a"].kv[0].keys.as_ptr());
            let reset = reused
                .generate_controlled("token8 ", 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            let metrics = reset.metrics.as_ref().unwrap();
            assert_eq!(metrics.cached_prompt_tokens, 0);
            assert_eq!(metrics.kv_buffer_reuse.source, Some("same_session_reset"));
            assert_eq!(metrics.kv_buffer_reuse.early_evicted_sessions, 0);
            assert_eq!(pointer, reused.sessions["a"].kv[0].keys.as_ptr());
            assert_eq!(
                reset.token_ids,
                model(fixture).generate("token8 ", 3).unwrap().token_ids
            );
        }
    }

    #[test]
    fn kv_buffer_reuse_selects_lru_and_keeps_total_retained_budget() {
        let mut reused = model("unnormalized-mha");
        reused.kv_buffer_reuse_enabled = true;
        for id in ["a", "b", "a"] {
            reused
                .generate_controlled("[BOS]token3 ", 3, Some(id), &|| Ok(()), || Ok(()), None)
                .unwrap();
        }
        let pointer_b = reused.sessions["b"].kv[0].keys.as_ptr();
        let retained = reused.expert_status().session_cache_allocated_bytes;
        reused.session_cache_bytes = retained + retained / 4;
        let output = reused
            .generate_controlled("[BOS]token8 ", 3, Some("c"), &|| Ok(()), || Ok(()), None)
            .unwrap();
        assert_eq!(
            output
                .metrics
                .unwrap()
                .kv_buffer_reuse
                .early_evicted_sessions,
            1
        );
        assert!(!reused.sessions.contains_key("b"));
        assert!(reused.sessions.contains_key("a") && reused.sessions.contains_key("c"));
        assert_eq!(pointer_b, reused.sessions["c"].kv[0].keys.as_ptr());
        assert!(reused.expert_status().session_cache_allocated_bytes <= reused.session_cache_bytes);
    }

    #[test]
    fn kv_buffer_reuse_does_not_evict_when_no_admission_or_room_remains() {
        for case in ["room", "sessionless", "oversized"] {
            let mut reused = model("unnormalized-mha");
            reused.kv_buffer_reuse_enabled = true;
            reused
                .generate_controlled("[BOS]token3 ", 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            let retained = reused.expert_status().session_cache_allocated_bytes;
            reused.session_cache_bytes = if case == "room" {
                retained * 3
            } else if case == "oversized" {
                0
            } else {
                retained + 1024
            };
            let output = reused
                .generate_controlled(
                    "[BOS]token8 ",
                    3,
                    if case == "sessionless" {
                        None
                    } else {
                        Some("b")
                    },
                    &|| Ok(()),
                    || Ok(()),
                    None,
                )
                .unwrap();
            let metrics = output.metrics.unwrap();
            assert_eq!(metrics.kv_buffer_reuse.early_evicted_sessions, 0, "{case}");
            assert_eq!(metrics.kv_buffer_reuse.reused_capacity_bytes, 0, "{case}");
            assert!(reused.sessions.contains_key("a"), "{case}");
        }
    }

    #[test]
    fn kv_buffer_reuse_rejects_incompatible_or_oversized_donor_without_removing_id() {
        for too_large in [false, true] {
            let mut reused = model("unnormalized-mha");
            reused
                .generate_controlled("[BOS]token3 ", 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            let kd = reused.config.hidden_size / reused.config.num_attention_heads
                * reused.config.num_key_value_heads;
            for kv in &mut reused.sessions.get_mut("a").unwrap().kv {
                if too_large {
                    kv.keys.reserve_exact(reused.context_limit * kd * 4);
                    kv.values.reserve_exact(reused.context_limit * kd * 4);
                } else {
                    kv.keys.shrink_to_fit();
                    kv.values.shrink_to_fit();
                }
            }
            let minimum = reused.layers.len()
                * (reused.context_limit * kd * 8 + std::mem::size_of::<Kv>())
                + std::mem::size_of::<SessionState>()
                + 2 * std::mem::size_of::<usize>()
                + 1;
            reused.session_cache_bytes = minimum + 64;
            reused.kv_buffer_reuse_enabled = true;
            let mut metrics = KvBufferReuseMetrics::default();
            let (_, _, prepared) = reused.take_prefix_kv(&[1, 8], Some("b"), kd, 3, &mut metrics);
            assert_eq!(metrics.early_evicted_sessions, 0);
            assert_eq!(metrics.reused_capacity_bytes, 0);
            assert!(prepared.is_none() && reused.sessions.contains_key("a"));
        }
    }

    #[test]
    fn kv_buffer_reuse_cancel_errors_remove_old_id_and_retry_is_fresh() {
        for kind in [
            io::ErrorKind::Interrupted,
            io::ErrorKind::InvalidData,
            io::ErrorKind::OutOfMemory,
        ] {
            let mut reused = model("unnormalized-mha");
            reused.kv_buffer_reuse_enabled = true;
            reused
                .generate_controlled("[BOS]token3 ", 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            reused.session_cache_bytes =
                reused.expert_status().session_cache_allocated_bytes + 1024;
            let mut emit =
                |_delta: &str| Err(io::Error::new(kind, "fixture error after KV handoff"));
            let error = reused
                .generate_controlled(
                    "[BOS]token3 ",
                    3,
                    Some("b"),
                    &|| Ok(()),
                    || Ok(()),
                    Some(&mut emit),
                )
                .unwrap_err();
            assert_eq!(error.kind(), kind);
            assert!(!reused.sessions.contains_key("a") && !reused.sessions.contains_key("b"));
            let status = reused.expert_status();
            assert_eq!(
                status
                    .last_kv_buffer_reuse
                    .as_ref()
                    .unwrap()
                    .early_evicted_sessions,
                1
            );
            let trace = status.last_memory_trace.unwrap();
            assert_eq!(trace.samples.last().unwrap().active_kv_capacity_bytes, 0);
            let retry = reused
                .generate_controlled("[BOS]token3 ", 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            assert_eq!(retry.metrics.as_ref().unwrap().cached_prompt_tokens, 0);
            assert_eq!(
                retry.token_ids,
                model("unnormalized-mha")
                    .generate("[BOS]token3 ", 3)
                    .unwrap()
                    .token_ids
            );
        }
    }

    #[test]
    fn kv_buffer_reuse_prefill_logits_match_fresh_and_parser_is_explicit() {
        assert!(!parse_kv_buffer_reuse_mode(None).unwrap());
        assert!(!parse_kv_buffer_reuse_mode(Some("off")).unwrap());
        assert!(parse_kv_buffer_reuse_mode(Some("on")).unwrap());
        assert!(parse_kv_buffer_reuse_mode(Some("true")).is_err());
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let mut reused = model(fixture);
            reused.kv_buffer_reuse_enabled = true;
            reused
                .generate_controlled("[BOS]token3 ", 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            reused.session_cache_bytes =
                reused.expert_status().session_cache_allocated_bytes + 1024;
            let kd = reused.config.hidden_size / reused.config.num_attention_heads
                * reused.config.num_key_value_heads;
            let mut metrics = KvBufferReuseMetrics::default();
            let (_, mut kv, prepared) =
                reused.take_prefix_kv(&[1, 8], Some("b"), kd, 3, &mut metrics);
            assert!(prepared.is_some() && metrics.reused_capacity_bytes > 0);
            assert!(kv
                .iter()
                .all(|kv| kv.keys.is_empty() && kv.values.is_empty()));
            let mut scratch = StepScratch::new(&reused.config);
            reused
                .prefill(
                    &[1, 8],
                    &mut kv,
                    &mut RequestProfile::default(),
                    &mut scratch,
                    &|| Ok(()),
                )
                .unwrap();
            let mut fresh = model(fixture);
            let logits = fresh.logits(&[1, 8]).unwrap();
            let error = scratch
                .logits
                .iter()
                .zip(logits.last().unwrap())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(error < 2e-5, "{fixture}: {error}");
        }
    }

    fn model(fixture: &str) -> OlmoeModel {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/olmoe")
            .join(fixture);
        let mut model =
            OlmoeModel::load_with_gpu_mode(path, 8192, 64, 10_000_000, GpuMode::Cpu).unwrap();
        model.memory_trace_enabled = true;
        model.kv_buffer_reuse_enabled = false;
        model
    }

    fn stage<'a>(trace: &'a RequestMemoryTrace, name: &str) -> &'a MemoryStageSample {
        trace.samples.iter().find(|s| s.stage == name).unwrap()
    }

    #[test]
    fn integration_fp8_switch_invalidates_memory_and_rejects_old_disk_precision() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let mut traced = model(fixture);
            let directory = tempfile::tempdir().unwrap();
            traced.set_session_dir(Some(directory.path().to_owned()));
            let prompt = "[BOS]token3 token4 ";
            traced
                .generate_controlled(prompt, 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            assert!(directory.path().join("a.moekv").exists());
            assert!(!traced.sessions["a"].kv[0].is_fp8);

            traced.set_kv_fp8(true);
            assert!(traced.sessions.is_empty());
            let output = traced
                .generate_controlled(prompt, 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            assert_eq!(output.metrics.as_ref().unwrap().cached_prompt_tokens, 0);
            assert!(traced.sessions["a"].kv.iter().all(|cache| cache.is_fp8));
            let expected = kv_capacity_bytes(
                &traced.sessions["a"].kv,
                traced.sessions["a"].kv.capacity() * std::mem::size_of::<Kv>(),
            );
            let lookup = stage(
                output
                    .metrics
                    .as_ref()
                    .unwrap()
                    .memory_trace
                    .as_ref()
                    .unwrap(),
                "after_kv_lookup",
            );
            assert_eq!(lookup.active_kv_capacity_bytes, expected);
            let payload = traced.sessions["a"]
                .kv
                .iter()
                .map(Kv::allocated_bytes)
                .sum::<usize>();
            assert!(payload > 0);
            assert_eq!(
                expected,
                payload + traced.sessions["a"].kv.capacity() * std::mem::size_of::<Kv>()
            );

            assert!(!traced.toggle_kv_fp8());
            assert!(traced.sessions.is_empty());
            let output = traced
                .generate_controlled(prompt, 3, Some("a"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            assert_eq!(output.metrics.as_ref().unwrap().cached_prompt_tokens, 0);
            assert!(traced.sessions["a"].kv.iter().all(|cache| !cache.is_fp8));
        }
    }

    #[test]
    fn integration_fp8_same_session_reset_and_cross_session_fallback() {
        let mut traced = model("unnormalized-mha");
        traced.set_kv_fp8(true);
        traced.kv_buffer_reuse_enabled = true;
        traced
            .generate_controlled(
                "[BOS]token3 token4 ",
                3,
                Some("a"),
                &|| Ok(()),
                || Ok(()),
                None,
            )
            .unwrap();
        let kd = traced.config.hidden_size / traced.config.num_attention_heads
            * traced.config.num_key_value_heads;
        let mut reuse = KvBufferReuseMetrics::default();
        let (prefix, buffers, _) =
            traced.take_prefix_kv(&[usize::MAX], Some("a"), kd, 3, &mut reuse);
        assert_eq!(prefix, 0);
        assert_eq!(reuse.source, Some("same_session_reset"));
        assert!(buffers.iter().all(|cache| cache.positions(kd) == 0
            && cache.fp8_keys.is_empty()
            && cache.fp8_values.is_empty()));
        assert_eq!(
            reuse.reused_capacity_bytes,
            kv_capacity_bytes(&buffers, buffers.capacity() * std::mem::size_of::<Kv>())
        );
        traced
            .generate_controlled(
                "[BOS]token3 token4 ",
                3,
                Some("a"),
                &|| Ok(()),
                || Ok(()),
                None,
            )
            .unwrap();
        traced.session_cache_bytes = traced.expert_status().session_cache_allocated_bytes;
        let result = traced.take_evicted_kv(&[3], "b", kd, 3, &mut reuse);
        assert!(result.is_none());
        assert!(traced.sessions.contains_key("a"));
    }

    #[test]
    fn expert_batch_prefill_parser_restricts_precision_and_backend() {
        assert!(!parse_expert_batch_prefill_mode(None, GpuMode::Cpu, "int8").unwrap());
        assert!(!parse_expert_batch_prefill_mode(Some("off"), GpuMode::Auto, "int4").unwrap());
        assert!(parse_expert_batch_prefill_mode(Some("on"), GpuMode::Cpu, "int8").unwrap());
        for (value, gpu, quant) in [
            ("auto", GpuMode::Cpu, "int8"),
            ("on", GpuMode::Auto, "int8"),
            ("on", GpuMode::Cpu, "int4"),
        ] {
            assert!(parse_expert_batch_prefill_mode(Some(value), gpu, quant).is_err());
        }
    }

    #[test]
    fn expert_batch_prefill_matches_tokenwise_logits_across_route_tails() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            for length in [1, 2, 3, 4, 5, 7, 9] {
                let mut baseline = model(fixture);
                let mut candidate = model(fixture);
                candidate.expert_batch_prefill_enabled = true;
                let tokens: Vec<_> = (0..length).map(|i| [0, 3, 4, 6][i % 4]).collect();
                let expected = baseline.logits(&tokens).unwrap().pop().unwrap();
                let mut kv: Vec<_> = (0..candidate.layers.len()).map(|_| Kv::default()).collect();
                let mut profile = RequestProfile::default();
                let mut scratch = StepScratch::new(&candidate.config);
                candidate
                    .prefill(&tokens, &mut kv, &mut profile, &mut scratch, &|| Ok(()))
                    .unwrap();
                for (a, b) in scratch.logits.iter().zip(expected) {
                    assert!(
                        (a - b).abs() < 2e-5,
                        "{fixture} length={length}: {a} vs {b}"
                    );
                }
                if length >= 7 {
                    assert!(profile.prefill_expert_batch4_calls > 0);
                }
                let kd = candidate.config.hidden_size / candidate.config.num_attention_heads
                    * candidate.config.num_key_value_heads;
                assert!(kv.iter().all(|k| k.positions(kd) == length));
            }
        }
    }

    #[test]
    fn expert_batch_prefill_preserves_sessions_and_accounts_local_buffers() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let mut baseline = model(fixture);
            let mut candidate = model(fixture);
            candidate.expert_batch_prefill_enabled = true;
            let extra = (candidate.config.hidden_size + candidate.config.intermediate_size) * 48;
            let prompt = "[BOS]token3 token4 token6 token3 token4 token6 token3 token4 token6 ";
            for (index, (text, id)) in [
                (prompt, "a"),
                (prompt, "a"),
                ("[BOS]token8 token3 token4 ", "b"),
            ]
            .into_iter()
            .enumerate()
            {
                let a = baseline
                    .generate_controlled(text, 3, Some(id), &|| Ok(()), || Ok(()), None)
                    .unwrap();
                let b = candidate
                    .generate_controlled(text, 3, Some(id), &|| Ok(()), || Ok(()), None)
                    .unwrap();
                assert_eq!(a.token_ids, b.token_ids);
                assert_eq!(a.finish_reason, b.finish_reason);
                let am = a.metrics.unwrap();
                let bm = b.metrics.unwrap();
                assert_eq!(am.cached_prompt_tokens, bm.cached_prompt_tokens);
                assert_eq!(am.expert_file_loads, bm.expert_file_loads);
                assert!(bm.expert_batch_prefill_enabled);
                assert_eq!(
                    am.prefill_expert_batch2_calls + am.prefill_expert_batch4_calls,
                    0
                );
                let at = am.memory_trace.unwrap();
                let bt = bm.memory_trace.unwrap();
                assert_eq!(at.samples.len(), bt.samples.len());
                if index == 0 {
                    assert!(bm.prefill_expert_batch4_calls > 0);
                    assert_eq!(
                        stage(&bt, "prefill_buffers_live").prefill_capacity_bytes,
                        stage(&at, "prefill_buffers_live").prefill_capacity_bytes + extra
                    );
                }
                assert_eq!(stage(&bt, "after_prefill").prefill_capacity_bytes, 0);
            }
            let failure = candidate.generate_controlled(
                prompt,
                3,
                Some("cancel"),
                &|| Err(io::Error::new(io::ErrorKind::Interrupted, "cancel")),
                || Ok(()),
                None,
            );
            assert!(failure.is_err());
            assert!(candidate.generate(prompt, 3).is_ok());
        }
    }

    #[test]
    fn gate_up_model_modes_preserve_logits_generation_and_prefix_sessions() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let mut paired = model(fixture);
            let mut separate = model(fixture);
            separate.gate_up_mode = GateUpMode::Separate;
            let tokens = [0, 3, 4, 6];
            let a = paired.logits(&tokens).unwrap();
            let b = separate.logits(&tokens).unwrap();
            for (a, b) in a.iter().flatten().zip(b.iter().flatten()) {
                assert!((a - b).abs() < 2e-5);
            }
            paired.reset_cache();
            separate.reset_cache();
            for (prompt, id) in [
                ("[BOS]token3 token4 ", "a"),
                ("[BOS]token3 token4 token6 ", "a"),
                ("[BOS]token8 ", "b"),
            ] {
                let a = paired
                    .generate_controlled(prompt, 3, Some(id), &|| Ok(()), || Ok(()), None)
                    .unwrap();
                let b = separate
                    .generate_controlled(prompt, 3, Some(id), &|| Ok(()), || Ok(()), None)
                    .unwrap();
                assert_eq!(a.token_ids, b.token_ids);
                assert_eq!(a.finish_reason, b.finish_reason);
                let ma = a.metrics.unwrap();
                let mb = b.metrics.unwrap();
                assert_eq!(ma.gate_up_mode, "paired");
                assert_eq!(mb.gate_up_mode, "separate");
                assert_eq!(ma.cached_prompt_tokens, mb.cached_prompt_tokens);
                assert_eq!(paired.expert_status().expert_quantization, "int8");
                assert_eq!(separate.expert_status().gate_up_mode, "separate");
            }
        }
        assert_eq!(
            "separate".parse::<GateUpMode>().unwrap(),
            GateUpMode::Separate
        );
        assert_eq!("paired".parse::<GateUpMode>().unwrap(), GateUpMode::Paired);
        for value in ["", "auto", "on"] {
            assert!(value.parse::<GateUpMode>().is_err());
        }
    }

    fn totals(trace: &RequestMemoryTrace) {
        assert_eq!(trace.schema_version, 1);
        assert!(trace.samples.len() <= 8);
        for s in &trace.samples {
            let encoded = serde_json::to_value(s).unwrap();
            assert!(encoded.get("thread_id").is_some());
            assert!(encoded["allocator"]["status"].is_string());
            if s.allocator.status == "glibc_mallinfo2" {
                assert_eq!(
                    s.allocator
                        .in_use_bytes
                        .unwrap()
                        .checked_add(s.allocator.free_bytes.unwrap()),
                    s.allocator.arena_bytes
                );
                assert!(s.allocator.libc_version.is_some());
            } else {
                assert!(s.allocator.in_use_bytes.is_none());
            }
            assert_eq!(
                s.tracked_host_bytes,
                s.dense_weight_payload_bytes
                    + s.expert_cache_capacity_bytes
                    + s.recycled_capacity_bytes
                    + s.retained_session_capacity_bytes
                    + s.active_kv_capacity_bytes
                    + s.scratch_capacity_bytes
                    + s.prefill_capacity_bytes
            );
            assert!(s.retained_kv_capacity_bytes <= s.retained_session_capacity_bytes);
        }
        assert!(trace
            .samples
            .windows(2)
            .all(|p| p[0].elapsed_ms <= p[1].elapsed_ms));
    }

    #[test]
    fn memory_trace_kv_transfer_reuse_eviction_and_mha_gqa_parity() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let mut traced = model(fixture);
            let mut reference = model(fixture);
            reference.memory_trace_enabled = false;
            for _ in 0..2 {
                let prompt = "[BOS]token3 token4 ";
                let actual = traced
                    .generate_controlled(prompt, 3, Some("a"), &|| Ok(()), || Ok(()), None)
                    .unwrap();
                let expected = reference
                    .generate_controlled(prompt, 3, Some("a"), &|| Ok(()), || Ok(()), None)
                    .unwrap();
                assert_eq!(actual.token_ids, expected.token_ids);
                assert!(expected.metrics.unwrap().memory_trace.is_none());
                let trace = actual
                    .metrics
                    .as_ref()
                    .unwrap()
                    .memory_trace
                    .as_ref()
                    .unwrap();
                totals(trace);
                assert_eq!(trace.outcome, "completed");
                assert_eq!(
                    trace.samples.iter().map(|s| s.stage).collect::<Vec<_>>(),
                    [
                        "before_request",
                        "after_kv_lookup",
                        "prefill_buffers_live",
                        "after_prefill",
                        "before_session_admission",
                        "after_session_admission",
                        "after_generate_inner"
                    ]
                );
                assert_eq!(
                    stage(trace, "after_kv_lookup").retained_session_capacity_bytes,
                    0
                );
                assert_eq!(stage(trace, "after_kv_lookup").scratch_capacity_bytes, 0);
                let active = stage(trace, "before_session_admission").active_kv_capacity_bytes;
                assert!(active > 0);
                assert_eq!(
                    stage(trace, "after_session_admission").active_kv_capacity_bytes,
                    0
                );
                assert_eq!(
                    stage(trace, "after_session_admission").retained_kv_capacity_bytes,
                    active
                );
                assert!(stage(trace, "prefill_buffers_live").prefill_capacity_bytes > 0);
                assert_eq!(stage(trace, "after_prefill").prefill_capacity_bytes, 0);
                assert!(stage(trace, "after_session_admission").scratch_capacity_bytes > 0);
                assert_eq!(
                    stage(trace, "after_generate_inner").scratch_capacity_bytes,
                    0
                );
            }
            traced.session_cache_bytes = traced.expert_status().session_cache_allocated_bytes;
            let output = traced
                .generate_controlled(
                    "[BOS]token3 token4 ",
                    3,
                    Some("b"),
                    &|| Ok(()),
                    || Ok(()),
                    None,
                )
                .unwrap();
            let trace = output
                .metrics
                .as_ref()
                .unwrap()
                .memory_trace
                .as_ref()
                .unwrap();
            totals(trace);
            let lookup = stage(trace, "after_kv_lookup");
            assert!(lookup.retained_kv_capacity_bytes > 0 && lookup.active_kv_capacity_bytes > 0);
            assert!(!traced.sessions.contains_key("a") && traced.sessions.contains_key("b"));
            assert_eq!(
                stage(trace, "after_session_admission").retained_kv_capacity_bytes,
                lookup.active_kv_capacity_bytes
            );
            traced.reset_cache();
            assert!(traced.expert_status().last_memory_trace.is_none());
            assert!(traced.memory_trace_enabled);
        }
    }

    #[test]
    fn memory_trace_unretained_and_sessionless_kv_drop() {
        for session in [Some("too-large"), None] {
            let mut model = model("unnormalized-mha");
            model.session_cache_bytes = 0;
            let output = model
                .generate_controlled("[BOS]token3 ", 3, session, &|| Ok(()), || Ok(()), None)
                .unwrap();
            let trace = output
                .metrics
                .as_ref()
                .unwrap()
                .memory_trace
                .as_ref()
                .unwrap();
            totals(trace);
            let after = stage(trace, "after_session_admission");
            assert_eq!(after.retained_session_capacity_bytes, 0);
            assert_eq!(after.active_kv_capacity_bytes > 0, session.is_none());
            assert_eq!(
                stage(trace, "after_generate_inner").active_kv_capacity_bytes,
                0
            );
            assert!(model.sessions.is_empty());
        }
    }

    #[test]
    fn memory_trace_cancel_oom_cleanup_and_retry() {
        for kind in [io::ErrorKind::Interrupted, io::ErrorKind::OutOfMemory] {
            let mut model = model("unnormalized-mha");
            model
                .generate_controlled("[BOS]token3 ", 3, Some("old"), &|| Ok(()), || Ok(()), None)
                .unwrap();
            let mut emit = |_delta: &str| Err(io::Error::new(kind, "fixture error after prefill"));
            let error = model
                .generate_controlled(
                    "[BOS]token3 ",
                    3,
                    Some("new"),
                    &|| Ok(()),
                    || Ok(()),
                    Some(&mut emit),
                )
                .unwrap_err();
            assert_eq!(error.kind(), kind);
            let status = model.expert_status();
            let trace = status.last_memory_trace.as_ref().unwrap();
            totals(trace);
            assert!(stage(trace, "after_prefill").active_kv_capacity_bytes > 0);
            let last = trace.samples.last().unwrap();
            assert_eq!(
                last.active_kv_capacity_bytes
                    + last.scratch_capacity_bytes
                    + last.prefill_capacity_bytes,
                0
            );
            assert!(!model.sessions.contains_key("new"));
            if kind == io::ErrorKind::OutOfMemory {
                assert_eq!(trace.outcome, "out_of_memory");
                assert_eq!(last.stage, "after_error_cleanup");
                assert_eq!(
                    last.retained_session_capacity_bytes
                        + last.expert_cache_capacity_bytes
                        + last.recycled_capacity_bytes,
                    0
                );
            } else {
                assert_eq!(trace.outcome, "interrupted");
                assert!(model.sessions.contains_key("old"));
            }
            assert!(model.generate("[BOS]token3 ", 3).is_ok());
        }
    }

    #[test]
    fn memory_trace_prefill_logits_and_parser() {
        assert!(!parse_memory_trace_mode(None).unwrap());
        assert!(!parse_memory_trace_mode(Some("off")).unwrap());
        assert!(parse_memory_trace_mode(Some("on")).unwrap());
        assert!(parse_memory_trace_mode(Some("yes")).is_err());
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let mut expected: Option<Vec<f32>> = None;
            for enabled in [false, true] {
                let mut model = model(fixture);
                let mut profile = RequestProfile::default();
                if enabled {
                    profile.memory_trace = Some(RequestMemoryTrace {
                        schema_version: 1,
                        outcome: "running",
                        samples: Vec::new(),
                    });
                    profile.memory_started = Some(Instant::now());
                }
                let mut kv: Vec<_> = (0..model.layers.len()).map(|_| Kv::default()).collect();
                let mut scratch = StepScratch::new(&model.config);
                model
                    .prefill(&[1, 3, 4], &mut kv, &mut profile, &mut scratch, &|| Ok(()))
                    .unwrap();
                if let Some(ref logits) = expected {
                    let error = logits
                        .iter()
                        .zip(&scratch.logits)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(error < 2e-5, "{fixture}: {error}");
                    assert_eq!(profile.memory_trace.as_ref().unwrap().samples.len(), 1);
                } else {
                    expected = Some(scratch.logits.clone());
                }
            }
        }
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn layer_major_prefill_matches_tokenwise_last_logits() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/olmoe")
                .join(fixture);
            let reference: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join("reference.json")).unwrap())
                    .unwrap();
            let tokens: Vec<_> = reference["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect();
            let mut model = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
            let expected = model.logits(&tokens).unwrap().pop().unwrap();
            let mut kv: Vec<_> = (0..model.layers.len()).map(|_| Kv::default()).collect();
            let mut profile = RequestProfile::default();
            let mut scratch = StepScratch::new(&model.config);
            model
                .prefill(&tokens, &mut kv, &mut profile, &mut scratch, &|| Ok(()))
                .unwrap();
            let error = scratch
                .logits
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 2e-5, "{fixture}: max logit error={error}");
            let kd = model.config.hidden_size / model.config.num_attention_heads
                * model.config.num_key_value_heads;
            assert!(kv.iter().all(|cache| cache.keys.len() / kd == tokens.len()));
        }
    }

    #[test]
    fn undersized_pinned_staging_falls_back_and_preserves_logits() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut model = OlmoeModel::load(&directory, 512, 64, 10_000_000).unwrap();
        let Some(storage) = model.gpu_dense.as_mut() else {
            eprintln!("SKIPPED undersized pinned CUDA execution: GPU unavailable");
            return;
        };
        storage.pinned_upload_buf =
            Some(crate::gpu::PinnedBuffer::alloc(&storage.engine, 8).unwrap());
        storage.expert_cache.max_bytes = 1024;
        storage.expert_pool =
            Some(crate::gpu::GpuExpertPool::new(&storage.engine, 2, 384, 128).unwrap());
        model.victim_expert_cache = true;
        model.set_gpu_enabled(false);
        let tokens = [1, 3, 4, 5, 6, 7, 8];
        let expected = model.logits(&tokens).unwrap();
        model.reset_cache();
        model.set_gpu_enabled(true);
        let actual = model.logits(&tokens).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            let error = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                error < 2e-5,
                "undersized pinned fallback logit error={error}"
            );
        }
        let storage = model.gpu_dense.as_ref().unwrap();
        assert!(storage.engine.is_healthy());
        assert!(
            !storage.expert_cache.is_empty(),
            "fallback must upload through the pool"
        );
        let pinned = storage.pinned_upload_buf.as_ref().unwrap();
        assert_eq!(pinned.len(), 8);
        assert!(
            unsafe { pinned.as_slice::<u8>() }
                .iter()
                .all(|&byte| byte == 0),
            "undersized staging must stay untouched"
        );
        eprintln!("EXECUTED undersized pinned staging fallback/logits on CUDA");
    }

    #[test]
    fn victim_cache_gpu_keeps_tiers_disjoint_and_matches_cpu_logits() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut model = OlmoeModel::load(&directory, 512, 64, 10_000_000).unwrap();
        let Some(storage) = model.gpu_dense.as_mut() else {
            eprintln!("SKIPPED victim cache CUDA execution: GPU unavailable");
            return;
        };
        storage.expert_cache.max_bytes = 1024;
        storage.expert_pool =
            Some(crate::gpu::GpuExpertPool::new(&storage.engine, 2, 384, 128).unwrap());
        model.victim_expert_cache = true;
        model.set_gpu_enabled(false);
        let tokens = [1, 3, 4, 5, 6, 7, 8];
        let expected = model.logits(&tokens).unwrap();
        model.reset_cache();
        model.set_gpu_enabled(true);
        let actual = model.logits(&tokens).unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            let error = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 2e-5, "victim CUDA logit error={error}");
        }
        let storage = model.gpu_dense.as_ref().unwrap();
        assert!(storage.engine.is_healthy());
        assert!(
            !storage.expert_cache.is_empty(),
            "evictions should populate VRAM"
        );
        for &(layer, expert) in storage.expert_cache.entries.keys() {
            assert!(!model.layers[layer]
                .cache
                .iter()
                .any(|entry| entry.id == expert));
        }
        model.reset_cache();
        let storage = model.gpu_dense.as_ref().unwrap();
        assert!(storage.expert_cache.is_empty());
        let pool = storage.expert_pool.as_ref().unwrap();
        assert_eq!(pool.available_slots(), pool.max_slots);
        // If RAM cannot admit even one expert, keep L1 useful as the only cache.
        model.cache_bytes_per_layer = 0;
        let storage = model.gpu_dense.as_mut().unwrap();
        storage.expert_cache.max_bytes = 8 * 512;
        storage.expert_pool =
            Some(crate::gpu::GpuExpertPool::new(&storage.engine, 8, 384, 128).unwrap());
        let result = model.generate("[BOS]token3 token4 ", 4).unwrap();
        assert!(model.layers.iter().all(|layer| layer.cache.is_empty()));
        let metrics = result.metrics.unwrap();
        assert!(metrics.expert_vram_upload_bytes > 0);
        assert!(metrics.expert_vram_loads > 0);
        eprintln!("EXECUTED victim cache CUDA: logits, disjoint tiers, pool reset passed");
    }

    #[test]
    fn expert_io_metrics_separate_phases_and_file_payload() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut model = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
        model.set_gpu_enabled(false);
        for limit in [1, 4] {
            model.reset_cache();
            let metrics = model
                .generate("[BOS]token3 token4 ", limit)
                .unwrap()
                .metrics
                .unwrap();
            assert!(metrics.expert_file_loads > 0);
            let payload = model.config.hidden_size * model.config.intermediate_size * 3
                + (model.config.intermediate_size * 2 + model.config.hidden_size) * 4;
            assert_eq!(
                metrics.expert_file_bytes_read,
                metrics.expert_file_loads * payload as u64
            );
            assert!((metrics.expert_io_ms - metrics.expert_file_read_ms).abs() < 1e-6);
            assert!(
                (metrics.expert_io_ms - metrics.prefill_expert_io_ms - metrics.decode_expert_io_ms)
                    .abs()
                    < 1e-6
            );
            assert_eq!(metrics.expert_vram_download_bytes, 0);
            assert_eq!(metrics.expert_vram_upload_bytes, 0);
            if limit == 1 {
                assert_eq!(metrics.decode_expert_io_ms, 0.0);
            }
        }
    }

    #[test]
    fn session_prefix_cache_matches_fresh_generation() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut cached = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
        cached
            .generate_controlled(
                "[BOS]token3 ",
                1,
                Some("test-session"),
                &|| Ok(()),
                || Ok(()),
                None,
            )
            .unwrap();
        let reused = cached
            .generate_controlled(
                "[BOS]token3 token4 ",
                3,
                Some("test-session"),
                &|| Ok(()),
                || Ok(()),
                None,
            )
            .unwrap();
        let mut fresh = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
        let expected = fresh.generate("[BOS]token3 token4 ", 3).unwrap();
        assert_eq!(reused.token_ids, expected.token_ids);
        assert!(
            reused.metrics.unwrap().cached_prompt_tokens > 0,
            "the exact token prefix should reuse KV"
        );
    }

    #[test]
    fn session_reuses_common_prefix_after_edit_and_repeated_prompt() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/olmoe")
                .join(fixture);
            let mut model = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
            model.set_gpu_enabled(false);
            for prompt in [
                "[BOS]token3 token4 token5 ",
                "[BOS]token3 token4 token6 ",
                "[BOS]token3 token4 token6 ",
                "[BOS]token3 ",
                "token7 token8 ",
            ] {
                let tokens: Vec<_> = model
                    .tokenizer
                    .as_ref()
                    .unwrap()
                    .encode(prompt, false)
                    .unwrap()
                    .get_ids()
                    .iter()
                    .map(|&id| id as usize)
                    .collect();
                let expected_cached = model.sessions.get("edited-session").map_or(0, |state| {
                    tokens
                        .iter()
                        .zip(&state.tokens)
                        .take_while(|(a, b)| a == b)
                        .count()
                        .min(tokens.len() - 1)
                });
                let reused = model
                    .generate_controlled(
                        prompt,
                        3,
                        Some("edited-session"),
                        &|| Ok(()),
                        || Ok(()),
                        None,
                    )
                    .unwrap();
                assert_eq!(
                    reused.metrics.unwrap().cached_prompt_tokens,
                    expected_cached,
                    "{fixture}, prompt={prompt}"
                );
                let expected = model.generate(prompt, 3).unwrap();
                assert_eq!(reused.token_ids, expected.token_ids, "{fixture}, {prompt}");
            }
        }
    }

    #[test]
    fn truncated_session_kv_preserves_logits_and_buffer_ownership() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/olmoe")
                .join(fixture);
            let mut model = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
            model.set_gpu_enabled(false);
            model
                .generate_controlled(
                    "[BOS]token3 token4 token5 ",
                    1,
                    Some("fork"),
                    &|| Ok(()),
                    || Ok(()),
                    None,
                )
                .unwrap();
            let addresses: Vec<_> = model.sessions["fork"]
                .kv
                .iter()
                .map(|cache| (cache.keys.as_ptr(), cache.values.as_ptr()))
                .collect();
            let tokens = [1, 3, 4, 6];
            let expected = model.logits(&tokens).unwrap().pop().unwrap();
            let kd = model.config.hidden_size / model.config.num_attention_heads
                * model.config.num_key_value_heads;
            let (prefix, mut kv, _) = model.take_prefix_kv(
                &tokens,
                Some("fork"),
                kd,
                1,
                &mut KvBufferReuseMetrics::default(),
            );
            assert_eq!(prefix, 3);
            assert!(!model.sessions.contains_key("fork"));
            for (cache, addresses) in kv.iter().zip(addresses) {
                assert_eq!((cache.keys.as_ptr(), cache.values.as_ptr()), addresses);
                assert_eq!(cache.keys.len(), prefix * kd);
                assert_eq!(cache.values.len(), prefix * kd);
            }
            let mut profile = RequestProfile::default();
            let mut scratch = StepScratch::new(&model.config);
            model
                .prefill(
                    &tokens[prefix..],
                    &mut kv,
                    &mut profile,
                    &mut scratch,
                    &|| Ok(()),
                )
                .unwrap();
            let error = scratch
                .logits
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 2e-5, "{fixture}: truncated KV logit error={error}");
        }
    }

    #[test]
    fn reset_cache_preserves_session_budget_and_allows_new_prefix_reuse() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut model = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
        let budget = model.session_cache_bytes;
        assert!(budget > 0);
        for cycle in 0..2 {
            model
                .generate_controlled(
                    "[BOS]token3 ",
                    1,
                    Some("reset-session"),
                    &|| Ok(()),
                    || Ok(()),
                    None,
                )
                .unwrap();
            assert!(!model.sessions.is_empty(), "cycle {cycle}");
            model.reset_cache();
            assert!(model.sessions.is_empty());
            assert!(model.layers.iter().all(|layer| {
                layer.cache.is_empty() && layer.recycled_buffers.is_empty() && layer.bytes == 0
            }));
            assert_eq!(
                model.session_cache_bytes, budget,
                "reset must retain budget"
            );
            let first = model
                .generate_controlled(
                    "[BOS]token3 ",
                    1,
                    Some("reset-session"),
                    &|| Ok(()),
                    || Ok(()),
                    None,
                )
                .unwrap();
            assert_eq!(first.metrics.unwrap().cached_prompt_tokens, 0);
            let reused = model
                .generate_controlled(
                    "[BOS]token3 token4 ",
                    3,
                    Some("reset-session"),
                    &|| Ok(()),
                    || Ok(()),
                    None,
                )
                .unwrap();
            let metrics = reused.metrics.unwrap();
            assert!(metrics.cached_prompt_tokens > 0, "cycle {cycle}");
            assert!(metrics.cached_prompt_tokens < reused.prompt_tokens);
            let expected = model.generate("[BOS]token3 token4 ", 3).unwrap();
            assert_eq!(reused.token_ids, expected.token_ids);
        }
    }

    #[test]
    fn prefix_prefill_matches_full_logits_for_every_split() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/olmoe")
                .join(fixture);
            let reference: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join("reference.json")).unwrap())
                    .unwrap();
            let tokens: Vec<_> = reference["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect();
            let mut model = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
            let expected = model.logits(&tokens).unwrap().pop().unwrap();
            for split in 1..tokens.len() {
                let mut kv: Vec<_> = (0..model.layers.len()).map(|_| Kv::default()).collect();
                let mut profile = RequestProfile::default();
                let mut scratch = StepScratch::new(&model.config);
                for part in [&tokens[..split], &tokens[split..]] {
                    model
                        .prefill(part, &mut kv, &mut profile, &mut scratch, &|| Ok(()))
                        .unwrap();
                }
                assert_eq!(scratch.logits.len(), expected.len());
                let error = scratch
                    .logits
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    error < 2e-5,
                    "{fixture}, split {split}: logit error={error}"
                );
                let kd = model.config.hidden_size / model.config.num_attention_heads
                    * model.config.num_key_value_heads;
                assert!(kv.iter().all(|cache| cache.keys.len() / kd == tokens.len()));
            }
        }
    }

    #[test]
    #[ignore = "release-only paired model prefill experiment; requires MOE_PREFILL_MODEL/REPORT/TESTBED"]
    fn benchmark_prefill_dispatch_paired() {
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(!cfg!(debug_assertions), "run with --release");
        }
        let testbed = std::env::var("MOE_PREFILL_TESTBED").expect("explicit i5/i7 required");
        assert!(matches!(testbed.as_str(), "i5" | "i7"));
        let model_path = std::env::var("MOE_PREFILL_MODEL").expect("set model directory");
        let output = std::env::var("MOE_PREFILL_REPORT").expect("set unique output path");
        let mut model =
            OlmoeModel::load(&model_path, 128 * 1024 * 1024, 512, 3 * 1024 * 1024 * 1024).unwrap();
        let prompt = model
            .render_chat(&[crate::engine::ChatMessage {
                role: "user".into(),
                content: "Where is Mars ?".into(),
            }])
            .unwrap();
        let gpu_start = model.expert_status().gpu;
        // Warm both paths before sampling; no cache reset between measurements.
        for serial in [true, false, true, false] {
            model.serial_prefill_reference = serial;
            model.generate(&prompt, 1).unwrap();
        }
        let mut samples = Vec::new();
        let mut expected = None;
        for pair in 0..9 {
            let order = if pair % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            };
            for serial in order {
                model.serial_prefill_reference = serial;
                let response = model.generate(&prompt, 1).unwrap();
                if let Some(tokens) = &expected {
                    assert_eq!(&response.token_ids, tokens);
                } else {
                    expected = Some(response.token_ids.clone());
                }
                eprintln!(
                    "pair {pair}, serial={serial}, prefill_ms={:.3}",
                    response.metrics.as_ref().unwrap().prefill_ms
                );
                samples.push(serde_json::json!({
                    "pair": pair, "variant": if serial { "before" } else { "after" },
                    "response": response
                }));
            }
        }
        let report = serde_json::json!({
            "schema_version": 1, "scope": "exploratory paired prefill experiment",
            "testbed": testbed, "model_directory": model_path,
            "prompt": "Where is Mars ?", "max_tokens": 1, "cache_mib_per_layer": 128,
            "context": 512, "threads": rayon::current_num_threads(),
            "gpu_start": gpu_start, "gpu_end": model.expert_status().gpu,
            "warmup": 4, "samples": samples,
            "notes": ["one loaded model; alternating order within each pair; expert caches evolve",
                "one output token isolates prefill; no session KV reuse",
                "same checkpoint as runner A/B; OS cache and desktop load uncontrolled",
                "serial reference is compiled only in tests; not a production runtime toggle"]
        });
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap();
        serde_json::to_writer_pretty(&mut file, &report).unwrap();
    }

    #[test]
    #[ignore = "release-only paired session experiment; requires MOE_PREFILL_MODEL/REPORT/TESTBED"]
    fn benchmark_session_prefix_paired() {
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(!cfg!(debug_assertions), "run with --release");
        }
        let testbed = std::env::var("MOE_PREFILL_TESTBED").expect("explicit i5/i7 required");
        assert!(matches!(testbed.as_str(), "i5" | "i7"));
        let model_path = std::env::var("MOE_PREFILL_MODEL").expect("set model directory");
        let output = std::env::var("MOE_PREFILL_REPORT").expect("set unique output path");
        let mut model =
            OlmoeModel::load(&model_path, 128 * 1024 * 1024, 512, 3 * 1024 * 1024 * 1024).unwrap();
        let context = "Context note: Mars is a planet in our solar system. ".repeat(8);
        let render = |model: &OlmoeModel, question: &str| {
            model
                .render_chat(&[crate::engine::ChatMessage {
                    role: "user".into(),
                    content: format!("{context}{question}"),
                }])
                .unwrap()
        };
        let seed_prompt = render(&model, "Where is Mars ?");
        let target_prompt = render(&model, "What is love ?");
        model
            .generate_controlled(
                &seed_prompt,
                1,
                Some("paired-session"),
                &|| Ok(()),
                || Ok(()),
                None,
            )
            .unwrap();
        let seed = model.sessions.remove("paired-session").unwrap();
        let gpu_start = model.expert_status().gpu;
        let mut samples = Vec::new();
        let mut expected = None;
        // Clone a fixed seed outside generation timing, preserving its reserved capacities.
        // Warm both variants once, then collect alternating paired samples.
        for pair in 0..6 {
            let order = if pair % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            };
            for strict in order {
                let kv = seed.kv.clone();
                model.sessions.insert(
                    "paired-session".into(),
                    SessionState {
                        tokens: seed.tokens.clone(),
                        kv,
                        last_used: model.session_clock,
                    },
                );
                model.strict_session_prefix_reference = strict;
                let response = model
                    .generate_controlled(
                        &target_prompt,
                        1,
                        Some("paired-session"),
                        &|| Ok(()),
                        || Ok(()),
                        None,
                    )
                    .unwrap();
                if let Some(tokens) = &expected {
                    assert_eq!(&response.token_ids, tokens);
                } else {
                    expected = Some(response.token_ids.clone());
                }
                let metrics = response.metrics.as_ref().unwrap();
                if strict {
                    assert_eq!(metrics.cached_prompt_tokens, 0);
                } else {
                    assert!(metrics.cached_prompt_tokens > 0);
                }
                eprintln!(
                    "pair {pair}, strict={strict}, prefill_ms={:.3}, cached={}",
                    metrics.prefill_ms, metrics.cached_prompt_tokens
                );
                if pair > 0 {
                    samples.push(serde_json::json!({
                        "pair": pair, "variant": if strict { "before" } else { "after" },
                        "response": response
                    }));
                }
            }
        }
        let report = serde_json::json!({
            "schema_version": 1, "scope": "exploratory edited-session prefix experiment",
            "testbed": testbed, "model_directory": model_path,
            "seed_prompt": seed_prompt, "target_prompt": target_prompt,
            "max_tokens": 1, "cache_mib_per_layer": 128, "context": 512,
            "threads": rayon::current_num_threads(), "gpu_start": gpu_start,
            "gpu_end": model.expert_status().gpu, "warmup_pairs": 1, "samples": samples,
            "notes": ["fixed seed KV restored outside generation timing; one model; alternating order",
                "edited last question sharing a long prefix; not the user's ordinary two-turn chat",
                "one output token isolates prefill; OS cache and desktop load uncontrolled",
                "strict reference is compiled only in tests; same final serial expert dispatch"]
        });
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap();
        serde_json::to_writer_pretty(&mut file, &report).unwrap();
    }

    #[test]
    fn parallel_prefill_routes_match_tokenwise_logits_with_tails() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    for fixture in ["unnormalized-mha", "normalized-gqa"] {
                        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                            .join("tests/fixtures/olmoe")
                            .join(fixture);
                        let mut model =
                            OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
                        model.set_gpu_enabled(false);
                        model.serial_prefill_reference = false;
                        for length in [1, 9, 17] {
                            let tokens: Vec<_> = (0..length).map(|i| [1, 3, 4, 5][i % 4]).collect();
                            let expected = model.logits(&tokens).unwrap().pop().unwrap();
                            let mut kv: Vec<_> =
                                (0..model.layers.len()).map(|_| Kv::default()).collect();
                            let mut profile = RequestProfile::default();
                            let mut scratch = StepScratch::new(&model.config);
                            model
                                .prefill(&tokens, &mut kv, &mut profile, &mut scratch, &|| Ok(()))
                                .unwrap();
                            let error = scratch
                                .logits
                                .iter()
                                .zip(&expected)
                                .map(|(actual, expected)| (actual - expected).abs())
                                .fold(0.0f32, f32::max);
                            assert!(
                                error < 2e-5,
                                "{fixture}, threads={threads}, tokens={length}: error={error}"
                            );
                        }
                    }
                });
        }
    }

    #[test]
    fn disconnect_after_content_stops_decode_and_preserves_model() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut model = OlmoeModel::load(directory, 10_000, 64, 10_000_000).unwrap();
        let prompt = "[BOS]token3 token4 ";
        let expected = model.generate(prompt, 8).unwrap();
        let mut chunks = 0;
        let error = model
            .generate_controlled(
                prompt,
                8,
                None,
                &|| Ok(()),
                || Ok(()),
                Some(&mut |_| {
                    chunks += 1;
                    Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "closed receiver",
                    ))
                }),
            )
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert_eq!(chunks, 1);
        assert_eq!(
            model.generate(prompt, 8).unwrap().token_ids,
            expected.token_ids
        );
    }

    #[test]
    fn cancellation_during_prefill_leaves_model_reusable() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut model = OlmoeModel::load(directory, 10_000, 64, 10_000_000).unwrap();
        let prompt = "[BOS]token3 token4 ";
        let expected = model.generate(prompt, 3).unwrap();
        let checks = Cell::new(0);
        let error = model
            .generate_controlled(
                prompt,
                3,
                None,
                &|| {
                    checks.set(checks.get() + 1);
                    if checks.get() == 5 {
                        Err(io::Error::new(io::ErrorKind::Interrupted, "cancel"))
                    } else {
                        Ok(())
                    }
                },
                || Ok(()),
                None,
            )
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert_eq!(checks.get(), 5);
        assert_eq!(
            model.generate(prompt, 3).unwrap().token_ids,
            expected.token_ids
        );
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod routing_and_topk_tests {
    use super::*;

    #[test]
    fn test_zero_alloc_topk_matches_full_sort() {
        let num_experts = 64;
        let top_k = 8;

        // Test 1: Các giá trị phân tán
        let mut router = vec![0.0f32; num_experts];
        for i in 0..num_experts {
            router[i] = ((i * 17 + 31) % 100) as f32 / 10.0;
        }

        let mut legacy_ids: Vec<_> = (0..num_experts).collect();
        legacy_ids.sort_by(|&a, &b| router[b].total_cmp(&router[a]).then_with(|| a.cmp(&b)));
        legacy_ids.truncate(top_k);

        let mut indices = [0usize; 64];
        for i in 0..num_experts {
            indices[i] = i;
        }
        let valid = &mut indices[..num_experts];
        valid.select_nth_unstable_by(top_k, |&a, &b| {
            router[b].total_cmp(&router[a]).then_with(|| a.cmp(&b))
        });
        let top = &mut valid[..top_k];
        top.sort_by(|&a, &b| router[b].total_cmp(&router[a]).then_with(|| a.cmp(&b)));

        assert_eq!(legacy_ids.as_slice(), top);

        // Test 2: Các phần tử trùng lặp / ties
        for i in 0..num_experts {
            router[i] = (i % 5) as f32;
        }
        let mut legacy_ids: Vec<_> = (0..num_experts).collect();
        legacy_ids.sort_by(|&a, &b| router[b].total_cmp(&router[a]).then_with(|| a.cmp(&b)));
        legacy_ids.truncate(top_k);

        for i in 0..num_experts {
            indices[i] = i;
        }
        let valid = &mut indices[..num_experts];
        valid.select_nth_unstable_by(top_k, |&a, &b| {
            router[b].total_cmp(&router[a]).then_with(|| a.cmp(&b))
        });
        let top = &mut valid[..top_k];
        top.sort_by(|&a, &b| router[b].total_cmp(&router[a]).then_with(|| a.cmp(&b)));

        assert_eq!(legacy_ids.as_slice(), top);
    }

    #[test]
    fn test_prefill_direct_routing_matches_legacy_scan() {
        let num_experts = 64;
        let num_tokens = 10;
        let top_k = 8;

        let mut token_routes: Vec<Vec<(usize, f32)>> = Vec::new();
        for t in 0..num_tokens {
            let mut routes = Vec::new();
            for s in 0..top_k {
                let expert = (t * 7 + s * 3) % num_experts;
                let weight = 1.0 / (s + 1) as f32;
                routes.push((expert, weight));
            }
            token_routes.push(routes);
        }

        // Method 1: Legacy flat_map scan
        #[derive(Clone, Copy)]
        struct LegacyRoute {
            expert: usize,
            weight: f32,
        }
        let legacy_token_routes: Vec<Vec<LegacyRoute>> = token_routes
            .iter()
            .map(|r| {
                r.iter()
                    .map(|&(expert, weight)| LegacyRoute { expert, weight })
                    .collect()
            })
            .collect();

        let mut legacy_uses_by_expert: Vec<Vec<(usize, usize, f32)>> = Vec::new();
        for expert_id in 0..num_experts {
            let uses: Vec<_> = legacy_token_routes
                .iter()
                .enumerate()
                .flat_map(|(token, routes)| {
                    routes
                        .iter()
                        .enumerate()
                        .filter(move |(_, route)| route.expert == expert_id)
                        .map(move |(slot, route)| (token, slot, route.weight))
                })
                .collect();
            legacy_uses_by_expert.push(uses);
        }

        // Method 2: Direct accumulation
        let mut direct_uses_by_expert: Vec<Vec<(usize, usize, f32)>> =
            vec![Vec::new(); num_experts];
        for (token, routes) in token_routes.iter().enumerate() {
            for (slot, &(expert, weight)) in routes.iter().enumerate() {
                direct_uses_by_expert[expert].push((token, slot, weight));
            }
        }

        assert_eq!(legacy_uses_by_expert, direct_uses_by_expert);
    }

    #[test]
    fn test_argmax_greedy_matches_legacy_closure() {
        let vocab_size = 50_304;

        // Case 1: Random logits with a clear peak
        let mut logits: Vec<f32> = (0..vocab_size)
            .map(|i| (i as f32 * 0.013).sin() * 5.0)
            .collect();
        logits[12_345] = 99.0;

        let legacy = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        let fast = argmax_greedy(&logits);
        assert_eq!(fast, legacy);
        assert_eq!(fast, 12_345);

        // Case 2: Duplicate maximums - check tie breaking (smallest index wins)
        let mut tie_logits = vec![0.0f32; 100];
        tie_logits[10] = 5.0;
        tie_logits[42] = 5.0;
        tie_logits[88] = 5.0;
        let legacy_tie = (0..tie_logits.len())
            .max_by(|&a, &b| {
                tie_logits[a]
                    .total_cmp(&tie_logits[b])
                    .then_with(|| b.cmp(&a))
            })
            .unwrap();
        let fast_tie = argmax_greedy(&tie_logits);
        assert_eq!(fast_tie, legacy_tie);
        assert_eq!(fast_tie, 10);

        // Case 3: All negative values
        let neg_logits: Vec<f32> = (0..100).map(|i| -100.0 + i as f32).collect();
        let legacy_neg = (0..neg_logits.len())
            .max_by(|&a, &b| {
                neg_logits[a]
                    .total_cmp(&neg_logits[b])
                    .then_with(|| b.cmp(&a))
            })
            .unwrap();
        let fast_neg = argmax_greedy(&neg_logits);
        assert_eq!(fast_neg, legacy_neg);
        assert_eq!(fast_neg, 99);
    }

    #[test]
    fn test_rope_fast_path_matches_legacy_rotate() {
        let hd = 128;
        let num_heads = 16;
        let d = hd * num_heads;
        let theta = 500_000.0f32;
        let position = 42;

        let original: Vec<f32> = (0..d).map(|i| (i as f32 * 0.07).sin()).collect();

        // Legacy rotate
        let mut legacy_x = original.clone();
        rotate(&mut legacy_x, hd, position, theta);

        // Fast cached rotate
        let half = hd / 2;
        let rope_bases: Vec<f32> = (0..half)
            .map(|i| theta.powf(2.0 * i as f32 / hd as f32))
            .collect();
        let mut sincos = vec![(0.0f32, 0.0f32); half];
        prepare_rope_sincos(&mut sincos, &rope_bases, position);

        let mut fast_x = original.clone();
        apply_rope_cached(&mut fast_x, hd, &sincos);

        for i in 0..d {
            let diff = (legacy_x[i] - fast_x[i]).abs();
            assert!(
                diff < 1e-6,
                "RoPE mismatch at {}: legacy={}, fast={}, diff={}",
                i,
                legacy_x[i],
                fast_x[i],
                diff
            );
        }
    }

    #[test]
    fn ram_pressure_rejects_and_clears_host_cache_then_allows_retry() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut model = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
        model.set_gpu_enabled(false);
        model
            .generate_controlled(
                "[BOS]token3 ",
                3,
                Some("ram-test"),
                &|| Ok(()),
                || Ok(()),
                None,
            )
            .unwrap();
        assert!(!model.sessions.is_empty());
        assert!(model.layers.iter().any(|layer| !layer.cache.is_empty()));
        let guard = model.ram_guard;
        model.ram_guard.rss_limit = 1;
        let ready = std::cell::Cell::new(false);
        let error = model
            .generate_controlled(
                "[BOS]token3 ",
                3,
                Some("ram-test"),
                &|| Ok(()),
                || {
                    ready.set(true);
                    Ok(())
                },
                None,
            )
            .err()
            .unwrap();
        assert_eq!(error.kind(), io::ErrorKind::OutOfMemory);
        assert!(!ready.get(), "reject before SSE ready");
        assert!(model.sessions.is_empty());
        assert!(model
            .layers
            .iter()
            .all(|l| l.bytes == 0 && l.cache.is_empty() && l.recycled_buffers.is_empty()));
        model.ram_guard = guard;
        assert!(model.generate("[BOS]token3 ", 3).is_ok());
    }

    #[test]
    fn expert_retained_budget_covers_cache_and_recycled_after_churn() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        for budget in [0, 512, 900, 1536] {
            let mut model = OlmoeModel::load(&directory, budget, 64, 10_000_000).unwrap();
            model.set_gpu_enabled(false);
            let mut reference = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
            reference.set_gpu_enabled(false);
            for prompt in [
                "[BOS]token3 token4 token5 ",
                "[BOS]token8 token9 ",
                "[BOS]token10 token11 token12 ",
            ] {
                let expected = reference.generate(prompt, 8).unwrap();
                let actual = model.generate(prompt, 8).unwrap();
                assert_eq!(actual.token_ids, expected.token_ids, "budget={budget}");
                for layer in &model.layers {
                    assert_eq!(
                        layer.bytes,
                        layer
                            .cache
                            .iter()
                            .map(|e| e.expert.allocated_bytes())
                            .sum::<usize>()
                    );
                    assert!(
                        layer.bytes + layer.recycled_bytes() <= budget,
                        "budget={budget}, cached={}, recycled={}",
                        layer.bytes,
                        layer.recycled_bytes()
                    );
                }
            }
            model.reset_cache();
            assert!(model
                .layers
                .iter()
                .all(|l| l.bytes + l.recycled_bytes() == 0));
        }
    }

    #[test]
    fn prefill_prefetch_is_bounded_and_recycled_capacity_is_charged() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut model = OlmoeModel::load(&directory, 900, 64, 10_000_000).unwrap();
        model.set_gpu_enabled(false);
        let layer = &mut model.layers[0];
        layer.recycle((Vec::with_capacity(384), Vec::with_capacity(32)), 900);
        assert_eq!(layer.recycled_bytes(), 512);
        layer.recycle((Vec::with_capacity(384), Vec::with_capacity(32)), 900);
        assert_eq!(layer.recycled_buffers.len(), 1);
        let (prefetched, _) = OlmoeModel::prefetch_missing_experts(
            0,
            &[0, 1, 2, 3],
            16,
            8,
            &model.index,
            layer,
            None,
            false,
            2,
        );
        assert_eq!(
            prefetched.len(),
            2,
            "do not allocate all missing experts at once"
        );
        assert!(
            prefetched
                .values()
                .map(Int8Expert::allocated_bytes)
                .sum::<usize>()
                <= 1024
        );
    }

    #[test]
    fn session_budget_counts_reserved_capacity_for_short_prompts() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut model = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
        model.set_gpu_enabled(false);
        model.session_cache_bytes = 1024;
        for id in ["short-a", "short-b"] {
            model
                .generate_controlled("[BOS]token3 ", 1, Some(id), &|| Ok(()), || Ok(()), None)
                .unwrap();
        }
        assert!(
            model.sessions.is_empty(),
            "full-context buffers exceed the small session budget"
        );
        model.session_cache_bytes = 20_000;
        for id in ["short-a", "short-b", "short-c"] {
            model
                .generate_controlled("[BOS]token3 ", 1, Some(id), &|| Ok(()), || Ok(()), None)
                .unwrap();
            let retained: usize = model
                .sessions
                .iter()
                .map(|(id, state)| {
                    id.capacity()
                        + std::mem::size_of::<SessionState>()
                        + state.tokens.capacity() * std::mem::size_of::<usize>()
                        + state.kv.capacity() * std::mem::size_of::<Kv>()
                        + state
                            .kv
                            .iter()
                            .map(|kv| (kv.keys.capacity() + kv.values.capacity()) * 4)
                            .sum::<usize>()
                })
                .sum();
            assert!(retained <= model.session_cache_bytes);
        }
        assert_eq!(model.sessions.len(), 1, "evict older reserved buffers");
        assert!(model.sessions.contains_key("short-c"));
    }

    #[test]
    fn test_kv_cache_preallocation_capacity() {
        let context_limit = 512;
        let kd = 1024;
        let kv = Kv::with_capacity(context_limit, kd);
        assert_eq!(kv.keys.capacity(), context_limit * kd);
        assert_eq!(kv.values.capacity(), context_limit * kd);
        assert_eq!(kv.keys.len(), 0);
        assert_eq!(kv.values.len(), 0);
    }

    #[test]
    fn test_zero_finite_overhead_still_catches_nan() {
        let mut x = vec![1.0f32; 2048];
        x[42] = f32::NAN;
        let result = crate::backend::finite(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_topk_bounds_safety() {
        // Kiểm tra trường hợp biên: top_k == num_experts không được gây panic
        let mut router = vec![0.1f32; 64];
        router[3] = 0.9;
        router[7] = 0.8;
        let mut expert_indices = [0usize; 64];
        let num_experts = 64;
        let top_k = 64; // top_k == valid_indices.len()
        for (i, idx) in expert_indices.iter_mut().enumerate().take(num_experts) {
            *idx = i;
        }
        let valid_indices = &mut expert_indices[..num_experts];
        if top_k < valid_indices.len() {
            valid_indices.select_nth_unstable_by(top_k, |&a, &b| {
                router[b].total_cmp(&router[a]).then_with(|| a.cmp(&b))
            });
        }
        let top_experts = &mut valid_indices[..top_k];
        top_experts.sort_by(|&a, &b| router[b].total_cmp(&router[a]).then_with(|| a.cmp(&b)));
        assert_eq!(top_experts[0], 3);
        assert_eq!(top_experts[1], 7);
        assert_eq!(top_experts.len(), 64);
    }

    #[test]
    fn test_sf_cache_partition_and_protection() {
        // Kiểm tra tính bảo vệ của phân vùng Segmented Frequency-Recency (SF-Cache)
        let d = 4;
        let intermediate = 4;
        let make_expert = || {
            let weights = vec![1u8; d * intermediate * 3];
            let scales = vec![1.0f32; intermediate * 2 + d];
            Int8Expert::from_raw(d, intermediate, weights, scales)
        };

        let expert_bytes = make_expert().resident_bytes();
        let cache_capacity = expert_bytes * 4; // Ngân sách chứa tối đa 4 experts

        let mut layer = Layer {
            input_norm: vec![],
            post_norm: vec![],
            q_norm: vec![],
            k_norm: vec![],
            q: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            k: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            v: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            o: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            router: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            cache: VecDeque::new(),
            bytes: 0,
            heat: vec![0; 64],
            clock: 0,
            recycled_buffers: Vec::new(),
            hot_pinned: 0,
            router_ema: vec![0.0; 64],
        };
        let mut profile = RequestProfile::default();

        // 1. Nạp expert 10 (miss) -> Probationary
        layer.clock = 1;
        layer.heat[10] = 1;
        OlmoeModel::admit_expert_l2(
            &mut layer,
            10,
            make_expert(),
            false,
            1,
            false,
            cache_capacity,
            &mut profile,
            &mut None,
            false,
            false,
            0,
        );
        assert_eq!(layer.cache.len(), 1);
        assert!(!layer.cache[0].is_protected);

        // 2. Expert 10 được hit (acquire_expert_tiered lấy ra khỏi cache để tính toán) -> Thăng hạng lên Protected
        let _ = layer.cache.pop_back().unwrap();
        layer.bytes -= expert_bytes;
        layer.clock = 2;
        layer.heat[10] = 2;
        OlmoeModel::admit_expert_l2(
            &mut layer,
            10,
            make_expert(),
            true,
            2,
            false,
            cache_capacity,
            &mut profile,
            &mut None,
            false,
            false,
            0,
        );
        assert_eq!(layer.cache.len(), 1);
        assert!(layer.cache[0].is_protected);

        // 3. Nạp expert 20 và thăng hạng lên Protected
        layer.clock = 3;
        layer.heat[20] = 1;
        OlmoeModel::admit_expert_l2(
            &mut layer,
            20,
            make_expert(),
            false,
            1,
            false,
            cache_capacity,
            &mut profile,
            &mut None,
            false,
            false,
            0,
        );
        let _ = layer.cache.pop_back().unwrap();
        layer.bytes -= expert_bytes;
        layer.clock = 4;
        layer.heat[20] = 2;
        OlmoeModel::admit_expert_l2(
            &mut layer,
            20,
            make_expert(),
            true,
            2,
            false,
            cache_capacity,
            &mut profile,
            &mut None,
            false,
            false,
            0,
        );

        // 4. Nạp expert 30 (probationary) và 40 (probationary) -> Bộ đệm đầy 4/4 slots
        layer.clock = 5;
        layer.heat[30] = 1;
        OlmoeModel::admit_expert_l2(
            &mut layer,
            30,
            make_expert(),
            false,
            1,
            false,
            cache_capacity,
            &mut profile,
            &mut None,
            false,
            false,
            0,
        );
        layer.clock = 6;
        layer.heat[40] = 1;
        OlmoeModel::admit_expert_l2(
            &mut layer,
            40,
            make_expert(),
            false,
            1,
            false,
            cache_capacity,
            &mut profile,
            &mut None,
            false,
            false,
            0,
        );
        assert_eq!(layer.cache.len(), 4);
        assert_eq!(layer.bytes, cache_capacity);

        // 5. Nạp expert mới 50 (miss). SF-Cache BUỘC PHẢI trục xuất phần tử Probationary (30), KHÔNG ĐƯỢC trục xuất 10 hoặc 20!
        layer.clock = 7;
        layer.heat[50] = 1;
        OlmoeModel::admit_expert_l2(
            &mut layer,
            50,
            make_expert(),
            false,
            1,
            false,
            cache_capacity,
            &mut profile,
            &mut None,
            false,
            false,
            0,
        );
        assert_eq!(layer.cache.len(), 4);
        assert_eq!(layer.bytes, cache_capacity);

        // Thẩm định: Expert 10 và 20 vẫn an toàn tuyệt đối trong cache!
        let cached_ids: Vec<usize> = layer.cache.iter().map(|e| e.id).collect();
        assert!(
            cached_ids.contains(&10),
            "Expert 10 (protected) không được bị trục xuất"
        );
        assert!(
            cached_ids.contains(&20),
            "Expert 20 (protected) không được bị trục xuất"
        );
        assert!(
            cached_ids.contains(&50),
            "Expert 50 mới phải được nạp vào cache"
        );
        assert!(
            !cached_ids.contains(&30),
            "Expert 30 (probationary cũ nhất) phải bị trục xuất"
        );
    }

    fn prepared_lookahead_model(directory: &Path, enabled: bool) -> OlmoeModel {
        let mut model =
            OlmoeModel::load_with_gpu_mode(directory, 8192, 64, 10_000_000, GpuMode::Cpu).unwrap();
        model.set_gpu_enabled(false);
        model.lookahead_enabled = enabled;
        // Tie-break routes to 0/1. Predict 0 (used) and 3 (unused) at the next layer.
        for layer in &mut model.layers {
            layer.router.values.fill(0.0);
        }
        for id in [0, 1] {
            let expert = Int8Expert::load(&model.index, 0, id, 16, 8, 512).unwrap();
            model.layers[0].bytes += expert.allocated_bytes();
            model.layers[0].cache.push_back(CachedExpert {
                is_pinned: false,
                id,
                expert,
                last_used: 0,
                hits_in_cache: 1,
                is_protected: false,
            });
            model.inter_layer_transitions[0][id * 64] = 10;
            model.inter_layer_transitions[0][id * 64 + 3] = 10;
        }
        model
    }

    #[test]
    fn lookahead_modes_preserve_logits_and_account_for_used_unused_buffers() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut expected = None;
        for enabled in [false, true] {
            let mut model = prepared_lookahead_model(&directory, enabled);
            let mut profile = RequestProfile::default();
            let mut scratch = StepScratch::new(&model.config);
            let mut kv: Vec<_> = (0..model.layers.len()).map(|_| Kv::default()).collect();
            model
                .step(1, &mut kv, &mut profile, &mut scratch, &|| Ok(()))
                .unwrap();
            let m = &profile.lookahead;
            assert_eq!(m.eligible_layers, 1);
            assert_eq!(m.evaluated_layers, 2);
            assert_eq!(m.no_next_layer, 1);
            assert_eq!(m.blocked_by_misses, 0);
            assert_eq!(m.miss_histogram, [1, 0, 0]);
            assert_eq!(m.disabled_eligible_layers, if enabled { 0 } else { 1 });
            assert_eq!(m.prediction_calls, if enabled { 1 } else { 0 });
            assert_eq!(m.positive_candidates, if enabled { 2 } else { 0 });
            assert_eq!(m.candidates_available, m.positive_candidates);
            assert_eq!(m.proposed, if enabled { 2 } else { 0 });
            assert_eq!(m.loaded, m.used + m.unused + m.abandoned + m.pending);
            assert_eq!(m.proposed, m.loaded + m.load_errors);
            assert_eq!(m.pending, 0);
            if enabled {
                assert_eq!((m.used, m.unused, m.load_errors), (1, 1, 0));
                assert_eq!(m.file_bytes, 2 * 512);
                assert_eq!(m.buffer_peak_bytes, 2 * 512);
            } else {
                assert_eq!(m.loaded, 0);
                assert_eq!(m.buffer_peak_bytes, 0);
            }
            assert!(profile.decode_expert_active_peak_bytes <= (2 + 2) * 512);
            assert!(profile.decode_expert_host_peak_bytes <= 2 * 8192 + (2 + 2) * 512);
            for layer in &model.layers {
                assert!(layer.bytes + layer.recycled_bytes() <= 8192);
            }
            if let Some(ref logits) = expected {
                let error = scratch
                    .logits
                    .iter()
                    .zip(logits)
                    .map(|(a, b): (&f32, &f32)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(error < 2e-5, "{error}");
            } else {
                expected = Some(scratch.logits.clone());
            }
            model.reset_cache();
            assert!(model
                .inter_layer_transitions
                .iter()
                .flatten()
                .all(|n| *n == 0));
            assert!(model
                .layers
                .iter()
                .all(|l| l.cache.is_empty() && l.recycled_buffers.is_empty()));
            assert_eq!(model.expert_status().lookahead_enabled, enabled);
        }
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let directory = directory.parent().unwrap().join(fixture);
            let mut model = OlmoeModel::load(directory, 1024, 64, 10_000_000).unwrap();
            model.set_gpu_enabled(false);
            model.lookahead_enabled = false;
            let off = model.generate("a", 8).unwrap();
            model.reset_cache();
            model.lookahead_enabled = true;
            let on = model.generate("a", 8).unwrap();
            assert_eq!(off.token_ids, on.token_ids);
        }
    }

    #[test]
    fn lookahead_reports_partial_and_full_current_misses_without_predicting() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        for missing in [1, 2] {
            let mut model = prepared_lookahead_model(&directory, true);
            for _ in 0..missing {
                let entry = model.layers[0].cache.pop_back().unwrap();
                model.layers[0].bytes -= entry.expert.allocated_bytes();
            }
            let mut profile = RequestProfile::default();
            let mut scratch = StepScratch::new(&model.config);
            let mut kv: Vec<_> = (0..model.layers.len()).map(|_| Kv::default()).collect();
            model
                .step(1, &mut kv, &mut profile, &mut scratch, &|| Ok(()))
                .unwrap();
            let m = &profile.lookahead;
            assert_eq!(m.evaluated_layers, 2);
            assert_eq!(m.no_next_layer, 1);
            assert_eq!(m.blocked_by_misses, 1);
            assert_eq!(m.miss_histogram[missing], 1);
            assert_eq!(m.miss_histogram.iter().sum::<u64>(), 1);
            assert_eq!(m.eligible_layers, 0);
            assert_eq!(m.prediction_calls, 0);
            assert_eq!(m.no_positive_score_calls, 0); // Not evaluated, rather than no score.
            assert_eq!(m.proposed, 0);
        }
    }

    #[test]
    fn lookahead_distinguishes_no_score_from_candidates_already_cached() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        for cached in [false, true] {
            let mut model = prepared_lookahead_model(&directory, true);
            if cached {
                for id in [0, 3] {
                    let expert = Int8Expert::load(&model.index, 1, id, 16, 8, 512).unwrap();
                    model.layers[1].bytes += expert.allocated_bytes();
                    model.layers[1].cache.push_back(CachedExpert {
                        is_pinned: false,
                        id,
                        expert,
                        last_used: 0,
                        hits_in_cache: 1,
                        is_protected: false,
                    });
                }
            } else {
                model.inter_layer_transitions[0].fill(0);
            }
            let mut profile = RequestProfile::default();
            let mut scratch = StepScratch::new(&model.config);
            let mut kv: Vec<_> = (0..model.layers.len()).map(|_| Kv::default()).collect();
            model
                .step(1, &mut kv, &mut profile, &mut scratch, &|| Ok(()))
                .unwrap();
            let m = &profile.lookahead;
            assert_eq!(m.eligible_layers, 1);
            assert_eq!(m.prediction_calls, 1);
            assert_eq!(m.miss_histogram, [1, 0, 0]);
            assert_eq!(m.no_positive_score_calls, if cached { 0 } else { 1 });
            assert_eq!(m.all_candidates_cached_calls, if cached { 1 } else { 0 });
            assert_eq!(m.positive_candidates, if cached { 2 } else { 0 });
            assert_eq!(m.candidates_in_ram, m.positive_candidates);
            assert_eq!(m.candidates_available, 0);
            assert_eq!(m.loaded, 0);
        }
    }

    #[test]
    fn lookahead_cancel_drops_pending_and_model_can_retry() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let mut model = prepared_lookahead_model(&directory, true);
        let mut profile = RequestProfile::default();
        let mut scratch = StepScratch::new(&model.config);
        let mut kv: Vec<_> = (0..model.layers.len()).map(|_| Kv::default()).collect();
        let checks = std::cell::Cell::new(0);
        let error = model
            .step(1, &mut kv, &mut profile, &mut scratch, &|| {
                checks.set(checks.get() + 1);
                if checks.get() == 2 {
                    Err(io::Error::new(io::ErrorKind::Interrupted, "cancel"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert_eq!(profile.lookahead.loaded, 2);
        assert_eq!(profile.lookahead.abandoned, 2);
        assert_eq!(profile.lookahead.pending, 0);
        model.reset_cache();
        assert!(model.generate("a", 3).is_ok());
    }

    #[test]
    fn lookahead_file_error_is_counted_and_not_hidden_as_success() {
        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let copy = tempfile::tempdir().unwrap();
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), copy.path().join(entry.file_name())).unwrap();
        }
        let mut model = prepared_lookahead_model(copy.path(), true);
        // Only this private test copy changes, after indexing/priming the current layer.
        fs::OpenOptions::new()
            .write(true)
            .open(copy.path().join("model.safetensors"))
            .unwrap()
            .set_len(0)
            .unwrap();
        let mut profile = RequestProfile::default();
        let mut scratch = StepScratch::new(&model.config);
        let mut kv: Vec<_> = (0..model.layers.len()).map(|_| Kv::default()).collect();
        assert!(model
            .step(1, &mut kv, &mut profile, &mut scratch, &|| Ok(()))
            .is_err());
        assert_eq!(profile.lookahead.proposed, 2);
        assert_eq!(profile.lookahead.load_errors, 2);
        assert_eq!(profile.lookahead.loaded, 0);
        assert_eq!(profile.lookahead.pending, 0);
    }

    #[test]
    fn lookahead_mode_parser_rejects_ambiguous_values() {
        assert!(parse_lookahead_mode(None).unwrap());
        assert!(parse_lookahead_mode(Some("on")).unwrap());
        assert!(!parse_lookahead_mode(Some("off")).unwrap());
        assert!(parse_lookahead_mode(Some("false")).is_err());
    }

    #[test]
    fn test_inter_layer_lookahead_prefetch_and_markov_transition() {
        // Kiểm tra cơ chế dự đoán Lookahead và cập nhật Ma trận chuyển tiếp liên tầng
        let num_layers = 4;
        let mut transitions = vec![vec![0u16; 64 * 64]; num_layers - 1];

        // 1. Tầng 0 kích hoạt expert 7 và 12
        let layer0_experts = vec![7, 12];
        // Tầng 1 kích hoạt expert 25 và 48
        let layer1_experts = vec![25, 48];

        // Cập nhật ma trận chuyển tiếp tầng 0 -> 1
        let trans0 = &mut transitions[0];
        for &src in &layer0_experts {
            for &dst in &layer1_experts {
                trans0[src * 64 + dst] = trans0[src * 64 + dst].saturating_add(10);
            }
        }

        assert_eq!(transitions[0][7 * 64 + 25], 10);
        assert_eq!(transitions[0][7 * 64 + 48], 10);
        assert_eq!(transitions[0][12 * 64 + 25], 10);
        assert_eq!(transitions[0][12 * 64 + 48], 10);
        assert_eq!(transitions[0][7 * 64], 0);

        // 2. Dự đoán cho tầng 1 dựa trên router tầng 0
        let mut next_layer_heat = vec![0u32; 64];
        next_layer_heat[25] = 20; // heat cao
        next_layer_heat[9] = 100; // heat rất cao nhưng không có transition

        let next_layer_cache = VecDeque::new();

        let (candidates, _) = OlmoeModel::predict_next_layer_experts(
            None,
            &transitions,
            0,
            &layer0_experts,
            &next_layer_heat,
            &next_layer_cache,
            None,
            false,
            3,
            None,
        );

        // Expert 25 có cả transition (10 + 10 = 20) và heat (20/4 = 5) -> Score = 25
        // Expert 48 có transition (10 + 10 = 20) và heat (0) -> Score = 20
        // Expert 9 có heat (100/4 = 25) và transition (0) -> Score = 25
        assert_eq!(candidates.len(), 3);
        assert!(
            candidates.contains(&25),
            "Expert 25 phải nằm trong top candidates"
        );
        assert!(
            candidates.contains(&48),
            "Expert 48 phải nằm trong top candidates"
        );
        assert!(
            candidates.contains(&9),
            "Expert 9 (high heat) phải nằm trong top candidates"
        );
    }

    #[test]
    fn test_pilot_router_lookahead_speculation() {
        // Kiểm tra cơ chế PILOT Router Lookahead từ Colibrì:
        // Tín hiệu logits động từ router tầng kế tiếp dự phóng chính xác ứng viên
        let num_layers = 4;
        let transitions = vec![vec![0u16; 64 * 64]; num_layers - 1];
        let layer0_experts = vec![1, 2];
        let next_layer_heat = vec![0u32; 64];
        let next_layer_cache = VecDeque::new();

        // Giả lập router logits tầng kế tiếp dự phóng expert 42 và 17 có điểm số cao vượt trội
        let mut pilot_router = vec![0.0f32; 64];
        pilot_router[42] = 8.5; // Top 1
        pilot_router[17] = 6.2; // Top 2
        pilot_router[5] = 4.1; // Top 3

        let (candidates, _) = OlmoeModel::predict_next_layer_experts(
            Some(&pilot_router),
            &transitions,
            0,
            &layer0_experts,
            &next_layer_heat,
            &next_layer_cache,
            None,
            false,
            2,
            None,
        );

        assert_eq!(candidates.len(), 2);
        // Top 2 candidates dự phóng từ PILOT router logits là 42 và 17
        assert!(candidates.contains(&42));
        assert!(candidates.contains(&17));
        // Kiểm tra thứ tự EID tăng dần để đảm bảo Sequential SSD Locality
        assert_eq!(candidates, vec![17, 42]);
    }

    #[test]
    fn test_pilot_router_momentum_ema() {
        // Kiểm tra cơ chế Routing Momentum EMA từ Colibrì (olmoe.c):
        let mut layer = Layer {
            input_norm: vec![],
            post_norm: vec![],
            q_norm: vec![],
            k_norm: vec![],
            q: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            k: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            v: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            o: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            router: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            cache: VecDeque::new(),
            bytes: 0,
            heat: vec![0; 64],
            clock: 0,
            recycled_buffers: Vec::new(),
            hot_pinned: 0,
            router_ema: vec![0.0; 64],
        };

        let logits_tok0 = vec![1.0f32; 64];
        let smooth = 0.35f32;
        // Lần đầu tiên cập nhật: EMA được gán bằng chính logits
        layer.update_router_ema(&logits_tok0, smooth);
        assert!((layer.router_ema[0] - 1.0).abs() < 1e-5);

        // Lần thứ hai cập nhật: logits = 2.0
        let logits_tok1 = vec![2.0f32; 64];
        layer.update_router_ema(&logits_tok1, smooth);
        // EMA = (1 - 0.35) * 2.0 + 0.35 * 1.0 = 1.30 + 0.35 = 1.65
        let expected = (1.0 - smooth) * 2.0 + smooth * 1.0;
        assert!((layer.router_ema[0] - expected).abs() < 1e-5);

        // Kiểm tra hàm blend_pilot_router
        let mut lookahead_logits = vec![3.0f32; 64];
        layer.blend_pilot_router(&mut lookahead_logits, smooth);
        let expected_blend = (1.0 - smooth) * 3.0 + smooth * expected;
        assert!((lookahead_logits[0] - expected_blend).abs() < 1e-5);
    }

    #[test]
    fn test_pilot_confidence_cutoff_pruning() {
        // Kiểm tra Cumulative Confidence Cutoff (pilot_conf_limit) từ Colibrì:
        let num_layers = 4;
        let transitions = vec![vec![0u16; 64 * 64]; num_layers - 1];
        let layer0_experts = vec![1, 2];
        let next_layer_heat = vec![0u32; 64];
        let next_layer_cache = VecDeque::new();

        let mut pilot_router = vec![0.0f32; 64];
        // Expert 33 có logit 10.0 (áp đảo, xác suất softmax > 99%)
        pilot_router[33] = 10.0;

        // Yêu cầu max_candidates = 4, nhưng với conf_limit = Some(0.90),
        // expert 33 đã vượt ngưỡng 90%, thuật toán phải cắt tỉa và chỉ trả về 1 candidate!
        let (candidates, _) = OlmoeModel::predict_next_layer_experts(
            Some(&pilot_router),
            &transitions,
            0,
            &layer0_experts,
            &next_layer_heat,
            &next_layer_cache,
            None,
            false,
            4,
            Some(0.90),
        );

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0], 33);
    }

    #[test]
    fn test_coarse_grained_multi_expert_scratch_and_reduction() {
        let hidden = 64;
        let intermediate = 32;
        let top_k = 4;

        // Tạo 4 experts giả lập
        let experts: Vec<Int8Expert> = (0..top_k)
            .map(|e| {
                let n = hidden * intermediate;
                let weights = (0..3 * n).map(|i| ((i * 13 + e * 7) % 256) as u8).collect();
                let scales = (0..2 * intermediate + hidden)
                    .map(|i| 0.002 * (1 + (i + e) % 5) as f32)
                    .collect();
                Int8Expert::from_raw(hidden, intermediate, weights, scales)
            })
            .collect();

        let input: Vec<f32> = (0..hidden).map(|i| ((i % 17) as f32 - 8.0) / 9.0).collect();
        let weights = [0.4f32, 0.3f32, 0.2f32, 0.1f32];

        // 1. Tính toán theo mô hình tuần tự (baseline)
        let mut expected_x = vec![0.0f32; hidden];
        for (i, expert) in experts.iter().enumerate() {
            let mut scratch = ExpertScratch::new(hidden, intermediate);
            let out = expert
                .forward_single_threaded(&input, &mut scratch)
                .unwrap();
            for (x_val, &o_val) in expected_x.iter_mut().zip(out) {
                *x_val += weights[i] * o_val;
            }
        }

        // 2. Tính toán theo mô hình song song Coarse-Grained Rayon (Bước 24)
        let mut expert_scratches: Vec<ExpertScratch> = (0..top_k)
            .map(|_| ExpertScratch::new(hidden, intermediate))
            .collect();

        experts
            .par_iter()
            .zip(expert_scratches.par_iter_mut())
            .for_each(|(expert, scratch)| {
                expert.forward_single_threaded(&input, scratch).unwrap();
            });

        let mut actual_x = vec![0.0f32; hidden];
        for (i, scratch) in expert_scratches.iter().enumerate() {
            for (x_val, &o_val) in actual_x.iter_mut().zip(&scratch.output) {
                *x_val += weights[i] * o_val;
            }
        }

        // Thẩm định bit-exact 100%
        for i in 0..hidden {
            assert_eq!(
                expected_x[i].to_bits(),
                actual_x[i].to_bits(),
                "Bit mismatch at index {i}: {} vs {}",
                expected_x[i],
                actual_x[i]
            );
        }
    }

    #[test]
    fn test_hot_expert_pinning_prevents_eviction() {
        // Kiểm tra cơ chế HOT Pinning từ Colibrì:
        // Expert có tần suất sử dụng cao nhất được ghim và không bị trục xuất khỏi cache dù cache đầy
        let hidden = 64;
        let intermediate = 32;
        let make_expert = || {
            let n = hidden * intermediate;
            let weights = vec![1u8; 3 * n];
            let scales = vec![0.001f32; 2 * intermediate + hidden];
            Int8Expert::from_raw(hidden, intermediate, weights, scales)
        };

        let expert_bytes = make_expert().resident_bytes();
        let cache_capacity = expert_bytes * 2; // Ngân sách chỉ chứa tối đa 2 experts

        let mut layer = Layer {
            input_norm: vec![],
            post_norm: vec![],
            q_norm: vec![],
            k_norm: vec![],
            q: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            k: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            v: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            o: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            router: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            cache: VecDeque::new(),
            bytes: 0,
            heat: vec![0; 64],
            clock: 0,
            recycled_buffers: Vec::new(),
            hot_pinned: 1, // Ghim top 1 hot expert
            router_ema: vec![0.0; 64],
        };
        let mut profile = RequestProfile::default();

        // 1. Nạp expert 5 với heat rất cao = 50 -> Trở thành HOT expert được ghim
        layer.heat[5] = 50;
        layer.clock = 1;
        OlmoeModel::admit_expert_l2(
            &mut layer,
            5,
            make_expert(),
            false,
            0,
            false,
            cache_capacity,
            &mut profile,
            &mut None,
            false,
            false,
            0,
        );
        assert_eq!(layer.cache.len(), 1);
        assert!(layer.cache[0].is_pinned);

        // 2. Nạp expert 10 (heat = 1) -> Vừa khít ngân sách 2 experts
        layer.heat[10] = 1;
        layer.clock = 2;
        OlmoeModel::admit_expert_l2(
            &mut layer,
            10,
            make_expert(),
            false,
            0,
            false,
            cache_capacity,
            &mut profile,
            &mut None,
            false,
            false,
            0,
        );
        assert_eq!(layer.cache.len(), 2);

        // 3. Nạp expert 20 (heat = 2) -> Bắt buộc phải trục xuất 1 expert.
        // Bình thường expert 5 (đến trước) hoặc expert 10 (heat thấp hơn) sẽ bị xem xét.
        // Nhờ HOT Pinning, expert 5 được ghim, nên expert 10 bị trục xuất!
        layer.heat[20] = 2;
        layer.clock = 3;
        OlmoeModel::admit_expert_l2(
            &mut layer,
            20,
            make_expert(),
            false,
            0,
            false,
            cache_capacity,
            &mut profile,
            &mut None,
            false,
            false,
            0,
        );

        assert_eq!(layer.cache.len(), 2);
        let resident_ids: Vec<usize> = layer.cache.iter().map(|e| e.id).collect();
        assert!(
            resident_ids.contains(&5),
            "Expert 5 được ghim (pinned) phải còn trong cache"
        );
        assert!(
            resident_ids.contains(&20),
            "Expert 20 mới nạp phải có trong cache"
        );
        assert!(
            !resident_ids.contains(&10),
            "Expert 10 không ghim phải bị trục xuất"
        );
    }

    #[test]
    fn test_route_trace_logger_format() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_route_trace_{}.log", std::process::id()));
        let _ = fs::remove_file(&path);

        let mut logger = RouteTraceLogger::new(&path).expect("Tạo logger file");
        logger.trace(0, 3, &[5, 12], &[0.65432, 0.34568]);
        logger.trace(1, 3, &[5, 21], &[0.81234, 0.18766]);
        logger.advance_call();
        logger.trace(0, 0, &[1, 2], &[0.5, 0.5]);
        drop(logger);

        let content = fs::read_to_string(&path).expect("Đọc trace file");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "0 0 3 5:0.6543 12:0.3457");
        assert_eq!(lines[1], "0 1 3 5:0.8123 21:0.1877");
        assert_eq!(lines[2], "1 0 0 1:0.5000 2:0.5000");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_persistent_usage_save_and_load_pins_hot_experts() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_usage_{}.usage", std::process::id()));
        let _ = fs::remove_file(&path);

        let mut layer = Layer {
            input_norm: vec![],
            post_norm: vec![],
            q_norm: vec![],
            k_norm: vec![],
            q: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            k: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            v: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            o: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            router: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            cache: VecDeque::new(),
            bytes: 0,
            heat: vec![0; 64],
            clock: 0,
            recycled_buffers: Vec::new(),
            hot_pinned: 1,
            router_ema: vec![0.0; 64],
        };
        layer.heat[17] = 250;
        layer.heat[4] = 80;

        let layers = vec![layer];
        OlmoeModel::save_usage_to_path(&path, &layers, 64).expect("Lưu usage file");

        let fresh_layer = Layer {
            input_norm: vec![],
            post_norm: vec![],
            q_norm: vec![],
            k_norm: vec![],
            q: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            k: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            v: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            o: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            router: Matrix {
                rows: 0,
                columns: 0,
                values: vec![],
            },
            cache: VecDeque::new(),
            bytes: 0,
            heat: vec![0; 64],
            clock: 0,
            recycled_buffers: Vec::new(),
            hot_pinned: 1,
            router_ema: vec![0.0; 64],
        };
        let mut fresh_layers = vec![fresh_layer];
        OlmoeModel::load_usage_from_path(&path, &mut fresh_layers, 64);

        assert_eq!(fresh_layers[0].heat[17], 250);
        assert_eq!(fresh_layers[0].heat[4], 80);
        assert_eq!(fresh_layers[0].heat[0], 0);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_kv_fp8_compression_and_numerical_parity() {
        for fixture in ["unnormalized-mha", "normalized-gqa"] {
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/olmoe")
                .join(fixture);
            let reference: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join("reference.json")).unwrap())
                    .unwrap();
            let tokens: Vec<_> = reference["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect();

            let mut model_fp32 = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
            assert!(!model_fp32.is_kv_fp8());
            let logits_fp32 = model_fp32.logits(&tokens).unwrap();

            let mut model_fp8 = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
            model_fp8.set_kv_fp8(true);
            assert!(model_fp8.is_kv_fp8());
            let logits_fp8 = model_fp8.logits(&tokens).unwrap();

            assert_eq!(logits_fp32.len(), logits_fp8.len());
            for (step, (l32, l8)) in logits_fp32.iter().zip(&logits_fp8).enumerate() {
                let top_fp32 = argmax_greedy(l32);
                let top_fp8 = argmax_greedy(l8);
                assert_eq!(
                    top_fp32, top_fp8,
                    "Fixture {fixture} step {step} greedy token mismatch"
                );

                let max_err = l32
                    .iter()
                    .zip(l8)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 0.25,
                    "Fixture {fixture} step {step} max logit delta too high: {max_err}"
                );
            }

            // Kiểm tra tỉ lệ nén bộ nhớ KV Cache
            let kd = model_fp8.config.hidden_size / model_fp8.config.num_attention_heads
                * model_fp8.config.num_key_value_heads;
            let mut kv_fp32 = Kv::with_capacity_precision(1024, kd, false);
            let mut kv_fp8 = Kv::with_capacity_precision(1024, kd, true);

            // Giả lập 256 tokens trong cache
            for _ in 0..256 {
                kv_fp32.keys.extend((0..kd).map(|i| i as f32));
                kv_fp32.values.extend((0..kd).map(|i| i as f32));

                kv_fp8.fp8_keys.extend((0..kd).map(|i| (i % 256) as u8));
                kv_fp8.fp8_values.extend((0..kd).map(|i| (i % 256) as u8));
                kv_fp8.fp8_key_scales.push(1.0);
                kv_fp8.fp8_value_scales.push(1.0);
            }

            assert_eq!(kv_fp32.positions(kd), 256);
            assert_eq!(kv_fp8.positions(kd), 256);

            let bytes_fp32 = kv_fp32.memory_bytes();
            let bytes_fp8 = kv_fp8.memory_bytes();
            let ratio = bytes_fp8 as f64 / bytes_fp32 as f64;
            let theoretical_ratio = (2.0 * kd as f64 + 8.0) / (2.0 * kd as f64 * 4.0);
            assert!(
                (ratio - theoretical_ratio).abs() < 1e-4,
                "Tỉ lệ dung lượng thực tế {ratio} lệch khỏi lý thuyết {theoretical_ratio}"
            );
            assert!(
                ratio <= 0.40,
                "Tỉ lệ dung lượng KV FP8/FP32 vượt quá 40%: {ratio}"
            );

            // Kiểm tra tỉ lệ nén trên kích thước thật của OLMoE (kd = 512, tiết kiệm ~74.8% RAM)
            let mut full_fp32 = Kv::with_capacity_precision(1024, 512, false);
            let mut full_fp8 = Kv::with_capacity_precision(1024, 512, true);
            for _ in 0..256 {
                full_fp32.keys.extend((0..512).map(|i| i as f32));
                full_fp32.values.extend((0..512).map(|i| i as f32));
                full_fp8.fp8_keys.extend((0..512).map(|i| (i % 256) as u8));
                full_fp8
                    .fp8_values
                    .extend((0..512).map(|i| (i % 256) as u8));
                full_fp8.fp8_key_scales.push(1.0);
                full_fp8.fp8_value_scales.push(1.0);
            }
            let full_ratio = full_fp8.memory_bytes() as f64 / full_fp32.memory_bytes() as f64;
            assert!(
                full_ratio < 0.255,
                "Full OLMoE kd=512 phải đạt tỉ lệ nén < 25.5%: {full_ratio}"
            );

            // Kiểm tra truncate_positions
            kv_fp8.truncate_positions(100, kd);
            assert_eq!(kv_fp8.positions(kd), 100);
            assert_eq!(kv_fp8.fp8_keys.len(), 100 * kd);
            assert_eq!(kv_fp8.fp8_values.len(), 100 * kd);
            assert_eq!(kv_fp8.fp8_key_scales.len(), 100);
            assert_eq!(kv_fp8.fp8_value_scales.len(), 100);
        }
    }

    #[test]
    fn test_kv_session_disk_persistence_roundtrip_and_crash_safety() {
        let temp_dir = std::env::temp_dir().join(format!("moe_test_kv_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);
        let file_path = temp_dir.join("test_session.moekv");

        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        let model = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
        let kd = model.config.hidden_size / model.config.num_attention_heads
            * model.config.num_key_value_heads;

        // 1. Kiểm thử lưu và nạp định dạng FP32
        let tokens: Vec<usize> = vec![10, 20, 30, 40, 50];
        let mut kv_fp32: Vec<Kv> = (0..model.layers.len())
            .map(|_| Kv::with_capacity_precision(128, kd, false))
            .collect();
        for layer in &mut kv_fp32 {
            for p in 0..tokens.len() {
                layer.keys.extend((0..kd).map(|i| (p * 100 + i) as f32));
                layer.values.extend((0..kd).map(|i| (p * 200 + i) as f32));
            }
        }

        model
            .save_session_to_file(&file_path, &tokens, &kv_fp32)
            .unwrap();
        assert!(file_path.exists());

        // Nạp lại khớp 100% tiền tố
        let (prefix, loaded_kv) = model
            .load_session_from_file(&file_path, &tokens, kd)
            .unwrap()
            .expect("Phải nạp thành công session từ đĩa");

        // Causal truncation giữ lại tối đa tokens.len() - 1 = 4
        assert_eq!(prefix, 4);
        assert_eq!(loaded_kv.len(), model.layers.len());
        for (layer_orig, layer_loaded) in kv_fp32.iter().zip(&loaded_kv) {
            assert_eq!(layer_loaded.positions(kd), 4);
            assert_eq!(&layer_loaded.keys, &layer_orig.keys[..4 * kd]);
            assert_eq!(&layer_loaded.values, &layer_orig.values[..4 * kd]);
        }

        // 2. Kiểm thử lưu và nạp định dạng nén FP8 E4M3
        let mut kv_fp8: Vec<Kv> = (0..model.layers.len())
            .map(|_| Kv::with_capacity_precision(128, kd, true))
            .collect();
        for layer in &mut kv_fp8 {
            for p in 0..tokens.len() {
                layer
                    .fp8_keys
                    .extend((0..kd).map(|i| ((p + i) % 255) as u8));
                layer.fp8_key_scales.push(1.5);
                layer
                    .fp8_values
                    .extend((0..kd).map(|i| ((p * 2 + i) % 255) as u8));
                layer.fp8_value_scales.push(2.5);
            }
        }

        model
            .save_session_to_file(&file_path, &tokens, &kv_fp8)
            .unwrap();
        let (prefix_fp8, loaded_fp8) = model
            .load_session_from_file(&file_path, &tokens, kd)
            .unwrap()
            .expect("Phải nạp thành công session FP8");
        assert_eq!(prefix_fp8, 4);
        for (layer_orig, layer_loaded) in kv_fp8.iter().zip(&loaded_fp8) {
            assert!(layer_loaded.is_fp8);
            assert_eq!(layer_loaded.positions(kd), 4);
            assert_eq!(&layer_loaded.fp8_keys, &layer_orig.fp8_keys[..4 * kd]);
            assert_eq!(
                &layer_loaded.fp8_key_scales,
                &layer_orig.fp8_key_scales[..4]
            );
            assert_eq!(&layer_loaded.fp8_values, &layer_orig.fp8_values[..4 * kd]);
            assert_eq!(
                &layer_loaded.fp8_value_scales,
                &layer_orig.fp8_value_scales[..4]
            );
        }

        // 3. Kiểm thử so khớp tiền tố không trùng (Diverging tokens)
        let alt_tokens: Vec<usize> = vec![10, 20, 999, 40, 50]; // Lệch tại vị trí 2
        let (prefix_div, loaded_div) = model
            .load_session_from_file(&file_path, &alt_tokens, kd)
            .unwrap()
            .expect("Phải nạp phần tiền tố khớp trước khi lệch");
        assert_eq!(prefix_div, 2); // Chỉ giữ lại 2 tokens [10, 20]
        assert_eq!(loaded_div[0].positions(kd), 2);

        // 4. Kiểm thử xử lý lỗi dữ liệu hỏng (Robustness on corruption)
        fs::write(&file_path, b"CORRUPTED_SHORT_HEADER").unwrap();
        let result = model
            .load_session_from_file(&file_path, &tokens, kd)
            .unwrap();
        assert!(
            result.is_none(),
            "Dữ liệu header hỏng phải trả về None thay vì panic"
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_session_resume_after_engine_restart() {
        let temp_dir =
            std::env::temp_dir().join(format!("moe_test_restart_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);

        let directory =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");

        // 1. Chạy lượt 1 với engine 1
        let mut model1 = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
        model1.set_session_dir(Some(temp_dir.clone()));
        let prompt1 = "Hello";
        let out1 = model1
            .generate_controlled(
                prompt1,
                2,
                Some("chat-session-1"),
                &|| Ok(()),
                || Ok(()),
                None,
            )
            .unwrap();
        assert_eq!(out1.finish_reason, "length");
        assert!(temp_dir.join("chat-session-1.moekv").exists());

        // 2. "Khởi động lại máy chủ": Tạo thực thể engine 2 hoàn toàn mới (RAM trống 100%)
        let mut model2 = OlmoeModel::load(&directory, 100_000, 64, 10_000_000).unwrap();
        assert!(model2.sessions.is_empty(), "Engine mới phải có RAM trống");
        model2.set_session_dir(Some(temp_dir.clone()));

        // Gửi lượt 2 tiếp nối hội thoại trên cùng session ID
        let prompt2 = format!("{} world", prompt1);
        let out2 = model2
            .generate_controlled(
                &prompt2,
                2,
                Some("chat-session-1"),
                &|| Ok(()),
                || Ok(()),
                None,
            )
            .unwrap();

        // Xác minh session được khôi phục thành công từ đĩa NVMe mà không phải prefill từ đầu
        let metrics = out2.metrics.expect("Phải có metrics");
        assert!(
            metrics.cached_prompt_tokens > 0,
            "Engine 2 sau khởi động lại phải khôi phục được tiền tố prompt từ đĩa NVMe: cached={}",
            metrics.cached_prompt_tokens
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
