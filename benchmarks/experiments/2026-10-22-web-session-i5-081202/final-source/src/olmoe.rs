//! CPU OLMoE decoder for merged INT8 checkpoints.
use crate::{
    backend::{
        causal_attention_into, configure_parallelism, finite, invalid, rms_norm_in_place,
        rms_norm_into, softmax, Matrix,
    },
    generation::{Generation, GenerationMetrics},
    int8_expert::{ExpertScratch, Int8Expert},
    safetensors::TensorIndex,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    fs, io,
    path::Path,
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
}

#[derive(Serialize, Clone, Debug, Default)]
pub struct SystemTelemetry {
    pub vm_rss_kib: Option<u64>,
    pub vm_hwm_kib: Option<u64>,
    pub io_read_bytes: Option<u64>,
    pub io_rchar: Option<u64>,
    pub gpu_vram_used_mib: Option<u64>,
}

#[derive(Serialize, Clone, Debug)]
pub struct LayerExpertStatus {
    pub resident_experts: Vec<usize>,
    pub heat: Vec<u32>,
    pub cache_bytes: usize,
}

#[derive(Serialize, Clone, Debug)]
pub struct ExpertStatusReport {
    pub expert_cache_policy: &'static str,
    pub num_layers: usize,
    pub num_experts: usize,
    pub cache_bytes_per_layer: usize,
    pub dense_bytes: usize,
    pub layers: Vec<LayerExpertStatus>,
    pub system_telemetry: SystemTelemetry,
    pub gpu: crate::gpu::GpuTierStatus,
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
}
#[derive(Default)]
struct Kv {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl Kv {
    fn with_capacity(capacity_tokens: usize, kd: usize) -> Self {
        let total = capacity_tokens * kd;
        Self {
            keys: Vec::with_capacity(total),
            values: Vec::with_capacity(total),
        }
    }
}

struct SessionState {
    tokens: Vec<usize>,
    kv: Vec<Kv>,
    last_used: u64,
}

#[derive(Default)]
struct RequestProfile {
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
    bytes_read: u64,
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
    logits: Vec<f32>,
    rope_bases: Vec<f32>,
    sincos: Vec<(f32, f32)>,
}

impl StepScratch {
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
            logits: vec![0.0; config.vocab_size],
            rope_bases,
            sincos,
        }
    }
}

pub struct OlmoeModel {
    // Retain the serial reference only in tests for paired in-process measurements.
    #[cfg(test)]
    serial_prefill_reference: bool,
    #[cfg(test)]
    strict_session_prefix_reference: bool,
    config: Config,
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
    gpu_tier: crate::gpu::GpuTierManager,
    gpu_dense: Option<crate::gpu::GpuDenseStorage>,
    inter_layer_transitions: Vec<Vec<u16>>,
}

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
        let expert_bytes = d
            .checked_mul(config.intermediate_size)
            .and_then(|count| count.checked_mul(3))
            .and_then(|weights| {
                (config.intermediate_size * 2 + d)
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|scales| weights.checked_add(scales))
            })
            .ok_or_else(|| invalid("expert size overflow"))?;
        let active_set_bytes = expert_bytes
            .checked_mul(config.num_experts_per_tok)
            .ok_or_else(|| invalid("active expert set size overflow"))?;
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
                if w.dtype != crate::safetensors::DType::I8
                    || w.shape != [3 * d * config.intermediate_size]
                    || s.dtype != crate::safetensors::DType::F32
                    || s.shape != [2 * config.intermediate_size + d]
                {
                    return Err(invalid(format!("{ep}: invalid merged expert layout")));
                }
            }
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
        let mut gpu_tier = crate::gpu::GpuTierManager::probe();
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
                        let weight_slot_bytes = 3 * config.hidden_size * config.intermediate_size;
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
        Ok(Self {
            #[cfg(test)]
            serial_prefill_reference: true,
            #[cfg(test)]
            strict_session_prefix_reference: false,
            config,
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
            gpu_tier,
            gpu_dense,
            inter_layer_transitions: vec![vec![0u16; 64 * 64]; num_layers.saturating_sub(1)],
        })
    }

    pub fn expert_status(&self) -> ExpertStatusReport {
        let layers = self
            .layers
            .iter()
            .map(|layer| LayerExpertStatus {
                resident_experts: layer.cache.iter().map(|entry| entry.id).collect(),
                heat: layer.heat.clone(),
                cache_bytes: layer.bytes,
            })
            .collect();
        let mut gpu_status = self.gpu_tier.status();
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
            expert_cache_policy: if self.victim_expert_cache {
                "victim"
            } else {
                "inclusive"
            },
            num_layers: self.layers.len(),
            num_experts: self.config.num_experts,
            cache_bytes_per_layer: self.cache_bytes_per_layer,
            dense_bytes: 3 * 1024 * 1024 * 1024,
            layers,
            system_telemetry,
            gpu: gpu_status,
        }
    }

    /// Đảo trạng thái Bật / Tắt GPU Tier khi chạy ứng dụng
    pub fn toggle_gpu(&mut self) -> crate::gpu::GpuTierStatus {
        let new_state = self.gpu_tier.toggle_enabled();
        if new_state {
            if let Some(storage) = self.gpu_dense.as_mut() {
                storage.engine.reset_circuit_breaker();
            }
        }
        self.gpu_tier.status()
    }

    /// Thiết lập trạng thái Bật / Tắt GPU Tier
    pub fn set_gpu_enabled(&mut self, enabled: bool) -> crate::gpu::GpuTierStatus {
        self.gpu_tier.set_enabled(enabled);
        if enabled {
            if let Some(storage) = self.gpu_dense.as_mut() {
                storage.engine.reset_circuit_breaker();
            }
        }
        self.gpu_tier.status()
    }

    /// Xóa toàn bộ bộ đệm (KV session cache, Host RAM L2 expert cache, GPU VRAM L1 expert cache)
    /// Đưa hệ thống về trạng thái Khởi động lạnh (Cold Start / 0 cache) hoàn toàn
    pub fn reset_cache(&mut self) {
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
                }
            }
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
            .collect();

        if missing_ids.len() < 2 {
            return (HashMap::new(), Duration::ZERO);
        }

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
    /// dựa trên:
    /// 1. Ma trận xác suất chuyển tiếp giữa 2 tầng liên tiếp (Inter-Layer Markov Transition)
    /// 2. Tín hiệu nhiệt (Heat/Recency) của tầng tiếp theo
    /// 3. Lọc bỏ các expert đã có sẵn trong L2 Host RAM Cache hoặc L1 GPU VRAM Cache
    #[allow(clippy::too_many_arguments)]
    fn predict_next_layer_experts(
        transitions: &[Vec<u16>],
        curr_layer_id: usize,
        curr_top_experts: &[usize],
        next_layer_heat: &[u32],
        next_layer_cache: &VecDeque<CachedExpert>,
        gpu_dense: Option<&crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        max_candidates: usize,
    ) -> Vec<usize> {
        if curr_layer_id >= transitions.len() || max_candidates == 0 {
            return Vec::new();
        }
        let trans = &transitions[curr_layer_id];
        let mut scores = [0u32; 64];

        // 1. Tích lũy xác suất chuyển tiếp từ các expert đang hoạt động ở tầng hiện tại
        for &src in curr_top_experts {
            if src < 64 {
                let row_offset = src * 64;
                for (dst, score) in scores.iter_mut().enumerate() {
                    *score = score.saturating_add(trans[row_offset + dst] as u32);
                }
            }
        }

        // 2. Kết hợp với tín hiệu nhiệt (Heat/Recency) của tầng tiếp theo
        for (dst, score) in scores.iter_mut().enumerate() {
            if dst < next_layer_heat.len() {
                *score = score.saturating_add(next_layer_heat[dst] / 4);
            }
        }

        // 3. Lọc các expert CHƯA CÓ trong cache của tầng tiếp theo (và chỉ lấy expert có điểm tin cậy > 0)
        let next_layer_id = curr_layer_id + 1;
        let mut candidates: Vec<(usize, u32)> = (0..64)
            .filter(|&eid| {
                if scores[eid] == 0 {
                    return false;
                }
                if next_layer_cache.iter().any(|entry| entry.id == eid) {
                    return false;
                }
                if let Some(storage) = gpu_dense {
                    if gpu_enabled && storage.expert_cache.contains(next_layer_id, eid) {
                        return false;
                    }
                }
                true
            })
            .map(|eid| (eid, scores[eid]))
            .collect();

        // 4. Sắp xếp giảm dần theo điểm số dự đoán
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // 5. Chọn top max_candidates ứng viên có xác suất cao nhất
        candidates
            .into_iter()
            .take(max_candidates)
            .map(|(eid, _)| eid)
            .collect()
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
                layer.bytes = layer.bytes.saturating_sub(entry.expert.resident_bytes());
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
                            } else {
                                layer.recycled_buffers.push((weights, scales));
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
        let bytes = expert.resident_bytes();
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
        // Ưu tiên trục xuất (evict) các phần tử trong Probationary segment trước
        while layer.bytes > cache_bytes_per_layer.saturating_sub(bytes) {
            let victim_idx = layer
                .cache
                .iter()
                .enumerate()
                .filter(|(_, e)| !e.is_protected)
                .min_by_key(|(_, e)| (layer.heat[e.id], e.last_used))
                .map(|(idx, _)| idx)
                .or_else(|| {
                    // Nếu không còn phần tử probationary nào, buộc phải evict phần tử nguội nhất trong protected
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
                layer.bytes = layer.bytes.saturating_sub(removed.expert.resident_bytes());
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
                if layer.recycled_buffers.len() < 16 {
                    layer.recycled_buffers.push(removed.expert.into_raw());
                }
                profile.cache_evictions += 1;
            }
        }

        if layer.bytes <= cache_bytes_per_layer - bytes {
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

        let (layers, gpu_dense, gpu_tier, index, cache_bytes_per_layer, transitions) = (
            &mut self.layers,
            &mut self.gpu_dense,
            &self.gpu_tier,
            &self.index,
            self.cache_bytes_per_layer,
            &mut self.inter_layer_transitions,
        );

        let num_layers = layers.len();
        let mut lookahead_prefetched: HashMap<usize, Int8Expert> = HashMap::new();
        let mut prev_top_experts: Vec<usize> = Vec::with_capacity(8);

        for id in 0..num_layers {
            check()?;
            let (prev_and_curr, remaining) = layers.split_at_mut(id + 1);
            let layer = &mut prev_and_curr[id];

            let attention_started = Instant::now();
            let position = kv[id].keys.len() / kd;
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
                    preloaded.insert(eid, exp);
                    hit_slots.push(slot);
                } else {
                    miss_slots.push(slot);
                    miss_ids.push(eid);
                }
            }

            // Hoàn trả các expert còn lại trong lookahead_prefetched (nếu không dùng đến) vào recycled_buffers
            for (_, unused_exp) in lookahead_prefetched.drain() {
                if layer.recycled_buffers.len() < 16 {
                    layer.recycled_buffers.push(unused_exp.into_raw());
                }
            }

            // 3. Dự đoán Speculative Lookahead Candidates cho tầng tiếp theo (id + 1):
            // Opportunistic Lookahead: Chỉ kích hoạt nạp trước khi tầng hiện tại KHÔNG CÓ MISS (toàn bộ Hit).
            // Nếu tầng hiện tại có miss, ưu tiên 100% băng thông I/O nạp miss để tránh hiện tượng Prefetch Contention.
            let next_candidates = if miss_slots.is_empty() {
                if let Some(next_l) = remaining.first() {
                    Self::predict_next_layer_experts(
                        transitions,
                        id,
                        top_experts,
                        &next_l.heat,
                        &next_l.cache,
                        gpu_dense.as_ref(),
                        gpu_tier.is_enabled(),
                        2, // Window = 2 candidates sáng giá nhất có score > 0
                    )
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

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

            if !hit_ready.is_empty() && has_io_work {
                let num_hits = hit_ready.len();
                let (hit_scratches, _) = scratch.expert_scratches[..top_k].split_at_mut(num_hits);
                let hidden = &scratch.hidden[..];

                let (io_res, compute_res) = rayon::join(
                    || {
                        let started = Instant::now();
                        let curr_loaded: HashMap<usize, Int8Expert> = miss_ids
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
                let curr_loaded: HashMap<usize, Int8Expert> = miss_ids
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

        let intermediate_size = c.intermediate_size;
        let weight_count = d * intermediate_size * 3;
        let scale_count = intermediate_size * 2 + d;

        let (layers, gpu_dense, gpu_tier, index, cache_bytes_per_layer) = (
            &mut self.layers,
            &mut self.gpu_dense,
            &self.gpu_tier,
            &self.index,
            self.cache_bytes_per_layer,
        );

        for (layer_id, layer) in layers.iter_mut().enumerate() {
            check()?;
            let base_position = kv[layer_id].keys.len() / kd;
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
                kv[layer_id].keys.extend_from_slice(k_slice);
                kv[layer_id].values.extend_from_slice(v_slice);

                let ctx_slice = &mut context_batch[local_position * d..(local_position + 1) * d];
                causal_attention_into(
                    q_slice,
                    &kv[layer_id].keys,
                    &kv[layer_id].values,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    ctx_slice,
                )?;
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
                for routes in uses.chunks(dispatch_width) {
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
            .map(|_| Kv::with_capacity(self.context_limit, kd))
            .collect();
        let mut profile = RequestProfile::default();
        let mut scratch = StepScratch::new(&self.config);
        let mut logits = Vec::with_capacity(tokens.len());
        for &token in tokens {
            self.step(token, &mut kv, &mut profile, &mut scratch, &|| Ok(()))?;
            logits.push(scratch.logits.clone());
        }
        Ok(logits)
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> io::Result<Generation> {
        self.generate_controlled(prompt, max_tokens, None, &|| Ok(()), || Ok(()), None)
    }

    fn take_prefix_kv(
        &mut self,
        tokens: &[usize],
        session_id: Option<&str>,
        kd: usize,
    ) -> (usize, Vec<Kv>) {
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
                    cache.keys.truncate(prefix * kd);
                    cache.values.truncate(prefix * kd);
                }
                return (prefix, state.kv);
            }
        }
        // Allocate only after lookup; a reusable session already owns its buffers.
        (
            0,
            (0..self.layers.len())
                .map(|_| Kv::with_capacity(self.context_limit, kd))
                .collect(),
        )
    }

    pub(crate) fn generate_controlled(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        session_id: Option<&str>,
        check: &dyn Fn() -> io::Result<()>,
        ready: impl FnOnce() -> io::Result<()>,
        mut emit: Option<&mut DeltaSink<'_>>,
    ) -> io::Result<Generation> {
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
        let (cached_prompt_tokens, mut kv) = self.take_prefix_kv(&tokens, session_id, kd);
        let mut profile = RequestProfile::default();
        let mut scratch = StepScratch::new(&self.config);
        let prefill_started = Instant::now();
        self.prefill(
            &tokens[cached_prompt_tokens..],
            &mut kv,
            &mut profile,
            &mut scratch,
            check,
        )?;
        let prefill_elapsed = prefill_started.elapsed();
        let prefill_expert_io = profile.expert_io;
        let mut output = Vec::new();
        let mut finish_reason = "length";
        let first_token_started = total_started.elapsed();
        let decode_started = Instant::now();
        let mut decode_steps = 0usize;
        let mut processed_tokens = tokens.clone();
        for step in 0..max_tokens {
            check()?;
            let token = argmax_greedy(&scratch.logits);
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
                self.step(token, &mut kv, &mut profile, &mut scratch, check)?;
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
        let total_elapsed = total_started.elapsed();
        let cache_bytes = self.layers.iter().map(|layer| layer.bytes).sum();
        let kv_bytes = kv
            .iter()
            .map(|cache| (cache.keys.len() + cache.values.len()) * std::mem::size_of::<f32>())
            .sum();
        if let Some(id) = session_id {
            if kv_bytes <= self.session_cache_bytes {
                while self.sessions.values().map(session_bytes).sum::<usize>() + kv_bytes
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
                self.sessions.insert(
                    id.to_owned(),
                    SessionState {
                        tokens: processed_tokens,
                        kv,
                        last_used: self.session_clock,
                    },
                );
            }
        }
        let seconds = decode_elapsed.as_secs_f64();
        Ok(Generation {
            text,
            token_ids: output,
            prompt_tokens: tokens.len(),
            finish_reason,
            metrics: Some(GenerationMetrics {
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

fn session_bytes(state: &SessionState) -> usize {
    state
        .kv
        .iter()
        .map(|cache| (cache.keys.len() + cache.values.len()) * std::mem::size_of::<f32>())
        .sum()
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
            let (prefix, mut kv) = model.take_prefix_kv(&tokens, Some("fork"), kd);
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
                let kv = seed
                    .kv
                    .iter()
                    .map(|cache| {
                        let mut copy = Kv::with_capacity(
                            model.context_limit,
                            cache.keys.len() / seed.tokens.len(),
                        );
                        copy.keys.extend_from_slice(&cache.keys);
                        copy.values.extend_from_slice(&cache.values);
                        copy
                    })
                    .collect();
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

        let candidates = OlmoeModel::predict_next_layer_experts(
            &transitions,
            0,
            &layer0_experts,
            &next_layer_heat,
            &next_layer_cache,
            None,
            false,
            3,
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
}
