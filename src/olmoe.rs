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
            || self.num_experts > 1024
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
    cache_hits: u64,
    cache_misses: u64,
    cache_evictions: u64,
    bytes_read: u64,
}

struct StepScratch {
    expert: ExpertScratch,
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
        Self {
            expert: ExpertScratch::new(hidden, config.intermediate_size),
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
    gpu_tier: crate::gpu::GpuTierManager,
    gpu_dense: Option<crate::gpu::GpuDenseStorage>,
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
                            engine,
                            layers: gpu_layers,
                            lm_head: lm_head_gpu,
                            total_allocated_bytes: total_bytes,
                            expert_cache: crate::gpu::GpuExpertCache::new(expert_cache_budget),
                            expert_pool,
                            pinned_upload_buf,
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

        Ok(Self {
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
            gpu_tier,
            gpu_dense,
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
        self.gpu_tier.toggle_enabled();
        self.gpu_tier.status()
    }

    /// Thiết lập trạng thái Bật / Tắt GPU Tier
    pub fn set_gpu_enabled(&mut self, enabled: bool) -> crate::gpu::GpuTierStatus {
        self.gpu_tier.set_enabled(enabled);
        self.gpu_tier.status()
    }

    /// Xóa toàn bộ bộ đệm (KV session cache, Host RAM L2 expert cache, GPU VRAM L1 expert cache)
    /// Đưa hệ thống về trạng thái Khởi động lạnh (Cold Start / 0 cache) hoàn toàn
    pub fn reset_cache(&mut self) {
        // 1. Xóa toàn bộ KV session cache
        self.sessions.clear();
        self.session_cache_bytes = 0;
        self.session_clock = 0;

        // 2. Xóa toàn bộ Host RAM L2 expert cache và đặt lại heatmap
        for layer in &mut self.layers {
            layer.cache.clear();
            layer.bytes = 0;
            layer.heat.fill(0);
            layer.clock = 0;
        }

        // 3. Xóa GPU VRAM L1 expert cache và hoàn trả các slot về VRAM Pool
        if let Some(storage) = self.gpu_dense.as_mut() {
            storage.clear_expert_cache();
        }
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

    /// Thăng hạng một Expert INT8 lên L1 VRAM Cache nếu còn ngân sách và chưa có trong cache
    fn promote_to_vram(
        gpu_dense: &mut Option<crate::gpu::GpuDenseStorage>,
        gpu_enabled: bool,
        layer_id: usize,
        expert_id: usize,
        weights: &[u8],
        scales: &[f32],
    ) {
        if let Some(storage) = gpu_dense.as_mut() {
            if gpu_enabled && storage.expert_cache.max_bytes > 0 {
                storage.upload_and_cache(layer_id, expert_id, weights, scales);
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
        layer: &Layer,
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

        let started = Instant::now();
        let loaded: HashMap<usize, Int8Expert> = missing_ids
            .into_par_iter()
            .filter_map(|eid| {
                Int8Expert::load(index, layer_id, eid, d, intermediate_size, 256 * 1024 * 1024)
                    .ok()
                    .map(|exp| (eid, exp))
            })
            .collect();
        let elapsed = started.elapsed();
        (loaded, elapsed)
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
        profile: &mut RequestProfile,
        uses_count: u64,
        preloaded: &mut HashMap<usize, Int8Expert>,
    ) -> io::Result<(Int8Expert, bool)> {
        let (expert, l2_hit) = if let Some(pos) =
            layer.cache.iter().position(|entry| entry.id == expert_id)
        {
            profile.cache_hits += uses_count;
            let entry = layer.cache.remove(pos).unwrap();
            layer.bytes = layer.bytes.saturating_sub(entry.expert.resident_bytes());
            (entry.expert, true)
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
                        let mut weights = vec![0u8; weight_count];
                        let mut scales = vec![0.0f32; scale_count];
                        if engine
                            .download_expert(gpu_weights, &mut weights, &mut scales)
                            .is_ok()
                        {
                            let exp = Int8Expert::from_raw(d, intermediate_size, weights, scales);
                            profile.expert_io += io_started.elapsed();
                            profile.bytes_read += exp.resident_bytes() as u64;
                            profile.cache_hits += uses_count;
                            from_vram = Some(exp);
                        }
                    }
                }
            }

            if let Some(exp) = from_vram {
                (exp, false)
            } else {
                // Cấp 2: Miss cả L1 và L2 -> Lấy từ preloaded (16-B) hoặc nạp từ SSD NVMe Cold Storage
                profile.cache_misses += 1;
                profile.cache_hits += uses_count.saturating_sub(1);
                let (exp, io_elapsed) = if let Some(p) = preloaded.remove(&expert_id) {
                    (p, Duration::ZERO)
                } else {
                    let io_started = Instant::now();
                    let loaded = Int8Expert::load(
                        index,
                        layer_id,
                        expert_id,
                        d,
                        intermediate_size,
                        256 * 1024 * 1024,
                    )?;
                    (loaded, io_started.elapsed())
                };
                profile.expert_io += io_elapsed;
                profile.bytes_read += exp.resident_bytes() as u64;

                // Tự động thăng hạng (promote) expert vừa đọc lên L1 VRAM Cache
                Self::promote_to_vram(
                    gpu_dense,
                    gpu_enabled,
                    layer_id,
                    expert_id,
                    exp.weights(),
                    exp.scales(),
                );
                (exp, false)
            }
        };

        if l2_hit {
            Self::promote_to_vram(
                gpu_dense,
                gpu_enabled,
                layer_id,
                expert_id,
                expert.weights(),
                expert.scales(),
            );
        }

        Ok((expert, l2_hit))
    }


    /// Quản lý nạp và trục xuất (admission & eviction) cho L2 Host RAM Cache
    fn admit_expert_l2(
        layer: &mut Layer,
        expert_id: usize,
        expert: Int8Expert,
        l2_hit: bool,
        cache_bytes_per_layer: usize,
        profile: &mut RequestProfile,
    ) {
        let bytes = expert.resident_bytes();
        if bytes <= cache_bytes_per_layer {
            let mut admit = l2_hit || layer.bytes <= cache_bytes_per_layer - bytes;
            while !admit && layer.bytes > cache_bytes_per_layer - bytes {
                let victim = layer
                    .cache
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, entry)| (layer.heat[entry.id], entry.last_used))
                    .map(|(position, entry)| (position, layer.heat[entry.id], entry.last_used));
                let Some(victim) = victim else {
                    break;
                };
                if layer.heat[expert_id] < victim.1 {
                    break;
                }
                if let Some(removed) = layer.cache.remove(victim.0) {
                    layer.bytes = layer.bytes.saturating_sub(removed.expert.resident_bytes());
                    profile.cache_evictions += 1;
                }
                admit = layer.bytes <= cache_bytes_per_layer - bytes;
            }
            if admit {
                layer.bytes += bytes;
                layer.cache.push_back(CachedExpert {
                    id: expert_id,
                    expert,
                    last_used: layer.clock,
                });
            }
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

        let (layers, gpu_dense, gpu_tier, index, cache_bytes_per_layer) = (
            &mut self.layers,
            &mut self.gpu_dense,
            &self.gpu_tier,
            &self.index,
            self.cache_bytes_per_layer,
        );

        for (id, layer) in layers.iter_mut().enumerate() {
            check()?;
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
            valid_indices.select_nth_unstable_by(top_k, |&a, &b| {
                scratch.router[b]
                    .total_cmp(&scratch.router[a])
                    .then_with(|| a.cmp(&b))
            });
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
            let (mut preloaded, prefetch_io) = Self::prefetch_missing_experts(
                id,
                top_experts,
                d,
                intermediate_size,
                index,
                layer,
                gpu_dense.as_ref(),
                gpu_tier.is_enabled(),
            );
            profile.expert_io += prefetch_io;

            for &expert_id in top_experts.iter() {
                check()?;
                layer.clock = layer.clock.wrapping_add(1);
                if layer.clock % 65_536 == 0 {
                    layer.heat.iter_mut().for_each(|heat| *heat /= 2);
                }
                layer.heat[expert_id] = layer.heat[expert_id].saturating_add(1);
                let (expert, l2_hit) = Self::acquire_expert_tiered(
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
                    profile,
                    1,
                    &mut preloaded,
                )?;


                let expert_started = Instant::now();
                let result = expert.forward_reuse(&scratch.hidden, &mut scratch.expert)?;
                profile.expert_compute += expert_started.elapsed();
                for (a, b) in scratch.x.iter_mut().zip(result.iter().copied()) {
                    *a += scratch.router[expert_id] / normalizer * b;
                }

                Self::admit_expert_l2(
                    layer,
                    expert_id,
                    expert,
                    l2_hit,
                    cache_bytes_per_layer,
                    profile,
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
        let route_count = tokens
            .len()
            .checked_mul(c.num_experts_per_tok)
            .ok_or_else(|| invalid("prefill route count overflow"))?;
        let contribution_count = route_count
            .checked_mul(d)
            .ok_or_else(|| invalid("prefill contribution size overflow"))?;
        let mut contributions = vec![0.0; contribution_count];
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
            for local_position in 0..tokens.len() {
                check()?;
                scratch
                    .x
                    .copy_from_slice(&batch[local_position * d..(local_position + 1) * d]);
                let attention_started = Instant::now();
                let position = base_position + local_position;
                rms_norm_into(
                    &scratch.x,
                    &layer.input_norm,
                    c.rms_norm_eps,
                    &mut scratch.hidden,
                )?;
                Self::dense_qkv(&*gpu_dense, gpu_tier.is_enabled(), layer_id, layer, scratch)?;
                rms_norm_in_place(&mut scratch.query, &layer.q_norm, c.rms_norm_eps)?;
                rms_norm_in_place(&mut scratch.key, &layer.k_norm, c.rms_norm_eps)?;
                prepare_rope_sincos(&mut scratch.sincos, &scratch.rope_bases, position);
                apply_rope_cached(&mut scratch.query, hd, &scratch.sincos);
                apply_rope_cached(&mut scratch.key, hd, &scratch.sincos);
                kv[layer_id].keys.extend_from_slice(&scratch.key);
                kv[layer_id].values.extend_from_slice(&scratch.value);
                causal_attention_into(
                    &scratch.query,
                    &kv[layer_id].keys,
                    &kv[layer_id].values,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    &mut scratch.context,
                )?;
                Self::dense_gemv(
                    &*gpu_dense,
                    gpu_tier.is_enabled(),
                    &layer.o,
                    |g| &g.layers[layer_id].o_proj,
                    &scratch.context,
                    &mut scratch.projected,
                )?;
                for (value, projected) in scratch.x.iter_mut().zip(&scratch.projected) {
                    *value += projected;
                }
                profile.attention += attention_started.elapsed();
                batch[local_position * d..(local_position + 1) * d].copy_from_slice(&scratch.x);

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
                valid_indices.select_nth_unstable_by(top_k, |&left, &right| {
                    scratch.router[right]
                        .total_cmp(&scratch.router[left])
                        .then_with(|| left.cmp(&right))
                });
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
                let (expert, l2_hit) = Self::acquire_expert_tiered(
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
                    profile,
                    uses.len() as u64,
                    &mut preloaded,
                )?;


                for &(token, slot, weight) in uses {
                    check()?;
                    let expert_started = Instant::now();
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
                }

                Self::admit_expert_l2(
                    layer,
                    expert_id,
                    expert,
                    l2_hit,
                    cache_bytes_per_layer,
                    profile,
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
        let mut cached_prompt_tokens = 0;
        let d = self.config.hidden_size;
        let kd = d / self.config.num_attention_heads * self.config.num_key_value_heads;
        let mut kv: Vec<_> = (0..self.layers.len())
            .map(|_| Kv::with_capacity(self.context_limit, kd))
            .collect();
        if let Some(id) = session_id {
            if let Some(state) = self.sessions.remove(id) {
                let prefix = state.tokens.len();
                if prefix < tokens.len() && tokens.starts_with(&state.tokens) {
                    cached_prompt_tokens = prefix;
                    kv = state.kv;
                }
            }
        }
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
}
