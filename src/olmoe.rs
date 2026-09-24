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
use serde::Deserialize;
use std::{
    collections::{HashMap, VecDeque},
    fs, io,
    path::Path,
    time::{Duration, Instant},
};

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
    context: Vec<f32>,
    projected: Vec<f32>,
    router: Vec<f32>,
    logits: Vec<f32>,
}

#[derive(Clone, Copy)]
struct Route {
    expert: usize,
    weight: f32,
}

impl StepScratch {
    fn new(config: &Config) -> Self {
        let hidden = config.hidden_size;
        let key_value = hidden / config.num_attention_heads * config.num_key_value_heads;
        Self {
            expert: ExpertScratch::new(hidden, config.intermediate_size),
            x: vec![0.0; hidden],
            hidden: vec![0.0; hidden],
            query: vec![0.0; hidden],
            key: vec![0.0; key_value],
            value: vec![0.0; key_value],
            context: vec![0.0; hidden],
            projected: vec![0.0; hidden],
            router: vec![0.0; config.num_experts],
            logits: vec![0.0; config.vocab_size],
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
        })
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
        scratch
            .x
            .copy_from_slice(&self.embedding.values[token * d..(token + 1) * d]);
        for (id, layer) in self.layers.iter_mut().enumerate() {
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
            layer
                .q
                .multiply_loaded_into(&scratch.hidden, &mut scratch.query)?;
            layer
                .k
                .multiply_loaded_into(&scratch.hidden, &mut scratch.key)?;
            layer
                .v
                .multiply_loaded_into(&scratch.hidden, &mut scratch.value)?;
            rms_norm_in_place(&mut scratch.query, &layer.q_norm, c.rms_norm_eps)?;
            rms_norm_in_place(&mut scratch.key, &layer.k_norm, c.rms_norm_eps)?;
            rotate(&mut scratch.query, hd, position, c.rope_theta);
            rotate(&mut scratch.key, hd, position, c.rope_theta);
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
            layer
                .o
                .multiply_loaded_into(&scratch.context, &mut scratch.projected)?;
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
            let mut ids: Vec<_> = (0..c.num_experts).collect();
            ids.sort_by(|&a, &b| {
                scratch.router[b]
                    .total_cmp(&scratch.router[a])
                    .then_with(|| a.cmp(&b))
            });
            ids.truncate(c.num_experts_per_tok);
            let normalizer = if c.norm_topk_prob {
                ids.iter().map(|&expert| scratch.router[expert]).sum()
            } else {
                1.
            };
            for expert_id in ids {
                check()?;
                layer.clock = layer.clock.wrapping_add(1);
                if layer.clock % 65_536 == 0 {
                    layer.heat.iter_mut().for_each(|heat| *heat /= 2);
                }
                layer.heat[expert_id] = layer.heat[expert_id].saturating_add(1);
                let (expert, cache_hit) =
                    if let Some(pos) = layer.cache.iter().position(|entry| entry.id == expert_id) {
                        profile.cache_hits += 1;
                        let entry = layer.cache.remove(pos).unwrap();
                        layer.bytes -= entry.expert.resident_bytes();
                        (entry.expert, true)
                    } else {
                        profile.cache_misses += 1;
                        let io_started = Instant::now();
                        let expert = Int8Expert::load(
                            &self.index,
                            id,
                            expert_id,
                            d,
                            c.intermediate_size,
                            256 * 1024 * 1024,
                        )?;
                        profile.expert_io += io_started.elapsed();
                        profile.bytes_read += expert.resident_bytes() as u64;
                        (expert, false)
                    };
                let expert_started = Instant::now();
                let result = expert.forward_reuse(&scratch.hidden, &mut scratch.expert)?;
                profile.expert_compute += expert_started.elapsed();
                for (a, b) in scratch.x.iter_mut().zip(result.iter().copied()) {
                    *a += scratch.router[expert_id] / normalizer * b;
                }
                let bytes = expert.resident_bytes();
                if bytes <= self.cache_bytes_per_layer {
                    let mut admit = cache_hit || layer.bytes <= self.cache_bytes_per_layer - bytes;
                    while !admit && layer.bytes > self.cache_bytes_per_layer - bytes {
                        let victim = layer
                            .cache
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, entry)| (layer.heat[entry.id], entry.last_used))
                            .map(|(position, entry)| {
                                (position, layer.heat[entry.id], entry.last_used)
                            })
                            .unwrap();
                        if layer.heat[expert_id] < victim.1 {
                            break;
                        }
                        let removed = layer.cache.remove(victim.0).unwrap();
                        layer.bytes -= removed.expert.resident_bytes();
                        profile.cache_evictions += 1;
                        admit = layer.bytes <= self.cache_bytes_per_layer - bytes;
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
            finite(&scratch.x)?;
        }
        let head_started = Instant::now();
        rms_norm_into(&scratch.x, &self.norm, c.rms_norm_eps, &mut scratch.hidden)?;
        self.head
            .multiply_loaded_into(&scratch.hidden, &mut scratch.logits)?;
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

        for (layer_id, layer) in self.layers.iter_mut().enumerate() {
            check()?;
            let base_position = kv[layer_id].keys.len() / kd;
            if base_position
                .checked_add(tokens.len())
                .is_none_or(|length| length > self.context_limit)
            {
                return Err(invalid("context exhausted"));
            }
            let mut routes = Vec::with_capacity(tokens.len());
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
                layer
                    .q
                    .multiply_loaded_into(&scratch.hidden, &mut scratch.query)?;
                layer
                    .k
                    .multiply_loaded_into(&scratch.hidden, &mut scratch.key)?;
                layer
                    .v
                    .multiply_loaded_into(&scratch.hidden, &mut scratch.value)?;
                rms_norm_in_place(&mut scratch.query, &layer.q_norm, c.rms_norm_eps)?;
                rms_norm_in_place(&mut scratch.key, &layer.k_norm, c.rms_norm_eps)?;
                rotate(&mut scratch.query, hd, position, c.rope_theta);
                rotate(&mut scratch.key, hd, position, c.rope_theta);
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
                layer
                    .o
                    .multiply_loaded_into(&scratch.context, &mut scratch.projected)?;
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
                let mut ids: Vec<_> = (0..c.num_experts).collect();
                ids.sort_by(|&left, &right| {
                    scratch.router[right]
                        .total_cmp(&scratch.router[left])
                        .then_with(|| left.cmp(&right))
                });
                ids.truncate(c.num_experts_per_tok);
                let normalizer = if c.norm_topk_prob {
                    ids.iter().map(|&expert| scratch.router[expert]).sum()
                } else {
                    1.0
                };
                routes.push(
                    ids.into_iter()
                        .map(|expert| Route {
                            expert,
                            weight: scratch.router[expert] / normalizer,
                        })
                        .collect::<Vec<_>>(),
                );
            }

            contributions.fill(0.0);
            for expert_id in 0..c.num_experts {
                let uses: Vec<_> = routes
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
                let (expert, cache_hit) = if let Some(position) =
                    layer.cache.iter().position(|entry| entry.id == expert_id)
                {
                    profile.cache_hits += uses.len() as u64;
                    let entry = layer.cache.remove(position).unwrap();
                    layer.bytes -= entry.expert.resident_bytes();
                    (entry.expert, true)
                } else {
                    profile.cache_misses += 1;
                    profile.cache_hits += uses.len().saturating_sub(1) as u64;
                    let io_started = Instant::now();
                    let expert = Int8Expert::load(
                        &self.index,
                        layer_id,
                        expert_id,
                        d,
                        c.intermediate_size,
                        256 * 1024 * 1024,
                    )?;
                    profile.expert_io += io_started.elapsed();
                    profile.bytes_read += expert.resident_bytes() as u64;
                    (expert, false)
                };
                for &(token, slot, weight) in &uses {
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
                let bytes = expert.resident_bytes();
                if bytes <= self.cache_bytes_per_layer {
                    let mut admit = cache_hit || layer.bytes <= self.cache_bytes_per_layer - bytes;
                    while !admit && layer.bytes > self.cache_bytes_per_layer - bytes {
                        let victim = layer
                            .cache
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, entry)| (layer.heat[entry.id], entry.last_used))
                            .map(|(position, entry)| {
                                (position, layer.heat[entry.id], entry.last_used)
                            })
                            .unwrap();
                        if layer.heat[expert_id] < victim.1 {
                            break;
                        }
                        let removed = layer.cache.remove(victim.0).unwrap();
                        layer.bytes -= removed.expert.resident_bytes();
                        profile.cache_evictions += 1;
                        admit = layer.bytes <= self.cache_bytes_per_layer - bytes;
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
        self.head
            .multiply_loaded_into(&scratch.hidden, &mut scratch.logits)?;
        profile.lm_head += head_started.elapsed();
        Ok(())
    }

    /// Logits at every prefix, for numerical comparison with a reference decoder.
    pub fn logits(&mut self, tokens: &[usize]) -> io::Result<Vec<Vec<f32>>> {
        if tokens.is_empty() || tokens.len() > self.context_limit {
            return Err(invalid("invalid token count"));
        }
        let mut kv: Vec<_> = (0..self.layers.len()).map(|_| Kv::default()).collect();
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
        let mut kv: Vec<_> = (0..self.layers.len()).map(|_| Kv::default()).collect();
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
            let token = (0..scratch.logits.len())
                .max_by(|&a, &b| {
                    scratch.logits[a]
                        .total_cmp(&scratch.logits[b])
                        .then_with(|| b.cmp(&a))
                })
                .unwrap();
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
fn rotate(x: &mut [f32], hd: usize, position: usize, theta: f32) {
    for head in x.chunks_exact_mut(hd) {
        for i in 0..hd / 2 {
            let angle = position as f32 / theta.powf(2. * i as f32 / hd as f32);
            let (sin, cos) = angle.sin_cos();
            let a = head[i];
            let b = head[i + hd / 2];
            head[i] = a * cos - b * sin;
            head[i + hd / 2] = b * cos + a * sin;
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
