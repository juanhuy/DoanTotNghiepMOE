//! Dump engine logits for real-model parity against `tools/stream_reference.py`.
//!
//! usage: dump_logits MODEL_DIR IDS_JSON OUT_DIR
//! IDS_JSON is `{"sequences": [[id, ...], ...]}`; writes `seq-<i>.f32` (little-endian,
//! positions x vocab, row-major) and `engine.json` (shapes, load/forward settings).
//! Respects the usual MOE_* environment (set MOE_GPU_MODE=cpu for a CPU-only reference).
use moe_tier_engine::olmoe::OlmoeModel;
use serde::Deserialize;
use std::{io::Write, path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Ids {
    sequences: Vec<Vec<usize>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: dump_logits MODEL_DIR IDS_JSON OUT_DIR".into());
    }
    let ids: Ids = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let out = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&out)?;
    let cache_mib: usize = std::env::var("DUMP_CACHE_MIB")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(64);
    let context = ids.sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let started = Instant::now();
    let mut model = OlmoeModel::load(&args[1], cache_mib << 20, context, 8 << 30)?;
    let load_s = started.elapsed().as_secs_f64();
    let status = model.expert_status();
    let mut shapes = Vec::new();
    for (i, sequence) in ids.sequences.iter().enumerate() {
        let t = Instant::now();
        let logits = model.logits(sequence)?;
        let vocab = logits.first().map_or(0, Vec::len);
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(out.join(format!("seq-{i}.f32")))?);
        for row in &logits {
            for v in row {
                file.write_all(&v.to_le_bytes())?;
            }
        }
        file.flush()?;
        eprintln!(
            "seq {i}: {} positions in {:.1}s",
            logits.len(),
            t.elapsed().as_secs_f64()
        );
        shapes.push(serde_json::json!({"positions": logits.len(), "vocab": vocab}));
    }
    let meta = serde_json::json!({
        "model_dir": args[1],
        "cache_mib_per_layer": cache_mib,
        "context": context,
        "load_seconds": load_s,
        "model_type": status.model_type,
        "dense_storage": status.dense_storage,
        "embedding_on_disk": status.embedding_on_disk,
        "gpu_dense_offloaded": status.gpu.dense_offloaded,
        "gpu_expert_compute_enabled": status.gpu_expert_compute_enabled,
        "shapes": shapes,
    });
    std::fs::write(out.join("engine.json"), serde_json::to_vec_pretty(&meta)?)?;
    Ok(())
}
