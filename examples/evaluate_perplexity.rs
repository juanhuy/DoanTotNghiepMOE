//! Perplexity and Cross-Entropy scientific evaluation suite for OLMoE.
//! Computes mathematical token-by-token cross entropy and perplexity
//! over test corpora for academic verification and thesis benchmarks.

use moe_tier_engine::InferenceEngine;
use serde_json::json;
use std::{env, fs, path::Path, time::Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.len() > 2 {
        eprintln!("Sử dụng: evaluate_perplexity MODEL_DIRECTORY [TEXT_OR_FILE_PATH]");
        eprintln!();
        eprintln!("Ví dụ:");
        eprintln!(
            "  evaluate_perplexity models/olmoe-1b-7b-int8 'Hello world, how are you today?'"
        );
        eprintln!(
            "  evaluate_perplexity models/olmoe-1b-7b-int8 benchmarks/suites/test_corpus.txt"
        );
        std::process::exit(1);
    }

    let model_dir = &args[0];
    let default_corpus = "Mixture of Experts models scale parameter count while maintaining constant compute per token. OLMoE is a sparse language model utilizing top-k routing across multiple feed-forward expert networks.";
    let eval_text = if args.len() == 2 {
        let input_arg = &args[1];
        if Path::new(input_arg).is_file() {
            fs::read_to_string(input_arg)?
        } else {
            input_arg.clone()
        }
    } else {
        default_corpus.to_string()
    };

    eprintln!("============================================================");
    eprintln!("     MOE-TIERENGINE HỌC THUẬT: ĐO LƯỜNG ĐỘ PHỨC TẠP (PERPLEXITY)");
    eprintln!("============================================================");
    eprintln!("Đường dẫn mô hình: {}", model_dir);
    eprintln!(
        "Độ dài văn bản thử nghiệm: {} ký tự",
        eval_text.chars().count()
    );

    let requested_context = env::var("MOE_CONTEXT_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let config_path = Path::new(model_dir).join("config.json");
    let actual_context = if let Ok(data) = fs::read(&config_path) {
        if let Ok(cfg) = serde_json::from_slice::<serde_json::Value>(&data) {
            cfg.get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .map(|v| (v as usize).min(requested_context))
                .unwrap_or(requested_context)
        } else {
            requested_context
        }
    } else {
        requested_context
    };

    let load_start = Instant::now();
    let engine = InferenceEngine::load_with_limits(
        model_dir,
        64 * 1024 * 1024,
        actual_context,
        3 * 1024 * 1024 * 1024,
    )
    .await?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Thời gian nạp mô hình: {:.2} ms", load_ms);

    if engine.model_name() != "olmoe" {
        return Err("Perplexity evaluation yêu cầu kiến trúc OLMoE".into());
    }

    eprintln!("Bắt đầu tính toán phân phối Logits và Cross-Entropy Loss...");
    let metrics = engine.compute_perplexity(&eval_text).await?;

    eprintln!();
    eprintln!("---------------- KẾT QUẢ ĐO LƯỜNG KHOA HỌC -----------------");
    eprintln!(
        "Tổng số token (Context Tokens):     {}",
        metrics.token_count
    );
    eprintln!(
        "Số token được đánh giá (N-1):       {}",
        metrics.evaluated_tokens
    );
    eprintln!(
        "Cross-Entropy Loss (NLL Mean):      {:.4} nats",
        metrics.cross_entropy_loss
    );
    eprintln!(
        "Perplexity (PPL = exp(Loss)):       {:.4}",
        metrics.perplexity
    );
    eprintln!(
        "Bits-Per-Character (BPC):           {:.4} bits/char",
        metrics.bits_per_character
    );
    eprintln!(
        "Thời gian thực thi (Latency):       {:.2} ms",
        metrics.eval_time_ms
    );
    eprintln!(
        "Tốc độ đánh giá (Throughput):       {:.2} tokens/s",
        metrics.tokens_per_second
    );
    eprintln!("------------------------------------------------------------");

    let output = json!({
        "schema_version": 1,
        "evaluation_type": "standard_language_modeling_perplexity",
        "model_directory": model_dir,
        "load_time_ms": load_ms,
        "metrics": metrics,
        "formula": {
            "cross_entropy": "- (1 / (N - 1)) * sum_{t=1}^{N-1} ln(P(x_{t+1} | x_{1..t}))",
            "perplexity": "exp(cross_entropy_loss)",
            "log_sum_exp_stable": "max(z) + ln(sum(exp(z - max(z))))"
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}
