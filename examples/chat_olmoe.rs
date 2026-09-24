use moe_tier_engine::{engine::ChatMessage, InferenceEngine};
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: chat_olmoe MODEL_DIRECTORY PROMPT [MAX_TOKENS]".into());
    }
    let max = args.get(2).map(|v| v.parse()).transpose()?.unwrap_or(16);
    let start = std::time::Instant::now();
    let engine =
        InferenceEngine::load_with_limits(&args[0], 128 * 1024 * 1024, 512, 3 * 1024 * 1024 * 1024)
            .await?;
    eprintln!(
        "Loaded {} in {:.2}s",
        engine.model_name(),
        start.elapsed().as_secs_f64()
    );
    let start = std::time::Instant::now();
    let output = engine
        .chat(
            vec![ChatMessage {
                role: "user".into(),
                content: args[1].clone(),
            }],
            max,
        )
        .await?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    eprintln!("Generation took {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}
