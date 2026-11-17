//! Render a checkpoint's chat template (template compatibility probe).
//! Usage: cargo run --release --example render_chat_probe -- <model_dir> '<messages JSON>'
//! The JSON is a list of {"role", "content"}; a plain string is one user message.
use moe_tier_engine::{engine::ChatMessage, olmoe::OlmoeModel};

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("model dir");
    let input = args.next().unwrap_or_else(|| "Hello".into());
    let messages: Vec<ChatMessage> = serde_json::from_str(&input).unwrap_or_else(|_| {
        vec![ChatMessage {
            role: "user".into(),
            content: input.clone(),
        }]
    });
    let model = OlmoeModel::load(&dir, 0, 32, usize::MAX / 4)?;
    print!("{}", model.render_chat(&messages)?);
    Ok(())
}
