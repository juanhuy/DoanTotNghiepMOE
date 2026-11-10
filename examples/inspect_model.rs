//! cargo run --example inspect_model -- /path/to/olmoe_merged
use moe_tier_engine::{int8_expert::Int8Expert, safetensors::TensorIndex};
use serde::Deserialize;
use std::{io, path::Path};

#[derive(Deserialize)]
struct Config {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args()
        .nth(1)
        .ok_or("usage: inspect_model DIRECTORY")?;
    let config: Config = serde_json::from_reader(std::fs::File::open(
        Path::new(&directory).join("config.json"),
    )?)?;
    if config.model_type != "olmoe" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected model_type olmoe").into());
    }
    let index = TensorIndex::open(&directory)?;
    println!("Indexed {} tensors (headers only)", index.tensors().len());
    let expert = Int8Expert::load(
        &index,
        0,
        0,
        config.hidden_size,
        config.intermediate_size,
        128 * 1024 * 1024,
    )?;
    println!(
        "Loaded layer 0 expert 0: {} resident bytes",
        expert.resident_bytes()
    );
    let output = expert.forward(&vec![1.0; config.hidden_size])?;
    println!(
        "Synthetic all-ones input: {} output values, first values {:?}",
        output.len(),
        &output[..output.len().min(8)]
    );
    println!("Expert smoke test only; not full OLMoE inference.");
    Ok(())
}
