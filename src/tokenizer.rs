//! Toy byte vocabulary, not a tokenizer for OLMoE or other pretrained models.
pub const BOS: usize = 256;
pub const EOS: usize = 257;
pub const VOCAB_SIZE: usize = 258;

pub fn encode(text: &str) -> Vec<usize> {
    std::iter::once(BOS)
        .chain(text.bytes().map(usize::from))
        .collect()
}

pub fn decode(tokens: &[usize]) -> String {
    let bytes: Vec<_> = tokens
        .iter()
        .copied()
        .filter(|&id| id < 256)
        .map(|id| id as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    #[test]
    fn vietnamese_round_trip() {
        assert_eq!(
            super::decode(&super::encode("Xin chào Việt Nam!")),
            "Xin chào Việt Nam!"
        );
    }
}
