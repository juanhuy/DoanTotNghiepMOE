#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    Int8,
    Int4,
}

pub fn dequantize(values: &[u8], scale: f32, format: Quantization) -> Vec<f32> {
    match format {
        Quantization::Int8 => values
            .iter()
            .map(|value| (*value as i8 as f32) * scale)
            .collect(),
        Quantization::Int4 => values
            .iter()
            .flat_map(|byte| {
                let low = (byte & 0x0f) as i8;
                let high = ((byte >> 4) & 0x0f) as i8;
                [decode_int4(low) * scale, decode_int4(high) * scale]
            })
            .collect(),
    }
}

fn decode_int4(value: i8) -> f32 {
    (if value < 8 { value } else { value - 16 }) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_signed_int8() {
        assert_eq!(
            dequantize(&[0x7f, 0x80], 0.5, Quantization::Int8),
            vec![63.5, -64.0]
        );
    }

    #[test]
    fn unpacks_signed_int4_nibbles() {
        assert_eq!(
            dequantize(&[0x18], 1.0, Quantization::Int4),
            vec![-8.0, 1.0]
        );
    }
}
