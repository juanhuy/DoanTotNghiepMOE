use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    Int8,
    Int4,
    Fp8E4m3,
}

/// Bảng tra cứu trước (LUT) ánh xạ 256 mã byte của định dạng FP8 E4M3 (OCP standard) sang giá trị f32.
/// Giải mã tức thời O(1) không tốn chu kỳ tính toán giải nén của CPU (kế thừa từ Colibrì kv_fp8.h).
pub static FP8_E4M3_LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut lut = [0.0f32; 256];
    for (b, entry) in lut.iter_mut().enumerate() {
        let e = ((b >> 3) & 0xF) as i32;
        let m = (b & 7) as f32;
        let pos_v = if e > 0 {
            (1.0 + m / 8.0) * (2.0f32).powi(e - 7)
        } else {
            m * (2.0f32).powi(-9)
        };
        let pos_v = if e == 15 && (b & 7) == 7 { 0.0 } else { pos_v };
        *entry = if (b & 0x80) != 0 { -pos_v } else { pos_v };
    }
    lut
});

/// Giải mã byte FP8 E4M3 sang f32 qua bảng tra cứu O(1)
#[inline(always)]
pub fn fp8_e4m3_decode(byte: u8) -> f32 {
    FP8_E4M3_LUT[byte as usize]
}

/// Mã hóa giá trị f32 sang byte FP8 E4M3 theo chuẩn OCP e4m3 (saturate tại +-448, round-to-nearest-even)
pub fn fp8_e4m3_encode(f: f32) -> u8 {
    if !f.is_finite() {
        return 0;
    }
    let s = if f.is_sign_negative() { 0x80u8 } else { 0u8 };
    let a = f.abs();
    if a > 448.0 {
        return s | 0x7e;
    }
    // Ngưỡng denormal: 2^-6 = 0.015625, bước nhảy 2^-9 = 1.0 / 512.0
    if a < 0.015625 {
        let k = (a * 512.0).round() as u8;
        return s | k;
    }
    let bits = a.to_bits();
    let f_exp = (((bits >> 23) & 0xFF) as i32) - 127;
    let mantissa = (bits & 0x7FFFFF) as f64 / 8388608.0;
    let m = 0.5 + mantissa * 0.5;
    let e = f_exp + 1;
    let mut big_e = e + 6;
    let mut k = (m * 16.0).round() as i32;
    if k == 16 {
        k = 8;
        big_e += 1;
    }
    if big_e > 15 || (big_e == 15 && k > 14) {
        return s | 0x7e;
    }
    s | (((big_e as u8) << 3) & 0x78) | (((k - 8) as u8) & 7)
}

/// Lượng tử hóa hàng f32 sang FP8 E4M3 với hệ số co giãn per-row scale = amax / 448.0
pub fn quantize_row_fp8(src: &[f32], dst: &mut [u8]) -> f32 {
    let mut amax = 0.0f32;
    for &val in src {
        let a = val.abs();
        if a > amax {
            amax = a;
        }
    }
    if amax <= 1e-35 || !amax.is_finite() {
        dst[..src.len()].fill(0);
        return 1.0;
    }
    let inv = 448.0 / amax;
    for (out, &val) in dst[..src.len()].iter_mut().zip(src) {
        *out = fp8_e4m3_encode(val * inv);
    }
    amax / 448.0
}

/// Giải lượng tử hóa hàng FP8 E4M3 sang mảng f32
pub fn dequantize_row_fp8(src: &[u8], scale: f32, dst: &mut [f32]) {
    for (out, &byte) in dst[..src.len()].iter_mut().zip(src) {
        *out = FP8_E4M3_LUT[byte as usize] * scale;
    }
}

/// Tích vô hướng giữa vector query f32 và hàng key FP8: sum(q * LUT[k]) * scale
#[inline(always)]
pub fn dot_fp8_f32(query: &[f32], key_fp8: &[u8], scale: f32) -> f32 {
    let mut sum = 0.0f32;
    for (&q, &k) in query.iter().zip(key_fp8) {
        sum += q * FP8_E4M3_LUT[k as usize];
    }
    sum * scale
}

/// Tích lũy có trọng số từ hàng value FP8 vào vector output: output += weight_scale * LUT[v]
#[inline(always)]
pub fn accumulate_fp8(output: &mut [f32], val_fp8: &[u8], weight_scale: f32) {
    for (out, &v) in output.iter_mut().zip(val_fp8) {
        *out += weight_scale * FP8_E4M3_LUT[v as usize];
    }
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
        Quantization::Fp8E4m3 => values
            .iter()
            .map(|&byte| FP8_E4M3_LUT[byte as usize] * scale)
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

    #[test]
    fn test_fp8_e4m3_lut_and_roundtrip() {
        // Kiểm tra tính đối xứng và các điểm mốc chính xác của FP8 E4M3
        assert_eq!(FP8_E4M3_LUT[0], 0.0);
        assert_eq!(FP8_E4M3_LUT[0x7e], 448.0);
        assert_eq!(FP8_E4M3_LUT[0xfe], -448.0);
        assert_eq!(FP8_E4M3_LUT[0x08], 0.015625); // 2^-6 (giá trị chuẩn hóa nhỏ nhất)
        assert_eq!(FP8_E4M3_LUT[1], 1.0 / 512.0); // 2^-9 (giá trị phi chuẩn nhỏ nhất)

        // Kiểm tra roundtrip cho toàn bộ 254 byte hợp lệ (ngoại trừ NaN codes 0x7F và 0xFF)
        for b in 0..256u16 {
            let byte = b as u8;
            if byte == 0x7f || byte == 0xff {
                continue;
            }
            let val = FP8_E4M3_LUT[byte as usize];
            let enc = fp8_e4m3_encode(val);
            assert_eq!(
                enc, byte,
                "Lỗi roundtrip mã hóa tại byte 0x{:02x} (val={}) -> 0x{:02x}",
                byte, val, enc
            );
        }
    }

    #[test]
    fn test_fp8_row_quantize_dequantize_parity() {
        let original = vec![0.5f32, -1.2, 3.4, -0.05, 12.0, -8.5, 0.0, 15.0];
        let mut fp8_buf = vec![0u8; original.len()];
        let scale = quantize_row_fp8(&original, &mut fp8_buf);

        let mut reconstructed = vec![0.0f32; original.len()];
        dequantize_row_fp8(&fp8_buf, scale, &mut reconstructed);

        for (i, (&orig, &rec)) in original.iter().zip(&reconstructed).enumerate() {
            let error = (orig - rec).abs();
            // Sai số lượng tử hóa tương đối của FP8 E4M3 <= 6%
            assert!(
                error <= (orig.abs() * 0.08).max(0.05),
                "Sai số lượng tử hóa quá lớn tại index {}: orig={}, rec={}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_dot_fp8_f32_parity() {
        let q = vec![1.0f32, 2.0, -0.5, 0.8];
        let k = vec![2.5f32, -1.0, 4.0, -2.0];
        let mut k_fp8 = vec![0u8; k.len()];
        let scale = quantize_row_fp8(&k, &mut k_fp8);

        let dot_fp8 = dot_fp8_f32(&q, &k_fp8, scale);
        let dot_ref: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();

        let diff = (dot_fp8 - dot_ref).abs();
        assert!(
            diff < 0.25,
            "Tích vô hướng FP8 sai lệch: dot_fp8={}, dot_ref={}",
            dot_fp8,
            dot_ref
        );
    }
}
