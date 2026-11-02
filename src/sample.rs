//! Token sampling suite inspired by Colibrì's `sample.h`.
//! Provides temperature scaling, top-p (nucleus) filtering, top-k truncation,
//! repetition penalty, and fast zero-dependency XorShift64* PRNG.

use serde::{Deserialize, Serialize};

/// Cấu hình tham số lấy mẫu (Sampling Parameters).
/// Khi temperature <= 1e-4, hệ thống tự động rơi về greedy argmax để đảm bảo bit-exact parity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Nhiệt độ điều chỉnh độ ngẫu nhiên. Mặc định 0.0 (Greedy Argmax).
    #[serde(default)]
    pub temperature: f32,
    /// Ngưỡng cắt xác suất tích lũy Nucleus Top-P (0.0 < top_p <= 1.0). Mặc định 1.0 (tắt).
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Giới hạn số lượng ứng viên Top-K cao nhất. Mặc định 0 (tắt).
    #[serde(default)]
    pub top_k: usize,
    /// Hệ số phạt lặp lại (Repetition Penalty >= 1.0). 1.0 = không phạt.
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
}

fn default_top_p() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.0
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingParams {
    /// Đọc cấu hình sampling từ các biến môi trường (MOE_TEMPERATURE, MOE_TOP_P, MOE_TOP_K, MOE_REPETITION_PENALTY)
    pub fn from_env() -> Self {
        let temperature = std::env::var("MOE_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0);
        let top_p = std::env::var("MOE_TOP_P")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0);
        let top_k = std::env::var("MOE_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let repetition_penalty = std::env::var("MOE_REPETITION_PENALTY")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0);

        Self {
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-4 && (self.repetition_penalty - 1.0).abs() <= 1e-5
    }
}

/// Bộ sinh số ngẫu nhiên XorShift64* cực nhanh, không phụ thuộc ngoại vi,
/// tương tự macro `rndu()` trong `sample.h` của Colibrì.
pub struct XorShift64 {
    state: u64,
}

impl Default for XorShift64 {
    fn default() -> Self {
        Self::new(0x9E3779B97F4A7C15)
    }
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Trả về số thực phân bố đều trong khoảng [0.0, 1.0)
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16777216.0)
    }
}

/// Lấy mẫu token kế tiếp từ logits dựa trên SamplingParams và danh sách token đã sinh
pub fn sample_token(
    logits: &[f32],
    params: &SamplingParams,
    past_tokens: &[usize],
    rng: &mut XorShift64,
) -> usize {
    if logits.is_empty() {
        return 0;
    }

    // Nếu greedy hoàn toàn và không có phạt lặp, dùng argmax trực tiếp
    if params.is_greedy() {
        return argmax(logits);
    }

    let mut filtered_logits = logits.to_vec();

    // 1. Áp dụng Repetition Penalty (nếu > 1.0)
    if params.repetition_penalty > 1.0 {
        for &tok in past_tokens {
            if tok < filtered_logits.len() {
                let v = filtered_logits[tok];
                if v > 0.0 {
                    filtered_logits[tok] = v / params.repetition_penalty;
                } else {
                    filtered_logits[tok] = v * params.repetition_penalty;
                }
            }
        }
    }

    // Nếu sau phạt lặp vẫn là nhiệt độ 0, trả về argmax
    if params.temperature <= 1e-4 {
        return argmax(&filtered_logits);
    }

    // 2. Chia cho nhiệt độ (Temperature Scaling)
    let inv_temp = 1.0 / params.temperature;
    for x in &mut filtered_logits {
        *x *= inv_temp;
    }

    // 3. Tính Softmax ổn định (trừ max)
    let max_logit = filtered_logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    if !max_logit.is_finite() {
        // Fallback về argmax nếu gặp NaN/Inf (theo triết lý Colibrì sample.h)
        return argmax(logits);
    }

    let mut probs: Vec<(usize, f32)> = filtered_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let p = if v.is_finite() {
                (v - max_logit).exp()
            } else {
                0.0
            };
            (i, p)
        })
        .collect();

    let sum_p: f32 = probs.iter().map(|(_, p)| *p).sum();
    if sum_p <= 0.0 || !sum_p.is_finite() {
        return argmax(logits);
    }

    let inv_sum = 1.0 / sum_p;
    for (_, p) in &mut probs {
        *p *= inv_sum;
    }

    // 4. Sắp xếp giảm dần theo xác suất
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 5. Áp dụng Top-K (nếu có cấu hình)
    if params.top_k > 0 && params.top_k < probs.len() {
        probs.truncate(params.top_k);
    }

    // 6. Áp dụng Top-P (Nucleus cutoff)
    if params.top_p > 0.0 && params.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut cutoff_idx = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            cumulative += *p;
            if cumulative >= params.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        probs.truncate(cutoff_idx);
    }

    // 7. Chuẩn hóa lại phân phối sau khi cắt tỉa
    let new_sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if new_sum <= 0.0 {
        return probs.first().map(|&(idx, _)| idx).unwrap_or(0);
    }

    // 8. Bốc mẫu ngẫu nhiên từ phân phối tích lũy
    let r = rng.next_f32() * new_sum;
    let mut acc = 0.0f32;
    for &(idx, p) in &probs {
        acc += p;
        if acc >= r {
            return idx;
        }
    }

    probs.last().map(|&(idx, _)| idx).unwrap_or(0)
}

fn argmax(logits: &[f32]) -> usize {
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
mod tests {
    use super::*;

    #[test]
    fn test_greedy_fallback_matches_argmax() {
        let logits = vec![1.0, 5.0, 3.2, 0.1, -2.0];
        let params = SamplingParams::default(); // temperature = 0.0
        let mut rng = XorShift64::default();
        let tok = sample_token(&logits, &params, &[], &mut rng);
        assert_eq!(tok, 1); // 5.0 is highest at index 1
    }

    #[test]
    fn test_repetition_penalty_depresses_past_token() {
        let logits = vec![5.0, 4.8, 1.0]; // Token 0 is highest
        let past = vec![0]; // Token 0 was previously generated
        let params = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.5, // 5.0 / 1.5 = 3.333 < 4.8
        };
        let mut rng = XorShift64::default();
        let tok = sample_token(&logits, &params, &past, &mut rng);
        assert_eq!(tok, 1); // Token 1 becomes highest due to penalty on Token 0
    }

    #[test]
    fn test_top_k_filtering() {
        let logits = vec![10.0, 9.0, 1.0, 0.5];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 2, // Only consider tokens 0 and 1
            repetition_penalty: 1.0,
        };
        let mut rng = XorShift64::default();
        for _ in 0..50 {
            let tok = sample_token(&logits, &params, &[], &mut rng);
            assert!(tok == 0 || tok == 1);
        }
    }

    #[test]
    fn test_non_finite_logits_fallback_gracefully() {
        let logits = vec![f32::NAN, 2.0, f32::INFINITY, 1.0];
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 5,
            repetition_penalty: 1.0,
        };
        let mut rng = XorShift64::default();
        let tok = sample_token(&logits, &params, &[], &mut rng);
        // Does not panic or crash, returns a valid token index
        assert!(tok < logits.len());
    }
}
