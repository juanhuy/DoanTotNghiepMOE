# Module 04 — Toán và số thực cho AI

## Mục tiêu

Nắm đủ đại số tuyến tính để đọc công thức Transformer, hiểu softmax và cách tính nó không bị tràn số, và hiểu máy tính biểu diễn số thực ra sao (FP32, BF16, FP16, INT8) cùng các sai số đi kèm.

**Dùng khi đi làm:** mọi vị trí ML/AI engineer, data scientist, đồ họa máy tính, mô phỏng khoa học, tài chính định lượng. Hiểu số thực giúp tránh các lỗi rất khó tìm: kết quả khác nhau giữa CPU và GPU, NaN xuất hiện sau vài giờ huấn luyện, tổng tiền lệch một xu.

## 1. Vector, ma trận và phép nhân

- Vector `x ∈ ℝ^d`: một dãy d số. Trong Transformer, mỗi token được biểu diễn bằng một vector `d = 2048` số.
- Ma trận `W ∈ ℝ^{m×d}`: bảng m hàng, d cột.
- **Nhân ma trận với vector (matvec):** `y = W x`, với `y_i = Σ_j W_ij · x_j`. Mỗi phần tử đầu ra là **tích vô hướng** giữa một hàng của W và x.

Chi phí của matvec `m×d`: `m·d` phép nhân–cộng, và phải đọc `m·d` phần tử của W. Với một expert OLMoE (3 ma trận `1024×2048`), một lần tính tốn khoảng 6,3 triệu phép nhân–cộng và đọc 6 MiB trọng số.

- **Nhân ma trận với ma trận (matmul/GEMM):** `Y = W X`, với X gồm n vector cột. Mỗi phần tử của W được dùng n lần. Đây là lý do xử lý nhiều token cùng lúc (prefill, batch) hiệu quả hơn từng token một: cùng một lần đọc trọng số phục vụ được nhiều token.

**Quy ước bố cục:** ma trận lưu theo hàng (row-major): hàng 0, rồi hàng 1, ... Trọng số expert trong đồ án lưu theo hàng nên mỗi tích vô hướng đọc một đoạn nhớ liên tục. Đây là bố cục thân thiện nhất với cache CPU (module 08).

## 2. Các hàm kích hoạt

| Hàm | Công thức | Dùng ở |
| --- | --- | --- |
| ReLU | `max(0, z)` | mạng cổ điển |
| Sigmoid | `σ(z) = 1/(1+e^{-z})` | cổng, phân loại nhị phân |
| SiLU (Swish) | `z · σ(z)` | expert của OLMoE, Qwen3 |
| Softmax | `e^{z_i} / Σ_j e^{z_j}` | attention, router, phân phối token |

SiLU mượt, không bị "chết" ở vùng âm như ReLU, và được dùng trong khối SwiGLU (module 06).

## 3. Softmax và cách tính ổn định

Softmax biến một vector điểm bất kỳ (logits) thành phân phối xác suất: mọi phần tử dương, tổng bằng 1.

**Vấn đề:** `e^{100}` đã vượt giới hạn của `f32` (khoảng `3,4·10^{38}`), cho ra `inf`, và `inf/inf = NaN`.

**Cách sửa:** trừ giá trị lớn nhất trước khi lấy mũ. Kết quả không đổi vì:

```
e^{z_i - M} / Σ_j e^{z_j - M}  =  (e^{z_i}·e^{-M}) / (e^{-M}·Σ_j e^{z_j})  =  e^{z_i} / Σ_j e^{z_j}
```

Sau khi trừ, số mũ lớn nhất bằng `e^0 = 1`, không bao giờ tràn. Code trong `src/backend.rs`:

```rust
pub(crate) fn softmax(values: &mut [f32]) -> io::Result<()> {
    finite(values)?;                                               // từ chối NaN/inf ngay đầu vào
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = values.iter_mut().map(|x| { *x = (*x - max).exp(); *x }).sum();
    for x in values { *x /= sum; }
    Ok(())
}
```

### Log-sum-exp

Nhiều chỗ cần `log Σ e^{z_j}` (ví dụ tính log-xác suất, cross-entropy, perplexity). Công thức ổn định:

```
LSE(z) = M + log Σ_j e^{z_j - M},   M = max_j z_j
log softmax(z)_i = z_i - LSE(z)
```

Công cụ so logits `tools/compare_logits.py` dùng đúng công thức này trong hàm `log_softmax` để tính KL divergence mà không tràn số.

### Softmax trực tuyến (online softmax)

Khi các điểm đến lần lượt (attention duyệt qua từng vị trí trong KV cache), có thể tính softmax trong **một lượt** mà không lưu tất cả điểm. Ta giữ giá trị lớn nhất hiện tại `M`, tổng chuẩn hóa `S`, và tổng có trọng số `O`. Khi gặp điểm mới `s` kèm vector giá trị `v`:

```
nếu s > M:   O ← O·e^{M-s} + v;   S ← S·e^{M-s} + 1;   M ← s
ngược lại:   w = e^{s-M};   O ← O + w·v;   S ← S + w
kết quả cuối: O / S
```

Mỗi khi gặp giá trị lớn hơn, ta co lại những gì đã cộng theo hệ số `e^{M_cũ - M_mới}`. `src/backend.rs::causal_attention_into` cài đúng thuật toán này. Đây cũng là ý tưởng cốt lõi của **FlashAttention**, kỹ thuật giúp attention trên GPU nhanh hơn nhiều lần nhờ không phải ghi ma trận điểm `T×T` ra bộ nhớ.

## 4. Biểu diễn số thực IEEE 754

Một số thực dấu phẩy động gồm ba phần: **dấu** (sign), **mũ** (exponent), **phần định trị** (mantissa):

```
giá trị = (-1)^sign × 1.mantissa × 2^(exponent - bias)
```

| Định dạng | Tổng bit | Mũ | Định trị | Phạm vi lớn nhất | Độ chính xác (số chữ số thập phân) |
| --- | --- | --- | --- | --- | --- |
| FP32 | 32 | 8 | 23 | ~3,4·10³⁸ | ~7 |
| FP16 | 16 | 5 | 10 | 65 504 | ~3,3 |
| BF16 | 16 | 8 | 7 | ~3,4·10³⁸ | ~2,4 |
| FP8 E4M3 | 8 | 4 | 3 | 448 | ~1 |

Nhận xét quan trọng:

- **BF16** giữ nguyên 8 bit mũ của FP32, nên có cùng phạm vi, chỉ kém độ chính xác. Đổi FP32 sang BF16 chỉ cần bỏ 16 bit thấp. Vì vậy BF16 là định dạng chuẩn để huấn luyện và lưu mô hình ngôn ngữ: không lo tràn số như FP16.
- **FP16** chính xác hơn BF16 nhưng phạm vi nhỏ (tối đa 65 504), dễ tràn khi activation lớn.

**Trong đồ án:** `src/dense.rs` lưu dense của Qwen3 ở BF16 và đổi sang FP32 khi tính:

```rust
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)    // đặt 16 bit BF16 vào nửa trên của FP32
}
```

Một dòng, không mất thông tin. Chiều ngược lại (`f32_to_bf16`) cần làm tròn đúng: làm tròn về số gần nhất, khi đúng giữa thì về số chẵn (round-to-nearest-even), không phải cắt bỏ.

### Các giá trị đặc biệt

- `inf`, `-inf`: tràn số hoặc chia cho 0.
- `NaN` (not a number): `0/0`, `inf - inf`, `sqrt(-1)`. NaN lan truyền: mọi phép tính có NaN đều ra NaN, và `NaN == NaN` là `false`.

Engine kiểm tra `finite(...)` ở đầu vào và đầu ra của nhiều phép tính (`src/backend.rs::finite`). Khi có lỗi số, lỗi được bắt ngay tại chỗ xuất hiện, không lan qua 48 lớp rồi mới thành câu trả lời rác.

## 5. Sai số làm tròn và thứ tự cộng

Mỗi phép tính số thực làm tròn kết quả về số biểu diễn được gần nhất. Hệ quả:

- **Cộng không kết hợp:** `(a + b) + c` có thể khác `a + (b + c)`. Ví dụ với FP32: `(1e8 + 1) - 1e8 = 0` nhưng `1e8 + (1 - 1e8)` cũng có thể ra kết quả khác.
- **Cộng số nhỏ vào số lớn bị mất:** `1e8f32 + 1.0 == 1e8f32` vì FP32 chỉ có khoảng 7 chữ số.
- **Sai số tích lũy:** cộng n số theo thứ tự tuần tự có sai số tăng theo n. Cộng theo cây (pairwise) hoặc nhiều bộ cộng song song giảm sai số.

Hệ quả thực tế, rất quan trọng khi đọc kết quả đồ án:

- Hai cách cài cùng một thuật toán gần như không bao giờ cho kết quả **bằng nhau từng bit**, vì thứ tự cộng khác nhau (SIMD 8 làn, chia khối, GPU cộng theo cây).
- Vì vậy kiểm tra tính đúng phải dùng **dung sai** (tolerance), ví dụ `|a - b| < 1e-4`, chứ không dùng `==`.
- Engine so với Transformers lệch tối đa `3·10⁻⁴` trên logits của Qwen3. Đây là sai số làm tròn tích lũy qua 48 lớp, không phải lỗi (module 12).
- Ngược lại, **cùng một binary, cùng đầu vào, cùng thứ tự** thì kết quả phải giống hệt từng bit. Engine được kiểm điều này: build lại và chạy lại cho logits giống từng byte. Đó là tính tất định (determinism).

Engine cũng giữ thứ tự cộng cố định khi tổng hợp đầu ra của 8 expert: dù các expert tính xong theo thứ tự nào, bước cộng cuối luôn theo thứ tự top-k (`src/olmoe.rs`, xem comment về thứ tự top-k).

## 6. Thống kê cần cho đọc kết quả

- **Trung vị (median):** giá trị đứng giữa sau khi sắp xếp. Ít bị ảnh hưởng bởi một lần đo quá chậm, nên phù hợp hơn trung bình khi đo thời gian.
- **Tỉ số (ratio):** `t_B / t_A`. Dưới 1 nghĩa là B nhanh hơn. Với dữ liệu ghép cặp, xem khoảng tỉ số của từng cặp thay vì tỉ số của hai trung bình.
- **Phần trăm thay đổi:** `(t_A - t_B)/t_A`. Nhanh hơn 2 lần nghĩa là giảm 50% thời gian, không phải 200%.
- **Phân phối xác suất và entropy:** với phân phối p, entropy `H(p) = -Σ p_i log p_i` đo độ "không chắc chắn".
- **KL divergence:** `D_KL(p‖q) = Σ p_i log(p_i/q_i)` đo q lệch khỏi p bao nhiêu. Bằng 0 khi hai phân phối giống hệt, luôn ≥ 0, không đối xứng. Đồ án dùng KL để đo engine lệch khỏi Transformers và INT8 lệch khỏi BF16.

---

## Bài tập

Dùng Python với numpy cho các bài tập này.

### Bài 1 ★ — Đếm chi phí

Với OLMoE (`d = 2048`, `m = 1024`, `k = 8`, `L = 16`), tính cho **một token**:

1. Số phép nhân–cộng của phần expert (k expert × L lớp, mỗi expert 3 ma trận).
2. Số byte trọng số expert INT8 phải đọc.
3. Nếu RAM cung cấp 20 GB/s, thời gian tối thiểu chỉ để đọc trọng số expert là bao nhiêu? Suy ra tốc độ sinh tối đa (token/s) khi mọi expert đều đã nằm trong RAM.

<details><summary>Lời giải</summary>

1. Mỗi expert: `3 × 2048 × 1024 ≈ 6,29·10⁶`. Nhân `8 × 16 = 128` expert: khoảng `8,05·10⁸` phép nhân–cộng.
2. Khoảng `128 × 6,3 MB ≈ 807 MB` (770 MiB, như bảng dung lượng trong báo cáo).
3. `0,807 GB / 20 GB/s ≈ 40 ms`, tức tối đa khoảng 25 token/s chỉ tính riêng phần expert. Trên i5 engine đo được khoảng 8–10 tok/s ở chế độ GPU, phần còn lại là attention, đọc tệp khi cache miss, v.v. Phép tính kiểu này (cận trên từ băng thông) là kỹ năng rất có giá trị: biết ngay trần tốc độ trước khi tối ưu.
</details>

### Bài 2 ★ — Softmax tràn số

Tính softmax của `[1000, 1001, 1002]` bằng công thức ngây thơ và bằng công thức trừ max trong numpy với `float32`. Giải thích kết quả.

<details><summary>Lời giải</summary>

```python
import numpy as np
z = np.array([1000, 1001, 1002], dtype=np.float32)
naive = np.exp(z) / np.exp(z).sum()          # [nan nan nan], kèm cảnh báo overflow
stable = np.exp(z - z.max()); stable /= stable.sum()
print(naive, stable)                          # stable ≈ [0.090 0.245 0.665]
```

`exp(1000)` tràn thành inf, `inf/inf` là NaN. Trừ max đưa về `[-2, -1, 0]`, kết quả đúng.
</details>

### Bài 3 ★★ — Online softmax

Viết hàm `attention_online(q, K, V)` (q là vector d, K và V là ma trận T×d) tính `softmax(K q / √d)ᵀ V` bằng thuật toán một lượt ở mục 3, chỉ duyệt K, V một lần, không lưu mảng điểm. So với bản tính trực tiếp, sai số dưới 1e-5 với T = 1000.

<details><summary>Lời giải</summary>

```python
def attention_online(q, K, V):
    d = q.shape[0]
    M, S, O = -np.inf, 0.0, np.zeros(V.shape[1])
    for k, v in zip(K, V):
        s = k @ q / np.sqrt(d)
        if s > M:
            r = np.exp(M - s)
            O = O * r + v; S = S * r + 1.0; M = s
        else:
            w = np.exp(s - M); O += w * v; S += w
    return O / S

rng = np.random.default_rng(0)
q, K, V = rng.normal(size=64), rng.normal(size=(1000, 64)), rng.normal(size=(1000, 64))
s = K @ q / 8.0; p = np.exp(s - s.max()); p /= p.sum()
print(np.abs(attention_online(q, K, V) - p @ V).max())   # khoảng 1e-15 với float64
```

So với `causal_attention_into` trong `src/backend.rs`: cùng thuật toán, viết cho nhiều đầu và có kiểm tra số hữu hạn.
</details>

### Bài 4 ★★ — BF16 bằng tay

1. Viết `f32_to_bf16(x)` bằng cách cắt bỏ 16 bit thấp, và một bản làm tròn round-to-nearest-even. Gợi ý: `np.float32(x).view(np.uint32)`.
2. Với 1 triệu số ngẫu nhiên phân phối chuẩn, đo sai số tương đối lớn nhất và trung bình của hai cách.
3. Biểu diễn số 1/3 ở FP32, BF16, FP16. Sai lệch bao nhiêu?

<details><summary>Lời giải</summary>

```python
def bf16_trunc(x):
    b = np.asarray(x, np.float32).view(np.uint32)
    return (b & 0xFFFF0000).view(np.float32)

def bf16_rne(x):
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    lsb = (b >> 16) & 1
    b = (b + 0x7FFF + lsb) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)

x = np.random.default_rng(0).normal(size=1_000_000).astype(np.float32)
for f in (bf16_trunc, bf16_rne):
    rel = np.abs(f(x) - x) / np.abs(x)
    print(f.__name__, rel.max(), rel.mean())
```

Cắt bỏ cho sai số tương đối tối đa khoảng `2^-7 ≈ 0,78%` và luôn lệch về phía 0 (có thiên lệch). Làm tròn cho tối đa khoảng `2^-8 ≈ 0,39%` và không thiên lệch. Đối chiếu với hàm `f32_to_bf16` trong `src/dense.rs`.
</details>

### Bài 5 ★★ — Thứ tự cộng

Tạo 10 triệu số `float32` ngẫu nhiên trong `[0, 1)`. Tính tổng bằng: (a) vòng lặp tuần tự bằng `float32`, (b) `np.sum` (dùng cộng theo cây), (c) `float64`. So sánh với (c). Vì sao (a) sai nhiều nhất?

<details><summary>Gợi ý</summary>

Viết (a) bằng vòng lặp Python với `acc = np.float32(0)`, hoặc nhanh hơn với `np.cumsum(x, dtype=np.float32)[-1]`. Khi tổng lên tới hàng triệu, mỗi số mới (nhỏ hơn 1) cộng vào bị làm tròn mất nhiều chữ số. Cộng theo cây giữ các tổng con ở cùng cỡ độ lớn nên sai số nhỏ hơn nhiều.
</details>

### Bài 6 ★★ — KL divergence

Viết `kl(p_logits, q_logits)` tính `D_KL(softmax(p) ‖ softmax(q))` bằng log-softmax ổn định. Kiểm tra:

1. `kl(z, z) == 0`.
2. `kl(z, z + 5) == 0` (cộng hằng số vào mọi logit không đổi phân phối).
3. Với `z` ngẫu nhiên 50 000 phần tử, thêm nhiễu `N(0, 0.01)` thì KL khoảng bao nhiêu?

Đối chiếu với `tools/compare_logits.py`.

<details><summary>Lời giải</summary>

```python
def log_softmax(x):
    m = x.max(); return x - m - np.log(np.exp(x - m).sum())

def kl(p, q):
    lp, lq = log_softmax(p), log_softmax(q)
    return float((np.exp(lp) * (lp - lq)).sum())
```

Ý 2 bằng 0 vì softmax bất biến với phép cộng hằng số. Đây cũng là lý do so logits trực tiếp có thể đánh lừa: hai bộ logits khác nhau một hằng số trông lệch rất lớn nhưng cho cùng phân phối. KL thì không bị đánh lừa. Với nhiễu 0,01, KL thường cỡ `10⁻⁵`. Engine so với Transformers có KL cỡ `10⁻¹¹`, nhỏ hơn rất nhiều.
</details>

## Tài liệu đọc thêm

- Gilbert Strang, *Introduction to Linear Algebra* (MIT OCW 18.06).
- David Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic* (1991).
- Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018).
- Tri Dao et al., *FlashAttention* (2022).
