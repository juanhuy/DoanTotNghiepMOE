# Kế hoạch Học tập 1 Tháng: Từ Zero → MoE Inference Expert

> **Mục tiêu kép:** Nắm chắc đồ án để bảo vệ xuất sắc + tích lũy kiến thức AI nền tảng để đi làm ngay sau tốt nghiệp.

---

## Nguyên tắc học tập

- **Học gắn với code thực**: Mọi lý thuyết đều soi chiếu vào code dự án thực tế.
- **Build mental model trước**: Hiểu tại sao, rồi mới đến công thức, rồi mới đến code.
- **Active recall**: Cuối mỗi ngày, đóng tài liệu và giải thích lại bằng ngôn ngữ của mình.

---

## TUẦN 1: Nền tảng — Neural Network & Ngôn ngữ

### Ngày 1–2: Mạng nơ-ron là gì? (Neural Network 101)

**Khái niệm cốt lõi cần hiểu:**
- Neuron = hàm: nhận số, tính tổng có trọng số, rồi qua hàm kích hoạt
- `y = activation(W·x + b)` — đây là mọi thứ
- Tại sao cần nhiều lớp (Deep Learning)?
- Backpropagation = "đổ lỗi ngược về" để điều chỉnh trọng số

**Tài nguyên học:**
- 📺 [3Blue1Brown — Neural Networks Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — cực kỳ trực quan (4 video, ~1h tổng)
- Bài tập: Mở Python và tính `y = relu(np.dot(W, x) + b)` với mảng ngẫu nhiên

---

### Ngày 3–4: Language Model là gì?

**Câu hỏi cốt lõi:** Làm sao máy tính "đọc hiểu" văn bản?

**Khái niệm cần hiểu:**
- Tokenization: Cắt câu thành các mảnh (token)
- Token ID → Embedding vector (số hóa ý nghĩa của từ)
- Language Model = máy đoán từ tiếp theo
- Autoregressive generation: Sinh từng token một theo thứ tự

**Thực hành:**
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
tokens = tokenizer("Xin chào Việt Nam")
print(tokens)  # Xem token IDs
```

**Tại sao quan trọng cho đồ án:** Engine của bạn nhận token IDs từ tokenizer, đây là điểm khởi đầu của mọi request trong `src/olmoe.rs`.

---

### Ngày 5–7: Ma trận & Phép toán Tuyến tính (Ôn lại)

**Tại sao cần:** 90% tính toán AI là phép nhân ma trận-vector. Hiểu cái này = hiểu được bottleneck của toàn bộ engine.

**Khái niệm cần ôn:**
- Ma trận nhân vector: `y = W·x` (shape: `[m,n] x [n] = [m]`)
- Dot product = "đo độ giống nhau"
- Softmax: chuyển điểm số → xác suất (tổng = 1)

**Thực hành trong Python:**
```python
import numpy as np

W = np.random.randn(4, 8)   # Ma trận trọng số [4x8]
x = np.random.randn(8)       # Vector đầu vào [8]
y = W @ x                    # Phép nhân: kết quả [4]

def softmax(z):
    e = np.exp(z - z.max())  # Ổn định số (numerical stability)
    return e / e.sum()

probs = softmax(y)
print(probs.sum())  # Luôn = 1.0
```

**Kết nối với đồ án:** `src/backend.rs` là nơi triển khai phép nhân này ở Rust với tối ưu 8 bộ cộng.

---

## TUẦN 2: Kiến trúc Transformer

### Ngày 8–9: Cơ chế Attention — "Từ nào liên quan đến từ nào?"

**Mental model trước khi học công thức:**

Hãy tưởng tượng câu: *"Con mèo ngồi trên chiếc ghế vì nó mệt."*
Khi dịch "nó" sang tiếng Anh, bạn cần biết "nó" = "con mèo" hay "chiếc ghế"?
Attention giải quyết bài toán này.

**Cơ chế Q-K-V:**
```
Mỗi từ được chiếu thành 3 vai trò:
  Q (Query)  = "Mình đang tìm kiếm thứ gì?"
  K (Key)    = "Mình đang quảng cáo thứ gì?"
  V (Value)  = "Nội dung thực sự của mình là gì?"

Attention(Q,K,V) = softmax(Q·Kᵀ / sqrt(d)) · V
```

**Tài nguyên bắt buộc:**
- 📖 [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

**Kết nối đồ án:** `src/attention.rs` — đây là nơi Q·Kᵀ được tính.

---

### Ngày 10–11: KV Cache — Tại sao decode nhanh hơn prefill?

**Vấn đề:** Khi sinh token thứ 100, nếu tính lại K và V của 99 token trước → Lãng phí khổng lồ!

**Giải pháp KV Cache:**
```
Step 1: Token "Xin" → K₁, V₁ (lưu vào cache)
Step 2: Token "chào" → K₂, V₂ + dùng lại K₁,V₁
Step 3: Token "Việt" → K₃, V₃ + dùng lại K₁,K₂,V₁,V₂
```

**Hai pha quan trọng:**
- **Prefill**: Xử lý toàn bộ prompt, điền đầy KV Cache (compute-bound)
- **Decode**: Sinh từng token, thêm 1 hàng vào cache mỗi bước (memory-bound)

**Kết nối đồ án:** `struct OlmoeState` trong `src/olmoe.rs` chứa KV Cache.

---

### Ngày 12–14: RMSNorm, RoPE, và Residual Connection

**1. Residual Connection:**
```
output = sublayer(x) + x
```
Gradient không bị "chết" khi truyền qua 16 layers.

**2. RMSNorm:**
```
RMSNorm(x) = x / sqrt(mean(x²)) * γ
```
Giữ độ lớn activation ổn định. Không có nó, số sẽ bùng nổ hoặc tiêu biến.

**3. RoPE:** Xoay vector Q và K theo góc tỉ lệ với vị trí từ. Dot product tự động phụ thuộc khoảng cách tương đối giữa hai từ.

**Thực hành:** Tìm `rmsnorm` và `apply_rope` trong `src/attention.rs`. Giải thích mỗi dòng bằng lời.

---

## TUẦN 3: Mixture of Experts (MoE) — Trái tim của Đồ án

### Ngày 15–16: Tại sao cần MoE?

```
Dense model: Mỗi token → 1 MLP khổng lồ (175B params)
MoE model:   Mỗi token → 8 MLP nhỏ chọn từ 64 MLPs

Kết quả:
  Total params:  7B  ← kiến thức phong phú
  Active params: 1B  ← chi phí tính toán thấp
```

**OLMoE cụ thể:**
- 16 layers, mỗi layer: 64 experts, mỗi token chọn **Top-8**
- Mỗi expert ≈ 6.3 MB (INT8)
- Tổng: 64 × 6.3 MB × 16 layers ≈ **6.4 GB** → không vừa RAM!

→ Đây là lý do cần **Tiered Memory**.

---

### Ngày 17–18: Router / Gating — Cơ chế chọn chuyên gia

```python
def router(hidden_state, W_gate, top_k=8):
    scores = hidden_state @ W_gate.T  # [64] điểm
    probs = softmax(scores)            # [64] xác suất
    top_indices = argsort(probs)[-8:] # 8 experts tốt nhất
    weights = probs[top_indices] / probs[top_indices].sum()  # chuẩn hóa
    return top_indices, weights

output = sum(weights[i] * expert[idx](hidden_state)
             for i, idx in enumerate(top_indices))
```

**Kết nối code:** `src/router.rs` — xem hàm `route()`.

---

### Ngày 19–20: SwiGLU Expert — Bên trong mỗi chuyên gia

```
Expert(x) = W_down · (SiLU(W_gate·x) ⊙ W_up·x)
```

- `SiLU(z) = z * sigmoid(z)`: hàm kích hoạt mượt hơn ReLU
- `⊙`: nhân element-wise — "cổng" kiểm soát thông tin đi qua

**Tại sao INT8?**
```
FP32: 4 bytes/số → Expert ~25 MB
INT8: 1 byte/số  → Expert ~6.3 MB (giảm 4x!)
```

Với row-wise scaling factor, sai số logits < 2e-5. Chất lượng không đổi.

**Kết nối code:** `src/int8_expert.rs` — đọc `forward_reuse()`.

---

### Ngày 21: Prefill Batching — Gom token theo expert

```
Token-major (cũ — chậm):
  Token 1: Layer1 → Layer2 → ... → Layer16 (nạp expert nhiều lần)
  Token 2: Lại nạp expert từ đầu!

Layer-major (mới — nhanh):
  Layer 1: Xử lý 136 tokens cùng lúc
            Gom token theo expert → nạp expert 1 lần cho nhiều tokens!
  Layer 2: Tiếp tục...
```

**Lợi ích:** Expert cache hit tăng, SSD read giảm mạnh.

---

## TUẦN 4: Hệ thống & Thực nghiệm

### Ngày 22–23: Tiered Memory Architecture

```
RAM (~2GB cache):
  ├─ Dense weights: embeddings, attention, norms, LM head (LUÔN Ở ĐÂY)
  └─ Expert Cache: ~21 experts/layer đang "nóng"

SSD NVMe (6.4GB):
  └─ 64 experts × 16 layers (Safetensors format)
     → Đọc theo offset khi cache miss
```

**Thuật toán Heat Cache:**
```
Khi expert được truy cập: heat[id] += 1
Khi cache đầy: loại expert có heat thấp nhất (victim)
Định kỳ: heat = heat / 2  (aging — thích nghi với ngữ cảnh mới)
```

**Kết quả benchmark thực của đồ án:**

| Cache Policy    | Cache Hit | SSD Read | Tổng thời gian |
|-----------------|-----------|----------|----------------|
| LRU 32MB/layer  | 242       | 18.6 GB  | 29.8s          |
| LRU 128MB/layer | 1,951     | 7.9 GB   | 18.2s          |
| Heat 128MB/layer| 2,553     | 4.1 GB   | 16.8s          |

---

### Ngày 24–25: Tối ưu CPU Kernel

**Tại sao 8 bộ cộng nhanh hơn?**

```rust
// CHẬM: dependency tuần tự, CPU phải đợi
let mut sum = 0.0f32;
for i in 0..N { sum += a[i] * b[i]; }

// NHANH: 8 luồng độc lập, CPU thực thi song song
let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
let (mut s4, mut s5, mut s6, mut s7) = (0.0, 0.0, 0.0, 0.0);
for i in (0..N).step_by(8) {
    s0 += a[i]*b[i]; s1 += a[i+1]*b[i+1]; // ... đến s7
}
let sum = s0+s1+s2+s3+s4+s5+s6+s7;
```

**Kết quả thực tế:** Expert compute giảm 71% (9.3s → 2.6s).

---

### Ngày 26–27: SSE Streaming & API

**Tại sao cần Streaming?** Thay vì đợi cả câu trả lời (30-60s), người dùng thấy từng từ hiện ra ngay.

```
HTTP Request
  → Axum Router
  → spawn_blocking() ← tính toán nặng, không block async runtime
  → Tokenizer → Prefill → Decode Loop
  → SSE Channel (queue 8 events, backpressure 30s timeout)
  → HTTP text/event-stream
```

---

### Ngày 28: Chạy Benchmark & Đọc Kết quả

```bash
cargo run --release --example benchmark_suite models/olmoe-1b-7b-int8 3
```

**Các chỉ số cần hiểu:**

| Chỉ số         | Ý nghĩa                                    |
|----------------|--------------------------------------------|
| TTFT           | Thời gian đến token đầu (trải nghiệm UX)   |
| Decode token/s | Tốc độ sinh token (throughput)             |
| Cache hit/miss | Hiệu quả chiến lược cache                  |
| Expert bytes   | Áp lực đọc SSD                             |
| Peak RSS       | RAM tối đa sử dụng (đảm bảo không OOM)    |

---

### Ngày 29–30: Báo cáo & Bảo vệ

**Cấu trúc báo cáo đề xuất:**
```
Chương 1: Giới thiệu
  Bài toán: Chạy LLM 7B trên CPU không GPU
  Đóng góp: Engine Rust + Tiered Memory + Heat Cache

Chương 2: Cơ sở lý thuyết
  Transformer & MoE Architecture
  Weight-only INT8 Quantization
  Tiered Memory Systems

Chương 3: Thiết kế & Triển khai
  Kiến trúc tổng thể
  Safetensors INT8 Checkpoint Format
  Heat Cache Algorithm
  Layer-major Prefill Batching
  8-Accumulator + Rayon Kernel
  SSE Streaming API

Chương 4: Thực nghiệm & Đánh giá
  Môi trường: i7-14650HX, 16GB RAM, SSD NVMe
  Benchmark: TTFT, decode token/s, cache hit/miss
  So sánh cache policies
  Phân tích bottleneck (Prefill vs Decode)

Chương 5: Kết luận & Hướng phát triển
```

**3 câu hội đồng hay hỏi:**

1. *Tại sao chọn MoE thay vì Dense?* → MoE có 7B total params nhưng chỉ activate 1B params/token. Kiến thức phong phú, tính toán rẻ, phù hợp offload.

2. *Bottleneck lớn nhất ở đâu?* → I/O bound (SSD read expert) và memory-bandwidth bound (matvec dense). Giải quyết bằng Heat Cache + Layer-major Prefill + 8-accumulator kernel.

3. *INT8 có làm giảm chất lượng không?* → Row-wise scaling factor bảo toàn sai số logits < 2e-5. Token output giống hệt bản FP32 gốc.

---

## Sau tốt nghiệp: Ứng dụng kiến thức đi làm

### Vị trí phù hợp:

| Vị trí                     | Skill chính từ đồ án                           |
|----------------------------|------------------------------------------------|
| AI/ML Engineer             | LLM inference, quantization, model optimization|
| LLM Inference Engineer     | Tiered memory, CPU opt, serving pipeline       |
| Systems Engineer (Rust)    | Zero-allocation, SIMD, async systems           |
| Backend Engineer (AI)      | SSE streaming, API design, concurrency         |

### Điểm nổi bật ghi CV:
```
• Implemented sparse MoE inference engine in Rust with tiered SSD/RAM offloading
• Designed heat-aware expert cache, reducing SSD reads by 57% (verified via benchmark)
• Optimized INT8 matmul with 8-accumulator ILP technique: 3.6× kernel speedup
• Built SSE streaming API with graceful cancellation (Tokio/Axum)
```

---

## Tài nguyên học theo thứ tự ưu tiên

1. 📺 **3Blue1Brown Neural Networks** (YouTube) — trực quan nhất
2. 📖 **The Illustrated Transformer** (jalammar.github.io) — bắt buộc
3. 📄 **OLMoE Paper** (arxiv:2409.02060) — hiểu model bạn đang chạy
4. 📖 **Rust Performance Book** (nnethercote.github.io) — lý do tối ưu code

---

## Checklist tự đánh giá cuối tháng

- [ ] Giải thích Q-K-V Attention bằng lời (không nhìn tài liệu)
- [ ] Vẽ sơ đồ Forward Pass qua 1 Layer MoE
- [ ] Giải thích tại sao KV Cache giúp decode nhanh hơn
- [ ] Mô tả vấn đề cache thrashing và cách Heat Cache giải quyết
- [ ] Đọc và giải thích `forward_reuse()` trong `src/int8_expert.rs`
- [ ] Chạy benchmark suite và đọc hiểu JSON output
- [ ] Trả lời: "Bottleneck của engine ở đâu? Tại sao?"
- [ ] Viết xong Chương 2 và Chương 3 của báo cáo
