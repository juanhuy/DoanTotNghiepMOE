# Module 05 — Transformer từ đầu

## Mục tiêu

Hiểu một mô hình ngôn ngữ biến câu hỏi thành câu trả lời như thế nào, từng bước một: tokenizer, embedding, attention, RoPE, RMSNorm, GQA, KV cache, prefill và decode, chọn token. Sau module này bạn tự viết được một Transformer nhỏ bằng numpy và đọc được hàm `step_inner` trong `src/olmoe.rs`.

**Dùng khi đi làm:** nền tảng của mọi công việc liên quan LLM: fine-tune, RAG, tối ưu inference, đánh giá mô hình, kể cả viết prompt tốt hơn (hiểu vì sao ngữ cảnh dài thì chậm và tốn bộ nhớ).

## 1. Bức tranh tổng thể

Mô hình ngôn ngữ tự hồi quy (autoregressive) chỉ làm một việc: **cho chuỗi token, dự đoán token tiếp theo**. Sinh câu trả lời là lặp việc đó:

```
"Where is Mars?"  ──tokenizer──►  [Where][ is][ Mars][?]  ──mô hình──►  điểm cho 50 304 token
                                                                          chọn "Mars"
[Where][ is][ Mars][?][Mars]  ──mô hình──►  chọn " is"
[Where][ is][ Mars][?][Mars][ is]  ──mô hình──►  chọn " the"
...  cho đến token kết thúc (EOS) hoặc đủ số token
```

Bên trong mô hình:

```
token ids ─► Embedding ─► [ Lớp 1 ] ─► [ Lớp 2 ] ─► ... ─► [ Lớp L ] ─► RMSNorm ─► LM head ─► logits
                             │
                 ┌───────────┴────────────────────────────────┐
                 │ h ─► RMSNorm ─► Attention ─► + h  (residual)│
                 │ h ─► RMSNorm ─► MoE / FFN ─► + h  (residual)│
                 └─────────────────────────────────────────────┘
```

## 2. Tokenizer

Mô hình không đọc chữ mà đọc số. **Tokenizer** cắt văn bản thành các mảnh (token) theo một từ vựng cố định và đổi mỗi mảnh thành một số nguyên.

- Phần lớn LLM dùng **BPE** (Byte Pair Encoding): bắt đầu từ từng byte, lặp lại việc gộp cặp xuất hiện nhiều nhất thành token mới cho đến khi đủ kích thước từ vựng.
- Từ thông dụng là một token (`" Mars"`), từ hiếm bị cắt thành nhiều mảnh. Tiếng Việt thường tốn nhiều token hơn tiếng Anh cho cùng nội dung, vì tokenizer được huấn luyện chủ yếu trên tiếng Anh.
- Token khác từ: `"Hello"` và `" Hello"` (có dấu cách) là hai token khác nhau.

**Khuôn mẫu hội thoại (chat template).** Mô hình Instruct được huấn luyện với định dạng hội thoại cố định, ví dụ:

```
<|im_start|>user
Where is Mars?<|im_end|>
<|im_start|>assistant
```

Khuôn mẫu được viết bằng ngôn ngữ Jinja và đi kèm checkpoint (`tokenizer_config.json`). Engine render nó bằng thư viện `minijinja` (`OlmoeModel::render_chat` trong `src/olmoe.rs`), rồi mã hóa bằng crate `tokenizers` của Hugging Face. Dùng sai khuôn mẫu thì mô hình vẫn chạy nhưng trả lời kém đi rõ rệt.

## 3. Embedding

Bảng embedding là ma trận `V × d` (OLMoE: `50 304 × 2048`). Token id `t` được biến thành vector bằng cách lấy hàng thứ `t`. Không có phép tính nào, chỉ là tra bảng.

**Trong đồ án:** với Qwen3, bảng này là `151 936 × 2048`, chiếm khoảng 1,24 GB ở FP32. Mỗi token chỉ cần một hàng, nên engine đọc đúng hàng đó từ tệp (`Embedding::copy_row` trong `src/olmoe.rs`) thay vì giữ cả bảng trong RAM. Tiết kiệm hơn 1 GB mà gần như không tốn thời gian, vì mỗi hàng chỉ 4 KB.

## 4. Attention

Attention cho phép mỗi token "nhìn" các token trước nó và lấy thông tin từ những token liên quan.

### 4.1 Query, key, value

Từ vector ẩn `h` của token, ba phép chiếu tạo ra:

- **query** `q = W_Q h`: "tôi đang tìm gì",
- **key** `k = W_K h`: "tôi có thông tin gì",
- **value** `v = W_V h`: "nội dung tôi đóng góp".

Với token ở vị trí t:

```
điểm_j  = q_t · k_j / √d_h            với mọi j ≤ t   (chỉ nhìn quá khứ: mặt nạ nhân quả)
trọng_số = softmax(điểm)
đầu ra   = Σ_j trọng_số_j · v_j
```

Chia cho `√d_h` để điểm không quá lớn khi chiều tăng; không có bước này, softmax quá "nhọn" và gradient khi huấn luyện bị triệt tiêu.

### 4.2 Nhiều đầu (multi-head)

Vector q, k, v được chia thành h phần, mỗi phần d_h chiều, gọi là một **đầu** (head). Mỗi đầu tính attention độc lập, có thể học một kiểu quan hệ khác nhau (cú pháp, đồng tham chiếu, vị trí...). Kết quả các đầu được nối lại rồi chiếu qua `W_O`.

OLMoE có 16 đầu × 128 chiều = 2048. Qwen3 có 32 đầu query × 128 = 4096 (lớn hơn chiều ẩn 2048; `W_Q` chiếu từ 2048 lên 4096, `W_O` chiếu ngược về).

### 4.3 Grouped-Query Attention (GQA)

Trong GQA, nhiều đầu query **dùng chung** một đầu key/value. Qwen3 có 32 đầu query nhưng chỉ 4 đầu KV: mỗi nhóm 8 đầu query dùng chung một cặp k, v.

Lợi ích: KV cache nhỏ đi 8 lần, đọc KV khi decode nhanh hơn, mà chất lượng giảm rất ít. Trong code (`src/backend.rs::causal_attention_into`):

```rust
let group = num_attention_heads / num_key_value_heads;   // Qwen3: 32 / 4 = 8
let key_start = (head / group) * head_width;             // đầu query `head` dùng đầu KV thứ head/group
```

Hai dòng này là toàn bộ sự khác biệt giữa MHA (multi-head) và GQA. Với OLMoE, `group = 1`: mỗi đầu query có KV riêng.

## 5. Mã hóa vị trí bằng RoPE

Attention bản thân không biết thứ tự: đảo các token thì tập điểm không đổi. Cần đưa thông tin vị trí vào.

**RoPE** (Rotary Position Embedding) xoay từng cặp tọa độ của q và k một góc tỉ lệ với vị trí:

```
θ_i = vị_trí / base^(2i / d_h)
[a', b'] = [a·cos θ_i − b·sin θ_i,  a·sin θ_i + b·cos θ_i]
```

Tính chất quan trọng: tích vô hướng `q_t · k_j` sau khi xoay chỉ phụ thuộc vào **khoảng cách** `t − j`, không phụ thuộc vị trí tuyệt đối. `base` là `rope_theta` trong cấu hình (Qwen3 dùng 1 000 000, cho phép ngữ cảnh dài).

Có hai quy ước ghép cặp, và dùng sai là mô hình cho kết quả rác:

- **Cặp kề nhau** `(x₀, x₁), (x₂, x₃), ...` — mô hình nhỏ dùng để kiểm thử trong `src/attention.rs::rope`.
- **Nửa đầu với nửa sau** `(x₀, x_{64}), (x₁, x_{65}), ...` (rotate-half) — quy ước của Hugging Face cho OLMoE và Qwen3, cài trong `src/olmoe.rs::apply_rope_cached`.

Engine tính sẵn bảng `sin`, `cos` cho mỗi vị trí một lần (`prepare_rope_sincos`) rồi dùng cho mọi đầu và mọi lớp. Test `test_rope_fast_path_matches_legacy_rotate` kiểm bản nhanh khớp với bản tính thẳng.

## 6. RMSNorm và QK-norm

**RMSNorm** chuẩn hóa độ lớn của vector trước mỗi khối:

```
RMSNorm(x)_i = γ_i · x_i / sqrt(mean(x²) + ε)
```

Đơn giản hơn LayerNorm (không trừ trung bình), rẻ hơn mà hiệu quả tương đương.

**QK-norm:** cả OLMoE và Qwen3 áp RMSNorm lên q và k trước khi tính điểm, để điểm attention không phình to. Khác biệt:

- OLMoE chuẩn hóa trên **toàn bộ** vector q (2048 chiều).
- Qwen3 chuẩn hóa **từng đầu** riêng (mỗi đầu 128 chiều).

Engine xử lý bằng `qk_norm_in_place` với độ rộng chuẩn hóa chọn theo `config.is_qwen3_moe()`. Đây là ví dụ điển hình: hai mô hình "giống nhau" trên giấy nhưng khác ở chi tiết, và chi tiết sai là logits sai. Đó là lý do cần so logits với Transformers (module 12).

## 7. Residual connection

Mỗi khối cộng đầu ra vào đầu vào: `h = h + Attention(RMSNorm(h))`. Đường tắt này giúp tín hiệu và gradient đi xuyên qua hàng chục lớp mà không bị suy giảm. Có thể hình dung `h` là một "dòng chảy" (residual stream) mà mỗi lớp đọc ra rồi ghi thêm thông tin vào.

## 8. KV cache, prefill và decode

### 8.1 Vì sao cần KV cache

Khi sinh token thứ t, attention cần k và v của **tất cả** token trước đó. Nếu tính lại từ đầu mỗi bước, sinh n token tốn O(n²) lượt chạy mô hình. Nhưng k, v của token cũ không đổi (mặt nạ nhân quả: token cũ không nhìn thấy token mới), nên ta **lưu lại** chúng: đó là KV cache. Mỗi bước mới chỉ tính q, k, v cho đúng một token rồi nối k, v vào cache.

Dung lượng KV:

```
B_KV = 2 (k và v) × L × T × h_kv × d_h × số byte mỗi số
```

OLMoE FP32: `2 × 16 × 16 × 128 × 4 = 256 KiB` mỗi token, 128 MiB cho 512 token. Qwen3 nhờ GQA chỉ 192 KiB mỗi token dù có 48 lớp. Với ngữ cảnh 100 000 token, KV cache có thể lớn hơn chính trọng số mô hình. Đó là lý do các hệ thống lớn dùng PagedAttention (vLLM), lượng tử hóa KV, GQA/MQA.

### 8.2 Hai giai đoạn

| | Prefill | Decode |
| --- | --- | --- |
| Đầu vào | toàn bộ prompt (n token) | 1 token vừa sinh |
| Phép tính chính | ma trận × ma trận (GEMM) | ma trận × vector (GEMV) |
| Giới hạn bởi | tính toán | băng thông bộ nhớ |
| Chỉ số đo | TTFT (time to first token) | tốc độ sinh (tok/s) |

Prefill xử lý nhiều token cùng lúc, nên mỗi trọng số được dùng cho nhiều token và GPU/SIMD được tận dụng tốt. Decode xử lý từng token, phải đọc toàn bộ trọng số cần dùng chỉ để làm một matvec. Hai giai đoạn có đặc tính hiệu năng khác hẳn nhau, nên đồ án đo chúng riêng.

**Trong đồ án:** `OlmoeModel::prefill` xử lý cả prompt theo từng lớp ("layer-major": mọi token đi qua lớp 1 rồi mới sang lớp 2). Nhờ vậy, mỗi expert được nạp một lần cho mọi token cần nó trong lớp đó. `OlmoeModel::step` xử lý một token khi decode.

### 8.3 Tái sử dụng tiền tố

Trong hội thoại, lượt hỏi sau chứa toàn bộ lượt trước. Nếu KV của phần chung còn giữ lại, chỉ cần prefill phần mới. Engine tìm **tiền tố chung dài nhất** giữa chuỗi token mới và chuỗi đã lưu của phiên (`take_prefix_kv`), cắt KV đến đó và chỉ tính phần còn lại. Thí nghiệm cho thấy prefill giảm 67,56% khi 104/116 token trùng. Kỹ thuật này gọi chung là **prefix caching**, có trong vLLM, SGLang và các API thương mại (giảm giá cho phần prompt được cache).

## 9. Từ logits đến token

LM head là ma trận `V × d` biến vector ẩn cuối thành điểm (logits) cho từng token trong từ vựng.

- **Greedy:** chọn token có điểm cao nhất (`argmax_greedy` trong `src/olmoe.rs`). Kết quả tất định. Đồ án dùng greedy cho mọi phép đo, để hai lượt chạy so sánh được với nhau.
- **Sampling:** chia logits cho **temperature** T rồi softmax, lấy mẫu ngẫu nhiên. T nhỏ thì gần greedy, T lớn thì đa dạng hơn.
- **Top-k / top-p:** chỉ lấy mẫu trong k token tốt nhất, hoặc trong nhóm nhỏ nhất có tổng xác suất ≥ p.
- **Repetition penalty:** giảm điểm token đã xuất hiện để tránh lặp.

Engine có các tùy chọn này trong `src/sample.rs`, dùng khi request truyền tham số `temperature`, `top_p`...

---

## Bài tập

### Bài 1 ★ — Tokenizer

Cài `pip install tokenizers` rồi nạp `tokenizer.json` từ thư mục checkpoint OLMoE. Mã hóa các câu: `"Where is Mars?"`, `"Sao Hỏa ở đâu?"`, `"Mars"`, `" Mars"`. Mỗi câu ra bao nhiêu token? Nhận xét gì về tiếng Việt?

<details><summary>Gợi ý</summary>

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("../Models/olmoe_merged/tokenizer.json")
for s in ["Where is Mars?", "Sao Hỏa ở đâu?", "Mars", " Mars"]:
    e = tok.encode(s); print(len(e.ids), e.tokens)
```

Câu tiếng Việt cùng nghĩa thường tốn gấp 2–3 lần số token, vì các ký tự có dấu bị cắt thành nhiều byte. Hệ quả: với cùng cửa sổ ngữ cảnh, nội dung tiếng Việt chứa được ít hơn và sinh chậm hơn.
</details>

### Bài 2 ★ — Dung lượng KV

Tính dung lượng KV cache FP32 cho 512 token và 32 768 token với: OLMoE (MHA 16 đầu KV), Qwen3 (GQA 4 đầu KV), và một Qwen3 giả định dùng MHA (32 đầu KV). GQA tiết kiệm bao nhiêu?

<details><summary>Lời giải</summary>

Mỗi token: `2 × L × h_kv × d_h × 4` byte.

- OLMoE: `2×16×16×128×4 = 262 144` B = 256 KiB. 512 token = 128 MiB; 32 768 token = 8 GiB.
- Qwen3 GQA: `2×48×4×128×4 = 196 608` B = 192 KiB. 512 token = 96 MiB; 32 768 token = 6 GiB.
- Qwen3 MHA giả định: gấp 8 lần GQA = 1,5 MiB mỗi token. 32 768 token = 48 GiB.

GQA tiết kiệm 87,5% so với MHA cùng số đầu query.
</details>

### Bài 3 ★★ — RoPE bảo toàn khoảng cách

Cài RoPE kiểu rotate-half bằng numpy cho vector 128 chiều, `base = 10000`. Chứng minh bằng thực nghiệm: với q, k ngẫu nhiên cố định, `rope(q, t) · rope(k, j)` chỉ phụ thuộc `t − j` (thử `(t, j) = (5, 2), (105, 102), (1005, 1002)`).

<details><summary>Lời giải</summary>

```python
def rope_half(x, pos, base=10000.0):
    h = x.shape[0] // 2
    inv = base ** (-np.arange(h) * 2.0 / x.shape[0])
    ang = pos * inv
    a, b = x[:h], x[h:]
    return np.concatenate([a*np.cos(ang) - b*np.sin(ang), b*np.cos(ang) + a*np.sin(ang)])

q, k = np.random.default_rng(1).normal(size=(2, 128))
for t, j in [(5, 2), (105, 102), (1005, 1002)]:
    print(rope_half(q, t) @ rope_half(k, j))      # ba giá trị bằng nhau (sai số làm tròn)
```

Lý do: phép xoay góc α rồi tích vô hướng với vector xoay góc β tương đương xoay một vector góc `α − β`.
</details>

### Bài 4 ★★ — Đếm trong code

Mở `src/backend.rs::causal_attention_into`. Trả lời:

1. Hàm kiểm tra những điều kiện nào về kích thước trước khi tính?
2. Khi nào nó chạy song song các đầu bằng Rayon?
3. Biến `normalizer` đóng vai trò gì trong online softmax?

<details><summary>Lời giải</summary>

1. Số đầu và số đầu KV khác 0, số đầu chia hết cho số đầu KV, `query` dài bằng `context`, `query` chia hết cho số đầu, `keys` dài bằng `values`, `keys` chia hết cho độ rộng KV, có ít nhất một vị trí, và mọi đầu vào là số hữu hạn.
2. Khi `num_attention_heads * positions >= 4096`. Với ngữ cảnh ngắn, chi phí chia việc cho các luồng lớn hơn lợi ích.
3. `normalizer` là tổng `S = Σ e^{s_j - M}`, được co lại mỗi khi gặp điểm lớn hơn max hiện tại. Cuối cùng đầu ra chia cho nó.
</details>

### Bài 5 ★★★ — Transformer mini bằng numpy

Viết bằng numpy một decoder 2 lớp, `d = 64`, 4 đầu, FFN SwiGLU với `m = 128`, từ vựng 100, trọng số ngẫu nhiên. Cài:

1. `forward_full(tokens)`: tính logits cho mọi vị trí bằng attention có mặt nạ nhân quả.
2. `forward_step(token, kv)`: tính logits cho một token mới dùng KV cache.
3. Kiểm chứng: sinh 20 token bằng cách lặp `forward_step` cho logits trùng với `forward_full` tại mọi vị trí (sai số dưới 1e-6).

<details><summary>Gợi ý</summary>

Đây là phiên bản thu nhỏ của chính bài kiểm tra trong đồ án: `layer_major_prefill_matches_tokenwise_last_logits` và `prefix_prefill_matches_full_logits_for_every_split` (trong `src/olmoe.rs`) kiểm rằng prefill nhiều token và decode từng token cho cùng logits. Bỏ RoPE ở bước đầu để dễ, sau đó thêm vào: lúc đó phải xoay k **trước** khi lưu vào cache, đúng vị trí của nó.

Mẹo kiểm lỗi: nếu `forward_step` lệch ở vị trí đầu tiên thì lỗi ở phép chiếu hoặc norm; nếu chỉ lệch từ vị trí thứ hai trở đi thì lỗi ở KV cache hoặc RoPE.
</details>

### Bài 6 ★★★ — So với mô hình thật

Dùng Hugging Face Transformers (trên máy có đủ RAM, hoặc dùng một mô hình nhỏ như `Qwen/Qwen3-0.6B`) để in: số lớp, số đầu, số đầu KV, `head_dim`, `rope_theta`, có QK-norm không, `tie_word_embeddings`. Đối chiếu với bảng ký hiệu trong `study/README.md` và với hàm `Config::validate` trong `src/olmoe.rs`.

<details><summary>Gợi ý</summary>

`AutoConfig.from_pretrained(...)` đủ để đọc cấu hình mà không tải trọng số. `Config::validate` liệt kê các điều kiện engine yêu cầu để từ chối cấu hình không hỗ trợ ngay khi nạp.
</details>

## Tài liệu đọc thêm

- Vaswani et al., *Attention Is All You Need* (2017).
- Jay Alammar, *The Illustrated Transformer*: https://jalammar.github.io/illustrated-transformer/
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (video) và nanoGPT.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021).
- Ainslie et al., *GQA* (2023).
- Hugging Face, *KV cache strategies*.
