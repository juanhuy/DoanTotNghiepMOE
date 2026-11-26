# Module 07 — Lượng tử hóa

## Mục tiêu

Hiểu cách nén trọng số mô hình từ 16 bit xuống 8 hoặc 4 bit, sai số sinh ra từ đâu và lớn cỡ nào, các biến thể (theo tensor, theo hàng, theo nhóm), và cách đánh giá đúng ảnh hưởng của lượng tử hóa lên đầu ra.

**Dùng khi đi làm:** lượng tử hóa là kỹ thuật số một để chạy LLM rẻ hơn: GGUF của llama.cpp, GPTQ, AWQ, bitsandbytes, FP8 trên GPU H100. Cũng dùng trong mô hình trên thiết bị di động (TensorFlow Lite, Core ML), xử lý ảnh và âm thanh nhúng.

## 1. Vì sao lượng tử hóa

Với mô hình chạy bằng cách đọc trọng số (module 06), **số byte trọng số chính là thời gian**. Giảm từ BF16 xuống INT8 làm:

- dung lượng tệp và RAM giảm một nửa,
- lượng dữ liệu phải đọc mỗi token giảm một nửa, nên tốc độ có thể tăng gần gấp đôi khi bị giới hạn bởi băng thông,
- số expert vừa một ngân sách cache tăng gấp đôi.

Qwen3-30B-A3B ở BF16 khoảng 61 GB; ở INT8 (expert) khoảng 30 GB. Không có lượng tử hóa thì đồ án không thể chạy mô hình này dưới trần 8 GB.

## 2. Lượng tử hóa đều đối xứng

Ý tưởng: biểu diễn số thực w bằng số nguyên q và một hệ số tỉ lệ s chung cho một nhóm số:

```
q = clamp(round(w / s), -128, 127)        lượng tử hóa
ŵ = s · q                                 giải lượng tử (xấp xỉ w)
```

Chọn `s = max|w| / 127` để số lớn nhất của nhóm vừa khít 127. Đây là cách **đối xứng** (zero-point bằng 0), phù hợp với trọng số vì phân bố của chúng gần đối xứng quanh 0.

**Cách bất đối xứng** thêm một điểm 0 (zero-point) z: `q = round(w/s) + z`. Dùng khi dữ liệu lệch về một phía (ví dụ activation sau ReLU luôn ≥ 0). Tận dụng hết dải số nguyên nhưng tính toán phức tạp hơn.

### Sai số

Phép làm tròn sinh sai số trong khoảng `[-s/2, s/2]`. Nếu sai số phân bố đều, độ lệch chuẩn của nó là `s/√12`. Hệ quả:

- **s càng nhỏ, sai số càng nhỏ.** Mà s tỉ lệ với số lớn nhất trong nhóm.
- **Một giá trị ngoại lai (outlier) làm hỏng cả nhóm.** Nếu nhóm có một số bằng 10 còn các số khác cỡ 0,1, thì s = 10/127 ≈ 0,079, và các số 0,1 chỉ còn được biểu diễn bằng 1 hoặc 2 mức. Gần như mất hết thông tin.

## 3. Độ chi tiết của hệ số tỉ lệ

| Kiểu | Một hệ số tỉ lệ cho | Chi phí phụ | Độ chính xác |
| --- | --- | --- | --- |
| Theo tensor | cả ma trận | không đáng kể | kém nhất, outlier ảnh hưởng toàn bộ |
| Theo hàng (per-row / per-channel) | mỗi hàng | 4 byte mỗi hàng | tốt, đồ án dùng cách này |
| Theo nhóm (per-group) | mỗi nhóm 32–128 phần tử | 2–4 byte mỗi nhóm | tốt nhất, chuẩn cho INT4 |

**Trong đồ án:** bộ chuyển đổi `tools/prepare_olmoe.py` lượng tử hóa theo hàng:

```python
scale = tensor.abs().amax(dim=1).clamp(min=1e-12) / 127.0
quantized = (tensor / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
```

`clamp(min=1e-12)` tránh chia cho 0 ở hàng toàn số 0. Chi phí phụ của hệ số tỉ lệ: với một expert OLMoE, `2m + d = 4096` hệ số × 4 byte = 16 KiB trên 6 MiB, khoảng 0,26%.

Tại sao theo hàng lại hợp với matvec: mỗi phần tử đầu ra `y_i` là tích vô hướng của **một hàng** với đầu vào. Với một hệ số tỉ lệ cho mỗi hàng:

```
y_i = Σ_j (s_i · q_ij) · x_j = s_i · Σ_j q_ij · x_j
```

Hệ số tỉ lệ ra ngoài tổng: tính tích vô hướng trên số nguyên trước, nhân `s_i` một lần ở cuối. Không bao giờ phải dựng lại ma trận số thực. Đây chính là cách kernel CPU (`src/int8_expert.rs`) và kernel GPU (`src/gpu_expert.rs`) làm.

## 4. INT4 và đóng gói bit

INT4 có 16 mức (-8 đến 7). Hai số INT4 được nhét vào một byte: 4 bit thấp và 4 bit cao. Khi đọc phải tách ra:

```rust
let lo = ((byte << 4) as i8) >> 4;   // 4 bit thấp, giữ dấu nhờ dịch phải số có dấu
let hi = (byte as i8) >> 4;          // 4 bit cao
```

Với chỉ 16 mức, lượng tử hóa theo hàng thường quá thô; cần theo nhóm (32–128 phần tử) và các thuật toán thông minh hơn làm tròn đơn giản:

- **GPTQ:** lượng tử hóa từng cột, rồi điều chỉnh các cột chưa lượng tử hóa để bù sai số, dùng thông tin bậc hai từ dữ liệu hiệu chỉnh.
- **AWQ:** nhân trọng số của các kênh "quan trọng" (kênh có activation lớn) với hệ số trước khi lượng tử hóa, để bảo vệ chúng.
- **GGUF k-quants (llama.cpp):** lượng tử hóa hai cấp, hệ số tỉ lệ của nhóm con cũng được lượng tử hóa.

Engine có đường INT4 (`dot_i4_f32` trong `src/int8_expert.rs`, `tools/prepare_olmoe_int4.py`). Ngày 23/11 nhóm quyết định không phát triển tiếp INT4. Đồ án **không** dùng INT4 làm cấu hình chính và không trích số liệu tốc độ hay chất lượng của nó, vì chưa có lượt đo đủ nguồn gốc. INT4 được nêu là hướng phát triển: nó giảm một nửa lượng byte phải đọc, đánh thẳng vào nút thắt I/O của Qwen3, nhưng phải đánh giá chất lượng trước.

## 5. Chỉ lượng tử hóa trọng số hay cả activation

- **Chỉ trọng số (weight-only):** trọng số INT8/INT4, activation vẫn FP32/FP16. Phép tính đổi trọng số sang số thực ngay trong vòng nhân. Lợi ích chủ yếu là giảm byte phải đọc. Đây là cách đồ án dùng, và là cách phổ biến cho inference trên máy cá nhân.
- **Cả trọng số và activation (W8A8):** tính tích vô hướng hoàn toàn bằng số nguyên (lệnh `VPDPBUSD` trên CPU, Tensor Core INT8 trên GPU). Nhanh hơn về tính toán nhưng activation có outlier lớn, khó lượng tử hóa (SmoothQuant giải quyết bằng cách chuyển độ khó từ activation sang trọng số).

Đồ án giữ dense ở BF16 (Qwen3) hoặc FP32 (OLMoE) và chỉ lượng tử hóa expert, vì expert chiếm 95% dung lượng còn dense được dùng ở mọi token nên đáng giữ chính xác.

## 6. Đánh giá ảnh hưởng của lượng tử hóa

Có nhiều mức đánh giá, từ rẻ đến đắt:

| Mức | Cách đo | Ưu điểm | Hạn chế |
| --- | --- | --- | --- |
| Sai số trọng số | MSE giữa w và ŵ | rất rẻ | không nói gì về đầu ra |
| Sai số logits | top-1 trùng, top-5 overlap, KL so với bản gốc | rẻ, nhạy, so được từng vị trí | không phải chất lượng tác vụ |
| Perplexity | trên một tập văn bản lớn | chuẩn, so được giữa các phương pháp | cần tập dữ liệu và thời gian |
| Benchmark tác vụ | MMLU, GSM8K, HumanEval... | đo chất lượng thực | đắt, nhiễu |

**Kết quả trong đồ án** (hồ sơ `benchmarks/experiments/2026-11-20-real-model-parity-i5`), so engine INT8 với Transformers dùng trọng số BF16 gốc:

| | Qwen3-30B-A3B | OLMoE-1B-7B |
| --- | --- | --- |
| Top-1 trùng | 157/159 (98,7%) | 184/186 (98,9%) |
| Top-5 overlap | 0,945 | 0,972 |
| KL trung bình | 0,0089 nat | 0,0077 nat |

Bốn vị trí đổi top-1 đều là vị trí **gần hòa**: hai token đứng đầu chênh nhau 0,004 đến 0,19 logit. Ví dụ Qwen3 trả lời "What is love?" bắt đầu bằng "Love" (BF16) hay "The" (INT8). Cả hai đều hợp lý. Đây là cách đọc đúng: đổi top-1 ở chỗ biên rất nhỏ là bình thường; đổi ở chỗ biên lớn mới là dấu hiệu hỏng.

Đồ án tách được sai số do lượng tử hóa khỏi sai số do engine nhờ so ba bộ logits (module 12):
- engine so với Transformers dùng **cùng** expert INT8: lệch cỡ 10⁻⁴, tức engine tính đúng;
- Transformers INT8 so với Transformers BF16: cho đúng các con số ở bảng trên, tức toàn bộ chênh lệch đến từ lượng tử hóa.

**Giới hạn cần nói rõ khi bảo vệ:** đánh giá này ở mức logits trên 6 chuỗi ngắn. Đồ án chưa chạy perplexity trên tập lớn hay benchmark tác vụ chuẩn.

---

## Bài tập

### Bài 1 ★ — Lượng tử hóa bằng tay

Hàng trọng số `w = [0.12, -0.50, 0.33, 0.01, -0.07]`.

1. Tính `s`, các giá trị `q`, và `ŵ` theo INT8 đối xứng theo hàng.
2. Sai số tuyệt đối lớn nhất là bao nhiêu? So với `s/2`.
3. Nếu thêm phần tử `5.0` vào hàng, làm lại. Phần tử `0.01` còn biểu diễn được không?

<details><summary>Lời giải</summary>

1. `s = 0.50/127 ≈ 0.003937`. `q = round(w/s) = [30, -127, 84, 3, -18]`. `ŵ = [0.1181, -0.5, 0.3307, 0.0118, -0.0709]`.
2. Lớn nhất khoảng 0.0019, nhỏ hơn `s/2 ≈ 0.00197`, đúng như lý thuyết.
3. `s = 5/127 ≈ 0.0394`. `q(0.01) = round(0.254) = 0`, nên `ŵ = 0`: mất hoàn toàn. `0.12` thành 3·0.0394 = 0.118, vẫn tạm ổn, nhưng `-0.07` thành -2·0.0394 = -0.079 (sai 13%). Một outlier làm cả hàng kém đi 10 lần.
</details>

### Bài 2 ★★ — Đo sai số trên trọng số thật

Dùng `safetensors` và `numpy` (hoặc `torch`) đọc một ma trận expert BF16 từ checkpoint gốc (nếu có) hoặc tạo ma trận `1024 × 2048` phân phối chuẩn với vài outlier. Lượng tử hóa INT8 theo tensor, theo hàng, theo nhóm 64. Với mỗi cách, tính:

1. MSE tương đối `‖w − ŵ‖² / ‖w‖²`.
2. Sai số tương đối của `W x` với x ngẫu nhiên: `‖Wx − Ŵx‖ / ‖Wx‖`.

<details><summary>Gợi ý</summary>

```python
def quant_rows(W, group=None):
    if group is None:
        s = np.abs(W).max(axis=1, keepdims=True) / 127
        return np.clip(np.round(W / s), -128, 127) * s
    Wg = W.reshape(W.shape[0], -1, group)
    s = np.abs(Wg).max(axis=2, keepdims=True) / 127
    return (np.clip(np.round(Wg / s), -128, 127) * s).reshape(W.shape)
```

Theo tensor kém nhất, theo nhóm tốt nhất. Với INT8, theo hàng thường đã đủ tốt (sai số tương đối cỡ 0,5–1%); với INT4 thì khoảng cách giữa các cách trở nên rất lớn.
</details>

### Bài 3 ★★ — INT4 đóng gói

Viết hàm `pack_int4(q: np.ndarray) -> bytes` gói mảng số nguyên trong `[-8, 7]` (độ dài chẵn) thành byte, phần tử chẵn ở 4 bit thấp; và `unpack_int4` ngược lại. Kiểm `unpack(pack(q)) == q` với mọi giá trị. So với cách engine giải nén trong `dot_i4_f32_scalar` (`src/int8_expert.rs`).

<details><summary>Lời giải</summary>

```python
def pack_int4(q):
    u = (q.astype(np.int16) & 0xF).astype(np.uint8)
    return (u[0::2] | (u[1::2] << 4)).tobytes()

def unpack_int4(b):
    a = np.frombuffer(b, np.uint8)
    lo = (a & 0xF).astype(np.int8); hi = (a >> 4).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo); hi = np.where(hi > 7, hi - 16, hi)
    out = np.empty(a.size * 2, np.int8); out[0::2], out[1::2] = lo, hi
    return out

q = np.random.default_rng(0).integers(-8, 8, size=1000)
assert (unpack_int4(pack_int4(q)) == q).all()
```

Bước dễ sai nhất là khôi phục dấu: 4 bit `1111` là -1 chứ không phải 15.
</details>

### Bài 4 ★★ — Lượng tử hóa và KL

Dùng Transformer mini của bài 5 module 05 (hoặc một ma trận LM head ngẫu nhiên `V=1000, d=64`). Lượng tử hóa các ma trận FFN về INT8 và INT4 (theo nhóm 32). Với 200 vị trí, đo top-1 trùng, top-5 overlap, KL trung bình so với bản FP32. Liệt kê các vị trí đổi top-1 kèm biên (hiệu hai logits lớn nhất của bản gốc). Có quy luật gì?

<details><summary>Gợi ý</summary>

Các vị trí đổi top-1 hầu như luôn có biên nhỏ. Vẽ biểu đồ biên của mọi vị trí, tô màu các vị trí bị đổi: chúng dồn về phía biên gần 0. Đây là lý do báo cáo phải ghi biên kèm mỗi vị trí đổi top-1, đúng như `tools/compare_logits.py` làm.
</details>

### Bài 5 ★★★ — Tự cài AWQ đơn giản

Tạo ma trận `W (256 × 512)` và tập activation `X (1000 × 512)` trong đó 5 kênh đầu vào có độ lớn gấp 50 lần các kênh khác. Lượng tử hóa INT4 theo nhóm 64:

1. Cách thường.
2. Cách kiểu AWQ: nhân cột j của W với `α_j = (mean|X_j|)^0.5`, lượng tử hóa, rồi chia đầu vào tương ứng cho `α_j` khi tính.

So sai số `‖XWᵀ − XŴᵀ‖`. Vì sao cách 2 tốt hơn?

<details><summary>Gợi ý</summary>

Kênh có activation lớn khuếch đại sai số của trọng số tương ứng. Nhân các cột đó lên trước khi lượng tử hóa làm chúng chiếm nhiều mức lượng tử hơn (sai số tương đối nhỏ hơn), đổi lại các cột khác chịu sai số lớn hơn một chút nhưng chúng ít quan trọng hơn. Phép biến đổi tương đương về toán học vì `(W·diag(α))·(diag(1/α)·x) = W x`.
</details>

## Tài liệu đọc thêm

- Gholami et al., *A Survey of Quantization Methods for Efficient Neural Network Inference* (2021).
- Dettmers et al., *LLM.int8()* (2022).
- Frantar et al., *GPTQ* (2022).
- Lin et al., *AWQ* (2023).
- Xiao et al., *SmoothQuant* (2022).
- Tài liệu lượng tử hóa của llama.cpp (định dạng GGUF, k-quants).
