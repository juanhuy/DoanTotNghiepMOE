# Module 06 — Mixture-of-Experts

## Mục tiêu

Hiểu vì sao MoE cho phép mô hình rất lớn mà mỗi token chỉ tốn ít phép tính, Router chọn expert ra sao, expert tính gì, và vì sao chạy MoE trên máy cá nhân là bài toán **di chuyển dữ liệu** chứ không phải bài toán tính toán. Đây là module trung tâm của đồ án.

**Dùng khi đi làm:** phần lớn các LLM mạnh hiện nay là MoE: Mixtral, DeepSeek-V3/R1, Qwen3-MoE, Llama 4, nhiều mô hình thương mại. Làm inference, fine-tune hay triển khai các mô hình này đều cần hiểu router, sparsity, expert parallelism và chi phí bộ nhớ.

## 1. Từ FFN dày đặc đến MoE

Trong Transformer thường, sau attention mỗi lớp có một mạng feed-forward (FFN) áp dụng cho từng token. FFN chiếm khoảng 2/3 số tham số của mô hình.

**Ý tưởng MoE:** thay một FFN lớn bằng E FFN nhỏ hơn (các **expert**) và một **router**. Với mỗi token, router chọn k expert (k ≪ E); chỉ k expert đó được tính.

```
                       ┌─► Expert 3 ─┐
h ─► RMSNorm ─► Router ─┼─► Expert 17 ┼─► Σ g_e · Expert_e(h) ─► + h
   (chọn top-k)         ├─► ...       │
                       └─► Expert 50 ─┘
                   (chỉ k trong E expert được tính)
```

Hệ quả:

| | OLMoE-1B-7B | Qwen3-30B-A3B |
| --- | --- | --- |
| Tổng tham số | ~7 tỷ | ~30 tỷ |
| Tham số dùng cho mỗi token | ~1 tỷ | ~3 tỷ |
| Expert mỗi lớp E, chọn k | 64, 8 | 128, 8 |
| Tổng số expert (E × L) | 1 024 | 6 144 |

"A3B" trong tên Qwen3-30B-A3B nghĩa là "active 3 billion": 3 tỷ tham số kích hoạt mỗi token. Mô hình có kiến thức của 30 tỷ tham số nhưng tốc độ tính như mô hình 3 tỷ.

**Cái giá:** tuy chỉ dùng 3 tỷ tham số mỗi token, toàn bộ 30 tỷ vẫn phải **nằm ở đâu đó** truy cập được, vì token tiếp theo có thể chọn expert khác. Đây chính là bài toán của đồ án.

## 2. Router

Router là một ma trận `W_r` kích thước `E × d`. Với vector ẩn h:

```
r = W_r h                         (E điểm, một điểm cho mỗi expert)
p = softmax(r)                    (xác suất)
S = top_k(p)                      (k expert có xác suất cao nhất)
y = Σ_{e ∈ S} g_e · f_e(h)        (tổng có trọng số các đầu ra expert)
```

Có hai cách tính trọng số `g_e`:

- **Không chuẩn hóa lại** (OLMoE): `g_e = p_e`. Tổng các g nhỏ hơn 1.
- **Chuẩn hóa lại** (Qwen3): `g_e = p_e / Σ_{j∈S} p_j`. Tổng các g bằng 1.

Cấu hình checkpoint ghi rõ qua `norm_topk_prob`. Dùng sai cách là đầu ra sai, dù mô hình vẫn chạy và sinh ra chữ.

**Code thật** (`src/olmoe.rs`, trong `step_inner`):

```rust
layer.router.multiply_loaded_into(&scratch.hidden, &mut scratch.router)?;   // r = W_r h
softmax(&mut scratch.router)?;                                              // p
// lấy k phần tử lớn nhất mà không sắp xếp toàn bộ E phần tử
valid_indices.select_nth_unstable_by(top_k, |&a, &b| {
    scratch.router[b].total_cmp(&scratch.router[a]).then_with(|| a.cmp(&b))
});
let top_experts = &mut valid_indices[..top_k];
top_experts.sort_by(/* cùng tiêu chí */);           // chỉ sắp k phần tử
let normalizer = if c.norm_topk_prob {
    top_experts.iter().map(|&e| scratch.router[e]).sum()
} else { 1.0 };
```

Điểm đáng chú ý:

- `select_nth_unstable_by` có độ phức tạp O(E), nhanh hơn sắp xếp toàn bộ O(E log E).
- Khi hai expert bằng điểm, chọn chỉ số nhỏ hơn (`then_with(|| a.cmp(&b))`). Nhờ quy tắc phá hòa cố định, kết quả tất định.
- `total_cmp` so sánh an toàn cả khi có NaN.

## 3. Expert: khối SwiGLU

Mỗi expert là một FFN dạng SwiGLU gồm 3 ma trận:

```
gate = W_g h          (m × d)
up   = W_u h          (m × d)
a    = SiLU(gate) ⊙ up      (⊙: nhân từng phần tử)
out  = W_d a          (d × m)
```

Nhánh `gate` sau SiLU đóng vai trò cái van điều chỉnh từng thành phần của nhánh `up`. Cấu trúc này cho chất lượng tốt hơn FFN dùng ReLU với cùng số tham số, nên hầu hết LLM hiện đại đều dùng.

**Trong đồ án:** `src/int8_expert.rs::forward_reuse` tính đúng ba bước trên với trọng số INT8:

- `fused_gate_up_swiglu`: tính gate và up **trong cùng một vòng duyệt** đầu vào (hàm `dot_i8_f32_pair` đọc hai hàng cùng lúc), rồi áp SiLU và nhân.
- `matvec_into`: tính `W_d a`.

Ba ma trận của một expert được lưu liền nhau trong một `Vec<u8>`: `[W_g | W_u | W_d]`, kèm hệ số tỉ lệ `[s_g | s_u | s_d]`. Một lần đọc tệp lấy cả expert.

## 4. Vì sao MoE bị giới hạn bởi bộ nhớ

Lấy decode một token của Qwen3:

- Mỗi lớp cần 8 expert, mỗi expert khoảng 4,7 MB INT8. 48 lớp: khoảng **1,8 GB dữ liệu expert mỗi token**.
- Số phép tính tương ứng khoảng 1,8 tỷ phép nhân–cộng: một CPU hiện đại làm trong vài chục mili giây.
- Nhưng nếu expert không có sẵn trong RAM mà phải đọc từ SSD với 2 GB/s, riêng việc đọc mất khoảng 0,9 giây.

Vậy bài toán là: **giữ được những expert nào ở bộ nhớ nhanh, và làm sao để phải di chuyển ít dữ liệu nhất**. Toàn bộ thiết kế của đồ án xoay quanh câu hỏi này:

| Vấn đề | Giải pháp trong đồ án | Module |
| --- | --- | --- |
| Không đủ RAM cho mọi expert | Cache expert theo từng lớp, có ngân sách byte | 09 |
| Expert nào giữ, expert nào bỏ | Eviction theo heat, có nhóm bảo vệ | 09 |
| Đọc từ SSD chậm | Đọc đúng vị trí trong tệp; page cache của hệ điều hành | 08 |
| GPU có VRAM nhưng ít | Pool expert trên VRAM, tính ngay tại VRAM | 10 |
| Expert lớn | Lượng tử hóa INT8 (nhỏ hơn BF16 một nửa) | 07 |
| Nhiều token cùng dùng một expert khi prefill | Nạp một lần, tính cho mọi token | 05, 02 |

## 5. Hành vi định tuyến: có quy luật để khai thác không?

Nếu router chọn expert hoàn toàn ngẫu nhiên và đều, mọi cache đều vô dụng: với cache chứa 11% expert, tỉ lệ trúng (hit rate) chỉ khoảng 11%. Thực tế tốt hơn nhờ ba hiện tượng:

1. **Phân bố lệch:** một số expert được chọn thường xuyên hơn hẳn các expert khác ("expert nóng").
2. **Tính cục bộ theo thời gian:** các token liền nhau trong cùng một đoạn văn thường chọn những expert giống nhau.
3. **Trùng lặp trong prefill:** n token của prompt cùng chọn từ E expert, nên nhiều token dùng chung một expert.

Với Qwen3 và cache 14/128 expert mỗi lớp, đồ án đo được hit rate 38–73%, cao hơn hẳn mức 11% của định tuyến ngẫu nhiên. Đó là bằng chứng có cấu trúc để khai thác, nhưng vẫn còn 27–62% số lần phải đọc tệp.

Mô hình được huấn luyện kèm **load balancing loss**, một hàm phạt khiến router phân bố đều tải cho các expert (để huấn luyện song song hiệu quả). Hàm phạt này làm định tuyến kém cục bộ hơn, tức khó cache hơn. Đây là xung đột tự nhiên giữa huấn luyện và inference trên máy nhỏ, và là lý do hướng "định tuyến có tính đến cache" (ưu tiên expert đã có trong bộ nhớ khi router gần hòa) được đề xuất ở phần hướng phát triển.

## 6. Các cách chạy MoE trong thực tế

| Bối cảnh | Cách làm |
| --- | --- |
| Trung tâm dữ liệu, nhiều GPU | **Expert parallelism:** mỗi GPU giữ một nhóm expert, token được gửi tới GPU chứa expert cần dùng (giao tiếp all-to-all). |
| Một GPU lớn | Giữ toàn bộ mô hình trong VRAM, gộp nhiều request thành lô để tận dụng tính toán. |
| Máy cá nhân, VRAM ít | **Offloading:** để expert ở RAM hoặc SSD, nạp lên khi cần; cache expert hay dùng; dự đoán expert sắp cần để nạp trước. llama.cpp, các nghiên cứu như Eliseev & Mazur (2023) đi theo hướng này. |
| Máy cá nhân, tính trên CPU | Tính expert ngay trên CPU nơi dữ liệu đang nằm, chỉ đưa phần dense lên GPU. |

Đồ án kết hợp các cách cuối: dense lên GPU khi vừa, expert ở VRAM thì tính trên GPU, expert ở RAM thì tính trên CPU, expert chưa có thì đọc từ SSD. Kết luận chính đã đo: **nơi tính quan trọng hơn nơi chứa**. Chứa expert trên VRAM mà vẫn tính trên CPU thì phải chép qua lại 33–41 GiB mỗi request và làm hệ thống chậm đi.

---

## Bài tập

### Bài 1 ★ — Router bằng tay

Một lớp có E = 6 expert, k = 2. Logits router là `[2.0, 0.5, 1.5, -1.0, 2.0, 0.0]`.

1. Tính xác suất softmax.
2. Chọn top-2 theo quy tắc phá hòa của engine.
3. Tính trọng số g theo cách của OLMoE và của Qwen3.

<details><summary>Lời giải</summary>

`e^z ≈ [7.389, 1.649, 4.482, 0.368, 7.389, 1.0]`, tổng ≈ 22.277, nên `p ≈ [0.332, 0.074, 0.201, 0.017, 0.332, 0.045]`.

Expert 0 và 4 bằng điểm cao nhất; phá hòa theo chỉ số nhỏ hơn, thứ tự là `[0, 4]`. Cả hai đều được chọn.

- OLMoE: `g = [0.332, 0.332]`, tổng 0.663.
- Qwen3: `g = [0.5, 0.5]`.

Với OLMoE, đầu ra MoE có độ lớn khoảng 0.66 lần so với Qwen3 cho cùng đầu ra expert. Mô hình được huấn luyện theo đúng cách của nó, nên dùng sai cách là lệch.
</details>

### Bài 2 ★ — Tham số tổng và kích hoạt

Ước lượng số tham số expert của Qwen3-30B-A3B: `L × E × 3 × d × m` với `L = 48, E = 128, d = 2048, m = 768`. Số tham số expert kích hoạt mỗi token là bao nhiêu? Phần còn lại (attention, embedding, LM head) chiếm bao nhiêu trong tổng 30,5 tỷ?

<details><summary>Lời giải</summary>

Expert: `48 × 128 × 3 × 2048 × 768 ≈ 29,0 tỷ`. Kích hoạt: `48 × 8 × 3 × 2048 × 768 ≈ 1,81 tỷ`. Phần còn lại khoảng 1,5 tỷ (attention khoảng 0,9 tỷ, embedding và LM head mỗi cái khoảng 0,31 tỷ). Tổng tham số kích hoạt khoảng 1,8 + 1,5 ≈ 3,3 tỷ, khớp với tên "A3B".

Nhận xét: 95% tham số nằm ở expert, nhưng mỗi token chỉ dùng 6% số đó. Phần dense (5%) thì luôn được dùng. Vì vậy chiến lược hợp lý là: giữ dense thường trú, còn expert thì phân tầng.
</details>

### Bài 3 ★★ — Hit rate dưới định tuyến ngẫu nhiên

Viết mô phỏng: một lớp có E = 128 expert, mỗi token chọn ngẫu nhiên đều 8 expert, cache LRU chứa C expert. Đo hit rate với C = 14, 32, 64 qua 10 000 token. So với tỉ lệ `C/E`. Sau đó cho phân bố lệch: expert i được chọn với xác suất tỉ lệ `1/(i+1)^0.8` (phân bố Zipf). Hit rate thay đổi thế nào?

<details><summary>Gợi ý</summary>

Với định tuyến đều, hit rate xấp xỉ `C/E` (14/128 ≈ 11%). Với phân bố Zipf, hit rate cao hơn rõ vì các expert nóng gần như luôn nằm trong cache. Bài tập này giải thích vì sao đồ án đo được 38–73% chứ không phải 11%. Module 09 sẽ dùng lại bộ mô phỏng này để so các chính sách cache.
</details>

### Bài 4 ★★ — Đọc code SwiGLU

Mở `src/int8_expert.rs`, tìm `fused_gate_up_swiglu` và `matvec_into`.

1. Hệ số tỉ lệ được nhân vào lúc nào, trước hay sau tích vô hướng?
2. Vì sao tính gate và up trong cùng một vòng (`dot_i8_f32_pair`) nhanh hơn tính hai vòng riêng?
3. Song song hóa bằng Rayon được đặt ở cấp nào: theo hàng hay theo expert?

<details><summary>Lời giải</summary>

1. Sau: tích vô hướng tính trên số nguyên đã đổi sang f32, rồi nhân với hệ số tỉ lệ của hàng. Nhân một lần mỗi hàng thay vì mỗi phần tử.
2. Vector đầu vào `h` chỉ phải đọc một lần cho cả hai hàng; với dữ liệu bị giới hạn bởi băng thông, bớt một lần đọc là nhanh hơn.
3. Theo hàng: mỗi luồng tính một nhóm hàng của ma trận (song song hạt mịn trong một expert). Hàm `forward_single_threaded` là biến thể cho trường hợp song song ở cấp expert (mỗi luồng tính trọn một expert).
</details>

### Bài 5 ★★ — Phân tích vết định tuyến

Engine có thể ghi vết định tuyến (route trace) và `tools/analyze_route_trace.py` phân tích nó. Đọc script đó: nó tính những thống kê nào? Nếu có thời gian, bật ghi vết (xem `RouteTraceLogger` trong `src/olmoe.rs` để biết biến môi trường), chạy một prompt, rồi trả lời: ở lớp 0, 10 expert nóng nhất chiếm bao nhiêu phần trăm số lượt chọn?

<details><summary>Gợi ý</summary>

Một trong các thống kê là hệ số Gini, đo độ lệch của phân bố (0: đều tuyệt đối, 1: dồn hết vào một expert). Lưu ý cách đọc: con số Gini từ một vài prompt ngắn chỉ mô tả workload đó. Đây là ví dụ về việc phân biệt quan sát sơ bộ với kết luận tổng quát, chủ đề của module 11.
</details>

### Bài 6 ★★★ — MoE layer bằng numpy

Viết một lớp MoE hoàn chỉnh bằng numpy: E = 16 expert SwiGLU (`d = 64, m = 128`), router, k = 4, có cờ `norm_topk_prob`. Viết hai cách tính cho một **lô** n token:

1. Theo token: mỗi token tính 4 expert của nó.
2. Theo expert: gom các token theo expert, mỗi expert tính một lần cho cả nhóm token của nó bằng matmul.

Kiểm hai cách cho cùng kết quả. Với n = 64, mỗi expert được dùng trung bình bao nhiêu lần? Cách 2 đọc ma trận expert ít hơn bao nhiêu lần?

<details><summary>Gợi ý</summary>

Cách 2 là ý tưởng của prefill layer-major và kernel expert nhiều token trong đồ án. Với n = 64 token, k = 4, E = 16: tổng 256 lượt chọn, trung bình 16 lượt mỗi expert. Cách 1 đọc trọng số 256 lần (mỗi lượt một lần), cách 2 chỉ đọc 16 lần. Khi decode (n = 1), hai cách như nhau. Đó là lý do prefill nhanh hơn decode tính trên mỗi token.
</details>

## Tài liệu đọc thêm

- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017).
- Fedus et al., *Switch Transformers* (2021).
- Muennighoff et al., *OLMoE: Open Mixture-of-Experts Language Models* (2024).
- Qwen Team, *Qwen3 Technical Report* (2025).
- Eliseev & Mazur, *Fast Inference of Mixture-of-Experts Language Models with Offloading* (2023).
- Hugging Face blog, *Mixture of Experts Explained*.
