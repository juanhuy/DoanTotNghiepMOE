# Module 12 — Kiểm thử tính đúng của code số học

## Mục tiêu

Biết cách chứng minh một cài đặt số học (mô hình AI, kernel, mô phỏng) tính đúng: dùng bản tham chiếu, chọn dung sai hợp lý, tạo fixture nhỏ, kiểm thử vi sai, kiểm thử tối ưu không đổi kết quả, và so logits trên mô hình thật lớn hơn RAM. Hiểu đồ án chứng minh "engine tính đúng" bằng những bằng chứng nào.

**Dùng khi đi làm:** mọi code có tính toán: port mô hình giữa PyTorch, ONNX, TensorRT, llama.cpp; viết kernel tối ưu; chuyển code Python sang C++/Rust; hệ thống tài chính, khoa học. Lỗi số học là loại lỗi nguy hiểm nhất vì chương trình vẫn chạy, vẫn ra kết quả trông hợp lý, chỉ là sai.

## 1. Vì sao code số học khó kiểm thử

- **Không có đáp án viết tay được.** Không ai tính tay logits của một mô hình 30 tỷ tham số.
- **Không so được bằng `==`.** Hai cài đặt đúng vẫn lệch nhau ở các chữ số cuối do thứ tự cộng (module 04).
- **Lỗi có thể rất tinh vi.** Ghép sai cặp RoPE, quên chuẩn hóa lại top-k, chuẩn hóa QK sai độ rộng: mô hình vẫn sinh chữ, đọc qua thấy "cũng được", nhưng chất lượng kém đi.
- **"Câu trả lời trông đúng" không phải bằng chứng.** So vài câu trả lời chỉ kiểm vài vị trí trên đường đi của giải mã tham lam; hàng nghìn vị trí khác không được kiểm.

## 2. Bản tham chiếu

Cách chuẩn là so với một **cài đặt tham chiếu** đã được tin cậy: với LLM, đó là Hugging Face Transformers, vì nó là bản chính thức đi kèm checkpoint.

Có hai mức:

1. **Fixture nhỏ:** tạo mô hình tí hon với trọng số ngẫu nhiên, chạy bằng Transformers, lưu đầu vào và logits ra tệp. Test Rust nạp cùng trọng số, tính lại, so với tệp đã lưu. Nhanh (mili giây), chạy được trong mọi lần `cargo test`.
2. **Mô hình thật:** so trên checkpoint thật để bắt các lỗi chỉ lộ ra ở kích thước thật (tràn số, cấu hình đặc biệt, độ sâu lớn).

### 2.1 Fixture nhỏ trong đồ án

`tools/make_olmoe_reference.py` tạo hai mô hình OLMoE tí hon: `hidden_size = 16`, 2 lớp, 4 expert, chọn 2, từ vựng 32. Có hai biến thể để phủ cả hai nhánh code:

- `unnormalized-mha`: không chuẩn hóa lại top-k, 2 đầu KV (MHA).
- `normalized-gqa`: chuẩn hóa lại top-k, 1 đầu KV (GQA).

Một thủ thuật thông minh trong script: expert được lượng tử hóa INT8, rồi **trọng số của mô hình Transformers được thay bằng bản đã giải lượng tử**:

```python
q = (weight / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
weight.copy_(q.float() * scale[:, None])        # mô hình tham chiếu dùng chính trọng số đã lượng tử
```

Nhờ vậy, tham chiếu và engine dùng **cùng một bộ trọng số** về mặt toán học. Mọi chênh lệch còn lại chỉ có thể là lỗi của engine (hoặc sai số làm tròn), không lẫn với sai số lượng tử hóa.

Test tương ứng trong `tests/olmoe_reference.rs`:

```rust
for name in ["unnormalized-mha", "normalized-gqa"] {
    for cache in [0, 900, 100_000] {                  // không cache, cache rất nhỏ, cache lớn
        let mut model = OlmoeModel::load(&dir, cache, 32, 10_000_000).unwrap();
        for _ in 0..2 {                               // gọi hai lần trên cùng mô hình
            let actual = model.logits(&reference.ids).unwrap();
            let error = /* sai số tuyệt đối lớn nhất */;
            assert!(error < 2e-5, "{name}, cache={cache}, max logit error={error}");
        }
    }
}
```

Test này kiểm cùng lúc nhiều tính chất:

- **Đúng toán học** so với Transformers.
- **Cache không đổi kết quả:** dù không có cache, cache nhỏ đến mức phải loại liên tục, hay cache đủ lớn, logits phải như nhau. Cache là tối ưu hiệu năng, không được ảnh hưởng tính đúng.
- **Trạng thái không rò rỉ:** gọi lần thứ hai trên cùng mô hình (cache đã ấm, bộ đệm đã tái sử dụng) vẫn đúng.

Qwen3 có test tương tự: `tests/qwen3moe_reference.rs` với `tools/make_qwen3moe_reference.py`, kiểm GQA, `head_dim` khác `hidden/heads`, QK-norm theo đầu.

## 3. Chọn dung sai

- **Tuyệt đối:** `|a − b| < ε`. Đơn giản, hợp khi các giá trị cùng cỡ độ lớn (logits thường cỡ 1–20).
- **Tương đối:** `|a − b| < ε·|b|`. Hợp khi giá trị trải rộng nhiều bậc.
- **Kết hợp:** `|a − b| < ε_abs + ε_rel·|b|` (như `numpy.allclose`).

Chọn ε bao nhiêu? Ước lượng sai số làm tròn kỳ vọng: FP32 có sai số tương đối mỗi phép cỡ `6·10⁻⁸`; qua một tích vô hướng n phần tử, sai số tích lũy cỡ `√n·6·10⁻⁸` đến `n·6·10⁻⁸`; qua nhiều lớp, cộng dồn tiếp. Với fixture tí hon, `2·10⁻⁵` có dư địa an toàn. Với mô hình thật 48 lớp, sai số `3·10⁻⁴` là bình thường.

**Hai lỗi khi chọn dung sai:**

- Quá chặt: test thất bại ngẫu nhiên khi đổi thứ tự cộng, rồi người ta có xu hướng nới lỏng bừa.
- Quá lỏng: lỗi thật lọt qua. Một lỗi RoPE nhỏ có thể chỉ gây lệch 0,01 ở lớp đầu.

Quy tắc của đồ án (AGENTS.md): giữ nguyên dung sai đã có; không sửa giá trị mong đợi để test pass. Nếu test đỏ, tìm lỗi trước.

## 4. Kiểm thử tối ưu bằng so sánh với bản đơn giản

Mỗi lần tối ưu một hàm (SIMD, chia khối, đường tắt), giữ lại bản đơn giản và viết test so hai bản trên nhiều đầu vào. Đây là **kiểm thử vi sai** (differential testing). Một số ví dụ trong `src/olmoe.rs`:

| Test | Kiểm |
| --- | --- |
| `test_zero_alloc_topk_matches_full_sort` | top-k bằng phân hoạch khớp top-k bằng sắp xếp toàn bộ |
| `test_rope_fast_path_matches_legacy_rotate` | RoPE dùng bảng sin/cos tính sẵn khớp RoPE tính trực tiếp |
| `test_argmax_greedy_matches_legacy_closure` | argmax nhanh khớp bản cũ |
| `layer_major_prefill_matches_tokenwise_last_logits` | prefill theo lớp cho cả prompt khớp tính từng token |
| `prefix_prefill_matches_full_logits_for_every_split` | dùng lại KV tiền tố tại **mọi** điểm cắt khớp tính lại toàn bộ |
| `expert_batch_prefill_matches_tokenwise_logits_across_route_tails` | kernel nhiều token khớp kernel một token, cả khi số token không chia hết cho lô |
| `gpu_expert_compute_matches_cpu_and_skips_vram_downloads` | expert tính trên GPU khớp CPU, và đúng là không chép VRAM về RAM |
| `disk_embedding_rows_match_ram_logits_for_olmoe_and_qwen3` | đọc embedding từ tệp khớp embedding trong RAM |

Chú ý cách đặt tên: đọc tên test biết ngay **bất biến** mà code phải giữ. Và chú ý các test kiểm **biên**: "for every split", "across route tails". Lỗi hay nằm ở biên: phần dư không chia hết, điểm cắt ở vị trí 0 hoặc cuối.

## 5. Kiểm thử GPU: cẩn thận với "pass mà không chạy"

Test GPU trên máy không có GPU có thể pass vì đường code GPU bị bỏ qua và chỉ chạy CPU. Khi báo "test GPU pass", phải xác nhận test **thực sự chạy trên GPU**. Đồ án ghi trạng thái `EXECUTED` khi test CUDA thực sự chạy, và kiểm thêm các bộ đếm (số lần tính expert trên GPU > 0, số lần chép VRAM → RAM = 0) để chắc đường GPU được đi qua.

Bài học chung: **test pass không có nghĩa code được kiểm**. Kiểm tra độ phủ (coverage) hoặc thêm khẳng định rằng nhánh mong muốn thực sự chạy.

## 6. So logits trên mô hình thật lớn hơn RAM

Fixture nhỏ không bắt được mọi lỗi. Đồ án so engine với Transformers trên chính checkpoint Qwen3-30B-A3B (57 GB ở BF16) trên máy 16 GB. Vấn đề: Transformers nạp cả mô hình vào RAM.

**Giải pháp: chạy tham chiếu từng lớp** (`tools/stream_reference.py`):

1. Tính embedding cho mọi chuỗi, giữ trạng thái ẩn trong RAM.
2. Với mỗi lớp: dựng lớp decoder của Transformers trên thiết bị `meta` (chưa cấp bộ nhớ); đọc trọng số của **riêng lớp đó** từ tệp; gán vào lớp (`load_state_dict(assign=True)`); chạy lớp trên mọi chuỗi; giải phóng.
3. Áp norm cuối và LM head, ghi logits.

RAM đỉnh chỉ khoảng 2,8 GB. Kỹ thuật này dùng được cho mọi việc "kiểm một mô hình lớn hơn bộ nhớ".

**Kiểm công cụ kiểm trước khi dùng nó.** Script tham chiếu có test riêng (`tools/test_stream_reference.py`): trên mô hình ngẫu nhiên nhỏ của cả hai kiến trúc, chạy từng lớp phải khớp chạy cả mô hình (sai số 10⁻⁵). Không có bước này, nếu bản tham chiếu sai thì mọi kết luận từ nó đều vô nghĩa.

**Một lỗi gặp khi làm:** lần chạy đầu bị OOM killer dừng ở lớp 22, vì giữ handle `safe_open` mở suốt thời gian chạy làm tensor đã đọc không được giải phóng (tăng 614 MiB mỗi lớp). Sửa bằng cách mở tệp riêng cho mỗi lần đọc. Bài học: công cụ kiểm thử cũng là code, cũng có lỗi, và cũng cần được ghi nhận trung thực.

### Ba bộ logits để tách nguồn sai số

| So sánh | Đo cái gì |
| --- | --- |
| engine với Transformers dùng **cùng expert INT8** | lỗi của engine |
| Transformers INT8 với Transformers BF16 | ảnh hưởng của lượng tử hóa |
| engine với Transformers BF16 | tổng hai cái trên |

Kết quả (module 07): engine với tham chiếu INT8 trùng top-1 ở 345/345 vị trí, KL cỡ 10⁻¹¹; toàn bộ chênh lệch so với BF16 đến từ lượng tử hóa. Nếu chỉ có phép so thứ ba, không thể biết 4 vị trí đổi top-1 là do engine sai hay do INT8.

### Đánh giá vị trí đổi top-1 kèm biên

Với mỗi vị trí top-1 khác nhau, `tools/compare_logits.py` ghi **biên** của tham chiếu (hiệu hai logits lớn nhất). Biên nhỏ (dưới 0,2) nghĩa là mô hình gốc cũng gần như phân vân giữa hai token; đổi ở đó là chấp nhận được. Biên lớn mà vẫn đổi là dấu hiệu lỗi. Quy tắc trong protocol: đạt nếu top-1 trùng ở mọi vị trí có biên ≥ 0,01 và KL trung bình ≤ 10⁻⁴.

## 7. Tính tất định

Cùng binary, cùng đầu vào, cùng cấu hình thì kết quả phải giống **từng bit**. Đồ án kiểm:

- build lại binary và chạy lại: logits Qwen3 giống từng byte (so mã băm tệp logits);
- hai máy khác nhau (i5 và i7) sinh văn bản giống hệt nhau ở 8/8 request.

Tất định quan trọng vì: nó cho phép dùng "văn bản và số token khớp" làm kiểm tra nhanh trong mọi benchmark (hai cấu hình chỉ khác về hiệu năng phải cho cùng văn bản); và nó giúp tái hiện lỗi.

Những thứ phá vỡ tất định cần tránh hoặc kiểm soát: thứ tự cộng phụ thuộc lịch chạy luồng (engine cộng đầu ra expert theo thứ tự top-k cố định, không theo thứ tự xong), sampling ngẫu nhiên không cố định seed, thuật toán GPU không tất định (một số phép `atomicAdd`).

## 8. Kiểm thử hồi quy cho lỗi

Khi sửa một lỗi, viết test **tái hiện đúng điều kiện gây lỗi** trước, chạy thấy đỏ, sửa code, chạy thấy xanh. Test ở lại mãi để lỗi không quay lại. Ví dụ trong đồ án: `undersized_pinned_staging_falls_back_and_preserves_logits` kiểm trường hợp bộ đệm ghim nhỏ hơn dữ liệu, `kv_buffer_reuse_rejects_incompatible_or_oversized_donor_without_removing_id` kiểm một trường hợp biên của tái sử dụng KV.

## 9. Tự động hóa

`tools/check.py` gom các kiểm tra thành một lệnh: test Python của công cụ, kiểm tài liệu và hồ sơ bằng chứng; thêm `--rust` để chạy `cargo fmt --check`, Clippy, test Rust; `--release` để chạy test bản release; `--converter` để chạy test cần PyTorch. Trước khi commit, chạy bộ kiểm tra phù hợp và báo rõ phần nào đã chạy, phần nào bỏ qua.

---

## Bài tập

### Bài 1 ★ — allclose bằng tay

Viết `allclose(a, b, rtol, atol)` và áp dụng cho `a = [1.0, 1000.0, 1e-8]`, `b = [1.0001, 1000.1, 2e-8]` với (rtol=1e-3, atol=0), (rtol=0, atol=1e-3), (rtol=1e-3, atol=1e-6). Giải thích từng kết quả.

<details><summary>Lời giải</summary>

Điều kiện: `|a − b| ≤ atol + rtol·|b|`.

- (1e-3, 0): phần tử 3: `|1e-8 − 2e-8| = 1e-8 > 1e-3·2e-8`, thất bại. Dung sai tương đối thuần không hợp với số gần 0.
- (0, 1e-3): phần tử 2: `0,1 > 1e-3`, thất bại. Dung sai tuyệt đối thuần không hợp với số lớn.
- (1e-3, 1e-6): cả ba đạt. Kết hợp hai loại là lựa chọn an toàn.
</details>

### Bài 2 ★★ — Kiểm thử vi sai

Bạn viết một hàm softmax "nhanh" chia mảng thành khối 8 và tính max theo khối. Viết test sinh 1 000 mảng ngẫu nhiên (độ dài từ 1 đến 100, giá trị từ −100 đến 100, kể cả độ dài không chia hết cho 8 và mảng có giá trị bằng nhau) so với softmax đơn giản. Cố ý cài một lỗi ở phần dư (bỏ qua phần tử cuối khi độ dài không chia hết cho 8): test có bắt được không, và với độ dài nào?

<details><summary>Gợi ý</summary>

Lỗi chỉ lộ ra khi độ dài không chia hết cho 8 và phần tử bị bỏ là phần tử lớn. Test chỉ dùng độ dài 64 hoặc 128 sẽ không bao giờ thấy. Đây là lý do test của đồ án nói rõ "across route tails", "for every split". Có thể dùng thư viện `hypothesis` (Python) hoặc `proptest` (Rust) để sinh đầu vào và tự thu nhỏ ví dụ thất bại.
</details>

### Bài 3 ★★ — Tạo fixture

Dùng Transformers tạo một mô hình `Qwen3MoeForCausalLM` tí hon (2 lớp, `hidden_size = 32`, 4 expert, chọn 2, `head_dim = 16`, 4 đầu query, 2 đầu KV). Lưu logits cho chuỗi `[1, 5, 9, 2]`. Sau đó tự cài forward bằng numpy (dùng lại Transformer mini ở module 05, thêm MoE, GQA, QK-norm theo đầu, chuẩn hóa top-k) và so với logits đã lưu.

<details><summary>Gợi ý</summary>

Bám sát `tools/make_qwen3moe_reference.py` để biết cấu hình. Đây là bài khó nhất của khóa học nếu làm từ đầu, nhưng sau khi làm xong, bạn sẽ hiểu tường tận từng chi tiết kiến trúc. Gỡ lỗi theo lớp: so trạng thái ẩn sau embedding, sau attention lớp 0, sau MoE lớp 0... (dùng `output_hidden_states=True` của Transformers). Lớp đầu tiên lệch chỉ ra chỗ sai.
</details>

### Bài 4 ★★ — Tách nguồn sai số

Trong Transformer mini của bạn, có ba bộ logits: (A) FP32 gốc, (B) FFN lượng tử INT8 tính bằng numpy (bản "tham chiếu INT8"), (C) cùng INT8 nhưng tính bằng một hàm matvec khác viết theo kiểu 8 bộ cộng (bản "engine"). Tính KL và top-1 cho các cặp (B, C), (A, B), (A, C). Cặp nào phản ánh lỗi cài đặt, cặp nào phản ánh lượng tử hóa?

<details><summary>Lời giải</summary>

(B, C) phản ánh lỗi cài đặt; phải cỡ sai số làm tròn (KL cỡ 10⁻¹² hoặc nhỏ hơn). (A, B) phản ánh lượng tử hóa. (A, C) là tổng; nếu (B, C) gần 0 thì (A, C) gần bằng (A, B). Cố ý cài một lỗi nhỏ vào C (ví dụ quên nhân hệ số tỉ lệ ở một hàng) và quan sát cặp nào thay đổi.
</details>

### Bài 5 ★★★ — Chạy từng lớp

Với một mô hình nhỏ thật (ví dụ `Qwen/Qwen3-0.6B`), viết script chạy từng lớp kiểu `tools/stream_reference.py`: không gọi `model(...)` cả khối, mà tự gọi embedding, từng `model.model.layers[i]`, norm, LM head. Kiểm kết quả khớp `model(...)` đến 10⁻⁵. Đo RAM đỉnh của hai cách (đọc `ru_maxrss` bằng `resource.getrusage`).

<details><summary>Gợi ý</summary>

Với Transformers mới, mỗi lớp decoder cần `position_embeddings` (cos, sin của RoPE) và mặt nạ attention; lấy chúng từ `model.model.rotary_emb` và hàm tạo mặt nạ nhân quả. Đọc `tools/stream_reference.py` để xem cách đồ án làm với phiên bản 4.51.3. Bước thêm: dựng lớp trên thiết bị `meta` và chỉ nạp trọng số của lớp đó từ safetensors, để RAM đỉnh chỉ còn bằng một lớp.
</details>

### Bài 6 ★★★ — Chạy và đọc test của đồ án

Chạy `cargo test --release -- --list` để liệt kê mọi test. Chọn 5 test có tên bắt đầu bằng `test_` hoặc chứa `matches`, đọc code của chúng, và với mỗi test viết một câu: "Test này đảm bảo rằng ... không bị phá khi ...". Sau đó cố ý sửa một dòng trong code được test (ví dụ đổi `then_with(|| a.cmp(&b))` thành `b.cmp(&a)` trong top-k), chạy lại, quan sát test nào đỏ, rồi hoàn tác thay đổi.

<details><summary>Gợi ý</summary>

Kỹ thuật cố ý cài lỗi để xem test có bắt được không gọi là **mutation testing**. Nếu cài lỗi mà không test nào đỏ, phần code đó chưa được kiểm đủ. Nhớ `git checkout -- src/` để hoàn tác sau khi thử.
</details>

## Tài liệu đọc thêm

- Higham, *Accuracy and Stability of Numerical Algorithms* (chương 1–4).
- McKeeman, *Differential Testing for Software* (1998).
- Tài liệu `hypothesis` (Python) và `proptest` (Rust) về property-based testing.
- Tài liệu kiểm tra tương đương số học khi chuyển mô hình của ONNX Runtime và TensorRT.
