# Khung báo cáo định kỳ lần 2

Bản khung để chuyển sang LaTeX (`baocao/latex/`). Báo cáo lần 2 nối tiếp quyển lần 1: giữ cấu trúc 5 chương, sửa chương 1–4 ở các mục ghi dưới đây và viết lại chương 5 cùng phần kết luận. Mỗi số liệu đều kèm link hồ sơ; chỉ trích hồ sơ P1, còn P2 chỉ dùng cho kết luận giữ hay bỏ một hướng ([nhãn](../../benchmarks/README.md)).

## Mạch trình bày

Câu hỏi nghiên cứu giữ nguyên: *trong ngân sách bộ nhớ cho trước, cách quản lý và di chuyển expert ảnh hưởng thế nào đến độ trễ, tốc độ sinh và lượng dữ liệu phải truyền?*

Kỳ 2 trả lời theo năm bước, mỗi bước có hồ sơ riêng:

1. **Dùng VRAM chỉ làm kho expert là chi phí ròng.** Expert vẫn tính trên CPU nên phải chép qua PCIe mỗi request. Tắt tầng VRAM nhanh hơn 8/8 trên cả i5 lẫn i7 (S5, i7-r2).
2. **Nút thắt nằm ở chỗ tính, không ở chỗ chứa.** Kernel INT8 SwiGLU tính ngay trên GPU (A1) biến tầng VRAM thành có lợi: nhanh hơn 8/8 trên cả hai máy.
3. **Lợi ích lớn hơn khi RAM bị giới hạn thật.** Thí nghiệm dùng cgroup và xóa page cache để ép đọc SSD thật (ram-limit i5/i7).
4. **Chạy được model lớn hơn RAM thật.** Qwen3-30B-A3B INT8 (30GB) chạy trên máy RAM 16GB, dưới trần 8–10GiB, không OOM, trên hai máy.
5. **Engine tính đúng, và chỉ ra được nút thắt còn lại.** Parity logits với Transformers trên cả hai model thật. Với Qwen3, I/O chiếm 58–76% thời gian request; phần này dẫn sang mục giới hạn và hướng phát triển.

## Chương 1 — Tổng quan (sửa nhỏ)

- 1.4 → "Phạm vi kỳ báo cáo lần 2":
  - thêm model thứ hai Qwen3-30B-A3B;
  - thêm tính expert trên GPU;
  - thêm thí nghiệm dưới trần RAM bằng cgroup;
  - thêm kiểm tính đúng trên model thật.
- 1.5 Phương pháp: thêm quy trình chốt protocol trước khi đo, đo xen kẽ ABBA, và khóa raw bằng checksum.

## Chương 2 — Cơ sở lý thuyết (bổ sung)

- 2.3.x **Cấu hình Qwen3-MoE**, đặt cạnh mục "Cấu hình OLMoE sử dụng". Nội dung là bảng so sánh hai model: số layer × expert (16×64 so với 48×128), top-k (8 và 8), GQA, head_dim 128, QK-norm theo head, kích thước checkpoint INT8, kích thước một expert.
- 2.6.x **Tính đúng bằng so sánh logits:** top-1, top-5 overlap, KL, max |Δlogit|; vì sao cần biên top-1 khi so ở vị trí gần hòa.
- 2.6.x **Định luật Amdahl:** dùng để giới hạn lợi ích khi tăng tốc một thành phần. Dẫn sang mục 5.8.

## Chương 3 — Thiết kế (bổ sung)

- 3.x **Hỗ trợ nhiều kiến trúc:** loader chọn theo `model_type` (olmoe, qwen3_moe); dense lưu BF16; embedding đọc từ đĩa thay vì giữ trong RAM.
- 3.x **Ba chế độ GPU:** dense trên GPU; tầng VRAM expert; tính expert trên GPU. Trình bày điều kiện tự chuyển về CPU khi VRAM không đủ.
- 3.x **Giao diện giám sát:** mô tả web hiển thị những gì, và mỗi số lấy từ trường nào của API. Không minh họa bằng số giả lập.
- 3.8 Preflight và RAM guard: thêm hành vi với Qwen3, ví dụ cache 128MiB/layer bị từ chối vì dự trữ 9,94GB vượt trần 9GiB.

## Chương 4 — Thuật toán (bổ sung)

- 4.x **Kernel INT8 SwiGLU trên GPU (A1):** biên dịch NVRTC; expert đã có trong VRAM được tính tại chỗ; không tải dữ liệu từ VRAM về RAM.
- 4.x **Kernel expert nhiều token trên CPU (S2):** dùng chung một lần đọc trọng số cho 2–4 token trong prefill.
- 4.3.1 Sửa mục "Lựa chọn expert bị loại khỏi RAM": chính sách là bỏ expert có heat thấp nhất, hòa thì bỏ expert dùng lâu nhất (`min_by_key((heat, last_used))`). Đây **không phải LRU thuần**; heat giảm một nửa sau mỗi 65.536 lượt truy cập của layer.
- 4.x **Tham chiếu chạy từng layer:** cách so logits model 57GB với RAM đỉnh khoảng 2,8GB.

## Chương 5 — Thực nghiệm kỳ 2 (viết lại)

Kết quả kỳ 1 thu gọn thành một mục, kèm link quyển lần 1; không tái lập bộ đo kỳ 1.

### 5.1 Phương pháp đo kỳ 2

Trình bày các nguyên tắc đo:
- protocol được commit trước khi đo;
- đo xen kẽ ABBA và cùng binary;
- tree sạch;
- power guard;
- cgroup `memory.max` kèm xóa page cache, có xác minh bằng `fincore`;
- checksum cho raw;
- nhãn P1/P2/P3.

Bảng môi trường hai máy:

| Máy | Phần cứng chính |
| --- | --- |
| i5 | GTX 1650 4GB, RAM 16GB |
| i7 | i7-14650HX, RTX 5060 Laptop 8GB |

Hai máy dùng chung checkpoint: OLMoE kiểm hash khớp ở mọi block đo, Qwen3 khớp 478/478 file.

### 5.2 Tính đúng trên model thật (P1)

Hồ sơ: [parity 20/11](../../benchmarks/experiments/2026-11-20-real-model-parity-i5/README.md)

| So sánh | Qwen3-30B-A3B | OLMoE-1B-7B |
| --- | --- | --- |
| Engine so với Transformers, cùng expert INT8: top-1 | 159/159 | 186/186 |
| max \|Δlogit\| | 3,0e-4 | 3,5e-5 |
| KL trung bình | ~1e-11 | ~1e-11 |
| INT8 so với BF16 gốc: top-1 | 98,7% | 98,9% |
| top-5 overlap | 0,945 | 0,972 |

Diễn giải:
- Các vị trí INT8 làm đổi top-1 đều gần hòa, biên ≤ 0,19.
- Checkpoint OLMoE dùng trong toàn bộ thí nghiệm trùng 2195/2195 tensor với bản 0924-Instruct convert lại.

Giới hạn: 6 chuỗi ngắn, chạy trên CPU; không phải perplexity hay benchmark tác vụ.

### 5.3 Tầng VRAM khi expert vẫn tính trên CPU (P1)

Hồ sơ: [S3 tier 10/11](../../benchmarks/experiments/2026-11-10-gpu-expert-tier-i5/README.md), [S5 chat 10/11](../../benchmarks/experiments/2026-11-10-gpu-tier-chat-i5/README.md), [i7-r2 11/11](../../benchmarks/experiments/2026-11-11-gpu-tier-i7-r2/README.md)

| Máy | Pool VRAM | Tier off so với on | Decode trung vị (on → off) | Upload mỗi request |
| --- | --- | --- | --- | --- |
| i5 | 298/1024 (29%) | off nhanh hơn 8/8, total 1,39–1,87× | 3,13 → 5,53 tok/s | 33–41GiB |
| i7 | 845–858/1024 (83%) | off nhanh hơn 8/8, total 1,78–2,00× | 4,92 → 9,75 tok/s | 33–40GiB |

Kết luận: kể cả khi pool gần đủ model (i7), việc chép dữ liệu qua PCIe vẫn làm tầng VRAM thành chi phí ròng. Vậy vấn đề nằm ở chỗ tính.

### 5.4 Tính expert trên GPU A1 (P1)

Hồ sơ: [A1 i5 14/11](../../benchmarks/experiments/2026-11-14-gpu-compute-i5/README.md), [A1 i7 14/11](../../benchmarks/experiments/2026-11-14-gpu-compute-i7/README.md)

| Máy | Bật tier + GPU compute so với tier off | Total on/off | Decode trung vị (off → on) |
| --- | --- | --- | --- |
| i5 | nhanh hơn 8/8 | 0,743–0,813 | 8,24 → 10,73 tok/s |
| i7 | nhanh hơn 8/8 | 0,086–0,124 | 4,64 → 50,67 tok/s |

Diễn giải:
- Không còn lượt tải VRAM→RAM nào. Upload 1,8GB một lần thay cho 33–41GiB mỗi request.
- Kernel GPU lệch logits < 1e-4 so với CPU (test CUDA EXECUTED).

Lưu ý bắt buộc phải ghi: phía off của i7 đọc file chậm bất thường (3,1–3,8 GB/s), nên không trích độ lớn tỉ số i7 như mức điển hình.

### 5.5 OLMoE dưới trần RAM bằng cgroup (P1)

Hồ sơ: [ram-limit i5 16/11](../../benchmarks/experiments/2026-11-16-ram-limit-i5/README.md), [ram-limit i7 16/11](../../benchmarks/experiments/2026-11-16-ram-limit-i7/README.md)

| Máy, trần | Total on/off | Decode trung vị (off → on) | Ghi chú |
| --- | --- | --- | --- |
| i5, 5120MiB | 0,389–0,616, 8/8 | 2,14 → 4,13 tok/s | Đọc SSD thật 1,3–2,6 GB/s |
| i5, 8192MiB | không nhất quán | — | Block 0 bị nhiễu bởi app ngoài; ghi trung thực |
| i7, 5120MiB | 0,068–0,104, 8/8 | 3,83 → 50,56 tok/s | Pool 93,75%: kết quả về dung lượng VRAM |
| i7, 8192MiB | 0,137–0,204, 8/8 | 7,85 → 50,27 tok/s | |

Không có OOM hay swap. Kết luận: lợi ích của phân tầng lớn hơn khi RAM thiếu, và tỉ lệ thuận với độ phủ của pool VRAM.

### 5.6 Qwen3-30B-A3B lớn hơn RAM thật (P1)

Hồ sơ: [Qwen3 i5 17/11](../../benchmarks/experiments/2026-11-17-qwen3-ram-cap-i5/README.md), [Qwen3 i7 18/11](../../benchmarks/experiments/2026-11-18-qwen3-ram-cap-i7/README.md)

| Máy, trần cgroup | Hoàn thành | Decode trung vị | TTFT trung vị | Anon đỉnh |
| --- | --- | --- | --- | --- |
| i5, 10240MiB | 8/8 | 1,34 tok/s | 9,8s | ~5,7GiB |
| i5, 8192MiB | 8/8 | 1,13 tok/s | 11,0s | ~5,7GiB |
| i7, 10240MiB | 8/8 | 1,53 tok/s | 9,9s | ~5,7GiB |
| i7, 8192MiB | 8/8 | 1,78 tok/s | 10,9s | ~5,7GiB |

Diễn giải:
- Checkpoint 30GB, gấp đôi RAM máy.
- Anon ≈ dense BF16 2,29GiB + cache expert 2,96GiB, tức khoảng 19% kích thước checkpoint.
- Văn bản sinh ra trên hai máy khớp 8/8 turn: engine cho kết quả tất định.

Ba yếu tố làm được điều này: dense BF16, embedding đọc từ đĩa, cache expert có ngân sách.

Giới hạn:
- Một lượt mỗi mức, không A/B.
- Không so tốc độ giữa hai máy. Mức 8192 trên i7 nhanh hơn 10240 là dao động giữa các lượt.

### 5.7 Các hướng đã thử nhưng không giữ (P1/P2, kết quả âm)

Ghi kết quả âm để cho thấy không chọn lọc kết quả.

| Hướng | Kết quả | Quyết định | Hồ sơ |
| --- | --- | --- | --- |
| Prefetch theo router (S4) | TTFT on/off không nhất quán | Không bật | [S4](../../benchmarks/experiments/2026-11-10-prefill-prefetch-i5/README.md) |
| Admission cache theo heat | Payload đọc tăng 2,56% | Gỡ khỏi code | [admission](../../benchmarks/experiments/2026-11-09-cache-admission-int8-i5/README.md) |
| Tăng số thread (A3) | 6–24 thread không hơn 4 thread trên cả hai máy | Giữ 4 thread | [i5](../../benchmarks/experiments/2026-11-12-threads-i5/README.md), [i7](../../benchmarks/experiments/2026-11-13-threads-i7/README.md) |
| Cache Qwen3 64 so với 96MiB/layer | Hit 50 → 60%, payload −20%, tốc độ không cải thiện nhất quán (page cache bị thu hẹp) | Giữ 64 | [cache A/B](../../benchmarks/experiments/2026-11-18-qwen3-cache-ab-i5/README.md) |
| Kernel expert nhiều token cho prompt dài (S2) | TTFT off/on 1,065–1,094, 4/4 so sánh | **Giữ**, opt-in | [S2](../../benchmarks/experiments/2026-11-10-expert-batch-long-i5/README.md) |

### 5.8 Nút thắt còn lại và giới hạn

Lấy nội dung từ [GIOI_HAN_VA_HUONG_PHAT_TRIEN.md](GIOI_HAN_VA_HUONG_PHAT_TRIEN.md), mục 1–2:
- I/O chiếm 58–76% thời gian request; dense chiếm 12–23%.
- Theo Amdahl, tối đa nhanh khoảng 1,13–1,30× nếu chỉ tăng tốc dense.
- Qwen3 chưa dùng được GPU trên i5 vì dense F32 khoảng 4,9GB vượt 4GB VRAM.

## Kết luận và kiến nghị (viết lại)

- **Đã làm được:** năm bước của mạch trình bày, một câu cho mỗi bước, kèm số liệu chính.
- **Chưa làm được:**
  - chưa có đánh giá chất lượng tác vụ;
  - các hồ sơ Qwen3 chỉ có một lượt;
  - Qwen3 chưa dùng GPU trên i5;
  - RAM guard là kiểm tra chủ động, không phải trần cứng.
- **Hướng phát triển** (mục 3 của file giới hạn), theo thứ tự ưu tiên: VRAM làm kho expert khi dense không vừa; dense 16-bit trên GPU; expert INT4 (chỉ ở mức hướng phát triển); định tuyến có tính đến cache.

## Hình cần vẽ

Vẽ từ raw đã khóa; script đặt trong `tools/`, không sửa raw.

1. Sơ đồ ba chế độ GPU: dense / tier / compute, với luồng dữ liệu của từng chế độ.
2. Cột decode tok/s cho bốn cấu hình tier off, tier on, A1 và cgroup 5120, chia theo từng máy. Không đặt hai máy chung một trục để so.
3. Thanh chia thời gian một request Qwen3: I/O, compute expert, attention, LM head.
4. Ảnh chụp web: ma trận expert với VRAM/RAM, ô "Trần RAM guard".

## Không đưa vào báo cáo

Bảng lịch sử trong [EVIDENCE_MATRIX](../../docs/thesis/EVIDENCE_MATRIX.md) còn các dòng thiếu hồ sơ đo đủ provenance. Những dòng sau **không trích số liệu**:
- INT4 so với INT8 ("+41,8% thông lượng", "PPL 103 → 114");
- hệ số Gini và hot pinning;
- KV FP8 ("giảm 74,95%", "4,07 tok/s");
- `.moekv` ("giảm 58,7% prefill");
- "Routing momentum EMA".

Nếu cần, chỉ mô tả chúng là tính năng có trong code.

Các lời cũng không được nói:
- "zero-copy", "hết OOM", "hard cap RAM";
- so tốc độ i5 với i7;
- cộng dồn phần trăm của nhiều thí nghiệm khác nhau;
- gọi chính sách eviction là LRU.

## Việc còn lại trước khi nộp

- [ ] Chuyển chương 5, kết luận và các mục bổ sung ở chương 1–4 sang LaTeX; cập nhật `metadata.tex` thành báo cáo lần 2.
- [ ] Vẽ 4 hình ở trên.
- [ ] Biên dịch bằng `bash baocao/latex/compile.sh`; kiểm mục lục, danh mục bảng và danh mục hình.
- [ ] Đối chiếu từng số trong bảng với README hồ sơ, đọc lại một lượt.
- [ ] Chạy thử demo theo kịch bản: `tools/demo.sh olmoe-gpu` rồi `tools/demo.sh qwen3`.
