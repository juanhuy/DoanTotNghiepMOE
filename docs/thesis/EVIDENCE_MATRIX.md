# Ma Trận Bằng Chứng Thực Nghiệm (Empirical Evidence Matrix)

Tài liệu này cung cấp ánh xạ đối chiếu trực tiếp giữa **Các Luận điểm Khoa học (Thesis Claims)** trong báo cáo đồ án với **Mã nguồn (Source Code)** và **Dữ liệu Đo lường Thô (Raw Benchmark Evidence)**. Hội đồng đánh giá có thể kiểm chứng độc lập bất kỳ luận điểm nào.

---

| STT | Luận điểm Khoa học (Scientific Claim) | Mã nguồn Hiện thực (Source Code) | Dữ liệu Thô Kiểm chứng (Raw JSON Artifact) | Kết quả Đo lường & Bằng chứng Cụ thể |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Tính đúng đắn số học so với PyTorch**<br>Engine Rust đạt kết quả tương đương 100% về logits với Hugging Face Transformers. | [`tests/olmoe_reference.rs`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/tests/olmoe_reference.rs)<br>[`src/olmoe.rs`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/src/olmoe.rs) | [`tests/fixtures/olmoe/`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/tests/fixtures/olmoe) | Sai số tuyệt đối tối đa giữa logits Rust và PyTorch `< 2e-5` trên cả MHA và GQA. Đạt 36/36 unit/integration tests. |
| **2** | **Loại bỏ xác thực lặp lại giúp tăng tốc 15%**<br>Bỏ quét bounds/finite lặp lại trong hot loop giúp giảm mạnh TTFT. | [`src/backend.rs`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/src/backend.rs#L69-L89) (`multiply_loaded_into`) | [`2026-09-22-skip-dense-revalidation.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-22-skip-dense-revalidation.json) | TTFT giảm từ 22.52s xuống 19.21s (-14.7%). Tổng thời gian sinh giảm từ 29.86s xuống 25.16s (-15.7%). |
| **3** | **Cơ chế Cache 128 MiB chống Cache Thrashing**<br>Ngân sách 32 MiB/layer không đủ active set gây thrashing; 128 MiB/layer giải quyết triệt để. | [`src/olmoe.rs`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/src/olmoe.rs#L491-L519) (`admit`, `victim`) | [`2026-09-22-baseline.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-22-baseline.json)<br>[`2026-09-22-cache-128mib.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-22-cache-128mib.json) | Cache 32 MiB đọc 18.6 GB từ SSD với 2878 lần eviction. Cache 128 MiB (~21 experts/layer) giữ trọn vẹn active set, giảm eviction về 0 ở warm runs. |
| **4** | **Layer-Major Expert-Batched Prefill**<br>Gom prompt tokens theo expert giúp mỗi expert chỉ đọc từ SSD 1 lần mỗi layer. | [`src/olmoe.rs`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/src/olmoe.rs#L531-L720) (`prefill`) | [`2026-09-23-batched-prefill-context.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-23-batched-prefill-context.json) | Số lần nạp expert từ đĩa giảm tỷ lệ thuận với độ dài prompt, duy trì thông lượng ổn định trên ngữ cảnh dài. |
| **5** | **Bão hòa Băng thông RAM khi mở rộng Luồng**<br>Tăng số worker Rayon quá 8 luồng không tăng tốc do nghẽn băng thông bộ nhớ. | [`src/backend.rs`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/src/backend.rs#L10-L36) (`configure_parallelism`) | [`2026-09-23-rayon-1-context.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-23-rayon-1-context.json)<br>[`2026-09-23-rayon-8-context.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-23-rayon-8-context.json)<br>[`2026-09-23-rayon-24-context.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-23-rayon-24-context.json) | 8 threads đạt điểm ngọt hiệu năng/công suất; 24 threads không cải thiện thêm do băng thông RAM DDR5 đạt ngưỡng bão hòa. |
| **6** | **Thực nghiệm Phản biện A/B: Attention Kernel**<br>Online Softmax lý thuyết chậm hơn Materialized Score có scratch tái sử dụng trên CPU. | [`src/backend.rs`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/src/backend.rs#L125-L210) | [`2026-09-24-attention-online-vs-materialized.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-24-attention-online-vs-materialized.json) | Microbenchmark A/B: Online Softmax mất 1.008 ms; Materialized Score chỉ mất 0.448 ms (nhanh hơn 2.25x do tránh rescale FP32 lặp lại). |
| **7** | **Khả năng chịu tải & Phục vụ Streaming**<br>Hệ thống phục vụ SSE streaming, hỗ trợ cooperative cancellation khi client ngắt kết nối. | [`src/main.rs`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/src/main.rs)<br>[`tools/test_streaming_smoke.py`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/tools/test_streaming_smoke.py) | [`2026-09-23-streaming-smoke.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-23-streaming-smoke.json)<br>[`2026-09-23-http-stress-fixture.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-23-http-stress-fixture.json) | Vượt qua stress test 50/50 HTTP requests; xử lý ngắt kết nối an toàn mà không rò rỉ bộ đệm hay treo worker thread. |
| **8** | **Chất lượng Suy luận Ngôn ngữ (Quality Smoke)**<br>Mô hình INT8 tạo câu trả lời đúng trên bộ 12 tác vụ deterministic. | [`benchmarks/suites/quality-v1.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/suites/quality-v1.json) | [`2026-09-23-quality-v1.json`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/benchmarks/results/2026-09-23-quality-v1.json) | Đạt 12/12 câu kiểm thử về suy luận logic, toán học, hiểu tiếng Việt và trích xuất ngữ cảnh. |

---

## Hướng dẫn Tái hiện Bằng chứng (Reproducibility Guide)

Để tái hiện lại toàn bộ bảng số liệu trên:
```bash
# 1. Chạy toàn bộ test suites kiểm chứng số học
cargo test --release

# 2. Sinh lại bảng tổng hợp báo cáo bằng chứng từ các file JSON
python tools/export_benchmark_evidence.py

# 3. Chạy một bài benchmark đo đạc mới
RAYON_NUM_THREADS=8 cargo run --release --example benchmark_suite -- \
  models/olmoe-1b-7b-int8 1 benchmarks/suites/suite-v1.json near-context-retrieval
```
