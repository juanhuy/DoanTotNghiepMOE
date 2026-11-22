# Học toàn bộ đồ án MoE-TierEngine

Bộ tài liệu này dạy lại mọi kiến thức nằm sau đồ án, từ ngôn ngữ Rust đến cách đo hiệu năng cho đúng. Mục tiêu có hai:

1. Hiểu đồ án đủ sâu để tự trả lời mọi câu hỏi của hội đồng, tự sửa code và tự thiết kế thí nghiệm mới.
2. Có kiến thức dùng được khi đi làm: lập trình hệ thống, backend, hạ tầng AI (ML infra / inference), tối ưu hiệu năng, GPU.

Mỗi module gồm ba phần:

- **Lý thuyết tổng quát**, viết như giáo trình và không phụ thuộc đồ án.
- **Trong đồ án**: chỉ ra đúng file, hàm trong repo đang dùng kiến thức đó.
- **Bài tập** từ dễ đến khó. Lời giải nằm trong khối `<details>` (bấm để mở); hãy tự làm trước khi mở.

Mỗi module cũng có mục **Dùng khi đi làm**, cho biết kiến thức đó xuất hiện ở công việc nào.

## Lộ trình

| # | Module | Nội dung chính | Thời gian gợi ý |
| --- | --- | --- | --- |
| 01 | [Rust cơ bản](01-rust-co-ban.md) | Ownership, borrowing, lifetime, enum/match, trait, xử lý lỗi, Cargo, test | 1 tuần |
| 02 | [Rust cho lập trình hệ thống và hiệu năng](02-rust-he-thong.md) | Bố cục bộ nhớ, slice, SIMD, Rayon, Arc/Mutex, `unsafe`, FFI, `dlopen` | 1 tuần |
| 03 | [Async, web server và streaming](03-async-web-streaming.md) | Tokio, Axum, `spawn_blocking`, Semaphore, channel có giới hạn, SSE, hủy request | 4 ngày |
| 04 | [Toán và số thực cho AI](04-toan-va-so-thuc.md) | Ma trận–vector, softmax ổn định, log-sum-exp, FP32/BF16/FP16, sai số cộng dồn | 4 ngày |
| 05 | [Transformer từ đầu](05-transformer.md) | Tokenizer, embedding, attention, RoPE, RMSNorm, GQA, KV cache, prefill/decode | 1 tuần |
| 06 | [Mixture-of-Experts](06-mixture-of-experts.md) | Router, top-k, SwiGLU, OLMoE so với Qwen3, vì sao MoE bị giới hạn bởi bộ nhớ | 3 ngày |
| 07 | [Lượng tử hóa](07-luong-tu-hoa.md) | INT8 theo hàng, INT4, sai số, đánh giá bằng logits và KL | 3 ngày |
| 08 | [Bộ nhớ và hệ điều hành](08-bo-nho-he-dieu-hanh.md) | Phân cấp bộ nhớ, roofline, page cache, RSS/anon/file, swap, OOM, cgroup v2, định dạng tệp | 1 tuần |
| 09 | [Chính sách cache](09-chinh-sach-cache.md) | LRU, LFU, SLRU, admission, prefetch, Belady, mô phỏng cache | 4 ngày |
| 10 | [GPU và CUDA](10-gpu-cuda.md) | Kiến trúc GPU, warp, coalescing, PCIe, cuBLAS, viết kernel, NVRTC | 1 tuần |
| 11 | [Đo lường hiệu năng như một nhà khoa học](11-do-luong-hieu-nang.md) | Nhiễu, trung vị, ABBA, protocol, provenance, Amdahl, profiler | 4 ngày |
| 12 | [Kiểm thử tính đúng số học](12-kiem-thu-tinh-dung.md) | Bản tham chiếu, tolerance, fixture, parity logits, tính tất định | 3 ngày |
| 13 | [Đọc hiểu MoE-TierEngine](13-doc-hieu-moe-tierengine.md) | Một request đi qua hệ thống từ đầu đến cuối, bản đồ mã nguồn | 3 ngày |
| 14 | [Dự án thực hành và bảo vệ đồ án](14-thuc-hanh-va-bao-ve.md) | 6 dự án nhỏ tự làm lại các phần cốt lõi, bộ câu hỏi hội đồng có đáp án | 2 tuần |

Tổng cộng khoảng 2–3 tháng nếu học đều mỗi ngày 2–3 giờ. Nếu cần gấp để bảo vệ, hãy đọc theo thứ tự **13 → 06 → 08 → 10 → 11 → 14** (khoảng 1 tuần), rồi quay lại các module còn lại sau.

## Bản đồ: đồ án dùng kiến thức nào

```
                 ┌─────────────── Trình duyệt (web/index.html) ───────────────┐
                 │   hội thoại, ma trận expert, số liệu từng request           │
                 └──────────────────────────┬──────────────────────────────────┘
                                            │ HTTP + SSE            (module 03)
                 ┌──────────────────────────▼──────────────────────────────────┐
                 │ src/main.rs  Axum router, SSE      src/engine.rs  Semaphore │
                 │ spawn_blocking, mpsc có giới hạn, hủy khi client ngắt       │
                 └──────────────────────────┬──────────────────────────────────┘
                                            │                       (module 01, 02)
                 ┌──────────────────────────▼──────────────────────────────────┐
                 │ src/olmoe.rs  OlmoeModel: generate → prefill / step          │
                 │  ├─ attention, RoPE, RMSNorm, KV cache        (module 05)   │
                 │  ├─ router top-k, SwiGLU expert               (module 06)   │
                 │  ├─ expert INT8, kernel AVX2                  (module 04, 07)│
                 │  ├─ cache expert theo lớp, eviction theo heat (module 09)   │
                 │  └─ RAM guard, cgroup                         (module 08)   │
                 └──────┬─────────────────────┬────────────────────────┬───────┘
                        │                     │                        │
             ┌──────────▼────────┐  ┌─────────▼──────────┐  ┌──────────▼────────┐
             │ VRAM: dense cuBLAS│  │ RAM: cache expert  │  │ SSD: safetensors  │
             │ pool + kernel INT8│  │ (64 MiB/lớp)       │  │ đọc theo offset   │
             │ src/gpu*.rs (10)  │  │                    │  │ src/safetensors.rs│
             └───────────────────┘  └────────────────────┘  └───────────────────┘

  Đo và kiểm chứng: tools/run_benchmark.py, ABBA, checksum (module 11)
                    tools/stream_reference.py, compare_logits.py (module 12)
```

## Chuẩn bị môi trường

```bash
# Rust (nếu chưa có)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup component add clippy rustfmt

# Python cho bài tập số học
python3 -m venv ~/venv-study && source ~/venv-study/bin/activate
pip install numpy matplotlib

# Kiểm tra repo build và test được
cd DoanTotNghiepMOE
cargo test --release            # chạy toàn bộ test Rust
python3 tools/check.py          # kiểm tooling và tài liệu
```

Các bài tập lập trình được làm trong thư mục riêng ngoài repo, ví dụ `~/study-ex/`. Mỗi bài tập Rust tạo bằng `cargo new ten_bai`. Không sửa code đồ án để làm bài tập, trừ khi đề bài yêu cầu đọc hoặc chạy thử.

## Cách học hiệu quả

- **Đọc lý thuyết, rồi đóng tài liệu và tự viết lại ý chính** trong 5 dòng. Không viết được nghĩa là chưa hiểu.
- **Làm bài tập trước khi mở lời giải.** Kẹt quá 30 phút mới mở, và chỉ mở đến gợi ý đầu tiên.
- **Luôn đối chiếu với code thật.** Mở file được chỉ tới, đặt breakpoint hoặc thêm `eprintln!` tạm thời (rồi xóa) để xem giá trị chạy qua.
- **Tự giải thích bằng lời** cho một người không biết gì: Router làm gì, vì sao tầng VRAM chậm hơn khi tính trên CPU... Đây cũng là cách luyện bảo vệ đồ án.

## Ký hiệu dùng chung

| Ký hiệu | Ý nghĩa | OLMoE-1B-7B | Qwen3-30B-A3B |
| --- | --- | --- | --- |
| L | số lớp | 16 | 48 |
| d | chiều ẩn | 2048 | 2048 |
| m | chiều trung gian của expert | 1024 | 768 |
| E | số expert mỗi lớp | 64 | 128 |
| k | số expert chọn cho mỗi token | 8 | 8 |
| h, h_kv | số đầu query, số đầu key/value | 16, 16 | 32, 4 |
| d_h | chiều mỗi đầu | 128 | 128 |
| V | kích thước từ vựng | 50 304 | 151 936 |

Đơn vị: GB = 10⁹ byte, GiB = 2³⁰ byte, MiB = 2²⁰ byte. Đây là chỗ hay nhầm nhất khi đọc số liệu.
