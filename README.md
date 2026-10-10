# MoE-TierEngine: Hệ Thống Suy Luận Mô Hình Sparse MoE Đa Tầng Cho Thiết Bị Biên

[![Rust](https://img.shields.io/badge/Rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-36%20passed-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/Model-OLMoE--1B--7B-purple.svg)](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
[![Paradigm](https://img.shields.io/badge/Methodology-AI--First%20Engineering-black.svg)]()

> **Đề tài Đồ án Tốt nghiệp & Nghiên cứu Khoa học**:  
> *"Nghiên cứu, Thiết kế và Hiện thực hóa Hệ thống Suy luận Tối ưu cho Mô hình Ngôn ngữ Lớn Kiến trúc Sparse Mixture-of-Experts trên Phần cứng Giới hạn."*

---

## 1. Giới Thiệu & Đặt Vấn Đề

Các mô hình ngôn ngữ lớn kiến trúc **Sparse Mixture-of-Experts (MoE)** như `OLMoE-1B-7B` (64 experts, top-8 active) sở hữu tổng dung lượng tri thức lớn (7B tham số) nhưng chi phí tính toán mỗi token chỉ tương đương mô hình 1B tham số. Tuy nhiên, rào cản lớn nhất khi triển khai trên phần cứng tiêu dùng (Consumer Laptops/PCs với RAM 16 GB, không có GPU máy chủ đắt tiền) là **nút thắt bộ nhớ (Memory Wall)**: toàn bộ 7B tham số không thể nằm trọn trong VRAM/RAM.

**MoE-TierEngine** là inference engine viết hoàn toàn bằng **Rust từ con số 0**, giải quyết bài toán này thông qua kiến trúc **Bộ nhớ Phân tầng (Tiered-Memory Architecture)** kết hợp các thuật toán tối ưu hóa cấp hệ thống:

```
                          ┌────────────────────────────────────────────────────────┐
                          │               MoE-TierEngine Architecture              │
                          └────────────────────────────────────────────────────────┘
                                                       │
               ┌───────────────────────────────────────┼────────────────────────────────────────┐
               ▼                                       ▼                                        ▼
   ┌───────────────────────┐               ┌───────────────────────┐                ┌───────────────────────┐
   │    RAM TIER (FP32)    │               │  CACHE TIER (INT8)    │                │   STORAGE TIER (SSD)  │
   │───────────────────────│               │───────────────────────│                │───────────────────────│
   │ • Token Embeddings    │               │ • 128 MiB/layer Cache │                │ • Safetensors Shards  │
   │ • Attention (Q/K/V/O) │◄─────────────►│ • Top-K Active Set    │◄──────────────►│ • 64 Experts/layer    │
   │ • Router Gate         │               │ • Heat/Aging Policy   │   (On-Demand   │ • Flat INT8 SwiGLU    │
   │ • RMSNorm & LM Head   │               │ • Session KV Cache    │   File Read)   │ • Per-row FP32 Scales │
   │ • Zero-Alloc Scratch  │               │   (256 MiB Prefix)    │                │                       │
   └───────────────────────┘               └───────────────────────┘                └───────────────────────┘
               │                                       │                                        │
               └───────────────────────────────────────┼────────────────────────────────────────┘
                                                       ▼
                          ┌────────────────────────────────────────────────────────┐
                          │               EXECUTION & COMPUTE ENGINE               │
                          │────────────────────────────────────────────────────────│
                          │ • Prefill: Layer-major + Expert-batched grouping       │
                          │ • Decode: Autoregressive greedy (Rayon worker pool)    │
                          │ • Matvec: INT8 row-scaled + 8-accumulator unrolling    │
                          │ • Serving: Axum HTTP, SSE Streaming, Co-op Cancel      │
                          └────────────────────────────────────────────────────────┘
```

---

## 2. Các Điểm Sáng Kỹ Thuật Cốt Lõi

1. **Bộ nhớ Phân tầng Thông minh (Hierarchical Tiered Memory)**:
   - **Dense Weights** (Attention, Router, RMSNorm, LM Head) giữ thường trực trong RAM (~3.0 GiB FP32).
   - **Sparse Experts** (INT8) lưu trữ dưới dạng Safetensors shards trên SSD; đọc on-demand theo byte offset không nạp toàn bộ.
   - **Expert Cache** (128 MiB/layer): Quản lý theo chính sách kết hợp **Tần suất kích hoạt (Heat) + Lão hóa (Aging) + Thời điểm truy cập (LRU)**.
   - **Session KV Cache**: Tái sử dụng tiền tố hội thoại (Prefix matching) lên tới 256 MiB.
2. **Layer-Major Expert-Batched Prefill**:
   - Chạy prefill theo từng layer và gom toàn bộ token prompt có chung router target lại. Mỗi expert chỉ cần nạp từ đĩa đúng 1 lần cho cả chuỗi prompt tại layer đó, triệt tiêu hiện tượng I/O thrashing.
3. **Hiệu năng Cấp Vi kiến trúc (Micro-architectural Efficiency)**:
   - Inference hot-loop đạt tiêu chí **Zero-Allocation**: Tái sử dụng vùng đệm tĩnh (`StepScratch`, `ExpertScratch`).
   - Matvec INT8 SwiGLU và FP32 dùng **8 bộ cộng độc lập** để LLVM tự động vectorize mã máy x86_64.
4. **Độ Tin Cậy Số Học Tuyệt Đối (Numerical Parity)**:
   - Đối chiếu sai số logits so với PyTorch Transformers 4.51.3 ở ngưỡng tuyệt đối $< 2 \times 10^{-5}$.
   - Vượt qua 36/36 unit, integration và benchmark tests.

---

## 3. Bằng Chứng Thực Nghiệm (Empirical Evidence)

Dự án sở hữu kho lưu trữ bằng chứng thực nghiệm gồm **30+ tệp benchmark JSON thô** và bảng tổng hợp:

* 📊 **Báo cáo Thực nghiệm Chi tiết**: Xem [`docs/BENCHMARKS.md`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/docs/BENCHMARKS.md)
* 📋 **Bảng Ma trận Bằng chứng Luận văn**: Xem [`docs/thesis/EVIDENCE_MATRIX.md`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/docs/thesis/EVIDENCE_MATRIX.md)
* 📑 **Đề cương Luận văn Tốt nghiệp**: Xem [`docs/thesis/OUTLINE.md`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/docs/thesis/OUTLINE.md)
* 📝 **Nhật ký Nâng cấp & Kỷ luật Kỹ thuật**: Xem [`docs/PROGRESS.md`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/docs/PROGRESS.md)

### Tóm tắt Một số Mốc Đột phá:
* **Tối ưu Bỏ Revalidation Dense**: Giảm TTFT từ `22.52s` xuống `19.21s` (-14.7%), tổng thời gian giảm 15.7%.
* **Kích thước Cache 128 MiB/layer**: Giảm tỷ lệ đọc SSD từ 18.6 GB xuống tiệm cận 0 byte ở warm runs.
* **Quy mô Luồng Rayon**: 8 threads đạt điểm ngọt hiệu năng/băng thông bộ nhớ trên CPU DDR5.

---

## 4. Hướng Dẫn Cài Đặt & Chạy Thử

### Yêu cầu Tiên quyết
* **Hệ điều hành**: Windows 10/11 hoặc Linux x86_64.
* **Rust**: 1.85 trở lên (`rustup default stable`).
* **RAM**: Khuyến nghị $\ge 16$ GB (hoặc 8 GB nếu cấu hình cache 64 MiB/layer).
* **Bộ nhớ trống**: $\ge 10$ GB SSD cho checkpoint INT8.

### Khởi chạy Server Phục vụ (HTTP & Web UI)
```bash
# Thiết lập biến môi trường và chạy binary release
MOE_MODEL_DIR=./models/olmoe-1b-7b-int8 \
MOE_BIND=127.0.0.1:8081 \
MOE_CONTEXT=512 \
MOE_CACHE_BYTES_PER_LAYER=134217728 \
RAYON_NUM_THREADS=8 \
cargo run --release --bin moe-tier-engine
```

Mở trình duyệt tại `http://127.0.0.1:8081` để tương tác trực tiếp qua Web UI nhúng sẵn (hỗ trợ SSE streaming, đo TTFT, tokens/s).

### Gọi API chuẩn OpenAI
```bash
curl -sS http://127.0.0.1:8081/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "olmoe",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 16,
    "stream": true
  }'
```

### Chạy Kiểm thử & Tái hiện Bằng chứng Benchmark
```bash
# 1. Chạy toàn bộ test suites
cargo test --release

# 2. Sinh lại bảng tổng hợp báo cáo bằng chứng từ các file JSON
python tools/export_benchmark_evidence.py
```

---

## 5. Quy Trình Kỹ Thuật AI-First (AI-First Squad)

Dự án áp dụng quy trình phát triển **AI-First Software Engineering** nghiêm ngặt với đội ngũ 5 chuyên gia đại diện (Subagents) tại [`.agents/plugins/moe-squad`](file:///c:/Users/kasiz/Documents/Studying/DoanTotNghiepMOE/.agents/plugins/moe-squad):

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ 0. Orchestrator │ ───►  │  1. Architect   │ ───►  │ 2. Rust-Engineer│
│ (Điều phối lộ   │       │  (Đặc tả kỹ     │       │ (Hiện thực code │
│  trình & log)   │       │   thuật & spec) │       │  zero-alloc)    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
         ▲                                                   │
         │                                                   ▼
┌─────────────────┐                                 ┌─────────────────┐
│   4. Reviewer   │ ◄────────────────────────────── │3. Profiler-Test │
│ (Anti-sycophancy│                                 │ (Đo TTFT, tokens│
│  & audit code)  │                                 │  /s, xuất JSON) │
└─────────────────┘                                 └─────────────────┘
```

1. **Không Vibe-Coding**: Mọi tính năng phải có đặc tả rõ ràng về bộ nhớ, bounds, và luồng dữ liệu trước khi viết mã.
2. **Kỷ luật TDD**: Viết test/fixture thất bại trước (Red), code pass (Green), rồi mới tối ưu (Refactor).
3. **Đánh giá Phản biện (Anti-Sycophancy)**: Dám ghi nhận các tối ưu lý thuyết thất bại (ví dụ: Online Softmax chậm hơn Materialized Score) để giữ tính trung thực học thuật.
4. **Bằng chứng Thống nhất**: Mọi số đo đều phải lưu thành file JSON có timestamp và cấu hình phần cứng đi kèm.

---

## 6. Cấu Trúc Thư Mục Dự Án

```text
DoanTotNghiepMOE/
├── .agents/                    # Cấu hình AI-First Squad, Skills và Rules
│   ├── plugins/moe-squad/      # Định nghĩa 5 chuyên gia AI
│   └── rules/                  # Tiêu chuẩn kỹ thuật Rust và ghi chép tiến độ
├── benchmarks/                 # Hệ thống bằng chứng thực nghiệm
│   ├── reports/                # Bảng biểu CSV & Markdown tự động tổng hợp
│   ├── results/                # 30+ tệp JSON chứa số liệu benchmark thô
│   └── suites/                 # Bộ đề kịch bản đo (suite-v1, long-context...)
├── docs/                       # Tài liệu nghiên cứu khoa học & đồ án
│   ├── architecture/           # Sơ đồ và đặc tả kiến trúc hệ thống
│   ├── thesis/                 # Đề cương luận văn và ma trận bằng chứng
│   ├── BENCHMARKS.md           # Báo cáo tổng hợp số liệu đo kiểm
│   └── PROGRESS.md             # Nhật ký kỹ thuật chi tiết qua từng bước
├── models/                     # Thư mục chứa checkpoints (Safetensors)
├── src/                        # Toàn bộ mã nguồn Engine viết bằng Rust
│   ├── backend.rs              # CPU kernels (FP32 Matvec, Attention)
│   ├── int8_expert.rs          # INT8 SwiGLU Expert compute & loading
│   ├── olmoe.rs                # OLMoE decoder, prefill, KV & Expert cache
│   ├── safetensors.rs          # Reader Safetensors on-demand
│   └── main.rs                 # Axum HTTP server & Web UI
├── tests/                      # Bộ fixture so khớp số học với PyTorch
├── tools/                      # Bộ script chuẩn bị model và xuất bằng chứng
└── archive/                    # Lưu trữ các phiên bản tài liệu lịch sử
```

---

## Giấy Phép & Tác Quyền
Phát hành theo giấy phép **MIT License**. Toàn bộ mã nguồn và dữ liệu thực nghiệm được phát triển phục vụ mục đích nghiên cứu học thuật và đồ án tốt nghiệp đại học.
