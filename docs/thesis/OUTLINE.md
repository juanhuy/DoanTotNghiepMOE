# Đề Cương Chi Tiết Đồ Án Tốt Nghiệp / Bài Báo NCKH

**Đề tài**: Nghiên cứu, Thiết kế và Hiện thực hóa Hệ thống Suy luận Tối ưu cho Mô hình Ngôn ngữ Lớn Kiến trúc Sparse Mixture-of-Experts trên Phần cứng Giới hạn (Tiered-Memory MoE Inference Engine)  
**Tên hệ thống**: `MoE-TierEngine`  
**Ngôn ngữ & Công nghệ**: Rust (Systems Programming, Zero-Allocation, Rayon Parallelism, Safe Concurrency)  
**Mô hình thực nghiệm**: OLMoE-1B-7B (64 Experts, Top-8 Active Routing)

---

## CHƯƠNG 1: MỞ ĐẦU & BỐI CẢNH NGHIÊN CỨU (INTRODUCTION)

### 1.1. Bối cảnh & Thách thức của Mô hình Ngôn ngữ Lớn (LLM)
- Sự bùng nổ của mô hình ngôn ngữ lớn và rào cản tài nguyên phần cứng (Memory Wall / VRAM Bottleneck).
- Nghịch lý triển khai LLM tại biên (Edge / Consumer Devices): Mô hình Dense đòi hỏi chi phí tính toán tỷ lệ thuận với số lượng tham số, trong khi người dùng cá nhân và doanh nghiệp vừa/nhỏ chỉ sở hữu laptop/PC phổ thông (RAM 16 GB, GPU 4-8 GB VRAM).

### 1.2. Kiến trúc Sparse Mixture-of-Experts (MoE) — Cơ hội và Thách thức
- **Cơ hội**: MoE cho phép mở rộng dung lượng tri thức (ví dụ: 7B tham số) nhưng chỉ kích hoạt một phần nhỏ tham số cho mỗi token (ví dụ: 1B active parameters), duy trì chi phí FLOPs thấp.
- **Thách thức cốt lõi**: Tổng trọng số vẫn phải lưu trữ toàn bộ. Khi không đủ VRAM/RAM, hệ thống gặp hiện tượng **I/O Thrashing** — hoán đổi trọng số liên tục qua bus PCIe/SSD, làm tốc độ suy luận giảm nghiêm trọng.

### 1.3. Mục tiêu & Nhiệm vụ Nghiên cứu của Đề tài
- Thiết kế hệ thống suy luận phân tầng bộ nhớ (Tiered Memory: VRAM/RAM/SSD) chuyên biệt cho MoE.
- Đảm bảo tính toán số học chính xác (Numerical Parity) so với framework tham chiếu chuẩn (PyTorch / Hugging Face).
- Tối ưu hóa toàn diện: Layer-major Prefill gom Expert, Lượng tử hóa INT8 SwiGLU, Cache thông minh (Heat + Aging + Recency).

### 1.4. Đóng góp Chính của Đồ án (Main Contributions)
1. Xây dựng một **Inference Engine hoàn chỉnh từ con số 0 bằng Rust** với hiệu năng cao, zero-allocation trong inference hot-loop.
2. Thuật toán **Layer-Major Expert-Batched Prefill** giúp loại bỏ I/O thrashing trong xử lý chuỗi prompt dài.
3. Chính sách bộ đệm **Heat-Aging Expert Cache** kết hợp **Session KV Prefix Cache** tối ưu cho hội thoại nhiều lượt.
4. Bộ công cụ và dữ liệu thực nghiệm chuẩn mực (**Empirical Evidence Suite**) gồm hơn 30 benchmark thô, đối chiếu A/B khoa học.

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & TỔNG QUAN TÀI LIỆU (RELATED WORK)

### 2.1. Cấu trúc Mô hình Sparse MoE & Transformer Decoder
- Cơ chế Causal Self-Attention (Multi-Head Attention - MHA, Grouped-Query Attention - GQA).
- Rotary Position Embedding (RoPE) và Root Mean Square Normalization (RMSNorm).
- Sparse Gating (Router Softmax, Top-K Selection, Renormalization).
- Khối FFN SwiGLU kích hoạt chuyên gia (`gate_proj`, `up_proj`, `down_proj`).

### 2.2. Kỹ thuật Lượng tử hóa Trọng số (Weight Quantization)
- Lượng tử hóa số nguyên INT8 đối xứng theo hàng (Per-row Scale).
- So sánh đánh đổi giữa độ chính xác số học (Perplexity / Logits) và dung lượng bộ nhớ.

### 2.3. Khảo sát các Hệ thống Suy luận & Cơ chế Offloading Hiện có
- **vLLM / Hugging Face Accelerate**: Giới hạn khi offload CPU/Disk do overhead của Python runtime.
- **llama.cpp**: Engine C++ phổ biến cho dense models, nhưng cơ chế nạp MoE chưa tối ưu sâu cho mô hình bộ nhớ phân tầng cấp layer.
- **MoE-Infinity / EdgeMoE / FMoE**: Các nghiên cứu hàn lâm kinh điển về MoE offloading trên cụm máy chủ và thiết bị biên.

---

## CHƯƠNG 3: THIẾT KẾ KIẾN TRÚC HỆ THỐNG MOE-TIERENGINE

### 3.1. Tổng quan Kiến trúc Đa tầng (Tiered Storage & Memory Layout)
- **RAM Tier (FP32)**: Embeddings, Attention (Q/K/V/O), Router Gate, RMSNorm, LM Head, Activation Buffers.
- **Cache Tier (INT8)**: Bộ đệm Expert Cache có giới hạn ngân sách (128 MiB/layer), Session KV Cache (256 MiB).
- **Secondary Storage Tier (SSD NVMe)**: Trọng số merged INT8 SwiGLU lưu dưới định dạng Safetensors, mở sẵn file handle, đọc trực tiếp theo byte offset không qua nạp toàn bộ.

### 3.2. Thuật toán Bộ đệm Thông minh (Expert Cache Policy)
- Cơ chế tính độ nóng (`heat`) và độ già hóa (`aging` chu kỳ $2^{16}$ step).
- Tiêu chí loại trừ (Eviction Criterion): $\min(\text{heat}, \text{last\_used})$.
- Ngưỡng tiếp nhận (Admission Control): Ngăn chặn ô nhiễm cache từ các expert "lạnh".

### 3.3. Thuật toán Prefill: Layer-Major Expert-Batched
- So sánh Token-Major vs Layer-Major Prefill.
- Kỹ thuật gom token theo Expert Target: Giảm số lần đọc đĩa của mỗi expert xuống còn 1 lần duy nhất cho mỗi layer trên toàn bộ prompt.

### 3.4. Kiến trúc Phục vụ (Serving & Streaming Engine)
- API chuẩn OpenAI Chat Completions dựa trên Axum.
- Server-Sent Events (SSE) Streaming.
- Cơ chế Hủy bỏ Hợp tác (Cooperative Cancellation) giữa các layer/expert khi client ngắt kết nối.

---

## CHƯƠNG 4: HIỆN THỰC HÓA & KỸ THUẬT TỐI ƯU HỆ THỐNG (IMPLEMENTATION)

### 4.1. Kỷ luật Lập trình Hệ thống bằng Rust
- Nguyên tắc Zero-Allocation trong inference loop: Phân bổ tĩnh `StepScratch`, `ExpertScratch`.
- Kiểm soát đa luồng với Rayon: Điều phối số worker tối ưu (8 threads) để tránh bão hòa băng thông RAM.

### 4.2. Tối ưu hóa Mức Vi kiến trúc (Micro-architectural Optimizations)
- Kỹ thuật tách chuỗi phụ thuộc lệnh: Sử dụng 8 bộ cộng độc lập (`sums = [0.0f32; 8]`) cho phép LLVM tự động sinh mã vector hóa SIMD.
- Bỏ xác thực tensor lặp lại trong hot-path (`multiply_loaded_into`).

### 4.3. Thiết kế & Thực nghiệm Causal Attention Kernel
- Blocked Causal Attention (tile 32 positions).
- Thực nghiệm A/B Online Softmax vs Materialized Score: Phân tích nguyên nhân và quyết định kiến trúc.

---

## CHƯƠNG 5: THỰC NGHIỆM, KẾT QUẢ & PHÂN TÍCH ĐÁNH GIÁ (EVALUATION)

### 5.1. Thiết lập Môi trường Đo kiểm & Bộ Benchmark
- Môi trường phần cứng: Intel Core i7-14650HX (16 Cores), 16 GB DDR5, SSD NVMe.
- Các bộ kịch bản đo: `suite-v1`, `suite-v2-long-context`, `quality-v1`.

### 5.2. Minh chứng 1: Độ Chính xác Số học (Numerical Parity)
- So khớp từng logits với PyTorch Transformers 4.51.3 trên cả MHA và GQA (Sai số $< 2 \times 10^{-5}$).
- Đánh giá chất lượng sinh nội dung với bộ smoke test 12 câu đa lĩnh vực.

### 5.3. Minh chứng 2: Tối ưu Hóa Tốc độ & Phân rã Độ trễ (Latency Breakdown)
- Tốc độ Time-to-First-Token (TTFT) và Decode (tokens/s).
- Biểu đồ phân rã thời gian: Attention vs Expert Compute vs Expert I/O vs LM Head.

### 5.4. Minh chứng 3: Hiệu quả Quản trị Bộ nhớ & Cache Hit-Rate
- Đo lường Peak RSS trong các kịch bản tải thực tế.
- Tác động của kích thước cache (32 MiB vs 128 MiB/layer) đối với số byte đọc từ SSD và hiện tượng cache thrashing.

### 5.5. Minh chứng 4: Phân tích Đánh đổi A/B & Khả năng Mở rộng Luồng
- Benchmark mở rộng số luồng Rayon: 1 vs 8 vs 24 threads (giới hạn băng thông RAM).
- Nghiên cứu so sánh A/B: Online Softmax vs Materialized Attention.

---

## CHƯƠNG 6: KẾT LUẬN & HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)

### 6.1. Tổng kết Kết quả Đạt được
- Đã xây dựng thành công một MoE Engine hoàn chỉnh, ổn định, chạy được mô hình 7B trên máy tính phổ thông với RAM thực tế chỉ ~4 GB.
- Hệ thống đạt độ tin cậy khoa học cao với quy trình ghi chép và thực nghiệm chuẩn chỉnh.

### 6.2. Hướng phát triển Đột phá cho Bài báo Khoa học
- **Predictive Expert Pre-fetching**: Dự đoán routing layer kế tiếp để ẩn độ trễ I/O bằng async I/O.
- **Hybrid 3-Tier Architecture**: Tận dụng VRAM của GPU phổ thông (GTX 1650 4GB / RTX 5060 8GB) để chứa Dense weights và Hot Experts.
- **Intel VNNI SIMD Intrinsics**: Tăng tốc tính toán INT8 ở cấp độ phần cứng.
