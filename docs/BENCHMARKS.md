# Báo Cáo Đo Lường & Bằng Chứng Thực Nghiệm Đa Phần Cứng

> **Tài liệu Bằng chứng Thực nghiệm cho Đồ án Tốt nghiệp & Nghiên cứu Khoa học**  
> Tự động trích xuất từ **31 tệp tin benchmark JSON thô** phân loại theo 2 Testbed.

## 1. Môi Trường Thực Nghiệm Của 2 Thành Viên (Hardware Testbeds)

| Thông số Phần cứng | Testbed A (Thành viên 1) | Testbed B (Thành viên 2) |
| :--- | :--- | :--- |
| **Vi xử lý (CPU)** | Intel Core i7-14650HX (16 Cores / 24 Threads) | Intel Core i5 (6 Cores / 12 Threads) |
| **Bộ nhớ trong (RAM)** | 16 GB DDR5 (~60 GB/s) | 16 GB DDR4 (~25-30 GB/s) |
| **Card đồ họa (GPU)** | NVIDIA GeForce RTX 5060 Laptop (8 GB GDDR6) | NVIDIA GeForce GTX 1650 Max-Q (4 GB GDDR5) |
| **Ổ cứng lưu trữ** | SSD NVMe PCIe 4.0 | SSD NVMe PCIe 3.0 |
| **Luồng song song (`RAYON_NUM_THREADS`)** | **8 Threads** | **4 Threads** |
| **Thư mục lưu kết quả** | `benchmarks/results/testbed_a_i7_14650hx/` | `benchmarks/results/testbed_b_i5_10400h/` |

## 2. Bảng So Sánh Đối Đầu (Cross-Hardware Side-by-Side Comparison)

So sánh trực diện hiệu năng khi chạy cùng kịch bản (`What is 2 + 2?`, Cache 128 MiB/layer, Max 16 tokens):

| Chỉ số Đánh giá | Testbed A (i7-14650HX) Cold | Testbed B (Core i5) Cold | Testbed A Warm | Testbed B Warm |
| :--- | ---: | ---: | ---: | ---: |
| **Time to First Token (TTFT)** | 13.9 s | **8.83 s** | 13.89 s | **4.9 s** |
| **Tốc độ Decode (tokens/s)** | 1.4 | **1.85** | 1.28 | **2.31** |
| **Tổng thời gian sinh token** | 18.18 s | **16.95 s** | 18.59 s | **11.39 s** |
| **Thời gian Causal Attention** | 3.01 s | 1.98 s | 3.04 s | 1.46 s |
| **Thời gian Tính Expert** | 8.61 s | 2.89 s | 8.85 s | 2.43 s |
| **Thời gian Đọc SSD (I/O)** | 5.41 s | 11.27 s | 5.55 s | 6.84 s |
| **Tỷ lệ Cache Hit (%)** | 60.97% | **68.69%** | 64.38% | **76.39%** |
| **Dung lượng đọc từ SSD** | 7.34 GB | 8.71 GB | 6.7 GB | 6.57 GB |

## 3. Nhật Ký Kết Quả Đo Lường Chi Tiết: Testbed B (Máy Core i5)

Thư mục nguồn: `benchmarks/results/testbed_b_i5_10400h/`

| Tệp Benchmark | State | TTFT (s) | Decode (tok/s) | Tổng (s) | Attention (s) | Expert Compute (s) | Expert I/O (s) | Cache Hit % | SSD Đọc (GB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2026-10-10-testbed-i5-10400h` | engine-cold | 8.83 | 1.85 | 16.95 | 1.98 | 2.89 | 11.27 | 68.69% | 8.71 |
| `2026-10-10-testbed-i5-10400h` | engine-warm | 4.9 | 2.31 | 11.39 | 1.46 | 2.43 | 6.84 | 76.39% | 6.57 |


## 4. Nhật Ký Kết Quả Đo Lường Chi Tiết: Testbed A (Máy Core i7)

Thư mục nguồn: `benchmarks/results/testbed_a_i7_14650hx/`

| Tệp Benchmark | State | TTFT (s) | Decode (tok/s) | Tổng (s) | Attention (s) | Expert Compute (s) | Expert I/O (s) | Cache Hit % | SSD Đọc (GB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2026-09-22-baseline.json` | engine-cold | 22.52 | 0.82 | 29.86 | 5.71 | 10.29 | 11.79 | 7.56% | 17.38 |
| `2026-09-22-baseline.json` | engine-warm | 21.85 | 0.84 | 28.96 | 5.46 | 9.96 | 11.48 | 7.81% | 17.33 |
| `2026-09-22-cache-128mib.json` | engine-cold | 13.9 | 1.4 | 18.18 | 3.01 | 8.61 | 5.41 | 60.97% | 7.34 |
| `2026-09-22-cache-128mib.json` | engine-warm | 13.89 | 1.28 | 18.59 | 3.04 | 8.85 | 5.55 | 64.38% | 6.7 |
| `2026-09-22-hot-cache-reused-` | engine-cold | 14.38 | 1.39 | 18.68 | 3.14 | 9.19 | 5.16 | 66.56% | 6.29 |
| `2026-09-22-hot-cache-reused-` | engine-warm | 12.5 | 1.41 | 16.77 | 3.21 | 9.24 | 3.12 | 78.69% | 4.01 |
| `2026-09-22-hot-cache-reused-` | engine-warm | 12.55 | 1.41 | 16.8 | 3.27 | 9.45 | 2.86 | 79.78% | 3.8 |
| `2026-09-22-mixed-workload-op` | run_0 | 15.45 | 1.37 | 19.82 | 3.76 | 9.96 | 4.76 | 66.56% | 6.29 |
| `2026-09-22-mixed-workload-op` | run_1 | 15.84 | 1.36 | 21.01 | 3.91 | 11.42 | 4.19 | 66.15% | 7.64 |
| `2026-09-22-mixed-workload-op` | run_2 | 31.29 | 1.53 | 35.86 | 7.21 | 20.73 | 5.22 | 67.61% | 13.39 |
| `2026-09-22-mixed-workload-op` | run_3 | 13.43 | 1.46 | 17.55 | 3.34 | 10.83 | 2.15 | 71.88% | 5.29 |
| `2026-09-22-mixed-workload-op` | run_4 | 15.4 | 1.4 | 20.4 | 3.99 | 11.77 | 3.14 | 64.84% | 7.93 |
| `2026-09-22-mixed-workload-op` | run_5 | 30.43 | 1.55 | 34.95 | 7.52 | 21.59 | 3.03 | 74.47% | 10.56 |
| `2026-09-22-skip-dense-revali` | engine-cold | 19.21 | 1.01 | 25.16 | 3.18 | 9.39 | 11.37 | 7.56% | 17.38 |
| `2026-09-22-skip-dense-revali` | engine-warm | 16.2 | 1.07 | 21.8 | 2.99 | 8.59 | 9.04 | 7.81% | 17.33 |
| `2026-09-23-batched-prefill-c` | run_0 | 27.25 | 2.47 | 32.92 | 8.51 | 14.08 | 8.31 | 90.75% | 10.43 |
| `2026-09-23-batched-prefill-c` | run_1 | 21.16 | 2.88 | 26.02 | 7.19 | 12.78 | 5.63 | 91.82% | 9.22 |
| `2026-09-23-batched-prefill-c` | run_2 | 20.73 | 2.91 | 25.54 | 7.26 | 13.01 | 4.9 | 92.09% | 8.92 |
| `2026-09-23-batched-prefill-n` | run_0 | 69.66 | 2.51 | 72.85 | 25.71 | 41.02 | 5.61 | 97.63% | 8.49 |
| `2026-09-23-dense-after-conte` | run_0 | 43.65 | 3.27 | 47.93 | 8.93 | 14.68 | 21.27 | 56.79% | 48.74 |
| `2026-09-23-dense-after-conte` | run_1 | 38.44 | 3.6 | 42.33 | 8.45 | 14.31 | 16.65 | 61.72% | 43.18 |
| `2026-09-23-dense-after-conte` | run_2 | 38.97 | 3.74 | 42.71 | 8.83 | 14.85 | 15.98 | 61.97% | 42.89 |
| `2026-09-23-dense-before-cont` | run_0 | 63.87 | 2.16 | 70.34 | 19.81 | 15.31 | 28.4 | 56.79% | 48.74 |
| `2026-09-23-dense-before-cont` | run_1 | 58.16 | 2.44 | 63.91 | 19.09 | 14.69 | 23.33 | 61.72% | 43.18 |
| `2026-09-23-dense-before-cont` | run_2 | 51.42 | 2.59 | 56.82 | 19.38 | 14.76 | 15.71 | 61.97% | 42.89 |
| `2026-09-23-flat-kv-context-3` | run_0 | 16.84 | 3.04 | 21.45 | 5.68 | 8.88 | 6.48 | 90.75% | 10.43 |
| `2026-09-23-flat-kv-context-3` | run_1 | 15.96 | 3.16 | 20.39 | 5.74 | 8.28 | 5.96 | 91.82% | 9.22 |
| `2026-09-23-flat-kv-context-3` | run_2 | 14.71 | 3.3 | 18.95 | 5.27 | 7.93 | 5.35 | 92.09% | 8.92 |
| `2026-09-23-flat-kv-context.j` | run_0 | 18.39 | 2.98 | 23.09 | 5.35 | 9.07 | 8.24 | 90.75% | 10.43 |
| `2026-09-23-int8-after.json` | engine-cold | 8.54 | 2.21 | 11.26 | 3.34 | 2.65 | 4.05 | 66.56% | 6.29 |
| `2026-09-23-int8-after.json` | engine-warm | 7.2 | 2.4 | 9.7 | 3.31 | 2.64 | 2.5 | 78.69% | 4.01 |
| `2026-09-23-int8-before.json` | engine-cold | 13.91 | 1.42 | 18.14 | 3.19 | 9.32 | 4.42 | 66.56% | 6.29 |
| `2026-09-23-int8-before.json` | engine-warm | 12.36 | 1.47 | 16.45 | 3.18 | 9.2 | 2.86 | 78.69% | 4.01 |
| `2026-09-23-quality-v1.json` | run_0 | 5.72 | 3.25 | 7.57 | 1.42 | 2.81 | 3.17 | 73.38% | 5.6 |
| `2026-09-23-quality-v1.json` | run_1 | 5.06 | 4.12 | 6.51 | 1.42 | 2.86 | 2.08 | 82.16% | 4.02 |
| `2026-09-23-quality-v1.json` | run_2 | 4.92 | 2.52 | 7.7 | 1.48 | 2.89 | 3.16 | 73.54% | 5.97 |
| `2026-09-23-quality-v1.json` | run_3 | 9.59 | 3.43 | 15.43 | 3.86 | 7.13 | 3.93 | 84.77% | 8.13 |
| `2026-09-23-quality-v1.json` | run_4 | 7.29 | 3.7 | 11.34 | 2.72 | 5.11 | 3.18 | 84.28% | 5.91 |
| `2026-09-23-quality-v1.json` | run_5 | 3.96 | 3.4 | 4.26 | 0.89 | 1.87 | 1.44 | 83.96% | 2.29 |
| `2026-09-23-quality-v1.json` | run_6 | 7.46 | 3.3 | 11.4 | 2.55 | 4.99 | 3.51 | 83.07% | 5.86 |
| `2026-09-23-quality-v1.json` | run_7 | 7.95 | 3.61 | 12.1 | 2.65 | 5.38 | 3.7 | 83.98% | 6.02 |
| `2026-09-23-quality-v1.json` | run_8 | 8.7 | 2.63 | 10.98 | 2.54 | 5.18 | 3.1 | 85.92% | 5.4 |
| `2026-09-23-quality-v1.json` | run_9 | 8.07 | 2.75 | 10.98 | 2.36 | 4.81 | 3.6 | 81.22% | 6.92 |
| `2026-09-23-quality-v1.json` | run_10 | 4.93 | 3.4 | 9.35 | 1.97 | 3.99 | 3.05 | 77.22% | 6.68 |
| `2026-09-23-quality-v1.json` | run_11 | 5.23 | 2.63 | 5.61 | 1.22 | 2.51 | 1.82 | 82.15% | 3.49 |
| `2026-09-23-rayon-1-context.j` | run_0 | 36.44 | 1.99 | 43.48 | 9.96 | 26.49 | 6.48 | 90.75% | 10.43 |
| `2026-09-23-rayon-24-context.` | run_0 | 17.89 | 2.67 | 23.13 | 5.56 | 10.09 | 7.06 | 90.75% | 10.43 |
| `2026-09-23-rayon-8-context.j` | run_0 | 12.95 | 3.59 | 16.86 | 4.45 | 6.06 | 6.01 | 90.75% | 10.43 |
| `2026-09-23-suite-v1.json` | run_0 | 9.41 | 2.34 | 11.97 | 3.58 | 2.81 | 4.22 | 65.71% | 7.22 |
| `2026-09-23-suite-v1.json` | run_1 | 11.77 | 2.59 | 17.95 | 5.75 | 4.48 | 5.57 | 70.26% | 9.84 |
| `2026-09-23-suite-v1.json` | run_2 | 25.96 | 2.47 | 34.06 | 10.57 | 8.11 | 11.46 | 66.07% | 20.41 |
| `2026-09-23-suite-v1.json` | run_3 | 23.79 | 1.78 | 27.16 | 7.65 | 6.05 | 10.62 | 62.21% | 16.2 |
| `2026-09-23-suite-v1.json` | run_4 | 73.02 | 1.7 | 81.27 | 20.09 | 15.62 | 38.07 | 45.42% | 61.56 |
| `2026-09-23-suite-v1.json` | run_5 | 10.04 | 1.81 | 13.35 | 4.05 | 3.19 | 4.6 | 66.66% | 7.02 |
| `2026-09-23-suite-v1.json` | run_6 | 11.47 | 2.59 | 17.66 | 6.05 | 4.77 | 4.55 | 68.96% | 10.27 |
| `2026-09-23-suite-v1.json` | run_7 | 23.31 | 2.71 | 30.68 | 10.93 | 8.44 | 7.13 | 66.46% | 20.18 |
| `2026-09-23-suite-v1.json` | run_8 | 20.25 | 2.2 | 22.98 | 7.63 | 5.92 | 6.5 | 64.21% | 15.34 |
| `2026-09-23-suite-v1.json` | run_9 | 64.84 | 2.05 | 71.66 | 20.6 | 15.8 | 27.55 | 48.51% | 58.08 |
| `2026-09-23-suite-v1.json` | run_10 | 8.18 | 2.44 | 10.63 | 3.77 | 2.92 | 2.49 | 66.29% | 7.1 |
| `2026-09-23-suite-v1.json` | run_11 | 11.34 | 2.73 | 17.19 | 6.19 | 4.81 | 3.85 | 68.13% | 10.54 |
| `2026-09-23-suite-v1.json` | run_12 | 22.22 | 2.63 | 29.82 | 10.84 | 8.31 | 6.53 | 65.84% | 20.55 |
| `2026-09-23-suite-v1.json` | run_13 | 20.25 | 2.11 | 23.1 | 7.44 | 5.75 | 7.05 | 64.75% | 15.11 |
| `2026-09-23-suite-v1.json` | run_14 | 63.41 | 2.22 | 69.72 | 19.41 | 14.75 | 28.43 | 49.32% | 57.17 |
| `2026-09-23-suite-v2-long-con` | run_0 | 84.56 | 3.24 | 87.03 | 23.94 | 36.8 | 18.58 | 74.54% | 77.53 |
| `2026-09-23-suite-v2-near-con` | run_0 | 94.37 | 3.43 | 96.71 | 28.58 | 42.92 | 16.11 | 75.05% | 89.51 |
| `2026-09-24-blocked-attention` | run_0 | 48.17 | 3.03 | 50.8 | 19.29 | 23.65 | 7.09 | 97.63% | 8.49 |
| `2026-09-24-online-softmax-ne` | run_0 | 50.3 | 3.11 | 52.87 | 21.35 | 20.36 | 9.88 | 97.63% | 8.49 |
| `2026-09-24-online-softmax-ne` | run_1 | 42.28 | 3.99 | 44.29 | 21.67 | 18.29 | 3.43 | 98.0% | 7.17 |
| `2026-09-24-online-softmax-ne` | run_2 | 42.5 | 4.0 | 44.5 | 22.09 | 18.23 | 3.33 | 98.05% | 7.01 |
| `2026-09-24-parallel-attentio` | run_0 | 49.07 | 2.3 | 52.55 | 20.39 | 24.94 | 6.41 | 97.63% | 8.49 |
| `2026-09-24-parallel-attentio` | run_1 | 54.5 | 2.24 | 58.07 | 23.35 | 28.11 | 5.68 | 98.0% | 7.17 |
| `2026-09-24-parallel-attentio` | run_2 | 52.81 | 2.44 | 56.09 | 22.13 | 27.93 | 5.17 | 98.05% | 7.0 |
| `2026-09-24-parallel-attentio` | run_0 | 41.38 | 3.69 | 43.55 | 17.37 | 20.84 | 4.68 | 97.63% | 8.49 |


## 5. Quy Ước Vận Hành Cho 2 Thành Viên (Team Protocol)

1. **Thành viên 1** khi chạy benchmark lưu vào: `benchmarks/results/testbed_a_i7_14650hx/`.
2. **Thành viên 2** khi chạy benchmark lưu vào: `benchmarks/results/testbed_b_i5_10400h/`.
3. Chạy `python tools/run_benchmark.py` để script tự động phát hiện máy và lưu vào đúng folder mà không lo nhầm lẫn!
