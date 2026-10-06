# Tiêu chuẩn Lập trình Rust cho MoE Inference Engine

## 1. Kiểm tra Tĩnh & Mã sạch (Static Analysis)
- Mọi thay đổi phải vượt qua kiểm tra nghiêm ngặt:
  ```bash
  cargo fmt --all -- --check
  cargo clippy --all-targets -- -D warnings
  cargo test --all-targets
  ```
- Không để lại cảnh báo compiler hoặc clippy warning.

## 2. Quản lý Bộ nhớ & Tối ưu Hot-loop (Zero-Allocation)
- **Tuyệt đối không cấp phát Heap (Vec/Box/String) trong inference hot-loop** (trong các hàm `forward`, `matvec`, `attention step`, `expert SwiGLU`).
- Tái sử dụng các vùng đệm đệm sẵn: `StepScratch`, `ExpertScratch`.
- Sử dụng slice `&[f32]`, `&mut [f32]`, `&[i8]` thay vì pass by value hoặc clone vector.

## 3. Tính toán & SIMD Alignment
- Các kernel ma trận/vector phải thiết kế để compiler dễ vector hóa (ví dụ: dùng các bộ cộng độc lập gom theo chunk 8 phần tử).
- Đảm bảo tính toán song song với Rayon được phân đoạn theo hàng (row-major) để tối ưu hóa CPU cache và tránh cache line bouncing.

## 4. Sai số Số học (Numerical Precision)
- Giữ vững ngưỡng sai số tuyệt đối so với PyTorch/Transformers baseline $\le 2 \times 10^{-5} \times (1 + |\text{ref}|)$.
- Các biến đổi lượng tử hóa INT8/INT4 phải xử lý cẩn thận `row_scale` và clamping.
