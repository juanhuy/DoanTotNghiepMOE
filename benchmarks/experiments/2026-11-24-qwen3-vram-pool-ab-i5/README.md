# Pool expert VRAM không cần dense GPU, Qwen3-30B-A3B, i5, 24/11/2026

Protocol [`qwen3-vram-pool-ab-v1`](../../SCENARIOS.md#pool-expert-vram-không-cần-dense-gpu-cho-qwen3-trên-i5-qwen3-vram-pool-ab-v1-2411), commit trước khi đo.

## Chuẩn bị (trước A/B)

- **Tính năng** (commit `d67a8c5`): `MOE_GPU_DENSE=off` giữ dense trên CPU, VRAM chỉ chứa pool expert và kernel A1; `MOE_GPU_EXPERT_PLACEMENT` nạp sẵn các expert dùng nhiều nhất theo hồ sơ tần suất, `reset_cache` không xóa placement. Test CUDA thật (GTX 1650) `gpu_expert_pool_without_dense_gpu_matches_cpu_with_placement`: logits/text khớp CPU, placement còn sau reset (test này đỏ trước khi sửa reset). `check.py --rust --release` đạt.
- **Hồ sơ tần suất** [`placement-long-prefill.txt`](placement-long-prefill.txt) (SHA-256 `3a5cea5d63f1150b…`): CPU-only, workload `long-prefill` × 4 (prompt khác `mixed`), `MOE_USAGE_PATH`, revision sạch ([precheck/profile-long-prefill](precheck/profile-long-prefill/)). Lượt chạy lại trên cây dirty trước đó cho file giống từng byte.
- **Lượt kiểm trước** ([precheck/smoke-pool](precheck/smoke-pool/), cấu hình B, revision sạch, một lượt, không phải A/B):
  - 713 slot (3218MiB), nạp sẵn 713 expert trong 4,4s; pool giữ đúng 713 expert theo hồ sơ ở 8/8 turn;
  - text khớp lượt CPU-only 8/8 turn; VmSwap 0; RSS sau request cuối 6,02GiB (CPU-only ~5,6GiB);
  - cache miss và số route tính GPU khớp **tuyệt đối** bộ mô phỏng ở 8/8 turn; miss bằng 0,724 lần CPU-only.
- Lượt kiểm trên cây dirty trước khi sửa `reset_cache` cho thấy lỗi: harness reset trước request đầu, pool bị xóa rồi nạp lại kiểu "đọc trước giữ trước" (chỉ 85/713 expert trùng hồ sơ). Lượt đó không giữ làm bằng chứng.

## Kết quả A/B

Chưa đo.
