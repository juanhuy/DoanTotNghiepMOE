# Pool expert VRAM không cần dense GPU, Qwen3-30B-A3B, i5, 24/11/2026

**Nhãn: P1.** Kết luận: **B nhanh hơn ở 16/16 so sánh** theo rule chốt trước; text khớp 16/16; không OOM, không swap.

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

- Revision `a84171f` (src như `d67a8c5`), worktree sạch, `git_dirty` false ở 4/4 block; binary `bb4f2a9cf13e2bcf…`. Thứ tự A B B A, mỗi block một process, checkpoint bị xóa khỏi page cache trước mỗi block (fincore 0 byte). [console.log](console.log), [summary.json](summary.json) (`tools/compare_ram_runs.py --protocol qwen3-vram-pool-ab-v1`).
- Block: [A0 CPU](block0-A-cpu/), [B1 pool](block1-B-pool/), [B2 pool](block2-B-pool/), [A3 CPU](block3-A-cpu/). Cả 4 block 8/8 request, `oom_kill` 0, VmSwap 0.

| | A (CPU-only) | B (pool VRAM 713 expert, dense CPU) |
| --- | ---: | ---: |
| Decode tok/s (trung vị 16 turn) | 0,94 | **1,33** |
| Total/request (trung vị) | 82,2s | **61,5s** |
| TTFT (trung vị) | 11,9s | 10,9s |
| Payload expert đọc từ file/request (trung vị) | 74,6GB | 52,1GB |
| Cache miss B/A theo cặp | | 0,685–0,788 (tổng 0,724, đúng dự đoán) |
| Expert tính trên GPU/request | 0 | 4.585–12.526 route, ~1,0s GPU |
| Thời gian expert trên CPU/request (trung vị theo block) | 24,3 / 24,6s | 17,5 / 14,6s |
| VmHWM sau request cuối | 5,64 / 5,67GiB | 6,01 / 6,00GiB |
| Peak scope anon / file | 5771 / 6046, 5795 / 6529MiB | 5878 / 6996, 5877 / 7099MiB |
| VRAM cấp phát | 0 | 3.374.577.664 byte (pool 713 slot, dense 0) |

- **16 so sánh:** total B/A 0,686–0,869, decode B/A 0,665–0,839, đều ≤ 1/1,05 → `b_faster: true`. Cặp A0–B1 có tỉ số 0,694–0,869, cặp A3–B2 0,686–0,826.
- **Chi phí:** RSS cao hơn khoảng 370MiB (CUDA context, thư viện, pinned buffer), anon +~100MiB; nạp pool lúc khởi động 4,2–4,5s (không tính vào request).

## Diễn giải

- Trên chính GTX 1650 4GB, Qwen3-30B-A3B (dense F32 không vừa VRAM) **dùng được GPU** nhờ đặt riêng expert lên VRAM: decode nhanh hơn ~1,42 lần, total ~1,34 lần (trung vị), nhất quán ở cả 16 cặp.
- Mức tăng vượt cận "chỉ tính I/O" ~1,2 lần ghi trong protocol. Số đo cho thấy hai nguồn: (1) bớt 28% lần đọc file và payload đọc giảm từ 74,6 xuống 52,1GB; (2) phần tính expert trên CPU giảm từ ~24s xuống 15–18s mỗi request vì các route trên pool chạy trên GPU (~1s). Dự đoán trước chỉ tính (1) nên thấp hơn. Không tách được đóng góp chính xác của từng nguồn từ A/B này.
- Đặt expert theo hồ sơ tần suất là điều kiện cần: mô phỏng cho thấy chính sách A1 cũ (fill-only, đọc trước giữ trước) chỉ giảm 19% lần đọc ở cùng cỡ pool.
- Baseline A hôm nay chậm hơn hồ sơ P1 17/11 (decode 0,94 so với 1,34 tok/s, cùng cấu hình) do điều kiện máy khác ngày; vì vậy chỉ so trong cùng ABBA, không so B với số 17/11.

## Giới hạn

- Một workload (`mixed`, 4 prompt tiếng Anh ngắn), một lượt ABBA, một máy (i5). Hồ sơ tần suất học từ `long-prefill` (2 prompt); chưa kiểm với workload xa hơn hoặc ngôn ngữ khác.
- Placement tĩnh cho cả process; chưa có cơ chế cập nhật khi phân bố routing đổi.
- Số slot phụ thuộc VRAM trống lúc khởi động (713 ở lượt này). Chưa chạy trên i7.
- Text khớp với CPU-only 16/16 turn; parity logits đo trên fixture (test CUDA), chưa đo trên model thật ở cấu hình B.
