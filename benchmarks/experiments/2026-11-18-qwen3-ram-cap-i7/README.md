# Qwen3-30B-A3B dưới trần RAM 10GB và 8GB, i7, 18/11/2026

## Câu hỏi và protocol

Thực nghiệm độc lập trên testbed i7 cho model Qwen3-30B-A3B INT8 (~30GB) theo protocol [`qwen3-ram-cap-v1`](../../SCENARIOS.md#qwen3-30b-a3b-dưới-trần-ram-10gb-qwen3-ram-cap-v1-1711). Kiểm tra xem engine có hoàn thành workload dưới giới hạn cgroup 10240MiB và 8192MiB (tính cả page cache, không swap) trên CPU hay không, và so sánh chéo tính nhất quán kết quả với i5 trên cùng một checkpoint đã convert.

Kèm một lượt chạy khám phá (P3, tùy chọn) với GPU (`--gpu auto --gpu-expert-tier on --gpu-expert-compute on`) tận dụng VRAM 8GB của RTX 5060 Laptop để đưa dense lên VRAM và duy trì pool L1 expert resident.

- **Checkpoint:** `Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
  - SHA256 16 shard và tokenizer khớp hoàn toàn Hub ([source-checkpoint-sha256.json](source-checkpoint-sha256.json)).
  - Convert sang INT8 bằng `tools/prepare_olmoe.py` trong 2 phút 37 giây trên i7 ([convert.log](convert.log)).
  - Kiểm tra đối chiếu hash 478 file checkpoint đã convert với bản của i5: **0 mismatched files of 478**. Hai máy chạy cùng bit-exact checkpoint.
- **Cấu hình engine (protocol chính):**
  - dense BF16 (`MOE_DENSE_STORAGE=bf16`, 2.459.435.008 byte), embedding trên đĩa, thinking off;
  - CPU-only (`--gpu cpu`), 4 threads (`--threads 4`);
  - cache 64MiB/layer × 48 (14/128 expert mỗi layer), context 512, max-tokens 64, victim, lookahead off, batch off;
  - guard cooperative đặt thấp hơn trần 1024MiB (`--ram-limit-mib 9216 --stop-rss-mib 9728 --headroom-mib 512`).
- **Workload:** `mixed` v1 gồm 4 prompt, có follow-up, chạy 8 turn liên tiếp với `--keep-cache`.
- **Điều kiện đo:**
  - Thứ tự chạy 10240MiB rồi 8192MiB;
  - Trước mỗi lượt, page cache được xóa và kiểm bằng `fincore` (về 0 byte);
  - Chạy dưới `systemd-inhibit --what=idle:sleep`, cắm sạc, tắt màn hình chờ, đóng Chrome/Slack;
  - Preflight: `eligible: true`, available 11698MiB, swap không tăng, AC on, GPU not throttled ([preflight.json](preflight.json)).

HEAD `82a4b8df4a90b4800f5995c93f794153324f5bb1`, binary sha256 `12a62ee910d02beaf5671ef969dc1a9a4fbc957219730c79530b4f50833156aa`, i7-14650HX, 16GB RAM (15,2GiB khả dụng), RTX 5060 Laptop GPU 8GB (driver 595.99.02 theo `hardware_observed`; kernel NVRTC biên dịch PTX `compute_75`, driver JIT cho GPU). [console.log](console.log).

## Kết quả

| Trần / Cấu hình | Hoàn thành | oom_kill | VmSwap | Đỉnh scope current / anon / file (MiB) | RSS peak (MiB) |
| --- | --- | ---: | ---: | --- | ---: |
| 10240MiB CPU ([limit-10240](limit-10240/)) | 8/8 | 0 | 0 | 7758,6 / 5739,9 / 4409,7 | 5749,3 |
| 8192MiB CPU ([limit-8192](limit-8192/)) | 8/8 | 0 | 0 | 8192,1 / 5745,0 / 4929,8 | 5754,3 |
| 10240MiB GPU explore ([gpu-explore](gpu-explore/)) | 8/8 | 0 | 0 | 10240,0 / 5836,0 / 6517,6 | 6074,4 |

| Turn 0–7 | Trần 10240 (CPU) | Trần 8192 (CPU) | GPU explore (10240) |
| --- | --- | --- | --- |
| TTFT (trung vị, min–max) | 9,9s (8,3–18,9) | 10,9s (9,1–20,2) | 6,9s (5,6–12,4) |
| Decode tok/s (trung vị, min–max) | 1,53 (1,24–1,86) | 1,78 (1,51–1,97) | 2,22 (1,73–2,68) |
| Total/request (trung vị, min–max) | 55,7s (42,3–61,1) | 49,8s (41,0–54,9) | 37,4s (29,0–43,3) |
| Tỉ lệ thời gian chờ đọc expert trên total | 0,59–0,65 | 0,55–0,62 | 0,59–0,66 |
| Tốc độ đọc file expert (trung vị, min–max) | 2,24 GB/s (2,14–2,30) | 2,57 GB/s (2,37–2,82) | 2,75 GB/s (2,61–2,93) |
| Cache hit expert | 38–73% | 38–73% (giống hệt) | 46–77% |
| Payload expert đọc/request | 60,6–91,0GB | 60,6–91,0GB | 49,3–79,2GB |
| Số tính toán expert trên GPU | 0 | 0 | 31.469 (1499,2 ms) |

### Tính nhất quán text và xác nhận chéo với i5

- Cả 8/8 turn ở cả 3 lượt chạy trên i7 cho văn bản sinh ra **khớp từng ký tự (bit-exact 8/8)** với kết quả của i5 (`benchmarks/experiments/2026-11-17-qwen3-ram-cap-i5/`).
- Mẫu câu trả lời (64 token):
  - Turn 0: "Mars is one of the eight planets in our solar system, and it is the fourth planet from the Sun. It is often called the \"Red Planet\" because its surface appears reddish due to iron oxide (rust) on its surface.\n\n### Location of Mars:\n- **Orbit**: Mars orbits the Sun at"
  - Turn 1 (follow-up): "You just asked about **Mars**.\n\nMars is the fourth planet from the Sun in our solar system, known for its reddish appearance, thin atmosphere, and features like Olympus Mons (the largest volcano) and Valles Marineris (a giant canyon system). Is there something specific you'd like to"
  - Turn 2: "A **Python dictionary** is a built-in data structure used to store collections of data in **key-value pairs**. It is mutable (can be changed), unordered (prior to Python 3.7) / insertion-ordered (Python 3.7 and later), and highly optimized for retrieving values when you know the key"

### Khám phá GPU (P3)

- RTX 5060 Laptop 8GB nạp toàn bộ dense lên VRAM dưới dạng **F32** (BF16 được nới thành F32 khi upload cho cuBLAS SGEMV, `allocated_mb` 4643) và cấp phát pool L1 VRAM Expert gồm **565 slots**. Bản dense BF16 trên RAM vẫn giữ, nên anon không giảm (~5,8GiB).
- Trong quá trình suy luận, `expert_gpu_computes` đạt 31.469 lượt (tổng thời gian tính toán expert GPU chỉ 1,5 giây trên NVRTC SwiGLU INT8 kernel).
- Cache hit tăng lên 46–77% nhờ tận dụng đồng thời cả L1 VRAM pool và L2 RAM cache.
- Decode trung vị 2,22 tok/s, TTFT trung vị 6,9s, text khớp 8/8. Đây là một lượt khám phá, đổi đồng thời dense lên GPU, pool L1 và tính expert trên GPU, nên không phải A/B và không quy được phần tăng tốc cho thành phần nào.

## Giới hạn

- Bằng chứng ổn định bộ nhớ dưới cgroup trần hẹp; không cấu thành benchmark so sánh năng lực IPC giữa hai kiến trúc CPU.
- Lượt GPU explore mang tính chất khảo sát cấu trúc bộ nhớ hỗn hợp VRAM+RAM; chưa đo toàn diện các kích thước pool khác nhau.
- Checkpoint INT8 30GB chạy thực tế với anon RSS ~5,7GiB (CPU) và ~6,0GiB (GPU), tương đương dưới 20% dung lượng mô hình trên đĩa.

## Đính chính khi review (owner, 20/11/2026)

Raw không đổi; chỉ README/mục lục/PROGRESS được sửa so với commit 08f7a9c:

- **Trung vị:** bản gốc dùng trung vị cao (`median_high`, phần tử thứ 5 trên 8), trong khi hồ sơ i5 và `tools/compare_ram_runs.py` dùng trung vị chuẩn (trung bình hai phần tử giữa). Đã tính lại decode/TTFT/total theo trung vị chuẩn: 10240 là 1,60→1,53 tok/s, 10,2→9,9s, 57,2→55,7s; 8192 là 1,79→1,78 tok/s, 52,6→49,8s; GPU là 2,30→2,22 tok/s, 7,0→6,9s, 39,0→37,4s.
- **Đơn vị:** tốc độ đọc và payload GPU ghi là GB nhưng thực ra là GiB. Đã đổi sang GB (10^9, `expert_file_bytes_read / expert_file_read_ms`) như hồ sơ i5.
- **GPU:** dense trên VRAM là F32 (nới từ BF16), không phải BF16. Driver là 595.99.02 theo raw, không phải 570.86.16/CUDA 12.8. `compute_75` là đích PTX của NVRTC, không phải chế độ của GPU.
- **8192 nhanh hơn 10240** (1,78 so với 1,53 tok/s) không phải hiệu ứng của trần: ở 10240, scope chỉ đạt 7758,6MiB, nên trần không chạm. Mỗi mức một lượt, thứ tự cố định, nên đây là dao động giữa hai lượt (tốc độ đọc 2,24 so với 2,57 GB/s), không kết luận theo chiều nào.
- Không so tốc độ i5 với i7: khác máy, khác SSD, khác ngày. Điểm so chéo hợp lệ là checkpoint 478/478 hash và text 8/8 khớp từng ký tự (đã kiểm lại cả 3 lượt).

## Bảo toàn payload

Toàn bộ artifact được khóa bằng SHA256 ([CHECKSUMS.sha256](CHECKSUMS.sha256)): 43 file (không tính README.md).
```bash
sha256sum -c CHECKSUMS.sha256
```
