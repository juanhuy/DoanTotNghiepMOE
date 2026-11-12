# Số thread CPU theo máy A3, i5, 12/11/2026

## Câu hỏi và protocol

Bước A3 của hướng cải tiến (PLAN). Các protocol S1–S5 cố định 4 thread Rayon để giữ so sánh. Câu hỏi: với cấu hình khuyến nghị (GPU dense, tầng VRAM expert tắt), tăng thread có làm chat/decode nhanh hơn trên i5 (i5-10500H, 6 nhân/12 luồng) không? Protocol `threads-chat-v1` chốt và commit vào [SCENARIOS](../../SCENARIOS.md#số-thread-cpu-theo-máy-a3-threads-chat-v1-1211) (commit 455bfea) trước khi đo.

- Thứ tự palindrome `4,6,8,12,12,8,6,4`, mỗi block là một process mới với 8 request mixed v1 ×2 nhóm (Mars, follow-up có lịch sử, Python, Love). Nhóm 1 (turn0–3) là warmup và được loại; nhóm 2 (turn4–7) được đo. Tổng 64 request.
- `--gpu auto`, tier off, cache 64MiB/layer, context 512, max64, victim, Gate–Up paired, batch off, lookahead off, FP32 KV, greedy. Chỉ `RAYON_NUM_THREADS` đổi giữa các block.
- Rule: mỗi N được so với 4 thread ở 8 so sánh (2 cặp gương × 4 prompt). N nhanh hơn nếu total_ms VÀ decode_ms 4/N ≥1,05 ở cả 8; N chậm hơn nếu ≤1/1,05 ở cả 8. Khuyến nghị là N nhanh nhất trong các N đạt; không N nào đạt thì giữ 4.

```bash
python3 tools/run_benchmark.py --testbed i5 --thread-sweep-model --gpu auto \
  --expert-batch-workload mixed --cache 64 --context 512 --max-tokens 64 --lookahead off
```

## Provenance

- Run 20261112T050426.127103Z-threads-d2fd10cb…, HEAD 455bfea, git_dirty false ở cả 8 block. Binary 64048653c5016318… (trùng S4/S5). Checkpoint chung (tokenizer b1fb1517…). tools/check.py PASS; fixture tier off 1/4/12 thread cho cùng text và số lần nạp.
- [preflight](preflight.json): available ≥12204MiB, pswpout không tăng, không engine/build khác; [nvidia-smi](nvidia-smi-pre.csv).

### Lượt đầu bị dừng (giữ nguyên làm hồ sơ)

[attempt1-stopped](attempt1-stopped/) (20261112T042107…): preflight đạt (available 7231MiB), nhưng các app khác chiếm khoảng 5GB anon (Chrome, desktop). Block0 (4 thread) hoàn tất 6/8 request. Đến giây 188, RAM trống còn khoảng 260MiB và engine bị swap 146MiB, watchdog dừng (ngưỡng 128MiB). Lượt này không có số đo. Người dùng đóng app rồi chạy lại một lần, không đổi ngưỡng/protocol.

## Kết quả

COMPLETED 64/64. Text/usage/finish và 5 work counter khớp giữa mọi block. Prompt tokens 19/107/22/20, completion 64.

**Rule: không N nào nhanh hơn hay chậm hơn 4 thread một cách nhất quán → khuyến nghị giữ 4 thread (`recommended_threads: 4`).**

| Thread | Decode tok/s trung vị | Total trung vị (ms) | total 4/N (8 so sánh) | decode 4/N | expert_compute 4/N |
| ---: | ---: | ---: | --- | --- | --- |
| 4 | 8,80 | 8436 | — | — | — |
| 6 | 8,87 | 8509 | 0,954–1,021 | 0,960–1,017 | 0,924–1,024 |
| 8 | 8,64 | 8584 | 0,927–1,009 | 0,958–0,999 | 0,903–0,988 |
| 12 | 8,67 | 8662 | 0,890–1,001 | 0,893–1,007 | 0,806–0,965 |

Warmup (loại) cùng chiều: total 4/N trong khoảng 0,910–1,103.

### Diễn giải

- Mọi chênh lệch nằm trong ±11%, không đạt ngưỡng 5% ở cả 8 so sánh. Chiều chung là thêm thread **không** giúp, và ở 8/12 thread thì expert_compute còn chậm hơn một chút (4/N 0,81–0,99). Điều này phù hợp với giả thuyết SwiGLU INT8 ở batch 1 bị giới hạn bởi băng thông bộ nhớ chứ không phải số nhân, và 12 luồng dùng hyperthreading trên 6 nhân. Chưa có profiler nên đây chưa phải kết luận nhân quả.
- Biến động theo vị trí lớn hơn hiệu ứng thread: cả bốn block nửa sau (12,8,6,4) đều nhanh hơn nửa đầu (decode khoảng 8,6–10,0 so với 7,1–9,2 tok/s, expert_io trung vị khoảng 3,9 so với 4,1–4,4s). Thiết kế palindrome giúp hai phía chịu biến động này như nhau.
- Quan sát phụ, **không phải đối chứng**: decode tok/s trung vị ở 4 thread là 8,80, trong khi S5 tier off cùng binary/cấu hình là 5,53. Khác biệt đáng chú ý về điều kiện máy: S5 có MemAvailable min khoảng 4,4–4,8GiB, lượt này khoảng 8,8–9,3GiB. Nhiều RAM trống hơn thì page cache có thể giữ được nhiều phần checkpoint 7GB hơn. Đây là gợi ý cho phép thử RAM hạn chế (giới hạn page cache bằng cgroup), chưa được kiểm.

### Tài nguyên

RSS max 3506–3525MiB mọi block, VmSwap 0, MemAvailable min 8833–9296MiB. pswpout hệ thống tăng 55835 page ở block4 (12 thread nửa sau), các block khác 0–2691; engine không bị swap.

## Kết luận và giới hạn

Trên i5, cấu hình khuyến nghị giữ **4 thread**; tăng tới 6/8/12 thread không cải thiện chat/decode một cách nhất quán. Chỉ một lượt, một máy, workload mixed max64, tier off/cache64; không đo prefill dài, cache khác, hoặc i7 (i7 chạy riêng tập 4,8,16,24).

## Bảo toàn payload

Chuyển nguyên hai run vào `run/` và `attempt1-stopped/`, kèm preflight/nvidia-smi của cả hai lượt. Khóa 121 file bằng CHECKSUMS.sha256 (không gồm README).

```bash
sha256sum -c CHECKSUMS.sha256
```
