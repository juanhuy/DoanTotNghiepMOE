# Số thread CPU theo máy A3, i7, 13/11/2026

## Câu hỏi và protocol

Bước A3 của hướng cải tiến (PLAN). Các protocol S1–S5 cố định 4 thread Rayon để giữ so sánh. Câu hỏi: với cấu hình khuyến nghị (GPU dense, tầng VRAM expert tắt), tăng thread có làm chat/decode nhanh hơn trên i7 (i7-14650HX, 16 nhân/24 luồng: 8 P-cores + 8 E-cores) không? Protocol `threads-chat-v1` chốt và commit vào [SCENARIOS](../../SCENARIOS.md#số-thread-cpu-theo-máy-a3-threads-chat-v1-1211) (commit 455bfea) trước khi đo.

- Thứ tự palindrome `4,8,16,24,24,16,8,4`, mỗi block là một process mới với 8 request mixed v1 ×2 nhóm (Mars, follow-up có lịch sử, Python, Love). Nhóm 1 (turn0–3) là warmup và được loại; nhóm 2 (turn4–7) được đo. Tổng 64 request.
- `--gpu auto`, tier off, cache 64MiB/layer, context 512, max64, victim, Gate–Up paired, batch off, lookahead off, FP32 KV, greedy. Chỉ `RAYON_NUM_THREADS` đổi giữa các block.
- Rule: mỗi N được so với 4 thread ở 8 so sánh (2 cặp gương × 4 prompt). N nhanh hơn nếu total_ms VÀ decode_ms 4/N ≥1,05 ở cả 8; N chậm hơn nếu ≤1/1,05 ở cả 8. Khuyến nghị là N nhanh nhất trong các N đạt; không N nào đạt thì giữ 4.

```bash
python3 tools/run_benchmark.py --testbed i7 --thread-sweep-model --gpu auto \
  --expert-batch-workload mixed --cache 64 --context 512 --max-tokens 64 --lookahead off \
  --model-dir models/olmoe-i5
```

## Provenance

- Run 20261113T152744.003149Z-threads-e8fe6ef6…, HEAD 16835a3, git_dirty false ở cả 8 block. Binary 5b091a4df13fbb6f… Checkpoint chung `models/olmoe-i5` (khớp 16/16 hash `olmoe-int8-i5.sha256`, tokenizer b1fb1517…). `tools/check.py --rust --release` PASS; GPU tests EXECUTED trên RTX 5060 Laptop GPU.
- [preflight](preflight.json): available ≥10205MiB, pswpout không tăng [229689, 229689, 229689], eligible: true, không engine/build khác; [nvidia-smi](nvidia-smi-pre.csv).

### Lượt đầu bị dừng (giữ nguyên làm hồ sơ)

[attempt1-stopped](attempt1-stopped/) (20261113T151932.178310Z-threads-c20876d6…): preflight đạt (available ≥10074MiB, [attempt1-preflight](attempt1-preflight.json), [attempt1-nvidia-smi](attempt1-nvidia-smi-pre.csv)). Hoàn tất block0 (t4), block1 (t8), block2 (t16) với VmSwap 0KiB. Ở block3 (t24), Chrome khởi động lại chiếm RAM gây áp lực bộ nhớ hệ thống, kswapd đẩy VmSwap của process engine lên 250MiB, vượt ngưỡng watchdog 128MiB. Lượt này không có số đo hợp lệ. Người dùng đóng hoàn toàn Chrome, kiểm tra preflight lại đạt và chạy lại một lần theo đúng hướng dẫn WORKFLOWS.md, không đổi ngưỡng/protocol.

## Kết quả

COMPLETED 64/64. Text/usage/finish và 5 work counter khớp giữa mọi block. Prompt tokens 19/107/22/20, completion 64.

**Rule: không N nào nhanh hơn hay chậm hơn 4 thread một cách nhất quán → khuyến nghị giữ 4 thread (`recommended_threads: 4`).**

| Thread | Decode tok/s trung vị | Total trung vị (ms) | total 4/N (8 so sánh) | decode 4/N | expert_compute 4/N |
| ---: | ---: | ---: | --- | --- | --- |
| 4 | 7,72 | 10825 | — | — | — |
| 8 | 7,21 | 11416 | 0,870–0,995 | 0,867–0,994 | 0,955–1,084 |
| 16 | 7,03 | 11617 | 0,871–0,980 | 0,895–0,979 | 0,867–0,992 |
| 24 | 7,02 | 11755 | 0,777–1,004 | 0,867–1,032 | 0,711–1,001 |

### Diễn giải

- Kết quả trên i7 hoàn toàn đồng thuận với i5: việc tăng số thread CPU từ 4 lên 8, 16 hay 24 **không** cải thiện tốc độ decode mà có xu hướng giảm nhẹ (decode tok/s trung vị giảm từ 7,72 xuống 7,21, 7,03, 7,02; latency tăng từ 10,8s lên 11,4s, 11,6s, 11,8s).
- Không có bất kỳ cấu hình N nào đạt ngưỡng nhanh hơn 5% (tỉ số 4/N ≥ 1,05) ở cả 8 cặp so sánh (`faster_than_4: false`).
- Điều này củng cố giả thuyết: ở batch size 1 (sequential token generation), SwiGLU INT8 inference bị nghẽn bởi memory bandwidth (nạp trọng số expert vào cache CPU/RAM) hơn là compute core; việc chia thêm luồng (nhất là khi vượt số P-core sang E-core hoặc hyperthread trên kiến trúc lai) có thể làm tăng chi phí đồng bộ hóa của Rayon. Chưa có profiler nên đây là giả thuyết, không phải kết luận nhân quả (ghi chú khi tích hợp 14/11).
- Do đó, cấu hình khuyến nghị vận hành cho cả hai testbed i5 và i7 thống nhất giữ **4 thread CPU**.

### Tài nguyên

RSS max 3416,0–3431,4MiB ở mọi block, VmSwap 0KiB ở cả 8 block, MemAvailable min 7485–7599MiB (đính chính khi tích hợp 14/11: bản đầu ghi "> 9,5GiB", không khớp memory.jsonl).

## Kết luận và giới hạn

Trên i7, cấu hình khuyến nghị giữ **4 thread** (`recommended_threads: 4`); tăng tới 8/16/24 thread không cải thiện chat/decode. Kết luận nhất quán giữa i5 và i7. Phép đo thực hiện trong phạm vi workload mixed max64, GPU dense auto, GPU expert tier off, cache 64MiB/layer; không đại diện cho batch > 1 hay prefill context cực dài.

## Bảo toàn payload

Chuyển nguyên hai run vào `run/` và `attempt1-stopped/`, kèm preflight/nvidia-smi của cả hai lượt. Khóa 154 file bằng CHECKSUMS.sha256 (không gồm README).

```bash
sha256sum -c CHECKSUMS.sha256
```
