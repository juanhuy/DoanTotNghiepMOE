# Qwen3-30B-A3B dưới trần RAM 10GB và 8GB, i5, 17/11/2026

## Câu hỏi và protocol

Model INT8 khoảng 30GB, gấp đôi RAM máy (15,4GiB), có chạy hết workload khi engine bị giới hạn RAM ở 10GB hoặc 8GB không? Giới hạn này tính cả page cache và không cho swap. Đây là quan sát ổn định có giới hạn, không phải so sánh A/B tốc độ. Protocol [`qwen3-ram-cap-v1`](../../SCENARIOS.md#qwen3-30b-a3b-dưới-trần-ram-10gb-qwen3-ram-cap-v1-1711) được commit trước khi đo (f695629).

- **Checkpoint:** `Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. SHA256 của 16 shard khớp Hub ([source-checkpoint-sha256.json](source-checkpoint-sha256.json), [convert.log](convert.log)). Convert bằng `tools/prepare_olmoe.py` trong 6 phút 18 giây, ra 30GB: expert INT8 theo hàng, dense và norm giữ BF16. Hash của checkpoint đã convert nằm trong `result.json` → `checkpoint`.
- **Cấu hình engine:**
  - dense BF16 (`MOE_DENSE_STORAGE=bf16`, 2.459.435.008 byte), embedding đọc từ đĩa, thinking off;
  - CPU-only (`--gpu cpu`), vì dense FP32 trên GPU khoảng 4,9GB, không vừa GTX 1650 4GB;
  - CPU4, cache 64MiB/layer × 48 (14 trên 128 expert mỗi layer), context 512, max64, victim, lookahead off, batch off.
- **Workload:** `mixed` gồm 4 prompt, có follow-up. Chạy 8 request liên tiếp với `--keep-cache`.
- **Điều kiện đo:**
  - mỗi mức chạy một lượt, thứ tự 10240 rồi 8192MiB;
  - guard hợp tác đặt thấp hơn trần 1024MiB;
  - trước mỗi lượt, checkpoint bị xóa khỏi page cache (8,1GB → 0 byte, kiểm bằng `fincore`);
  - `memory.max` của scope engine được đọc lại và khớp ở cả hai lượt.

HEAD f695629, git_dirty false, binary 9fedf60d57ca9fa0…, i5-10500H, 15,4GiB RAM. [preflight](preflight.json): eligible, available 9736MiB, pswpout không đổi, AC. [console](console.log).

## Kết quả

| Trần | Hoàn thành | oom_kill | VmSwap | Đỉnh scope current / anon / file (MiB) | RSS sau request cuối |
| --- | --- | ---: | ---: | --- | ---: |
| 10240MiB ([limit-10240](limit-10240/)) | 8/8 | 0 | 0 | 10240 / 5755,7 / 4457,4 | 5762MiB |
| 8192MiB ([limit-8192](limit-8192/)) | 8/8 | 0 | 0 | 8192 / 5740,3 / 4518,0 | 5749MiB |

| Turn 0–7 | Trần 10240 | Trần 8192 |
| --- | --- | --- |
| TTFT (trung vị, min–max) | 9,8s (8,0–20,6) | 11,0s (9,1–24,9) |
| Decode tok/s (trung vị, min–max) | 1,34 (1,06–1,59) | 1,13 (0,90–1,37) |
| Total/request (trung vị) | 61,5s | 72,1s |
| Tỉ lệ thời gian chờ đọc expert trên total | 0,63–0,76 | 0,58–0,72 |
| Tốc độ đọc file expert | 1,60–2,20 GB/s | 1,43–1,67 GB/s |
| Cache hit expert RAM | 38–73% | 38–73% (giống hệt) |
| Payload expert đọc/request | 60,6–91,0GB | 60,6–91,0GB |

- Text, usage và số lần cache hit/miss giống hệt nhau giữa hai mức, vì cùng routing và cùng cache. Chỉ thời gian đọc khác nhau.
- Mẫu câu trả lời (64 token):
  - turn0: "Mars is one of the eight planets in our solar system, and it is the fourth planet from the Sun…"
  - turn1 (follow-up): "You just asked about **Mars**…"
  - turn2: "A **Python dictionary** is a built-in data structure that stores **key-value pairs**…"
- **Thành phần anon ~5,6GiB:** dense BF16 2,29GiB, cache expert tối đa 2,96GiB, KV 30MiB, còn lại là buffer/allocator/tokenizer. Phần file (~4,5GB) là page cache của checkpoint nằm trong scope, kernel thu hồi khi chạm trần.

### Diễn giải

- **Đạt cả ba tiêu chí nghiệm thu** của nhánh Qwen3:
  - parity logits trên fixture (đạt từ trước);
  - chạy hết workload dưới trần 10GB, và cả 8GB, không OOM, không swap;
  - có tok/s thật trên i5.
- Như vậy model có checkpoint 30GB đã chạy được với RAM engine khoảng 5,7GiB, tức khoảng 19% kích thước checkpoint.
- **Nút thắt là I/O, không phải RAM.**
  - Cache 14/128 expert mỗi layer chỉ đạt hit 38–73%.
  - Mỗi request 64 token đọc lại 60–91GB payload expert, tức 2–3 lần toàn bộ checkpoint.
  - 58–76% thời gian là chờ đọc file.
  - Siết trần từ 10 xuống 8GB làm page cache nhỏ đi, nên tốc độ đọc giảm từ 1,6–2,2 xuống 1,4–1,7 GB/s, decode giảm từ 1,34 xuống 1,13 tok/s.
- **Hướng cải thiện (chưa đo):**
  - cache lớn hơn trong phần RAM còn trống: anon mới khoảng 5,7 trên 10GB;
  - expert INT4 để mỗi lần đọc nhỏ đi một nửa;
  - dense FP16 trên GPU để i5 dùng được VRAM.

## Giới hạn

- Một lượt mỗi mức, một máy, CPU-only, 8 request × 64 token. Không có đối chứng tier on/off và không so tốc độ với OLMoE.
- Chưa đánh giá chất lượng ngoài việc đọc câu trả lời. Chưa so logits với Transformers trên model 30B thật, vì máy không đủ RAM để chạy bản tham chiếu. Parity chỉ chứng minh trên fixture nhỏ ngẫu nhiên (`tests/qwen3moe_reference.rs`). *Bổ sung 20/11: đã có parity trên model 30B thật bằng tham chiếu chạy từng layer, xem [hồ sơ parity](../2026-11-20-real-model-parity-i5/README.md).*
- Tốc độ đọc là payload engine chia cho thời gian đọc của engine, không phải I/O thiết bị. Trần là cgroup của riêng engine, không phải RAM vật lý của máy.
- Thinking tắt. Bật thinking sẽ sinh nhiều token hơn trước khi ra câu trả lời.

## Bảo toàn payload

Hai run được chuyển nguyên từ `benchmarks/runs/i5/` vào `limit-10240/` (run `20261117T044548.140906Z-ram-a25a0147…`) và `limit-8192/` (run `20261117T045514.352650Z-ram-3dfbbd7b…`). Kèm theo là preflight, console, log kiểm SHA/convert và hash checkpoint nguồn. Toàn bộ được khóa bằng CHECKSUMS.sha256 (30 file, không gồm README).

```bash
sha256sum -c CHECKSUMS.sha256
```
