# Batch prefill trên prompt dài S2 trên i5, 10/11/2026

## Câu hỏi và protocol

Bước S2 của [PLAN](../../../docs/PLAN.md), protocol `expert-batch-long-v1` chốt và commit vào [SCENARIOS](../../SCENARIOS.md#batch-prefill-trên-prompt-dài-s2-expert-batch-long-v1-1011) (commit 9cf3b0b) trước khi đo. Câu hỏi: với prompt 64 và 128 token, MOE_EXPERT_BATCH_PREFILL=on có giảm TTFT/prefill so với off không? Chạy sau khi [gate an toàn S1](../2026-11-10-ram-window-i5/README.md) đạt.

- Một ABBA off→on→on→off, 4 process mới; mỗi process 4 request, mỗi request một session mới: turn0–1 warmup bakery/library (thu riêng), turn2–3 đo bakery/library. Reset chỉ trước turn0, keepcache. 16 request = 8 sample chính + 8 warmup.
- Max1, CPU4/victim/cache64MiB-layer/context512, Gate–Up paired, FP32KV/KVreuse off/lookahead off/trace on/greedy. Chỉ khác batch flag.
- Primary: TTFT VÀ prefill off/on ≥1,05 ở từng prompt trong CẢ hai cặp (cặp0 = block0/1, cặp1 = block3/2).
- Prompt đóng băng trong [prompts.json](prompts.json): bakery 64 token, library 128 token sau chat template, kèm SHA256 text và token IDs. Đếm bằng cùng render_chat/tokenizer encode như engine; harness kiểm usage.prompt_tokens ở mọi request.

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --expert-batch-model-ab \
  --expert-batch-workload long-prefill --threads 4 --cache 64 --context 512 \
  --max-tokens 1 --lookahead off
```

## Provenance và preflight

- Run 20261110T045405.090137Z-expert-batch-model-65369188464f4382b341cac1fe4308be, HEAD 9cf3b0b, git_dirty false ở cả4block. Binary d12a3939… và 24 file Rust/Cargo trùng snapshot model 09/11; chỉ harness Python khác (long-prefill workload, tests). Checkpoint manifest trùng 09/11. Source không đổi giữa các block (harness tự kiểm).
- Kiểm trước đo: tools/check.py PASS 91 Python. Rust/release/fixture logits/prefix/cancel/trace không chạy lại vì engine source không đổi; dùng lại kiểm 09/11 (133 tests/profile, HTTP fixture batch on/off).
- [preflight](preflight.json) 3 mẫu cách 5s: available ≥9996MiB, pswpout 480777 không đổi, không engine/build khác. [preview](preview.json) là dry-run trên tree sạch.

## Kết quả

COMPLETED 16/16, không watchdog violation. Mọi request prompt 64/128 token đúng, completion 1. Text/usage/finish và 5 cache/file counter khớp ở mọi turn của 4 block (text đầu ra là một token xuống dòng; parity 1 token là kiểm yếu, không thay logits parity). Block on: batch2/batch4 calls bakery 473/1683, library 518/3708; block off 0/0. File loads/payload turn đo bakery 871/5494128640B, library 904/5702287360B.

**Primary ĐẠT: 4/4 so sánh có TTFT và prefill ≥1,05.**

| Cặp | Prompt | TTFT off/on (ms) | TTFT ratio | Prefill ratio | expert_compute off/on (ms) | Compute ratio |
| --- | --- | --- | ---: | ---: | --- | ---: |
| 0 | bakery64 | 4445.0 / 4108.1 | 1.0820 | 1.0822 | 1124.9 / 794.3 | 1.4163 |
| 0 | library128 | 7728.8 / 7063.9 | 1.0941 | 1.0943 | 2034.7 / 1360.7 | 1.4953 |
| 1 | bakery64 | 4063.5 / 3816.8 | 1.0646 | 1.0645 | 1106.3 / 767.6 | 1.4412 |
| 1 | library128 | 7243.3 / 6650.4 | 1.0892 | 1.0896 | 2010.0 / 1335.0 | 1.5056 |

Warmup (không tính vào rule, công bố đầy đủ): TTFT ratio 1.1135/1.1019 ở cặp0 và **1.0335**/1.0880 ở cặp1. Warmup bakery cặp1 dưới 1,05; với protocol đã chốt nó không thuộc sample, nhưng cho thấy biên lợi ích ở prompt 64 token nhỏ và nhạy điều kiện.

Expert I/O biến động theo block (turn đo bakery off0 1122.7, on1 1130.7, on2 851.3, off3 762.3 ms) do OS cache/I/O, không phải batch flag; các timer không cộng thành wall. TTFT giảm 247–665ms, cùng bậc với expert_compute giảm 330–675ms.

RSS sampled max 3020.0–3026.0MiB, process VmSwap 0, MemAvailable min ≥7235MiB. pswpout hệ thống tăng 480777→677792 page (≈769MiB) trong block0–3, chủ yếu block0–2; engine không bị swap, tác nhân chưa xác định.

## Kết luận và giới hạn

Trong workload đã thử (2 prompt 64/128 token, max1, CPU4, cache64, một ABBA), batch prefill giảm TTFT 6,1–8,6% (ratio off/on 1,065–1,094) và expert_compute 29,4–33,6%; lợi ích lớn hơn ở prompt 128 token. Đây là một lượt với 2 cặp/prompt, không phải ước lượng thống kê; không suy cho decode, chat nhiều lượt, prompt khác, i7 hoặc GPU. Không tự bật mặc định; giữ opt-in và ghi phạm vi dùng "prefill prompt ≥64 token trên i5 CPU4". Bổ sung cho kết quả 09/11 (prompt ngắn Mars/Python không đạt), không thay hoặc sửa kết quả cũ.

## Bảo toàn payload

Chuyển nguyên 37 file run từ benchmarks/runs/i5/ vào run/, thêm preflight/preview/prompts; khóa 40 file bằng CHECKSUMS.sha256.

```bash
sha256sum -c CHECKSUMS.sha256
```
