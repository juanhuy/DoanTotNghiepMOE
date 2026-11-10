# Chat/decode với tầng VRAM expert bật/tắt S5, i5, 10/11/2026

## Câu hỏi và protocol

Bước S5 của [PLAN](../../../docs/PLAN.md). Tầng VRAM expert bật/tắt đã đo ở [prefill/max1](../2026-11-10-gpu-expert-tier-i5/README.md); câu hỏi bổ sung: với decode max64 và chat nhiều lượt (follow-up có lịch sử/prefix cache, chuyển phiên), kết quả có cùng chiều không. Protocol `gpu-tier-chat-v1` chốt và commit vào [SCENARIOS](../../SCENARIOS.md#chatdecode-với-tầng-vram-expert-bậttắt-s5-gpu-tier-chat-v1-1011) (commit 2cea0cd) trước khi đo.

- Mỗi process 2 nhóm workload mixed v1: Mars, follow-up "What planet did I just ask about?" cùng phiên với lịch sử thật, Python phiên mới, Love phiên mới. Nhóm 1 (turn0–3) warmup thu riêng, nhóm 2 (turn4–7) đo. Reset chỉ trước turn0/keepcache.
- ABBA tier on→off→off→on, 4 process mới, `--gpu auto` (dense cuBLAS cả bốn), max64, CPU4/victim/cache64MiB-layer/context512, Gate–Up paired, batch off, lookahead off, FP32KV/KVreuse off/trace on/greedy. Tổng 32 request.
- Rule đối xứng trên 8 so sánh: total_ms VÀ decode_ms on/off ≥1,05 ở cả 8 → tắt nhanh hơn.

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu auto --gpu-tier-model-ab \
  --expert-batch-workload mixed --threads 4 --cache 64 --context 512 \
  --max-tokens 64 --lookahead off
```

## Provenance

- Run 20261110T081514.310194Z-gpu-tier-model-55a1e303b6684334940638eace6a0c1e, HEAD 2cea0cd, git_dirty false cả 4 block. Binary 64048653c5016318647985aaf44b3af845e301910ff3c10d0e15e2e9c3d46fcb (engine như S4; Rust 136 tests/profile PASS ở S4); tools/check.py PASS 101 Python. Checkpoint trùng 09/11.
- [preflight](preflight.json): available ≥8402MiB, pswpout không tăng. [nvidia-smi trước chạy](nvidia-smi-pre.csv): 242MiB, 0%.

## Kết quả

COMPLETED 32/32. Text/usage/finish và lịch sử follow-up khớp mọi turn ở cả 4 block: prompt tokens 19/107/22/20, completion 64, cached prompt 11 ở follow-up. Đây là parity text/token-count trên 64 token qua HTTP, không phải logits parity. Block off không có VRAM traffic.

**Rule: tắt tầng VRAM nhanh hơn — ĐẠT 8/8.**

| Cặp | Prompt | total on/off (ms) | total ratio | TTFT ratio | decode ratio | decode tok/s on/off |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | Mars | 21887 / 13427 | 1.630 | 1.317 | 1.693 | 3.33 / 5.64 |
| 0 | Follow-up | 25738 / 17575 | 1.465 | 1.099 | 1.654 | 3.29 / 5.45 |
| 0 | Python | 23535 / 14072 | 1.672 | 1.088 | 1.821 | 3.08 / 5.61 |
| 0 | Love | 25390 / 18267 | 1.390 | 1.291 | 1.404 | 2.81 / 3.94 |
| 1 | Mars | 23214 / 15655 | 1.483 | 1.056 | 1.593 | 3.18 / 5.06 |
| 1 | Follow-up | 25726 / 17303 | 1.487 | 1.011 | 1.757 | 3.25 / 5.71 |
| 1 | Python | 24893 / 13285 | 1.874 | 1.504 | 1.952 | 2.94 / 5.75 |
| 1 | Love | 24857 / 15167 | 1.639 | 1.499 | 1.662 | 2.92 / 4.85 |

Decode tok/s là `decode_tokens_per_second` của engine (mẫu số là decode tokens). Trung vị on 3,13, off 5,53 tok/s; total giảm 28–47% khi tắt. Warmup (loại theo protocol) cùng chiều: total ratio 1,381–1,577.

### Diễn giải theo thành phần

- Ở decode, tầng VRAM phục vụ 2745–3886 lần nạp mỗi request, nên file loads giảm khoảng một nửa (ví dụ Mars 2748 so với 5532). Đổi lại, nó upload 33182–40816MiB lên VRAM mỗi request, expert_vram_upload_ms 8108–10402ms, đồng bộ trên đường xử lý. Đây là đánh đổi bất lợi rõ ràng trên PCIe của GTX1650 Max-Q.
- expert_compute ở decode thấp hơn khi bật tầng (ví dụ 4503 so với 5915ms). Chiều này ngược với prefill, nên giả thuyết tranh chấp CPU ở hồ sơ trước không giải thích được mọi trường hợp; các timer chồng lấn và chưa có profiler.
- Chênh lệch chính nằm ở decode; TTFT cũng tốt hơn khi tắt ở cả 8 so sánh (1,011–1,504), dù 2 so sánh dưới 1,05.

### Tài nguyên

RSS max on 3462.1/3472.3, off 3422.0/3457.8MiB; VmSwap 0 mọi block. MemAvailable min 4415–4757MiB, thấp hơn các lượt trước nhưng vẫn trên sàn watchdog 1536MiB. pswpout hệ thống tăng 5175 page ở block0, sau đó ít; engine không bị swap.

## Kết luận và giới hạn

Với dense trên GPU, tắt tầng VRAM expert nhanh hơn rõ rệt cả ở chat nhiều lượt/decode (total 1,39–1,87×, decode tok/s trung vị 3,13 → 5,53) lẫn prefill (hồ sơ trước). Tiêu chí chốt trước để cân nhắc đổi mặc định MOE_GPU_EXPERT_TIER đã đạt, nhưng phạm vi chỉ là i5 (GTX1650 Max-Q 4GB), cache 64MiB/layer, policy victim. Đổi mặc định là quyết định riêng vì ảnh hưởng máy i7 và UI.

Một lượt, 2 cặp mỗi prompt, 64 token; không đo cache budget khác, inclusive, i7 hoặc logits parity 7B. Kết quả không bác bỏ thiết kế tầng VRAM có admission chọn lọc hoặc upload bất đồng bộ. VRAM là số logical theo engine.

## Bảo toàn payload

Chuyển nguyên 53 file run vào run/, thêm preflight/preview/nvidia-smi-pre; khóa 56 file bằng CHECKSUMS.sha256.

```bash
sha256sum -c CHECKSUMS.sha256
```
