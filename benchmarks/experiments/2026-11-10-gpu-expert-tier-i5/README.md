# Tầng VRAM expert bật/tắt khi GPU dense bật, i5, 10/11/2026

## Câu hỏi và protocol

Tiếp theo [đối chứng backend S3](../2026-11-10-backend-cpu-gpu-i5/README.md): khi dense đã chạy cuBLAS, tầng VRAM expert (mỗi expert bị evict khỏi RAM được upload đồng bộ lên VRAM) làm TTFT/prefill tốt hơn hay kém hơn? Công tắc mới `MOE_GPU_EXPERT_TIER` (xem [ARCHITECTURE](../../../docs/ARCHITECTURE.md#công-tắc-tầng-vram-expert-moe_gpu_expert_tier-1011)) tách riêng tầng này, dense GPU giữ nguyên hai phía. Protocol `gpu-tier-long-v1` chốt và commit vào [SCENARIOS](../../SCENARIOS.md#tầng-vram-expert-bậttắt-s3-gpu-tier-long-v1-1011) (commit 4f319b7) trước khi đo.

- Workload long-prefill v1 (bakery 64/library 128 token, warmup turn0–1 thu riêng, đo turn2–3, session mới mỗi request, reset chỉ trước turn0). ABBA tier on→off→off→on, 4 process mới, `--gpu auto` cả bốn. Max1, CPU4/victim/cache64MiB-layer/context512, Gate–Up paired, batch off, FP32KV/KVreuse off/lookahead off/trace on/greedy.
- Rule đối xứng chốt trước: on/off ≥1,05 ở cả 4 (TTFT VÀ prefill) → tắt nhanh hơn; ≤1/1,05 ở cả 4 → tầng VRAM có lợi; còn lại → không nhất quán.

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu auto --gpu-tier-model-ab \
  --expert-batch-workload long-prefill --threads 4 --cache 64 --context 512 \
  --max-tokens 1 --lookahead off
```

## Provenance và kiểm trước đo

- Run 20261110T070546.777634Z-gpu-tier-model-bc406499970b4639bbaf80af4369fdf3, HEAD 4f319b7, git_dirty false cả 4 block. Binary mới bae0afbea6a1be12a568f182d36232f54f684811b4de9b4ed48ecea724b29e90 (thêm công tắc, mặc định on); source hash khớp workspace; checkpoint trùng 09/11. Không so số tuyệt đối với S2/S3 (binary khác).
- python3 tools/check.py --rust --release --web PASS: Rust 134 tests/profile (115 lib + 6 + 7 + 6), 8 ignored, fmt/Clippy; 97 Python; 7 Node. Test CUDA mới tier off EXECUTED (dense GPU, có eviction RAM, 0 upload/0 VRAM load, logits so CPU <2e-5). Fixture server auto on/off: status đúng, text khớp.
- [preflight](preflight.json): available ≥9268MiB, pswpout không tăng, không engine/build khác. [nvidia-smi trước chạy](nvidia-smi-pre.csv): 242MiB desktop, 0%.

## Kết quả

COMPLETED 16/16, không watchdog violation; text/usage/finish/prompt tokens khớp mọi turn (đầu ra 1 token xuống dòng, parity text yếu). Cả 4 block GPU enabled/dense offloaded. Block off: 0 VRAM upload/load/resident, VRAM logical 1417MiB (chỉ dense). Block on: 298–299 expert resident, VRAM logical 3210–3216MiB.

**Rule: tắt tầng VRAM nhanh hơn — ĐẠT 4/4.**

| Cặp | Prompt | TTFT on/off (ms) | TTFT ratio on/off | expert_compute on/off (ms) | File loads on/off | VRAM upload on (MiB) |
| --- | --- | --- | ---: | --- | --- | ---: |
| 0 | bakery64 | 5826.7 / 4029.4 | 1.446 | 1250.5 / 1152.4 | 800 / 871 | 5240 |
| 0 | library128 | 7339.7 / 5563.1 | 1.319 | 2261.5 / 2065.9 | 834 / 904 | 5438 |
| 1 | bakery64 | 4084.1 / 3371.9 | 1.211 | 1438.6 / 1205.8 | 800 / 871 | 5240 |
| 1 | library128 | 5678.4 / 3973.4 | 1.429 | 2378.7 / 2093.8 | 834 / 904 | 5438 |

Prefill ratio bằng TTFT ratio tới 0,001. Tắt tầng VRAM làm TTFT giảm 17,4–30,8% (0,71–1,80s). Attention và LM head gần như không đổi (ratio 0,98–1,05), xác nhận dense GPU giống nhau hai phía.

Warmup (loại theo protocol): 1,720/1,481 ở cặp0, 1,119/**0,951** ở cặp1. Có một warmup (library cặp1) mà tier on nhanh hơn; công bố như ngoại lệ.

### Diễn giải theo thành phần (timer chồng lấn, không cộng thành wall)

- Tầng VRAM tránh được 70–71 lần đọc file mỗi request (khoảng 427–421MiB payload), nhưng phải upload 5240–5438MiB lên VRAM. Thực tế pool 298 slot bị xoay vòng: mỗi expert evict khỏi RAM đều được đẩy lên, rồi đẩy expert VRAM cũ ra.
- expert_compute khi tắt tầng thấp hơn ở cả 4 so sánh (ratio on/off 1,085–1,193) dù SwiGLU giống hệt. Điều này phù hợp với giả thuyết tranh chấp CPU/bộ nhớ do upload đồng bộ và driver, nhưng chưa có profiler nên chưa chứng minh nguyên nhân.
- expert_io (gồm upload) biến động lớn giữa block; cặp1 bakery lại có expert_io on thấp hơn off (1048.6 vs 1934.7ms) nhưng TTFT on vẫn cao hơn. Không quy toàn bộ chênh cho một timer.

### Tài nguyên

RSS max on 3408.9/3414.5MiB, off 3403.1/3401.6MiB — gần như bằng nhau (pool là VRAM, pinned staging chỉ một expert); VmSwap 0 mọi block; MemAvailable min ≥6773MiB. Tắt tầng giải phóng khoảng 1,8GiB VRAM logical. pswpout hệ thống tăng tổng cộng khoảng 4082 page trong run, engine không bị swap.

## Kết luận và giới hạn

Câu hỏi "chi phí truyền qua lại có ăn hết lợi ích GPU không" có câu trả lời tách bạch: **dense trên GPU có lợi (S3 backend), còn tầng VRAM expert theo thiết kế hiện tại (victim, upload mọi expert bị evict) làm chậm TTFT 17–31% trên workload này.** Cấu hình tốt nhất quan sát được trên i5 là GPU dense + tầng VRAM expert tắt.

Chỉ một lượt, 2 cặp mỗi prompt, max1; một warmup đi ngược chiều; không đo decode, chat nhiều lượt, cache budget khác, i7 hoặc policy inclusive. Không đổi mặc định từ một lượt: mặc định vẫn on cho tới khi có phép decode/nhiều lượt. Kết quả cũng không bác bỏ ý tưởng tầng VRAM nói chung — một thiết kế có admission (chỉ đẩy expert nóng) hoặc upload bất đồng bộ có thể khác, nhưng chưa được thử. VRAM là số logical theo engine.

## Bảo toàn payload

Chuyển nguyên 37 file run vào run/, thêm preflight/preview/nvidia-smi-pre; khóa 40 file bằng CHECKSUMS.sha256.

```bash
sha256sum -c CHECKSUMS.sha256
```
