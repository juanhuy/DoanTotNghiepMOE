# Đối chứng backend CPU/GPU S3 trên i5, 10/11/2026

## Câu hỏi và protocol

Bước S3 của [PLAN](../../../docs/PLAN.md), protocol `backend-long-v1` chốt và commit vào [SCENARIOS](../../SCENARIOS.md#đối-chứng-backend-cpugpu-s3-backend-long-v1-1011) (commit c5a7766) trước khi đo. Câu hỏi: trên i5 (GTX1650 Max-Q), `--gpu auto` thay `--gpu cpu` thay đổi TTFT/prefill và công việc expert thế nào với cùng binary/checkpoint/cấu hình?

**Đối chứng gói backend, không phải một yếu tố đơn lẻ:** auto vừa chạy dense (QKV/O, LM head) bằng cuBLAS vừa dùng VRAM còn lại làm tầng lưu expert L1 (victim). Expert SwiGLU tính CPU ở cả hai phía; causal attention/softmax cũng vẫn CPU.

- Workload long-prefill v1 như [S2](../2026-11-10-expert-batch-long-i5/README.md): bakery 64/library 128 token, 4 request/process (warmup turn0–1 thu riêng, đo turn2–3), session mới mỗi request, reset chỉ trước turn0. ABBA cpu→auto→auto→cpu, 4 process mới. Max1, CPU4/victim/cache64MiB-layer/context512, Gate–Up paired, batch prefill off, FP32KV/KVreuse off/lookahead off/trace on/greedy.
- Rule: TTFT VÀ prefill cpu/auto ≥1,05 ở từng prompt trong cả hai cặp.

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu auto --backend-model-ab \
  --expert-batch-workload long-prefill --threads 4 --cache 64 --context 512 \
  --max-tokens 1 --lookahead off
```

## Provenance và kiểm trước đo

- Run 20261110T064618.256640Z-backend-model-7726284126004623b5a1fe094ec66730, HEAD c5a7766, git_dirty false ở cả 4 block. Binary d12a3939… và source Rust/Cargo trùng snapshot 09/11; checkpoint trùng 09/11.
- Probe trước đo (PROGRESS 10/11): 11/11 GPU unit test với cuBLAS GEMV/SGEMM thực thi; lib 114 PASS/8 ignored, 3 test model EXECUTED trên CUDA (victim cache logits so CPU <2e-5, tầng không trùng, reset pool). Harness backend có 4 test mới; tools/check.py PASS 95 Python.
- [preflight](preflight.json): available ≥9240MiB, pswpout không tăng, không engine/build khác. [nvidia-smi trước chạy](nvidia-smi-pre.csv): 242MiB đã dùng bởi desktop, tải 0%. [preview](preview.json) dry-run trên tree sạch.

## Kết quả

COMPLETED 16/16, không watchdog violation. Text/usage/finish/prompt tokens khớp mọi turn (đầu ra 1 token xuống dòng; parity text yếu, không thay logits parity 7B). Block cpu: GPU disabled/0 VRAM, không map CUDA. Block auto: gpu enabled, dense offloaded, VRAM logical 1417MiB lúc start và 3210–3216MiB sau khi pool expert dùng (298–299 expert resident). Cùng một thông báo status ghi "0 L1 Experts" lúc start trước khi pool được nạp.

**Rule ĐẠT 4/4.**

| Cặp | Prompt | TTFT cpu/auto (ms) | TTFT ratio | attention cpu/auto (ms) | expert_compute cpu/auto (ms) | expert_io cpu/auto (ms) |
| --- | --- | --- | ---: | --- | --- | --- |
| 0 | bakery64 | 6533.6 / 4038.2 | 1.618 | 2390.9 / 138.1 | 1353.4 / 1247.4 | 2681.7 / 1279.7 |
| 0 | library128 | 10230.3 / 6518.5 | 1.569 | 4334.5 / 459.3 | 2256.1 / 2673.9 | 3497.7 / 1805.7 |
| 1 | bakery64 | 5124.1 / 4150.4 | 1.235 | 2093.0 / 138.5 | 1318.0 / 1237.4 | 1608.8 / 1407.5 |
| 1 | library128 | 8892.2 / 5650.1 | 1.574 | 4724.0 / 455.5 | 2338.4 / 2353.1 | 1682.2 / 1320.4 |

Prefill ratio bằng TTFT ratio tới 0,001. LM head cpu/auto 14–17 / 2,8ms. Warmup (loại theo protocol): TTFT ratio **0,887**/1,143 ở cặp0 và 1,153/1,329 ở cặp1. Warmup bakery đầu tiên của block auto chậm hơn cpu, trùng lúc pool VRAM rỗng được nạp lần đầu (upload 4812MiB trong request đó).

### Đóng góp theo thành phần (timer chồng lấn, không cộng thành wall)

- **Dense/attention trên GPU là thành phần lớn nhất:** attention_ms (RMSNorm, chiếu QKV, RoPE, causal attention CPU, chiếu O) giảm khoảng 2,0–4,3s mỗi request, ratio 9,4–17,3×. Mức giảm này lớn hơn mức giảm TTFT ở 3/4 so sánh (trừ bakery cặp0, nơi expert I/O block cpu cao bất thường), tức các thành phần khác trong auto tốn thêm thời gian.
- **Tầng VRAM expert lợi ít, tốn nhiều băng thông:** ở turn đo, VRAM phục vụ 70–71 expert, file loads giảm 871→800 (bakery) và 904→834 (library), payload file ít hơn khoảng 427–421MiB. Nhưng mỗi expert bị evict khỏi RAM đều được upload lên VRAM: upload 5240–5438MiB/request, expert_vram_upload_ms 1274–1409ms; download 421–427MiB/89–95ms. Upload chạy đồng bộ trong `promote_to_vram` khi evict. Chưa tách được lợi/hại ròng của tầng này vì engine chưa có công tắc riêng.
- **expert_compute không giảm** (ratio 0,844–1,085): SwiGLU vẫn CPU; chênh có thể do tranh chấp CPU với đường upload/driver, chưa xác định.
- expert_io của block cpu biến động lớn giữa block0 và block3 (bakery 2681.7 vs 1608.8ms), nên ratio cặp0 lớn hơn cặp1. Đây là biến động OS cache/I/O giữa process, không do backend.

### Tài nguyên

| Block | RSS/HWM max (MiB) | VmSwap max (MiB) | MemAvailable min (MiB) |
| --- | ---: | ---: | ---: |
| cpu0 | 3028.9 | 0.1 | 6327.2 |
| auto1 | 3411.8 | 0.0 | 6213.1 |
| auto2 | 3411.8 | 0.0 | 6262.5 |
| cpu3 | 3032.9 | 0.0 | 6356.2 |

Auto tốn thêm khoảng 380MiB host RSS (CUDA context/thư viện, pinned staging); dense vẫn giữ một bản trên host. Block cpu0 có VmSwap 0,1MiB (khác 0, không gọi là "không swap"). pswpout hệ thống tăng 631 page ở block0, gần như không đổi ở các block sau. VRAM là logical theo engine (dense 1417MiB + pool), không phải toàn bộ cấp phát vật lý của driver/context; nvidia-smi sau run trở về 242MiB.

## Kết luận và giới hạn

Trên i5, workload prompt 64/128 token, max1, CPU4, cache64: chọn backend auto giảm TTFT/prefill (ratio 1,235–1,618, tức giảm 19–38%) ở cả 4 so sánh, chủ yếu nhờ dense/attention chạy cuBLAS. Tầng VRAM expert phục vụ khoảng 8% miss nhưng tạo khoảng 5,3GiB upload PCIe mỗi request. Lợi ích ròng của riêng tầng này chưa được chứng minh và là câu hỏi cho đối chứng sau, cần công tắc VRAM expert riêng.

Một lượt, 2 cặp mỗi prompt; warmup đầu của auto bất lợi; không đo decode/chat nhiều lượt/i7. Không suy GPU tính expert; không gọi VRAM logical là cấp phát vật lý. Không so số tuyệt đối với S2 (cpu block ở đây có I/O khác).

## Bảo toàn payload

Chuyển nguyên 37 file run từ benchmarks/runs/i5/ vào run/, thêm preflight/preview/nvidia-smi-pre; khóa 40 file bằng CHECKSUMS.sha256.

```bash
sha256sum -c CHECKSUMS.sha256
```
