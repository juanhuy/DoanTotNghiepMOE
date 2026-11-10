# Prefetch theo router: prefill bật/tắt và coverage lookahead, i5, 10/11/2026

## Câu hỏi

Bước S4 của [PLAN](../../../docs/PLAN.md), kiểm hai cơ chế prefetch riêng:

1. **Prefill nạp trước expert router đã chọn** (công việc đã biết): ở mỗi layer, nạp song song tối đa 8 expert đang thiếu, phần còn lại nạp theo nhu cầu. Câu hỏi: preload có giảm TTFT/prefill so với demand-only không?
2. **Decode lookahead** (dự đoán layer kế): gate chỉ mở khi layer hiện tại không miss. Câu hỏi: trong workload RAM cố định, cơ chế có cơ hội chạy không?

## Cơ chế 2: coverage lookahead hồi cứu (không đo mới)

Lấy từ raw [S1](../2026-11-10-ram-window-i5/README.md) (24 request mixed, max64, cache64MiB/layer, CPU4, lookahead off nhưng gate vẫn được đánh giá và đếm). Tổng 1512 token decode:

| Counter | Giá trị |
| --- | ---: |
| evaluated_layers | 24192 |
| no_next_layer (layer cuối) | 1512 |
| blocked_by_misses | 22665 |
| eligible_layers | 15 (0,066% của 22680) |

Histogram số expert miss/8 ở layer hiện tại: 0→15, 1→62, 2→394, 3→1073, 4→3173, 5→5778, 6→6506, 7→4280, 8→1399. Với cache 64MiB/layer (khoảng 10/64 expert mỗi layer), hầu như mọi layer decode đều có miss, nên gate (dự đoán chỉ khi I/O rảnh) gần như không mở. Không nới gate hoặc tăng cache để tạo cơ hội; kết luận: **chưa có cơ hội chứng minh lợi ích lookahead ở cấu hình này**. Chức năng được kiểm bằng fixture: 10 test lookahead/prefetch PASS (logits giữ nguyên giữa các mode, used/unused buffer, hủy pending, lỗi file được đếm, phân biệt miss một phần/toàn phần).

## Cơ chế 1: prefill prefetch on/off

### Protocol

Công tắc mới `MOE_PREFILL_PREFETCH` (xem [ARCHITECTURE](../../../docs/ARCHITECTURE.md#prefetch-prefill-theo-router-moe_prefill_prefetch-1011)) và counter preload. Trước đây preload lỗi bị bỏ qua im lặng; nay được đếm, đường demand vẫn nạp lại. Protocol `prefill-prefetch-long-v1` chốt và commit vào [SCENARIOS](../../SCENARIOS.md#prefetch-prefill-bậttắt-s4-prefill-prefetch-long-v1-1011) (commit ed0f3b4) trước khi đo.

- Workload long-prefill v1 (bakery 64/library 128 token, warmup turn0–1 thu riêng, đo turn2–3). ABBA on→off→off→on, 4 process mới, `--gpu cpu`, CPU4/victim/cache64MiB-layer/context512, Gate–Up paired, batch off, lookahead off, max1.
- Text/usage/finish **và 5 work counter phải khớp** mọi block (prefetch chỉ đổi thời điểm nạp). Rule đối xứng: on/off ≤1/1,05 ở cả 4 → có lợi; ≥1,05 ở cả 4 → có hại; còn lại → không nhất quán.

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --prefill-prefetch-model-ab \
  --expert-batch-workload long-prefill --threads 4 --cache 64 --context 512 \
  --max-tokens 1 --lookahead off
```

### Provenance và kiểm trước đo

- Run 20261110T074027.992428Z-prefetch-model-ffcb630bf1c74503b9188c3a68702e14, HEAD ed0f3b4, git_dirty false cả 4 block. Binary 64048653c5016318647985aaf44b3af845e301910ff3c10d0e15e2e9c3d46fcb; source khớp workspace; checkpoint trùng 09/11.
- python3 tools/check.py --rust --release --web PASS (Rust 136 tests/profile = 117 lib + 6 + 7 + 6, 8 ignored, fmt/Clippy; 99 Python; 7 Node). Test mới: on/off cho token, file loads/bytes/hits bằng nhau trên hai fixture; preload lỗi được đếm trên bản sao checkpoint bị cắt. Fixture server on/off text và loads khớp.
- [preflight](preflight.json): available ≥8789MiB, pswpout không tăng. [preview](preview.json) dry-run trên tree sạch.

### Kết quả

COMPLETED 16/16; text/usage/finish và 5 work counter khớp mọi block. On: 128 preload/request (8 expert × 16 layer) trên 871–964 lần nạp, tức 13–15%; off: 0; không lỗi preload.

**Rule: không nhất quán** (không đạt "có lợi" cũng không đạt "có hại").

| Cặp | Prompt | TTFT on/off (ms) | TTFT ratio on/off | prefill_expert_io on/off (ms) |
| --- | --- | --- | ---: | --- |
| 0 | bakery64 | 8194.8 / 6609.2 | 1.240 | 4350.9 / 3080.6 |
| 0 | library128 | 12259.9 / 10464.0 | 1.172 | 5775.4 / 3829.9 |
| 1 | bakery64 | 6609.1 / 6891.2 | 0.959 | 2913.6 / 3221.7 |
| 1 | library128 | 9474.8 / 10214.0 | 0.928 | 3069.1 / 3373.8 |

Warmup (loại): 1,046/1,182/0,934/1,016. Chiều kết quả đổi theo cặp: block on đầu tiên (block0) có I/O cao hơn hẳn block on cuối (bakery 4350.9 vs 2913.6ms cùng mode), lớn hơn mọi chênh on/off. Biến động giữa process/OS cache áp đảo hiệu ứng của preload. Attention/LM head không đổi có hệ thống.

RSS max on 3018.9/3027.7MiB, off 2918.9/2921.7MiB: preload giữ thêm khoảng 100MiB tại đỉnh. VmSwap 0; MemAvailable min ≥5767MiB; pswpout tăng 341 page trong block0, sau đó không đổi.

Chú ý: TTFT tuyệt đối trong lượt này cao hơn các lượt CPU trước trong ngày; không so giữa các run (binary và trạng thái máy khác).

## Kết luận và giới hạn

- **Prefill prefetch:** cơ chế chạy đúng (counter, parity, đếm lỗi), nhưng chỉ phủ 13–15% số lần nạp do giới hạn 8 expert/layer. Trong một ABBA, hiệu ứng lên TTFT không nhất quán và nhỏ hơn biến động giữa các process. Chưa chứng minh lợi ích; tốn thêm khoảng 100MiB RSS đỉnh. Giữ mặc định on (không có bằng chứng để đổi).
- **Decode lookahead:** gate đủ điều kiện 0,066% ở cache 64MiB/layer; chưa có cơ hội chứng minh lợi ích, không tối ưu thêm predictor.
- Một lượt, max1; không đo decode, cache khác, GPU hoặc i7. Kết luận hợp lệ cho báo cáo: "cơ chế prefetch dựa trên router đã được hiện thực và kiểm chức năng; lợi ích hiệu năng chưa được chứng minh trong cấu hình đã đo".

## Bảo toàn payload

Chuyển nguyên 37 file run vào run/, thêm preflight/preview; khóa 39 file bằng CHECKSUMS.sha256.

```bash
sha256sum -c CHECKSUMS.sha256
```
