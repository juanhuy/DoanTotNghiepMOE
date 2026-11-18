# Cache expert 64 so với 96MiB/layer cho Qwen3-30B-A3B dưới trần 10GB, i5, 18/11/2026

## Câu hỏi và protocol

Cùng trần cgroup 10240MiB, tăng cache expert của engine từ 64 lên 96MiB/layer, tức từ 14 lên 21 trong 128 expert mỗi layer. Engine dùng thêm khoảng 1,5GiB anon, và page cache của OS bớt lại tương ứng. Như vậy Qwen3 có nhanh hơn không? Protocol [`qwen3-cache-ab-v1`](../../SCENARIOS.md#cache-expert-lớn-hơn-cho-qwen3-dưới-trần-10gb-qwen3-cache-ab-v1-1811) được commit trước khi đo (fba1931).

- **Cấu hình:** giữ nguyên `qwen3-ram-cap-v1`: i5, CPU-only, dense BF16, embedding từ đĩa, mixed × 8, keep-cache, max64, CPU4, trần 10240, guard 9216. Chỉ đổi `--cache`.
- **Thứ tự:** A0 = 64, B1 = 96, B2 = 96, A3 = 64. Mỗi block là một process mới, checkpoint được xóa khỏi page cache trước mỗi block (còn 0 byte, 4/4).
- **Rule:** B nhanh hơn nếu tỉ số B/A của total và decode ≤ 1/1,05 ở cả 16 so sánh. Phân tích bằng [summary.json](summary.json) (`tools/compare_ram_runs.py`).

HEAD fba1931 sạch, binary 9fedf60d… (trùng lượt `qwen3-ram-cap-v1`). [preflight](preflight.json): eligible, available 8276MiB, pswpout 0, AC. [console](console.log).

**Điều kiện máy:** trong lúc đo, Chrome, Slack và VS Code đang mở, nên available thấp hơn lượt 17/11 (9736MiB). Thiết kế ABBA cân bằng nhiễu này giữa A và B. Tuy vậy, tốc độ tuyệt đối không so được với hồ sơ 17/11.

## Kết quả: không nhất quán, giữ 64MiB/layer

| Block | Cache | Decode trung vị | Total trung vị | Đọc file/request (trung vị) | Tốc độ đọc file | Cache expert | Đỉnh anon / file (MiB) |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| A0 | 64 | 0,96 tok/s | 78,5s | 74,6GB | 0,95–1,38 GB/s | 2,96GiB | 5751 / 4640 |
| B1 | 96 | 1,03 tok/s | 80,1s | 59,6GB | 0,95–1,08 GB/s | 4,44GiB | 7259 / 5562 |
| B2 | 96 | 1,05 tok/s | 79,3s | 59,6GB | 0,95–1,04 GB/s | 4,44GiB | 7269 / 5600 |
| A3 | 64 | 0,96 tok/s | 85,3s | 74,6GB | 1,02–1,19 GB/s | 2,96GiB | 5757 / 6102 |

- **`b_faster: false`, `b_slower: false`.**
  - Total B/A: 0,787–1,045. Decode B/A: 0,748–1,080.
  - Cặp A0–B1 gần như ngang nhau (0,79–1,05).
  - Cặp A3–B2 nghiêng về 96 (0,84–0,94).
- Text khớp 16/16. Cả 4 block: 8/8 request, oom_kill 0, VmSwap 0, scope chạm trần 10240.
- **Cache hit tăng nhưng thời gian không giảm.**
  - Hit rate trung vị tăng từ 50% lên 60% (A: 38–73%, B: 47–78%), và lượng payload đọc file giảm 20% (74,6 → 59,6GB/request).
  - Nhưng tốc độ đọc file của B thấp hơn (0,95–1,08 so với 0,95–1,38 GB/s), vì page cache trong cùng trần nhỏ đi nên nhiều lần đọc phải xuống SSD thật hơn.
  - Hai hiệu ứng gần như triệt tiêu nhau.
- **Kết luận:** dưới một trần RAM cố định, chuyển RAM từ page cache sang cache expert của engine không cho lợi ích tốc độ nhất quán. Cấu hình Qwen3 giữ cache 64MiB/layer, vì nhẹ hơn 1,5GiB anon với cùng tốc độ.
- Muốn nhanh hơn thì phải giảm khối lượng cần đọc (expert INT4, routing có tính đến cache) hoặc có thêm bộ nhớ (VRAM). Phân chia lại RAM đã có là không đủ.

## Giới hạn

- Một lượt ABBA, một máy, CPU-only, 8 request × 64 token. Có ứng dụng desktop chạy nền.
- Tốc độ đọc là payload engine chia cho thời gian đọc của engine, không phải I/O thiết bị.
- Chỉ thử hai mức. Mức 128 bị guard loại từ trước (dự tính khoảng 9,9GB > 9216MiB). Kết quả không suy ra cho trần khác hoặc máy khác.

## Bảo toàn payload

Bốn run được chuyển nguyên từ `benchmarks/runs/i5/` vào `block0-cache64/` (`20261118T025152.863142Z-ram-07ed77a9…`), `block1-cache96/` (`20261118T030414.485195Z-ram-0f1ec8c4…`), `block2-cache96/` (`20261118T031625.331194Z-ram-54b0f1d1…`) và `block3-cache64/` (`20261118T032822.908924Z-ram-1cfcca1a…`). Kèm theo là preflight, console và summary. Toàn bộ được khóa bằng CHECKSUMS.sha256 (55 file, không gồm README).

```bash
sha256sum -c CHECKSUMS.sha256
```
