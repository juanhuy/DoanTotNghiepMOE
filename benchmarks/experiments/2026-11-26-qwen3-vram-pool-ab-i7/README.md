# Pool expert VRAM không cần dense GPU, Qwen3-30B-A3B, i7, 26/11/2026

**Nhãn: P1.** Kết luận: **B nhanh hơn ở 16/16 so sánh** theo rule chốt trước; text khớp 16/16; không OOM, không swap.

Protocol [`qwen3-vram-pool-ab-v1`](../../SCENARIOS.md#pool-expert-vram-không-cần-dense-gpu-cho-qwen3-trên-i5-qwen3-vram-pool-ab-v1-2411) chạy độc lập trên testbed `i7` (`i7-14650HX`, 16GB RAM, RTX 5060 Laptop GPU 8GB / 7,53 GiB khả dụng). Kèm một lượt khám phá (P3, không thuộc A/B) với `--gpu-dense on`.

## Chuẩn bị (trước A/B)

- **Revision & kiểm thử:** `main` commit `ee990c3ce6eae2cf4905638742306417b8b781e1`, `git_dirty: false`, binary `966fb8659f882fc7…`.
  - `python3 tools/check.py --rust --release` **PASS**.
  - Test CUDA thật `gpu_expert_pool_without_dense_gpu_matches_cpu_with_placement` (`cargo test --release --lib gpu_expert_pool_without_dense -- --nocapture`) in **`EXECUTED GPU pool without dense on CUDA: placement, logits/text parity`** và **PASS**.
- **Checkpoint & hồ sơ tần suất:**
  - Dùng lại checkpoint Qwen3-30B-A3B INT8 (30GB) từ nhiệm vụ 18/11; kiểm SHA-256 đối chiếu với i5: **`mismatched files: 0 of 478`**.
  - Dùng đúng file hồ sơ tần suất của i5 [`placement-long-prefill.txt`](../2026-11-24-qwen3-vram-pool-ab-i5/placement-long-prefill.txt) (SHA-256 `3a5cea5d63f1150b62b0d72248c99d1e32bc78e203f3be187512558d52b5ca35`).
- **Preflight ([preflight.json](preflight.json)):** `eligible: true`, `on_ac: true`, `min_available_mib: 10354,7`, `pswpout` ổn định `[176, 176, 176]`, `gpu_throttle: []`, đóng Chrome/Slack.
  - Ghi chú driver: hệ thống dùng driver `595.99.02` (CUDA `13.2`, `DynamicPowerManagement: 2`); khi không bật persistence mode, GPU tự `suspended` trong 5s sleep của preflight và báo `sw_power_cap` nhất thời lúc thức dậy ở xung `225–232 MHz`. Đã bật Persistence Mode (`nvidia-smi -pm 1`) trước preflight và trong khi đo để giữ GPU ở trạng thái sẵn sàng hợp lệ.

## Kết quả A/B (`--testbed i7`)

- Thứ tự `A B B A`, mỗi block một process mới, trần cgroup `10240MiB`, checkpoint bị xóa khỏi page cache trước mỗi block (`checkpoint_resident_bytes_after_evict: 0` kiểm bằng `fincore`). Chạy dưới `systemd-inhibit --what=idle:sleep`.
- Block: [A0 CPU](block0-A-cpu/), [B1 pool](block1-B-pool/), [B2 pool](block2-B-pool/), [A3 CPU](block3-A-cpu/). [console.log](console.log), [summary.json](summary.json).
- Cả 4 block hoàn thành `8/8` request, `oom_kill: 0`, `VmSwap: 0 KiB`.

| | A (CPU-only) | B (pool VRAM 1594 expert, dense CPU) |
| --- | ---: | ---: |
| Decode tok/s (trung vị 16 turn) | 1,92 | **3,74** |
| Total/request (trung vị) | 47,12s | **25,78s** |
| TTFT (trung vị) | 9,16s | **6,25s** |
| Payload expert đọc từ file/request (trung vị) | 74,60GB | **25,37GB** |
| Cache miss B/A theo cặp | | 0,309–0,483 (tổng 48.474 / 128.467 = **0,377**) |
| Cache hit rate (min–max) | 38,2%–72,9% | **70,1%–90,6%** |
| Expert tính trên GPU/request | 0 | 11.514–27.611 route (~0,66s GPU) |
| Thời gian expert trên CPU/request (trung vị theo block) | 13,08 / 13,15s | 5,13 / 5,25s |
| `VmHWM` sau request cuối | 5,63 / 5,62 GiB | 5,90 / 5,90 GiB |
| Peak scope `current / anon / file` (MiB) | A0: 9237,0 / 5753,2 / 3551,6<br>A3: 10240,0 / 5743,9 / 6531,2 | B1: 9981,4 / 5807,8 / 5089,3<br>B2: 10240,0 / 5806,3 / 7671,0 |
| VRAM cấp phát | 0 | 7.544.287.232 byte (7,03 / 7,53 GiB; **1594 slot**, dense 0) |
| Thời gian nạp pool lúc khởi động (`server.log`) | 0 | 5,91s (B1) / 7,18s (B2) |

- **16 so sánh:** `b_faster: true`, `b_slower: false`, `text_match_all: true` (16/16 turn khớp từng ký tự).
  - `total_ratio_range`: **0,500–0,604** (A0–B1 `0,500–0,604`, A3–B2 `0,518–0,602`).
  - `decode_ratio_range`: **0,466–0,590**.
- **Chi phí:** `VmHWM` cao hơn ~280MiB (`5,90` so với `5,62 GiB`), `anon` cao hơn ~60MiB (`5807` so với `5748 MiB`) cho CUDA context/pinned buffer; nạp pool mất `5,9–7,2s` lúc khởi động (ngoài thời gian request).

## Khám phá tùy chọn (P3, không thuộc A/B): `--gpu-dense on`

- Chạy thêm 1 lượt B với `--gpu-dense on` ([explore-dense-on](explore-dense-on/), [explore-console.log](explore-console.log)) dưới cùng trần `10240MiB` và cùng placement profile.
- Nạp toàn bộ dense lên VRAM (`allocated_mb=4643`, tức 4643MiB; checkpoint lưu BF16 nhưng engine nới sang F32 khi đưa lên GPU) + pool L1 nhỏ hơn (**565 slot**, nạp pool `2,14s`, tổng VRAM cấp phát `7.542.642.688` byte = 7,02 GiB). Hoàn thành 8/8 turn, `oom_kill: 0`, `VmSwap: 0`, text khớp A0 `8/8`. `VmHWM: 5,92 GiB`, peak scope `10240,0 / 5828,2 / 7021,6 MiB`.
- Kết quả lượt khám phá:
  - Decode trung vị **2,44 tok/s** (tổng 16→8 turn, TTFT trung vị **8,06s**, total/request trung vị **35,42s**).
  - Payload expert đọc/request trung vị **57,00GB**, hit rate `48,2%–79,2%`, GPU computes `3.575–10.007` route (~0,39s GPU), thời gian expert CPU trung vị `12,64s`.

## Diễn giải

- Trên i7 (RTX 5060 Laptop 8GB), giữ dense trên CPU và dành toàn bộ 7,03 GiB VRAM cho **1594 expert slot** (25,9% trong tổng 6144 expert của Qwen3-30B-A3B) giúp cắt 62,3% số lần cache miss SSD (từ 128.467 xuống 48.474), giảm payload đọc từ 74,60 GB xuống 25,37 GB, đưa decode trung vị từ **1,92 lên 3,74 tok/s** (~1,95×) và thời gian mỗi request từ **47,1s xuống 25,8s** (total nhanh hơn 1,66–2,00 lần ở 16 cặp so sánh; *đính chính khi review 26/11: bản đầu ghi "gấp đôi ở tất cả 16 cặp"*).
- Ngoài giảm I/O, phần tính expert trên CPU cũng giảm từ ~13,1s xuống ~5,2s mỗi request (trung vị theo block) vì các route trên pool chạy trên GPU (~0,66s); A/B này không tách được đóng góp từng phần (*bổ sung khi review 26/11*).
- Qua lượt khám phá `--gpu-dense on` (P3), dù dense (4643MiB F32) cùng 565 expert trong VRAM 8GB, tốc độ chỉ đạt **2,44 tok/s** (chậm hơn rõ rệt so với **3,74 tok/s** khi `--gpu-dense off`). Giải thích khả dĩ (chưa kiểm bằng A/B): với Qwen3-30B-A3B dưới trần RAM, đọc expert từ SSD là nút thắt chính, nên dành ~4,6GiB VRAM cho thêm ~1029 expert (payload 57,0 → 25,4GB) có lợi hơn đưa dense lên cuBLAS. Đây là một lượt đơn lẻ so với hai block B ở thời điểm khác, chỉ dùng làm chiều tham khảo (*đính chính khi review 26/11*).

## Giới hạn

- Một workload (`mixed`, 4 prompt tiếng Anh ngắn), một lượt ABBA trên máy i7, placement tĩnh học từ `long-prefill` trên i5.
- Lượt `--gpu-dense on` là khám phá đơn lẻ (P3), không phải ABBA đối chứng trực tiếp với B.
- Độ lớn VRAM cấp phát và số slot (1594) phụ thuộc VRAM trống thực tế của RTX 5060 8GB lúc khởi động.

## Bảo toàn payload

Toàn bộ artifact (4 block A/B, `explore-dense-on/`, `summary.json`, `console.log`, `explore-console.log`, `preflight.json`) được khóa bằng SHA-256 trong [CHECKSUMS.sha256](CHECKSUMS.sha256) (69 file, không gồm `README.md`).

```bash
sha256sum -c CHECKSUMS.sha256
```
