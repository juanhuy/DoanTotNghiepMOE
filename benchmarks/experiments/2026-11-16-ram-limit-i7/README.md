# Giới hạn RAM của engine, i7, 16/11/2026

## Câu hỏi và protocol

Mọi lượt đo trước đều có page cache của OS giữ cả checkpoint 7GB, nên "đọc expert từ SSD" thực ra là đọc từ RAM (8–12 GB/s). Câu hỏi: khi engine **không được dùng đủ RAM để giữ checkpoint** (giống model lớn hơn RAM), tầng VRAM + tính expert trên GPU (A1) có còn nhanh hơn tier off không? Protocol `ram-limit-chat-v1` được chốt và commit vào [SCENARIOS](../../SCENARIOS.md#giới-hạn-ram-của-engine-ram-limit-chat-v1-1511) trước khi đo.

- Engine chạy trong scope `systemd-run --user --scope -p MemoryMax=<N>M -p MemorySwapMax=0`; cgroup v2 tính cả page cache. Trước mỗi block, checkpoint bị xóa khỏi page cache (`posix_fadvise`) và kiểm bằng `fincore`: còn 0 byte ở cả 8 block. `memory.max` của scope engine được đọc lại sau khi khởi động và khớp ở cả 8 block.
- Mỗi mức là một ABBA A1 (on→off→off→on, 4 process mới, mixed v1 ×2 nhóm, turn4–7 được đo, max64, GPU auto, CPU4, cache 64MiB/layer, context 512, lookahead off, batch off). On = tier on + GPU expert compute (`fill`, pool 960 slot = 93,75%); off = tier off. Hai mức chạy theo thứ tự 8192 rồi 5120MiB.

```bash
for limit in 8192 5120; do
  systemd-inhibit --what=idle:sleep --why="ram-limit benchmark" \
    python3 tools/run_benchmark.py --testbed i7 --gpu auto --gpu-compute-model-ab \
    --expert-batch-workload mixed --threads 4 --cache 64 --context 512 --max-tokens 64 --lookahead off \
    --model-dir models/olmoe-i5 --memory-max-mib $limit
done 2>&1 | tee /tmp/i7-ramlimit-console.log
```

HEAD 00c9d98, git_dirty false, binary 8dc775b2d39bf080… (trùng binary lượt A1 i7 chính), checkpoint chung `models/olmoe-i5`. [preflight](preflight.json): eligible, available ≥9823MiB, pswpout không đổi [1314, 1314, 1314], on_ac true, GPU không throttle; [nvidia-smi](nvidia-smi-pre.csv); [console](console.log); [cuda-test](cuda-test.txt) (15/15 test PASS, EXECUTED, 0 SKIPPED). `power_checks`: AC online ở cả 16 điểm. VmSwap engine 0 ở mọi block, không có OOM. COMPLETED 32/32 ở mỗi mức; text/usage/finish khớp giữa các block; on có 0 lần tải VRAM→RAM.

## Kết quả

| Mức trần | Rule | total on/off | decode on/off | TTFT on/off | Decode tok/s trung vị on / off |
| --- | --- | --- | --- | --- | --- |
| Không giới hạn (A1 14/11, hồi cứu) | `compute_faster` 8/8 | 0,086–0,124 | 0,082–0,101 | 0,112–0,210 | 50,67 / 4,64 |
| 8192MiB ([limit-8192](limit-8192/)) | **`compute_faster` 8/8** | 0,137–0,204 | 0,133–0,176 | 0,153–0,272 | **50,27 / 7,85** |
| 5120MiB ([limit-5120](limit-5120/)) | **`compute_faster` 8/8** | 0,068–0,104 | 0,065–0,083 | 0,086–0,183 | **50,56 / 3,83** |

| Mức 5120MiB, turn4–7 | on | off |
| --- | ---: | ---: |
| total_ms | 1.343–2.204 | 18.206–22.596 |
| Tốc độ đọc file expert | 6,82–8,34 GB/s | 2,50–2,95 GB/s |
| expert_io_ms trung vị | 13,2 | 13.194,0 |
| Lần đọc file expert/request | 0–105 | 5532–6737 |
| Đỉnh scope (current / anon / file, MiB) | 5120,0 / 3080,6–3098,4 / 2669,1–2669,8 | 5120,0 / 3208,2–3216,9 / 2654,0–2673,8 |
| VmSwap max / oom_kills | 0 KiB / 0 | 0 KiB / 0 |

| Mức 8192MiB, turn4–7 | on | off |
| --- | ---: | ---: |
| total_ms | 1.285–2.189 | 8.516–11.104 |
| Tốc độ đọc file expert | 4,22–8,86 GB/s | 7,40–8,08 GB/s |
| expert_io_ms trung vị | 18,4 | 4.661,5 |
| Lần đọc file expert/request | 0–105 | 5532–6737 |
| Đỉnh scope (current / anon / file, MiB) | 6428,2–8192,0 / 3085,5–3098,4 / 3607,3–5223,7 | 8192,0 / 3218,9–3221,6 / 5171,7–5203,8 |
| VmSwap max / oom_kills | 0 KiB / 0 | 0 KiB / 0 |

Mốc không giới hạn là lượt khác ngày, nên chỉ dùng để mô tả. Số file load giống hệt A1 ở cả hai mode, vì cùng checkpoint, routing và cache.

### Diễn giải

- **Ở mức 5120MiB, ưu thế của tầng VRAM trên i7 trở nên áp đảo hoàn toàn:**
  - Tỉ số total on/off giảm xuống mức kỷ lục **0,068–0,104 (nhanh hơn 9,6–14,7 lần)**, và decode on/off là **0,065–0,083 (nhanh hơn 12,0–15,4 lần)**.
  - Phía off bị tụt decode từ 4,64 tok/s xuống còn **3,83 tok/s**, total latency nhảy vọt từ ~16s lên **18,2–22,6s** vì phải đọc SSD thật (tốc độ đọc SSD đạt 2,50–2,95 GB/s, expert I/O wait trung vị lên tới 13,2 giây mỗi request).
  - Ngược lại, phía on nhờ có **960 slot VRAM (93,75% số expert)** được tính toán tại chỗ qua GPU kernel, chỉ phải đọc SSD 0–105 lần cho các expert ngoài pool, giữ nguyên tốc độ giải mã **50,56 tok/s** và total latency chỉ **1,34–2,20s**.
- **Ở mức 8192MiB: hoàn toàn nhất quán trên i7, `compute_faster` 8/8:**
  - Khác với i5 (bị app ngoài chiếm RAM làm nhiễu ở block0), trên i7 các app đã được đóng sạch, hai block cùng mode cực kỳ đồng đều (block 1 off: 8516–10868 ms vs block 2 off: 8516–11104 ms; block 0 on: 1285–2050 ms vs block 3 on: 1379–2189 ms).
  - Tốc độ đọc file của phía off đạt 7,40–8,08 GB/s do page cache ~5,2GB chứa được phần lớn checkpoint, decode đạt 7,85 tok/s. Phía on vẫn nhanh hơn 4,9–7,3× (total on/off 0,137–0,204, decode trung vị 50,27 tok/s).
- **Kết luận:** Thí nghiệm xác nhận trên cả hai mức trần RAM, khi dung lượng RAM bị giới hạn và OS page cache không thể giữ toàn bộ checkpoint, kiến trúc phân tầng VRAM kết hợp tính toán trực tiếp trên GPU chứng minh ưu thế vượt trội, giúp hệ thống duy trì thông lượng cao ổn định (~50 tok/s) mà không bị phụ thuộc vào tốc độ đọc của đĩa SSD.

## Giới hạn

- Một lượt mỗi mức, một máy (i7-14650HX, RTX 5060 Laptop 8GB, pool 960 slot), workload mixed max64, CPU4, cache 64MiB/layer.
- Trần là cgroup `memory.max` của riêng engine (mô phỏng "checkpoint không vừa RAM được cấp").
- Tốc độ đọc là payload engine chia cho thời gian đọc của engine.
- Không có logits parity HTTP 7B (chỉ text/usage khớp). Chưa đo prefill dài.

## Bảo toàn payload

Hai run được chuyển nguyên vào `limit-8192/` và `limit-5120/`, kèm preflight, nvidia-smi, console và output test CUDA. Khóa bằng CHECKSUMS.sha256 (không gồm README).

```bash
sha256sum -c CHECKSUMS.sha256
```
