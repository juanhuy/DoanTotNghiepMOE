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

HEAD 00c9d98, git_dirty false, binary 8dc775b2d39bf080… (trùng binary lượt A1 i7 chính), checkpoint chung `models/olmoe-i5`. [preflight](preflight.json): eligible, available ≥9823MiB, pswpout không đổi [1314, 1314, 1314], on_ac true, GPU không throttle lúc preflight; [nvidia-smi](nvidia-smi-pre.csv); [console](console.log); [cuda-test](cuda-test.txt) (15/15 test PASS, EXECUTED, 0 SKIPPED). `power_checks`: AC online ở cả 16 điểm. VmSwap engine 0 ở mọi block, không có OOM. COMPLETED 32/32 ở mỗi mức; text/usage/finish khớp giữa các block; on có 0 lần tải VRAM→RAM.

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

*Mục này được owner i5 viết lại khi tích hợp (17/11). Bản gốc của i7 dùng từ ngữ quảng bá ("kỷ lục", "áp đảo") và so sánh khác ngày; xem ghi chú bên dưới.*

- **5120MiB:** off phải đọc SSD thật (2,50–2,95 GB/s, chờ I/O trung vị 13,2s/request), decode trung vị 3,83 tok/s. On giữ 960/1024 expert (93,75%) trong VRAM và tính tại chỗ, chỉ đọc file 0–105 lần/request, decode trung vị 50,56 tok/s. Tỉ số total on/off 0,068–0,104, decode 0,065–0,083, `compute_faster` 8/8.
- **8192MiB:** `compute_faster` 8/8, total 0,137–0,204. Page cache trong scope giữ được ~5,2GB nên off đọc file 7,40–8,08 GB/s, decode 7,85 tok/s. Hai block cùng mode đồng đều (off 8516–10868 so với 8516–11104 ms; on 1285–2050 so với 1379–2189 ms). Không có nhiễu ngoài engine như block0 i5.
- **Ý nghĩa:** trên i7, trần RAM gần như không ảnh hưởng phía on (decode ~50 tok/s ở cả hai mức), vì hầu như toàn bộ expert đã nằm trong VRAM. Phía off chậm dần khi trần siết lại. Đây là kết quả về **dung lượng VRAM đủ chứa gần hết model**, chưa phải bằng chứng cho trường hợp model lớn hơn tổng RAM + VRAM. Ngược lại, i5 (pool 299 slot ≈ 29%) vẫn phải đọc SSD ở phía on, nên lợi ích chỉ ở mức 0,389–0,616.

## Ghi chú khi tích hợp (17/11, owner i5)

- Đã kiểm lại: `sha256sum -c` khớp 110 file. Ở cả hai mức `compute_faster: true`, `on_faster: true`, git_dirty false, HEAD 00c9d98. Tỉ số total và decode tính lại từ `pairs` khớp README. `checkpoint_resident_bytes_after_evict` = 0 và `memory.max` của scope khớp ở 8/8 block. 0 OOM, `expert_vram_upload_bytes` = 0 trong các turn đo.
- `power_checks`: AC online 16/16. **Một điểm GPU ghi `throttled: true`** (8192, block0 trước khi chạy: `sw_power_cap` lúc GPU nhàn rỗi, 9,5W, 45°C). Không có thermal slowdown nào. Tooling chỉ ghi chứ không chặn điểm này, nên câu "GPU không throttle" chỉ đúng với preflight.
- **Không so sánh khác ngày.** Bản gốc viết "off tụt từ 4,64 xuống 3,83 tok/s", tức so với lượt A1 không giới hạn ngày 14/11. Nhưng baseline off ngày 14/11 yếu bất thường (đọc file 3,1–3,8 GB/s, xem ghi chú tích hợp A1 i7). Bằng chứng là ở trần 8192 (ít RAM hơn), off lại nhanh hơn (7,85 tok/s). Hàng "Không giới hạn" trong bảng chỉ để tham khảo.
- Phía on nạp gần hết checkpoint lên VRAM lúc khởi động (`fill`). Thời gian đó không nằm trong turn4–7 được đo, và file page của phía on (~2,7–5,2GB) nhiều khả năng đến từ lần nạp này (suy luận, không đo riêng).
- Tốc độ đọc phía on (4,22–8,86 GB/s) chỉ tính trên 0–105 lần đọc, nên không đại diện cho SSD.

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
