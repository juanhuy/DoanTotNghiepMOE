# Giới hạn RAM của engine, i5, 16/11/2026

## Câu hỏi và protocol

Mọi lượt đo trước đều có page cache của OS giữ cả checkpoint 7GB, nên "đọc expert từ SSD" thực ra là đọc từ RAM (9–12 GB/s). Câu hỏi: khi engine **không được dùng đủ RAM để giữ checkpoint** (giống model lớn hơn RAM), tầng VRAM + tính expert trên GPU (A1) có còn nhanh hơn tier off không? Protocol `ram-limit-chat-v1` được chốt và commit vào [SCENARIOS](../../SCENARIOS.md#giới-hạn-ram-của-engine-ram-limit-chat-v1-1511) (commit a752868) trước khi đo.

- Engine chạy trong scope `systemd-run --user --scope -p MemoryMax=<N>M -p MemorySwapMax=0`; cgroup v2 tính cả page cache. Trước mỗi block, checkpoint bị xóa khỏi page cache (`posix_fadvise`) và kiểm bằng `fincore`: còn 0 byte ở cả 8 block. `memory.max` của scope engine được đọc lại sau khi khởi động và khớp ở cả 8 block.
- Mỗi mức là một ABBA A1 (on→off→off→on, 4 process mới, mixed v1 ×2 nhóm, turn4–7 được đo, max64, GPU auto, CPU4, cache 64MiB/layer, context 512, lookahead off, batch off). On = tier on + GPU expert compute (`fill`, pool 299 slot); off = tier off. Hai mức chạy theo thứ tự 8192 rồi 5120MiB.

```bash
for limit in 8192 5120; do
  systemd-inhibit --what=idle:sleep --why="ram-limit benchmark" \
    python3 tools/run_benchmark.py --testbed i5 --gpu auto --gpu-compute-model-ab \
    --expert-batch-workload mixed --threads 4 --cache 64 --context 512 --max-tokens 64 --lookahead off \
    --memory-max-mib $limit
done
```

HEAD a752868, git_dirty false, binary 372532cb2099bc35… (trùng binary lượt A1 i5 chính; lượt này chỉ đổi tooling Python), checkpoint chung. [preflight](preflight.json): eligible, available ≥8658MiB, pswpout không đổi, on_ac true, GPU không throttle; [nvidia-smi](nvidia-smi-pre.csv); [console](console.log). `power_checks`: AC online, GPU không throttle ở cả 16 điểm. VmSwap engine 0 ở mọi block, không có OOM. COMPLETED 32/32 ở mỗi mức; text/usage/finish khớp giữa các block; on có 0 lần tải VRAM→RAM.

## Kết quả

| Mức trần | Rule | total on/off | decode on/off | TTFT on/off | Decode tok/s trung vị on / off |
| --- | --- | --- | --- | --- | --- |
| Không giới hạn (A1 14/11, hồi cứu) | `compute_faster` | 0,743–0,813 | 0,743–0,811 | 0,711–0,866 | 10,73 / 8,24 |
| 8192MiB ([limit-8192](limit-8192/)) | **không nhất quán** | 0,566–1,057 | 0,561–1,074 | 0,605–1,017 | 9,46 / 7,70 |
| 5120MiB ([limit-5120](limit-5120/)) | **`compute_faster` 8/8** | 0,389–0,616 | 0,389–0,599 | 0,381–0,758 | **4,13 / 2,14** |

| Mức 5120MiB, turn4–7 | on | off |
| --- | ---: | ---: |
| total_ms | 12.459–23.522 | 29.602–42.094 |
| Tốc độ đọc file expert | 1,61–2,56 GB/s | 1,28–1,71 GB/s |
| expert_io_ms trung vị | 11.594–11.976 | 27.516–28.039 |
| Lần đọc file expert/request | 3420–4731 | 5532–6737 |
| Đỉnh scope (current / anon / file, MiB) | 5120 / 3262–3289 / 2597–2627 | 5120 / 3268–3279 / 2608–2614 |

Mốc không giới hạn là lượt khác ngày, nên chỉ dùng để mô tả. Số file load giống hệt A1 ở cả hai mode, vì cùng checkpoint, routing và cache; trần chỉ làm mỗi lần đọc chậm hơn.

### Diễn giải

- **Ở mức 5120MiB, tầng VRAM có ích nhiều hơn so với khi không giới hạn.** Tỉ số total giảm từ 0,743–0,813 xuống 0,389–0,616, và decode trung vị nhanh khoảng 1,9× thay vì khoảng 1,3×. Lý do: mỗi lần đọc file giờ đi xuống SSD (1,3–2,6 GB/s), nên mỗi lần đọc mà 299 slot VRAM tránh được (khoảng 30–39% số lần đọc) có giá trị lớn hơn nhiều.
- So với khi không giới hạn, off chậm đi khoảng 3,9× (8,24→2,14 tok/s), còn on chậm đi khoảng 2,6× (10,73→4,13 tok/s). Engine vẫn hoàn tất mọi request trong trần, không OOM và không swap. Phần anon đỉnh khoảng 3,3GB, phần còn lại của trần là page cache.
- Phía on đọc file nhanh hơn phía off (1,6–2,6 so với 1,3–1,7 GB/s), dù cùng trần. Giả thuyết (chưa đo I/O thiết bị): on đọc ít expert hơn nên page cache 2,6GB giữ được tỉ lệ checkpoint cần dùng lớn hơn. Đây là `expert_file_read_ms` của engine, không phải I/O thiết bị.

### Mức 8192MiB: không nhất quán, có nhiễu môi trường ở block0

- Cặp 1 (block3 on / block2 off) cho total on/off 0,566–0,833 ở 4/4 turn. Cặp 0 (block0 on / block1 off) cho 1,024 / 1,057 / 0,831 / 0,841: on chậm hơn ở turn4–5.
- Trong block0, `AnonPages` của toàn hệ thống lên tới 5,6–8,5GiB, trong khi ở block2/3 chỉ 1,5–4,8GiB. Nghĩa là có tiến trình ngoài engine chiếm thêm khoảng 2–4GiB RAM (chưa xác định là tiến trình nào). Scope engine ở block0 chỉ đạt đỉnh 6908MiB, chưa chạm trần 8192: page cache của engine bị ép bởi áp lực bộ nhớ toàn hệ thống, không phải bởi trần của nó. Đọc file của block0 chỉ đạt 3,6–5,1 GB/s, so với 7,5–9,5 GB/s ở block3 cùng mode. MemAvailable min 5588MiB vẫn trên guard, VmSwap 0, nên không có điều kiện dừng nào bị kích.
- Theo protocol, lượt này có số đo nên không chạy lại; kết quả ghi là không nhất quán. Không dùng cặp 1 riêng để kết luận.
- Ở mức 8192MiB, page cache chứa được khoảng 5,1GB trên 7GB checkpoint, và block off đọc file 5,2–7,8 GB/s. Mức này gần trạng thái không giới hạn hơn là mức 5120MiB.

## Giới hạn

- Một lượt mỗi mức, một máy (i5, GTX1650 4GB, pool 299 slot), workload mixed max64, CPU4, cache 64MiB/layer. Mốc không giới hạn đo khác ngày.
- Trần là cgroup `memory.max` của riêng engine. Đây là mô phỏng "checkpoint không vừa RAM được cấp", không phải chạy model lớn hơn RAM thật. Phần còn lại của hệ thống vẫn có RAM, và áp lực toàn hệ thống vẫn ảnh hưởng (như block0 ở mức 8192).
- Tốc độ đọc là payload engine chia cho thời gian đọc của engine, không phải I/O thiết bị. Chưa có `iostat` hay blktrace.
- Không có logits parity HTTP 7B (chỉ text/usage khớp). Chưa đo trên i7 hay prefill dài.

## Bảo toàn payload

Hai run được chuyển nguyên vào `limit-8192/` và `limit-5120/`, kèm preflight, nvidia-smi và console. Khóa bằng CHECKSUMS.sha256 (không gồm README).

```bash
sha256sum -c CHECKSUMS.sha256
```
