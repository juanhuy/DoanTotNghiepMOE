# Tính expert trên GPU A1, i7, 14/11/2026

## Câu hỏi và protocol

Bước A1 (PLAN): so với cấu hình khuyến nghị (GPU dense, tầng VRAM expert tắt), bật tầng VRAM **kèm tính expert ngay trên GPU** (`MOE_GPU_EXPERT_COMPUTE=on`) có làm chat/decode nhanh hơn không? Protocol `gpu-compute-chat-v1` được chốt và commit vào [SCENARIOS](../../SCENARIOS.md#tính-expert-trên-gpu-a1-gpu-compute-chat-v1-1211) (commit a8ffb71) trước khi đo. ABBA on→off→off→on, mỗi block là một process mới với 8 request mixed v1 (nhóm 1 là warmup, nhóm 2 turn4–7 được đo), max64, CPU4, cache 64MiB/layer, context 512, checkpoint chung `models/olmoe-i5` (khớp 16/16 SHA-256).

```bash
systemd-inhibit --what=idle:sleep --why="A1 benchmark" \
  python3 tools/run_benchmark.py --testbed i7 --gpu auto --gpu-compute-model-ab \
  --expert-batch-workload mixed --threads 4 --cache 64 --context 512 --max-tokens 64 --lookahead off \
  --model-dir models/olmoe-i5
```

## Kết quả lượt chính

Run 20261114T123957.730046Z-gpu-compute-model-1322346100c8…, HEAD 771ca28, git_dirty false, binary 8dc775b2d39bf080… Chạy dưới `systemd-inhibit --what=idle:sleep`. [preflight](preflight.json): available ≥10449MiB, pswpout không đổi [274814, 274814, 274814], on_ac true, GPU không throttle. [nvidia-smi](nvidia-smi-pre.csv), [cuda-test](cuda-test.txt) (15/15 test PASS, EXECUTED không có SKIPPED). `power_checks`: AC online, GPU không throttle ở cả 8 điểm trước/sau block.

COMPLETED 32/32. Text/usage/finish khớp giữa các block. Block on có `expert_gpu_computes` 7734–16269/request và `expert_vram_loads` = 0 ở mọi turn; block off không có GPU compute.

**Rule: `compute_faster: true`, `compute_slower: false`.** Bật tầng VRAM kèm tính expert trên GPU nhanh hơn cấu hình khuyến nghị hiện tại (tier off) ở cả 8/8 so sánh, cả total_ms lẫn decode_ms.

| Chỉ số (turn4–7, 2 cặp × 4 prompt) | on/off min–max |
| --- | --- |
| total_ms | 0,086–0,124 |
| decode_ms | 0,082–0,101 |
| TTFT | 0,112–0,210 |
| prefill_ms | 0,110–0,209 |
| expert_compute_ms | 0,091–0,149 |
| attention_ms | 0,781–0,885 |
| lm_head_ms | 0,890–0,917 |

| | on | off |
| --- | ---: | ---: |
| Decode tok/s trung vị | 50,67 | 4,64 |
| expert_io_ms trung vị | 15,3 | 11667,0 |
| Lần đọc file expert/request (turn4–7) | 0–105 | 5532–6737 |
| Upload VRAM (turn4–7) | 0 MiB | 0 MiB |
| expert_gpu_compute_ms trung vị | 313,2 | 0 |
| Pool VRAM expert resident | 960 / 1024 (93,75%) | 0 |

Warmup (loại) cùng chiều: total on/off 0,094–0,370. Hai block cùng mode rất sát nhau (decode tok/s lệch ≤3%), máy hoạt động ổn định trong suốt lượt.

### Diễn giải

- Trên i7 với GPU RTX 5060 Laptop (8GB VRAM), pool VRAM chứa được tới **960 / 1024 expert (93,75%)**.
- Khi tầng VRAM bật nhưng không có GPU compute (ở các thí nghiệm S5/r2 trước đây), việc nạp expert từ VRAM về CPU qua PCIe là gánh nặng lớn. Nhưng khi kết hợp với `MOE_GPU_EXPERT_COMPUTE=on` (A1), toàn bộ 960 expert được tính toán ngay tại chỗ trên GPU qua NVRTC INT8 SwiGLU kernel.
- Nhờ vậy, số lần đọc file expert từ SSD/RAM giảm triệt để (từ 5532–6737 lần xuống chỉ còn 0–105 lần mỗi request), expert I/O wait giảm từ ~11,7s xuống chỉ còn 15ms.
- Tốc độ giải mã decode tăng vọt từ **4,64 tok/s lên 50,67 tok/s (tăng gấp ~10,9 lần)**, và total latency giảm từ 15,3–18,0s xuống chỉ còn 1,34–2,24s (tỉ số on/off 0,086–0,124).
- Kết quả này củng cố mạnh mẽ kết luận từ i5: nút thắt không phải là nơi lưu trữ mà là nơi thực thi tính toán. Khi tính trực tiếp trên GPU, tầng VRAM phát huy hiệu quả tối đa.

## Ghi chú khi tích hợp (15/11, owner i5)

Raw không đổi; checksum 64 file khớp. Rule `compute_faster: true` 8/8 giữ nguyên. Các điểm dưới đây đính chính phần diễn giải ở trên:

- **Baseline off trên i7 yếu bất thường, nên độ lớn ~10× không đại diện cho i7.** Tốc độ đọc file expert phía off (`expert_file_bytes_read / expert_file_read_ms`, turn4–7) chỉ đạt **3,09–3,77 GB/s**, so với 6,10–7,65 GB/s ở mọi block tier off của A3 i7 (13/11) và 10,89–12,12 GB/s ở block off của r2 i7 (11/11). Cùng số file load (5532–6737/request, trùng i5 vì cùng checkpoint và routing), nên `expert_io_ms` phía off (~11,7s) bị phóng to theo. Trước lượt, máy đã dùng ~842MiB swap (sau lượt 1 bị Chrome chiếm RAM); `Cached` hệ thống trong block off chỉ 6,7–8,3GiB. Giả thuyết là page cache không còn giữ đủ checkpoint nên một phần lần đọc đi xuống NVMe, nhưng chưa đo trực tiếp. Phía on hầu như không đọc file nên ít bị ảnh hưởng. Vì vậy chiều kết quả (on nhanh hơn) đáng tin, còn tỉ số 0,086–0,124 và 4,64→50,67 tok/s là của riêng lượt này, không trích như mức tăng điển hình của i7.
- **RAM i7 là ~14,8GiB** (`MemTotal` 15.537.172 KiB trong memory.jsonl), không phải 32GB như PROGRESS của branch ghi.
- **Có upload VRAM khi lấp pool:** mỗi block on upload 5775MiB ở turn0–3 (5618MiB ở request đầu), rồi 0 ở turn4–7. Pool đạt 934 slot sau request đầu và 960 slot từ turn sau đó.
- Câu "nút thắt không nằm ở capacity" là quá mức. Ở i7, pool chứa 93,75% expert nên lợi ích đến từ cả hai yếu tố: chỗ chứa đủ lớn để gần như bỏ I/O file, và tính ngay trên GPU để bỏ download VRAM→RAM. Lượt này không tách riêng được hai yếu tố đó.

## Giới hạn

Một lượt, một máy (i7, RTX 5060 Laptop GPU 8GB, pool 960 slot), workload mixed max64, CPU4, cache 64MiB/layer. Kernel GPU dùng NVRTC compute_75, kiểm logits so với CPU <1e-4 ở fixture; không có logits parity HTTP trên model 7B (chỉ text/usage khớp). Chưa đo prefill dài, ngân sách cache khác. Mặc định `MOE_GPU_EXPERT_COMPUTE` trong code vẫn off theo quy tắc.

## Lượt trước bị dừng (giữ nguyên làm hồ sơ)

### Lượt 1: engine bị swap ([attempt1-swap-stopped](attempt1-swap-stopped/))

Run 20261114T123814.488508Z-gpu-compute-model-b2790d5d…, status failed. Preflight đạt ([attempt1-preflight](attempt1-preflight.json), [attempt1-nvidia-smi](attempt1-nvidia-smi-pre.csv)). Ở block0 (on), sau khi server nạp xong dense và pool 960 slot, Chrome khởi động lại chiếm RAM gây áp lực bộ nhớ, kswapd đẩy VmSwap của process engine lên 208MiB, vượt ngưỡng watchdog 128MiB ở giây 6,8. Lượt này không có số đo, được lưu giữ nguyên trạng và chạy lại một lần sau khi đóng sạch Chrome theo đúng quy trình WORKFLOWS.md.

## Bảo toàn payload

Hai run được chuyển nguyên vào `run/` (lượt chính) và `attempt1-swap-stopped/`, kèm preflight, nvidia-smi hai lượt và output kiểm tra CUDA. Khóa 64 file bằng CHECKSUMS.sha256 (không gồm README).

```bash
sha256sum -c CHECKSUMS.sha256
```
