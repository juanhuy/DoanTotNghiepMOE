# Tính expert trên GPU A1, i5, 14/11/2026

## Câu hỏi và protocol

Bước A1 (PLAN): so với cấu hình khuyến nghị (GPU dense, tầng VRAM expert tắt), bật tầng VRAM **kèm tính expert ngay trên GPU** (`MOE_GPU_EXPERT_COMPUTE=on`) có làm chat/decode nhanh hơn không? Protocol `gpu-compute-chat-v1` được chốt và commit vào [SCENARIOS](../../SCENARIOS.md#tính-expert-trên-gpu-a1-gpu-compute-chat-v1-1211) (commit a8ffb71) trước khi đo. ABBA on→off→off→on, mỗi block là một process mới với 8 request mixed v1 (nhóm 1 là warmup, nhóm 2 turn4–7 được đo), max64, CPU4, cache 64MiB/layer, context 512.

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu auto --gpu-compute-model-ab \
  --expert-batch-workload mixed --threads 4 --cache 64 --context 512 --max-tokens 64 --lookahead off
```

## Kết quả lượt chính

Run 20261114T081740.484078Z-gpu-compute-model-0486f8…, HEAD 47008a1 (protocol đã thêm power guard, xem cuối trang), git_dirty false, binary 372532cb2099bc35… (trùng hai lượt hỏng; bản sửa chỉ đổi tooling Python). Chạy dưới `systemd-inhibit --what=idle:sleep`. [preflight](preflight.json): available ≥11998MiB, pswpout không đổi, on_ac true, GPU không throttle. [nvidia-smi](nvidia-smi-pre.csv), [console](console.log). `power_checks`: AC online, GPU P0, không lý do throttle ở cả 8 điểm trước/sau block.

COMPLETED 32/32. Text/usage/finish khớp giữa các block. Block on có `expert_gpu_computes` 2240–5884/request và `expert_vram_loads` = 0 ở mọi turn; block off không có GPU compute.

**Rule: `compute_faster: true`, `compute_slower: false`.** Bật tầng VRAM kèm tính expert trên GPU nhanh hơn cấu hình khuyến nghị hiện tại (tier off) ở cả 8/8 so sánh, cả total_ms lẫn decode_ms.

| Chỉ số (turn4–7, 2 cặp × 4 prompt) | on/off min–max |
| --- | --- |
| total_ms | 0,743–0,813 |
| decode_ms | 0,743–0,811 |
| TTFT | 0,711–0,866 |
| prefill_ms | 0,709–0,869 |
| expert_compute_ms | 0,667–0,750 |
| attention_ms | 0,952–0,973 |
| lm_head_ms | 0,982–1,011 |

| | on | off |
| --- | ---: | ---: |
| Decode tok/s trung vị | 10,73 | 8,24 |
| expert_io_ms trung vị | 2628 | 4132 |
| Lần đọc file expert/request (turn4–7) | 3420–4731 | 5532–6737 |
| Upload VRAM | 1799MiB ở request đầu (lấp pool), sau đó 0 | 0 |
| expert_gpu_compute_ms trung vị | 491 | 0 |

Warmup (loại) cùng chiều: total on/off 0,745–0,863. Hai block cùng mode rất sát nhau (decode tok/s lệch ≤0,34), nên máy ổn định trong suốt lượt.

### Diễn giải

- A1 chuyển expert đã nằm trong VRAM sang tính tại chỗ trên GPU. Nhờ vậy CPU bớt khoảng 30% thời gian SwiGLU (expert_compute 0,67–0,75), và số lần đọc file expert giảm khoảng 30–39% vì 299 slot VRAM giữ thêm expert ngoài cache RAM. Chi phí upload chỉ còn 1,8GB một lần, thay vì 33–41GiB/request như tầng VRAM victim ở S5.
- So với S5 và i7 r2 (tầng VRAM có lợi về chỗ chứa nhưng thua vì phải kéo expert về CPU), lượt này xác nhận luận điểm: **nút thắt là chỗ tính, không phải chỗ chứa.** Khi tính ngay trên GPU, tầng VRAM chuyển từ chậm hơn sang nhanh hơn.
- attention gần như không đổi (0,95–0,97) vì dense đã chạy trên GPU ở cả hai phía.

## Giới hạn

Một lượt, một máy (i5, GTX1650 4GB, pool 299 slot), workload mixed max64, CPU4, cache 64MiB/layer. Kernel GPU dùng NVRTC compute_75, kiểm logits so với CPU <1e-4 ở fixture; không có logits parity HTTP trên model 7B (chỉ text/usage khớp). Chưa đo prefill dài, i7, ngân sách cache khác. Mặc định `MOE_GPU_EXPERT_COMPUTE` vẫn off; đổi khuyến nghị cần thêm lượt i7.

## Hai lượt trước, không thành kết quả

Cả hai lượt dưới đây hỏng vì điều kiện máy, không phải vì engine. Chúng được giữ để minh bạch và làm căn cứ cho bản sửa protocol (power guard, 14/11). Cả hai chạy ở HEAD a8ffb71, cùng binary và checkpoint chung (tokenizer b1fb1517…); trước khi đo `tools/check.py --rust --release` PASS, test CUDA EXECUTED.

### Lượt 1, 12/11: engine bị swap ([attempt1-swap-stopped](attempt1-swap-stopped/))

Run 20261112T070938.316912Z-gpu-compute-model-cf3458…, status failed. Preflight đạt nhưng file preflight/nvidia-smi của lượt này nằm trong scratchpad của phiên trước và đã mất, nên không có trong hồ sơ.

- Block0 (on) hoàn tất 8/8, decode 8,9–11,4 tok/s, 0 lần tải expert từ VRAM về RAM, upload 1,8GB chỉ ở request đầu.
- Block1 (off) bị watchdog dừng ở giây 101, sau 5 request (3,1–6,8 tok/s): VmSwap của engine 154MiB, vượt ngưỡng 128MiB. Lúc đó Chrome/Slack đã được mở lại: AnonPages hệ thống khoảng 6,9GiB, swap hệ thống dùng khoảng 2,0GiB.
- Lượt này không có số đo, nên theo quy tắc được chạy lại một lần.

### Lượt 2, 14/11: chạy bằng pin, GPU bị bóp giữa chừng ([attempt2-battery](attempt2-battery/))

Run 20261114T080033.939288Z-gpu-compute-model-1c3659…, status completed 32/32. Text/usage/finish khớp giữa các block. Block on có `expert_gpu_computes` > 0 và `expert_vram_loads` = 0 ở mọi turn. Rule cho kết quả `compute_faster=false`, `compute_slower=false`.

- [preflight](attempt2-preflight.json) đạt: available ≥11999MiB, pswpout không đổi, không engine/build khác. [nvidia-smi](attempt2-nvidia-smi-pre.csv), [console](attempt2-console.log).
- RSS max 3517–3539MiB, VmSwap 0, MemAvailable min 8650–8910MiB ở cả 4 block. Đợt này bộ nhớ không phải vấn đề.
- **Máy chạy bằng pin suốt lượt:** AC offline từ 14:14 (giờ máy, +07) và chỉ được cắm lại lúc 15:12, sau khi lượt đo kết thúc ([trích journal](attempt2-journal-power.txt)). Preflight lúc đó chưa kiểm nguồn điện.

| Block | Chế độ | Decode tok/s turn0–7 | attention_ms turn4–7 (GPU) |
| --- | --- | --- | --- |
| 0 | on | 8,48 9,15 9,18 8,50 · 10,09 9,04 9,57 8,63 | 988–1974 |
| 1 | off | 6,30 6,68 6,06 5,95 · 7,23 6,83 6,80 6,56 | 1055–2079 |
| 2 | off | 7,29 7,23 7,06 6,84 · 7,80 7,18 **3,08 2,83** | 1042–2062, rồi **9883 / 11152** |
| 3 | on | **3,37 3,36 3,52 3,50 · 3,62 3,37 3,57 3,19** | **8049–9902** |

- Từ block2 turn6 (khoảng 15:05), mọi phần chạy trên GPU chậm đi khoảng 5–8 lần: attention khoảng 8×, thời gian kernel expert trên GPU ở block3 khoảng 7× so với block0. Các phần trên CPU (expert_io, expert_compute) giữ nguyên mức cũ. Thời điểm này trùng với lúc màn hình mờ khi idle (15:03:49–15:04:00) và màn hình khóa (15:06:50), khi đang chạy pin. Lúc kiểm sau lượt đo (pin 20%), GPU ở P8, 300MHz, `sw_power_cap` và `sw_thermal_slowdown` Active. Sau khi cắm sạc, GPU về P0 và không còn lý do throttle nào. Nguyên nhân chính xác (chính sách nguồn khi idle trên pin hay pin yếu) chưa được tách riêng.
- Vì vậy cặp gương thứ hai bị lệch: block3 (on) bị bóp toàn bộ, block2 (off) chỉ bị ở 2 turn cuối. Tỉ số total on/off: cặp 0 (block0/1) là 0,686 / 0,764 / 0,697 / 0,760; cặp 1 (block3/2) là 2,028 / 1,923 / 0,894 / 0,903. Rule "không nhất quán" ở đây phản ánh sự cố nguồn điện, **không** phản ánh A1.

#### Quan sát phụ của lượt 2

- Ở cặp sạch (block0/1, cả hai bình thường), A1 giảm total 24–31% và expert_compute khoảng 28–35% ở 4/4 turn đo; decode khoảng 9,1 so với 6,7 tok/s.
- Ở turn6–7 của cặp 1, khi cả hai phía đều bị bóp GPU, on vẫn có total thấp hơn (0,89–0,90).
- Lượt chính sau đó cho cặp sạch cùng chiều (0,743–0,813).

## Sửa protocol sau hai lượt hỏng (power guard, 14/11; áp dụng cho lượt chính)

- `tools/preflight_memory.py` thêm hai điều kiện để eligible: máy đang cắm sạc (sysfs Mains online; máy không có Mains thì coi là có AC) và GPU không có lý do throttle `hw_slowdown`/`hw_thermal`/`sw_thermal`/`hw_power_brake`/`sw_power_cap` (nvidia-smi; không có GPU thì ghi nhận, không chặn).
- Harness A1 ghi trạng thái AC/GPU trước và sau mỗi block (`power_checks`). Nếu máy rời AC thì dừng như environment stop. Throttle trong lúc tải chỉ được ghi lại, không chặn.
- Chạy dưới `systemd-inhibit --what=idle:sleep` và tắt tự khóa/tự mờ màn hình trong lúc đo.
- Không đổi workload, mode, rule hay ngưỡng bộ nhớ.

## Bảo toàn payload

Ba run được chuyển nguyên vào `run/` (lượt chính), `attempt1-swap-stopped/` và `attempt2-battery/`, kèm preflight, nvidia-smi, console và trích journal của lượt 2. Mọi file trừ README được khóa bằng CHECKSUMS.sha256.

```bash
sha256sum -c CHECKSUMS.sha256
```
