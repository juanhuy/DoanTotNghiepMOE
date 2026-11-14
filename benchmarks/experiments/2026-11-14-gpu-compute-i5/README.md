# Tính expert trên GPU A1, i5: hai lượt chưa thành kết quả (12/11 và 14/11/2026)

## Câu hỏi và protocol

Bước A1 (PLAN): so với cấu hình khuyến nghị (GPU dense, tầng VRAM expert tắt), bật tầng VRAM **kèm tính expert ngay trên GPU** (`MOE_GPU_EXPERT_COMPUTE=on`) có làm chat/decode nhanh hơn không? Protocol `gpu-compute-chat-v1` được chốt và commit vào [SCENARIOS](../../SCENARIOS.md#tính-expert-trên-gpu-a1-gpu-compute-chat-v1-1211) (commit a8ffb71) trước khi đo. ABBA on→off→off→on, mỗi block là một process mới với 8 request mixed v1 (nhóm 1 là warmup, nhóm 2 turn4–7 được đo), max64, CPU4, cache 64MiB/layer, context 512.

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu auto --gpu-compute-model-ab \
  --expert-batch-workload mixed --threads 4 --cache 64 --context 512 --max-tokens 64 --lookahead off
```

**Hồ sơ này không có kết luận theo rule.** Cả hai lượt đều hỏng vì điều kiện máy, không phải vì engine. Hồ sơ được giữ để minh bạch và làm căn cứ cho bản sửa protocol (power guard, 14/11).

## Provenance chung

HEAD a8ffb71 (branch `work/a1-gpu-expert-compute`), git_dirty false. Binary 372532cb2099bc35… ở mọi block của cả hai lượt. Checkpoint chung (tokenizer b1fb1517…). Trước khi đo, `tools/check.py --rust --release` PASS, test CUDA kernel/fixture EXECUTED.

## Lượt 1, 12/11: engine bị swap ([attempt1-swap-stopped](attempt1-swap-stopped/))

Run 20261112T070938.316912Z-gpu-compute-model-cf3458…, status failed. Preflight đạt nhưng file preflight/nvidia-smi của lượt này nằm trong scratchpad của phiên trước và đã mất, nên không có trong hồ sơ.

- Block0 (on) hoàn tất 8/8, decode 8,9–11,4 tok/s, 0 lần tải expert từ VRAM về RAM, upload 1,8GB chỉ ở request đầu.
- Block1 (off) bị watchdog dừng ở giây 101, sau 5 request (3,1–6,8 tok/s): VmSwap của engine 154MiB, vượt ngưỡng 128MiB. Lúc đó Chrome/Slack đã được mở lại: AnonPages hệ thống khoảng 6,9GiB, swap hệ thống dùng khoảng 2,0GiB.
- Lượt này không có số đo, nên theo quy tắc được chạy lại một lần.

## Lượt 2, 14/11: chạy bằng pin, GPU bị bóp giữa chừng ([attempt2-battery](attempt2-battery/))

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

### Quan sát phụ, không phải kết luận

- Ở cặp sạch (block0/1, cả hai bình thường), A1 giảm total 24–31% và expert_compute khoảng 28–35% ở 4/4 turn đo; decode khoảng 9,1 so với 6,7 tok/s.
- Ở turn6–7 của cặp 1, khi cả hai phía đều bị bóp GPU, on vẫn có total thấp hơn (0,89–0,90).
- Đây chỉ là một cặp, không đủ cho rule 8/8. Cần một lượt mới theo protocol đã sửa.

## Sửa protocol sau hồ sơ này (power guard, 14/11)

- `tools/preflight_memory.py` thêm hai điều kiện để eligible: máy đang cắm sạc (sysfs Mains online; máy không có Mains thì coi là có AC) và GPU không có lý do throttle `hw_slowdown`/`hw_thermal`/`sw_thermal`/`hw_power_brake`/`sw_power_cap` (nvidia-smi; không có GPU thì ghi nhận, không chặn).
- Harness A1 ghi trạng thái AC/GPU trước và sau mỗi block (`power_checks`). Nếu máy rời AC thì dừng như environment stop. Throttle trong lúc tải chỉ được ghi lại, không chặn.
- Chạy dưới `systemd-inhibit --what=idle:sleep` và tắt tự khóa/tự mờ màn hình trong lúc đo.
- Không đổi workload, mode, rule hay ngưỡng bộ nhớ.

## Bảo toàn payload

Hai run được chuyển nguyên vào `attempt1-swap-stopped/` và `attempt2-battery/`, kèm preflight, nvidia-smi, console và trích journal của lượt 2. Mọi file trừ README được khóa bằng CHECKSUMS.sha256.

```bash
sha256sum -c CHECKSUMS.sha256
```
