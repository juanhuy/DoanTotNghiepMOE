# Tầng VRAM expert bật/tắt trên i7 (RTX 5060 Laptop 7,53 GiB), 11/11/2026

## Câu hỏi và protocol

Theo [WORKFLOWS — Nhiệm vụ i7: tầng VRAM expert bật/tắt](../../../docs/WORKFLOWS.md#nhi%E1%BB%87m-v%E1%BB%A5-i7-t%E1%BA%A7ng-vram-expert-b%E1%BA%ADtt%E1%BA%AFt-giao-10112026): trên i5 (GTX 1650 Max-Q 4 GB, pool ~298/1024 expert), tắt tầng VRAM expert (`MOE_GPU_EXPERT_TIER=off`, vẫn giữ dense trên GPU) nhanh hơn ở cả prefill (`1,21–1,45×`) lẫn chat/decode (`total 1,39–1,87×`, decode trung vị `3,13 → 5,53 tok/s`). Máy i7 có GPU lớn hơn (`RTX 5060 Laptop GPU`, `7,53 GB` VRAM theo OS/CUDA) nên có thể chứa phần lớn expert; hai phép đo dưới đây kiểm chứng trên chính i7 bằng **cùng checkpoint INT8 chung với i5**, không so thời gian tuyệt đối giữa hai máy.

Lượt chạy đầu trên nhánh `i7-memory-20261111` (commit `b3dbec3`) dừng ở `block-0-on` (A do lệch token theo tokenizer cũ của i7, B do swap) và không sinh số đo; hồ sơ này là lượt đo lại (`-r2`) đầy đủ trên `main` mới nhất (`8941d29`).

- **Protocol**: giữ nguyên [SCENARIOS](../../SCENARIOS.md): `gpu-tier-long-v1` (Phép A, 16 request = 4 block `on→off→off→on`, `max_tokens=1`, warmup turn 0–1, đo turn 2–3) và `gpu-tier-chat-v1` (Phép B, 32 request = 4 block `on→off→off→on`, `max_tokens=64`, nhóm 1 turn 0–3 warmup, nhóm 2 turn 4–7 đo).
- **Cấu hình cố định**: `--gpu auto` (dense cuBLAS hai phía), CPU `4` threads (cố định theo protocol, không phải cấu hình tối ưu của i7), cache `64 MiB/layer`, `context=512`, `victim`, `Gate–Up paired`, batch prefill `off`, `lookahead off`, `FP32 KV`, `KV reuse off`, `trace on`, `greedy`.

```bash
# Phép A — prefill (16 request, max 1)
python3 tools/run_benchmark.py --testbed i7 --gpu auto --gpu-tier-model-ab \
  --expert-batch-workload long-prefill --threads 4 --cache 64 --context 512 \
  --max-tokens 1 --lookahead off --model-dir models/olmoe-i5

# Phép B — chat nhiều lượt / decode (32 request, max 64)
python3 tools/run_benchmark.py --testbed i7 --gpu auto --gpu-tier-model-ab \
  --expert-batch-workload mixed --threads 4 --cache 64 --context 512 \
  --max-tokens 64 --lookahead off --model-dir models/olmoe-i5
```

## Provenance và kiểm trước đo

- **Branch & commit**: `work/i7-gpu-tier` tại HEAD `8941d2977b7541c9f04bb05c88aaf44ea1337fe0`, `git_dirty: false` ở cả 8 block.
- **Phần cứng & OS quan sát**: `Intel(R) Core(TM) i7-14650HX` (24 logical CPUs, ~14,8 GiB physical RAM), `NVIDIA GeForce RTX 5060 Laptop GPU` (`8151 MiB` theo `nvidia-smi` / `7,53 GB` theo CUDA `cuMemGetInfo`, driver `595.99.02`), Linux `7.0.0-42-generic` x86_64, `rustc 1.98.1 (48a229cea 2026-09-01)`.
- **Binary release**: `target/release/moe-tier-engine`, SHA-256 `5b091a4df13fbb6f528907157bbe05f138adfaa29a5e5a00101c7587dec29aa9`.
- **Checkpoint**: `models/olmoe-i5`, khớp **16/16** SHA-256 trong [olmoe-int8-i5.sha256](../../checkpoints/olmoe-int8-i5.sha256), không có `.moe_usage`.
- **Kiểm trước đo**:
  - `python3 tools/check.py --rust --release --web`: PASS (104 Python, fmt/Clippy, Rust debug/release, 7 Node).
  - `cargo test --locked --lib gpu -- --nocapture --test-threads=1`: PASS 13/13 (`Available: true`, `Total VRAM: 7.53 GB`, `Dense Offloaded: true`, 3 test CUDA `EXECUTED`).
  - Fixture `check_ram_stability` (`on`/`off`): cả 2 `completed`, text khớp (`['token18 token12 token24', 'token30 token6 token21']`), `gpu_expert_tier_enabled` đúng `True`/`False`.
  - [Preflight A](preflight-a.json): `eligible: true`, min `MemAvailable` `10954,4 MiB`, `pswpout: [536827, 536827, 536827]`; [nvidia-smi A](nvidia-smi-pre-a.csv): `13 MiB`, `0 %`, `56 °C`.
  - [Preflight B](preflight-b.json): `eligible: true`, min `MemAvailable` `11694,4 MiB`, `pswpout: [537097, 537097, 537097]`; [nvidia-smi B](nvidia-smi-pre-b.csv): `13 MiB`, `0 %`, `53 °C`.

## Kết quả Phép A — Prefill (`gpu-tier-long-v1`, max 1)

- **Run**: `run-a-prefill` (`20261111T164415.269086Z-gpu-tier-model-73de7ce20cdb4e0ab863fb2d41782016`), **COMPLETED 16/16**, không watchdog violation. `prompt_tokens` `[64, 128, 64, 128]`, `completion_tokens=1`, text/usage khớp mọi block.
- **Pool VRAM thực tế (`on`)**: `vram_experts_resident` đạt **845 / 1024** expert (~82,5%), `6500 MiB` VRAM logical (`1417 MiB` ở block `off`, `0` VRAM upload/load).
- **Rule đối xứng (`on/off ≥ 1,05` ở cả 4 → tắt nhanh hơn; `≤ 1/1,05` → bật nhanh hơn)**: **`off_faster: False`, `on_faster: False`** (không nhất quán — `on` nhanh hơn ở 1/4, `off` nhanh hơn ở 3/4).

| Cặp | Prompt (turn) | TTFT on/off (ms) | TTFT ratio | expert_compute on/off (ms) | File loads on/off | VRAM upload on (MiB / ms) | VRAM download on (ms) |
| --- | --- | --- | ---: | --- | --- | --- | ---: |
| 0 | bakery64 (turn 2) | 2874,7 / 3353,3 | 0,857 | 1042,8 / 917,8 | 0 / 871 | 5240 / 843,8 | 850,1 |
| 0 | library128 (turn 3) | 3839,8 / 3237,2 | 1,186 | 1706,5 / 1695,7 | 0 / 904 | 5438 / 876,3 | 895,1 |
| 1 | bakery64 (turn 2) | 3094,0 / 2424,8 | 1,276 | 1164,9 / 1031,5 | 0 / 871 | 5240 / 876,9 | 898,9 |
| 1 | library128 (turn 3) | 4099,0 / 3080,9 | 1,330 | 1904,6 / 1652,5 | 0 / 904 | 5438 / 918,5 | 933,8 |

Warmup loại theo protocol (total ratio `on/off`): cặp 0 turn 0 `1,236`, turn 1 `0,824`; cặp 1 turn 0 `0,924`, turn 1 `0,787` (lượt `turn 1` cả hai cặp đều nhanh hơn khi bật `on`, dù phải upload 5799 MiB + 919 VRAM loads).

## Kết quả Phép B — Chat nhiều lượt / Decode (`gpu-tier-chat-v1`, max 64)

- **Run**: `run-b-chat` (`20261111T164600.363795Z-gpu-tier-model-c75eabd5d9a744b2a755c44084f018c3`), **COMPLETED 32/32**, không watchdog violation. `prompt_tokens` `[19, 107, 22, 20, 19, 107, 22, 20]`, `completion_tokens=64`, `cached_prompt_tokens=11` ở follow-up, text/usage khớp mọi turn giữa các block (parity HTTP text/token-count, không phải logits parity).
- **Pool VRAM thực tế (`on`)**: `vram_experts_resident` đạt **858 / 1024** expert (~83,8%), `6578 MiB` VRAM logical (`1417 MiB` ở block `off`, `0` VRAM upload/load).
- **Rule đối xứng (`total_ms` VÀ `decode_ms` ratio `on/off ≥ 1,05` ở cả 8)**: **`off_faster: True` — ĐẠT 8/8.**

| Cặp | Prompt (turn) | total on/off (ms) | total ratio | TTFT ratio | decode ratio | decode tok/s on/off | File loads on/off | VRAM upload on (MiB) |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 0 | Mars (turn 4) | 13592,1 / 6871,4 | 1,978 | 1,899 | 1,988 | 5,21 / 10,36 | 0 / 5532 | 33357 |
| 0 | Follow-up (turn 5) | 15709,1 / 8804,7 | 1,784 | 1,390 | 1,937 | 5,13 / 9,93 | 0 / 5994 | 36124 |
| 0 | Python (turn 6) | 14703,3 / 7533,0 | 1,952 | 1,565 | 2,007 | 4,77 / 9,57 | 0 / 6319 | 37977 |
| 0 | Love (turn 7) | 15881,6 / 8280,7 | 1,918 | 1,644 | 1,951 | 4,38 / 8,54 | 0 / 6737 | 40557 |
| 1 | Mars (turn 4) | 13583,7 / 6873,7 | 1,976 | 2,022 | 1,970 | 5,26 / 10,37 | 0 / 5532 | 33357 |
| 1 | Follow-up (turn 5) | 15608,1 / 8673,0 | 1,800 | 1,359 | 1,965 | 5,08 / 9,98 | 0 / 5994 | 36124 |
| 1 | Python (turn 6) | 14852,4 / 7852,4 | 1,891 | 1,580 | 1,932 | 4,69 / 9,07 | 0 / 6319 | 37977 |
| 1 | Love (turn 7) | 15911,0 / 7963,0 | 1,998 | 1,835 | 2,019 | 4,42 / 8,93 | 0 / 6737 | 40557 |

Warmup loại theo protocol (nhóm 1, turn 0–3) cùng chiều: `total` ratio `1,735–2,078`, decode tok/s `on` `4,27–5,20` so với `off` `8,68–10,05`.

### Diễn giải theo thành phần

- **Tỉ lệ pool trên i7 đạt 82,5–83,8% (845–858 / 1024 expert)**, nên ở cả Phép A và Phép B khi bật `on`, `expert_file_loads` thực tế **bằng `0`** trong các turn đo (mọi miss L2 RAM 64 MiB/layer đều được phục vụ từ VRAM thay vì đọc file).
- Tuy nhiên với policy `victim`, mọi expert bị evict khỏi RAM 64 MiB/layer vẫn bị đẩy lên VRAM (và đẩy ngược expert cũ ra khỏi pool 845–858 slot). Ở Phép B (64 token decode), mỗi request phải thực hiện **5545–6742 lần load VRAM**, upload **33357–40557 MiB (32,6–39,6 GiB/request, mất 5,46–6,71 s)** và download **5,27–6,43 s**. Trong khi đó khi tắt tầng VRAM (`off`), OS page cache trên RAM trống phục vụ 5532–6737 lần đọc file chỉ mất **2,86–3,53 s** (`expert_file_read_ms`).
- Ở Phép B, `expert_compute_ms` khi bật `on` thấp hơn `off` (`0,607–0,719×`, ví dụ `2098–3212 ms` vs `3303–4464 ms`), nhưng chi phí truyền PCIe upload+download (~10,7–13,1 s/request) lớn hơn nhiều nên **tổng thời gian tắt `off` vẫn nhanh hơn `1,784–1,998×`** (decode tok/s trung vị tăng từ **`4,92` lên `9,75 tok/s`**).
- Ở Phép A (chỉ prefill 1 token, lượng upload chỉ 5,2–5,4 GiB / 0,84–0,92 s), chi phí đọc file của `off` ở `block-1-off` lượt đầu (`bakery64` 2,24 s) cao hơn chi phí upload+download của `block-0-on` (`1,69 s`), khiến cặp 0 turn 2 nghiêng về `on` (`0,857×`) còn 3 lượt còn lại (khi page cache ấm hoặc prompt 128 token) nghiêng về `off` (`1,186–1,330×`).

### Tài nguyên

- **Phép A**: RSS max `on` `3252,2 / 3251,8 MiB`, `off` `3313,0 / 3313,1 MiB`; `VmSwap = 0,0 MiB` ở mọi block; `MemAvailable min` `8492,3–8528,7 MiB`; `pswpout` tăng `112` page ở `block-0-on`, `0` ở 3 block sau.
- **Phép B**: RSS max `on` `3427,9 / 3428,4 MiB`, `off` `3418,4 / 3423,8 MiB`; `VmSwap = 0,0 MiB` ở mọi block; `MemAvailable min` `8350,8–8366,0 MiB`; `pswpout` tăng `0` page ở cả 4 block.

## Kết luận và giới hạn

- **Trên i7 (RTX 5060 Laptop 7,53 GiB VRAM, 845–858/1024 expert resident), tắt tầng VRAM expert (`off`) nhanh hơn rõ 8/8 ở chat/decode (`total 1,78–2,00×`, decode tok/s trung vị `4,92 → 9,75`), còn ở prefill max 1 cho kết quả không nhất quán (`off_faster: False, on_faster: False`, `on` thắng 1/4, `off` thắng 3/4).**
- Đối chiếu bảng quyết định mục 6 trong [WORKFLOWS](../../../docs/WORKFLOWS.md#nhi%E1%BB%87m-v%E1%BB%A5-i7-t%E1%BA%A7ng-vram-expert-b%E1%BA%ADtt%E1%BA%AFt-giao-10112026): Phép B đạt `off_faster` 8/8 và Phép A không nhất quán (không có `on_faster` ở bất kỳ phép nào). Chưa đổi mặc định `MOE_GPU_EXPERT_TIER` trong code ở bước này — chuyển kết quả để hai bên thống nhất quyết định cuối cùng.
- Giới hạn: một lượt ABBA cho mỗi workload, CPU cố định `4` threads (không phải cấu hình tối ưu của i7), `cache=64 MiB/layer`, policy `victim`, không đo với cache lớn hơn, `inclusive` hay upload bất đồng bộ / admission chọn lọc; `VRAM` là số logical do engine báo.

## Bảo toàn payload

Giữ nguyên toàn bộ payload hai run (`run-a-prefill/`, `run-b-chat/`) cùng 4 file preflight/nvidia-smi và `README.md`; khóa toàn bộ bằng `CHECKSUMS.sha256`.

```bash
sha256sum -c CHECKSUMS.sha256
```

