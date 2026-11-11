# Thử nghiệm Tầng VRAM Expert A/B trên Testbed i7 (11/11/2026)

Hồ sơ ghi nhận lượt chạy đo tầng VRAM expert (`MOE_GPU_EXPERT_TIER=on/off`) trên testbed i7 (`RTX 5060 Laptop GPU`, 8151 MiB VRAM).

## 1. Môi trường & Thiết bị

- **Máy chủ**: Testbed i7 (`x86_64`, Linux 6.8.0-52-generic)
- **CPU**: Intel Core i7
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (Total VRAM: 8151 MiB, Driver 595.99.02)
- **Rust binary**: `target/release/moe-tier-engine` (SHA-256: `5b091a4df13fbb6f528907157bbe05f138adfaa29a5e5a00101c7587dec29aa9`)
- **Git Commit**: `b3dbec3d2b9b5fe76d3580159b4a7f23ff4bfb2f` (clean)
- **Checkpoint**: `models/olmoe-1b-7b-int8` (checkpoint kỳ 1 trên testbed i7)

## 2. Kiểm tra sơ bộ (Pre-check)

- `python3 tools/check.py --rust --release --web`: Toàn bộ test pass.
- `cargo test --locked --lib gpu -- --nocapture --test-threads=1`:
  - `Available: true`
  - `Total VRAM: 7.53 GB`
  - `Dense Offloaded: true`
  - 13 passed, 0 failed, các test CUDA thực thi đúng (EXECUTED).
- Fixture hai chế độ:
  - `--gpu-expert-tier on`: completed
  - `--gpu-expert-tier off`: completed
  - Output text khớp nhau và khớp kỳ vọng (`['token18 token12 token24', 'token30 token6 token21']`).

## 3. Kết quả các phép đo (mỗi lệnh chạy đúng 1 lần)

### Phép A: Prefill (`--expert-batch-workload long-prefill --max-tokens 1`)
- **Lệnh**:
  ```bash
  python3 tools/run_benchmark.py --testbed i7 --gpu auto --gpu-tier-model-ab \
    --expert-batch-workload long-prefill --threads 4 --cache 64 --context 512 \
    --max-tokens 1 --lookahead off --model-dir models/olmoe-1b-7b-int8
  ```
- **Kết quả**: `failed` ở block 0 (`run-a-prefill/result.json`).
- **Nguyên nhân**: `prompt token count drift`. Hằng số kỳ vọng của harness là `[64, 128, 64, 128]` (khớp tokenizer checkpoint i5). Tokenizer hiện có của checkpoint i7 encode ra `[61, 125, 61, 125]`.

### Phép B: Mixed Chat / Decode (`--expert-batch-workload mixed --max-tokens 64`)
- **Lệnh**:
  ```bash
  python3 tools/run_benchmark.py --testbed i7 --gpu auto --gpu-tier-model-ab \
    --expert-batch-workload mixed --threads 4 --cache 64 --context 512 \
    --max-tokens 64 --lookahead off --model-dir models/olmoe-1b-7b-int8
  ```
- **Kết quả**: `watchdog_stopped` ở block 0 (`run-b-chat/result.json`).
- **Nguyên nhân**: `process swap watchdog threshold`. `VmSwap: 188360 KiB` (> 128 MiB) do máy trạm có swap sẵn từ hệ thống nền.

## 4. Bảo toàn nguyên bản (Raw artifacts)
- `run-a-prefill/`: nguyên vẹn thư mục run của phép A.
- `run-b-chat/`: nguyên vẹn thư mục run của phép B.
- `nvidia-smi-pre.csv`, `preflight.json`: thông tin trạng thái bộ nhớ và GPU trước chạy.
