# MoE-TierEngine

Đồ án tốt nghiệp nghiên cứu suy luận Sparse Mixture-of-Experts trên phần cứng giới hạn. Engine Rust chạy OLMoE merged INT8, đọc expert theo nhu cầu từ file, cache ở RAM/VRAM; cuBLAS có thể tính dense attention và LM head. API chat hỗ trợ SSE và hủy yêu cầu.

Hai testbed cộng tác là **i5 (máy của Sơn)** và **i7 (máy đồng nghiệp)**. Nhãn máy được khai báo khi đo; CPU/GPU/RAM thực tế phải thu nhận trong từng run. Dữ liệu lịch sử chưa đủ provenance để kết luận máy nào nhanh hơn. Xem [giao thức benchmark](benchmarks/README.md).

## Nhận bản bàn giao i5 → i7

Branch: **handoff/i5-memory-20261108**. Đọc [WORKFLOWS — tiếp nhận i7](docs/WORKFLOWS.md#tiếp-nhận-trên-testbed-i7) để nhận đúng source, chạy checks và smoke fixture trước model thật. [PLAN](docs/PLAN.md) giữ bước phát triển tiếp, [SYSTEM_GUIDE](docs/SYSTEM_GUIDE.md) giải thích hệ thống; các số đo i5 không phải kết quả i7.

## Bắt đầu

Cần Rust stable với Cargo/rustfmt/Clippy và Python 3.11+ cho công cụ stdlib. Fixture nhỏ đã có sẵn, không cần tải model để test.

```bash
python3 tools/check.py --rust
cargo build --release --locked --bin moe-tier-engine
```

Sau khi chuẩn bị checkpoint tại `models/olmoe-1b-7b-int8`, chạy từ root repo:

```bash
cargo run --release --locked --bin moe-tier-engine
```

Mở `http://127.0.0.1:8080`. [CONTRIBUTING.md](CONTRIBUTING.md) hướng dẫn setup; cấu hình demo được nêu ngay bên dưới.

Cache expert mặc định dùng `victim`: VRAM giữ expert bị đẩy khỏi RAM, tránh giữ bản sao trùng ở hai tầng. Đặt `MOE_EXPERT_CACHE_POLICY=inclusive` để dùng chính sách cũ khi đối chứng. [Thí nghiệm HTTP Mars→Love trên i5](benchmarks/experiments/2026-10-22-web-session-i5-081202/README.md) ghi từng mẫu, cấu hình128MiB/layer/threads8 và giới hạn; không cam kết mức tăng tốc cho mọi cấu hình.

Server mặc định cache64MiB/layer cố định, không tự tăng theo RAM trống; `MOE_CACHE_BYTES_PER_LAYER` override tường minh. RAM preflight/guard có thể từ chối load hoặc trả lỗi request khi RSS/headroom không đủ. Đây là bảo vệ hợp tác, chưa hard RSS cap hay bảo đảm hết OOM; xem [kiến trúc](docs/ARCHITECTURE.md) và [quy trình kiểm RAM](docs/WORKFLOWS.md). Mức cache chung để ghi báo cáo i5/i7 vẫn cần kiểm ổn định trên từng máy.

Để chạy CPU ngay từ lúc nạp, đặt `MOE_GPU_MODE=cpu`; chế độ này bỏ qua CUDA/VRAM và cần khởi động lại với `auto` nếu muốn dùng GPU. Runner `--gpu cpu` tự đặt lựa chọn này. Các record cũ CPU tắt sau load có điều kiện khởi tạo khác.

Candidate tái sử dụng buffer KV chọn bằng `MOE_KV_BUFFER_REUSE=on|off`, mặc định off. Khi cần loại session theo ngân sách, candidate có thể chuyển buffer của session LRU sang request mới sau khi xóa trạng thái KV; nếu request mới hủy, phiên cũ phải recompute từ lịch sử client gửi. Đã kiểm fixture và một cặp model thật: tránh overlap một bộKV128MiB, outputHTTP2/2khớp; lợi ích RSS chỉ là tín hiệu có giới hạn, chưa chứng minh tốc độ hoặc mức giảm tổng quát. Xem [điều kiện và đối chứng](benchmarks/SCENARIOS.md#36-candidate-tái-sử-dụng-buffer-kv-05112026).

## Demo và kiểm nhanh

Chọn checkpoint bằng `MOE_MODEL_DIR`; đổi địa chỉ bằng `MOE_BIND`. Ví dụ dùng cache64MiB/lớp và context512:

```bash
MOE_MODEL_DIR=models/olmoe-1b-7b-int8 \
MOE_BIND=127.0.0.1:8080 MOE_CONTEXT=512 \
MOE_CACHE_BYTES_PER_LAYER=67108864 target/release/moe-tier-engine
```

Thử hỏi ngắn, bấm Stop giữa câu trả lời rồi gửi lại. Phản hồi lỗi/hủy không được lưu vào lịch sử hoàn tất. Xóa chat phía client khác reset cache server; reset server không xóa OS page cache hoặc bảo đảm trả hết RAM/VRAM vật lý. Hình routing và VRAM logical trên web không thay thế profiler.

Kiểm parser bằng `python3 tools/check.py --web`; kiểm giao thức HTTP trên fixture bằng `python3 tools/check.py --http`. Smoke nội dung model thật, chạy riêng sau khi đã dừng server:

```bash
cargo run --release --locked --example evaluate_quality -- models/olmoe-1b-7b-int8 benchmarks/suites/quality-v1.json
```

Smoke không phải đánh giá chất lượng chuẩn hóa. Khi đo hiệu năng, dùng [kịch bản benchmark](benchmarks/SCENARIOS.md); không lấy thời gian thao tác demo làm đối chứng tối ưu.

## Bản đồ project

| Vị trí | Vai trò |
| --- | --- |
| `src/` | Engine, HTTP, kernel CPU, loader, CUDA và tiny reference model |
| `web/index.html` | Giao diện nhúng trong binary |
| `tests/fixtures/olmoe/` | Fixture MHA/GQA và logits tham chiếu |
| `examples/` | Driver chat, benchmark và quality smoke |
| `tools/` | Converter, kiểm tra repo và công cụ bằng chứng |
| `benchmarks/suites/` | Kịch bản đo được version control |
| `benchmarks/results/` | 40 artifact lịch sử bất biến, có manifest SHA-256 |
| `benchmarks/runs/` | Run mới theo testbed, có ID/metadata/log riêng |
| `.agents/skills/`, `.codex/` | Skills và cấu hình agent dùng chung |
| `docs/`, `baocao/` | Tài liệu kỹ thuật và báo cáo kỳ 1 |

## Coding agents và bằng chứng

Đọc [AGENTS.md](AGENTS.md), [kiến trúc](docs/ARCHITECTURE.md), [workflow](docs/WORKFLOWS.md). Một owner tích hợp; chỉ tách nhánh có phạm vi độc lập. Hai thành viên dùng cùng lệnh local/CI và cùng giao thức benchmark.

Không cần tài khoản OpenAI API, plugin hoặc MCP để build/test engine. Coding agent là công cụ phát triển, không phải dependency runtime.

- [Học đồ án theo từng module](docs/research/LEARNING_PLAN.md): ví dụ nhỏ, đường đọc code, bài tập và đáp án.
- [Báo cáo kỳ 1](baocao/README.md): bản nộp, cách biên dịch và hướng dẫn thực nghiệm.
- [Báo cáo i5 phase1-v1](benchmarks/experiments/2026-10-22-phase1-i5/README.md): bộ đo cuối trên checkpoint sạch,4 nhóm ×5 mẫu.
- [PLAN.md](docs/PLAN.md): công việc tiếp theo và giới hạn cần kiểm chứng.
- [PROGRESS.md](docs/PROGRESS.md): lịch sử, số đo, đính chính và kết quả mới.
- [BENCHMARKS.md](docs/BENCHMARKS.md): danh mục sinh từ manifest.
- [Ma trận luận điểm](docs/thesis/EVIDENCE_MATRIX.md): bằng chứng và giới hạn.
- [Review hệ thống](docs/reviews/2026-10-22-system-review.md): R1–R9, chưa coi đã sửa chỉ vì test pass.

Phạm vi hiện tại là demo/nghiên cứu local, greedy decoding, một request OLMoE tại một thời điểm. INT4 chưa tích hợp decoder. Fixture logits, quality smoke và benchmark hiệu năng là ba loại kiểm chứng khác nhau. Chưa có bằng chứng toàn đường chạy zero-allocation, zero-copy SSD hoặc hết page faults.

## Tài liệu hệ thống

Bắt đầu đọc [Hướng dẫn hệ thống MoE-TierEngine](docs/SYSTEM_GUIDE.md) để hiểu luồng xử lý, thuật toán, tối ưu, thuật ngữ và các file benchmark.
