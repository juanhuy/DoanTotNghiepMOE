# Demo local

Chạy từ root repo:

```bash
python3 tools/check.py --rust --web
cargo build --release --locked --bin moe-tier-engine
```

Checkpoint cần ở `models/olmoe-1b-7b-int8` hoặc đường dẫn `MOE_MODEL_DIR`. Không đưa model vào Git. Trên Linux:

```bash
MOE_MODEL_DIR=./models/olmoe-1b-7b-int8 MOE_BIND=127.0.0.1:8081 MOE_CONTEXT=512 MOE_CACHE_BYTES_PER_LAYER=134217728 target/release/moe-tier-engine
```

Mở `http://127.0.0.1:8081/`. Demo hỏi ngắn, theo dõi stream, bấm Stop giữa câu trả lời dài rồi gửi lại. Xóa lịch sử phía client không đồng nghĩa xóa mọi cache server; R3 đã vá: reset xóa expert/KV entries và giữ ngân sách session, không xóa OS page cache hoặc giải phóng toàn bộ VRAM pool. Không dùng routing animation như bằng chứng expert thật hoặc VRAM display như số đo physical allocation.

Web chỉ lưu history/metrics sau terminal và DONE đầy đủ; stream lỗi/EOF thiếu terminal/Stop hiển thị trạng thái chưa hoàn tất và không lưu vào history. Kiểm parser bằng `python3 tools/check.py --web` (cần Node.js); chưa thay thế kiểm trình duyệt trực tiếp. [PLAN](PLAN.md) theo dõi các lỗi, [review](reviews/2026-10-22-system-review.md) có trigger.

Quality smoke riêng (cần model thật; không phải đánh giá chất lượng chuẩn hóa):

```bash
cargo run --release --locked --example evaluate_quality -- models/olmoe-1b-7b-int8 benchmarks/suites/quality-v1.json
```

Stress giao thức fixture:

```bash
python3 tools/check.py --http
```

Không redirect vào artifact lịch sử. Phép đo performance dùng [runner](../benchmarks/README.md), không trộn smoke demo với baseline hiệu năng.


RAM: server mặc định cache64MiB/layer; dùng MOE_CACHE_BYTES_PER_LAYER để chọn tường minh. Guard RSS/headroom có thể từ chối load hoặc request thay vì tiếp tục cấp phát. Đây là cơ chế hợp tác, chưa bảo đảm hết OOM; không tăng MOE_RAM_LIMIT_BYTES chỉ để bỏ qua lỗi. Giảm cache/context hoặc giải phóng ứng dụng khác rồi thử lại. Sau khi sửa runtime phải restart server. Theo dõi RAM dài bằng tools/check_ram_stability.py sau khi build và dừng server khác; xem WORKFLOWS/PLAN.
