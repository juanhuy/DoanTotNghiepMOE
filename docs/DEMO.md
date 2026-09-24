# Kịch bản demo local

## Kiểm tra trước khi trình bày

Từ thư mục `MoE`, chạy:

```bash
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo build --release --bin moe-tier-engine
```

Checkpoint phải có tại `models/olmoe-1b-7b-int8`. Model không được đưa vào Git,
vì vậy cần sao chép hoặc chuyển đổi lại khi chạy trên máy khác.

## Mở ứng dụng

```bash
MOE_MODEL_DIR=./models/olmoe-1b-7b-int8 \
MOE_BIND=127.0.0.1:8081 \
MOE_CONTEXT=512 \
MOE_CACHE_BYTES_PER_LAYER=134217728 \
target/release/moe-tier-engine
```

Mở `http://127.0.0.1:8081/`. Trình tự demo đề xuất:

1. Hỏi `What is 2 + 2?` và chỉ ra nội dung streaming cùng TTFT/decode metrics.
2. Hỏi tiếp một câu có liên quan để thể hiện lịch sử hội thoại được gửi lại.
3. Bắt đầu câu trả lời dài rồi bấm **Dừng**; gửi câu mới để xác nhận permit đã được giải phóng.
4. Tải lại trang để kiểm tra lịch sử còn trong trình duyệt, sau đó bấm **Xóa**.

## Bằng chứng có thể tái lập

```bash
target/release/examples/evaluate_quality \
  models/olmoe-1b-7b-int8 benchmarks/quality-v1.json \
  > benchmarks/results/quality-v1.json

python3 tools/stress_http.py tests/fixtures/olmoe/unnormalized-mha \
  benchmarks/results/http-stress.json 50
```

Quality-v1 là smoke test exact/substring 12 câu, không phải benchmark chuẩn hóa.
Stress fixture xác nhận giao thức và vòng đời request, không đại diện thời gian
hay RSS của checkpoint thật. Các benchmark dài và giới hạn phép đo được ghi tại
[`PROGRESS.md`](PROGRESS.md).
