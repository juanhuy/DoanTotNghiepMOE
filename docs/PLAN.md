# Kế hoạch hoàn thiện MoE-TierEngine

Mục tiêu bản đầu: OLMoE merged INT8 chạy CPU + RAM + SSD, có chat streaming,
hủy yêu cầu, giao diện sử dụng được và số đo tái lập. Không cam kết model trả
lời đúng mọi câu hỏi. Chất lượng checkpoint phải đánh giá riêng với tính đúng
của engine/giao thức.

## Tiến độ và nghiệm thu

| Giai đoạn | Trạng thái | Điều kiện nghiệm thu |
| --- | --- | --- |
| 1. Baseline và phạm vi | Hoàn thành bản đầu | Suite v1 5 case × 3 vòng đã đo; có JSON cấu hình/metrics/RSS; token ổn định, đánh giá nội dung ban đầu trong PROGRESS |
| 2. SSE và hủy | Hoàn thành bản đầu cho OLMoE | Chunk content trước khi hoàn tất; validation trước SSE; DONE/usage; hủy trong prefill/decode; nhận request mới sau hủy |
| 3. API và hồi quy | Hoàn thành bản local | Có validation, giới hạn request, queue timeout, lỗi stream, session KV prefix cache 256 MiB và smoke 429/health; còn TTL/timeout tổng nếu vận hành production |
| 4. Giao diện chat | Hoàn thành bản local | Streaming, lịch sử bền có giới hạn, chọn max token, dừng/xóa, usage và metrics; HTML nhúng trong binary |
| 5. Tối ưu và đánh giá | Hoàn thành baseline v1 | Đã tối ưu INT8/dense đa lõi và prefill gom expert; giữ logits fixture, đo prompt 469 token; quality smoke 12/12 và stress HTTP 50/50; còn attention dài và GPU hybrid |
| 6. Đóng gói và báo cáo | Hoàn thành bản demo local | README, nhật ký số đo, dữ liệu thô và kịch bản demo tái lập; chưa có installer/container hay production deployment |

Ước lượng ban đầu 4–6 tuần là định hướng tổ chức, không phải thời hạn đã xác
nhận. Mỗi bước cập nhật PROGRESS và lưu kết quả thô; không coi chạy smoke là
đánh giá chất lượng hoàn chỉnh.

## Phạm vi bản đầu

- Một yêu cầu OLMoE tại một thời điểm; yêu cầu chồng nhau nhận 429.
- Greedy decoding, giới hạn context cấu hình, cache expert hiện có.
- Streaming và hủy qua đóng kết nối HTTP. Chế độ không streaming giữ hành vi
  cũ, chưa hủy worker khi client ngắt.
- Chạy local; chưa triển khai public production hoặc xác thực nhiều người dùng.
- Không thêm GPU, INT4, nhiều họ model hoặc inference đồng thời trong bản đầu.
- SIMD chuyên biệt/đa luồng chỉ triển khai khi đo xác nhận lợi ích và còn đủ
  thời gian kiểm chứng. Prefill layer-major đã triển khai bản đầu.

## Thứ tự tiếp tục

1. Mở rộng quality-v1 bằng tập dữ liệu độc lập lớn hơn nếu cần kết luận về chất lượng model; suite 12 câu hiện chỉ là smoke heuristic.
2. Tối ưu attention causal theo batch/SIMD hoặc bổ sung backend GPU hybrid; giữ suite 469 token làm hồi quy.
3. Chạy stress dài bằng checkpoint thật; bổ sung TTL/session observability và timeout tổng request nếu mục tiêu chuyển sang production.
4. Chỉ bổ sung installer/container, xác thực và quan sát vận hành khi phạm vi chuyển từ demo local sang triển khai.
