# Đề cương đồ án tốt nghiệp

Đề tài: suy luận Sparse Mixture-of-Experts trên hardware giới hạn, hiện thực Rust OLMoE INT8 với cache RAM/VRAM và đọc file theo nhu cầu. Đây là đề cương tổ chức, không phải nguồn xác nhận số đo.

## Chương 1: Bài toán và phạm vi

- Memory footprint toàn model so với số expert active; tài nguyên hai testbed i5/i7.
- Phân biệt tính đúng engine, chất lượng model, hiệu năng và khả năng phục vụ HTTP.
- Phạm vi local, một request OLMoE, greedy/context cấu hình; giới hạn đã biết.

## Chương 2: Cơ sở và nghiên cứu liên quan

- Transformer/RoPE/MHA/GQA, sparse MoE và routing Top-K, lượng tử hóa INT8.
- Cache/bộ nhớ phân tầng, I/O/page cache, CPU Rayon và dense cuBLAS.
- Đọc tài liệu/bài báo gốc, ghi nguồn/version; không dùng mô tả do agent tạo làm bằng chứng thực nghiệm.

## Chương 3: Thiết kế và hiện thực

- Theo [ARCHITECTURE](../ARCHITECTURE.md), chỉ rõ expert CPU/dense GPU, ownership/lifetime và cancellation.
- Loader/validation, prefill grouping, KV sessions, cache accounting và fallback.
- Phân biệt invariant mong muốn với lỗi chưa vá trong [PLAN](../PLAN.md).

## Chương 4: Phương pháp và kết quả

- Dùng [protocol](../../benchmarks/README.md), công bố CPU/GPU/RAM/driver quan sát thực, checkpoint/revision/suite/config/hash.
- Tính đúng fixture với tolerance; quality smoke tách khỏi đánh giá chuẩn hóa.
- Hiệu năng từng mẫu, load/prefill/TTFT/decode/RSS; cache state và timer overlap.
- Thí nghiệm tối ưu cùng máy/config tách khỏi thí nghiệm so sánh hai máy cùng revision/config.
- Trình bày thất bại, overhead và điều kiện không cải thiện; chỉ đưa claim có ánh xạ trong [EVIDENCE_MATRIX](EVIDENCE_MATRIX.md).

## Chương 5: Đánh giá và hướng phát triển

- Giới hạn fixture nhỏ/provenance lịch sử/counter/RSS so với capacity và OS cache.
- Chưa khẳng định zero-copy, zero-allocation, hết page faults hoặc tối ưu toàn cục khi chưa đo.
- Kế hoạch correctness/accounting trước baseline mới, rồi INT4/backend khác nếu còn phạm vi.

## Phụ lục

Lệnh tái lập, hashes/raw/log/run ID, test commands và kết quả, workflow AI hỗ trợ phát triển, đóng góp hai thành viên và giới hạn vai trò agents. Chỉ mô tả công cụ đã thực sự sử dụng; không gọi file cấu hình là hệ thống agent đã được kiểm chứng thực thi.
