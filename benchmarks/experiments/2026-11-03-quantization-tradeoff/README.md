# Khảo sát INT8/INT4 và routing từ nhánh Huy

Hai JSON được tiếp nhận nguyên byte từ commit c780d11 (nhánh Huy sau đó ở fdaa885), không phải phép đo mới trong lần merge.

- quantization_tradeoff_report.json: đoạn250ký tự/42token,41token được đánh giá. Disk giảm43,43%, evalthroughput tăng22,76%, perplexity103,19→113,82 (+10,30%). Thiếu source/checkpoint hashes, hardware và nội dung corpus đầy đủ trong record; chưa dùng làm kết luận chất lượng chuẩn hóa hoặc chatdecode tăng23%.
- route_analysis_report.json: trace_summary có1call/496records; không đại diện mọi workload hoặc chứng minh hotpin/predictor tăng tốc.
- CHECKSUMS.sha256 khóa hai payload tiếp nhận. Không chỉnh số hoặc bổ sung provenance hồi tố. Công cụ và hồi quy có trong repo; converterPython cần torch/numpy/safetensors, khác unit test loaderINT4 của Rust.

Đánh giá trên phiên bản hợp nhất cần protocol riêng: cùng checkpoint nguồn/corpus/tokenizer/backend/threads/budget, pin precision/sampling/usagecache, lưu từng mẫu và output/logits/chất lượng phù hợp. Không ghép hai JSON này với các sối5 làm đối chứng phần cứng.
