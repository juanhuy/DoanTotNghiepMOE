# Ma trận luận điểm và bằng chứng

Cập nhật sau review 2026-10-22. Raw lịch sử bất biến được liệt kê tại [catalog](../BENCHMARKS.md); thiếu revision/hash/điều kiện không được tự điền. Mọi số đo cũ dưới đây là **hồi cứu**, không phải kết quả mới của lần chuẩn hóa.

| Luận điểm có thể trình bày | Mã/bằng chứng | Phạm vi và giới hạn |
| --- | --- | --- |
| Rust logits phù hợp tham chiếu trong tolerance fixture | [test](../../tests/olmoe_reference.rs), [fixtures](../../tests/fixtures/olmoe) | Test đặt sai số tuyệt đối `<2e-5` trên toy MHA/GQA; không nói 100% logits 7B hoặc bit-exact mọi backend |
| Có loader INT8 theo nhu cầu và cache expert | [loader](../../src/int8_expert.rs), [OLMoE](../../src/olmoe.rs) | Cache budget không bằng tổng RSS; R1/R4 đã vá capacity/bounded prefetch với hồi quy fixture, xem PROGRESS; đã có observed HWM7B trong bộ phase1-v1 dưới đây, không coi sampler là chứng minh peak tuyệt đối |
| Ngân sách cache tác động churn ở các lượt lịch sử | [32 MiB](../../benchmarks/results/testbed_a_i7_14650hx/2026-09-22-baseline.json), [128 MiB](../../benchmarks/results/testbed_a_i7_14650hx/2026-09-22-cache-128mib.json) | 128 MiB còn 913 cold/1140 warm evictions, không nói loại bỏ thrashing; không coi payload là SSD physical bytes |
| Prefill layer-major và grouping expert đã có | [prefill](../../src/olmoe.rs), [record](../../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-batched-prefill-context.json) | Hiệu quả phải đo với cùng config; grouping không chứng minh zero allocation |
| Có thí nghiệm các mức Rayon threads | [1](../../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-rayon-1-context.json), [8](../../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-rayon-8-context.json), [24](../../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-rayon-24-context.json) | Đó là các điểm đã thử; chưa đo RAM bandwidth/power để kết luận bão hòa hoặc tối ưu toàn cục |
| Có A/B microbenchmark attention | [raw](../../benchmarks/results/testbed_a_i7_14650hx/2026-09-24-attention-online-vs-materialized.json), [kernel](../../src/backend.rs) | Chỉ điều kiện microbenchmark đã ghi, không suy ra end-to-end hay mọi hardware |
| HTTP hỗ trợ SSE/hủy hợp tác, fixture có hồi quy | [HTTP tests](../../src/main.rs), [stress](../../tools/stress_http.py), [review evidence mới](../reviews/2026-10-22-evidence.json) | Fixture 30 vòng mới và record lịch sử là hai lần khác; không đại diện tải 7B; frontend R5 đã có parser terminal/DONE và7 hồi quy Node; chưa kiểm browser trực tiếp |
| INT8 qua bộ quality smoke lịch sử | [suite](../../benchmarks/suites/quality-v1.json), [raw](../../benchmarks/results/testbed_a_i7_14650hx/2026-09-23-quality-v1.json) | 12 câu exact/substring là heuristic, không benchmark chất lượng chuẩn hóa |
| Có backend cuBLAS và cache VRAM | [GPU](../../src/gpu.rs), [các raw lịch sử](../BENCHMARKS.md) | Review gốc không có CUDA; lượt vá R2 đã chạy host CUDA pinned/undersized fallback và logits fixture; R6 legacy/R7 accounting còn giới hạn, xem PROGRESS |
| So sánh có kiểm soát i5/i7 | [protocol](../../benchmarks/README.md), [machine labels](../../benchmarks/machines.json) | **Chưa có phép đo mới đủ provenance được nghiệm thu**; không dùng bảng cũ khác revision làm đối đầu hardware |

Không dùng các kết luận “zero-copy SSD”, “hết page faults”, “an toàn bộ nhớ tuyệt đối”, “8 threads tối ưu” hoặc tỷ lệ tăng tốc độc lập từ việc đổi đồng thời thuật toán/cache/backend nếu chưa có phép đo phù hợp. Khi bổ sung luận điểm, ghi run ID/checkpoint/revision/config/sample size và review điều kiện tương đương.


Bổ sung kiểm chứng đóng giai đoạn1 (2026-10-22): [artifact mới](../../benchmarks/experiments/2026-10-22-stage1-validation/README.md) giữ source/hash/log và cả lượt harness thất bại. Smoke7B i5:16 request hoàn tất,1 cố ý hủy rồi reset/retry đạt; quality heuristic10/12, lỗi thủ đô Việt Nam và retrieval-day. Có hoạt động swap hệ thống. Đây là dữ liệu exploratory một máy, chưa baseline revision sạch hoặc đánh giá chất lượng chuẩn hóa; không suy ra speedup của R1/R2/R4/R5.


Bổ sung RAM guard: [raw/source/time series i5](../../benchmarks/experiments/2026-10-22-ram-guard-i5/README.md),24 requests chia hai lượt32/128 token với cache64MiB/guard5GiB, observed HWM max3,47/3,53GiB, process swap0, không watchdog stop. Preflight từ chối cache256MiB trong kiểm âm tính có giám sát. Không suy ra % giảm RAM so OOM cũ hoặc chứng minh không leak; config khác, source dirty và thời gian hữu hạn. Guard hợp tác không hard RSS cap; chưa đo i7 hoặc causal speedup.


Bổ sung bộ báo cáo **phase1-v1 nguồn sạch**: [báo cáo i5 chính](../../benchmarks/experiments/2026-10-22-phase1-i5/README.md), source `d4adefd9e53627bdc1fd250eabab436c3ac54b0d`, checkpoint/source/binary hashes khớp3 records; cache64MiB/layer/context512/threads8/max128/victim/guard5GiB/headroom1GiB/GPU enabled. Cold/warm/conversation,20 mẫu +5 warmup, mỗi nhóm5 mẫu: decode median3,663/3,987/3,681/3,228 steps/s (cold Mars/warm Mars/conversation Mars/Love). Observed HWM max3,494GiB, process swap0, không watchdog stop; text/usage khớp trong nhóm, finish length128. Dùng để mô tả vận hành i5 trên phiên bản này; không chứng minh causal speedup, không chuẩn hóa chất lượng, không thay kết quả i7 chưa đo. Tổng timer có overlap và file payload không là SSD physical bytes.
