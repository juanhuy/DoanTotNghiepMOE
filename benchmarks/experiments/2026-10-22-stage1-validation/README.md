# Đóng giai đoạn 1: kiểm chứng bản vá, chưa nghiệm thu baseline

Ngày 2026-10-22. **Exploratory i5**, revision gốc trong raw nhưng working tree dirty. Đây là smoke chức năng/quality, không phải A/B hiệu năng hay bằng chứng baseline sạch. Không sửa corpus40 raw lịch sử.

- [Record hoàn tất](model-smoke/result.json), [runner thực thi](model-smoke/smoke_runner.py), [server log](model-smoke/server.log), [source snapshot](model-smoke/source_snapshot.zip).
- [Record harness thất bại](failed-harness/result.json) giữ nguyên: Mars/Love hoàn tất, sau đó Python AttributeError do alias module `http`, trước khi thử cancellation; không quy lỗi này cho engine. Lượt sau dùng run ID và server mới, không gộp mẫu để suy ra speedup.
- Hash checkpoint/source/binary, hardware quan sát, config, từng input/output/session/process counters trong record. Source snapshot gồm Rust/Python/JS/HTML/Cargo, không chứa model weights. Kiểm payload bất biến bằng `sha256sum -c CHECKSUMS.sha256` trong thư mục này.

## Điều kiện và kiểm tra

Người dùng đã xác nhận i5; host probe GTX1650 Max-Q khả dụng. GPUauto enabled, victim policy, Rayon8, cache128MiB/layer, context512, dense budget3GiB. Một server riêng, không build/tests/benchmark đồng thời. Hash checkpoint trước đo làm ấm OS cache. Source runtime giữ nguyên trong smoke; không đổi expected/tolerance.

Mars→Love cùng history/session, budget32 mỗi lượt. Hủy TCP stream sau content đầu tiên của request128-token, reset retry429 đến khi model nhả lock, rồi request16-token hoàn tất. Reset trước prompt context dài, budget16. Quality suite12 case dùng budget gốc từng case, session riêng; cache expert tiếp tục ấm qua suite. Endpoint `/chat_stream.js` có serve asset dùng thật.

16 request hoàn tất, thêm1 request cố ý hủy; reset/retry sau hủy đạt. Protocol thành công không đồng nghĩa nội dung đúng. Quality heuristic **10/12**, lỗi `capital-vietnamese` (trả Hồ Chí Minh) và `retrieve-day`; xem nguyên văn trong từng record. Exact check sau trim/substring case-insensitive, không benchmark chất lượng chuẩn hóa và không so FP32/INT8 trên model7B.

## Từng request hoàn tất

Timing chỉ mô tả lượt smoke với budget khác nhau. Không gộp thành tốc độ đại diện, không so trực tiếp với các lượt128 output tokens trước đó.

| Case | Prompt tokens | Completion tokens | Wall s | TTFT s | Decode steps/s |
| --- | --- | --- | --- | --- | --- |
| mars | 19 | 32 | 17.958 | 5.162 | 2.423 |
| love | 71 | 32 | 24.923 | 9.031 | 1.951 |
| after-abort-reset | 19 | 16 | 12.518 | 5.816 | 2.238 |
| long-context | 212 | 16 | 30.547 | 24.597 | 2.521 |
| quality-math-add | 25 | 16 | 11.543 | 5.376 | 2.432 |
| quality-math-multiply | 27 | 14 | 9.864 | 5.070 | 2.712 |
| quality-capital-france | 26 | 24 | 16.241 | 5.118 | 2.068 |
| quality-capital-vietnamese | 54 | 24 | 17.868 | 7.832 | 2.292 |
| quality-follow-vietnamese | 38 | 7 | 8.008 | 6.144 | 3.219 |
| quality-follow-exact-ok | 21 | 2 | 4.029 | 3.708 | 3.109 |
| quality-python-add | 36 | 64 | 26.036 | 6.072 | 3.156 |
| quality-python-even | 38 | 64 | 23.943 | 5.842 | 3.481 |
| quality-remember-name | 50 | 8 | 11.205 | 8.466 | 2.555 |
| quality-retrieve-day | 44 | 32 | 17.711 | 6.486 | 2.762 |
| quality-translate-hello | 27 | 24 | 10.896 | 3.556 | 3.134 |
| quality-opposite-hot | 28 | 12 | 6.823 | 3.376 | 3.191 |

## Bộ nhớ và giới hạn

Sample VmRSS sau request khoảng3,78–4,58GiB (không phải peak toàn hệ thống hay A/B RAM); từng sample còn giữ VmHWM/VmSwap và process I/O. Cache+recycled ở status cuối trong budget mỗi layer. SwapFree hệ thống từ4.194.300 xuống3.488.264KiB, quan sát trước/sau process; có hoạt động swap và tải desktop không kiểm soát, không quy toàn bộ swap cho engine hoặc gọi bộ nhớ đã ổn định. Payload/cache capacity không đồng nghĩa RSS/SSD bytes.

Targeted host CUDA tests kiểm pinned bounds/zero-init, undersized fallback, victim disjoint tiers/pool reset và logits fixture tolerance2e-5. Fixture HTTP30 vòng20 complete/10 cancel; Node parser7 tests. Các test này được ghi ở PROGRESS, khác phép smoke model thật ở đây. Không kiểm browser trực tiếp vì môi trường thiếu browser/Playwright; không tự tải thêm dependency.

Giai đoạn1 còn: baseline cold/warm/conversation >=5 mẫu/policy trên revision sạch, kiểm browser trực tiếp, review giới hạn chất lượng và swap. i7 chưa đo mới. Các sửa R1/R2/R4/R5 không tự chứng minh tăng tốc; trần prefetch mới có thể đổi hiệu năng. Xem [PLAN](../../../docs/PLAN.md) và [PROGRESS](../../../docs/PROGRESS.md).
