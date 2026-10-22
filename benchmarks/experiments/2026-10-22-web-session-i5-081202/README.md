# Thí nghiệm HTTP/SSE Mars → Love trên i5 — 2026-10-22

**Phạm vi:** exploratory, working tree dirty, một máy và hai prompt. Đây là phép đo mới đúng chuỗi request web, không phải số liệu hồi cứu hoặc benchmark sửa lịch sử/max_tokens1. Không suy ra trần phần cứng hoặc mức tăng tốc phổ quát.

## Workload và nguồn

`POST /api/cache/reset` → session/history mới → SSE `Where is Mars` max_tokens128 → đưa nguyên text assistant vào history → SSE `What is love ?` max_tokens128 trong cùng session. Mỗi lần đổi policy đồng thời reset entries/heat/KV; giữ model/dense/pool trong cùng process. Không clear OS cache. Không chạy build/tests/inference khác đồng thời khi timing.

Người dùng xác nhận i5; hardware quan sát i5-10500H, GTX1650 Max-Q. Cấu hình đo tường minh: cache128MiB/layer, context512, threads8, GPUauto. Server cũ đã dừng trước khi đo; **không xác minh được cache thật của lượt web cũ** từ nhãn UI128MB/L. Vì vậy chỉ so các mẫu đối chứng tại đây, không dùng số chat người dùng gửi làm baseline causal.

Checkpoint hash nằm trong từng result.json; giống nhau ở mọi run. Runner hash checkpoint trước khi đo nên ảnh hưởng OS page cache. GPU start/end enabled và dense_offloaded; actual CUDA fixture được chạy riêng, không có profiler xác nhận từng kernel 7B. Binary/source hashes, source snapshots, build/server logs được lưu cùng raw. Các timer là đường xử lý, có thể chồng lấn; không cộng thành wall.

## Các lượt process riêng trước phép đo chính

Những lượt này cho thấy drift và không được gộp với AB/BA cùng process để tạo phần trăm đẹp hơn. Baseline đầu chưa có field policy; source ở đó chính là hành vi inclusive.

| Raw | Vòng | Prompt | Wall (s) | Decode steps/s |
| --- | ---: | --- | ---: | ---: |
| [web-before-229376390f414f8dad8](i5/20261022T081641.076048Z-web-before-229376390f414f8dad800bb24b0e9290/result.json) | 0 | Mars | 62.752 | 2.246 |
| [web-before-229376390f414f8dad8](i5/20261022T081641.076048Z-web-before-229376390f414f8dad800bb24b0e9290/result.json) | 0 | Love | 65.587 | 2.387 |
| [web-before-229376390f414f8dad8](i5/20261022T081641.076048Z-web-before-229376390f414f8dad800bb24b0e9290/result.json) | 1 | Mars | 56.496 | 2.508 |
| [web-before-229376390f414f8dad8](i5/20261022T081641.076048Z-web-before-229376390f414f8dad800bb24b0e9290/result.json) | 1 | Love | 54.306 | 2.973 |
| [web-victim-a817c94a46384dd289d](i5/20261022T082613.396790Z-web-victim-a817c94a46384dd289dddca47117b703/result.json) | 0 | Mars | 41.870 | 3.428 |
| [web-victim-a817c94a46384dd289d](i5/20261022T082613.396790Z-web-victim-a817c94a46384dd289dddca47117b703/result.json) | 0 | Love | 49.395 | 3.312 |
| [web-victim-a817c94a46384dd289d](i5/20261022T082613.396790Z-web-victim-a817c94a46384dd289dddca47117b703/result.json) | 1 | Mars | 40.828 | 3.519 |
| [web-victim-a817c94a46384dd289d](i5/20261022T082613.396790Z-web-victim-a817c94a46384dd289dddca47117b703/result.json) | 1 | Love | 50.579 | 3.193 |
| [web-reference-recheck-34a18b82](i5/20261022T083106.771312Z-web-reference-recheck-34a18b824aa64966b94b59c106468fab/result.json) | 0 | Mars | 43.956 | 3.265 |
| [web-reference-recheck-34a18b82](i5/20261022T083106.771312Z-web-reference-recheck-34a18b824aa64966b94b59c106468fab/result.json) | 0 | Love | 46.411 | 3.408 |
| [web-victim-recheck-9355a779f0f](i5/20261022T083300.630574Z-web-victim-recheck-9355a779f0f04930a6990e04abab3883/result.json) | 0 | Mars | 31.557 | 4.505 |
| [web-victim-recheck-9355a779f0f](i5/20261022T083300.630574Z-web-victim-recheck-9355a779f0f04930a6990e04abab3883/result.json) | 0 | Love | 47.715 | 3.443 |

## Phép đo chính: cùng server/model, AB/BA

Raw: [result.json](i5/20261022T092416.839618Z-web-paired-abba-443258fd6bdd4e2c84bde18575709b5e/result.json). Thứ tự cycles0–3: **inclusive → victim → victim → inclusive**. Hai mẫu/policy/prompt; mỗi output đúng128 tokens. Text UTF-8 đầy đủ và usage khớp giữa mọi mẫu của từng prompt (không có token IDs trên API HTTP, không gọi đây là bit-exact logits7B). Mars19 prompt tokens/cached0; Love166 prompt tokens/cached11.

Từng mẫu theo thứ tự trước khi tổng hợp:

| Cycle | Policy | Prompt | TTFT (s) | Decode steps/s | Wall (s) | Legacy I/O (s) | Process read delta (GiB) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | inclusive | Mars | 6.911 | 2.254 | 63.253 | 42.695 | 28.983 |
| 0 | inclusive | Love | 12.311 | 2.338 | 66.629 | 41.165 | 26.507 |
| 1 | victim | Mars | 5.467 | 3.203 | 45.111 | 29.419 | 17.971 |
| 1 | victim | Love | 10.423 | 2.851 | 54.970 | 31.064 | 17.925 |
| 2 | victim | Mars | 5.498 | 2.736 | 51.914 | 33.482 | 19.300 |
| 2 | victim | Love | 11.844 | 2.661 | 59.565 | 33.548 | 19.223 |
| 3 | inclusive | Mars | 5.947 | 2.384 | 59.214 | 39.064 | 26.870 |
| 3 | inclusive | Love | 14.429 | 2.479 | 65.664 | 37.643 | 23.863 |

Trung vị mô tả (n=2/policy/prompt, không có khoảng tin cậy):

| Prompt | Wall inclusive → victim (s) | Thay đổi wall | TTFT inclusive → victim (s) | Decode inclusive → victim (steps/s) |
| --- | --- | ---: | --- | --- |
| Mars | 61.234 → 48.512 | -20.77% | 6.429 → 5.483 | 2.319 → 2.970 |
| Love | 66.147 → 57.268 | -13.42% | 13.370 → 11.133 | 2.408 → 2.756 |

Cả hai cặp theo thứ tự AB và BA có wall victim thấp hơn cho cả hai prompt. Tuy vậy tải desktop, nhiệt/tần số và OS page cache chưa cố định; biến động giữa các samples còn lớn. Kết luận là **policy tốt hơn trong các mẫu/config đã đo**, không phải ước lượng cố định cho mọi lượt web hoặc mức tối ưu tối đa.

## Giải thích bằng lượng công việc và timers

| Prompt/policy | File payload (GiB) | File loads | H2D payload (GiB) | D2H payload (GiB) | Prefill I/O (s), median | Decode I/O (s), median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mars/inclusive | 32.786 | 5581 | 52.272 | 13.371 | 4.439 | 36.440 |
| Mars/victim | 24.867 | 4233 | 41.528 | 25.772 | 4.183 | 27.268 |
| Love/inclusive | 31.999 | 5447 | 52.208 | 21.989 | 4.925 | 34.479 |
| Love/victim | 24.973 | 4251 | 50.545 | 33.967 | 3.847 | 28.459 |

Victim dùng VRAM chứa expert bị evict khỏi RAM, xóa bản sao L1 khi admit lại RAM. File loads/payload giảm; **D2H tăng** vì thêm hit VRAM, không khẳng định mọi truyền dữ liệu đều giảm. File payload đếm load/prefetch thành công kể cả speculative chưa dùng, không phải byte SSD. Process read delta lấy /proc/<pid>/io:read_bytes, vẫn là attribution OS không phải đo bandwidth NVMe riêng. Upload timer gồm staging và upload, không nằm trong legacy expert_io_ms. Timer load/prefetch gồm scheduling và có thể overlap compute.

Love vẫn chỉ reuse11/166 tokens, không phải146 token của toàn history trước. Điều này giải thích vì sao phải prefill lại nhiều; chưa kết luận nguyên nhân chính xác tại tokenizer/template và không tự đổi whitespace/input để tạo speedup. Không đo GPU/NVMe/PCIe saturation; chưa thể gọi đây là trần hardware. R1/R4 accounting và R7 telemetry physical VRAM còn mở.

## Tính đúng và cách dùng

Fixture CUDA thật: logits so CPU tolerance2e-5, tiers không trùng sau admission, reset trả đủ pool slots và cache RAM0 vẫn có hit L1. CPU phase/file-payload invariants đạt. Hồi quy HTTP đổi policy valid/invalid/reset mặc định đạt; SSE parser bắt lỗi/truncated/nonfinite. Debug/release68 Rust tests/profile pass,5 ignored/profile; tooling30 tests, fmt/Clippy/catalog40 pass. HTTP fixture30 vòng (20 hoàn tất/10 hủy), steady RSS spread164KiB. Logs trong validation/. Converter/MSRV/Windows/i7 và quality benchmark chuẩn hóa chưa chạy.

Mặc định runtime mới là victim; giữ inclusive để đối chứng. Thay đổi default/validation parser sau đo không đổi đường compute hay policy được ép tường minh trong phép đo chính. API cũ reset không query vẫn giữ policy. Không thêm unsafe; remove_with_pool là nội bộ, chỉ gọi sau synchronous transfer và dưới ownership độc quyền cache/pool, trước khi slot được tái sử dụng.

```bash
# Chạy web với policy mới (cũng là mặc định khi không đặt biến này):
MOE_EXPERT_CACHE_POLICY=victim cargo run --release --locked --bin moe-tier-engine
# Đối chứng policy cũ, giữ các biến cache/threads khác giống nhau:
MOE_EXPERT_CACHE_POLICY=inclusive cargo run --release --locked --bin moe-tier-engine

# Preview phép đo chính; bỏ dry-run sau khi hoàn tất checks và không còn inference khác:
python3 tools/run_benchmark.py --http-session --compare-cache-policies \
  --testbed i5 --gpu auto --threads 8 --cache 128 --max-tokens 128 \
  --runs 2 --allow-dirty --tag web-abba --dry-run
```

Các run dirty là exploratory; không đưa thành baseline luận văn đã nghiệm thu. Không so trực tiếp với workload max_tokens16 hoặc sửa history/max_tokens1. Giữ nguyên raw lịch sử và mọi raw experiment; checksums.json kiểm tính toàn vẹn byte. Source default và API/metrics được ghi tại final-source/ cùng diff; report này giữ cả thử nghiệm không kết luận được và phép đo đối chứng chính.
