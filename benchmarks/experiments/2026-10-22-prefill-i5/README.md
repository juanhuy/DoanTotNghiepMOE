# Thí nghiệm prefill/session trên i5 — 2026-10-22

Đây là thí nghiệm local có kiểm soát cấu hình, chưa phải baseline nghiệm thu hai máy. Máy i5 được người dùng xác nhận; runner quan sát Intel Core i5-10500H, GTX 1650 Max-Q. Checkpoint/hash/hardware/build flags nằm trong từng raw runner record. OS page cache, tần số CPU, nhiệt và tải desktop chưa kiểm soát. Không chạy build/tests/inference khác đồng thời với phép đo.

## Dispatch token theo nhóm: không bật mặc định

Bốn tiến trình theo thứ tự before → after → after → before, cùng checkpoint, prompt `Where is Mars ?`, max_tokens16, cache128MiB/layer, context512, threads4, GPUauto. Mỗi tiến trình có một lượt engine-cold và ba lượt warm. Hash checkpoint và token IDs khớp ở cả 16 mẫu. Engine-cold không chứng minh OS/SSD-cold. Snapshot Git sạch là bản sao local của cây làm việc đang sửa; không phải revision phát hành. Source từng variant và hash được giữ trong `before/`, `after/`, `experiment.json`, diff tại `optimization.patch`.

Các mẫu theo thứ tự chạy, trước khi tổng hợp:

| Raw | Lượt | TTFT (s) | Decode steps/s | Wall (s) |
| --- | ---: | ---: | ---: | ---: |
| [prefill-before-0](i5/20261022T072146.481346Z-prefill-before-0-f2ea55cf8d4d46cb8561d6bd33bad38c/result.json) | 0 (engine-cold) | 5.811 | 2.337 | 12.228 |
| [prefill-before-0](i5/20261022T072146.481346Z-prefill-before-0-f2ea55cf8d4d46cb8561d6bd33bad38c/result.json) | 1 (engine-warm) | 4.162 | 2.571 | 9.998 |
| [prefill-before-0](i5/20261022T072146.481346Z-prefill-before-0-f2ea55cf8d4d46cb8561d6bd33bad38c/result.json) | 2 (engine-warm) | 4.025 | 2.581 | 9.838 |
| [prefill-before-0](i5/20261022T072146.481346Z-prefill-before-0-f2ea55cf8d4d46cb8561d6bd33bad38c/result.json) | 3 (engine-warm) | 4.027 | 2.557 | 9.894 |
| [prefill-after-1](i5/20261022T072253.909169Z-prefill-after-1-da0f2f0e602e42468661c9c8eb8fe432/result.json) | 0 (engine-cold) | 5.214 | 2.516 | 11.175 |
| [prefill-after-1](i5/20261022T072253.909169Z-prefill-after-1-da0f2f0e602e42468661c9c8eb8fe432/result.json) | 1 (engine-warm) | 3.130 | 3.087 | 7.989 |
| [prefill-after-1](i5/20261022T072253.909169Z-prefill-after-1-da0f2f0e602e42468661c9c8eb8fe432/result.json) | 2 (engine-warm) | 3.304 | 3.189 | 8.008 |
| [prefill-after-1](i5/20261022T072253.909169Z-prefill-after-1-da0f2f0e602e42468661c9c8eb8fe432/result.json) | 3 (engine-warm) | 3.045 | 3.393 | 7.466 |
| [prefill-after-2](i5/20261022T072351.419505Z-prefill-after-2-a64487993ec244c5b6af9be5a2ee08b2/result.json) | 0 (engine-cold) | 3.941 | 2.914 | 9.088 |
| [prefill-after-2](i5/20261022T072351.419505Z-prefill-after-2-a64487993ec244c5b6af9be5a2ee08b2/result.json) | 1 (engine-warm) | 3.693 | 3.254 | 8.302 |
| [prefill-after-2](i5/20261022T072351.419505Z-prefill-after-2-a64487993ec244c5b6af9be5a2ee08b2/result.json) | 2 (engine-warm) | 3.092 | 3.461 | 7.425 |
| [prefill-after-2](i5/20261022T072351.419505Z-prefill-after-2-a64487993ec244c5b6af9be5a2ee08b2/result.json) | 3 (engine-warm) | 3.137 | 3.574 | 7.334 |
| [prefill-before-3](i5/20261022T072445.372227Z-prefill-before-3-cda9087ed27945b3bcef1ad94d2db912/result.json) | 0 (engine-cold) | 4.891 | 2.985 | 9.916 |
| [prefill-before-3](i5/20261022T072445.372227Z-prefill-before-3-cda9087ed27945b3bcef1ad94d2db912/result.json) | 1 (engine-warm) | 3.262 | 3.568 | 7.466 |
| [prefill-before-3](i5/20261022T072445.372227Z-prefill-before-3-cda9087ed27945b3bcef1ad94d2db912/result.json) | 2 (engine-warm) | 3.062 | 3.761 | 7.050 |
| [prefill-before-3](i5/20261022T072445.372227Z-prefill-before-3-cda9087ed27945b3bcef1ad94d2db912/result.json) | 3 (engine-warm) | 2.931 | 3.812 | 6.866 |

Có drift lớn: baseline cuối nhanh hơn baseline đầu. Không dùng trung vị gộp để khẳng định tăng tốc. Đo tiếp trong cùng một process/binary, một model, max_tokens1, warmup4, 9 cặp xen kẽ thứ tự. [Raw paired dispatch](paired-prefill.json), [log](paired-prefill.log), [source](paired-source.rs), [provenance](paired-provenance.json).

| Cặp | Serial prefill (ms) | Grouped prefill (ms) |
| --- | ---: | ---: |
| 0 | 2571.892 | 2132.050 |
| 1 | 2184.676 | 2143.930 |
| 2 | 1988.396 | 2176.767 |
| 3 | 2004.289 | 2151.255 |
| 4 | 1945.666 | 2108.076 |
| 5 | 2150.463 | 2248.372 |
| 6 | 2046.680 | 2129.426 |
| 7 | 2050.890 | 2121.548 |
| 8 | 1985.965 | 2181.052 |

Grouped chậm hơn 7/9 cặp; trung vị serial 2046,680 ms, grouped 2143,930 ms. Trung vị thay đổi theo từng cặp +4,55%. Candidate không được bật mặc định; chỉ giữ trong cfg(test) để tái lập. Không tuyên bố cải thiện decode từ thí nghiệm này.

## Prefix chung sau khi sửa câu hỏi: được giữ

Bản cũ bỏ KV nếu toàn bộ token prefix đã lưu không khớp; bản mới lấy longest common prefix, truncate KV tới phần còn đúng và luôn tính lại ít nhất token cuối prompt để lấy logits. Cấp KV mới sau lookup, tránh cấp một bộ buffer rồi bỏ khi đã có cache dùng lại. Chưa đo số allocation hoặc RSS giảm do thay đổi này; R1 capacity accounting vẫn mở.

Đo trong cùng model/binary, serial expert dispatch, cache128MiB/layer, context512, threads4/GPUauto. Cố định seed KV bên ngoài timing; thay câu hỏi cuối trên phần ngữ cảnh chung. Mỗi biến thể warmup1, sau đó 5 cặp xen kẽ. Target116 tokens; bản cũ reuse0, bản mới reuse104, còn12 token prefill. Max_tokens1 để cô lập prefill. Đây là workload sửa lịch sử, không phải bản tái hiện hai lượt Mars/Love do người dùng báo.

[Raw session](paired-session.json), [log](paired-session.log), [source tại thời điểm đo](session-source.rs), [provenance](session-provenance.json), [summary dẫn xuất](summary.json). Strict reference chỉ được biên dịch trong tests; không phải công tắc runtime. Token đầu sinh ra khớp ở mọi mẫu.

| Cặp | Cũ: prefill (ms) | Mới: prefill (ms) |
| --- | ---: | ---: |
| 1 | 8827.777 | 2922.145 |
| 2 | 9352.213 | 3006.887 |
| 3 | 9349.372 | 2870.175 |
| 4 | 8846.971 | 2703.790 |
| 5 | 8434.258 | 2629.063 |

Trung vị prefill: **8846,971 → 2870,175 ms (giảm 67,56%)**; TTFT: **8847,387 → 2870,468 ms**. Đây là lợi ích tránh tính lại prefix trong trường hợp được đo, không phải tăng tốc kernel hoặc cam kết cho chat thường/128 output tokens. Không đo decode với max_tokens1. Expert cache tiếp tục tiến hóa giữa samples; cache OS/tải nền chưa cố định.

## Kiểm chứng và tái lập

Hồi quy trước sửa prefix FAIL (reuse0 thay vì3); sau sửa đạt trên MHA/GQA. Tests bao phủ sửa/lặp/rút ngắn/khác hoàn toàn prompt, buffer ownership và logits sau truncate ở tolerance2e-5. `python3 tools/check.py --rust --release --http`: 64 Rust tests/profile pass, 5 ignored/profile; 24 tooling tests, fmt/Clippy và catalog40 đạt; HTTP20 complete/10 cancel, steady RSS spread172KiB trên fixture. [Validation log](validation.log). Hai ignored experiment đã được chạy riêng ở release. Converter và kiểm thử chất lượng chuẩn hóa chưa chạy. GPU được quan sát enabled/dense offloaded trong raw model runs; không có profiler xác nhận từng kernel.

Tái lập phép đo riêng; hoàn tất build/tests và đóng mọi inference khác trước khi chạy:

```bash
RAYON_NUM_THREADS=4 MOE_PREFILL_TESTBED=i5 \
MOE_PREFILL_MODEL=models/olmoe-1b-7b-int8 \
MOE_PREFILL_REPORT=/tmp/session-prefix-unique.json \
cargo test --release --locked --lib benchmark_session_prefix_paired -- --ignored --nocapture --test-threads=1
```

Dùng tên output mới cho mỗi phép đo (create_new, không ghi đè). Đổi test filter sang `benchmark_prefill_dispatch_paired` để đo lại candidate bị loại. `checksums.json` khóa byte của payload/source/log, không tự chứng minh provenance phần cứng hoặc chất lượng model. Corpus lịch sử40 artifact và manifest không thay đổi.
