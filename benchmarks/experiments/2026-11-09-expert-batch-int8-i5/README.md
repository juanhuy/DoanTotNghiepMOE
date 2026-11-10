# [P2 — BỔ TRỢ] Expert INT8 nhiều token: compute có lợi, TTFT chưa ổn định

Ngày09/11/2026. **Đã có lợi ích compute trong phạm vi kiểm chứng.** Kernel dùng trọng số thật đạt median1,410× với2token và1,381× với4token; cảhai thắng81/81cặp. Trên model thật, expert_compute thấp hơn trong cả4so sánh, khoảng17,5–29,2%. Tuy nhiên TTFT chỉ tốt hơn3/4so sánh, không đạt quy tắc≥1,05× cho mọi cặp/prompt.

Giữ tính năng CPU/INT8 ở dạng opt-in, **mặc địnhoff**. Không ghi toàn chat nhanh38–41%, không suy cải thiện decode hoặc kết quả i7. Hồ sơ này chứa hai giai đoạn riêng, một mục lục; không phải42benchmark độc lập.

## Thuật toán và điều kiện dùng

Code production trước thay đổi có dispatch_width=1, gọi forward_reuse từng token và chia hàng qua Rayon. Nhánh dispatch nhiều token song song cũ chỉ cfg(test); đã hiệu chỉnh baseline đúng trước timing, không dùng candidate cũ làm đối chứng.

Kernel mới lấy một khối8trọng số INT8, chuyển sang FP32 trong register và dùng cho2/4input. Lặp đủ4bank để giữ thứ tự cộng như dot SIMD hiện có; không mở rộng ma trận trọng số sang FP32. Gate/Up/SiLU/Down đều dùng đường batch, đổi layout activation/output có tính vào timing.

Prefill vẫn nhóm token theo expert như trước. Với opt-in, xử lý nhóm4 rồi2 rồi1; nhóm1 giữ đường cũ. Ghi contributions về đúng token/Top-Kslot và giảm tổng theo thứ tự cũ. Cache/admission/predictor/precision/sampling không đổi; decode không dùng kernel mới.

Bật bằng MOE_EXPERT_BATCH_PREFILL=on cùng MOE_GPU_MODE=cpu và checkpoint expert INT8. Off là mặc định; on với GPUauto hoặc INT4 bị từ chối trước khi dùng kernel. Giữ cấu hình RAM/threads/guard theo [WORKFLOWS](../../../docs/WORKFLOWS.md); không tự tăng cache khi bật.

API status/metrics công bố expert_batch_prefill_enabled; prefill_expert_batch2_calls/batch4_calls đếm số lần gọi kernel expert, không số prompt token duy nhất. Chọn Gate–Up separate/paired vẫn áp dụng đường từng token; đường batch dùng phép dot chia sẻ trọng số riêng.

Scratch local cho batch2+batch4 =48×(hidden+intermediate)byte,147456B/144KiB ở model thật. Thêm vào reserve request và prefill_capacity_bytes; giải phóng trước mốc after_prefill. Không tăng cache budget hoặc suy RSS giảm từ capacity.

## 1. Kernel resident — đạt tiêu chí compute

[Protocol expert-batch-int8-v1](../../SCENARIOS.md) chốt trước timing: layers0/7/15, expert0 cố định, hidden2048/intermediate1024;3payload thật18,046875MiB/budget256MiB. Activations synthetic xorshift64 seeds17/73/991, không hidden states hội thoại.

Baseline production forward_reuse từng token tuần tự, chia hàng qua cùng Rayon4; Gate–Up paired. Candidate batch2/batch4 chia hàng4worker. Cả full expert, layout copies và finite checks được tính; buffers cấp trước, không file I/O trong vùng compute. Giảm dispatch và chia sẻ trọng số cùng góp vào thay đổi, chưa phân lập phần lợi ích riêng từng yếu tố.

Mỗi layer/seed/scope10warmup/variant,9cặpAB/BA luân phiên,40iteration/sample:18paritycase/324sample,81cặp/scope. Primary CẢhai scope phải medianratio≥1,10 và thắng≥7/9 ở mọi layer/seed; đã đạt. Tỷ số dưới đây là baseline/candidate.

| Layer | Seed | Batch2 ratio | Thắng | Batch4 ratio | Thắng |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 17 | 1.405379× | 9/9 | 1.411664× | 9/9 |
| 0 | 73 | 1.430360× | 9/9 | 1.395971× | 9/9 |
| 0 | 991 | 1.394820× | 9/9 | 1.359495× | 9/9 |
| 7 | 17 | 1.395914× | 9/9 | 1.370162× | 9/9 |
| 7 | 73 | 1.425728× | 9/9 | 1.389838× | 9/9 |
| 7 | 991 | 1.397940× | 9/9 | 1.399003× | 9/9 |
| 15 | 17 | 1.409988× | 9/9 | 1.383416× | 9/9 |
| 15 | 73 | 1.403624× | 9/9 | 1.386352× | 9/9 |
| 15 | 991 | 1.410535× | 9/9 | 1.366109× | 9/9 |

18paritycase maxabs0 trong tolerance2e-5; không chứng minh bit-exact mọi input. Scratch/input capacity của cảhai variant tối đa180224B/176KiB, dưới256KiB. VmHWM/VmRSS cuối26980KiB, VmSwap0; read_bytes0→0. Process6thread gồm4Rayon workers. Testbinary wall3,620s gồm setup/harness, không tổng kernel time.

[Preflight kernel](kernel-preflight.json) có swap hệ thống tồn tại nhưng pswpout không tăng qua3mẫu5s, available≥10641052KiB; không engine/build khác. Không pin affinity/tần số/counter hardware; một process ngắn và input/lớp chọn cố định, không ước lượng tốc độ mọi workload.

[Kernel record](kernel-run/result.json), [từng sample](kernel-run/kernel.json), [checks](kernel-checks.log). Prototype khi đó chỉ cfg(test); snapshot/binary khớp ngay sau đo trước tích hợp. Nguồn kernel khác nguồn model phía dưới, không đổi hash hồi tố.

## 2. Model thật — compute thấp hơn, không đạt quy tắc TTFT

[Protocol expert-batch-model-v1](../../SCENARIOS.md): đúng một ABBA off→on→on→off,4process mới, mỗi process hai phiên độc lập Mars/Python, max1/cachedprompt0; reset đầu block rồi keepcache, không warmup riêng.

CPU4/victim/cache64MiB-layer/context512, cùng binary/source/checkpoint, per-token Gate–Up paired, FP32KV/reuseoff/lookaheadoff/memorytraceon/greedy/temp0/top-p1/top-k0/repetition1/hotpin0. Không usage side state/disk session. Hashcheckpoint làm ấm OS cache; không SSD-cold đồng nhất. Guard5120/headroom1024/watchdogRSS5632/available1536/processswap128MiB giữ nguyên.

8/8request completed, mỗi request sinh1token. Text/usage/finish và5counter cache/file khớp mọi block. Candidate thực sự gọi batch2/batch4: Mars287/366calls, Python281/462calls; baseline0. Trace7mốc/status/mode/INT8/CPU/FP32 đúng, GPUdisabled/VRAM0/maps không CUDA/cuBLAS.

| Block | Mode | Prompt | Prefill s | TTFT s | Expert compute ms | Expert I/O s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | off | Mars | 2.957 | 2.964 | 370.849 | 1.915 |
| 0 | off | Python | 3.162 | 3.167 | 438.091 | 1.935 |
| 1 | on | Mars | 2.658 | 2.665 | 293.814 | 1.695 |
| 1 | on | Python | 2.082 | 2.087 | 310.235 | 0.933 |
| 2 | on | Mars | 2.745 | 2.752 | 318.589 | 1.678 |
| 2 | on | Python | 2.847 | 2.855 | 352.373 | 1.562 |
| 3 | off | Mars | 2.933 | 2.963 | 409.206 | 1.837 |
| 3 | off | Python | 2.610 | 2.616 | 427.155 | 1.400 |

Cặp1=block0off/1on; cặp2=block3off/2on. Tỷ sốoff/on>1 có lợi cho on:

| Cặp | Prompt | TTFT ratio | Expert compute ratio | Thời gian compute giảm |
| --- | --- | ---: | ---: | ---: |
| 1 | Mars | 1.112214× | 1.262189× | 20.77% |
| 1 | Python | 1.517745× | 1.412124× | 29.18% |
| 2 | Mars | 1.076490× | 1.284432× | 22.14% |
| 2 | Python | 0.916257× | 1.212222× | 17.51% |

Primary phải cảTTFT/prefill ratio≥1,05 cho từng prompt/cặp. Cặp2Python cóTTFT0,916257× nên **không đạt**; không loại mẫu hoặc rerun/tuning. Expert compute ở cặp này giảm427,155→352,373ms nhưng expert I/O tăng1400,372→1561,513ms, TTFT tăng2616,349→2855,475ms. Đây là mô tả timer, không chứng minh nguồn chậm là SSD thiết bị.

Tất cả lượt Mars620fileloads/3910860800B; Python560/3532390400B. Payload không byte SSDphysical. Timers compute/I/O có thể chồng lấn, không cộng thành total. max1 không kiểm throughput decode dài hoặc chất lượng chuẩn hóa; HTTP không cung cấp logits/tokenIDs model thật.

## Bộ nhớ và pressure model

| Block | Sampled peak RSS MiB | Max VmSwap MiB | Min available MiB | Δpswpout pages | Δpswpin pages | Δprocess read_bytes GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2880.160 | 0.000 | 7946.109 | 52133 | 4 | 2.230 |
| 1 | 2874.621 | 0.000 | 7928.883 | 1227 | 1 | 1.006 |
| 2 | 2877.164 | 0.000 | 7569.066 | 0 | 376 | 1.360 |
| 3 | 2878.504 | 0.000 | 7256.609 | 0 | 2 | 1.397 |

Page4096B. [Preflight model](model-preflight.json): swap hệ thống đã tồn tại, pswpout ổn định/available≥10581960KiB trước tải; không engine/build khác. Trong run cóswap-out hệ thống/read I/O khác nhau. Processswap0 trong sample, khôngwatchdog/OOM trong cửa sổ đo; không chứng minh plateau/no-leak/hard-cap tổng RAM hoặc RSS candidate thấp hơn tổng quát.

Trace xác nhận prefill_capacity_bytes on−off đúng147456B ở cùng prompt; after_prefill=0 cảhai mode. Phần local được drop không đồng nghĩa allocator trả ngay trang về OS.

## Kiểm tra và provenance

- Kernel: [checks](kernel-checks.log)130Rust/profile (111lib+6HTTP+7loader+6reference),8ignored/profile,84Python,7Node; fmt/Clippy và targeted signed/tails/full-expert/invalid-input PASS.
- Model: [checks](validation/model-checks.log)133Rust/profile (114lib+6HTTP+7loader+6reference),8ignored/profile,87Python,7Node; fmt/Clippy PASS. Logits MHA/GQA tolerance2e-5 ở độ dài1/2/3/4/5/7/9, KV positions, token/session/finish/prefix/fileload parity, accounting và cancel/retry smoke PASS. Không sửa expected/tolerance.
- [Build server](validation/model-build.log) PASS; HTTP release fixture2request/mode với [off](validation/fixture-off.json)/[on](validation/fixture-on.json), cache1/context64/max3/2threads: mode/calls/parity/trace7mốc PASS. Đây là kiểm chức năng, không hiệu năng7B.
- Sau khi kết thúc timing, chạy MOE_GPU_MODE=cpu MOE_EXPERT_BATCH_PREFILL=on python3 tools/stress_http.py tests/fixtures/olmoe/unnormalized-mha /tmp/moe-batch-http-stress.json 30: [raw stress](validation/http-stress.json), 20completed/10cancelled, steady RSS spread72KiB; PASS. Selector do lệnh khởi chạy quy định, raw stress không tự lưu environment. Đây là fixture, không tải7B.
- CUDAcompute/converterPython không chạy ở bước này. Không suy GPU unit test pass thành CUDA đã kiểm. Không build/tests/inference khác đồng thời timing.

Kernel run20261109T031604.028889Z-expert-batch-34c0f3c2e88a4fc795363fb774d2025a: HEAD5663c7ab7c8f905edb2a218a3f524fda342b6179/dirty, 53sourcehash; binarye3fa5a62a5358f36bde441b06428285281c4058996cc597453f88a40b8bb1934.

Model run20261109T032523.246467Z-expert-batch-model-dddc6d8edb9a41dba3aa0d42bcfacc4d: HEAD5663c7ab7c8f905edb2a218a3f524fda342b6179/dirty, cùng56sourcehash ở4child; binaryd12a3939a7405d483a665e8642fa8c3404176e31decb0dad6eece7cd7b90c3fc. Source ZIP/currentsource/binary đã kiểm ngay sau timing, trước cập nhật tài liệu. Checkpoint/config9f1312f8… đầy đủ trong childmanifest, trùng checkpoint INT8. Hai giai đoạn là hai source snapshot khác nhau.

Hardware model: i5-10500H/12logicalCPU/RAM16571342848B; inventory thấy GTX1650Max-Q4096MiB/driver615.78.08 nhưng chỉ CPU compute. Kernel inventory GPU unavailable trong raw riêng. Không khái quát cho i7.

[CHECKSUMS.sha256](CHECKSUMS.sha256) khóa42payload (7kernel +34model/validation +1HTTPstress), giữ nguyên số gốc/đường dẫn lúc chạy. README là diễn giải/P2. Mapping child record model:

- Block0/off: [result](model-run/block-0-off/i5/20261109T032523.253842Z-ram-595d2cfe60c9482d8fd23f82e5f1558f/result.json); source.zip/memory.jsonl/server.log/turn-*.json cùng thư mục.
- Block1/on: [result](model-run/block-1-on/i5/20261109T032548.301752Z-ram-3c6fbc9eedbb4e27a0146f4ed69004c3/result.json); source.zip/memory.jsonl/server.log/turn-*.json cùng thư mục.
- Block2/on: [result](model-run/block-2-on/i5/20261109T032612.060120Z-ram-b52523a450f446f083737d3badee3e49/result.json); source.zip/memory.jsonl/server.log/turn-*.json cùng thư mục.
- Block3/off: [result](model-run/block-3-off/i5/20261109T032635.266196Z-ram-0b4fe125114440cc965a1a244800e892/result.json); source.zip/memory.jsonl/server.log/turn-*.json cùng thư mục.
