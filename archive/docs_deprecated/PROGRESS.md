# Nhật ký nâng cấp MoE

## Quy tắc ghi kết quả

Theo yêu cầu người dùng: mỗi bước triển khai phải ghi lại kết quả trong file này trước khi báo hoàn thành. Ghi cả thử nghiệm thất bại và giới hạn còn tồn tại. Các mục cũ được giữ làm lịch sử; sửa kết luận bằng mục đính chính mới nếu cần.

- Nêu mục tiêu, thay đổi thực tế và file liên quan.
- Ghi lệnh kiểm tra, kết quả pass/fail, số test và lỗi còn lại.
- Với hiệu năng: ghi checkpoint, prompt, số token, cấu hình, trạng thái cache, cách đo, số lần chạy và số đo trước/sau.
- Không có số đo thì ghi **chưa đo**; không suy diễn tốc độ tăng từ thay đổi code.
- Phân biệt thời gian nạp model, prefill, generation và tổng thời gian.
- Kết quả từ lịch sử hội thoại phải đánh dấu là hồi cứu, không coi là kiểm thử vừa chạy.
- Khi có công cụ benchmark, lưu kết quả thô vào `benchmarks/results/` và liên kết từ nhật ký; không lưu nội dung nhạy cảm vào Git.

## Lộ trình

| Bước | Nội dung | Trạng thái | Kết quả |
| --- | --- | --- | --- |
| 1 | Thống kê hiệu năng và benchmark tái lập | Hoàn thành bản đầu mở rộng | Có suite v1 5 tình huống × 3 vòng, cấu hình/metrics và peak RSS; giữ các baseline cũ |
| 2 | Tối ưu CPU, bộ đệm, đa luồng và SIMD | Đang triển khai | Đã tối ưu INT8 và dense FP32 bằng 8 bộ cộng, tái sử dụng buffer; còn prefill batch, đa luồng/SIMD chuyên biệt |
| 3 | SSE streaming và hủy yêu cầu | Hoàn thành bản đầu cho OLMoE | Có validation trước SSE, decoder Unicode, queue giới hạn và hủy giữa layer/expert; HTTP smoke đạt |
| 4 | Cache tổng, tải trước và tối ưu I/O | Đang triển khai | Có cache 128 MiB/layer, heat/aging, giữ shard handle và benchmark nhiều prompt; còn cache tổng/prefetch |
| 5 | Prefill theo batch token | Hoàn thành bản đầu | Layer-major causal attention, gom route theo expert và LM head một lần; logits/hồi quy dài đạt |
| 6 | Expert INT4 và đánh giá chất lượng | Chưa triển khai | Decoder thật hiện dùng INT8 |
| 7 | Cấu hình, hồi quy và tài liệu vận hành | Đang triển khai | Có PLAN, SSE/UI local, giới hạn request và hồi quy HTTP; còn đóng gói và tải dài |

## 2026-09-22 — Mốc ban đầu (hồi cứu)

Nguồn: kết quả công cụ đã ghi trong phiên làm việc trước khi tạo nhật ký; không chạy lại trong bước ghi chép này.

### Trạng thái chức năng

- Engine Rust đã chạy checkpoint OLMoE INT8 trên CPU, có tokenizer, chat template, cache expert và API.
- Đã tách khỏi dự án bên cạnh: converter riêng `tools/prepare_olmoe.py`, expert `src/int8_expert.rs`, checkpoint riêng `models/olmoe-1b-7b-int8`.
- Đã kiểm tra bản sao checkpoint không có symlink và không chung inode với bản nguồn.
- 23 test Rust đạt; Clippy với `-D warnings` đạt; test converter Python đạt.
- Logits fixture nhỏ được đối chiếu Transformers 4.51.3, ngưỡng sai số tuyệt đối `2e-5`.

### Lệnh đã chạy và kết quả

```bash
cargo test --manifest-path MoE/Cargo.toml --quiet
cargo clippy --manifest-path MoE/Cargo.toml --all-targets -- -D warnings
/tmp/moe-reference-env/bin/python MoE/tools/test_prepare_olmoe.py
cargo run --release --manifest-path MoE/Cargo.toml --example chat_olmoe -- \
  MoE/models/olmoe-1b-7b-int8 'What is 2 + 2?' 12
```

Lần chạy checkpoint độc lập:

| Chỉ số | Kết quả |
| --- | --- |
| Nạp model | 4,94 giây |
| Prefill + sinh output | 34,70 giây |
| Prompt tokens | 19 |
| Completion tokens (gồm EOS) | 7 |
| Output | `2 + 2 equals 4.` |
| Finish reason | `stop` |
| Context limit | 512 |
| Cache expert mỗi layer | 32 MiB |
| Dense budget | 3 GiB |
| Số lần chạy cho số đo này | 1 |
| Cache hệ điều hành | Không kiểm soát |
| Prefill riêng / TTFT / decode tokens mỗi giây | Chưa đo |

Một lần HTTP trước khi tách checkpoint ghi nhận peak RSS `2407344 kB` (khoảng 2,30 GiB), không tính page cache toàn hệ thống. Đây là phép đo khác, không ghép thành số đo RAM của lần CLI trên. HTTP đã trả 200 cho chat, 400 cho input không hợp lệ và 429 cho yêu cầu chồng nhau; health vẫn phản hồi trong lúc sinh token.

Cấu hình phần cứng do người dùng cung cấp: Intel i7-14650HX, RAM 16 GB, RTX 5060 8 GB. Engine thử nghiệm chạy CPU, không sử dụng GPU. Số luồng và tải nền chưa được ghi cố định.

**Giới hạn:** các số trên là smoke test, chưa phải baseline benchmark; không dùng để khẳng định mức tăng tốc. Chưa đo chất lượng/ngữ cảnh dài của checkpoint 7B.

## 2026-09-22 — Thiết lập nhật ký

- Mục tiêu: lưu kết quả của từng bước trong lộ trình nâng cấp.
- Thay đổi: tạo `docs/PROGRESS.md`, thêm liên kết README và hướng dẫn ghi chép trong `AGENTS.md`.
- Kiểm tra: đọc lại file và xác nhận liên kết tương đối.
- Không thay đổi runtime; không chạy lại test hoặc benchmark cho thay đổi tài liệu.
- Bước tiếp theo: triển khai thống kê hiệu năng và benchmark chuẩn ở bước 1.

## 2026-09-22 — Bước 1: đo hiệu năng và baseline

- Mục tiêu: tách thời gian của đường sinh OLMoE và tạo benchmark JSON có thể chạy lại.
- Thay đổi:
  - `src/generation.rs`: thêm `GenerationMetrics` vào kết quả sinh.
  - `src/olmoe.rs`: đo tokenizer, prefill, TTFT, decode, attention, expert compute, LM head, expert I/O; đếm cache hit/miss/eviction, byte đọc, cache expert và KV cache.
  - `src/main.rs`: trả trường `metrics` qua HTTP.
  - `examples/benchmark_olmoe.rs`: chạy nhiều lượt trên cùng model và xuất JSON.
  - `tests/olmoe_reference.rs`: kiểm tra tính nhất quán cơ bản của metrics.
  - Kết quả thô: [`../benchmarks/results/2026-09-22-baseline.json`](../benchmarks/results/2026-09-22-baseline.json).

Lệnh kiểm thử:

```bash
cargo fmt --all
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo run --release --example benchmark_olmoe -- \
  models/olmoe-1b-7b-int8 'What is 2 + 2?' 16 2
```

Kết quả kiểm thử: **đạt**; 23 test logic/integration đạt, các target example biên dịch, Clippy sạch.

Điều kiện baseline: checkpoint OLMoE INT8 độc lập; prompt 19 token sau chat template; output 7 token gồm EOS; cache expert 32 MiB/layer; context 512; dense budget 3 GiB; 24 logical CPU được hệ điều hành báo; 2 lượt trong cùng tiến trình. Page cache hệ điều hành không được xóa hoặc đo, tải nền và tần số CPU chưa được cố định.

| Chỉ số | Lượt 1, engine-cold | Lượt 2, engine-warm |
| --- | ---: | ---: |
| Load model | 1.578 giây | dùng model đã nạp |
| Prefill | 22.521 giây | 21.853 giây |
| Time to first token | 22.522 giây | 21.853 giây |
| Decode | 7.336 giây | 7.102 giây |
| Decode model steps/s | 0.818 | 0.845 |
| Tổng generate | 29.858 giây | 28.955 giây |
| Attention | 5.706 giây | 5.456 giây |
| Expert compute | 10.294 giây | 9.964 giây |
| LM head | 2.014 giây | 2.005 giây |
| Expert I/O | 11.788 giây | 11.476 giây |
| Cache hit / miss | 242 / 2958 | 250 / 2950 |
| Eviction | 2878 | 2950 |
| Byte expert đọc | 18.659 GB | 18.608 GB |
| Cache expert cuối request | 504.6 MB | 504.6 MB |
| KV cache cuối request | 6.55 MB | 6.55 MB |

Nhận định dựa trên baseline: cache 32 MiB/layer bị thrash nặng; lượt warm gần như không tăng hit rate và vẫn đọc khoảng 18,6 GB. Ba chi phí lớn là expert I/O, expert compute và attention. Đây là dữ liệu định hướng cho bước 2/4, chưa chứng minh một tối ưu cụ thể sẽ tăng tốc bao nhiêu.

Giới hạn phép đo:

- `time_to_first_token_ms` gồm tokenizer + prefill + chọn token đầu; chưa gồm render chat template trong `engine.rs`.
- `decode_tokens_per_second` đếm các model step sau token đầu, không tính token đầu hoặc EOS như một model step độc lập.
- Các timer thành phần có thể cộng lớn hơn wall time nếu sau này có song song; hiện đường chạy tuần tự.
- `expert_bytes_read` là payload được reader yêu cầu, không chứng minh từng byte đến từ thiết bị SSD vì hệ điều hành có page cache.
- Chưa đo RSS tự động, năng lượng, CPU utilization hoặc prompt dài.

Kết luận: bước 1 bản đầu hoàn thành. Bước tiếp theo là tối ưu CPU an toàn trước (bỏ xác thực trọng số lặp lại và giảm cấp phát), sau đó chạy lại cùng benchmark; cache/I/O chuyên sâu vẫn giữ ở bước 4.

## 2026-09-22 — Bước 2a: bỏ xác thực dense weights lặp lại

- Mục tiêu: loại bỏ một lượt quét toàn bộ trọng số dense trước mỗi matrix-vector multiply, vì loader OLMoE đã kiểm tra shape và giá trị hữu hạn một lần khi nạp.
- Thay đổi:
  - `backend.rs`: giữ `Matrix::multiply` có kiểm tra đầy đủ cho API/test chung; thêm đường nội bộ `multiply_loaded`, chỉ kiểm tra input và output.
  - `olmoe.rs`: dùng đường đã xác thực cho Q/K/V/O, router và LM head.
  - Thêm unit test xác nhận kết quả hai đường giống nhau và đường nhanh vẫn từ chối input sai chiều/NaN.
- Kết quả thô: [`../benchmarks/results/2026-09-22-skip-dense-revalidation.json`](../benchmarks/results/2026-09-22-skip-dense-revalidation.json).

Lệnh kiểm thử và benchmark giống bước 1. Kết quả: **đạt**, 24 test Rust đạt, Clippy sạch, output/token IDs không đổi.

| Chỉ số | Baseline cold | Sau tối ưu cold | Thay đổi |
| --- | ---: | ---: | ---: |
| Tổng generate | 29.858 s | 25.160 s | -15,7% |
| TTFT | 22.522 s | 19.208 s | -14,7% |
| Decode | 7.336 s | 5.952 s | -18,9% |
| Decode step/s | 0.818 | 1.008 | +23,3% |
| Attention | 5.706 s | 3.184 s | -44,2% |
| LM head | 2.014 s | 1.179 s | -41,4% |

| Chỉ số | Baseline warm | Sau tối ưu warm | Thay đổi |
| --- | ---: | ---: | ---: |
| Tổng generate | 28.955 s | 21.805 s | -24,7% |
| TTFT | 21.853 s | 16.198 s | -25,9% |
| Decode step/s | 0.845 | 1.070 | +26,7% |

Giới hạn: mỗi phiên bản mới chỉ được đo một cặp cold/warm; page cache, tải nền và CPU frequency không cố định nên phần trăm chỉ là chỉ báo ban đầu. Byte expert và cache hit/miss gần như không đổi, phù hợp vì tối ưu này chỉ tác động dense compute. Bước 2 vẫn đang triển khai; tiếp theo là giảm allocation/bộ đệm rồi đo lại.

## 2026-09-22 — Bước 4a: giảm cache thrashing

- Mục tiêu: cache phải chứa ít nhất tập expert hoạt động của một token và giảm tải lại expert liên tục.
- Nguyên nhân xác nhận: checkpoint có expert resident khoảng 6,3 MB và Top-K=8; cache cũ 32 MiB/layer chỉ giữ khoảng 5 expert, nhỏ hơn active set khoảng 50,5 MB.
- Thay đổi:
  - Tăng mặc định từ 32 lên 128 MiB/layer, tối đa khoảng 2 GiB cho 16 layer.
  - Loader tính kích thước active set và phát cảnh báo nếu cấu hình cache nhỏ hơn mức này.
  - Benchmark nhận tham số `CACHE_MIB_PER_LAYER` để so sánh chính sách cache.
  - Cập nhật CLI example và README theo mặc định mới.
- Kết quả thô: [`../benchmarks/results/2026-09-22-cache-128mib.json`](../benchmarks/results/2026-09-22-cache-128mib.json).

Lệnh:

```bash
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo run --release --example benchmark_olmoe -- \
  models/olmoe-1b-7b-int8 'What is 2 + 2?' 16 2 128
```

Kết quả: **đạt**, 24 test đạt, Clippy sạch, output và token IDs không đổi.

| Chỉ số cold | Cache 32 MiB/layer | Cache 128 MiB/layer | Thay đổi |
| --- | ---: | ---: | ---: |
| Cache hit | 242 | 1.951 | +706% |
| Cache miss | 2.958 | 1.249 | -57,8% |
| Eviction | 2.878 | 913 | -68,3% |
| Payload expert yêu cầu | 18,66 GB | 7,88 GB | -57,8% |
| Expert I/O | 11,37 s | 5,41 s | -52,4% |
| Tổng generate | 25,16 s | 18,18 s | -27,7% |
| TTFT | 19,21 s | 13,90 s | -27,6% |
| Decode step/s | 1,008 | 1,400 | +38,9% |
| Cache resident cuối request | 504,6 MB | 2.119 MB | +1.615 MB |

Lượt warm 128 MiB đạt 2.060 hit / 1.140 miss và đọc 7,19 GB; cache vẫn còn churn nhưng không còn nhỏ hơn active set. Tổng warm là 18,59 giây so với 21,80 giây ở 32 MiB (-14,7%).

Đánh đổi: cache resident tăng khoảng 1,6 GB. Mức 128 MiB/layer được chọn phù hợp hơn với máy RAM 16 GB hiện tại; chưa tăng lên 256–400 MiB/layer vì có thể tạo thêm 4–6,5 GB cache và gây swap/OOM khi hệ thống đang chỉ còn khoảng 5,5 GiB available, swap đã đầy trong lần kiểm tra.

Giới hạn: `expert_bytes_read` vẫn là byte payload qua reader, có thể đến từ page cache; benchmark chỉ một prompt ngắn và một cặp cold/warm. Bước 4 chưa hoàn thành: còn cache tổng toàn model, tránh tải trùng đồng thời, giữ file handle/prefetch và chính sách ưu tiên expert nóng.

## 2026-09-22 — Bước 2b/4b: expert nóng và tái sử dụng buffer SwiGLU

- Mục tiêu: giữ expert thường dùng xuyên nhiều request và bỏ cấp phát 4 vector cho mỗi lần chạy expert.
- Thay đổi cache:
  - Mỗi layer giữ heat counter cho toàn bộ expert, tăng trên cả hit và miss.
  - Khi đầy, chọn victim có heat thấp nhất; truy cập cũ nhất dùng để phá hòa.
  - Expert lạnh hơn victim không được nhận vào cache, tránh scan hiếm làm mất expert nóng.
  - Heat giảm một nửa sau mỗi 65.536 lượt truy cập/layer để thích nghi khi workload đổi.
- Thay đổi CPU:
  - `ExpertScratch` cấp phát gate, up, activation và output một lần/request.
  - `Int8Expert::forward_reuse` ghi trực tiếp vào buffer thay vì tạo bốn `Vec` cho mỗi expert.
  - API `forward` cũ vẫn giữ để tương thích và dùng cùng kernel.
- Test bổ sung kiểm tra warm request không có nhiều miss hơn cold request trên fixture. Logits và token output không đổi.
- Kết quả thô: [`../benchmarks/results/2026-09-22-hot-cache-reused-buffers.json`](../benchmarks/results/2026-09-22-hot-cache-reused-buffers.json).

Lệnh kiểm thử: `cargo test --all-targets`, `cargo clippy --all-targets -- -D warnings`; kết quả **đạt**, 24 test đạt, Clippy sạch.

So sánh lượt warm ổn định hơn (lượt 3 chính sách mới) với lượt warm LRU 128 MiB trước đó:

| Chỉ số warm | LRU | Heat-aware lượt 3 | Thay đổi |
| --- | ---: | ---: | ---: |
| Cache hit | 2.060 | 2.553 | +23,9% |
| Cache miss | 1.140 | 647 | -43,2% |
| Eviction | 1.140 | 174 | -84,7% |
| Payload expert yêu cầu | 7,19 GB | 4,08 GB | -43,2% |
| Expert I/O | 5,55 s | 2,86 s | -48,4% |
| TTFT | 13,89 s | 12,55 s | -9,7% |
| Tổng generate | 18,59 s | 16,80 s | -9,7% |

Trong chính sách mới, miss giảm dần 1.070 → 682 → 647 qua ba lượt và output vẫn là `2 + 2 equals 4.`. Cold total là 18,68 giây, gần mức 18,18 giây của LRU; chính sách mới chủ yếu có lợi sau khi học workload.

Giới hạn và diễn giải:

- Benchmark gộp hai thay đổi nên không tách riêng phần tăng tốc của buffer reuse. `expert_compute_ms` không giảm trong phép đo đơn này; chưa có bằng chứng wall-time rằng tái sử dụng buffer đã tăng tốc, dù số lần cấp phát trên đường expert đã giảm theo cấu trúc code.
- Chính sách heat có thể ưu tiên workload cũ; aging giảm rủi ro nhưng chưa benchmark prompt thay đổi hoặc nhiều người dùng.
- Cache lookup/victim hiện quét tuyến tính tối đa khoảng 21 entry/layer, chi phí nhỏ ở cấu hình hiện tại nhưng cần index/heap nếu cache lớn hơn.
- Bước tiếp theo: benchmark bộ prompt đa dạng để kiểm tra thích nghi, sau đó giữ file handle và giảm allocation ở attention/dense matvec.

## 2026-09-22 — Bước 4c: hoàn thiện kiểm chứng shard handle và workload nhiều prompt

- Mục tiêu: kiểm chứng phần giữ file shard mở và đo cache khi chuyển chủ đề theo bước tiếp theo của nhật ký.
- Trạng thái lúc bắt đầu: `TensorIndex` đã giữ `Vec<Mutex<File>>`, `examples/benchmark_workload.rs` đã có trong workspace nhưng chưa có kết quả benchmark trong nhật ký. Đây là phần tiếp tục kiểm chứng mã có sẵn, không phải toàn bộ được viết mới trong bước này.
- Thay đổi:
  - `src/safetensors.rs`: chuyển cấp phát payload ra trước mutex; giữ cặp seek/read trong cùng vùng khóa để các lượt đọc không làm sai vị trí file của nhau.
  - `tests/int8_loader.rs`: thêm kiểm thử 8 luồng, mỗi luồng đọc 100 lần xen kẽ hai tensor khác offset/kích thước trên cùng shard; kiểm tra giới hạn byte vẫn bị từ chối và lượt đọc hợp lệ tiếp theo trả đúng dữ liệu.
  - README: ghi rõ vòng đời file handle và giới hạn đọc tuần tự trong cùng shard.
  - Cập nhật bảng lộ trình để phản ánh buffer reuse và heat cache đã có ở bước 2b/4b.

Lệnh kiểm thử và benchmark (chạy mới trong bước này):

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo run --release --example benchmark_workload -- \
  models/olmoe-1b-7b-int8 8 2 128 \
  > benchmarks/results/2026-09-22-mixed-workload-open-shards.json
```

Kết quả: **25 test đạt** (18 unit + 5 loader + 2 reference/integration), Clippy sạch; benchmark hoàn tất 6 request. Đã đọc JSON và kiểm tra token IDs giống nhau giữa hai vòng cho cả ba prompt. Không ghi nhận lỗi kiểm thử/build.

Điều kiện: checkpoint `models/olmoe-1b-7b-int8`, release CPU, context 512, dense budget 3 GiB, cache 128 MiB/layer; ba prompt cố định theo thứ tự toán → Python → tiếng Việt, lặp hai vòng trong cùng tiến trình, tối đa 8 output token/request. Prompt lần lượt có 19/23/48 token; output có 7/8/8 token. Load model 1,337 giây. Cache expert cuối mỗi request là 2.119.434.240 byte. Chỉ request đầu có engine cache rỗng; các request còn lại kế thừa cache và heat history.

Kết quả thô: [`../benchmarks/results/2026-09-22-mixed-workload-open-shards.json`](../benchmarks/results/2026-09-22-mixed-workload-open-shards.json).

| Prompt | Tổng generate vòng 1 | Tổng generate vòng 2 | Cache miss vòng 1 → 2 |
| --- | ---: | ---: | ---: |
| `What is 2 + 2?` | 19,824 s | 17,550 s | 1.070 → 900 |
| `Write a short Python function that adds two integers.` | 21,006 s | 20,401 s | 1.300 → 1.350 |
| `Hãy trả lời ngắn gọn: thủ đô của Việt Nam là gì?` | 35,860 s | 34,954 s | 2.280 → 1.797 |
| Tổng | 76,690 s | 72,905 s | 4.650 → 4.047 |

- Tổng hit: 9.430 → 10.033; eviction: 1.305 → 311; payload expert yêu cầu: 29,331 → 25,528 GB; expert I/O: 14,173 → 8,306 s.
- Cache miss tổng giảm khoảng 13,0%, nhưng prompt Python tăng 50 miss. Kết quả này không chứng minh heat cache cải thiện mọi prompt; workload trước đó ảnh hưởng việc giữ expert.
- Câu toán trả `2 + 2 equals 4.` và dừng EOS. Python và tiếng Việt dừng vì giới hạn 8 token, nên câu trả lời chưa hoàn chỉnh; không dùng phép đo này để kết luận chất lượng model.
- Chưa đo trước/sau riêng cho giữ shard handle hoặc chuyển allocation ngoài mutex; không quy mức thay đổi thời gian giữa hai vòng cho các thay đổi này. Không có baseline LRU với cùng workload để so sánh chính sách.
- Page cache, tải nền và tần số CPU không được kiểm soát; test/Clippy chạy trong lúc benchmark ở phần đầu có thể ảnh hưởng thời gian. Không đo peak RSS, không diễn giải payload reader thành lượng đọc vật lý từ SSD.
- Một handle được giữ cho mỗi shard, dùng tài nguyên file descriptor cho đến khi model được giải phóng. Mutex bảo vệ cùng shard nhưng chưa có prefetch hoặc I/O song song trong đường inference. Test đồng thời là kiểm chứng reader, không phải benchmark throughput.

Kết luận: hoàn thành bước kiểm chứng workload nhiều prompt và phần giữ shard handle. Bước 2/4 vẫn đang triển khai; tiếp theo giảm allocation attention/dense hoặc đo đối chứng cache trên cùng workload trước khi đổi chính sách.

## 2026-09-23 — Bước 2c: kernel INT8 với các bộ cộng độc lập

- Mục tiêu: đo riêng kernel expert rồi tối ưu phần tính toán chiếm thời gian lớn nhất.
- `src/int8_expert.rs`: thay dot product cộng tuần tự bằng 8 bộ cộng FP32 độc lập, cộng gộp rồi xử lý phần dư và nhân row scale. Mã Rust an toàn, portable; không thêm dependency, thread hoặc yêu cầu AVX. Tạo điều kiện compiler vector hóa nhưng chưa kiểm tra assembly nên không khẳng định tập lệnh SIMD cụ thể.
- Giữ kernel scalar cũ trong module test làm đối chứng. Thêm kiểm tra với FP64 ở chiều 1/2/7/8/9/15/16/17/1024/2048/2051, gồm INT8 âm/dương, row scale khác nhau và phần dư không chia hết cho 8. Ngưỡng sai số `2e-5 * (1 + abs(reference))`; không yêu cầu bitwise equality vì thứ tự cộng đổi.
- Thêm microbenchmark ignored trong module test, lưu JSON và hướng dẫn trong README: buffer tổng hợp cố định, một luồng, không I/O/cấp phát trong vùng đo; warmup 10 lượt mỗi kernel; 9 mẫu, 30 matvec/mẫu, luân phiên thứ tự scalar/mới. Dùng release target mặc định, không `target-cpu=native`.

Lệnh chạy từ `MoE`:

```bash
# Trước thay đổi kernel:
cargo run --release --example benchmark_olmoe -- \
  models/olmoe-1b-7b-int8 'What is 2 + 2?' 16 2 128 \
  > benchmarks/results/2026-09-23-int8-before.json
# Sau thay đổi:
cargo fmt --all
cargo test --all-targets --quiet
MOE_KERNEL_REPORT=benchmarks/results/2026-09-23-int8-kernel.json \
cargo test --release --lib int8_expert::kernel_tests::benchmark_int8_kernel -- --ignored --exact
cargo test --release --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo run --release --example benchmark_olmoe -- \
  models/olmoe-1b-7b-int8 'What is 2 + 2?' 16 2 128 \
  > benchmarks/results/2026-09-23-int8-after.json
```

Kiểm thử: **26 test đạt** ở debug và release (19 unit, 5 loader, 2 integration/reference), 1 microbenchmark ignored trong suite thường và đã chạy riêng đạt. Logits fixture MHA/GQA vẫn nằm trong ngưỡng tuyệt đối `2e-5` so với Transformers. Clippy cuối cùng sạch.

Lỗi và cách xử lý:

- Sandbox không khởi tạo được (`mountinfo path is not absolute`), kể cả `pwd`; thực thi qua quyền ngoài sandbox sau khi được duyệt.
- Clippy ban đầu từ chối assertion hằng dùng để ngăn benchmark chạy debug. Thêm allow cục bộ có chú thích vì test ignored vẫn phải biên dịch trong debug; không tắt lint cho kernel hoặc toàn dự án. Chạy lại Clippy đạt.

Microbenchmark mới trên Intel i7-14650HX, median của 9 mẫu:

| Projection (rows × cols) | Scalar | 8 bộ cộng | Tỷ lệ scalar/mới | Sai số tuyệt đối tối đa so với scalar |
| --- | ---: | ---: | ---: | ---: |
| 1024 × 2048 | 1,009 ms | 0,278 ms | 3,63× | 4,77e-7 |
| 2048 × 1024 | 1,027 ms | 0,270 ms | 3,81× | 9,54e-7 |

Benchmark checkpoint thật mới: OLMoE INT8 trong workspace, CPU release, prompt `What is 2 + 2?` (19 token), max output 16, thực tế 7 token gồm EOS; context 512, dense budget 3 GiB, cache 128 MiB/layer. Mỗi phiên bản chạy 2 request liên tiếp trong một tiến trình; trước/sau ở hai tiến trình khác nhau. Không chạy test/build khác đồng thời với phần đo model. Load trước/sau 1,415/1,352 giây, tách khỏi generate.

| Chỉ số | Trước cold | Sau cold | Trước warm | Sau warm |
| --- | ---: | ---: | ---: | ---: |
| Generate | 18,143 s | 11,261 s | 16,446 s | 9,701 s |
| TTFT | 13,912 s | 8,541 s | 12,360 s | 7,201 s |
| Decode | 4,230 s | 2,719 s | 4,086 s | 2,500 s |
| Expert compute | 9,324 s | 2,648 s | 9,201 s | 2,642 s |
| Expert I/O | 4,418 s | 4,046 s | 2,857 s | 2,499 s |

- Generate giảm 37,9% cold và 41,0% warm trong cặp đo này; expert compute giảm khoảng 71%.
- Đã so sánh JSON: token IDs trước/sau giống nhau ở cả hai lượt; output `2 + 2 equals 4.`. Cache hit/miss giữ nguyên: cold 2.130/1.070, warm 2.518/682; payload cold 6.749.388.800 byte, warm 4.301.946.880 byte.
- Dữ liệu mới: [kernel](../benchmarks/results/2026-09-23-int8-kernel.json), [trước](../benchmarks/results/2026-09-23-int8-before.json), [sau](../benchmarks/results/2026-09-23-int8-after.json). Các mốc trước đó được giữ nguyên.

Giới hạn: kernel benchmark dùng dữ liệu tổng hợp resident; chưa đo đa luồng, SIMD intrinsics hoặc nhiều kiến trúc. Model benchmark chỉ một prompt/cặp cold-warm mỗi phiên bản; CPU frequency, page cache và tải nền không kiểm soát nên phần I/O thay đổi không quy cho kernel. Chưa đo chất lượng dài hoặc kiểm tra logits toàn checkpoint thật; token giống nhau ở prompt này không đảm bảo mọi prompt đều giống. Không tuyên bố tốc độ toàn model tăng 3,6–3,8 lần.

Kết luận: hoàn thành tối ưu kernel bản đầu và kiểm chứng trước/sau. Bước 2 vẫn đang triển khai; tiếp theo giảm allocation attention/dense và mở rộng workload trước khi quyết định thêm SIMD chuyên biệt hoặc đa luồng.

## 2026-09-23 — Bước 3a: SSE streaming và hủy khi client ngắt

- Tiếp tục kế hoạch hoàn thiện: chốt phạm vi tại `docs/PLAN.md`, thêm `benchmarks/suite-v1.json` và `examples/benchmark_suite.rs` cho baseline 5 tình huống. Kết quả suite được ghi riêng sau khi đo xong.
- Trạng thái mã lúc bắt đầu đã có `StepScratch` và các phép dense/attention ghi vào buffer. Giữ nguyên phần này; không quy thay đổi buffer có sẵn cho bước streaming.
- `src/olmoe.rs`: chung đường tính toán giữa generate thường và streaming; callback báo validation xong, decoder tăng dần từ tokenizers, flush phần đuôi khi EOS/limit; kiểm tra hủy trước prefill, giữa layer/expert và mỗi bước decode.
- `src/engine.rs`: `chat_stream` giữ semaphore đến khi worker thoát; queue giới hạn 8 event, handshake trả lỗi input/context trước khi mở HTTP stream; receiver bị drop báo worker hủy và đánh thức blocking_send. Theo dõi JoinHandle để lỗi panic không kết thúc stream âm thầm.
- `src/main.rs`: `stream:true` cho OLMoE trả SSE role/content/final usage+metrics và `[DONE]`, keepalive; lỗi sau headers thành JSON error rồi DONE. `stream:false` giữ JSON cũ. Demo chưa hỗ trợ streaming.
- Dependency: thêm trực tiếp `futures-core` cho trait Stream và dev dependency `tower` util cho test Router (đã có transitively trong lockfile).
- Kiểm thử: ghép nội dung stream so với thường và token IDs trên fixture; validation trước stream; hủy trong prefill, hủy khi emit trong decode, nhiều chu kỳ drop receiver → chạy tiếp; HTTP SSE framing; lỗi đọc shard sau headers không trả finish thành công.

Lệnh:

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo build --release --bin moe-tier-engine --example benchmark_suite
python3 -m py_compile tools/test_streaming_smoke.py
python3 tools/test_streaming_smoke.py models/olmoe-1b-7b-int8 \
  benchmarks/results/2026-09-23-streaming-smoke.json
```

Kết quả mới: **31 test đạt** (21 lib, 2 HTTP binary, 5 loader, 3 integration), 1 microbenchmark ignored có chủ đích; Clippy sạch. Logits reference fixture tiếp tục đạt ngưỡng cũ. Smoke HTTP chạy server local do script tự mở/dừng, không can thiệp server khác.

Smoke checkpoint thật, CPU release, context 512, cache 128 MiB/layer, max output 32:

| Prompt | Content đầu tại client | Tổng stream | Output token / chunk content | Kết quả |
| --- | ---: | ---: | ---: | --- |
| `What is 2 + 2?` | 9,925 s | 12,826 s | 7 / 6 | `2 + 2 equals 4.` |
| `Hãy viết đúng hai từ: Việt Nam` | 11,879 s | 13,182 s | 6 / 3 | `Đức` |

- Cả hai stream khớp hoàn toàn text/usage/finish reason của request thường chạy tiếp sau đó; kết thúc `stop` và DONE. Token Unicode có thể cần gom nhiều token trước khi phát content.
- Health phản hồi trong prefill và đang nhận token; request chồng nhau trả 429.
- Sau ngắt stream trong prefill, quan sát semaphore giải phóng sau **11,071 ms** qua request probe. Một lần đo, gồm HTTP round trip/poll; không phải giới hạn hủy đảm bảo. Worker không ngắt ngay matvec/I/O đang chạy.
- **Giới hạn chất lượng xác nhận:** câu tiếng Việt trả `Đức`, sai yêu cầu. Đây không phải đạt chất lượng trả lời; stream chỉ được xác nhận khớp đường generate thường. Cần đánh giá checkpoint riêng.
- Dữ liệu mới: [streaming smoke](../benchmarks/results/2026-09-23-streaming-smoke.json). Không so tốc độ với benchmark trước vì prompt/cache/trình tự request khác.

Lỗi phát hiện và xử lý: sandbox tiếp tục lỗi bubblewrap, đã thực thi bằng quyền ngoài sandbox được duyệt. Clippy báo type_complexity cho callback, sửa bằng type alias. Benchmark JSON dùng literal 3 GiB mặc định i32 gây compile error, đổi sang u64; các lệnh cuối đạt.

Giới hạn còn lại: chưa hủy request không streaming; chưa timeout cho client giữ kết nối nhưng đọc chậm; chưa có request-ID cancel endpoint hay toàn bộ trường OpenAI; chưa thử tải nhiều giờ. TTFT metrics nội bộ khác thời gian content đầu thực nhận do giải mã/queue/mạng. Streaming không giảm thời gian tính prefill. Khi worker panic, model mutex có thể bị poison và cần khởi động lại; không tuyên bố tự phục hồi mọi lỗi.

Kết luận: SSE/hủy bản đầu đã kiểm chứng fixture và checkpoint thật. Bước tiếp theo là hoàn tất phân tích baseline suite rồi hồi quy client chậm/timeout và giao diện chat theo PLAN.

## 2026-09-23 — Baseline hoàn thiện dự án: suite v1

- Phạm vi và nghiệm thu lưu tại [PLAN.md](PLAN.md). `benchmarks/suite-v1.json` cố định 5 case với messages và max_tokens riêng; case hội thoại gửi cả lịch sử, không dùng KV xuyên request.
- `examples/benchmark_suite.rs` mặc định 3 vòng; JSON nhúng suite, config model, cấu hình context/cache/dense, phiên bản Rust, OS/arch/CPU logical, metrics mỗi request và VmHWM trên Linux. Tiến độ in stderr, stdout chỉ có JSON.
- Lệnh mới đã chạy sau khi toàn bộ test/Clippy/build và smoke HTTP kết thúc:

```bash
target/release/examples/benchmark_suite models/olmoe-1b-7b-int8 3 \
  > benchmarks/results/2026-09-23-suite-v1.json
```

- Kết quả: **15/15 request thành công**, 5 case × 3 vòng; token IDs giống nhau giữa 3 lượt cho từng case; tất cả `finish_reason=stop`, không bị cắt bởi max_tokens. Đã parse JSON, kiểm tra số lượt và so sánh token IDs.
- Điều kiện: CPU i7-14650HX, Linux x86_64, 24 logical CPU được runtime báo, Rust 1.98.1, release; checkpoint merged INT8 trong workspace; context 512, cache 128 MiB/layer, dense budget 3 GiB. Load 1,227 s, không cộng vào generate. Không có phép đo/test/build khác của phiên làm việc chạy đồng thời với suite; tải nền hệ thống, page cache và tần số CPU vẫn không kiểm soát.
- Raw data mới: [suite v1](../benchmarks/results/2026-09-23-suite-v1.json). Suite/metrics cũ được giữ nguyên, không so phần trăm tốc độ giữa hai suite khác nhau.

Median của 3 lượt theo từng case (không gọi cả vòng đầu là cold vì cache kế thừa giữa các case):

| Case | Prompt / output tokens | TTFT nội bộ | Tổng generate | Decode model steps/s |
| --- | ---: | ---: | ---: | ---: |
| math-short | 22 / 7 | 9,406 s | 11,968 s | 2,342 |
| python | 28 / 17 | 11,466 s | 17,655 s | 2,590 |
| vietnamese | 60 / 21 | 23,309 s | 30,683 s | 2,631 |
| multi-turn | 51 / 7 | 20,250 s | 23,098 s | 2,105 |
| context-retrieval | 136 / 15 | 64,841 s | 71,664 s | 2,052 |

- Peak RSS tích lũy của tiến trình: **4.010.576 KiB ≈ 3,82 GiB**. VmHWM không bao gồm toàn bộ page cache hệ thống, không phải bộ nhớ riêng của request và không phải hard cap.
- Nội dung rà soát ban đầu: toán trả `2 + 2 equals 4.`; Python trả hàm `add_integers(a, b)` với `return a + b`; tiếng Việt trả `Thủ đô của Việt Nam là Hà Nội.`; hội thoại nhớ tên Linh; tìm thông tin trả Lan/Wednesday. Đây là 5 ví dụ đơn giản, không chấm điểm chất lượng chuẩn và không phủ nhận câu tiếng Việt làm theo yêu cầu bị sai trong smoke trước.
- Case đoạn dài chỉ có 136 prompt token, chưa gần giới hạn 512; cần mở rộng đánh giá ngữ cảnh dài và nhiều kiểu câu hỏi. TTFT ở case này cho thấy thời gian xử lý prompt vẫn là vấn đề trải nghiệm, không khẳng định mọi chi phí nằm ở attention: phải xem thêm expert/I/O trong metrics trước khi chọn tối ưu.
- Chỉ request đầu toàn suite có cache expert rỗng; cache/heat kế thừa giữa các case/vòng, KV mới mỗi request. TTFT là timer nội bộ, không phải content đầu qua HTTP. Decode tốc độ đếm model steps sau token đầu theo định nghĩa metrics hiện tại.

Kết luận: hoàn thành giai đoạn baseline/phạm vi bản đầu và SSE/hủy bản đầu theo kế hoạch. Giai đoạn tiếp theo là hoàn thiện hồi quy API cho client chậm/timeout, sau đó giao diện chat; không đánh dấu toàn dự án hoặc đánh giá chất lượng đã hoàn thành.

## 2026-09-23 — Bước 3b/4: backpressure timeout, giới hạn API và giao diện chat local

- Mục tiêu: hoàn thiện phần API còn lại trong kế hoạch và tạo luồng chat sử dụng được mà không cần toolchain frontend.
- `src/engine.rs`: dùng chung validation cho stream/thường; giới hạn 128 messages và 64 KiB tổng content. Queue SSE vẫn có 8 slot; mọi lần gửi delta/final/error có timeout. Khi queue đầy quá thời gian cấu hình, worker trả `TimedOut`, thoát và giải phóng semaphore thay vì chờ vô hạn.
- `src/main.rs`: thêm `MOE_STREAM_SEND_TIMEOUT_MS`, mặc định 30.000 ms và từ chối `0`; giới hạn body HTTP 128 KiB; phục vụ trang chat tại `/`.
- `web/index.html`: HTML/CSS/JS nhúng bằng `include_str!`; giữ lịch sử messages, đọc SSE qua `fetch`, hiển thị content/usage/lỗi, nút Dừng dùng `AbortController`, hỗ trợ Enter và Shift+Enter. Không thêm npm/Node dependency runtime.
- README và PLAN cập nhật cấu hình, cách mở UI, hành vi timeout và phạm vi local.

Lệnh kiểm tra:

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo build --release --bin moe-tier-engine
MOE_MODEL_DIR=tests/fixtures/olmoe/unnormalized-mha \
MOE_BIND=127.0.0.1:18082 MOE_CONTEXT=64 \
MOE_CACHE_BYTES_PER_LAYER=10000 MOE_DENSE_BYTES=10000000 \
MOE_STREAM_SEND_TIMEOUT_MS=1000 target/release/moe-tier-engine
curl http://127.0.0.1:18082/
curl -N http://127.0.0.1:18082/v1/chat/completions ...
```

Kết quả: **32 test đạt** (22 lib, 2 HTTP binary, 5 loader, 3 integration), 1 microbenchmark ignored có chủ đích; Clippy sạch; release build đạt; JavaScript inline đạt `node --check`. Test mới xác nhận queue đầy trả `TimedOut`, body trên 128 KiB trả 413, content trên 64 KiB và trên 128 messages bị từ chối, trang `/` chứa endpoint và `AbortController`.

Kiểm tra HTTP fixture thực tế: `/` trả `200`, `content-type: text/html`; SSE trả role, 3 content delta, final usage/metrics và `[DONE]`; request content 65 KiB trả `400`. Server thử nghiệm trên cổng 18082 đã được dừng sau kiểm tra. HTML cuối cùng là 5.532 byte sau chỉnh trạng thái nút; thay đổi cuối đã qua test/Clippy/`node --check` nhưng không cần chạy lại smoke vì endpoint và parser không đổi. Đây là smoke chức năng, không phải benchmark hiệu năng mới; **chưa đo** ảnh hưởng wall time/RAM của UI hoặc timeout.

Lỗi và cách xử lý: sandbox bubblewrap vẫn lỗi `mountinfo path is not absolute`. `apply_patch` ban đầu cũng kế thừa profile lỗi; bỏ riêng biến `CODEX_PERMISSION_PROFILE` khi gọi công cụ patch ngoài sandbox đã khôi phục đúng workflow patch. Lần test đầu sau đổi AppState không biên dịch vì hai Router test còn dùng `Arc<InferenceEngine>` trực tiếp; chuyển sang `AppState`. Clippy sau đó báo test module đứng trước item runtime; chuyển module xuống cuối file. Các lệnh cuối đạt.

Giới hạn:

- Timeout chỉ đo thời gian worker chờ queue engine → HTTP adapter có chỗ, không phải deadline tổng inference hay cam kết phát hiện ngay mọi TCP client chậm; buffer mạng có thể trì hoãn backpressure.
- Chế độ không streaming vẫn không hủy compute khi client ngắt. Chưa có request ID/cancel endpoint, authentication, persistence, Markdown sanitizer/renderer hoặc upload.
- UI được kiểm tra bằng unit/HTTP smoke và kiểm tra nội dung HTML; chưa chạy browser automation đa trình duyệt. Khi dừng sau một phần output, UI giữ phần assistant đã thấy trong lịch sử.
- Giới hạn byte bảo vệ chi phí parse/tokenize sơ bộ; context token của model vẫn là giới hạn quyết định cuối.

Kết luận: hoàn thành API/backpressure và giao diện local bản đầu. Theo PLAN, bước tiếp theo là mở rộng đánh giá gần context 512 và tối ưu prefill dựa trên số đo; tải dài/timeout tổng chỉ cần ưu tiên nếu mục tiêu chuyển sang production.

## 2026-09-23 — Bước 2d/5a: kernel dense FP32 và hồi quy context dài

- Mục tiêu: giảm thời gian prefill dựa trên phân rã suite, rồi kiểm tra hoạt động gần context 512.
- Phân tích baseline suite v1 trước khi sửa: case 136 token có median attention 20,095 s, expert compute 15,620 s, expert I/O 28,431 s và LM head 7,279 s. Vì dense nằm trong Q/K/V/O/router và LM head, đây là mục tiêu CPU có thể tối ưu độc lập với cache.
- `src/backend.rs`: đường trọng số đã validate dùng dot product với 8 bộ cộng FP32 độc lập, cộng gộp rồi xử lý tail. Không thêm dependency, thread, intrinsics hoặc cờ CPU. Kernel scalar chỉ giữ trong test làm đối chứng.
- Test mới so với tích lũy FP64 ở chiều 1/2/7/8/9/15/16/17/1024/2048/2051 với ngưỡng `2e-5 * (1 + abs(reference))`. Logits fixture MHA/GQA vẫn đạt ngưỡng tuyệt đối cũ `2e-5`.
- `examples/benchmark_suite.rs`: thêm đối số tùy chọn `CASE_ID`, từ chối ID không có trong suite và ghi `selected_case` vào JSON. Việc lọc không thay đổi nội dung case.
- `benchmarks/suite-v2-long-context.json`: case retrieval cố định 469 prompt token, max output 24, tổng ngân sách 493/512.

Lệnh kiểm thử và đo:

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo test --release --all-targets --quiet
MOE_DENSE_REPORT=benchmarks/results/2026-09-23-dense-kernel.json \
cargo test --release --lib backend::tests::benchmark_dense_kernel -- --ignored --exact
target/release/examples/benchmark_suite models/olmoe-1b-7b-int8 3 \
  benchmarks/suite-v1.json context-retrieval \
  > benchmarks/results/2026-09-23-dense-{before|after}-context.json
target/release/examples/benchmark_suite models/olmoe-1b-7b-int8 1 \
  benchmarks/suite-v2-long-context.json near-context-retrieval \
  > benchmarks/results/2026-09-23-suite-v2-near-context.json
```

Kết quả kiểm thử cuối: **33 test đạt** (23 lib, 2 HTTP binary, 5 loader, 3 integration), 2 microbenchmark ignored có chủ đích; debug/release đều đạt, Clippy sạch.

Microbenchmark dense, median 7 mẫu × 10 matvec/mẫu, warmup 5, một thread, buffer tổng hợp resident:

| Ma trận rows × columns | Scalar | 8 bộ cộng | Tỷ lệ scalar/mới |
| --- | ---: | ---: | ---: |
| 2.048 × 2.048 | 1,791 ms | 0,388 ms | 4,62× |
| 8.192 × 2.048 | 6,584 ms | 2,814 ms | 2,34× |

So sánh toàn model dùng đúng case `context-retrieval`: checkpoint/config giống nhau, release CPU, 136 prompt token, output 15 token và dừng EOS, 3 lượt liên tiếp trong mỗi tiến trình. Median:

| Chỉ số | Trước | Sau | Thay đổi |
| --- | ---: | ---: | ---: |
| Tổng generate | 63,905 s | 42,714 s | -33,2% |
| Prefill | 58,162 s | 38,967 s | -33,0% |
| TTFT nội bộ | 58,162 s | 38,967 s | -33,0% |
| Attention (gồm dense Q/K/V/O) | 19,383 s | 8,826 s | -54,5% |
| LM head | 6,620 s | 2,930 s | -55,7% |
| Expert compute | 14,758 s | 14,682 s | -0,5% |
| Expert I/O | 23,331 s | 16,653 s | -28,6% |

- Token IDs và output giống nhau trước/sau ở từng lượt: `Lan tested the chat interface and fixed a display problem on Wednesday.`
- Cache hit/miss median giữ nguyên 11.850/7.350; eviction 502; payload expert 46.362.624.000 byte. Vì byte/counter không đổi và page cache không kiểm soát, không quy phần I/O giảm cho dense kernel.
- Peak RSS gần như không đổi: 4.016.444 → 4.016.504 KiB. Tải nền/tần số CPU/page cache không cố định; phần trăm là kết quả của hai bộ 3 lượt, không phải cam kết mọi workload.
- Raw data: [microbenchmark](../benchmarks/results/2026-09-23-dense-kernel.json), [trước](../benchmarks/results/2026-09-23-dense-before-context.json), [sau](../benchmarks/results/2026-09-23-dense-after-context.json).

Đánh giá context dài sau tối ưu:

- Thử nghiệm trung gian 397 prompt token + 24 ngân sách output (421/512) trả đúng `The access code is ORCHID.`, TTFT 84,557 s; lưu tại [JSON 397 token](../benchmarks/results/2026-09-23-suite-v2-long-context.json). Không gọi đây là gần giới hạn.
- Suite cuối tăng lên 469 prompt token + 24 output = **493/512**; sinh 9 token, dừng EOS và trả đúng cùng câu. Một lượt: TTFT 94,368 s, tổng 96,705 s; attention 28,580 s, expert compute 42,919 s, expert I/O 16,111 s; đọc logic 96.106.250.240 byte, KV 106.168.320 byte, peak RSS 3.972.156 KiB (~3,79 GiB). Raw data: [near-context](../benchmarks/results/2026-09-23-suite-v2-near-context.json).
- Phép đo 493/512 chỉ một lượt, prompt có filler lặp và retrieval đơn giản; xác nhận chức năng/giới hạn, không đại diện chất lượng ngữ cảnh dài hoặc phân bố prompt thật.

Lỗi/giới hạn thực hiện: hai lệnh Cargo release được khởi động cùng lúc; microbenchmark chờ file lock và chỉ chạy sau test, nên không tranh CPU trong vùng đo. Sandbox bubblewrap tiếp tục lỗi; các lệnh dùng quyền ngoài sandbox đã được duyệt và patch bỏ biến profile lỗi như bước trước.

Kết luận: dense FP32 bản đầu hoàn thành và giảm khoảng một phần ba TTFT trên case 136 token. Ở 469 token, expert compute và lượng expert đọc trở thành chi phí lớn; bước tiếp theo nên giảm cache churn/I/O trong prefill hoặc triển khai prefill batch, với suite dài này làm hồi quy.

## 2026-09-23 — Bước 5b: prefill layer-major và gom route theo expert

- Mục tiêu: giảm cache churn/I/O và phép LM head thừa trong prefill mà không đổi causal attention, routing hoặc thứ tự cộng contribution Top-K.
- `src/olmoe.rs`: thêm đường `prefill` layer-major. Embedding của toàn prompt nằm trong batch; với từng layer, attention vẫn chạy position 0→N-1 và ghi cùng KV causal như đường tokenwise. Router lưu Top-K/weight cho từng token; các route được gom theo expert để một lần tải expert phục vụ mọi token trong layer. Output expert được lưu theo `(token, slot)` rồi cộng vào residual theo đúng thứ tự slot Top-K ban đầu. LM head chỉ chạy một lần trên token prompt cuối.
- Đường `step` tokenwise vẫn dùng cho decode và cho `logits()` tham chiếu. Hủy được kiểm tra giữa token attention, expert và từng expert invocation.
- Thêm test trực tiếp cho cả fixture MHA và GQA: logits cuối của prefill layer-major so với đường tokenwise, sai số tuyệt đối tối đa phải `<2e-5`; xác nhận chiều dài KV bằng số prompt token.
- Bộ đếm cache trong prefill coi các lần dùng tiếp theo của expert đã tải trong cùng batch là hit; miss/bytes phản ánh số lần tải vật lý qua loader. Vì vậy hit/miss mới hữu ích cho I/O nhưng không so nghĩa tuyệt đối với mọi phiên bản cũ ngoài bảng trước/sau được ghi rõ.

Lệnh:

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo test --release --all-targets --quiet
cargo build --release --example benchmark_suite
target/release/examples/benchmark_suite models/olmoe-1b-7b-int8 3 \
  benchmarks/suite-v1.json context-retrieval \
  > benchmarks/results/2026-09-23-batched-prefill-context.json
target/release/examples/benchmark_suite models/olmoe-1b-7b-int8 1 \
  benchmarks/suite-v2-long-context.json near-context-retrieval \
  > benchmarks/results/2026-09-23-batched-prefill-near-context.json
```

Kiểm thử: **34 test đạt** (24 lib, 2 HTTP binary, 5 loader, 3 integration), 2 microbenchmark ignored có chủ đích; debug/release đạt, Clippy sạch. Test logits layer-major/tokenwise bổ sung ngoài hồi quy Transformers đã có.

So sánh với kernel dense đã tối ưu, cùng case 136 prompt token, output 15 token, 3 lượt/process; median:

| Chỉ số | Token-major | Layer-major | Thay đổi |
| --- | ---: | ---: | ---: |
| Tổng generate | 42,714 s | 26,023 s | -39,1% |
| Prefill | 38,967 s | 21,161 s | -45,7% |
| TTFT | 38,967 s | 21,163 s | -45,7% |
| Attention | 8,826 s | 7,256 s | -17,8% |
| Expert compute | 14,682 s | 13,009 s | -11,4% |
| LM head | 2,930 s | 0,300 s | -89,8% |
| Expert I/O | 16,653 s | 5,632 s | -66,2% |
| Cache miss | 7.350 | 1.570 | -78,6% |
| Payload expert | 46,363 GB | 9,903 GB | -78,6% |

- Token IDs/output giống nhau ở cả ba cặp lượt. KV giữ 39.321.600 byte. Eviction median 502→353. Peak RSS 4.016.504→3.748.356 KiB; thay đổi admission/cache order có thể ảnh hưởng resident cache, không diễn giải là batch luôn giảm RAM.
- Raw data: [token-major dense](../benchmarks/results/2026-09-23-dense-after-context.json), [layer-major](../benchmarks/results/2026-09-23-batched-prefill-context.json).

Hồi quy 469 prompt token + 24 output budget = 493/512, một lượt trước/sau:

| Chỉ số | Token-major | Layer-major | Thay đổi |
| --- | ---: | ---: | ---: |
| Tổng generate | 96,705 s | 72,845 s | -24,7% |
| TTFT | 94,368 s | 69,657 s | -26,2% |
| Expert I/O | 16,111 s | 5,610 s | -65,2% |
| LM head | 8,738 s | 0,177 s | -98,0% |
| Cache miss | 15.236 | 1.445 | -90,5% |
| Eviction | 4.372 | 351 | -92,0% |
| Payload expert | 96,106 GB | 9,115 GB | -90,5% |

- Output/token IDs giống nhau: `The access code is ORCHID.`, 9 token, EOS. KV giữ 125.042.688 byte. Peak RSS 3.972.156→4.126.784 KiB (~+151 MiB), phù hợp có batch/hidden/contribution tạm nhưng là một lượt không kiểm soát tải nền.
- Raw data: [token-major gần context](../benchmarks/results/2026-09-23-suite-v2-near-context.json), [layer-major gần context](../benchmarks/results/2026-09-23-batched-prefill-near-context.json).

Giới hạn:

- Attention vẫn O(N²) và dense Q/K/V/O vẫn chạy từng token; chưa dùng matrix-matrix/BLAS. Expert compute vẫn gọi matvec từng route, chỉ gom vòng đời tải expert.
- Buffer contribution có kích thước `tokens × top_k × hidden × 4`; với 469×8×2048 khoảng 29,3 MiB, ngoài batch/hidden/KV. Context/config đã giới hạn kích thước và phép nhân được checked.
- So sánh 469 token chỉ một lượt; phần trăm không phải benchmark thống kê. Page cache/tần số CPU/tải nền không kiểm soát.
- Heat/cache admission được cập nhật theo nhóm expert thay vì thứ tự token-major, nên resident cache cuối request có thể khác dù logits giống. Aging vẫn dựa trên tổng lượt route.

Kết luận: hoàn thành prefill gom expert bản đầu. Trên case 136 token, TTFT giảm thêm 45,7% sau tối ưu dense; trên 469 token giảm 26,2%. Điểm nghẽn tiếp theo là attention dài và expert compute, không còn tải lại expert theo từng token ở mức cũ.

## 2026-09-23 — Bước 6: hoàn thiện UI, stress HTTP, quality smoke và demo

- `web/index.html`: thêm lựa chọn 32/64/128/256 output token, lưu lịch sử và cấu hình trong `localStorage`, giới hạn 100 message/48 KiB, nút xóa, trạng thái điều khiển và hiển thị usage/TTFT/decode tok/s/tổng thời gian/finish reason. Inline JavaScript đã qua `node --check`; binary test xác nhận các marker UI/metrics được nhúng.
- `tools/stress_http.py`: tự quản lý release server và chạy 50 vòng fixture, cứ ba vòng có một lần hủy SSE. Kết quả: 33 hoàn tất, 17 hủy, 50/50 health/permit đạt; RSS steady 10.436–10.552 KiB, spread 116 KiB. Raw data: [HTTP stress](../benchmarks/results/2026-09-23-http-stress-fixture.json). Fixture không đại diện tải checkpoint thật.
- `benchmarks/quality-v1.json` và `examples/evaluate_quality.rs`: 12 ca greedy deterministic thuộc toán, kiến thức, tiếng Việt, instruction, code, multi-turn, retrieval, dịch và ngôn ngữ. Chấm exact hoặc contains-all không phân biệt hoa thường; từ chối ID trùng và xuất response/metrics đầy đủ.
- Hai lượt quality độc lập đều đạt **12/12**. Artifact lượt lưu có median TTFT 6,505 s, tổng 10,162 s và decode 3,350 token/s; load model 1,452 s. Raw data: [quality-v1](../benchmarks/results/2026-09-23-quality-v1.json). Đây là smoke heuristic nhỏ, không đủ để kết luận chất lượng tổng quát; một ca tiếng Việt kết thúc do giới hạn token nhưng vẫn qua điều kiện substring.
- `docs/DEMO.md`: thêm preflight, cách mở UI, luồng trình bày, lệnh tạo lại quality/stress artifact và giới hạn diễn giải.

Kiểm chứng cuối:

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo build --release --bin moe-tier-engine
cargo build --release --example evaluate_quality
node --check /tmp/moe-index-script.js
python3 tools/stress_http.py tests/fixtures/olmoe/unnormalized-mha \
  benchmarks/results/2026-09-23-http-stress-fixture.json 50
target/release/examples/evaluate_quality models/olmoe-1b-7b-int8 \
  benchmarks/quality-v1.json \
  > benchmarks/results/2026-09-23-quality-v1.json
```

Kết quả: 34 test đạt, 2 microbenchmark ignored có chủ đích; Clippy sạch; release binary/example build thành công. Stress và quality đều đạt điều kiện đã khai báo.

Kết luận: phạm vi demo local v1 đã có UI sử dụng được, kiểm tra vòng đời HTTP, smoke chất lượng và kịch bản tái lập. Công việc tiếp theo có giá trị nhất là tối ưu attention/expert compute, mở rộng đánh giá độc lập và chỉ bổ sung yêu cầu production khi có mục tiêu triển khai cụ thể.

## 2026-09-23 — Bước 7: kernel dense/expert đa lõi có giới hạn

- Mục tiêu: giảm inference CPU ở phần dense và expert compute mà không thay đổi thứ tự cộng trong từng output row hoặc token sinh.
- Thêm Rayon trực tiếp. `Matrix::multiply_loaded_into` và expert INT8 `matvec_into` chia song song theo output row khi payload từ 262.144 phần tử; phép tính nhỏ giữ đường tuần tự để tránh overhead. Mỗi row vẫn dùng đúng 8 accumulator và thứ tự gộp cũ.
- `OlmoeModel::load` khởi tạo global pool tối đa 8 worker theo mặc định; `RAYON_NUM_THREADS` cho phép ghi đè và từ chối giá trị không hợp lệ. Nếu ứng dụng nhúng đã khởi tạo global Rayon pool, engine dùng pool có sẵn.

Kiểm chứng:

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo build --release --bin moe-tier-engine
RAYON_NUM_THREADS={1,8,24} target/release/examples/benchmark_suite \
  models/olmoe-1b-7b-int8 1 benchmarks/suite-v1.json context-retrieval \
  > benchmarks/results/2026-09-23-rayon-{1,8,24}-context.json
```

Kết quả test: **34 test đạt**, 2 microbenchmark ignored có chủ đích; Clippy sạch; release build đạt. Cả ba benchmark sinh cùng 15 token và đúng output `Lan tested the chat interface and fixed a display problem on Wednesday.`

Cùng case 136 prompt token, một lượt/process, release CPU, cache expert 128 MiB/layer:

| Worker | TTFT | Tổng | Attention | Expert compute | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36,440 s | 43,480 s | 9,957 s | 26,488 s | 1,989 |
| 8 | 12,955 s | 16,857 s | 4,450 s | 6,056 s | 3,588 |
| 24 | 17,891 s | 23,132 s | 5,556 s | 10,086 s | 2,671 |

Trên máy 24 CPU logic này, 8 worker giảm TTFT **64,4%** và tổng thời gian **61,2%** so với 1 worker. 24 worker chậm hơn 8, phù hợp giới hạn băng thông RAM/overhead; vì vậy chọn trần mặc định 8. RSS peak gần như bằng nhau: 4.010.588 KiB ở 1 worker và 4.010.992 KiB ở 8 worker.

Raw data: [1 worker](../benchmarks/results/2026-09-23-rayon-1-context.json), [8 worker](../benchmarks/results/2026-09-23-rayon-8-context.json), [24 worker](../benchmarks/results/2026-09-23-rayon-24-context.json).

Giới hạn: mỗi cấu hình mới chỉ đo một lượt, page cache/tần số CPU/tải nền không kiểm soát; không diễn giải tỷ lệ thành cam kết trên CPU khác. Parallelism theo row tăng mức sử dụng CPU tức thời. Attention softmax/context vẫn tuần tự và causal prefill vẫn O(N²). Backend GPU chưa được thêm.

Kết luận: đa lõi làm giảm mạnh điểm nghẽn expert compute trên máy hiện tại mà giữ output. Bước tiếp theo là attention batch/SIMD hoặc backend GPU hybrid nếu có phần cứng mục tiêu.

## 2026-09-23 — Bước 8: KV prefix cache theo session

- Thêm `session_id` tùy chọn cho API thường và SSE, validation 1–128 ký tự ASCII an toàn. Các method public cũ vẫn hoạt động với KV mới mỗi request.
- OLMoE lưu token đã thực sự đi qua decoder và KV tương ứng. Request sau chỉ tái sử dụng khi toàn bộ token cache là prefix chính xác và prompt còn suffix; mọi mismatch tự fallback về prefill đầy đủ. RoPE/prefill suffix dùng vị trí tuyệt đối từ độ dài KV đã có.
- Cache session dùng LRU theo lượt truy cập với tổng ngân sách 256 MiB. State bị lấy khỏi cache khi request bắt đầu và chỉ ghi lại sau thành công; request bị hủy không giữ state đang dùng. Metric mới `cached_prompt_tokens` báo số token đã tái sử dụng.
- UI giữ một session ID trong `localStorage`, gửi ID trên mỗi request, hiển thị số token cache và tạo ID mới khi xóa hội thoại.
- Test mới so generation dùng prefix cache với model/KV mới hoàn toàn và xác nhận token output giống nhau; HTTP test kiểm tra session ID lỗi bị từ chối và HTML chứa integration session.

Benchmark HTTP checkpoint thật, 8 Rayon worker, cache expert 128 MiB/layer, context 512, một lần cho mỗi nhánh:

| Lượt hai 60 prompt token | Cached token | TTFT | Tổng | Output |
| --- | ---: | ---: | ---: | --- |
| Session tiếp nối | 37 | 4,098 s | 6,182 s | `Your name is Linh.` |
| Session mới | 0 | 9,679 s | 12,517 s | `Your name is Linh.` |

TTFT giảm 57,7% và tổng giảm 50,6% trong phép đo này. Raw data: [session HTTP](../benchmarks/results/2026-09-23-session-cache-http.json). Đây là một hội thoại ngắn, một lượt đo, page cache/tần số CPU không kiểm soát; lợi ích phụ thuộc phần prefix có thể tái sử dụng.

Kiểm chứng cuối: `cargo test --all-targets --quiet` đạt **35 test** (25 lib, 2 HTTP binary, 5 loader, 3 integration), 2 microbenchmark ignored; `cargo clippy --all-targets -- -D warnings` sạch; release server build thành công; JavaScript inline qua `node --check`.

Giới hạn: cache nằm trong tiến trình và mất khi restart; không chia sẻ giữa replicas, chưa có TTL hay endpoint quan sát/xóa trực tiếp. Xóa UI đổi ID nên state cũ được LRU loại sau. Cache chỉ hỗ trợ exact token prefix; chỉnh lịch sử hoặc tokenization khác sẽ miss an toàn. Ngân sách session cộng thêm expert cache/dense/KV request và không phải hard cap RSS.

Kết luận: hội thoại nhiều lượt giờ tránh prefill lại phần lịch sử đã xử lý khi prefix khớp. Việc tiếp theo là attention batch/SIMD hoặc GPU hybrid; production cần TTL và telemetry session.

## 2026-09-23 — Bước 9: layout KV liên tục và attention dot kernel

- Thay `Vec<Vec<f32>>` cho key/value từng token bằng hai `Vec<f32>` liên tục ở mỗi layer. Attention duyệt token bằng `chunks_exact(key_value_width)`; append suffix dùng `extend_from_slice`. Việc này bỏ một allocation riêng cho mỗi key/value token và giúp session cache lưu payload liền mạch.
- Dùng chung kernel dot product 8 accumulator đã kiểm chứng cho Q·K attention score. Phần weighted-value/context vẫn cộng theo thứ tự token cũ để giữ sai số số học thấp.
- Cập nhật vị trí KV, byte accounting, session LRU và test chiều dài KV cho layout mới. MHA, GQA, logits tham chiếu và session prefix-cache đều qua test.

Kiểm chứng: **35 test đạt**, 2 microbenchmark ignored; Clippy sạch; release binary build thành công. Case 136 token/8 worker sinh cùng 15 token và cùng output ở cả ba lượt.

Benchmark mới 3 lượt/process có median TTFT 15,959 s, tổng 20,391 s, attention 5,677 s, expert compute 8,276 s và decode 3,159 token/s. Raw data: [flat KV 3 lượt](../benchmarks/results/2026-09-23-flat-kv-context-3x.json). Một lượt riêng lưu tại [flat KV](../benchmarks/results/2026-09-23-flat-kv-context.json).

Không tuyên bố tăng tốc end-to-end ở bước này: mốc Rayon-8 trước đó chỉ có một lượt (TTFT 12,955 s), còn các lượt mới cho thấy expert compute cũng dao động mạnh dù code expert không đổi. Điều kiện tải nền/tần số CPU/page cache không đủ ổn định cho so sánh phần trăm. Lợi ích chắc chắn là giảm số allocation KV và layout tuần tự; cần microbenchmark attention cô lập hoặc nhiều cặp chạy xen kẽ để định lượng tốc độ.

Giới hạn: KV vẫn là FP32, attention causal vẫn O(N²), Q/K/V/O vẫn matvec theo token và weighted-value chưa vector hóa riêng. Buffer liên tục có thể phải realloc khi lớn dần; chưa reserve toàn context để tránh tăng RSS cho prompt ngắn.

Kết luận: hoàn tất nền tảng layout cần cho attention block/batch và GPU transfer sau này, giữ đúng logits/output. Bước tiếp theo nên là attention block benchmark cô lập hoặc backend GPU hybrid trên phần cứng mục tiêu.

## 2026-09-24 — Bước 10: attention theo head trên KV liên tục

- Mục tiêu: dùng layout KV liên tục của bước 9 để tách attention theo head ở ngữ cảnh dài, không đổi thứ tự cộng theo token trong từng phần tử context.
- `src/backend.rs`: thêm `causal_attention_into`; xác thực chiều MHA/GQA và scratch, tính score/softmax rồi cộng weighted value trên slice head riêng. Khi số score từ 4.096 trở lên, các head độc lập chạy qua Rayon; context ngắn giữ đường tuần tự để tránh overhead.
- `src/olmoe.rs`: cả decode `step` và prefill layer-major cùng gọi kernel mới. Scratch score cấp phát một lần bằng `num_attention_heads × max_position_embeddings`, nên không còn `clear/extend` cho từng head/token.
- Test mới đối chiếu trực tiếp kernel với tham chiếu scalar cho MHA (2/2 head) và GQA (4/2), 17 vị trí. Hồi quy logits fixture hiện có vẫn chạy.

Lệnh kiểm chứng:

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo test --release --all-targets --quiet
cargo build --release --example benchmark_suite
RAYON_NUM_THREADS=8 target/release/examples/benchmark_suite \
  models/olmoe-1b-7b-int8 1 benchmarks/suite-v2-long-context.json \
  near-context-retrieval \
  > benchmarks/results/2026-09-24-parallel-attention-near-context.json
```

Kết quả: **36 test đạt** (26 lib, 2 HTTP binary, 5 loader, 3 integration), 2 microbenchmark ignored có chủ đích; debug/release đều đạt, Clippy sạch. Benchmark JSON mới parse hợp lệ; output/token IDs giữ nguyên `The access code is ORCHID.` (9 token gồm EOS), `finish_reason=stop`.

So với artifact layer-major 469 prompt token + 24 output budget ngày 2026-09-23, cả hai chạy một lượt release với cache 128 MiB/layer và context 512:

| Chỉ số | Trước | Sau | Thay đổi quan sát |
| --- | ---: | ---: | ---: |
| TTFT | 69,657 s | 41,379 s | -40,6% |
| Tổng generate | 72,845 s | 43,550 s | -40,2% |
| Attention | 25,712 s | 17,374 s | -32,4% |
| Expert compute | 41,023 s | 20,841 s | -49,2% |
| Cache miss / payload expert / KV | 1.445 / 9,115 GB / 125,0 MB | không đổi | — |
| Peak RSS | 4.126.784 KiB | 4.161.848 KiB | +34 MiB |

Raw data mới: [parallel attention near context](../benchmarks/results/2026-09-24-parallel-attention-near-context.json).

Lỗi thực hiện: hai benchmark ban đầu chạy chồng lên nhau và cùng ghi một JSON, nên artifact bị hỏng (hai object nối tiếp); không dùng số đó. Đã chạy lại một tiến trình duy nhất và xác thực bằng `python3 -m json.tool` trước khi ghi số liệu trên.

Giới hạn: đối chứng trước/sau chỉ một lượt và `expert_compute_ms` cũng giảm dù kernel expert không đổi. Page cache, tần số CPU và tải nền không kiểm soát, vì vậy không quy toàn bộ chênh lệch end-to-end cho kernel attention; chỉ xác nhận đúng output, đúng cache counters và số đo mới. Attention vẫn causal O(N²), Q/K/V/O vẫn là matvec theo token; chưa có block matmul/SIMD intrinsics riêng hoặc GPU.

Kết luận: hoàn thành kernel attention dùng chung cho decode/prefill và hồi quy chức năng. Bước kế tiếp nên chạy benchmark xen kẽ nhiều vòng cho attention dài trước khi kết luận tốc độ, sau đó cân nhắc block attention hoặc backend GPU hybrid.

## 2026-09-24 — Bổ sung đo 3 vòng attention gần context

- Mục tiêu: kiểm tra độ dao động của kernel attention mới trước khi xem số đo một lượt ở bước 10 là xu hướng.
- Không đổi mã nguồn. Chạy checkpoint OLMoE release, `RAYON_NUM_THREADS=8`, cache expert 128 MiB/layer, context 512; suite `near-context-retrieval` có 469 prompt token và ngân sách output 24.

```bash
RAYON_NUM_THREADS=8 target/release/examples/benchmark_suite \
  models/olmoe-1b-7b-int8 3 benchmarks/suite-v2-long-context.json \
  near-context-retrieval \
  > benchmarks/results/2026-09-24-parallel-attention-near-context-3x.json
python3 -m json.tool \
  benchmarks/results/2026-09-24-parallel-attention-near-context-3x.json > /dev/null
```

Kết quả: JSON hợp lệ, **3/3** request `finish_reason=stop`; token IDs giống nhau ở cả ba lượt và output đều là `The access code is ORCHID.`. Raw data: [3 lượt attention gần context](../benchmarks/results/2026-09-24-parallel-attention-near-context-3x.json).

| Chỉ số | Lượt 1 | Lượt 2 | Lượt 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 49,074 s | 54,502 s | 52,811 s | 52,811 s |
| Tổng generate | 52,554 s | 58,066 s | 56,086 s | 56,086 s |
| Attention | 20,390 s | 23,345 s | 22,129 s | 22,129 s |
| Expert compute | 24,942 s | 28,111 s | 27,925 s | 27,925 s |
| Peak RSS tiến trình | 4.161.980 KiB | 4.170.624 KiB | 4.170.628 KiB | 4.170.624 KiB |

Giới hạn: đây là các lượt liên tiếp trong một tiến trình nên expert cache/heat history được kế thừa; không phải cold benchmark. Page cache, tần số CPU và tải nền không kiểm soát. Median 3 lượt là ổn định hơn phép đo một lượt, nhưng vẫn không phải đối chứng A/B xen kẽ với kernel cũ; không dùng nó để kết luận phần trăm tăng tốc của bước 10.

Kết luận: output ổn định ở 493/512 token; độ dao động TTFT khoảng 5,4 giây giữa ba lượt xác nhận cần benchmark đối chứng xen kẽ nếu muốn định lượng chính xác. Bước kỹ thuật tiếp theo vẫn là block attention/matmul hoặc GPU hybrid, sau khi xác định mục tiêu phần cứng.

## 2026-09-24 — Bước 11: blocked attention CPU

- Mục tiêu: chia duyệt KV liên tục thành tile cố định để đường attention dài có đơn vị làm việc rõ ràng cho cache/SIMD sau này, nhưng không thay đổi toán học hoặc thứ tự cộng hiện hữu.
- `src/backend.rs`: `causal_attention_into` dùng tile 32 vị trí cho cả score Q·K và weighted-value. Mỗi tile vẫn duyệt token theo thứ tự cũ và context vẫn được cộng tuần tự; không cấp phát thêm. Decode và prefill cùng dùng kernel này qua integration ở bước 10.
- Test MHA/GQA scalar hiện có tiếp tục đối chiếu output; không thêm benchmark synthetic mới vì phép đo phải bao gồm cache/parallelism của checkpoint thật.

Lệnh kiểm chứng:

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo test --release --all-targets --quiet
cargo build --release --example benchmark_suite
RAYON_NUM_THREADS=8 target/release/examples/benchmark_suite \
  models/olmoe-1b-7b-int8 1 benchmarks/suite-v2-long-context.json \
  near-context-retrieval \
  > benchmarks/results/2026-09-24-blocked-attention-near-context.json
```

Kết quả: **36 test đạt** (26 lib, 2 HTTP binary, 5 loader, 3 integration), 2 microbenchmark ignored; Clippy sạch. Checkpoint 469 prompt token/24 output budget trả `The access code is ORCHID.`, 9 token IDs giống mốc cũ, `finish_reason=stop`. Raw data: [blocked attention near context](../benchmarks/results/2026-09-24-blocked-attention-near-context.json).

Một lượt release với cache 128 MiB/layer, context 512, 8 worker: TTFT 48,166 s; tổng 50,802 s; attention 19,295 s; expert compute 23,650 s; expert I/O 7,092 s; peak RSS 4.162.380 KiB. **Không** so phần trăm với lượt trước hoặc median 3 vòng: chỉ một lượt, expert I/O/compute dao động và page cache, CPU frequency, tải nền không kiểm soát.

Giới hạn: tile 32 chỉ tổ chức lại traversal; không thực hiện FlashAttention, online softmax, matrix-matrix, SIMD intrinsics hoặc giảm độ phức tạp O(N²). Lợi ích thực tế phải được xác nhận bằng benchmark A/B xen kẽ cùng điều kiện. Tile size hiện là hằng nội bộ, chưa tự điều chỉnh theo CPU/head width.

Kết luận: blocked attention bản đầu đã tích hợp và giữ output/logits fixture. Bước tiếp theo nên là benchmark A/B xen kẽ giữa tile 1 và tile 32, hoặc thiết kế online-softmax theo block nếu cần giảm scratch/băng thông ở context dài hơn.

## 2026-09-24 — Bước 12: softmax online theo block

- Mục tiêu: bỏ buffer score `heads × context` và lượt đọc value thứ hai trong causal attention, vẫn dùng tile 32 KV đã thêm ở bước 11.
- `src/backend.rs`: `causal_attention_into` duyệt từng tile K/V một lần. Với mỗi score, giữ `maximum` và `normalizer` ổn định; khi maximum mới xuất hiện, context tạm được rescale rồi cộng value có trọng số. Cuối head chia context cho normalizer. Kiểm tra score/normalizer hữu hạn được thêm vào.
- `src/olmoe.rs`: bỏ `StepScratch::scores`; prefill và decode dùng API online chung. Với context tối đa 512 và 16 head của checkpoint hiện tại, bỏ khoảng 32 KiB scratch/request; lợi ích chính dự kiến là không cần materialize score và không phải duyệt value lần hai, không phải một tuyên bố tốc độ đã đo.
- Test MHA/GQA scalar đã cập nhật sang API mới và vẫn so kết quả attention; các fixture logits Transformers MHA/GQA vẫn là hồi quy end-to-end.

Lệnh kiểm chứng:

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
cargo test --release --all-targets --quiet
```

Kết quả: **36 test đạt** ở debug và release (26 lib, 2 HTTP binary, 5 loader, 3 integration), 2 microbenchmark ignored có chủ đích; Clippy sạch. Hồi quy logits fixture tiếp tục đạt ngưỡng tuyệt đối `< 2e-5` đã có. **Chưa đo checkpoint thật** cho bước này, nên không có artifact hay số liệu trước/sau và không suy diễn tăng tốc.

Lỗi thực hiện: bản vá đầu tiên không khớp format code hiện hành nên không áp dụng; không làm thay đổi workspace. Sau khi áp dụng đúng, test unit còn truyền score scratch cũ vào API mới và không biên dịch; đã cập nhật test rồi chạy lại toàn bộ suite đạt.

Giới hạn: online softmax đổi phép kết hợp FP32 so với softmax materialized; fixture nhỏ đạt nhưng chưa phải kiểm tra mọi checkpoint/prompt. Thuật toán vẫn O(N²), context vẫn được cập nhật theo từng value và chưa có SIMD intrinsics/matrix-matrix. Cần benchmark checkpoint 469 token nhiều lượt trước khi đánh giá băng thông hoặc wall time.

Kết luận: hoàn thành nền tảng online-softmax không cấp phát score trên attention. Bước tiếp theo là benchmark 3 vòng gần context với cùng cấu hình và so token IDs/RSS, sau đó mới quyết định tối ưu SIMD hoặc GPU hybrid.

## 2026-09-24 — Đo checkpoint thật cho softmax online

- Mục tiêu: kiểm chứng end-to-end thay đổi softmax online ở bước 12, thay vì suy luận từ fixture hoặc cấu trúc code.
- Không đổi mã sau bước 12. Chạy release checkpoint OLMoE với 8 Rayon worker, cache expert 128 MiB/layer, context 512; case `near-context-retrieval` có 469 prompt token và ngân sách 24 output token.

```bash
cargo build --release --example benchmark_suite
RAYON_NUM_THREADS=8 target/release/examples/benchmark_suite \
  models/olmoe-1b-7b-int8 3 benchmarks/suite-v2-long-context.json \
  near-context-retrieval \
  > benchmarks/results/2026-09-24-online-softmax-near-context-3x.json
python3 -m json.tool \
  benchmarks/results/2026-09-24-online-softmax-near-context-3x.json > /dev/null
```

Kết quả: artifact JSON hợp lệ; **3/3** request dừng EOS, token IDs giống nhau giữa ba lượt và đúng output `The access code is ORCHID.`. Raw data: [online softmax 3 lượt](../benchmarks/results/2026-09-24-online-softmax-near-context-3x.json).

| Chỉ số | Lượt 1 | Lượt 2 | Lượt 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 50,296 s | 42,281 s | 42,503 s | 42,503 s |
| Tổng generate | 52,873 s | 44,287 s | 44,504 s | 44,504 s |
| Attention | 21,352 s | 21,672 s | 22,090 s | 21,672 s |
| Expert compute | 20,363 s | 18,290 s | 18,233 s | 18,290 s |
| Expert I/O | 9,883 s | 3,426 s | 3,329 s | 3,426 s |
| Peak RSS tiến trình | 3.740.132 KiB | 3.790.192 KiB | 3.791.236 KiB | 3.790.192 KiB |

Đối chiếu với median ba lượt trước softmax online (bước bổ sung attention ngày 2026-09-24): TTFT 52,811→42,503 s, tổng 56,086→44,504 s và attention 22,129→21,672 s. Cache expert, page cache, heat history, tải nền và CPU frequency không được reset/cố định; expert compute/I/O cũng đổi mạnh dù kernel expert không đổi. Vì thế bảng chỉ là quan sát giữa hai phiên chạy, **không** dùng để quy 19,5% TTFT hay 20,7% tổng thời gian cho softmax online. Bằng chứng chắc chắn là output ổn định và attention median không thoái lui trong mẫu này.

Giới hạn: các vòng kế thừa cache trong cùng tiến trình; không phải cold benchmark, không có A/B xen kẽ cùng binary/cùng trạng thái hệ điều hành. RSS là VmHWM tiến trình, không phải hard cap hay toàn bộ page cache. Attention vẫn O(N²).

Kết luận: softmax online đã xác nhận trên checkpoint thật gần context limit và giữ output. Bước tiếp theo nên là benchmark A/B xen kẽ có công tắc materialized/online hoặc chuyển sang SIMD/GPU hybrid nếu ưu tiên thời gian triển khai hơn độ chính xác đo vi mô.

## 2026-09-24 — Bước 13: A/B attention online và materialized

- Mục tiêu: đo tách biệt online-softmax và materialized-score trên đúng chiều attention dài, tránh quy khác biệt giữa hai phiên checkpoint cho một thay đổi kernel.
- `src/backend.rs`: thêm microbenchmark ignored `benchmark_attention_online_vs_materialized`. Dữ liệu K/V FP32 resident tổng hợp với 16 head × 128 width × 469 positions, 5 warmup mỗi biến thể, 7 mẫu × 10 lượt, thứ tự biến thể luân phiên. Baseline materialized dùng score vector mới cho mỗi head/lượt, nên không đại diện hoàn toàn scratch tái sử dụng của production nhưng là đối chứng bảo thủ cho online về allocation.

Lệnh:

```bash
cargo fmt --all
cargo test --all-targets --quiet
cargo clippy --all-targets -- -D warnings
MOE_ATTENTION_REPORT=benchmarks/results/2026-09-24-attention-online-vs-materialized.json \
  cargo test --release --lib \
  backend::tests::benchmark_attention_online_vs_materialized -- --ignored --exact
```

Kết quả: **36 test thường đạt**, 3 microbenchmark ignored; Clippy sạch. Microbenchmark chạy riêng đạt. Raw data: [online vs materialized](../benchmarks/results/2026-09-24-attention-online-vs-materialized.json).

| Kernel attention cô lập | Median 7 mẫu |
| --- | ---: |
| Online softmax theo block | 1,008 ms |
| Materialized score | 0,448 ms |
| Sai số tuyệt đối tối đa output | 3,47e-8 |

Kết quả **không đạt mục tiêu hiệu năng**: online chậm khoảng 2,25× trong benchmark này. Đây không mâu thuẫn với checkpoint end-to-end vì attention chỉ là một phần thời gian, nhưng loại bỏ cơ sở để gọi online là tối ưu tốc độ. Hơn nữa baseline materialized ở benchmark còn cấp phát score mỗi lượt, trong khi đường production cũ tái sử dụng scratch; do đó phép đo không có lợi cho materialized về allocation.

Giới hạn: buffer tổng hợp resident không đo I/O/expert, chỉ một cấu hình Rayon/máy, CPU frequency/tải nền không kiểm soát. Không dùng tỷ lệ này để dự đoán toàn model. Tuy vậy chênh lệch cùng kernel/cùng tiến trình đủ để yêu cầu đối chứng production trước khi giữ online làm mặc định.

Kết luận: đã có A/B tái lập và phát hiện online-softmax hiện không phải nâng cấp hiệu năng CPU. Bước tiếp theo là khôi phục materialized scratch làm mặc định hoặc tối ưu online (giảm rescale context khi max đổi) rồi chạy lại A/B; không tiếp tục tuyên bố online giúp tốc độ.

## Mẫu cho mỗi bước tiếp theo

```text
Ngày / Bước / Trạng thái:
Mục tiêu:
Thay đổi và file liên quan:
Lệnh kiểm thử:
Kết quả kiểm thử (pass/fail):
Điều kiện benchmark và liên kết dữ liệu thô:
Số đo trước / sau (hoặc chưa đo):
Lỗi, thử nghiệm thất bại và cách xử lý:
Giới hạn còn lại:
Kết luận và bước tiếp theo:
```
