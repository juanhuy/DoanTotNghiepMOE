# Review hệ thống ngày 2026-10-22

Review tại commit `1d98587`, cây làm việc sạch khi bắt đầu. Không sửa runtime trong đợt này.

## Phạm vi và kết luận

Đã rà soát đường OLMoE từ loader Safetensors/INT8, tokenizer/chat template, prefill/decode, cache expert và session đến engine, API HTTP/SSE và JavaScript giao diện. Đã xem các module tiny model, memory manager, quantization, công cụ converter, benchmark runner/exporter, test và tài liệu tiến độ. Archive không thuộc phạm vi mã đang vận hành.

Phát hiện **2 vấn đề P1 và 7 vấn đề P2** bên dưới. P1 cần ưu tiên xử lý vì liên quan giới hạn bộ nhớ hoặc tính an toàn của API Rust; P2 gây sai chức năng, sai số liệu hoặc mất bằng chứng. Bộ test hiện có đạt nhưng chưa bao phủ các trường hợp này.

## Các phát hiện theo mức ưu tiên

### R1 — P1: Cache session tính bộ nhớ theo chiều dài, bỏ qua capacity đã cấp phát

- Vị trí: `src/olmoe.rs:2049`, `src/olmoe.rs:2123`, `src/olmoe.rs:2185`.
- Mỗi request cấp KV cho toàn bộ `context_limit` qua `Kv::with_capacity`. Khi giữ session, điều kiện admission/eviction và `session_bytes` chỉ đếm `keys.len() + values.len()`.
- Nhiều session với prompt ngắn giữ allocation lớn hơn đáng kể ngân sách 256 MiB; bộ đếm vẫn cho phép thêm session. Đây là rủi ro tăng bộ nhớ ngoài dự kiến, có thể gây áp lực RAM/OOM khi tải nhiều session. Capacity là bộ nhớ allocator giữ, không đồng nghĩa tất cả các trang đã resident trong RSS.
- **Tái hiện mới:** trên fixture, đặt ngân sách session 1.024 byte; hai session được giữ với tổng byte được tính là 1.024 nhưng capacity KV thực tế là **32.768 byte**, gấp 32 lần ngân sách. Chưa đo RSS của nhiều session trên checkpoint 7B.
- Đề xuất: tính capacity và metadata khi admission/eviction, hoặc thu gọn KV trước khi giữ; tránh tạo KV mới toàn context trước khi biết có thể tái sử dụng session cũ. Thêm hồi quy nhiều session ngắn với ngân sách nhỏ.

### R2 — P1: Metadata pinned buffer cho phép phá invariant từ safe Rust

- Vị trí: `src/gpu.rs:1228`, `src/gpu.rs:1672`.
- `PinnedBuffer.len` là trường `pub` có thể sửa. `GpuDenseStorage::upload_and_cache` là safe API nhưng tin giá trị này để gọi `as_mut_slice::<u8>()` trong khối unsafe và ghi vào vùng nhớ host.
- Caller có thể cấp pinned buffer nhỏ, tăng `len` bằng safe Rust rồi đưa buffer vào `GpuDenseStorage.pinned_upload_buf`. Lần upload lớn hơn allocation thật sẽ tạo slice vượt allocation và có thể ghi ngoài vùng nhớ. Việc caller không gọi unsafe không bảo đảm an toàn khi invariant này bị mở công khai.
- **Bằng chứng:** phân tích đường gọi và visibility trong mã. Chưa thực thi trên CUDA, chưa tái hiện ghi ngoài vùng nhớ. Chưa xác định đường HTTP cho phép sửa metadata; phát hiện này thuộc API thư viện Rust.
- Đề xuất: ẩn pointer/length/capacity, cung cấp getter; mọi safe method phải kiểm tra kích thước theo allocation nội bộ không thể bị caller sửa. Rà soát tương tự các trường pointer/dimension/ownership công khai trong lớp GPU.

### R3 — P2: Reset cache tắt vĩnh viễn việc giữ KV session

- Vị trí: `src/olmoe.rs:569`.
- `reset_cache()` đặt `session_cache_bytes = 0`, dù trường này là ngân sách, không phải bộ đếm dung lượng đang dùng. Sau `POST /api/cache/reset`, mọi request có KV khác rỗng đều không được giữ session nữa, cho đến khi nạp lại model.
- **Tái hiện mới:** với cùng chuỗi prompt tăng dần, `cached_prompt_tokens` trước reset là **2**, sau reset là **0**; ngân sách còn lại là **0**.
- Đề xuất: chỉ xóa entries và clock, giữ ngân sách cấu hình; test reset → tạo session → tái sử dụng prefix.

### R4 — P2: Pool buffer expert nằm ngoài giới hạn RAM cache

- Vị trí: `src/olmoe.rs:1163`, `src/olmoe.rs:870`, `src/olmoe.rs:2122`.
- Eviction trừ `layer.bytes` nhưng giữ nguyên weights/scales trong `recycled_buffers`, tối đa 16 cặp mỗi layer. Pool này không được tính vào ngân sách hoặc `expert_cache_bytes`. Prefill còn cấp buffer cho toàn bộ missing experts trước khi thu kết quả, không giới hạn số expert đang nạp theo ngân sách.
- **Tái hiện mới:** tổng ngân sách L2 hai layer **1.800 byte**; cache được báo **1.024 byte**, nhưng recycled buffers giữ thêm **3.072 byte**, tổng hai phần là **4.096 byte**.
- Với kích thước OLMoE hidden=2.048/intermediate=1.024, trần 16 buffer × 16 layer tương đương khoảng **1,50 GiB** ngoài cache. Đây là phép tính từ cấu hình và giới hạn mã, không phải số đo RSS mới. Bộ nhớ tạm prefill còn có thể tăng thêm.
- Đề xuất: có ngân sách byte chung cho cache và pool; giới hạn prefetch theo số byte đang giữ; báo riêng dung lượng resident, recycled và inflight.

### R5 — P2: Web bỏ qua lỗi SSE và chấp nhận stream thiếu kết quả cuối

- Vị trí: `web/index.html:1893`, `web/index.html:1924`.
- Backend gửi `{"error":...}` khi sinh lỗi sau khi HTTP headers đã trả 200. Frontend chỉ đọc `choices` và `metrics`, không kiểm tra `event.error`. Khi EOF đến, nó vẫn thêm `assistantMsg` vào history/localStorage. Không có điều kiện bắt buộc phải nhận finish event và `[DONE]`.
- Người dùng có thể thấy câu trả lời rỗng hoặc bị cắt nhưng lượt hội thoại được lưu như thành công, và các lượt sau dùng lịch sử sai.
- **Tái hiện mới:** chạy nguyên vòng xử lý event đã trích từ web bằng Node, đầu vào là event `generation_error` rồi `[DONE]`; không có lỗi được đưa ra (`surfaced_error=false`). Test HTTP hiện có xác nhận backend thực sự phát loại event này.
- Đề xuất: xử lý `event.error`, theo dõi finish/DONE và từ chối EOF bất thường; chỉ ghi history thành công sau khi kiểm tra terminal event. Thêm hồi quy frontend với lỗi sau delta đầu tiên và stream bị cắt.

### R6 — P2: Byte đọc expert trộn dữ liệu từ VRAM với dữ liệu từ file

- Vị trí: `src/olmoe.rs:1023`, `src/olmoe.rs:2176`; nơi gắn nhãn SSD: `tools/export_benchmark_evidence.py:195`.
- Nhánh tải expert từ VRAM cộng `resident_bytes()` vào cùng `profile.bytes_read` mà nhánh đọc checkpoint dùng. Giá trị được trả thành `expert_bytes_read` rồi exporter gọi là dung lượng đọc SSD.
- Một request chỉ miss L2 nhưng hit L1 vẫn tăng chỉ số được báo là SSD. Vì vậy số liệu này không thể dùng để kết luận SSD đã đọc bao nhiêu. Đọc file cũng có thể được phục vụ từ page cache.
- **Bằng chứng:** phân tích đường cập nhật bộ đếm và exporter; chưa đo lại trên GPU.
- Đề xuất: tách `expert_file_bytes_read`, `vram_download_bytes`, `vram_upload_bytes`, byte prefetch bị bỏ; dùng `/proc/.../io` hoặc công cụ hệ thống nếu cần số đo thiết bị lưu trữ.

### R7 — P2: VRAM telemetry bỏ qua dung lượng pool đã cấp phát

- Vị trí: `src/gpu.rs:1335`, `src/olmoe.rs:515`.
- `GpuExpertPool::new` cấp phát toàn bộ weights/scales cho `max_slots` ngay lúc khởi tạo. `expert_status()` lại tính `allocated_vram_bytes` bằng dense weights cộng byte expert đang có trong cache, bỏ qua các slot pool còn trống và scratch của engine.
- Ngay sau load hoặc reset, báo cáo có thể cho thấy nhiều VRAM trống dù các block pool vẫn được cấp phát. `free_vram_bytes = total - logical_used` còn bỏ qua bộ nhớ của tiến trình khác. UI dùng trực tiếp các số này để hiển thị dung lượng VRAM.
- **Bằng chứng:** phân tích mã; chưa đối chiếu với số đo thiết bị CUDA. Trường `dense_bytes` cùng report cũng đang cố định 3 GiB thay vì tổng trọng số đã nạp.
- Đề xuất: tách logical cache occupancy và physical allocated/reserved; lưu kích thước allocation pool/scratch; query free/total thực tế nếu muốn hiển thị bộ nhớ thiết bị.

### R8 — P2: Benchmark runner ghi đè kết quả cùng ngày và tag

- Vị trí: `tools/run_benchmark.py:87`, `tools/run_benchmark.py:120`.
- Tên file chỉ gồm ngày và tag; `open(..., "w")` ghi đè không kiểm tra tồn tại. Chạy lại cùng lệnh, hoặc thay cache/prompt nhưng giữ tag mặc định, sẽ mất kết quả thô cũ. Điều này trái yêu cầu giữ mốc so sánh trong `AGENTS.md`.
- **Tái hiện mới:** mock subprocess trong thư mục tạm; chạy runner hai lần cùng tag với hai JSON khác nhau. Chỉ còn **1 file**, nội dung đầu bị thay bằng nội dung lượt hai. Không chạy benchmark model và không sửa dữ liệu thô hiện có.
- Đề xuất: tên chứa thời gian/run ID và cấu hình, tạo file độc quyền (`x`) hoặc chọn suffix khi trùng; ghi commit, checkpoint, thread count và trạng thái GPU thực tế trong metadata.

### R9 — P2: Sparse routing visualization hiển thị expert ngẫu nhiên

- Vị trí: `web/index.html:1906`.
- Mỗi delta văn bản kích hoạt `Math.random()` để chọn một expert ở các layer chẵn. Đây không phải top-8 của model và không phải sự kiện route theo token. Nhãn giao diện lại là `Top-8 Sparse Routing (12.5% active)`.
- Bảng hiện tại có thể gây hiểu sai khi dùng demo/luận văn để trình bày expert nào đang hoạt động. Snapshot backend trong lúc inference cũng có thể là trạng thái cũ do model đang bị khóa.
- **Bằng chứng:** đọc logic frontend và engine snapshot. Không có số đo routing mới.
- Đề xuất: gắn nhãn animation minh họa rõ ràng hoặc truyền route events thật từ backend; phân biệt snapshot sau request với telemetry trực tiếp.

## Kiểm chứng đã thực hiện

Bằng chứng lưu tại [2026-10-22-evidence.json](2026-10-22-evidence.json). Mã ba probe Rust nằm trong [2026-10-22-review-probes.rs](2026-10-22-review-probes.rs): để tái hiện, tạo bản sao source/manifest/lockfile/fixture trong thư mục tạm, nối file probe vào `src/olmoe.rs` của bản sao và chạy `cargo test --manifest-path <bản-sao>/Cargo.toml --lib review_probes -- --nocapture`. Không nối vào source vận hành khi chỉ cần đọc báo cáo.

| Kiểm tra | Kết quả mới |
| --- | --- |
| `cargo test --all-targets --locked` | 59 pass, 0 fail, 3 ignored |
| `cargo test --release --all-targets --locked` | 59 pass, 0 fail, 3 ignored |
| `cargo clippy --all-targets --locked -- -D warnings` | Đạt |
| `cargo fmt --all -- --check` | Không đạt; diff trong `src/olmoe.rs`, `tests/int8_loader.rs` |
| `cargo test --locked gpu::tests::test_gpu_probe -- --nocapture` | Available=false, không có CUDA |
| `cargo build --release --locked --bin moe-tier-engine` | Đạt |
| `python3 tools/stress_http.py tests/fixtures/olmoe/unnormalized-mha /tmp/moe-review-stress.json 30` | 30 vòng đạt: 20 hoàn tất, 10 hủy; RSS steady spread 124 KiB |
| `python3 tools/test_prepare_olmoe.py` | Không chạy được test: thiếu module `torch` |
| `node --check /tmp/moe-review-web.js` | Cú pháp JavaScript đạt |
| Ba probe Rust trên bản sao `/tmp/moe-review-n9s2ykj7` | Đạt điều kiện tái hiện R1/R3/R4; đây là test xác nhận lỗi tồn tại, không phải xác nhận đã vá |
| Parser SSE trích từ web, chạy bằng Node | Tái hiện R5 |
| Benchmark runner với mock subprocess/thư mục tạm | Tái hiện R8 |

Stress HTTP lần đầu bị sandbox chặn tạo socket (`PermissionError`); đã chạy lại sau khi được cấp quyền. Không có lỗi ứng dụng trong lần chạy lại. RSS trên fixture không đại diện tải checkpoint 7B hoặc nhiều session. Các test GPU dùng nhánh `if let Some(engine)` nên có thể trả thành công khi GPU không khả dụng; không coi đây là xác nhận CUDA.

Môi trường lệnh: Linux, rustc `1.99.1 (58cf8f9c3 2026-10-13)`, Python `3.14.8`. Chưa kiểm tra MSRV Rust 1.85, Windows hoặc browser E2E. Không tải thêm dependency, không chạy benchmark 7B hoặc đánh giá chất lượng model mới, không audit CVE online. Không suy ra tăng/giảm hiệu năng từ đợt review này.

## Bằng chứng và giới hạn của các kết luận cũ

- Các mốc hiệu năng trong `PROGRESS.md` là lịch sử; đợt review không chạy lại và không xác nhận phần trăm tăng tốc đó. Không dùng so sánh cache 128/256 MiB như bằng chứng riêng của tối ưu code.
- `Int8Expert::load_reusing` vẫn có ba `format!` tại `src/int8_expert.rs:66`; decode còn tạo Vec/HashMap. Tái sử dụng payload lớn không chứng minh zero heap allocations cho toàn đường chạy.
- `read_exact_at` ở đây đọc qua `File::open` thông thường vào buffer host. Mã này không đủ để chứng minh đường SSD→host là zero-copy DMA, không dùng page cache hoặc không còn page faults.
- Chưa có phép đo allocation/page-fault mới; cần profiler/counter có dữ liệu thô trước khi sử dụng các khẳng định tuyệt đối trong báo cáo nghiên cứu.
- Runner nhận `--cache` nhưng khi chạy `--suite` không truyền cache cho binary; `benchmark_suite.rs` hiện cố định 128 MiB. CLI nên từ chối cấu hình không hỗ trợ hoặc thực sự truyền cấu hình để tránh hiểu nhầm.
- Default path của một số examples còn trỏ `benchmarks/suite-v1.json`/`benchmarks/quality-v1.json`, trong khi file hiện nằm dưới `benchmarks/suites/`; cần đồng bộ trước khi dùng lệnh mặc định.

## Thứ tự xử lý đề xuất

1. R1/R2: đóng invariant bộ nhớ session và pinned buffer.
2. R3/R4/R5: sửa reset, ngân sách RAM chung và trạng thái lỗi frontend; thêm hồi quy theo trigger đã tái hiện.
3. R6/R7/R8/R9: chuẩn hóa counters, VRAM telemetry, bảo toàn dữ liệu benchmark và nhãn visualization.
4. Chạy lại CUDA trên máy có GPU, converter trong môi trường có dependency và kiểm thử browser; sau đó mới đo hiệu năng với checkpoint/cấu hình tương đương.

Trạng thái: review hoàn thành trong phạm vi trên; **các phát hiện chưa được vá**.
