
# MoE-TierEngine

Engine Rust chạy **OLMoE với expert INT8 theo định dạng merged INT8 trên CPU + RAM + SSD**. Không yêu cầu GPU. Đã nối loader Safetensors, decoder OLMoE, tokenizer, chat template và API. Chế độ `tiny-moe` dùng trọng số ngẫu nhiên vẫn có để kiểm thử.

Yêu cầu Rust 1.85 trở lên; lần build đầu cần tải các dependency tokenizer/template. Nên dùng **`--release`** khi chạy model thật.

## Chạy checkpoint thật đã có trong workspace

Từ thư mục `MoE`:

```bash
MOE_MODEL_DIR=./models/olmoe-1b-7b-int8 \
MOE_BIND=127.0.0.1:8081 \
MOE_CONTEXT=512 \
MOE_CACHE_BYTES_PER_LAYER=134217728 \
RAYON_NUM_THREADS=8 \
cargo run --release --bin moe-tier-engine
```

Terminal khác:

```bash
curl -sS http://127.0.0.1:8081/health
curl -sS http://127.0.0.1:8081/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"olmoe","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":16}'
```

Mở `http://127.0.0.1:8081/` để dùng giao diện chat nhúng trong binary. Giao
diện lưu lịch sử và lựa chọn số token trong `localStorage`, cho phép dừng SSE,
xóa hội thoại, đồng thời hiển thị usage, TTFT, tốc độ decode, tổng thời gian và
lý do kết thúc. Lịch sử phía trình duyệt được giới hạn 100 message/48 KiB nội
dung để tránh request tăng không giới hạn.

Hoặc chạy một câu trực tiếp, không cần mở server:

```bash
cargo run --release --example chat_olmoe -- \
  ./models/olmoe-1b-7b-int8 'What is 2 + 2?' 16
```

Đây là CPU reference implementation, chưa dùng BLAS hay SIMD chuyên biệt. Prefill chạy theo layer, gom các token theo expert để tái sử dụng expert đã tải và chỉ tính LM head ở token prompt cuối. Prompt dài vẫn chậm vì attention, expert compute và I/O. API hỗ trợ SSE cho OLMoE để hiển thị nội dung dần sau prefill. Khả năng tiếng Việt phụ thuộc checkpoint, không phải chỉ engine.

Các matvec dense và expert đủ lớn chạy song song theo hàng bằng Rayon. Mặc định
engine dùng tối đa 8 worker vì phép tính này thường bị giới hạn bởi băng thông
RAM; nhiều luồng hơn không nhất thiết nhanh hơn. Đặt `RAYON_NUM_THREADS=1` để
chạy đơn luồng hoặc benchmark 4/8/16 trên máy đích để chọn mức phù hợp.

Không chạy `--init-demo` cho checkpoint thật. Nếu cổng bị chiếm, đặt `MOE_BIND` sang cổng khác hoặc dừng server cũ trong terminal của nó.

## Bộ nhớ và cấu hình

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `MOE_MODEL_DIR` | `./models/olmoe-1b-7b-int8` | Thư mục model; có `model.json` chọn demo, nếu không chọn OLMoE |
| `MOE_BIND` | `127.0.0.1:8080` | Địa chỉ HTTP |
| `MOE_CONTEXT` | `512` | Giới hạn tổng prompt + output của OLMoE, không quá config model |
| `MOE_CACHE_BYTES_PER_LAYER` | `134217728` | Ngân sách expert INT8 cho mỗi layer; `0` tắt cache |
| `MOE_DENSE_BYTES` | `3221225472` | Giới hạn trọng số dùng chung sau chuyển sang FP32 |
| `MOE_STREAM_SEND_TIMEOUT_MS` | `30000` | Thời gian tối đa worker chờ queue SSE có chỗ; phải lớn hơn `0` |
| `RAYON_NUM_THREADS` | tối đa 8 | Số worker cho dense/expert matvec lớn; có thể điều chỉnh theo CPU và băng thông RAM |

Với 16 layer, cache mặc định tối đa 2 GiB tổng. Mỗi expert của checkpoint hiện tại khoảng 6,3 MB; 128 MiB/layer giữ khoảng 21 expert và ít nhất chứa được tập 8 expert đang hoạt động của một token. Engine cảnh báo khi cache nhỏ hơn active set này. Embeddings, attention, router, norm và LM head giữ trong RAM FP32; expert giữ INT8 và scale theo hàng. Cache này tách biệt với page cache của hệ điều hành. Giới hạn dense/cache **không phải hard cap RSS**: còn bộ đệm đọc, tokenizer, metadata, KV cache và activation. Context ngắn giúp giảm KV memory. Không nạp toàn bộ expert vào RAM lúc khởi động, chỉ kiểm tra descriptor.

OLMoE phục vụ một yêu cầu sinh token tại một thời điểm; yêu cầu chồng nhau trả `429`. Phép tính và I/O chạy qua `spawn_blocking`, nên không chặn async executor. Với `stream:true`, ngắt kết nối sẽ hủy sinh tại lần kiểm tra kế tiếp giữa các layer/expert; không ngắt tức thời một phép matvec hay I/O đang chạy. Lượt chạy được giải phóng sau khi worker dừng. Với `stream:false`, phép sinh vẫn hoàn thành khi client ngắt.

## Định dạng checkpoint hỗ trợ

Dùng converter độc lập `tools/prepare_olmoe.py` trong dự án này:

```text
olmoe_merged/
  config.json
  tokenizer.json
  tokenizer_config.json       # chat_template, bos_token, eos_token
  model-00000.safetensors
  model-00001.safetensors
  ...
```

- Trọng số dùng chung F32/F16/BF16 được giải mã FP32, kiểm tra shape và giá trị hữu hạn.
- Expert: `model.layers.L.mlp.experts.E.merged_weight` là vector I8 theo thứ tự gate/up/down; `.qs` là vector F32 scale theo hàng.
- Header shard được lập chỉ mục một lần; giữ một file handle cho mỗi shard đến khi model được giải phóng. Đọc payload theo offset khi cần, kiểm tra ngân sách byte và kích thước. Các lượt đọc cùng shard dùng mutex để bảo vệ cặp seek/read; chưa có prefetch I/O song song.
- Hỗ trợ MHA/GQA, Q/K RMSNorm trên vector projection, RoPE kiểu chia nửa, SiLU/SwiGLU, routing softmax toàn expert rồi Top-K và tùy chọn `norm_topk_prob`.
- Từ chối config có attention bias, tied embeddings, clip QKV hoặc rope scaling. Không hỗ trợ GGUF, raw HF expert BF16 hay họ model khác.
- Tokenizer chạy bằng thư viện Hugging Face `tokenizers`; template đọc từ checkpoint và render bằng MiniJinja. Hiện kiểm chứng template chuỗi của checkpoint OLMoE trong workspace; không tuyên bố hỗ trợ mọi Jinja template.
- EOS lấy từ `config.json`. Sinh greedy; chưa áp dụng các tùy chọn sampling/generation tùy ý từ `generation_config.json`.

Không sửa checkpoint trong khi server đang chạy. Trường `model` của request chỉ xác nhận model đang được nạp, không tải/chuyển model động.

## Kiến trúc phần mềm

```text
main.rs / examples/chat_olmoe.rs
  → engine.rs: chọn demo/OLMoE, giới hạn một yêu cầu OLMoE
  → olmoe.rs: config, tokenizer/template, decoder, KV cache, greedy generation
      → safetensors.rs: lập chỉ mục và đọc tensor theo nhu cầu
      → int8_expert.rs: expert INT8 SwiGLU
      → backend.rs: dense FP32 CPU
```

Cache OLMoE lưu các `Int8Expert` theo layer. Chính sách theo dõi tần suất truy cập xuyên request, dùng thời điểm truy cập để phá hòa và chỉ nhận expert mới khi nó không lạnh hơn expert sắp bị thay. Bộ đếm được giảm một nửa định kỳ để lịch sử cũ không chi phối mãi. Cache không dùng file `expert-N.bin` hay vùng VRAM mô phỏng của demo. Key/value của mỗi layer nằm trong hai buffer FP32 liên tục theo token; `session_id` có thể giữ các buffer này giữa những lượt có token-prefix khớp.

Trong prefill, engine xử lý causal attention theo thứ tự token ở từng layer,
sau đó gom các route cùng expert. Một lần tải expert phục vụ mọi prompt token
được route đến expert đó trong layer. Các contribution được lưu theo slot và
cộng lại đúng thứ tự Top-K ban đầu để giữ kết quả số. Đường decode sau token
đầu vẫn xử lý từng token. Buffer contribution tạm thời tăng theo
`prompt_tokens × top_k × hidden_size` và được giải phóng sau request.

API hỗ trợ `messages` (system/user/assistant), `model`, `max_tokens`, `stream:false/true` (SSE chỉ cho OLMoE); trường chưa hỗ trợ bị từ chối. Trả `choices`, `finish_reason`, `usage`. `400` cho sai model/context/messages hoặc streaming trên demo; `429` khi OLMoE đang bận.

`session_id` tùy chọn cho phép OLMoE tái sử dụng KV của token-prefix trùng khớp
giữa các lượt. ID gồm 1–128 ký tự ASCII chữ/số/`.`/`-`/`_`. Cache session có
ngân sách tổng 256 MiB và loại session cũ nhất khi cần; request vẫn đúng khi
cache miss vì engine tự prefill lại. Metric `cached_prompt_tokens` cho biết số
token prompt đã bỏ qua. UI tự tạo ID và đổi ID khi bấm **Xóa hội thoại**.

## Model demo

```bash
cargo run -- --init-demo demo-model  # chỉ chạy một lần, không ghi đè thư mục cũ
MOE_MODEL_DIR=./demo-model cargo run
```

Gọi API với `"model":"tiny-moe"`. Model demo có 2 layer và trọng số ngẫu nhiên, không trả lời có nghĩa. Các module `model/`, `attention.rs`, `expert.rs`, `generation.rs`, `tokenizer.rs` phục vụ demo. `moe.rs` giữ lớp tuyến tính INT8 MEX1 cũ; `quantization.rs` có giải mã INT4 nhưng đường OLMoE hiện dùng INT8. Chưa có GPU/VRAM thật, `io_uring` hay `O_DIRECT`.

## Kiểm chứng

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --example inspect_model -- ./models/olmoe-1b-7b-int8
```

`tests/fixtures/olmoe` chứa checkpoint rất nhỏ và logits tham chiếu từ Transformers 4.51.3. Trọng số expert được lượng tử hóa INT8 rồi giải mã lại trong mô hình tham chiếu để so sánh cùng phép tính. Test so sánh mọi logits của từng prefix với sai số tuyệt đối < `2e-5`: MHA không chuẩn hóa lại Top-K và GQA có chuẩn hóa lại, cache tắt/nhỏ/lớn, nhiều yêu cầu liên tiếp. Có test template/tokenizer, context, input lỗi và ngân sách dense.

Tái tạo fixture (không cần cho `cargo test` thường ngày):

```bash
python -m venv --system-site-packages /tmp/moe-reference-env
/tmp/moe-reference-env/bin/pip install torch 'transformers==4.51.3' safetensors
/tmp/moe-reference-env/bin/python tools/make_olmoe_reference.py tests/fixtures/olmoe
```

Các test nhỏ kiểm chứng tính toán, không thay thế đánh giá chất lượng toàn bộ checkpoint 7B. Lượng tử hóa INT8 có thể đổi logits/token so với trọng số pretrained nguyên bản. Chưa có benchmark chất lượng/ngữ cảnh dài hoặc tốc độ có tính đại diện.

Bộ smoke chất lượng có 12 câu deterministic cho toán, kiến thức, tiếng Việt,
làm theo chỉ dẫn, code và truy xuất ngữ cảnh:

```bash
cargo run --release --example evaluate_quality -- \
  models/olmoe-1b-7b-int8 benchmarks/quality-v1.json \
  > benchmarks/results/quality-v1.json
```

Đây là phép chấm exact/substring nhỏ, không phải benchmark chất lượng chuẩn hóa.
Kết quả tham chiếu hiện tại nằm tại
[`benchmarks/results/2026-09-23-quality-v1.json`](benchmarks/results/2026-09-23-quality-v1.json).

Kiểm tra 50 vòng HTTP bằng fixture, xen kẽ hoàn tất SSE và hủy kết nối:

```bash
python3 tools/stress_http.py tests/fixtures/olmoe/unnormalized-mha \
  benchmarks/results/http-stress.json 50
```

Script tự mở/tắt release server, kiểm tra health, permit sau hủy, DONE/usage và
độ ổn định RSS. Fixture chỉ kiểm thử vòng đời HTTP nhanh; nó không mô phỏng tải
CPU/I/O của checkpoint thật.

Kiểm tra thực tế trong workspace ngày 2026-09-22: checkpoint `./models/olmoe-1b-7b-int8` trả `2 + 2 equals 4.` cho câu hỏi `What is 2 + 2?` qua HTTP, sinh 7 token gồm EOS. Một lần thử CLI với prompt `Hello! What is 2 + 2?` mất 29,52 giây cho prefill + generation, chưa tính 1,91 giây nạp model. Lần thử HTTP ghi nhận peak RSS khoảng 2,30 GiB (không tính page cache toàn hệ thống). Đây chỉ là smoke test một câu ngắn, không phải benchmark tổng quát. Đã kiểm tra `/health` vẫn phản hồi khi sinh token, yêu cầu chồng nhau trả 429 và input sai trả 400.


## Chuẩn bị model độc lập

Toàn bộ source, converter và checkpoint chạy hiện tại nằm trong `MoE`. Không cần thư mục hay executable của dự án khác. Checkpoint hiện có tại `models/olmoe-1b-7b-int8` là bản sao riêng, không dùng symlink/hardlink. Thư mục `models/` không được đưa vào Git; khi chuyển máy cần sao chép model hoặc tải/chuyển đổi lại.

Chuẩn bị từ checkpoint Hugging Face gốc:

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy safetensors huggingface_hub
.venv/bin/python tools/prepare_olmoe.py \
  --repo allenai/OLMoE-1B-7B-0125-Instruct \
  --out models/olmoe-new-int8
```

Hoặc dùng `--source /path/to/original-hf-checkpoint` thay cho `--repo`. Converter đọc từng tensor, buffer output khoảng 64 MiB (tensor đơn lớn hơn ngưỡng vẫn được ghi nguyên vẹn), gộp expert INT8 và scale. Download giữ checkpoint gốc trong cache Hugging Face, nên cần dung lượng cho cả bản gốc và bản đã chuyển đổi. Python chỉ dùng lúc chuẩn bị model; runtime là Rust.

Converter từ chối ghi đè thư mục đã tồn tại. Nếu chuyển đổi lỗi, thư mục có dấu `INCOMPLETE` và engine từ chối chạy; chọn thư mục mới để thử lại. Không hỗ trợ resume. Có thể dùng `--revision` để chốt revision nguồn.

Kiểm thử converter: `python tools/test_prepare_olmoe.py` trong môi trường Python đã cài torch và safetensors.

## Nhật ký nâng cấp

Theo dõi lộ trình, thay đổi từng bước, kết quả kiểm thử và số đo trước/sau tại [docs/PROGRESS.md](docs/PROGRESS.md). Các số đo chưa thực hiện được đánh dấu rõ; smoke test hiện tại chưa phải baseline benchmark chuẩn.

Benchmark chuẩn xuất JSON gồm thời gian load, tokenizer, prefill, token đầu tiên,
decode, attention, expert, LM head, I/O và cache:

```bash
mkdir -p benchmarks/results
cargo run --release --example benchmark_olmoe -- \
  models/olmoe-1b-7b-int8 'What is 2 + 2?' 16 2 128 \
  > benchmarks/results/baseline.json
```

Lần chạy đầu có expert cache rỗng; lần sau dùng cache trong tiến trình. Công cụ
không xóa hay kiểm soát page cache của hệ điều hành, nên file JSON ghi rõ giới hạn này.

Kiểm tra cache nóng với ba prompt khác nhau qua hai vòng:

```bash
cargo run --release --example benchmark_workload -- \
  models/olmoe-1b-7b-int8 8 2 128 \
  > benchmarks/results/workload.json
```


### Đo riêng kernel expert INT8

Kernel INT8 dùng 8 bộ cộng FP32 độc lập để giảm phụ thuộc giữa các phép cộng;
compiler có thể vector hóa với target mặc định. Không cần cờ CPU riêng, không
thêm thread và không giải nén toàn bộ ma trận sang FP32. Thứ tự cộng khác bản
scalar nên có thể có chênh lệch làm tròn; kiểm thử vẫn đối chiếu logits fixture.

Microbenchmark so sánh trực tiếp kernel scalar cũ với kernel mới, dùng cùng
buffer đã nạp, 9 mẫu luân phiên thứ tự, mỗi mẫu 30 phép matvec:

```bash
MOE_KERNEL_REPORT=benchmarks/results/int8-kernel.json \
cargo test --release --lib int8_expert::kernel_tests::benchmark_int8_kernel -- --ignored --exact
```

JSON chứa thời gian từng mẫu và sai số so với scalar. Đây là dữ liệu tổng hợp
ở kích thước projection OLMoE, không gồm đọc SSD, SwiGLU hay toàn bộ decoder;
không diễn giải mức tăng tốc kernel thành mức tăng tốc cả model.

### Đo riêng kernel dense FP32

Q/K/V/O, router và LM head dùng dot product với 8 bộ cộng FP32 độc lập. Cách
này portable như kernel INT8, không yêu cầu `target-cpu=native` và không thêm
thread. Microbenchmark scalar/mới:

```bash
MOE_DENSE_REPORT=benchmarks/results/dense-kernel.json \
cargo test --release --lib backend::tests::benchmark_dense_kernel -- --ignored --exact
```

Thứ tự cộng FP32 thay đổi nên kết quả không bitwise-identical với scalar; test
đối chiếu FP64 và logits fixture bảo vệ ngưỡng sai số. JSON microbenchmark chỉ
đo buffer resident, không gồm expert, attention softmax hoặc I/O.


## SSE và hủy yêu cầu

```bash
curl -N http://127.0.0.1:8081/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"olmoe","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":32,"stream":true}'
```

Mỗi event `data:` chứa JSON với `choices[0].delta`: role đầu tiên, rồi content
được giải mã tăng dần. Một chunk có thể chứa nhiều token để giữ nguyên ký tự
Unicode. Chunk cuối chứa `finish_reason`, `usage`, `metrics`, sau đó là
`data: [DONE]`. Role/keepalive không phải nội dung đầu tiên; thời gian nhận
content thực tế có thể lớn hơn TTFT nội bộ do decoder, queue và mạng.

Ctrl+C ở curl hoặc `AbortController.abort()` ở client sẽ đóng kết nối. Queue
worker giới hạn 8 event; client chậm tạo backpressure. Việc ngắt kết nối giải
phóng queue và báo worker dừng, kể cả trong prefill. Kiểm tra input/context và
lượt chạy bận diễn ra trước HTTP 200. Lỗi sau khi mở SSE được gửi bằng JSON
`error` rồi `[DONE]`; client phải kiểm tra error, không coi `[DONE]` là thành công.
Streaming không làm giảm thời gian tính prefill. Request/response thường giữ
định dạng cũ. Nếu queue 8 event vẫn đầy quá `MOE_STREAM_SEND_TIMEOUT_MS`, worker
dừng và giải phóng lượt chạy. Đây là timeout backpressure, không phải giới hạn
tổng thời gian sinh. Hiện chưa có endpoint hủy theo request ID và chưa cung cấp
đầy đủ mọi trường giao thức OpenAI.

Request HTTP tối đa 128 KiB; phần `messages` tối đa 128 phần tử và tổng content
64 KiB. Tokenizer/context vẫn áp dụng giới hạn nhỏ hơn theo model.

## Giao diện chat local

Sau khi server báo sẵn sàng, mở `http://127.0.0.1:8081/` (hoặc cổng trong
`MOE_BIND`). Giao diện được nhúng trong binary, không cần Node/npm. Nó giữ lịch
sử `messages`, đọc SSE tăng dần, hiển thị usage và dùng `AbortController` cho
nút **Dừng**. Enter gửi, Shift+Enter xuống dòng.

Nếu dừng sau khi đã nhận một phần câu trả lời, phần đó được giữ trong lịch sử
để lượt sau có đúng ngữ cảnh người dùng đã thấy. Giao diện chỉ dành cho chạy
local: chưa có đăng nhập, lưu lịch sử lâu dài, Markdown renderer hoặc upload.

Smoke test HTTP tự mở và dừng server local của nó:

```bash
cargo build --release --bin moe-tier-engine
python3 tools/test_streaming_smoke.py models/olmoe-1b-7b-int8 \
  benchmarks/results/streaming-smoke.json
```

## Bộ benchmark hoàn thiện dự án

`benchmarks/suite-v1.json` cố định 5 tình huống: toán, code Python, tiếng Việt,
hội thoại nhiều lượt và tìm thông tin trong đoạn dài. Mỗi case có messages và
ngân sách output riêng. Chạy 3 vòng theo cùng thứ tự:

```bash
cargo run --release --example benchmark_suite -- models/olmoe-1b-7b-int8 3 \
  > benchmarks/results/suite-v1.json
```

Có thể thêm `CASE_ID` sau đường dẫn suite để đo riêng một case mà vẫn nhúng
nguyên suite vào JSON, ví dụ `context-retrieval`. Suite
`benchmarks/suite-v2-long-context.json` có 469 prompt token; với ngân sách 24
output token, tổng tối đa là 493/512.

JSON nhúng nguyên suite, config model, phiên bản Rust, cấu hình RAM/context và
metrics mỗi request. `process_peak_rss_kib` lấy VmHWM trên Linux: peak tích lũy
của tiến trình, không phải RAM riêng từng request hay page cache hệ thống.
Kiểm tra nội dung câu trả lời và finish reason trước khi chấm chất lượng;
không suy ra chất lượng từ thời gian chạy. Bộ này chưa phải benchmark chất
lượng chuẩn hoặc thử nghiệm gần giới hạn context 512 token.

Kế hoạch hoàn thiện và điều kiện nghiệm thu: [docs/PLAN.md](docs/PLAN.md).
