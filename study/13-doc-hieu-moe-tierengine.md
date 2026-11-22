# Module 13 — Đọc hiểu MoE-TierEngine

## Mục tiêu

Theo một request đi qua toàn bộ hệ thống, từ lúc khởi động server đến lúc token cuối cùng hiện trên trình duyệt, biết mỗi bước nằm ở file và hàm nào. Sau module này bạn tự tìm được chỗ cần sửa khi có yêu cầu thay đổi, và giải thích được kiến trúc trước hội đồng mà không cần nhìn tài liệu.

**Dùng khi đi làm:** kỹ năng đọc một codebase lạ là việc đầu tiên ở mọi công việc mới. Phương pháp trong module này (đi theo một request, vẽ bản đồ, đặt câu hỏi rồi tìm trong code) áp dụng cho mọi dự án.

## 1. Bản đồ thư mục

```
DoanTotNghiepMOE/
├── Cargo.toml              dependency Rust
├── src/
│   ├── main.rs             server HTTP: route, SSE, ánh xạ lỗi           (873 dòng)
│   ├── engine.rs           InferenceEngine: Semaphore, spawn_blocking, stream, hủy (674)
│   ├── olmoe.rs            OlmoeModel: nạp mô hình, prefill, decode, cache expert (8 249)
│   ├── int8_expert.rs      expert INT8: nạp, kernel SwiGLU CPU, AVX2       (1 859)
│   ├── int8_expert/batch_prototype.rs   kernel nhiều token
│   ├── gpu.rs              CUDA driver/cuBLAS qua dlopen, dense GPU, pool expert (2 563)
│   ├── gpu_expert.rs       kernel INT8 SwiGLU trên GPU, NVRTC              (583)
│   ├── backend.rs          dot product, attention online softmax, RMSNorm, softmax (787)
│   ├── dense.rs            ma trận dense FP32/BF16                         (219)
│   ├── safetensors.rs      đọc tệp checkpoint theo offset                  (348)
│   ├── ram.rs              RAM guard, đọc /proc, cgroup                   (337)
│   ├── sample.rs           temperature, top-k, top-p                       (305)
│   ├── quantization.rs     FP8, INT4 giải mã
│   ├── tokenizer.rs, router.rs, attention.rs, moe.rs, model/   mô hình tí hon để kiểm thử
├── web/index.html          giao diện (nhúng vào binary bằng include_str!)
├── tests/                  so logits với Transformers trên fixture
├── examples/dump_logits.rs ghi logits để so trên mô hình thật
├── tools/                  chuyển đổi checkpoint, chạy benchmark, so logits, demo.sh
├── benchmarks/             protocol (SCENARIOS.md) và hồ sơ đo (experiments/)
├── docs/                   PROGRESS (nhật ký), PLAN, ARCHITECTURE, WORKFLOWS
└── baocao/                 báo cáo LaTeX
```

`src/olmoe.rs` chiếm gần một nửa code. Đừng đọc từ đầu đến cuối; hãy đi theo luồng dưới đây.

## 2. Chuẩn bị checkpoint (một lần, ngoài engine)

`tools/prepare_olmoe.py` đọc checkpoint BF16 gốc của Hugging Face và ghi checkpoint mới:

- Với mỗi expert, lượng tử hóa ba ma trận `gate_proj`, `up_proj`, `down_proj` về INT8 theo hàng, **nối lại thành một tensor** `model.layers.{l}.mlp.experts.{e}.merged_weight` cùng tensor hệ số tỉ lệ `...qs`. Nhờ vậy một expert là một lần đọc liên tục.
- Phần dense giữ nguyên BF16.
- Chép tokenizer và cấu hình.

Kết quả: OLMoE khoảng 7 GB, Qwen3 khoảng 30 GB.

## 3. Khởi động server

`tools/demo.sh olmoe-gpu` đặt biến môi trường (`MOE_MODEL_DIR`, `MOE_CACHE_BYTES_PER_LAYER`, `MOE_GPU_MODE`, `MOE_GPU_EXPERT_TIER`, `MOE_GPU_EXPERT_COMPUTE`, `MOE_RAM_LIMIT_BYTES`...), build bản release, rồi chạy binary.

Trong `src/main.rs`:

1. Đọc cấu hình từ biến môi trường.
2. Gọi `InferenceEngine::load_with_gpu_mode` (`src/engine.rs`). Hàm này chạy việc nạp trong `spawn_blocking` vì nạp mất vài giây đến vài chục giây.
3. Dựng router Axum (`app`) và lắng nghe cổng.

Trong `OlmoeModel::load_with_gpu_mode` (`src/olmoe.rs`), theo thứ tự:

| Bước | Việc | Ở đâu |
| --- | --- | --- |
| 1 | Đọc `config.json`, nhận diện OLMoE hay Qwen3 (`model_type`), kiểm cấu hình hỗ trợ được | `Config::validate`, `is_qwen3_moe` |
| 2 | Quét header mọi shard safetensors, lập bảng tên tensor → (shard, offset, kích thước) | `TensorIndex::open` trong `src/safetensors.rs` |
| 3 | Nạp tokenizer, khuôn mẫu hội thoại | |
| 4 | Ước lượng bộ nhớ cần (dense + cache + KV + dự trữ) và so với trần RAM; vượt thì dừng với lỗi rõ ràng | RAM guard, `src/ram.rs` |
| 5 | Nạp dense (FP32 hoặc BF16 theo `MOE_DENSE_STORAGE`), embedding vào RAM hoặc để trên đĩa | `dense_mat`, `embedding_on_disk` |
| 6 | Nếu chế độ GPU: khởi tạo CUDA, ước lượng VRAM; đủ thì đưa dense lên VRAM và cấp pool expert; không đủ thì chuyển CPU | `src/gpu.rs` |
| 7 | Nếu bật tính expert trên GPU: biên dịch kernel bằng NVRTC | `src/gpu_expert.rs` |
| 8 | Khởi tạo cache rỗng cho mỗi lớp | `Layer` |

Expert **không** được nạp lúc khởi động. Chúng chỉ được đọc khi router chọn.

## 4. Một request chat

Người dùng gõ "Where is Mars?" và bấm gửi. Trình duyệt gửi `POST /v1/chat/completions` với `"stream": true`.

### 4.1 Lớp HTTP (`src/main.rs`, hàm `chat`)

1. Axum parse JSON thành `ChatRequest` (từ chối trường lạ, body tối đa 128 KB).
2. Kiểm tên mô hình yêu cầu khớp mô hình đang nạp (`accepts_model`).
3. Gọi `engine.chat_stream_with_session_and_sampling(...)` và **chờ** đến khi engine báo sẵn sàng (đầu vào đã hợp lệ).
4. Trả `Sse::new(SseEvents { ... })`, header `200 OK` được gửi.

### 4.2 Lớp điều phối (`src/engine.rs`)

1. Kiểm tin nhắn và mã phiên.
2. Lấy giấy phép Semaphore (`try_acquire_owned`); đang bận thì trả lỗi 429 ngay.
3. Tạo channel có giới hạn 8 phần tử và kênh `oneshot` báo sẵn sàng.
4. `spawn_blocking`: trong luồng riêng, khóa mô hình, render prompt, gọi `generate_controlled_with_sampling` với ba closure:
   - `check`: kiểm client còn kết nối (`tx.is_closed()`) và chưa quá hạn tổng;
   - `ready`: gửi tín hiệu sẵn sàng;
   - `on_delta`: gửi đoạn văn bản mới vào channel, có timeout.

### 4.3 Sinh văn bản (`src/olmoe.rs`, `generate_inner`)

```
1. Kiểm RAM guard, render khuôn mẫu, mã hóa thành token.
2. Kiểm: số token đầu vào + max_tokens ≤ cửa sổ ngữ cảnh.
3. ready()  ──► lớp HTTP gửi header SSE.
4. take_prefix_kv: tìm phiên, dùng lại KV của tiền tố chung dài nhất.
5. prefill(phần token còn lại)            → logits của vị trí cuối.  [đo TTFT đến đây]
6. Lặp:
     token = argmax(logits)  (hoặc sample nếu có tham số)
     giải mã thành văn bản, gửi qua on_delta
     nếu token là EOS hoặc đủ max_tokens: dừng
     logits = step(token)                                          [decode]
     check()
7. Lưu KV của phiên (nếu đủ ngân sách), trả Generation kèm metrics.
```

### 4.4 Một bước decode (`step_inner`)

Với mỗi lớp `l` trong L lớp:

```
 x (trạng thái ẩn)
 │
 ├─ RMSNorm ─► Q, K, V = dense_qkv(...)              (GPU cuBLAS nếu bật, không thì CPU)
 │             QK-norm; RoPE (bảng sin/cos tính sẵn)
 │             nối K, V vào kv[l]
 │             causal_attention_into(Q, kv[l])        (online softmax, GQA)
 │             O = dense_gemv(W_O, context)
 ├─ x = x + O
 │
 ├─ RMSNorm ─► router: softmax, top-k (select_nth), trọng số g (chuẩn hóa nếu Qwen3)
 │             phân loại từng expert đã chọn:
 │               • có trong cache RAM            → hit, tính CPU
 │               • có trong pool VRAM (A1 bật)   → tính GPU tại chỗ
 │               • không có                      → miss, phải đọc tệp
 │             rayon::join(
 │                 đọc song song các expert miss từ tệp,
 │                 tính song song các expert hit trên CPU )
 │             tính các expert vừa đọc; tính các expert trên GPU
 │             admit_expert_l2: đưa expert vào cache, loại expert heat thấp nếu đầy
 │             y = Σ g_e · out_e   (cộng theo thứ tự top-k cố định)
 └─ x = x + y
```

Sau lớp cuối: RMSNorm, LM head (GPU nếu bật), ra logits.

Điểm thiết kế đáng nhớ trong bước MoE:

- **Che độ trễ I/O bằng tính toán:** `rayon::join` cho phép trong lúc đọc các expert miss từ tệp, CPU tính các expert đã có. Nếu có 5 hit và 3 miss, thời gian gần bằng `max(đọc 3, tính 5)` thay vì tổng.
- **Đọc miss theo thứ tự id tăng dần** (`sorted_miss_ids.sort_unstable()`), để các lần đọc gần nhau trong tệp hơn.
- **Tái sử dụng bộ đệm:** đọc expert vào bộ đệm lấy từ `recycled_buffers`, không cấp phát mới.
- **Thứ tự cộng cố định** để kết quả tất định.

### 4.5 Prefill khác decode thế nào (`prefill`)

Prefill xử lý **mọi token của prompt qua lớp 1, rồi qua lớp 2**, ... (layer-major):

1. Dense cho cả lô bằng GEMM (`dense_gemm`, `dense_gemm_qkv`).
2. Router cho mọi token, rồi gom theo expert: expert e được những token nào chọn.
3. Mỗi expert được lấy (từ cache, VRAM hoặc tệp) **một lần** và tính cho mọi token cần nó. Nếu bật `MOE_EXPERT_BATCH_PREFILL`, tính theo lô 2 hoặc 4 token bằng kernel nhiều token.
4. Có thể nạp trước tối đa k expert miss của lớp (prefill prefetch, mặc định tắt).

### 4.6 Trở lại trình duyệt

Mỗi đoạn văn bản đi qua channel tới `SseEvents::poll_next`, thành một sự kiện SSE. Sự kiện cuối chứa `usage` và `metrics` (TTFT, decode tok/s, `expert_io_ms`, `expert_file_bytes_read`...), rồi `[DONE]`. Trang web cập nhật bảng số liệu, rồi gọi `GET /experts` để lấy trạng thái từng lớp (expert nào trong VRAM, trong RAM, bộ đếm heat) và vẽ lại ma trận expert.

## 5. Các đường dẫn đặc biệt

| Tình huống | Điều gì xảy ra | Ở đâu |
| --- | --- | --- |
| Người dùng bấm Stop | channel đóng → `check()` trả lỗi `Interrupted` → dừng, trả giấy phép | `src/engine.rs` |
| Request thứ hai khi đang bận | `try_acquire` thất bại → 429 | `src/engine.rs`, `api_error` |
| Prompt quá dài | lỗi `InvalidInput` trước khi gửi header → 400 | `generate_inner` |
| RAM vượt ngưỡng khi chạy | RAM guard trả `OutOfMemory`, dọn cache RAM → 503 hoặc sự kiện lỗi | `src/ram.rs`, `generate_inner` |
| Lỗi GPU khi chạy | bật circuit breaker, các lần sau dùng CPU | `src/gpu.rs` |
| Không có driver NVIDIA | `dlopen` thất bại → chạy CPU | `CudaDriverApi::load` |
| `POST /api/cache/reset` | xóa cache RAM, phiên, bộ đệm tái sử dụng | `OlmoeModel::reset_cache` |

## 6. Cách tự tìm đường trong code

Khi có câu hỏi "X được làm ở đâu?", dùng các cách sau theo thứ tự:

1. **Tìm theo tên biến môi trường hoặc trường JSON:** `grep -rn "MOE_GPU_EXPERT_COMPUTE" src/`, `grep -rn "expert_file_bytes_read" src/`.
2. **Tìm theo tên test:** `grep -n "fn .*matches" src/olmoe.rs` liệt kê các bất biến đã được kiểm.
3. **Theo kiểu dữ liệu:** muốn biết cache expert gồm gì, tìm `struct CachedExpert` rồi tìm mọi chỗ dùng nó.
4. **Dùng IDE:** rust-analyzer trong VS Code cho "Go to Definition" và "Find All References".
5. **Đọc nhật ký:** `docs/PROGRESS.md` ghi lý do của từng thay đổi lớn; `git log -S "tên_hàm"` cho biết hàm được thêm khi nào và vì sao.

---

## Bài tập

### Bài 1 ★ — Lần theo biến môi trường

Với mỗi biến sau, tìm nơi nó được đọc và giá trị mặc định: `MOE_CACHE_BYTES_PER_LAYER`, `MOE_DENSE_STORAGE`, `MOE_EMBEDDING_STORAGE`, `MOE_GPU_EXPERT_COMPUTE`, `MOE_EXPERT_BATCH_PREFILL`, `MOE_RAM_LIMIT_BYTES`, `RAYON_NUM_THREADS`.

<details><summary>Gợi ý</summary>

`grep -rn "\"MOE_" src/` liệt kê tất cả. `MOE_DENSE_STORAGE` mặc định `f32` (`src/dense.rs`); `MOE_EMBEDDING_STORAGE` mặc định `auto` (đĩa khi bảng lớn hơn ngưỡng); `MOE_RAM_LIMIT_BYTES` mặc định một nửa RAM, tối đa 6 GiB (`src/ram.rs`). Tự tìm phần còn lại.
</details>

### Bài 2 ★ — Từ chỉ số đến code

Trường `expert_io_ms` trong kết quả được cộng ở những chỗ nào? Nó gồm thời gian đọc tệp, thời gian chép từ VRAM, hay cả hai?

<details><summary>Lời giải</summary>

Trong `acquire_expert_tiered`, `profile.expert_io` được cộng cả khi tải từ VRAM (`profile.expert_io += elapsed; profile.vram_io += elapsed`) lẫn khi đọc tệp (`profile.expert_io += io_elapsed`), và thêm ở nhánh đọc song song khi decode. Vì vậy `expert_io_ms` là tổng thời gian chờ có được expert, từ mọi nguồn. Các trường `expert_file_read_ms` và `expert_vram_download_ms` tách riêng từng nguồn.
</details>

### Bài 3 ★★ — Thêm một chỉ số

Thêm vào metrics một trường `expert_gpu_bytes_avoided`: số byte expert lẽ ra phải đọc từ tệp nhưng được tính trên GPU (bằng số lần tính expert trên GPU nhân kích thước một expert). Tìm nơi đếm `expert_gpu_computes`, thêm trường vào `RequestProfile` và vào struct metrics trong `src/generation.rs`, chạy test. **Làm trên một nhánh git riêng và không merge**; đây là bài tập.

<details><summary>Gợi ý</summary>

Lần từ `pub expert_gpu_computes: u64` trong `src/generation.rs` ngược lên chỗ nó được gán từ `RequestProfile`. Có thể có nhiều chỗ khởi tạo struct metrics (ví dụ trong test); trình biên dịch sẽ chỉ ra hết các chỗ thiếu trường mới. Đây chính là lợi ích của kiểu dữ liệu chặt chẽ trong Rust (module 01, mục 6).
</details>

### Bài 4 ★★ — Vẽ sơ đồ của riêng bạn

Không nhìn module này, vẽ (trên giấy hoặc draw.io) sơ đồ một request từ trình duyệt đến khi token đầu tiên hiện ra, ghi tên file và hàm ở mỗi bước. Sau đó so với mục 4 và sửa chỗ thiếu.

### Bài 5 ★★ — Đọc API thật

Chạy `tools/demo.sh olmoe-cpu`, gửi một request, rồi:

```bash
curl -s http://127.0.0.1:8080/experts | python3 -m json.tool | head -60
```

Giải thích từng trường ở cấp cao nhất (`cache_bytes_per_layer`, `ram_rss_limit_bytes`, `expert_payload_bytes`, `layers[0]`...). Với `layers[0]`, expert nào đang ở RAM, heat của chúng là bao nhiêu? Đối chiếu với `ExpertStatusReport` và `LayerExpertStatus` trong `src/olmoe.rs`.

### Bài 6 ★★★ — Đề xuất thay đổi

Hướng phát triển số 1 của đồ án là "dùng VRAM làm kho expert khi dense không vừa" (để Qwen3 trên GTX 1650 dùng được pool expert). Không viết code, hãy viết một bản thiết kế 1 trang:

1. Hiện tại pool expert được cấp ở đâu và vì sao nó phụ thuộc vào dense trên GPU?
2. Kernel GPU nhận đầu vào từ đâu (RAM hay VRAM)? Điều đó có cho phép tách pool khỏi dense không?
3. Những hàm nào phải sửa?
4. Bạn sẽ đo lợi ích bằng thí nghiệm nào (viết protocol theo module 11)?

<details><summary>Gợi ý</summary>

Pool expert nằm trong `GpuDenseStorage` (`src/gpu.rs`), chỉ được tạo khi dense lên GPU thành công. Kernel nhận vector đầu vào từ RAM (`cuMemcpyHtoD` trong `src/gpu_expert.rs`) và trả kết quả về RAM, nên về nguyên tắc không cần dense trên GPU. Thay đổi chính: tách `expert_pool`, `expert_cache` và kernel ra khỏi `GpuDenseStorage` thành một cấu trúc riêng có thể tồn tại khi dense ở CPU. Thí nghiệm: ABBA trên Qwen3, i5, trần 10 GiB, tắt và bật pool, chỉ số chính là tổng thời gian và decode tok/s, chỉ số công việc là số lần đọc tệp. Đây là bài tập tốt nhất để chuẩn bị cho câu hỏi "nếu có thêm thời gian, em sẽ làm gì?".
</details>
