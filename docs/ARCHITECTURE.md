# Kiến trúc hiện tại

Mô tả mã hiện có; invariant mong muốn không đồng nghĩa đã được bảo đảm. Phát hiện gốc: [review R1–R9](reviews/2026-10-22-system-review.md); trạng thái vá hiện hành ở [PLAN](PLAN.md).

```mermaid
flowchart LR
    UI[Web HTML nhúng] --> HTTP[Axum HTTP / SSE]
    HTTP --> Engine[InferenceEngine: gate / blocking worker]
    Engine --> OLMoE[OlmoeModel: prefill / decode]
    OLMoE --> Dense[Dense attention / LM head: CPU hoặc cuBLAS]
    OLMoE --> Router[Top-K router]
    Router --> RAM[Expert cache RAM L2]
    RAM --> Compute[INT8 SwiGLU: CPU Rayon]
    OLMoE --> VRAM[Expert cache VRAM L1]
    VRAM --> RAM
    OLMoE --> Files[Safetensors / positional reads]
    Files --> RAM
    OLMoE --> KV[KV request / prefix session cache]
```

## Ranh giới module

| Module | Trách nhiệm | Invariant khi sửa |
| --- | --- | --- |
| `src/main.rs` | HTTP/status/control, SSE serialize, HTML nhúng | 400/429 trước SSE; lỗi sau headers thành error event; DONE một lần |
| `src/engine.rs` | API thư viện, một request OLMoE, spawn_blocking, queue/cancel | Không block Tokio; hủy/timeout nhả permit; snapshot có thể cũ |
| `src/olmoe.rs` | Load/config, prefill/decode, RoPE/KV/route, caches/metrics | Causal positions, reduction xác định, budget tất cả buffer, model dùng lại sau lỗi |
| `src/int8_expert.rs` | Merged layout/scales, payload recycling, CPU SwiGLU | Shape/scales hợp lệ, không mở rộng toàn expert FP32 |
| `src/safetensors.rs` | Header/offset/shape/overlap, đọc theo nhu cầu | File bất biến sau index, kiểm budget trước allocation khi API có budget |
| `src/backend.rs` | Dense/norm/softmax/attention CPU, Rayon | Tolerance fixture, finite input/output, threads thực |
| `src/gpu.rs` | CUDA/cuBLAS FFI, matrix/pinned/pool/cache | Allocation length không giả mạo, context/lifetime/drop order, scratch đồng bộ/fallback |
| `src/model/`, attention/expert/moe/memory/router/tokenizer | Tiny/reference building blocks | Không nhầm tiny format với OLMoE |
| `src/generation.rs` | Kết quả/metrics và tiny generation | Token count, finish reason, định nghĩa throughput |

Expert có thể tìm RAM trước, tải từ VRAM về host, rồi fallback file. **L1/L2 mô tả tầng lưu trữ**, không khẳng định thứ tự tra cứu hoặc expert compute trên GPU. SwiGLU hiện chạy CPU; dense GPU giữ bản weights CPU cho fallback.

Mặc định `MOE_EXPERT_CACHE_POLICY=victim`: expert đang ở RAM không upload bản sao lên VRAM; khi evict RAM, upload victim trước khi thu hồi host buffer. Admit trở lại RAM xóa bản sao L1 và trả slot sau khi download đồng bộ đã xong. Nếu expert không vừa L2, L1 vẫn có thể giữ nó. `inclusive` giữ hành vi cũ để đối chứng. Đổi policy qua `POST /api/cache/reset?expert_cache_policy=victim|inclusive` chỉ khi model idle, cùng model lock với reset; request không query giữ policy hiện hành. Không đổi thứ tự route/reduction hoặc trọng số.

## Request và bộ nhớ

1. Validate messages/model/session/context; gate trả WouldBlock/429 nếu OLMoE bận.
2. Blocking worker render template/tokenize; streaming báo ready sau validation.
3. Prefill theo layer/gom route expert, token của mỗi expert xử lý lần lượt. Candidate dispatch nhóm token chỉ còn ở cfg(test) vì A/B i5 cho thấy chậm hơn. Decode dispatch expert song song rồi cộng theo Top-K. Delta qua bounded queue; disconnect/deadline hủy hợp tác.
4. Text/token IDs/finish/metrics trả về. Prefix KV có thể giữ theo session; client vẫn gửi toàn lịch sử. Khi lượt mới khác phần cuối hoặc ngắn hơn, giữ longest common token prefix và truncate KV theo đúng vị trí causal; luôn prefill ít nhất token cuối vì session không lưu logits. Chỉ cấp KV mới nếu không có prefix dùng lại, tránh cấp bộ buffer tạm trước lookup.

Non-stream chưa hủy worker theo disconnect. Status khi suy luận có thể là snapshot cũ. Frontend chỉ commit history/metrics sau terminal+DONE và EOF hợp lệ; error/abort không commit, reader được cancel/release. Routing/VRAM placement hiển thị là minh họa, không dùng làm trace token.

Dense host/GPU, pool/scratch, L2 cache, recycled/inflight, KV capacity và tokenizer là vùng riêng. Cache limit không phải tổng RSS. Session admission/eviction tính capacity keys/values, token IDs, vector KV, SessionState và session ID; chưa gồm HashMap buckets/allocator metadata hoặc KV của request đang chạy. Metric `kv_cache_bytes` vẫn là logical keys/values bytes của request, không phải toàn bộ allocation session. Cache expert và recycled pool dùng chung ngân sách capacity mỗi layer; status báo recycled_bytes riêng. Prefill chỉ prefetch tối đa topK experts, phần còn lại đọc khi cần. Active expert payload có headroom riêng tối đa (topK+2) × bytes/expert, bao gồm decode ready/miss và lookahead2 hoặc prefill preloaded+expert đang xử lý; chưa gồm metadata, scratch/dense/KV/pinned staging và không phải tổng RSS. Reset xóa session/expert entries và recycled buffers nhưng giữ ngân sách KV để lượt sau tiếp tục lưu session. VRAM report thiếu physical pool/scratch; `expert_bytes_read` trộn file/PCIe. Timer I/O/compute chồng lấn không cộng thành wall time. Theo dõi trong PLAN trước khi dùng số liệu.

PinnedBuffer sở hữu pointer/length private, allocation portable không rỗng và không quá isize::MAX; host bytes được zero-init trước khi có slice. Getter chỉ đọc kích thước thật. Typed slice vẫn unsafe: caller chịu trách nhiệm validity/alignment của T, aliasing và đồng bộ DMA; giữ driver sống đến Drop. Upload kiểm kích thước staging bằng checked arithmetic, fallback host slices khi buffer quá nhỏ. Đây là invariant pinned buffer đã kiểm, không phải tuyên bố an toàn toàn bộ CUDA API.

Metrics mới tách `prefill_expert_io_ms`/`decode_expert_io_ms`, `expert_file_read_ms`, `expert_vram_download_ms`, `expert_vram_upload_ms` cùng payload/load counters. File timer là wall của đường load/prefetch (gồm scheduling), không phải riêng thời gian syscall; upload gồm đường staging/upload và không nằm trong legacy `expert_io_ms`. File payload đếm các load thành công, gồm speculative chưa dùng; không phải physical SSD bytes. HTTP runner ghi delta `/proc/<pid>/io:read_bytes` riêng. Không cộng các timer chồng lấn hoặc suy băng thông thiết bị từ payload/timer này.

## Hướng thay đổi

Sửa invariant với hồi quy trước, rồi tách accounting/telemetry khi cần. Chưa chia crate/di chuyển `olmoe.rs` hoặc `gpu.rs` trong lượt chuẩn hóa; refactor lớn cần PR riêng giữ API và fixture parity để không trộn thay đổi kiến trúc với bug fixes.


## Bảo vệ RAM hợp tác (đang kiểm ổn định)

Server không override cache dùng64MiB/layer cố định, không tăng theo MemAvailable. Model load kiểm kế hoạch dense decoded FP32 + toàn bộ cache cấu hình +256MiB session + working request estimate +512MiB overhead trước khi đọc weights lớn. Dense bytes trên status là số bytes weights thực đã load, không còn hardcode dense budget3GiB. Estimate không thay allocator/RSS profiler.

Guard mặc định RSS limit=min(6GiB,1/2 RAM hiệu dụng), system headroom=min(1GiB,1/8 RAM hiệu dụng); MOE_RAM_LIMIT_BYTES/MOE_RAM_HEADROOM_BYTES override bằng byte dương. Linux đọc VmRSS/MemAvailable và cgroup-v2 memory.max/current nếu nhìn thấy; ancestor constraints ngoài namespace/cgroup-v1 hoặc thiếu /proc chưa được bao phủ đầy đủ. Không có telemetry thì cảnh báo, không tuyên bố bảo vệ toàn nền tảng.

Generation kiểm reserve trước tokenization/clone/scratch, sau đó lấy mẫu tối đa mỗi25ms tại cancellation boundaries (cancel vẫn kiểm mọi lần). Áp lực RAM trả OutOfMemory: trước SSE là503, sau ready là generation_error; worker bỏ session/cache/recycled host sau khi ngừng dùng. Giữ dense và VRAM cache, không force allocator trả trang; retry chỉ thành công nếu headroom thực đủ. Guard là kiểm hợp tác, có thể overshoot giữa samples/cấp phát/driver khác, không phải hard RSS cap và chưa chứng minh hết leak/OOM. `logits()` chẩn đoán trực tiếp chưa dùng wrapper generation này.

Status bổ sung limit/headroom, session capacity/budget, VmSwap và MemAvailable; khi model bận status có thể là snapshot cũ. Đo ổn định dùng sampler /proc độc lập và watchdog, không dựa vào polling status như peak RSS.

### Vòng đời bộ nhớ request (rà tĩnh 04/11/2026)

| Vùng | Ownership/lifetime xác định từ mã | Giới hạn accounting |
| --- | --- | --- |
| KV dùng lại đúng session | take_prefix_kv remove state khỏi sessions, truncate length nhưng giữ capacity rồi chuyển KV vào request | Khi đang chạy, state không còn nằm trong tổng session status |
| KV phiên mới | Cấp theo context_limit sau lookup không dùng lại được; các session khác vẫn có thể còn sống | KV request có thể cùng tồn tại KV các phiên được giữ; admission/eviction diễn ra sau generation |
| Admission session | KV chuyển vào SessionState; evict session cũ tới khi vừa budget, rồi insert. State quá budget bị drop | Tổng cuối request không phải peak trong request, chưa gồm HashMap buckets/allocator |
| StepScratch và buffer prefill | StepScratch local trong generate_inner; batch Q/K/V, contributions và các buffer prefill local trong prefill | Drop khi thoát scope thành công/lỗi, không nằm trong session capacity |
| Tokenizer phục vụ SSE | Bản clone và decoder local theo request streaming | Chưa có allocation trace cho nội bộ clone; không suy lượng RSS từ tên thao tác |
| Lỗi OutOfMemory | Sau khi generate_inner trả lỗi và local đã ra scope, wrapper xóa sessions/cache/recycled host | Drop trả ownership cho allocator, không bảo đảm RSS giảm hoặc OS thu hồi ngay |

Đây là rà source, chưa đo byte cấp phát/giải phóng theo mốc hoặc chứng minh nguyên nhân RSS tăng. Muốn tách phần live allocation với allocator giữ trang cần quan sát riêng sau lookup KV, sau prefill, trước/sau session admission và sau generate_inner trả về; không suy ra từ một snapshot cuối request hoặc thêm malloc_trim như bản vá khi chưa có bằng chứng.

Runner RAM/HTTP hiện ghi telemetry reclaim/swap/PSI và cgroup của child trong `memory.jsonl`, không đổi guard engine. Schema, đơn vị và phạm vi namespace ở [SCENARIOS3.4](../benchmarks/SCENARIOS.md#34-đọc-telemetry-reclaim-mới-04112026). Dữ liệu mới hỗ trợ phân biệt reclaim với allocation; chưa xác định tác nhân swap của lượt cũ.

### Trace capacity theo mốc request (05/11/2026)

`MOE_MEMORY_TRACE=on|off` chọn tại load, mặc định off; giá trị sai báo lỗi. Khi tắt không quét capacity hoặc đọc RSS tại các mốc này. Khi bật, metrics của Generation thành công có `memory_trace`; `/experts` có `memory_trace_enabled` và `last_memory_trace` của request đã kết thúc (kể cả lỗi). Status khi busy vẫn có thể là snapshot cũ. Reset xóa last trace, giữ chế độ và ngân sách như trước. Trace chỉ giữ số liệu, không giữ tham chiếu/ownership của KV, scratch hay expert.

Bảy mốc thành công: before_request, after_kv_lookup (trước cấp StepScratch), prefill_buffers_live (các buffer chính đã cấp, trước vòng layer), after_prefill (buffer local đã ra scope), before_session_admission, after_session_admission và after_generate_inner (mọi local của inner đã ra scope). Lỗi có thể dừng sớm; OutOfMemory thêm after_error_cleanup sau khi xóa host cache/session/recycled. Không tạo Generation thành công để biểu diễn lỗi; trace lỗi đọc qua status. Đây là các boundary, không phải peak allocator hoặc mọi thời điểm ở từng layer/expert.

Mỗi sample ghi elapsed_ms tính từ wrapper, RSS/VmSwap byte (null nếu không đọc được), dense decoded weight payload, expert cache/recycled capacity, retained session capacity, active KV (keys/values và backing Vec<Kv>), StepScratch và các vector prefill chính. `retained_kv_capacity_bytes` là phần nằm trong retained_session_capacity_bytes, chỉ dùng phân tích ownership, không cộng lại vào `tracked_host_bytes`. KV đã chuyển vào session hoặc đã bị drop không còn tính ở active; sessionless KV vẫn active sau admission cho tới khi inner kết thúc.

`tracked_host_bytes` cộng dense payload và các capacity được liệt kê, không phải RSS/budget mới/hard cap. Chưa bao phủ tokenizer/clone/encoding, request token/text/output, container metadata ngoài accounting session hiện có, allocator bookkeeping/trang giữ lại, trace tự thân, index/heat/transitions hoặc GPU/driver/pinned và mọi expert tạm giữa các mốc. Prefill đếm batch/hidden/QKV/norm/context/projected/contributions và expert_uses, chưa bao phủ mọi allocation sâu bên trong loader/kernel. Không lấy RSS trừ tổng này để kết luận leak hoặc gán riêng vùng chưa tính. Không đổi guard/admission, thứ tự tính toán, routing/reduction hay chính sách cache.

### Candidate tái sử dụng buffer KV (05/11/2026)

`MOE_KV_BUFFER_REUSE=on|off` chọn tại load, mặc định off, giá trị sai bị từ chối. Chỉ điều khiển tái sử dụng vùng cấp phát khi không có prefix hợp lệ; valid prefix KV vẫn được dùng lại ở cả hai mode. Đường mặc định/fallback giữ thời điểm cấp KV/admission như bảng lifetime phía trên.

Khi bật, cùng session nhưng prefix0 có thể clear length keys/values và dùng lại buffers của chính nó; không dùng nội dung cũ làm KV hợp lệ. Với ID mới, candidate chỉ lấy từ session LRU khi:

- Ngay cả lower bound state mới cộng phần đang giữ cũng vượt session budget; không lấy cache khi còn chỗ hoặc request không có session.
- Donor có đúng số layer và từng buffer đủ capacity cho context hiện tại; không mượn donor thiếu capacity.
- Metadata chuẩn bị của ID/token buffer và KV donor tính chính xác vẫn vừa budget sau khi loại đúng một donor. Token buffer được reserve trước theo giới hạn sinh, tránh tăng capacity làm admission vượt dự tính; nếu EOS sớm, capacity token này có thể lớn hơn đường cũ và vẫn được tính đầy đủ vào budget. Nếu cần hơn một evict hoặc donor quá lớn thì fallback, không tạo pool ngoài ngân sách.

Map bỏ ID cũ trước khi clear/ghi đè; KV chuyển sang ownership của request, không có alias với session cũ. Clear length không zero toàn bộ capacity nhưng mọi vị trí sử dụng được ghi lại trước attention; prefix0 bảo đảm recompute từ đầu. Các session khác giữ nguyên; admission cuối vẫn kiểm budget thực và dùng LRU như cũ. Không đổi weights/router/reduction, không thêm unsafe/dependency hoặc nới guard.

Evict sớm có tác động có chủ đích: nếu request mới hủy/lỗi sau handoff, ID cũ đã mất KV cache và không được phục hồi bằng buffer đã ghi đè. Retry phiên cũ phải recompute từ lịch sử client gửi đầy đủ. Buffer request ra scope được drop; OOM vẫn áp cleanup host hiện hành. Đây là thay đổi trạng thái cache, không thay nội dung hội thoại hoặc dùng chéo KV. Candidate còn off mặc định để đối chứng trước khi chốt cấu hình vận hành.

Metrics có `kv_buffer_reuse`: enabled, reused_capacity_bytes (KV và backing Vec<Kv>), early_evicted_sessions, source (`same_session_reset`/`evicted_session`/null). Prefix hợp lệ chỉ báo qua cached_prompt_tokens như trước, không tăng counter candidate. Status có kv_buffer_reuse_enabled và last_kv_buffer_reuse của request kết thúc kể cả lỗi; reset xóa last counters, giữ mode. Các counter không chứa ID hoặc địa chỉ buffer. Fixture MHA/GQA kiểm pointer reuse, logits tolerance2e-5, budget/LRU và hủy/lỗi/retry. Cặp model thật05/11 đã xác nhận handoff và capacityKV256→128MiB/outputHTTP2/2khớp; RSS có tín hiệu thuận lợi nhưng một cặp chịu pressure khác, chưa chứng minh mức giảm tổng quát/hếtOOM hoặc speedup.


## Điều khiển và đo lookahead (03/11/2026)

`MOE_LOOKAHEAD_PREFETCH=on|off` chọn tại load, mặc định on, giá trị khác báo lỗi. Chỉ điều khiển dự đoán expert lớp kế tiếp ở decode; giữ cập nhật transition/heat, nạp theo nhu cầu và nạp trước expert đã biết trong prefill. `/experts` và metrics báo `lookahead_enabled`; reset không đổi chế độ.

Metrics request bổ sung `lookahead` với đề xuất/tải thành công/lỗi/được chọn/không dùng/bỏ khi lỗi/còn chờ, payload byte thành công và capacity buffer đỉnh. Payload còn chờ bị drop khi step lỗi, ghi cảnh báo và accounting abandoned trong profile; HTTP lỗi không trả một Generation thành công có metrics. `decode_expert_active_peak_bytes`/`decode_expert_host_peak_bytes` là mẫu capacity payload tại mốc decode, host gồm cache+recycled và active; không gồm prefill/dense/KV/scratch/pinned/metadata/allocator, không thay RSS hay hard cap. Budget/dự phòng không tăng khi bật lookahead. Thiết kế/lệnh đối chứng duy nhất ở [SCENARIOS mục3.1](../benchmarks/SCENARIOS.md#31-thiết-kế-đối-chứng-prefetch-dự-đoán-03112026).


## Chọn CPU trước khởi tạo GPU (04/11/2026)

`MOE_GPU_MODE=auto|cpu` được đọc trước load, mặc định auto. API typed `OlmoeModel::load_with_gpu_mode` và `InferenceEngine::load_with_gpu_mode` cho phép caller truyền `GpuMode` trực tiếp; hai example benchmark áp lựa chọnCLI qua API này. CPU tạo tier không có driver/device/budget, bỏ probeCUDA và không tạo denseVRAM/pool/pinned. `/experts.gpu_mode_at_load` ghi lựa chọn ban đầu. GPU toggle/set không khởi tạo muộn khi đã chọnCPU; cần restart bằng auto. Auto giữ đường probe/fallback hiện có.

RunnerHTTP/RAM pin MOE_GPU_MODE theo --gpu và kiểm trạng thái trước sinh. CPUmode của lượt cũ khác CPU trướcload; protocol lookahead mới làv2. Không suy bỏGPUinit đã chữa nguyênnhânswap nếu chưa kiểmRAM có giám sát.


## Lý do không kích hoạt lookahead

Metrics request có evaluated_layers/no_next_layer, histogram miss chỉ ở lớp có lớp kế tiếp, blocked_by_misses và disabled_eligible_layers. Điều kiện “không có demand miss” và cửa sổ tối đa2 không đổi. prediction_calls chỉ tăng khi thực sự gọi predictor; không chấm điểm shadow khi bị chặn/tắt. Phân loại positive candidates thành in_ram/in_vram/available theo cùng thứ tự lọc cũ; no_positive_score_calls và all_candidates_cached_calls phân biệt hai lý do không có đề xuất. Bộ đếm không thay route, cập nhật heat/transition, admission hoặc thứ tự cộng kết quả. Runner kiểm tổng nhất quán trước khi nhận request/cặp. Định nghĩa và giới hạn ở SCENARIOS3.2; lỗi/hủy giữa request không tạo metric hoàn tất hồi tố.

### Allocator và thread trong memory trace (08/11/2026)

MemoryStageSample bổ sung additive `thread_id` và `allocator`; schema_version1 và các trường capacity cũ giữ nghĩa. Chỉ thu khi memory trace bật, sau kiểm tra early-return; không thêm allocator guard/policy mới.

`thread_id` đọc Linux /proc/thread-self, null nếu thiếu/ngoàiLinux; đây là thread gọi boundary, không phải danh sách workerRayon. `allocator.status` là glibc_mallinfo2, symbol_unavailable hoặc unsupported_platform. Trường libc_version có khiLinuxGNU; preload_env_present chỉ báo biếnLD_PRELOAD hiện diện, không chứng minh activeallocator hay mọiinterposition.

TrênLinuxGNU, dlsymRTLD_DEFAULT tra mallinfo2 một lần quaOnceLock; không liên kết trực tiếp symbolGLIBC_2.33 nên libcGNU thiếuAPI có thể báo unavailable. ABIrepr(C) gồm đúng10size_t nhưmalloc.h, fn trảstruct byvalue; symbol và versionstring thuộc processlibc, không giữpointer vàoheaparena, không free/mutateallocator. Tính tương thích nền tảng khác chỉ là cfgfallback, chưa kiểm chạy trên mọiOS/libc. NgoàiGNU/Linux không gọiGNUFFI.

Allocator fields đều Option/byte: arena_bytes, in_use_bytes(uordblks), free_bytes(fordblks), mmap_bytes(hblkhd); null khiunavailable, không thay bằng0. query_ns đo thời gian chính lời gọi mallinfo2, không gồm /proc/capacityscan/serialize/dlsyminit/versionclone và toàn traceoverhead. Các phép đo này tự có allocation/overhead, không cam kết<1ms.

in_use_bytes+mmap_bytes là chỉ báo bookkeepingallocator, không exactliveRustpayload. Tcache/metadata/nonmallocmapping/interposition/cácarenađọc tuần tự giới hạn diễn giải. RSS và allocator không snapshotđồngthời, không cộng hai nguồn vào tracked_host_bytes; trackedcapacity cũ không gồm chỉbáoallocator để tránhđếmtrùng. Không gọi mallopt/malloc_trim hoặc đổi routing/session/cachebudget. Phép đo cũ không có trường mới không được điền hồi tố.
