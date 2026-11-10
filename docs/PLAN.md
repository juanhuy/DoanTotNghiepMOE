# [HIỆN HÀNH] Kế hoạch hoàn thiện và kiểm chứng đồ án

Cập nhật 10/11/2026. Đây là kế hoạch duy nhất để tiếp tục công việc; lịch sử nằm trong [PROGRESS](PROGRESS.md), kết luận có nguồn trong [EVIDENCE_MATRIX](thesis/EVIDENCE_MATRIX.md). Không tạo thêm status/report bàn giao song song.

**Việc tiếp theo: S3 trên i5.** S0/S1/S2 đã xong ngày10/11: gate an toàn RAM S1 đạt trong cửa sổ24request ([P1](../benchmarks/experiments/2026-11-10-ram-window-i5/README.md)), chưa hard-cap/plateau dài hạn; S2 batch prefill đạt primary ở prompt64/128token ([P1](../benchmarks/experiments/2026-11-10-expert-batch-long-i5/README.md)), giữ opt-in. S3 bắt đầu bằng đọc P1 inclusive/victim và kiểm fixture tiering, chưa đo. Chưa chuyển nghiên cứu sang i7, chưa mở model16B hoặc thay precision.

## Bắt đầu ở chat mới — đọc phần này trước

1. Đọc AGENTS.md, phần quy tắc và mục mới nhất của PROGRESS; không đọc toàn bộ nhật ký.
2. Đọc kế hoạch này, sau đó chỉ đọc mục P1/P2 liên quan trong [mục lục benchmark](../benchmarks/README.md). Hiểu hệ thống qua [SYSTEM_GUIDE](SYSTEM_GUIDE.md); sửa engine thì đọc [ARCHITECTURE](ARCHITECTURE.md) và skill moe-engine; đo/diễn giải thì đọc skill moe-benchmark.
3. Xác minh Git/môi trường trước khi hành động. Theo yêu cầu ngày10/11, code/tools/bằng chứng mới đã commit và đưa vào main local; nhánh work/i5-gate-up-ab giữ lại. Lấy commit hiện hành bằng git rev-parse HEAD, không dùng mốc5663c7a cũ làm nguồn mới. Chưa push origin/main trong tác vụ commit này.
4. Bắt đầu bằng git status. Nếu phát sinh thay đổi mới thì giữ nguyên, không reset/clean/stash hoặc đổi nhánh/pull để làm sạch. Main local hiện đã chứa phần mới; không mặc định origin/main hoặc máy i7 đã nhận các commit đó.
5. Tiếp tục đúng bước đang mở. Không chạy lại benchmark chỉ vì chuyển chat, không tự bật các candidate đã loại.

### Trạng thái cần nhớ

- Prototype OLMoE INT8/API/UI/session/cache/tiering đã có. Expert SwiGLU tính CPU; CUDA/cuBLAS hỗ trợ phần dense và tầng lưu expert VRAM, tùy phần cứng.
- Kernel nhiều token đã có trong code, CPU/INT8 opt-in MOE_EXPERT_BATCH_PREFILL=on, mặc định off. Prompt ngắn09/11 TTFT không đạt; S2 10/11 prompt64/128token đạt TTFT/prefill≥1,05 ở4/4 so sánh (max1, một ABBA). Không tự bật mặc định; chưa kiểm decode/i7.
- Nghiệm thu tổng RAM/giai đoạn1 vẫn mở. Allocator giữ vùng đã free là bằng chứng chính của tăng RSS đã điều tra; chưa chứng minh hard cap, plateau dài hạn hoặc hết leak nhỏ. S1 10/11: 24/24request, RSS max3188,6MiB, spread8request cuối14MiB, VmSwap engine0; swap-out hệ thống do process khác vẫn xảy ra.
- Kỳ1 i5/i7 dùng checkpoint khác nhau. Dữ liệu cũ không tự là số đo của revision hiện tại hoặc đối chứng CPU hai máy.
- Kiểm cuối source hiện có ngày09/11:133Rust tests/profile,8ignored/profile,87Python,7Node; HTTP fixture bật batch30vòng=20completed/10cancelled.434payload checksum. Đây là kết quả cũ có nguồn, không phải kiểm mới ngày10/11.

## 1. Sáu mục tiêu và phần cần nghiệm thu

| Mục tiêu của đồ án | Đã có | Còn phải kiểm/chốt |
| --- | --- | --- |
| Prototype MoE trên phần cứng hạn chế | Suy luận INT8, API/UI, streaming, session, CPU/GPU tùy cấu hình | Một cấu hình vận hành hữu hạn, nguồn truy được; không tuyên bố production-ready |
| Phân bổ/di chuyển expert VRAM–RAM–SSD | Đọc file vào RAM, upload/download VRAM, inclusive/victim | Xác nhận đường GPU thực trên source được chọn; SSD là checkpoint gốc, không phải cache ghi trả động; expert vẫn tính CPU |
| Expert cache cho expert thường dùng | Budget, heat/recency/protected/pin và eviction | Đối chứng hữu hạn cùng budget/workload; không cần tìm policy tối ưu toàn cục |
| Prefetch dựa trên router | Prefill nạp expert đã biết; lookahead dự đoán lớp kế tiếp khi decode | Kiểm chức năng/counter/gate; chỉ kết luận speedup nếu cơ chế thực sự được gọi và có đối chứng |
| Theo dõi/quản lý tài nguyên | RSS/HWM/swap/available, capacity, guard, I/O counters, trạng thái VRAM | Nhãn/đơn vị/coverage đúng; logical VRAM khác physical, payload file khác I/O thiết bị, guard khác hard cap |
| Thực nghiệm các chiến lược | Có công cụ và P1/P2/P3, số đo TTFT/token rate/I/O/cache/memory | Chọn bộ bằng chứng cuối thống nhất điều kiện, công bố kết quả âm và giới hạn; hoàn thiện báo cáo |

Một chiến lược không có lợi vẫn trả lời mục tiêu thực nghiệm nếu phép đối chứng đúng. Không bắt mọi hướng tối ưu phải thành công để hoàn thiện đồ án.

## 2. Bằng chứng giữ lại và quyết định đã khép

| Hồ sơ cần đọc khi liên quan | Kết luận được giữ | Không được suy ra |
| --- | --- | --- |
| [Batch expert09/11 — P2](../benchmarks/experiments/2026-11-09-expert-batch-int8-i5/README.md) | Kernel weights thật/resident batch2/batch4:1,410×/1,381×; model expert_compute thấp hơn17,5–29,2% trong4so sánh; buffer local144KiB đã tính vào budget | TTFT chỉ tốt hơn3/4; không đạt rule cũ, không claim chat/decode/i7 nhanh tương ứng. Model thật mới max1 |
| [Allocator08/11 — P1](../benchmarks/experiments/2026-11-08-allocator-i5/README.md) | RSS+284,547MiB, indicator in-use+mmap+0,234MiB; phù hợp allocator giữ vùng free | Không exact live heap, không loại hết leak nhỏ/hardcap. Không mở lại tìm leak lớn271MiB nếu chưa có triệu chứng mới |
| [RAM08/11 — P1](../benchmarks/experiments/2026-11-08-ram-budget-i5/README.md) |12request hoàn tất trong cấu hình đã đo; cache budget khác RSS | Không phải ổn định dài hạn hoặc baseline của mọi revision |
| [KV buffer reuse — P2](../benchmarks/experiments/2026-11-05-kv-reuse-narrow-i5/README.md) | Một cặp tránh overlap KV khoảng256→128MiB, output2/2khớp; mặc định off | Không chứng minh giảm RSS tổng quát hoặc hết OOM |
| [Gate–Up resident — P2](../benchmarks/experiments/2026-11-09-gate-up-resident-int8-i5/README.md) | Không đạt primary, đã dừng nghiên cứu thêm hướng này | Không chọn scope có lợi để cứu kết quả; paired kernel đã tiếp nhận vẫn trong code |
| [Heat-admission — P2](../benchmarks/experiments/2026-11-09-cache-admission-int8-i5/README.md) | Loads/payload tăng1,53%/2,56%, candidate đã gỡ | Không bật lại MOE_CACHE_ADMISSION/CLI cũ; source thử chỉ trong ZIP |

Đối chứng cache VRAM, KV prefix và vận hành kỳ1 nằm trong P1. Chỉ mở README liên quan trước; cần kiểm một claim mới đọc đúng run/trường raw. Không đọc lặp result chứa turns và các turnfile tương ứng.

## 3. Thứ tự việc cần làm

S0/S1/S2 có kết quả ngày10/11 (xem PROGRESS); S3–S6 chưa có kết quả. Mã S1 không đồng nghĩa giai đoạn1 của đồ án đã được đóng.

| Bước | Việc | Điều kiện chuyển bước |
| --- | --- | --- |
| **S0 — xong10/11** | Rà nguồn local, chọn snapshot/cấu hình và chuẩn bị phép kiểm | Nguồn/binary/checks truy được, giữ dirty changes, protocol và ngưỡng ghi rõ |
| **S1 — gate đạt10/11** | Một cửa sổ vận hành i5 có giám sát | Đạt gate an toàn hữu hạn; nếu không đạt thì ghi failure/xử lý trigger trước phép đo model nặng |
| **S2 — đạt10/11** | Đánh giá batch2/4 trên prompt dài được chốt trước | Có parity, counters, TTFT/compute/RAM và quyết định phạm vi sử dụng; không cần ép kết quả dương |
| **S3 — đang mở** | Hoàn thiện bằng chứng cache và VRAM thực | Có đối chứng áp dụng được hoặc công bố phần chưa kiểm; CPU-only không xác nhận lợi ích VRAM |
| **S4 — prefetch** | Kiểm coverage router-prefetch/lookahead | Cơ chế/counter đúng; giới hạn cơ hội kích hoạt được định lượng, không bắt predictor phải tăng tốc |
| **S5 — bộ kết quả cuối** | Bổ sung đúng các metric/kịch bản còn thiếu | Đủ số liệu cho sáu mục tiêu, không chạy lại toàn bộ lịch sử/cross-product |
| **S6 — báo cáo/bàn giao** | Tổng hợp luận điểm, giới hạn, tái lập và demo | Mọi kết luận có nguồn; công việc i7/publish có nhiệm vụ riêng, không suy diễn đã xong |

### S0. Xác minh nguồn và đóng băng điều kiện

- Kiểm Git branch/HEAD/diff/untracked, compiler/binary/source hash, checkpoint/tokenizer hash và side state. Không bỏ thay đổi chưa lưu, không tự commit/push/merge chỉ để hết dirty.
- Source main local mới có src/int8_expert/batch_prototype.rs và các cờ batch. Raw09/11 vẫn ghi HEAD5663c7a/dirty của lúc đo; không sửa hồi tố. Dùng source hash/ZIP khi kiểm phạm vi áp dụng số đo.
- Nếu source/config/toolchain không đổi và kiểm được fingerprint với hồ sơ09/11 thì dùng lại kết quả checks phù hợp, ghi nguồn. Khi có sửa engine: python3 tools/check.py --rust --release --web; thêm HTTP/CUDA/converter theo đường sửa, công bố skipped.
- Chọn một nguồn cố định trước đo, build trước timing. Normal baseline cần revision sạch; dirty run chỉ exploratory, lưu fullsource ZIP/hash/binary và không sửa nhãn dirty/commit hồi tố.
- Kiểm dùng cùng checkpoint INT8, FP32KV, greedy; không usage side state/disk session ẩn. Không tự xóa dữ liệu của người dùng để vượt kiểm môi trường.
- Đầu ra: trạng thái nguồn, checks reuse/new, config/protocol có tên trong SCENARIOS, dự toán RAM và preview. Chưa có số đo thì ghi “chưa đo”.

Cấu hình i5 đã khảo sát dùng làm điểm xuất phát: CPU4, victim, cache64MiB/layer, context limit512, per-token Gate–Up paired, lookahead decode off, KV buffer reuse off, FP32KV, temperature0/top-p1/top-k0/repetition1, hotpin0. Batch off cho baseline. Đây chưa phải cache chung/nghiệm thu của i5/i7.

### S1. Kiểm RAM hữu hạn trước phép đo model mới

**Đề xuất phải chốt vào SCENARIOS trước chạy, chưa thực hiện:** một process,24request theo workload mixed hiện có, max64, context512/cache64/CPU4, reset đầu lượt rồi keepcache, memory trace on; batch off và lookahead off. Không thay cùng lúc pool/cache/allocator/KV.

Ngưỡng làm việc đề xuất cho cửa sổ này:
- 24/24request hoàn tất; không OutOfMemory, guard từ chối hoặc watchdog violation.
- Sampled RSS và VmHWM không vượt5120MiB; watchdog giữ5632MiB, system headroom1024MiB, minavailable1536MiB, process swap threshold128MiB. Không nới ngưỡng để pass. Nếu có swap tiến trình phải báo max, không gọi kết quả “không swap”.
- Báo RSS từng request và spread max−min ở8request cuối;64MiB là ngưỡng phân loại độ phẳng đề xuất, phải chốt trước đo. Đây là chỉ báo bổ sung, không thay gate an toàn ở hai dòng trên. Vượt ngưỡng thì chưa kết luận phẳng, không tự kéo thêm request tới khi đẹp.
- Đủ trace/capacity và raw để phân biệt bộ đệm còn giữ, live indicator và RSS. RSS tăng đầu lượt không tự là leak.

Preflight3mẫu cách5s: không build/tests/inference khác, available đủ, pswpout không tăng. Giữ guard/watchdog; pressure môi trường thì hoãn hoặc ghi failure, không sửa OS/allocator.

Preview tham khảo sau S0, không phải lệnh đã chạy:

    python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --threads 4 \
      --cache 64 --context 512 --cycles 24 --max-tokens 64 --workload mixed --keep-cache \
      --lookahead off --memory-trace on --kv-buffer-reuse off --expert-batch-prefill off \
      --gate-up-mode paired --ram-limit-mib 5120 --headroom-mib 1024 \
      --stop-rss-mib 5632 --min-available-mib 1536 --stop-swap-mib 128 --dry-run

Dirty tree cần công bố exploratory và dùng --allow-dirty nếu thực hiện khảo sát; không gọi đó là baseline sạch. Đạt gate an toàn S1 cho phép S2 có giám sát dù chưa chứng minh plateau; phải công bố độ phẳng còn thiếu. Chỉ được viết “vận hành được trong cấu hình/cửa sổ đã thử”, không “hard cap/không leak/ổn định vô hạn”. Chưa tự đóng toàn bộ giai đoạn1.

Nếu không đạt: dừng, ghi trigger/mốc/raw; chỉ chọn một bản vá có căn cứ. Logic bug cần hồi quy trước/sau; patch allocation/I/O cần đối chứng cùng điều kiện. Không cài profiler/trim/đổi allocator hoặc lặp điều tra08/11 nếu dữ liệu mới không đặt câu hỏi khác.

### S2. Kiểm batch prefill ở đúng workload dài

Chỉ sau gate an toàn S1. Giữ context limit512 để KV capacity theo cấu hình không đổi; chọn hai prompt có tổng số token sau chat template/tokenizer gần64 và128. Lưu chính xác text/history/token count/hash; không dùng số từ làm số token.

**Thiết kế đề xuất, chưa chốt protocol/chưa chạy:**
- Một ABBA off→on→on→off;4process mới, mỗi process hai prompt đã đóng băng.
- Trước đo, mỗi process warmup một lượt cho từng prompt; thu riêng, không tính vào sample. Sau đó đo một request/prompt:8sample chính +8warmup, hai cặp mỗi prompt. Cùng reset/keepcache/session mode hai phía.
- Max1 để đo TTFT/prefill; cùng nguồn/checkpoint/CPU4/cache64/context512/sampling/guard. Chỉ khác batch flag, buffer thêm144KiB phải tính vào budget/trace.
- Text/usage/finish và cache/fileworkcounters phải khớp; on có batchcalls, off0; fixture logits/token/prefix/cancel/trace đạt trước model. HTTP text parity không thay logits parity model thật.
- Primary đề xuất: TTFT và prefill off/on≥1,05 ở từng prompt trong cảhai cặp. Ghi mọi sample và compute/I/O/clientwall/RSS/pressure. Không đổi ngưỡng hay warmup sau khi thấy kết quả.
- Một lượt có kế hoạch. Không đạt thì giữ kết luận compute hẹp và off mặc định; không discard mẫu hay rerun để tìm phần trăm. Đạt cũng chỉ kết luận trong workload đã thử, chưa tự bật mặc định cho mọi máy.

**Tooling chưa hỗ trợ trực tiếp bộ prompt này:** --expert-batch-model-ab hiện cố định Mars/Python ngắn/max1, không nhận suite tùy ý. Cần mở rộng harness dùng lại code hiện có với workload/protocol mới và tests; giữ protocol cũ nguyên nghĩa. Không sửa text trong wrapper rồi giả như cùng thí nghiệm cũ.

Nếu muốn kết luận về decode/tokens-per-second, thiết kế phép max64 riêng ở S5; không suy từ max1.

### S3. Chốt cache và ba tầng lưu trữ

- Đọc P1 inclusive/victim và vận hành cũ trước. Dùng được kết luận hẹp thì trích đúng revision/config; không bắt chạy lại chỉ vì có nhiều raw.
- Với luận điểm source hiện tại, xác nhận fixture upload/download/eviction/reset, capacity cache+pool+inflight và lỗi/retry. GPU tests pass nhưng probe không chạy CUDA phải ghi skipped.
- Khi GPU có thật và nguồn đã chọn: công bố backend/threads, VRAM budget và allocation pool/staging/dense; đối chiếu output/tolerance. Chỉ sau đó chọn một đối chứng inclusive/victim cùng checkpoint/budget/workload; không tăng RAM/VRAM đồng thời đổi policy. Sơ đồ placement trên UI không thay counter/trace chuyển expert thực.
- Đầu ra tối thiểu: policy, cache hit/miss, số tải/filepayload/PCIe riêng, TTFT/tokenrate, peakmemory và giới hạn. Không lấy CPU-only làm lợi ích VRAM hoặc logical allocation làm physical.
- GPU không khả dụng thì giữ phần đã kiểm CPU và thiếu kiểm GPU tường minh. Chưa giao thí nghiệm cho i7 trong bước hiện tại; xem WORKFLOWS khi có nhiệm vụ bàn giao.

### S4. Kiểm prefetch theo đúng hai cơ chế

1. Prefill nạp trước expert mà router đã chọn: là công việc đã biết, không dự đoán token kế tiếp.
2. Decode lookahead: ước lượng expert lớp kế tiếp; gate hiện chỉ cho cơ hội khi lớp hiện tại không miss. --lookahead off không tắt prefetch đã biết của prefill.

Ưu tiên fixture/hồi quy existing: dữ liệu nạp đúng, output đúng, lỗi nạp xử lý được, counter loaded/used/unused/abandoned/pending và budget cân bằng. Sau đó tra coverage ở workload RAM cố định.

Chỉ so hiệu năng khi có đủ cơ hội thực gọi cơ chế. Nếu gate gần như luôn chặn thì báo số eligible/blocked/calls và giới hạn, không tăng cache hoặc nới gate chỉ để tạo số đẹp. Kết luận “chưa chứng minh lợi ích ở cấu hình này” vẫn hợp lệ cho báo cáo.

Đối chứng demand-only với prefetch đã biết cần công tắc riêng nếu chưa có; không giả cờ lookahead làm việc đó. Không bắt mở lại predictor thành điều kiện hoàn thành đồ án.

### S5. Chuẩn hóa metric và bộ kết quả cuối

- Dùng lại S1 cho RAM, S2 cho compute/TTFT, S3 cho cache/tiering, S4 cho coverage prefetch. Chỉ bổ sung phép còn thiếu, không chạy mọi tổ hợp model×cache×threads×precision.
- Chọn bộ nhỏ có prompt đơn, chat tiếp/prefix và chuyển phiên; giữ prompt/history/số lượt/context/tokenlimit tường minh. Decode max64 là phép riêng với TTFT max1. Giữ token IDs/usage/finish thực tế để kiểm lượng công việc.
- Mỗi so sánh chỉ thay một yếu tố. Cùng nguồn/checkpoint và backend/threads khi đánh giá code; so hai máy là phép riêng, không xếp hạng CPU từ checkpoint khác.
- Bảng kết quả phải có TTFT, tokenrate với mẫu số rõ, thời gian/load/payload expert, hit/miss và peak RAM/VRAM. Đơn vị MiB/layer và byte đúng; timer chồng lấn không cộng, filepayload không SSDphysical, engine-cold không OS-cache-cold.
- Chốt phạm vi theo dõi tài nguyên: RSS/HWM/swap/available; capacity KV/cache/pool/inflight; GPU enabled/budget và logical/physical được ghi đúng; file/PCIe/processIO phân biệt. Chỉ thêm GPU utilization/SSD device counters nếu một luận điểm cụ thể cần, không biến prototype thành bộ giám sát toàn hệ điều hành.
- Lưu raw duy nhất/checksum/source và README ngắn; xếp đúng một nhãn P1/P2/P3 trong mục lục. Kết quả âm được giữ; corpus historical_unverified không tự thành nghiệm thu mới.

### S6. Báo cáo cuối và bàn giao hai máy

- Mỗi mục tiêu có câu hỏi → cơ chế/code → protocol → bằng chứng → kết luận → giới hạn. Không dùng “có code” thay “đã chứng minh hiệu quả”.
- Nêu rõ SSD là nơi chứa checkpoint; expert compute CPU; VRAM là tầng lưu tùy hardware; tổng RAM chưa hard-cap. Không gọi số đo GPU logical là toàn bộ cấp phát vật lý.
- Báo cáo kỳ1 đã xong; bản cuối phải cập nhật source/feature và chọn bằng chứng phù hợp, không nhập mọi nhật ký/JSON. Chỉ xếp phạm vi hoàn tất khi có tiêu chí nội bộ đã chốt; phần trăm tiến độ không là điểm theo rubric giảng viên.
- i7 tiếp nhận sau khi có nhiệm vụ đồng bộ nguồn và protocol riêng, cùng checkpoint nếu so sánh. Xác nhận commit/source/hash thực trên từng máy, lưu raw riêng, đọc [WORKFLOWS](WORKFLOWS.md). Không mặc định công việc local đã ở origin/main.
- Người dùng đã yêu cầu commit vào main ngày10/11; xem Git log và PROGRESS. Push remote hoặc tích hợp nhánh khác cần nhiệm vụ tương ứng, không tự suy main local đã được publish; không force-push hoặc bỏ thay đổi mới.

## 4. Tiêu chí kết thúc một bước để không đi vòng

Một bước kết thúc bằng một trong ba trạng thái: đạt trong phạm vi đã chốt; không đạt và dừng candidate; hoặc hoãn/thiếu kiểm do điều kiện cụ thể. Ghi thay đổi, lệnh, pass/fail/skipped, raw, giới hạn và quyết định tiếp theo vào PROGRESS.

Không kéo thêm mẫu tới khi đẹp, không tiếp tục tối ưu một nhánh đã loại khi chưa có giả thuyết mới. Không cần tất cả chiến lược thắng baseline. Không đóng RAM bằng một microbenchmark hoặc khẳng định chất lượng mô hình từ vài câu text.

## 5. Prompt sao chép sang chat mới

> Tiếp tục đồ án trong workspace hiện tại. Đọc AGENTS.md, docs/PLAN.md (bản10/11/2026), quy tắc và mục mới nhất của docs/PROGRESS.md; sau đó chỉ mở hồ sơ P1/P2 liên quan.
>
> S0/S1/S2 đã xong ngày10/11 (hồ sơ P1 2026-11-10-ram-window-i5 và 2026-11-10-expert-batch-long-i5). Kiểm git status trên main local; giữ mọi thay đổi mới, không reset/clean hoặc đổi nhánh để làm sạch. Chưa mặc định origin/main/i7 đã được cập nhật.
>
> Tiếp S3: đọc P1 inclusive/victim, xác nhận fixture upload/download/eviction/reset và probe CUDA; chỉ thiết kế đối chứng GPU khi đường CUDA thực chạy. Không lặp S1/S2, không chạy S4–S6 đồng thời hoặc mở thêm hướng mới.
>
> Giữ INT8/FP32KV và ngân sách tường minh; batch off mặc định, chưa bật predictor/INT4/FP8/model16B. Không lặp các lượt09/11 để tìm phần trăm đẹp. Chỉ sửa engine khi có trigger cụ thể và kiểm chứng phù hợp. Cập nhật PROGRESS trước khi báo xong; báo kết luận và giới hạn, không đổ toàn bộ raw vào ngữ cảnh.
