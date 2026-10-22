# Lộ trình hiện hành

Trạng thái ngày 2026-10-22. [PROGRESS.md](PROGRESS.md) giữ lịch sử; hoàn thành triển khai không đồng nghĩa hết lỗi hoặc xác thực mọi kết luận hiệu năng.

## Tóm tắt để review — chỉ cần đọc mục này trước

**Đã hoàn tất bộ kiểm chứng và báo cáo i5 phase1-v1 trên checkpoint sạch `d4adefd`. Phần triển khai/đo i5 đã chốt; nghiệm thu đầy đủ hai testbed còn chờ i7 chạy cùng protocol. Không mở thêm tối ưu để kéo dài giai đoạn1 trên i5.**

| Câu hỏi | Tình hình hiện tại |
| --- | --- |
| Engine/web chạy được không? | Có. Bạn đã kiểm7 thao tác web đạt; Rust/HTTP/parser và targeted CUDA tests đạt trong phạm vi đã ghi. |
| RAM/OOM đã xử lý tới đâu? | Thêm guard và cache mặc định64MiB/layer. Test i5 dùng guard5GiB:24 request chia hai lượt khoảng3,5 và8,1 phút, observed RSS max3,47/3,53GiB, process swap0. Chưa chứng minh hết leak/OOM mọi workload. |
| Tốc độ bản hiện tại bao nhiêu? | Đã đo 5 mẫu/nhóm: cold Mars3,66, warm Mars3,99, conversation Mars3,68, conversation Love3,23 steps/s; bảng/raw trong báo cáo chính. Không suy speedup so số UI/config cũ. |
| Chất lượng câu trả lời? | Smoke12 câu đạt10 tiêu chí; có câu trả lời sai. Đây là heuristic, không phải đánh giá chất lượng chuẩn hóa. |
| Hai máy dùng bộ test nào? | Mục tiêu cùng revision/checkpoint/bộ test/cache để ghi báo cáo riêng i5/i7. i5 đã đo cache64MiB/layer; i7 dùng cùng source/checkpoint/protocol/cache sau preflight tại máy đó. Chưa có run i7 mới, không gọi cache64 đã được nghiệm thu trên i7. |
| Việc tiếp theo? | Chạy phần i7 độc lập từ checkpoint nguồn d4adefd, kiểm hashes/config và bổ sung báo cáo. i5 không cần chạy lại nếu engine/protocol không đổi; INT4/backend mới để giai đoạn2. |

**Đường đọc:** PLAN là trạng thái/quyết định; [báo cáo i5 chính](../benchmarks/experiments/2026-10-22-phase1-i5/README.md) là trang review duy nhất cho bộ đo cuối; báo cáo RAM cũ chỉ tra khi cần kiểm guard; [SCENARIOS](../benchmarks/SCENARIOS.md) chỉ cần khi thực sự chạy phép đo. PROGRESS là nhật ký tra cứu, không phải danh sách bắt bạn review lại mọi mục.

| Nhóm artifact đã có | Vai trò, không phải bộ báo cáo cuối |
| --- | --- |
| prefill-i5 | Thử ý tưởng prefill/KV; dispatch nhóm bị loại, lợi ích prefix chỉ áp dụng workload phù hợp. |
| web-session-i5 | Thử inclusive/victim để quyết định policy; bằng chứng tối ưu cũ, cache128MiB, không đại diện tốc độ bản cache64 hiện tại. |
| stage1-validation | Smoke chức năng/quality, có cả record script thất bại được giữ để minh bạch. |
| ram-guard-i5 | Kiểm RAM mới có watchdog; record failed của cache256 là phép từ chối mong đợi, không phải24 request RAM thất bại. |

Trong mỗi artifact, README là bản giải thích; JSON/log/time series/source ZIP/checksum là bằng chứng lưu trữ, chỉ mở khi cần kiểm một claim hoặc tái lập. Không cần duyệt từng file. Các bảng chi tiết dưới đây là backlog/handoff kỹ thuật; mục tóm tắt này phản ánh trạng thái mới nhất.

| Việc | Trạng thái | Điều kiện hoàn tất |
| --- | --- | --- |
| AI-first/workflow/bằng chứng hai máy | Hoàn thành nền tảng | Entry point chung, cấu hình agent kiểm tra được, raw duy nhất/checksum, tooling/CI/docs khớp |
| RAM preflight/guard và kiểm OOM | Đã triển khai, hai lượt i5 có giám sát đạt trong phạm vi thử | 12×32 token có reset +12×128 giữ cache, cache64MiB/guard5GiB; RSS observed max3,53GiB, VmSwap0; chưa hard cap/khẳng định hết leak hoặc thử i7 |
| R1/R2: KV capacity, pinned allocation invariant | Đã vá và kiểm fixture/CUDA thực | Session admission tính capacity/token metadata; pinned length private, getter/compile-fail và undersized fallback/logits đạt; không tuyên bố tổng RSS hoặc mọi GPU API đã an toàn |
| R3: reset và KV session | Đã vá, hồi quy fixture đạt | Reset xóa entries nhưng giữ ngân sách; session mới reuse prefix, token parity và logits fixture đạt; chưa đo tăng tốc 7B |
| Prefill: dispatch token dùng chung expert theo nhóm | Không bật; A/B i5 cho thấy chậm hơn | Candidate giữ ở cfg(test); 7/9 cặp chậm hơn, không tuyên bố tăng tốc |
| KV session: reuse prefix chung, cấp buffer sau lookup | Đã triển khai, fixture và thí nghiệm i5 đạt | Logits/tokens/ownership đạt; edited-session reuse104/116 tokens, prefill median8,847→2,870s; không suy rộng cho decode/chat thường; [bằng chứng](../benchmarks/experiments/2026-10-22-prefill-i5/README.md) |
| VRAM victim cache và HTTP/SSE Mars→Love128 | Đã triển khai, mặc định victim; đo exploratory i5 | AB/BA cùng model: wall median Mars61,23→48,51s, Love66,15→57,27s, n2/policy/prompt; text/usage khớp; GPU fixture thực đạt; [bằng chứng/giới hạn](../benchmarks/experiments/2026-10-22-web-session-i5-081202/README.md) |
| R4/R5: budget RAM tổng, SSE frontend | Vá thành phần đạt; manual web pass và guard đã thử i5 hữu hạn | Cache+recycled theo capacity cùng budget; active expert payload có trần riêng topK+2 experts, prefill prefetch tối đa topK; web chỉ lưu sau terminal+DONE; guard hợp tác/RAM test mới có giới hạn nêu ở tóm tắt, chưa hard RSS cap |
| R6/R7/R9: I/O/VRAM/routing telemetry | Nhãn demo đã phân biệt giới hạn; accounting VRAM còn mở | Expert load wait không gọi SSD wait; VRAM ghi logical thiếu pool/scratch; placement/routing ghi minh họa, bỏ random token pulses; expert_bytes_read legacy vẫn trộn |
| R8: runner/provenance không overwrite | Hoàn thành tooling, đã kiểm fixture | Unique ID, revision/hash/config/hardware, hồi quy giữ kết quả cũ và lỗi |
| Baseline phase1-v1 i5/i7 | i5 hoàn tất; i7 chờ máy đồng nghiệp | i5 source sạch d4adefd, cold/warm/conversation,5 mẫu/nhóm; i7 giữ cùng source/checkpoint/config/protocol, kiểm RAM riêng |
| Runner HTTP nhiều workload | Đã có preset và kiểm protocol fixture | Cold/warm/conversation/topics/long-context/cache-sweep; preview trước đo, AB/BA mặc định, warmup tách mẫu; đã đo 7B cold/warm/conversation trên i5; các preset còn lại chưa đo |
| Chất lượng/độ ổn định model thật | Sau OOM đã có guard/test RAM hữu hạn; chưa nghiệm thu stability tổng quát | 16 request hoàn tất +1 hủy/reset/retry, input dài; quality10/12 heuristic, có swap; manual web người dùng đạt, chưa tải dài/baseline sạch |
| INT4/mở rộng backend | Chưa triển khai decoder | Sau correctness/accounting và baseline đủ tin cậy |

Phạm vi demo/nghiên cứu local, một request OLMoE, greedy decoding/context cấu hình; CUDA/cuBLAS tùy hardware. Chưa có production public/auth đa người dùng hoặc cam kết đúng mọi câu trả lời.

[Review gốc](reviews/2026-10-22-system-review.md) phản ánh commit tại lúc review; theo dõi trạng thái vá tại đây/PROGRESS. Lượt chuẩn hóa không sửa thuật toán engine.

## Checklist lịch sử trước bộ đo (được cập nhật bởi quyết định cuối trang)

1. Điều tra OOM/tổng RAM trước: xác minh cache thực và allocations theo request, chừa headroom OS/IDE, kiểm chạy dài có giám sát; không cố gây OOM để lấy bằng chứng.
2. Sau khi bộ nhớ ổn định, chốt protocol chung i5/i7: cùng revision/checkpoint, preset/prompt/history/context/token limit, cache MiB/layer tường minh, số mẫu/reset/warmup. Ghi kết quả độc lập để mô tả khả năng vận hành trên hai cấu hình, không xếp hạng CPU. Backend/threads khác phải ghi rõ; mức cache chung chưa được chốt.
3. Bộ báo cáo dùng runtime victim, ít nhất5 mẫu/cấu hình/workload, warmup tách riêng. --single-policy khác AB/BA policy experiment; không tự coi default wrapper là protocol đã nghiệm thu. Chốt giai đoạn1 sau tiêu chí bộ nhớ và bằng chứng tương ứng; INT4/backend mới để sau.

## Kế hoạch đóng giai đoạn 1 — handoff ngày 2026-10-22

Mục tiêu: chốt nền tảng engine OLMoE INT8, cache file/RAM/VRAM và web demo local có kiểm chứng. Chưa nghiệm thu giai đoạn1; không yêu cầu đạt một ngưỡng tok/s tùy ý. INT4, VNNI, Paged KV và mở rộng backend để giai đoạn2, sau khi nền tảng đạt các điều kiện dưới đây.

Thực hiện theo thứ tự, cập nhật PROGRESS sau mỗi bước:

1. **R1/R2 — bộ nhớ và an toàn pinned buffer.** Đọc review, skill `moe-engine` và ARCHITECTURE; tái hiện trigger trước sửa. Tính KV theo capacity thực, đóng khả năng phá metadata allocation từ safe API; giữ invariant ownership/lifetime/kích thước/thread/context. Nghiệm thu bằng hồi quy budget/lifetime, logits fixture giữ tolerance và kiểm GPU thực có probe cho đường liên quan.
2. **R4 — ngân sách RAM expert.** Tính cả buffer pool/recycled/inflight hoặc giới hạn riêng được công bố rõ; kiểm cả budget nhỏ/0, eviction/prefetch/reset/cancel. Phân biệt capacity với RSS; kiểm nguồn allocation và đo phù hợp trước khi tuyên bố giảm RAM/I/O.
3. **R5 và tính trung thực của demo.** Web xử lý SSE error, EOF thiếu terminal, Stop và retry; không lưu stream lỗi như thành công. R6/R7/R9: sửa nhãn payload/physical VRAM/routing hoặc hiển thị rõ giới hạn/mô phỏng trước khi demo. Đồng bộ DEMO với R3 đã vá; không để hướng dẫn lỗi thời.
4. **Checks và thử model thật.** Chạy `python3 tools/check.py --rust --release --http`; GPU probe và targeted GPU tests trên host có thiết bị. Thử model7B: chat nhiều lượt, Stop→gửi lại, reset→gửi lại, input dài trong context; giữ logs/output và theo dõi RSS/swap. Quality smoke tách riêng, không gọi là chất lượng chuẩn hóa. Báo pass/fail/skipped; fixture không chứng minh độ ổn định7B.
5. **Đo ghi báo cáo và nghiệm thu.** Sau khi kiểm RAM ổn định, dùng skill `moe-benchmark`, protocol và [runner preset](../benchmarks/SCENARIOS.md). Preview rồi chạy cold/warm/conversation, ít nhất5 mẫu/cấu hình/workload, victim với `--single-policy`, warmup không tính mẫu đo. i5/i7 dùng cùng revision/checkpoint/bộ test/cache MiB-layer/context/token limit/số lượt/reset-warmup đã chốt; ghi hardware/backend/threads thực riêng. Mục tiêu đánh giá khả năng vận hành trên hai cấu hình; không tự suy ra causal CPU comparison. AB/BA inclusive/victim là đợt tối ưu policy riêng nếu được giao. Không chạy inference/build/tests đồng thời. Giữ raw/log/hash và mọi kết quả lỗi/không cải thiện. Revision sạch cho nghiệm thu; dirty chỉ exploratory, không tự commit thay đổi người dùng để làm sạch. Threads8 chưa phải tối ưu; cache chung chưa chốt.
6. **Đóng bằng chứng và tài liệu.** Rà EVIDENCE_MATRIX, README/DEMO và bản thảo báo cáo: mỗi claim có nguồn/phạm vi, bỏ khẳng định tuyệt đối chưa chứng minh về memory safety/zero-allocation/SSD I/O/two-testbed. Giữ raw bất biến, đính chính lịch sử bằng mục mới. Ghi revision/checkpoint/config đã nghiệm thu và backlog chuyển giai đoạn2; chỉ đánh dấu hoàn tất khi các mục bắt buộc đạt, còn giới hạn phải nêu rõ.

Điểm bắt đầu handoff ban đầu là bước1; tiến độ thực hiện mới nhất ở PROGRESS. R1/R2/R4 và parser SSE đã triển khai/kiểm fixture; chưa được suy ra rằng các bước nghiệm thu còn lại đã đạt. Repo đang có nhiều thay đổi từ các lượt trước; giữ nguyên, không reset/overwrite hoặc commit/push khi chưa được giao. Kết quả gần nhất là kiểm runner fixture và preview6 preset, chưa có phép đo7B mới từ preset. Bằng chứng victim cache hiện có là exploratory i5, n2/policy/prompt; không thay thế baseline ở bước5.

### Tiến độ đóng giai đoạn1 sau thực hiện trong cùng session

- Bước1/2: R1/R2/R4 đã vá, fixture và targeted CUDA thực đạt; giới hạn accounting/RSS được công bố.
- Bước3: parser SSE và commit history sau terminal/DONE đã sửa,7 Node tests đạt; nhãn logical VRAM/overlap/routing minh họa đã bổ sung. Sau đó người dùng đã kiểm thủ công7 bước đạt chức năng; không có automation/Console log, xem mục cập nhật OOM bên dưới.
- Bước4: debug/release/HTTP fixture đạt; smoke7B i5 có16 request hoàn tất,1 hủy rồi reset/gửi lại đạt, quality heuristic10/12. Có swap và chưa tải dài. [Raw/source/log và giới hạn](../benchmarks/experiments/2026-10-22-stage1-validation/README.md).
- Bước5 **chưa chạy baseline nghiệm thu**: working tree còn nhiều thay đổi từ trước, không tự commit để làm sạch. Cần revision sạch được chốt hoặc cho phép snapshot revision riêng, rồi chạy cold/warm/conversation >=5 mẫu/cấu hình/workload theo protocol báo cáo mới; không thay bằng timing32-token của smoke.
- Bước6: đã đính chính bản thảo/EVIDENCE_MATRIX; còn quyết định nghiệm thu cuối sau xử lý OOM, kiểm RAM dài và bộ đo đã chốt. **Giai đoạn1 chưa đánh dấu hoàn tất.**

### Cập nhật kiểm web và OOM do người dùng báo — 2026-10-22

Người dùng xác nhận thủ công7 bước chat/history/Stop/retry/lỗi context/reset/reload hoạt động; không còn coi kiểm browser là chưa có bằng chứng, nhưng đây là báo cáo của người dùng, không phải automation hay xác nhận Console không lỗi. Trong cùng đợt, VS Code bị terminated. Kernel log22:05 giờ Việt Nam xác nhận OOM giết moe-tier-engine PID22818 với anon-rss10.416.220KiB. [Log](reviews/2026-10-22-i5-oom.log). Chưa nghiệm thu ổn định RAM.

Ưu tiên tiếp: xác minh cache thực của lượt lỗi (startup adaptive có thể chọn256MiB/layer, smoke trước cố định128MiB), ghi RSS/PSS/VmSwap/cache/session capacity theo request; tách cache growth, active allocations, allocator retention và leak. Rà adaptive budget có chừa dense/request/IDE/OS hay không, xây headroom và giới hạn tổng phù hợp. Chỉ chạy thử dài có giám sát với budget nhỏ sau điều tra, dừng trước áp lực RAM; không cố tái hiện OOM hoặc chạy benchmark tốc độ lúc này. Các bản vá R1/R4 chỉ giới hạn những phần đã nêu, không chứng minh tổng RSS an toàn.

### Tiếp tục xử lý OOM: bảo vệ hợp tác và kiểm có giám sát

Đã triển khai cache mặc định64MiB/layer cố định, RAM preflight trước weights và guard RSS/headroom trên generation,503 trước SSE/error sau SSE, giải phóng cache host khi guard từ chối. Chưa xác định duy nhất nguyên nhân RSS10GiB hoặc chứng minh hết OOM; chưa có hard RSS cap. Theo dõi PROGRESS cho kết quả kiểm12 request7B với watchdog. Ngân sách chung i5/i7 vẫn chưa nghiệm thu,64MiB mới là cấu hình thử/default thận trọng.

Runner mới tools/check_ram_stability.py yêu cầu testbed/backend/cache tường minh, binary đã build; không build trong phép thử. Theo dõi RSS/VmSwap/MemAvailable250ms, dừng child riêng khi mất telemetry hoặc vượt ngưỡng; giữ raw/time series/source/binary/checkpoint hashes. Không tự dừng server của người dùng. Hoàn tất checks trước chạy, không mở hai model đồng thời. Sau phép thử cần xem các request sau reset có tiếp tục tăng, phân biệt allocator retention/cache/leak; không gọi12 request là tải dài đã nghiệm thu.

Kết quả thử RAM sau guard: [artifact i5](../benchmarks/experiments/2026-10-22-ram-guard-i5/README.md) giữ3 records/source/time series/checksums. Hai lượt hoàn tất24 request, observed HWM max3,47GiB và3,53GiB, process swap0, không watchdog stop. Cấu hình cache256MiB bị từ chối trước weights trong phép kiểm âm tính có watchdog. Đây là dữ liệu exploratory hữu hạn, không chứng minh leak đã hết hoặc baseline tốc độ; RSS vẫn có tăng trong lượt giữ cache. Điểm tiếp: giữ config đã thử làm candidate, xác nhận độ ổn định dài hơn/allocator khi cần, chốt config báo cáo trên revision sạch và kiểm độc lập i7; chưa đánh dấu hoàn tất giai đoạn1. Không yêu cầu người dùng lặp lại7 bước UI đã đạt trừ khi sửa flow.

### Bước đo báo cáo tiếp theo — protocol phase1-v1

Đã chốt ứng viên i5 từ kiểm RAM: cache64MiB/layer/context512/threads8/max_tokens128/victim/guard5GiB/headroom1GiB. Một bộ cold/warm/conversation,5 mẫu mỗi nhóm; tổng20 measured requests +5 warmups, bảng cuối4 dòng. Protocol/lệnh duy nhất trong [SCENARIOS](../benchmarks/SCENARIOS.md); không mở thêm thí nghiệm tối ưu hoặc ép i7 có kết quả trước khi máy đó preflight. Runtime source không đổi trong bước này.

Runner HTTP bổ sung memory watchdog và giữ memory.jsonl/watchdog.json; cùng cơ chế/ngưỡng cho bộ báo cáo. Cần checkpoint commit nguồn sạch (workspace có cả thay đổi từ trước); đang chờ người dùng chọn cho phép checkpoint local hoặc tự commit, không tự commit/push. Trong lúc chờ chỉ chuẩn bị/check fixture/preview, không lách bằng dirty run rồi gọi baseline sạch.


### Kết quả chốt i5 phase1-v1 và việc còn lại

Bộ đo nguồn sạch `d4adefd9e53627bdc1fd250eabab436c3ac54b0d` đã hoàn tất, cùng binary/source/checkpoint hashes và config cache64MiB-layer/context512/threads8/max128/victim/guard5GiB/headroom1GiB. [Một báo cáo i5 chính](../benchmarks/experiments/2026-10-22-phase1-i5/README.md) giữ20 mẫu +5 warmup,4 nhóm mỗi nhóm5 mẫu; output/usage khớp trong nhóm. RSS/HWM quan sát cao nhất3,494GiB, process swap0, không watchdog violation. Đây là mức hiệu năng hiện tại, không A/B tăng tốc.

**Chốt phần triển khai/kiểm chứng/báo cáo i5 của giai đoạn1.** Không cần lặp7 bước web đã đạt hoặc tiếp tục tối ưu để đạt ngưỡng tok/s tùy ý. Phần nghiệm thu hai testbed còn mở duy nhất việc chạy/kiểm bộ i7 tương ứng sau preflight, rồi bổ sung bằng chứng và quyết định đóng toàn bộ giai đoạn1. Không dùng kết quả i5 thay i7. Những checklist “chưa commit/chưa baseline” phía trên mô tả thời điểm trước bộ này và đã được cập nhật bởi mục này.

Checkpoint nguồn đã push lên nhánh `phase1-report-i5`; người dùng cho phép commit/push, không merge main. Agent i7 lấy branch này nhưng tạo branch đo từ **commit nguồn d4adefd** để giữ cùng revision với i5 (commit báo cáo sau đó chỉ bổ sung tài liệu/bằng chứng). Theo SCENARIOS phase1-v1, đối chiếu checkpoint hash và settings; đổi đúng nhãn testbed, giữ cache64/context512/max128/threads8/victim/guard5GiB/headroom1GiB và5 mẫu/nhóm. Nếu máy không đáp ứng, dừng/ghi lý do; không tự giảm cache rồi coi cùng protocol.

Giới hạn chuyển tiếp: guard hợp tác không hard cap; chưa chứng minh hết leak/OOM hoặc tối ưu toàn cục; quality smoke10/12 cũ có lỗi, không phải chất lượng chuẩn hóa; file payload không bằng SSD physical bytes, timer chồng lấn và logical VRAM còn giới hạn. INT4/backend mới/quality chuẩn hóa/accounting đầy đủ là backlog sau nền tảng này, không phải thành tích đã có.
