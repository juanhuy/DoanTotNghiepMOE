# Kế hoạch hoàn thiện đồ án

Cập nhật08/11/2026 cho bản tích hợp chung i5–Huy. Báo cáo kỳ1 đã hoàn thành; **nghiệm thu tổng RAM/giai đoạn1 vẫn chưa đóng**. Nhật ký và lệnh nằm trong [PROGRESS](PROGRESS.md); tài liệu này giữ trạng thái và quyết định hiện hành.

## 1. Câu hỏi chính

**Trong ngân sách bộ nhớ cho trước, cách quản lý và di chuyển expert ảnh hưởng thế nào đến độ trễ, tốc độ sinh và lượng dữ liệu phải truyền khi suy luận OLMoE?**

Ba câu hỏi thành phần:
1. Engine có suy luận đúng trong phạm vi kiểm chứng khi chỉ giữ một phần expert trong RAM/VRAM không?
2. Trong cấu hình/ngân sách xác định, cache và nạp expert làm thay đổi lượng công việc và hiệu năng thế nào?
3. Bộ nhớ thực tế và giới hạn vận hành là gì, kể cả dense/KV/buffer tạm ngoài cache expert?

Phạm vi: inference checkpoint OLMoE mergedINT8 làm baseline; codeINT4/FP8 từ Huy đã tiếp nhận thành tính năng nghiên cứu riêng, chưa nghiệm thu lợi ích/chất lượng. Expert SwiGLU chạy CPU, dense/GPU tùy cấu hình. Chưa mở huấn luyện, model mới hoặc đa người dùng. Kết quả âm/đánh đổi cũng có giá trị, không buộc mọi cơ chế đều tăng tốc.

## 2. Kết quả hiện tại và mức chắc chắn

| Nhánh | Kết quả / bằng chứng | Giới hạn |
| --- | --- | --- |
| Vận hành kỳ1 | i5/i7 chạy cold/warm/conversation trong từng cấu hình. | Khác checkpoint, không so CPU hoặc ghép thành speedupcode. |
| Cache RAM/VRAM | Đối chứng inclusive/victim sát câu hỏi quản lý expert. | Ít mẫu, điều kiện cũ; CPU-only không kiểm lợi ích VRAM. |
| PrefixKV/dispatch | PrefixKV có lợi trong workload đã thử; dispatchcandidate chậm7/9cặp nên không chọn. | Prefix thử max_tokens1; không suy decode dài. |
| KVbufferreuse opt-in | Cặp05/11 tránh một bộKV128MiB khi buộc evict, outputHTTP2/2khớp. | Mặc định off; một cặp RSS chịu pressure khác, không claim giảm RSS tổng quát/speedup. |
| RAM12request trước allocator | Hoàn tất08/11, RSS tăng271,168MiB dù cache/session cuối gần cố định. | Chưa đủ xác định nguồn tăng chỉ bằng capacity. |
| Allocator12request | RSS+284,547MiB, indicatorin-use+mmap+0,234MiB; arenafree+205,777MiB. Admission chuyển128MiB sangfree màRSSkhônggiảm. | Ủng hộ giữ vùng đãfree, chưa thấy livegrowth tương ứng RSS lớn; không loại hết leak nhỏ/plateau/hardcap. |

Nguồn chi tiết: [Evidence Matrix](thesis/EVIDENCE_MATRIX.md), [benchmark index](../benchmarks/README.md), [allocator08/11](../benchmarks/experiments/2026-11-08-allocator-i5/README.md). Không dùng các lượt khác revision/checkpoint/cache/backend thành chuỗi tăng tốc. Tính đúng fixture khác chất lượng mô hình/ảnh hưởng lượng tử hóa chưa đánh giá chuẩn hóa.

## 3. Quyết định đối với RAM

Phép allocator có12/12request hoàn tất, sampledRSSpeak3182,594MiB, HWMmax3186,004MiB, processswapmax0,117MiB, MemAvailablemin7,752GiB. 84stage dùng glibc2.44, callerTID không đổi. Counterallocator là bookkeeping, không exactliveRustpayload; query_ns chỉ đo lời gọi, không toàn traceoverhead.

**Khép nhánh tìm “leak lớn tương ứng271MiB” ở phạm vi cửa sổ đã đo.** Bằng chứng mới nghiêng mạnh về allocator giữ vùng đãfree và trang trở thành resident. Không cần tự cài heaptrack hoặc lặp12request để đuổi mọi byte. Chỉ mở lại khi có triệu chứng mới hoặc câu hỏi quyết định thiết kế.

Cấu hình cache64MiB/layer vận hành được trong cửa sổ đã kiểm, chưa phải cache tối ưu/chung hai máy. Không nâng cache từ capacityKV tiết kiệm; không đổi OS/guard để vượt failure. Chưa tuyên bố ổn định dài hạn, không leak nhỏ hoặc hardcap tổngRAM.

## 4. Gốc chung sau tích hợp i5 và Huy

Theo yêu cầu người dùng, hợp nhất bản i5 và Huy lênmain để hai máy nhận cùng source. Gốc tiếp tục làorigin/main; nhận commit chính xác từ thông báo push rồi gitfetch/switchmain/pullff-only theoWORKFLOWS. Branchhandoff giữ snapshot trướcmerge; raw/sourceZIP cũ không đổi commit/dirtystatus hồi tố.

Bản hợp nhất giữ CPU-first/guard/trace của i5 cùng INT4/evaluation/routes/pilot/sampling/FP8/persistence của Huy. Đã xử lý accountingFP8, modechange/session và precisiondisk; kiểm debug/release/HTTP theoPROGRESS. Đây là tích hợp tínhnăng, không phép đo xác nhận mọi tối ưu đều tốt hơn.

i7 nhận mã và chạychecks/smokefixture trước. Checkpoint không nằm trongGit. Kết quải5 trướcmerge không tựlà kếtquảphiênbảnmới hoặc i7. Khi đo hai máy cần cùngrevision/checkpoint/workload/config vàrawriêng.

## 5. Bước phát triển tiếp trên i5

| Thứ tự | Việc | Điều kiện kết thúc |
| --- | --- | --- |
| 1 | Review, kiểm tra và publish bản bàn giao | Source/tooling/docs/evidence đủ trên branch; checks và checksum đạt; i7 có thể nhận đúng phiên bản. Đây là công việc tích hợp, không phải tối ưu hiệu năng mới. |
| 2 | Chọn một kết luận còn thiếu cho câu hỏi expertcache/I/O | Dùng Evidence Matrix: nếu dữ liệu cũ đủ cho kết luận hẹp thì viết; nếu cần kết luận trên phiên bản hiện tại thì chốt đúng một đối chứng tương ứng. |
| 3 | Thực hiện đối chứng đã chọn trong phạm vi an toàn | Chốt số mẫu/thứ tự/prompt/history/cache/backend trước tải; chỉ thay một yếu tố. Giữ tính đúng, đo RAM và thời gian thực tế; báo cả kết quả âm. |
| 4 | Viết kết quả/giới hạn và quyết định giữ hoặc bỏ | Một chuỗi câu hỏi→cơ chế→điều kiện→từngmẫu→kếtluận→giớihạn, không kéo dài đo chỉ để tìm phần trăm đẹp. |

Backlog để lựa chọn, chưa tự triển khai cùng lúc:
- **Vòng đời buffer expert:** chỉ làm nếu có luận điểm giảm cấp phát trong cùng ngân sách; phải tính mọi bufferpool/cache/inflight, kiểm lỗi/parity và đo. Giả thuyết churn chưa tự đủ để chọn poolpatch.
- **RAM cache–payload–độ trễ:** số ít mức cache đã dự toán/kiểm; giữ KV/prefetch/code cố định. Mức cache thấp cũng cần activepayload ngoài cache; chưa chọn mức hoặc chạy sweep.
- **Prefetch expert đã biết ở prefill:** cần công tắc riêng nếu đối chứng; khác predictor lớp kế tiếp. Không đổi cùng cache/pool.
- **Victim/inclusive trên GPU:** cần probe và VRAM thực hoạt động/ngân sách công bố, không dùng CPU-only để so policyVRAM.

Lookahead tiếp tục tạm gác; không đặt việc làm predictor hoạt động thành điều kiện hoàn thành đồ án. KV là cơ chế hỗ trợ ngân sách tổng, không nhánh tối ưu vô hạn.

## 6. Bàn giao cho Claude: phân tích RAM đã có kết quả mới

Giữ mục này làm điểm đọc cho công cụ phân tích độc lập. Đọc AGENTS và hai skillmoe-engine/moe-benchmark, [SYSTEM_GUIDE](SYSTEM_GUIDE.md), [ARCHITECTURE](ARCHITECTURE.md), [RAM_BUDGET](RAM_BUDGET.md), rồi hai hồ sơ:
- [RAM trước allocator](../benchmarks/experiments/2026-11-08-ram-budget-i5/README.md):271MiB và bảngstage.
- [Allocator model thật](../benchmarks/experiments/2026-11-08-allocator-i5/README.md):284,547MiB RSS,0,234MiB indicator và free sauadmission.

Raw mỗi experiment nằm trong run/: result/turn/memory/sourceZIP/log, checksum ở thư mục cha. Có thể kiểm các hàm generate_inner/prefill/sessionadmission/record_memory, prefetch_missing_experts/acquire_expert_tiered/admit_expert_l2/recycle trong src/olmoe.rs; load_reusing trong src/int8_expert.rs; src/ram.rs cho allocatorFFI; src/engine.rs cho workerlifetime.

Nếu review tiếp, trả findings theo file/hàm và mức bằng chứng; không mặc định leak hoặc exactliveheap, không lấy VmData−RssAnon làm sốheapchưachạm. Nếu chưa đủ phân biệt một câu hỏi mới, đề xuất một phép kiểm có kết quả dự đoán khác nhau. Không tự lặp cuộc đo đã hoàn tất hoặc sửa raw.

Prompt hiện hành:

> Đọc AGENTS.md và docs/PLAN.md. Điều tra RSS hiện đã có allocator12request: RSS tăng284,547MiB nhưng in-use+mmap tăng0,234MiB; admission chuyển128MiB sangfree màRSSkhônggiảm. Đánh giá mức chắc chắn/giới hạn từ raw và source snapshot, không lặp phân tích như thể chưa có telemetry. Nếu đề xuất tối ưu tiếp, gắn nó với câu hỏi quản lý expert trong ngân sách cố định, nêu điều kiện giữ/bỏ; trước mắt chỉ review, chưa tự sửa engine hoặc chạy model.

## 7. Quy tắc tiếp tục

Không tự commit/push/merge công việc ngoài phạm vi được người dùng giao. Yêu cầu hiện tại đã cho phép publish bản bàn giao này, không mặc định cho phép merge branch khác.

Giữ raw/checksum bất biến; mọi chỉnh chính ở diễn giải, không sửa giá trị gốc. Hoàn tất checks/build trước benchmark; một workload đo trên một máy mỗi lúc. Ghi PROGRESS sau bước có ý nghĩa. [WORKFLOWS](WORKFLOWS.md) giữ thao tác, không tạo thêm statusreport trùng nội dung.
