# Ngân sách RAM và thiết kế phép kiểm i5

Ngày lập: 08/11/2026. Protocol dưới đây được chốt trước tải. **Đã thực thi08/11:12/12request hoàn tất trong ngưỡng nhưng RSS còn tăng271,168MiB; chưa chốt cấu hình nghiệm thu.** [Hồ sơ và raw](../benchmarks/experiments/2026-11-08-ram-budget-i5/README.md). Dữ liệu hồi cứu lấy từ [cặp KV 05/11](../benchmarks/experiments/2026-11-05-kv-reuse-narrow-i5/README.md). Câu hỏi: với cấu hình cụ thể, hệ thống dùng RAM cho những gì và có hoàn tất một cửa sổ tải định trước trong các ngưỡng đã đặt không?

## 1. Phạm vi cấu hình

Dùng checkpoint/source phải ghi hash thực tế khi thực thi; hồ sơ cũ chỉ làm cơ sở dự toán. Cấu hình được đề xuất giữ CPU trước load, 4 threads, policy victim, cache64 MiB/layer, context512, max_tokens64, memory_trace on, lookahead off, KV buffer reuse off. Giữ off để đánh giá đường vận hành mặc định hiện có; đây không phải đối chứng đánh giá lại KVreuse. Không bật adaptive cache.

Checkpoint trong hồ sơ có hidden_size2048, intermediate_size1024, 16 layer, 16 attention heads, 16 KV heads, 64 expert/layer, Top-K8. Các số này thuộc checkpoint cụ thể, không mặc định đúng cho mọi model. Nhãn i5 phải đi cùng hardware thực thu, không suy ra từ threads hoặc tên folder cũ.

## 2. Bảng ngân sách: tính được gì?

Đơn vị: MiB = 1.048.576 byte. Các dòng gồm cả mức trần và số đo minh họa; **không cộng toàn bộ cột thành RSS**.

| Thành phần | Giá trị với cấu hình trên | Căn cứ / cách dùng |
| --- | ---: | --- |
| Payload dense đã load | 1.818,508 MiB | `status_start.dense_bytes` trong [record gốc](../benchmarks/experiments/2026-11-05-kv-reuse-narrow-i5/on/result.json); không gồm metadata/allocator. |
| Ngân sách cache expert RAM | 16 × 64 = 1.024 MiB | Trần cấu hình cache+recycled theo layer; không phải trần RSS. |
| Cache expert cuối request đã quan sát | 962,5 MiB | Do độ hạt expert, không nhất thiết lấp đúng toàn bộ 1.024MiB. Đây là số hồi cứu, không ngân sách thêm vào cache. |
| Một payload expert | 6,015625 MiB | `3*d*m + 4*(2*m+d)` byte: INT8 weights và FP32 scales, d=2048/m=1024. |
| Dự phòng payload expert đang xử lý | 60,15625 MiB | `(TopK+2)*expert_bytes`, tổng10expert; không phải nhân thêm16layer. Không cộng lần nữa nếu dùng metric host đã gồm cache+active. |
| Một bộ KV request theo context | 128 MiB + backing Vec/metadata | `context * kv_dim * 8 * layers`; kv_dim = hidden/attention_heads * KV_heads =2048. Hai tensor FP32 K/V. Capacity trace thực thấy128MiB+768byte backing Vec. |
| Ngân sách retained session | 256 MiB | Session accounting gồm capacity KV/token/ID/state. Chưa gồm HashMap/allocator; một bộ KV128MiB cộng metadata làm hai bộ không vừa đúng256MiB. |
| KV active ngoài retained session | Tối đa một bộ request theo đường generation tuần tự | Có thể cùng tồn tại retained session tới admission, đặc biệt khi reuse off. Không lấy budget session256MiB làm trần mọi KV đang sống. |
| Hạng dự toán buffer theo context | 80 MiB | `context*d*4*(TopK+12)`, công thức kế hoạch của engine, không số đo chính xác mọi scratch/prefill. |
| Hạng dự phòng khi load | 512 MiB | Khoản dự toán cho transient/tokenizer/allocator; không đảm bảo đủ mọi trường hợp, không allocation thực. |
| Vùng còn phải quan sát | Chưa định lượng đầy đủ | Tokenizer/clone/text/output, index/heat/transitions, container metadata, allocator giữ trang, allocation tạm giữa mốc, thư viện/runtime; GPU/pinned/driver nếu đổi sang auto. |

Công thức tham khảo nằm trong [load/status/generation](../src/olmoe.rs) và [RAM guard](../src/ram.rs). Với GQA, KV thực dùng kv_dim; reserve trước generation hiện dùng hidden_size nên có thể bảo thủ hơn. Checkpoint đang xét có kv_dim=hidden_size, hai hạng KV bằng nhau (128MiB trước metadata).

### Dự toán engine kiểm khi load

`dense_planned + layers*cache + session256MiB + overhead512MiB + context_buffers80MiB + active_KV128MiB + expert_payload60,15625MiB`.

Nếu dense_planned bằng payload dense ghi nhận ở hồ sơ này, tổng kế hoạch là **3.878,664 MiB**. Cần tính lại từ index/checkpoint thực trước chạy. Đây là reserve đưa vào guard cùng RSS hiện có, không phải dự đoán peakRSS. Không cộng thêm cache quan sát962,5MiB vào tổng này vì đã có cache budget1.024MiB.

Trước generation, reserve cho cấu hình này là `80 +128 +60,15625 +64 =332,15625MiB`. Guard cộng reserve với RSS quan sát lúc kiểm, rồi định kỳ kiểm với forward expert reserve. Hạng64MiB thuộc reserve request, không cộng vào loadplan theo công thức mã hiện tại. Guard lấy mẫu ở cancellation boundaries khoảng25ms, không kiểm mọi allocation.

## 3. Vì sao còn phải đo RSS/swap?

Capacity là chỗ dành cho các vector, RSS là trang đang resident theo OS. Trang có thể chưa được chạm, bị swap hoặc được allocator giữ lại sau drop. Vì vậy phép tính kế hoạch không chứng minh RSS sẽ thấp hơn ngưỡng. RSS thấp khi swap tăng cũng không tự chứng minh tối ưu bộ nhớ.

Hồi cứu cặp hẹp05/11: on/off hoàn tất2request mỗi process; peakRSS lấy mẫu2893,664/2945,574MiB; processswapmax43,121/0MiB. Một lượt khác12request hoàn tất nhưng RSS tăng288,094MiB từ request đầu đến cuối. Những số này cho thấy cần kiểm cửa sổ dài hơn hai request, chưa cho phép nâng cache. Xem [hồ sơ12request](../benchmarks/experiments/2026-11-05-ram-reclaim-i5/README.md).

## 4. Phép kiểm hữu hạn được đề xuất

Phạm vi: **một process, tối đa12request**, workload `mixed` hiện có (Mars, follow-up cùng session, Python mới, Love mới) lặp3nhóm, keepcache/chỉ reset ban đầu. Mỗi request tối đa64completion; giữ context512. Lưu đầy đủ messages thực: follow-up phụ thuộc câu trả lời trước, không coi mọi nhóm là chuỗi token giống nhau. Không chạy thêm lượt tự động sau thất bại, không mở i7 hoặc sweep cache. Chưa dùng phép này để so tốc độ CPU hay tối ưu code.

Giữ guard RSS5120MiB/headroom1024MiB; watchdog RSS5632MiB/MemAvailable1536MiB/processswap128MiB. Watchdog là hàng rào ngoài engine nên cao hơn guard, không có nghĩa vùng giữa hai ngưỡng được nghiệm thu. Không sửa OS/swappiness hoặc ngưỡng để vượt failure.

Trước tải: không có build/tests/inference khác; xác nhận source/binary/checkpoint/hardware/config và kiểm tương thích telemetry. Thu3mẫu cách5giây gồm MemAvailable, process/systemswap nếu có, pswpin/out và PSI. Nếu pswpout đang tăng giữa các mẫu hoặc MemAvailable dưới1536MiB, hoãn tải và giữ preflight, không lặp chờ tìm cửa sổ đẹp trong cùng tác vụ. Preflight thuận lợi không bảo đảm tình trạng khi load/inference. Source dirty chỉ được báo là exploratory; không tự commit để có revision sạch.

### Tiêu chí kết quả chốt trước chạy

| Nhóm kiểm | Tiêu chí | Cách kết luận |
| --- | --- | --- |
| Hồ sơ | Đủ config/hardware/checkpoint/source/binary/workload và raw; thiếu telemetry ghi unavailable. | Thiếu provenance quan trọng: không đủ điều kiện diễn giải như phép kiểm có kiểm soát. |
| Hoàn tất | Load và12/12request thành công, có terminal/DONE hợp lệ, không guard OOM, không watchdog stop. | Bất kỳ vi phạm: phép kiểm không hoàn tất; giữ cả raw lỗi. |
| Biên RAM | Sampler RSS không vượt5632MiB, processswap không vượt128MiB, MemAvailable không dưới1536MiB. Ghi cả HWM và các mốc trace. | Đạt ngưỡng chỉ đúng trong mẫu/cửa sổ này; không bảo đảm đỉnh bị bỏ sót hoặc thời gian dài hơn. |
| Accounting | Cache+recycled theo layer đúng budget, session sau admission đúng budget, trace không đếm retainedKV hai lần. | Sai invariant phải điều tra trước khi chọn tối ưu cache. |
| Nội dung | Response hợp lệ, usage/finish/cachedprompt và lịch sử phù hợp; ghi số token thực. | Smoke chức năng; không phải parity với model tham chiếu hay đánh giá chất lượng chuẩn hóa. |
| Xu hướng | Ghi RSS/swap cuối mỗi request, cuối từng nhóm4request, peak và chênh nhóm. | Không đặt ngưỡng plateau tùy tiện. Nếu tiếp tục tăng thì ghi chưa chứng minh ổn định, dù cửa sổ hoàn tất. |

Kết quả có thể là “không hoàn tất”, “hoàn tất cửa sổ nhưng còn tăng bộ nhớ” hoặc “hoàn tất và chưa thấy tăng tiếp trong cửa sổ này”. Cả ba đều không tương đương chứng minh hết leak/OOM. Phân tích tải nền/reclaim riêng, không quy toàn bộ pswpout hệ thống cho model. Nếu thất bại, bước sau là giải thích mốc thất bại từ raw, không tăng cache hay lặp ngay.

## 5. Khi nào chuyển sang expert cache/I/O?

Sau khi có hồ sơ cửa sổ với các giới hạn bộ nhớ rõ ràng, mới chọn một câu hỏi cache hẹp: với cùng ngân sách RAM, thay đổi cách giữ/nạp expert có giảm miss/filepayload hoặc thời gian chờ mà không đổi đầu ra và không tăng peak memory không? Cache64MiB/layer hiện chỉ là cấu hình khảo sát, chưa phải mức tối ưu hoặc cấu hình chung i5/i7. Chưa coi giai đoạn1 đã đóng.

Protocol này được ghi trước thực thi; kết quả08/11 giữ riêng ở hồ sơ liên kết đầu tài liệu và PROGRESS. Các bảng dự toán phía trên không được biến thành số đo RSS.

Phân tích hồi cứu08/11 đã có trong READMEexperiment liên kết đầu tài liệu: khoảngprefillrequest2–12 cộng+243,195MiB RSS; endpointRssAnon tăng271,168MiB/RssFilekhôngđổi. Chưa xác định liveallocation hay allocator giữtrang, cần profilerallocation/lifetime phù hợp trước bảnvá.

Instrumentation08/11 sau phépđocũ: memorytrace hiện có thêm allocatorindicator vàcallerTID, đã kiểmfixture và đo model thật; [hồ sơ allocator](../benchmarks/experiments/2026-11-08-allocator-i5/README.md) ủng hộ retainedmemory, không livegrowth tương ứng mức tăngRSS. Run kế tiếp cầnghi source/binary mới, không nhận hai revision là cùngcode đểsotốcđộ. XemARCHITECTURE/PROGRESS cho schema vàgiớihạn.
