# Chọn và chạy benchmark

Đọc [quy tắc chung](README.md) trước khi đo. Mục tiêu là sử dụng hiệu quả bộ nhớ cho trước; các số liệu i5/i7 đã có là kết quả vận hành riêng, không phải cuộc thi tốc độ CPU.

## Khi nào cần chạy?

**Không tự động chạy benchmark model lớn sau mỗi thay đổi, mỗi commit hoặc mỗi phiên làm việc.** Chỉ đo khi có câu hỏi cần số liệu để quyết định hoặc kết luận:

| Thay đổi hoặc nhu cầu | Kiểm chứng phù hợp |
| --- | --- |
| Sửa văn bản, format báo cáo, liên kết | Kiểm tài liệu; không benchmark |
| Sửa tooling hoặc cách ghi JSON, không đổi đường suy luận | Test công cụ/giao thức; không tự chạy lại model lớn |
| Sửa logic, tính toán hoặc giao thức engine | Hồi quy fixture và kiểm liên quan; không mặc định đo tốc độ |
| Tối ưu cache, prefetch, I/O hoặc kernel và cần đánh giá hiệu quả | Đối chứng trước/sau hoặc bật/tắt trên workload liên quan, sau khi kiểm tính đúng |
| Thay đổi cấp phát/ngân sách bộ nhớ hoặc điều tra RAM tăng | Kiểm bộ nhớ có giám sát phù hợp trigger; chỉ đo tốc độ nếu cần kết luận hiệu năng |
| Đổi checkpoint/backend/hardware hoặc cần ghi một mốc vận hành mới | Chọn các kịch bản vận hành cần thiết; ghi đây là cấu hình mới, không tự nhận là speedup của code |

Trước khi đo, ghi ngắn mục tiêu, biến thể đối chứng, workload/cấu hình, số lượt và điều kiện dừng. Ưu tiên đọc hồ sơ cũ nếu đã trả lời đúng câu hỏi. Nếu chưa có câu hỏi cụ thể, chưa chạy benchmark.

Chỉ chạy lại bộ vận hành đầy đủ khi cần ghi một mốc mới hoặc thay đổi có phạm vi rộng cần kiểm hiệu năng tổng thể. Dừng khi đã đủ dữ liệu cho câu hỏi; chỉ mở rộng/lặp lại nếu kết quả nhiễu, lỗi hoặc xuất hiện giả thuyết cụ thể. Giữ kết quả bất lợi, không lặp đến khi có số đẹp. Không cần chạy lại cả i5/i7 cho mỗi tối ưu.

## 1. Ba nhóm phép thử

| Nhóm | Khi nào dùng? | Công cụ |
| --- | --- | --- |
| Tính đúng và ổn định | Trước đo; sau khi sửa phần tính toán, bộ nhớ hoặc giao thức liên quan | `check.py`, test CUDA liên quan, `check_ram_stability.py` |
| Vận hành | Ghi nhận một phiên bản/cấu hình | `run_scenarios.py`: cold, warm, conversation |
| Đối chứng cơ chế | Khi cần quyết định giữ một cải tiến | Policy, KV, dispatch; prefetch còn cần thiết kế |

Không chạy tất cả các nhóm trên model lớn sau mọi sửa đổi. Lệnh chi tiết của các phép thử cũ ở [sổ tay báo cáo lần 1](../baocao/HUONG_DAN_LENH_THUC_NGHIEM.md).

## 2. Bộ đo vận hành chính

Đầu vào đầu tiên luôn là `Where is Mars`; câu hỏi tiếp trong hội thoại là `What is love ?` (giữ nguyên khoảng trắng trước dấu hỏi).

| Kịch bản | Một vòng làm gì? | Với 5 vòng |
| --- | --- | --- |
| `cold` | Reset cache ứng dụng, mở phiên mới, hỏi câu đầu | 5 mẫu |
| `warm` | Reset, hỏi câu đầu để làm ấm; mở phiên mới rồi hỏi lại, giữ cache expert | 5 mẫu + 5 làm ấm |
| `conversation` | Reset, hỏi câu đầu; gửi lịch sử gồm câu trả lời thực tế rồi hỏi câu tiếp, cùng phiên | 10 mẫu, tách kết quả lượt đầu/lượt tiếp |

Tổng bộ này: 20 mẫu đo + 5 yêu cầu làm ấm. Warm tách phiên để tránh dùng lại KV; conversation gồm cả tác động lịch sử và KV. Một lần làm ấm không có nghĩa cache đã đạt trạng thái ổn định.

Runner hiện mặc định **một policy victim**, cache 64 MiB/lớp, context 512, tối đa 128 token đầu ra, 5 vòng. Giữ `--single-policy` để tương thích lệnh cũ; muốn AB/BA phải thêm `--compare-cache-policies`. `--runs` là số vòng mỗi policy, không phải tổng request.

Preview trên i5, không build/hash/nạp model:

```bash
MOE_RAM_LIMIT_BYTES=5368709120 MOE_RAM_HEADROOM_BYTES=1073741824 \
python3 tools/run_scenarios.py cold \
  --testbed i5 --gpu auto --threads 8 --cache 64 \
  --context 512 --max-tokens 128 --runs 5 \
  --expert-cache-policy victim --allow-dirty --dry-run
```

Đổi `cold` thành `warm` hoặc `conversation` khi cần. Chọn `--model-dir` theo checkpoint thực tế. Máy i7 đổi nhãn; máy không có GPU chọn `--gpu cpu`. HTTP `auto` yêu cầu GPU hoạt động, không âm thầm chuyển CPU. CPU mode hiện chọn trước load qua `MOE_GPU_MODE=cpu`; record cũ trước bản vá 04/11 có thể đã khởi tạo VRAM.

Đo thật chỉ sau kiểm môi trường/RAM và dừng inference khác: bỏ `--dry-run` và, với lượt đối chứng source sạch, bỏ `--allow-dirty`. Các ngưỡng minh họa 5 GiB/headroom 1 GiB không bảo đảm an toàn trên mọi máy. HTTP runner lưu sampler/watchdog độc lập; guard engine là kiểm hợp tác, không hard cap RSS.

Đây là cách gọi cấu hình phase1-v1; không bắt chạy lại bộ cũ để bắt đầu giai đoạn tiếp.

## 3. Đối chứng quản lý expert trong ngân sách cố định

Mỗi phép thử phải chốt sáu điều: **câu hỏi, đầu vào, điều kiện giữ nguyên, yếu tố thay đổi, chỉ số, tiêu chí kết luận**. Đối chứng trước hết trong cùng máy và checkpoint.

| Cơ chế | Yếu tố thay đổi | Trạng thái |
| --- | --- | --- |
| Inclusive/victim | Cách bố trí expert giữa RAM/VRAM, giữ ngân sách hai tầng | Có thí nghiệm cũ; số mẫu nhỏ |
| Tái sử dụng tiền tố KV | Cách tận dụng phần token chung | Có thí nghiệm cũ; chưa đo đầy đủ đánh đổi dung lượng KV |
| Dispatch theo nhóm | Cách tổ chức tính toán | Có đối chứng, candidate chậm hơn trong phép thử và không bật runtime |
| Prefetch dự đoán | Bật/tắt nạp trước expert dự đoán, giữ đường nạp bắt buộc | Có công tắc/bộ đếm và runner; fixture đạt, lượt model lớn dừng do swap trước cặp đầu |

Ưu tiên thiết kế prefetch trước khi triển khai thêm thuật toán: xác định đúng loại nạp trước, tính cả buffer đang chờ vào kế hoạch bộ nhớ, ghi số dự đoán được dùng/không dùng và kiểm đầu ra nhất quán. Đo thời gian phản hồi, tốc độ sinh, file payload, truyền GPU, RAM và thời gian chờ. Không nhận hit rate tăng là đủ chứng minh nhanh hơn.

Policy comparison được chọn tường minh, ví dụ preview:

```bash
python3 tools/run_scenarios.py conversation \
  --testbed i5 --gpu auto --threads 8 --cache 64 \
  --compare-cache-policies --runs 2 --allow-dirty --dry-run
```

Thứ tự AB/BA, hai policy mỗi vòng; hai vòng là 8 request. Giới hạn RAM cache bằng nhau chưa chứng minh tổng RSS thực tế bằng nhau, nên vẫn phải quan sát RSS/VRAM/buffer. Không đồng thời đổi policy, checkpoint và backend rồi gán chênh lệch cho một cơ chế.

### 3.1. Thiết kế đối chứng prefetch dự đoán (03/11/2026)

Cập nhật 04/11: protocol hiện hành `lookahead-v2` chọn CPU trước load. Không ghép kết quả với v1 (CPU tắt sau load). Flag CLI giữ nguyên; source revision và `gpu_mode_at_load` phân biệt điều kiện thực.

**Câu hỏi:** Khi giữ cùng ngân sách bộ nhớ, dự đoán và tải trước expert của lớp kế tiếp có giảm thời gian sinh mà không tải thừa quá nhiều không?

**Phạm vi đã đối chiếu mã:** `prefetch_missing_experts` trong prefill nạp expert có nhu cầu đã biết, giữ nguyên ở cả hai chế độ. Trong `step`, lookahead dự đoán tối đa hai expert của lớp kế tiếp khi lớp hiện tại không còn miss. Điểm dự đoán là số đếm chuyển tiếp cộng heat/4 lấy phần nguyên, không phải xác suất chuẩn hóa. `reset_cache` xóa cả cache expert, recycled buffers, heat, KV và bảng chuyển tiếp.

| Thành phần | Quy định cho phép thử đầu |
| --- | --- |
| A / B | A tắt lookahead dự đoán; B bật thuật toán hiện tại, tối đa hai ứng viên. Không tắt nạp theo nhu cầu hoặc nạp trước prefill |
| Cách triển khai cần bổ sung | Hai chế độ trong cùng binary: `MOE_LOOKAHEAD_PREFETCH=on/off`, runner chọn và kiểm trạng thái/metric thực tế. Mặc định runtime vẫn on |
| Máy/model | i5, một checkpoint INT8 cố định, ghi hash; chưa yêu cầu i7 |
| Cấu hình ứng viên | CPU, 4 Rayon threads, victim, cache64MiB/lớp, context512, tối đa64 token sinh. Guard5GiB/headroom1GiB chỉ dùng sau kiểm RAM tại máy thử; không tự tăng ngưỡng nếu không đủ |
| VRAM | Từ 04/11, CPU mode bỏ qua probe CUDA/VRAM lúc load; ghi `gpu_mode_at_load=cpu` và trạng thái 0 VRAM. Record v1 trước bản vá vẫn có thể khởi tạo VRAM |
| Đầu vào | Theo thứ tự: `Where is Mars`, `What is love ?`, `Explain a Python dictionary.`, `Why did the Roman Republic end?` |
| Phiên và lịch sử | Mỗi câu phiên mới, một user message; không gửi câu trả lời trước làm lịch sử. Giữ cache expert và bảng chuyển tiếp giữa bốn câu |
| Một block | Một chế độ chạy bốn câu trên model vừa load; reset trước câu đầu, không reset giữa các câu. Không warmup riêng; ghi đây là quá trình học/ấm lên, không steady state |
| Số lượt | Năm cặp block, thứ tự AB / BA / AB / BA / AB; tổng40 request đo. Mỗi block process riêng, tải model ngoài timing request; cùng binary/checkpoint |

**Ngân sách:** cả hai chế độ giữ cùng trần cache RAM và cùng dự phòng expert đang xử lý, bao gồm buffer lookahead. Công thức hiện tại dự phòng `(topK + 2) × bytes/expert` là kế hoạch payload, chưa là hard cap tổng RAM. Hai metric `decode_expert_active_peak_bytes` và `decode_expert_host_peak_bytes` lấy mẫu capacity tại các mốc giữ expert/buffer ở decode; host gồm cache+recycled các lớp cộng payload đang giữ/nạp. Runner kiểm với dự phòng active và tổng cache+dự phòng; đây không phải profiler mọi cấp phát tức thời. Prefill, dense, KV, scratch, pinned staging, metadata và allocator không nằm trong hai metric này; RSS/watchdog theo dõi tổng tiến trình riêng. Không cộng ngân sách mới cho chế độ B. Cùng ngân sách tối đa không có nghĩa RSS thực tế bằng nhau; báo RSS/HWM/swap/VRAM quan sát cùng phần buffer sử dụng. Nếu chưa xác nhận được accounting thì chỉ gọi kết quả exploratory, chưa kết luận ngân sách được bảo toàn.

**Bộ đếm đã bổ sung:** số lần đủ điều kiện dự đoán; ứng viên được đề xuất; lượt tải thành công/lỗi; expert dự đoán được dùng ở lớp kế tiếp; expert không dùng; request kết thúc/hủy còn expert đang chờ. Metric `lookahead` trong kết quả có `eligible_layers`, `proposed`, `loaded`, `load_errors`, `used`, `unused`, `abandoned`, `pending`, `file_bytes`, `buffer_peak_bytes`. `used` nghĩa expert đã được Router chọn và lấy cho lớp kế tiếp, không chứng minh toàn request đã hoàn tất. Byte đếm payload lượt tải thành công; không đo byte đọc dở khi lỗi. Đếm riêng byte tải speculative và capacity buffer đỉnh. Phân biệt “đề xuất” với “đã tải”; tổng expert tải thành công phải giải thích được bằng dùng/không dùng/còn chờ. Giữ việc cập nhật transition và heat ở cả A/B để chỉ thay tác vụ dự đoán/tải speculative. Không suy mọi byte speculative thành lãng phí SSD vật lý.

**Tính đúng:** fixture có đường thực thi lookahead thật, kiểm logits theo tolerance hiện có và token IDs cả chuỗi; kiểm A không phát sinh speculative load, B có trường hợp dùng trúng/sai, reset, lỗi tải và buffer được thu hồi. Hồi quy mới ép đường I/O thật trên fixture với một expert được chọn, một không dùng; so logits, hủy trước lớp kế tiếp, lỗi file trên bản sao riêng và reset/retry. Fixture MHA/GQA kiểm token đầu ra. Đây vẫn chưa là parity toàn model7B. Trong model thật, lưu token IDs nếu đường đo hỗ trợ; nếu chỉ HTTP thì kiểm text/token usage và công bố thiếu logits parity. Nếu đầu ra khác, dừng việc diễn giải cặp đó là cùng workload để tìm nguyên nhân; không xóa mẫu khác biệt.

**Kết quả chính:** thời gian decode và steps/s sau token đầu, wall, TTFT; file payload/load count, thời gian chờ, bộ đếm speculative và RAM/VRAM. Prefill giữ nguyên nên TTFT chủ yếu là chỉ số kiểm tác động phụ. Tách kết quả theo bốn đầu vào và so từng cặp A/B; báo trung vị chênh lệch theo cặp cùng min–max, không chỉ chọn mẫu tốt nhất hoặc gộp tốc độ các câu.

**Điều kiện chạy và dừng:** chỉ đo sau công tắc/telemetry/hồi quy đạt và kiểm RAM có giám sát. Kế thừa watchdog HTTP: dừng khi RSS>5,5GiB, MemAvailable<1,5GiB, process swap>128MiB hoặc mất telemetry; không chạy build/test/server khác đồng thời. Lỗi protocol, model hoặc watchdog giữ status lỗi, dừng bộ đo. Nếu không có speculative load, kết quả chỉ cho thấy cơ chế không được kích hoạt trong workload này; không nhận đó là bằng chứng prefetch luôn vô ích, cũng không mở thêm workload tự động. Nếu chênh lệch nhỏ so dao động thì kết luận chưa rõ lợi ích, không chạy lặp đến khi có speedup.

**Trạng thái:** đã triển khai điều khiển/telemetry/runner paired và kiểm fixture. [Kiểm RAM CPU trước load 04/11](experiments/2026-11-04-cpu-load-i5/README.md) hoàn tất4request nhưng counter đủ điều kiện dự đoán đều0; cần rà kích hoạt trước khi quyết định bộ đối chứng tốc độ. [Lượt model lớn](experiments/2026-11-03-lookahead-i5/README.md) dừng do swap trong block đầu, chưa đủ cặp để kết luận hiệu năng. CLI dùng entry point chung, không tạo runner độc lập cho người dùng.

Preview cấu hình ứng viên (không build, hash checkpoint hoặc nạp model):

```bash
python3 tools/run_benchmark.py --http-session --http-scenario topics \
  --compare-lookahead --testbed i5 --gpu cpu --threads 4 \
  --cache 64 --context 512 --max-tokens 64 --runs 5 \
  --model-dir models/olmoe-1b-7b-int8 --tag lookahead-v2 \
  --allow-dirty --dry-run
```

`--runs 5` ở đây là năm cặp, tổng40request. Đo có kiểm soát trên revision sạch: bỏ **cả** `--allow-dirty` và `--dry-run` sau khi hoàn tất kiểm RAM/môi trường. Không tự commit hoặc dùng dirty run rồi gọi là kết quả chính thức. Protocol `lookahead-v2` hiện chỉ nhận `--gpu cpu`; chưa mở đối chứng GPU vì cần chốt trần VRAM tường minh. `--lookahead on/off` dành cho lượt chỉ một chế độ; trong paired, block plan quyết định từng chế độ, không so policy đồng thời.

Run giữ một binary/source.zip/hash/checkpoint manifest chung. Mỗi block có process/server log/time series/watchdog, snapshot status-start trước request, status-end và result nếu block hoàn tất; từng cặp có đối chiếu, kết quả tổng tách bốn đầu vào. Kiểm source/binary trước mỗi block và cuối phép thử; kiểm checkpoint theo danh sách file, kích thước và mtime sau hash ban đầu (không chứng minh bất biến nếu file bị đổi rồi phục hồi đúng metadata). Không chỉnh checkpoint khi đo. Root record status failed nếu gặp sai khác hoặc vi phạm; dữ liệu block/request đã ghi vẫn giữ nguyên. HTTP không trả token IDs nên parity ở đây là text/usage/finish reason, không full logits.

Để kiểm riêng runner trên fixture nhỏ (không đánh giá tốc độ 7B), dùng cùng lệnh nhưng model-dir là `tests/fixtures/olmoe/unnormalized-mha`, cache1/context64/max-tokens3/runs2/threads2 và output-root thư mục tạm; dirty run ở đây chỉ kiểm giao thức. Fixture có thể báo `no_speculative_loads`; các hồi quy Rust riêng xác nhận nhánh nạp dự đoán thực sự được thực thi.

Việc thêm telemetry cũng có overhead lấy mẫu/duyệt cache ở decode. Hai chế độ cùng chạy instrumentation; kết quả mới không được ghép trước/sau với binary kỳ 1 để gán mọi chênh lệch cho prefetch.

### 3.2. Chẩn đoán vì sao lookahead không kích hoạt

Câu hỏi hẹp: **Ở cùng ngân sách 64 MiB/lớp, cơ chế bị chặn ở điều kiện thiếu expert hay đã dự đoán nhưng không tìm được ứng viên?** Chưa nới gating, tăng cache hoặc đổi cách xếp hạng. Nếu dữ liệu cũ chỉ có eligible_layers=0, không dựng lại phân bố miss từ tổng cache hit rate.

| Nhóm bộ đếm trong metrics.lookahead | Ý nghĩa |
| --- | --- |
| evaluated_layers / no_next_layer | Lượt ghé lớp ở decode; lớp cuối không có lớp kế tiếp để dự đoán |
| miss_histogram | Phần tử thứ m đếm lượt còn thiếu m expert sau tra cache/lookahead trước đó; chỉ gồm các lớp có lớp kế tiếp |
| blocked_by_misses / eligible_layers | Bị chặn vì còn ít nhất 1 miss / đủ điều kiện vì không còn miss; độc lập công tắc bật/tắt |
| disabled_eligible_layers | Đủ điều kiện nhưng tắt cơ chế; không gộp với trường hợp chưa đủ điều kiện |
| prediction_calls | Số lần thực sự chấm điểm; khi bị chặn thì không chấm điểm ngầm để đo thử |
| no_positive_score_calls | Đã gọi dự đoán nhưng chưa có ứng viên điểm dương |
| all_candidates_cached_calls | Có ứng viên điểm dương nhưng tất cả ứng viên dự đoán đã ở RAM/VRAM; không có nghĩa Router lớp sau chắc chắn không cần expert khác |
| positive_candidates / candidates_in_ram / candidates_in_vram / candidates_available | Tổng ứng viên điểm dương trước lọc, bị lọc do RAM, do VRAM, và còn lại trước giới hạn cửa sổ 2 |

Tổng kiểm: evaluated=no_next+blocked+eligible; tổng histogram=blocked+eligible; histogram[0]=eligible; prediction_calls=eligible-disabled_eligible. Tổng positive bằng in_ram+in_vram+available. Không diễn giải các bộ đếm không có ứng viên nếu prediction_calls=0: khi đó ứng viên chưa được đánh giá. Các bộ đếm chỉ thuộc decode, không mô tả nạp theo nhu cầu đã biết trong prefill.

Một lượt chẩn đoán ban đầu là một request `Where is Mars`, phiên mới/reset đầu, CPU trước load, 4 threads/cache 64/context 512/max 64, lookahead **on**, cùng guard/watchdog. Dùng RAMrunner để giám sát và giữ raw; các timer được lưu nhưng không dùng so tốc độ. Preview:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead on \
  --cache 64 --context 512 --cycles 1 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty --dry-run
```

Đây là khảo sát chẩn đoán nếu dùng nguồn dirty; không mở bộ 40 request. Trên fixture đã ép 1/2 miss, chưa có điểm và ứng viên dự đoán đã có trong RAM; chọn expert và logits giữ nguyên. Lượt model thật 04/11 đầu tiên [dừng do swap trước khi request hoàn tất](experiments/2026-11-04-cpu-load-i5/README.md), không có histogram để báo số từ lượt đó.

Sau yêu cầu tiếp tục, [một request chẩn đoán tiếp theo đã hoàn tất](experiments/2026-11-04-cpu-load-i5/README.md#lượt-chẩn-đoán-tiếp-theo-hoàn-tất): histogram `[0,4,19,41,156,249,261,144,71]`, tổng 945 lượt có lớp kế tiếp, eligible/prediction_calls đều 0. Có 881/945 lượt (93,23%) thiếu 4–8 expert; gate không miss chặn toàn bộ predictor trong mẫu này. Không suy từ một câu sang mọi workload hoặc chất lượng predictor. Chưa cần lặp chẩn đoán hay đối chứng tốc độ; bước tiếp đề xuất kiểm xu hướng tổng RAM với lookahead tắt theo PLAN, giữ ngưỡng hiện có.

### 3.3. Kiểm xu hướng tổng RAM hữu hạn trên i5 (04/11/2026)

**Câu hỏi:** Với CPU trước load, 4 threads, victim, cache64 MiB/layer, context512, tối đa64 token và lookahead tắt, RSS qua chuỗi giữ cache còn tăng ra sao? Đây là kiểm bộ nhớ, không đo lợi ích tốc độ hoặc so bật/tắt lookahead.

**Phạm vi chốt trước chạy:** một process, tối đa12 request (ba nhóm bốn request), không warmup tách, reset chỉ trước request đầu. Runner `--cycles` đếm request, không phải nhóm. Mỗi nhóm hỏi `Where is Mars`, rồi `What planet did I just ask about?` cùng phiên/lịch sử chứa câu trả lời thật, tiếp `Explain a Python dictionary.` và `What is love ?` mỗi câu mở phiên mới. Giữ cache expert giữa các request, không thay ngân sách session. Không mở thêm lượt tự động sau kết quả; thời gian thực được sampler ghi lại.

**Điều kiện và dừng:** source/binary đã kiểm, không build/tests/inference khác đồng thời; quan sát RAM/swap-out/PSI trước tải, preview rồi mới chạy. Giữ guardRSS5GiB/headroom1GiB, watchdogRSS5,5GiB/MemAvailable1,5GiB/processswap128MiB và sampler250ms. Vi phạm/mất telemetry/lỗi giao thức thì giữ raw và dừng, không nới ngưỡng hoặc lặp ngay. Dirty source chỉ exploratory.

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --cache 64 --context 512 --cycles 12 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty --dry-run
```

Bỏ `--dry-run` sau preflight; giữ các cờ khác cho khảo sát nguồn dirty này. Đọc từng request: RSS/HWM/VmSwap, cache/recycled/session capacity, prompt/completion tokens, trạng thái guard/watchdog. Báo biến thiên RSS trong từng nhóm và chênh lệch cuối nhóm1→2→3, đối chiếu capacity cùng thời điểm; xem time series cho peak giữa request. Capacity không bằng RSS, phần chênh chưa giải thích không tự là leak. Hoàn tất12request trong ngưỡng chỉ chứng minh quan sát hữu hạn của workload này; không gọi là hard RAM cap, plateau dài hạn hoặc hết OOM. Nếu còn tăng rõ, ưu tiên rà ownership/lifetime/session/allocator theo bằng chứng trước thay cache policy.

**Kết quả đã chạy:** [hồ sơ chuỗi RAM](experiments/2026-11-04-ram-sequence-i5/README.md) giữ10request hoàn tất rồi watchdog dừng trong request11 ở207,649s vì processswap137,336MiB. Chưa hoàn tất nhóm3. RSSmax3,097691GiB, MemAvailablemin7,077408GiB; capacity cache/session cuối request gần cố định. Đoạn cuối RSS giảm bằng VmSwap tăng, chưa xác định tác nhân reclaim. Không đo lại ngay; bổ sung telemetry reclaim/cgroup trước theo PLAN. Gate đủ điều kiện4/9450lượt lớp trong10request, modeoff không gọi predictor; không suy kết quả gate0 của một request trước thành luôn bị chặn.

### 3.4. Đọc telemetry reclaim mới (04/11/2026)

Sampler RAM/HTTP dùng chung `tools/memory_telemetry.py`, bổ sung object `reclaim` vào từng dòng `memory.jsonl`. Không thêm hồi tố vào raw cũ; các trường `process`/`system` và ngưỡng watchdog giữ nguyên. Đã kiểm fixture/đọc-only và [một lượt12request model thật ngày05/11](experiments/2026-11-05-ram-reclaim-i5/README.md).

| Trường trong reclaim | Nội dung và đơn vị |
| --- | --- |
| schema_version / page_size_bytes | Schema1; kích thước page của host theo byte, null nếu không đọc được |
| vmstat | pswpin/pswpout và pgscan_*/pgsteal_*/allocstall_* của toàn hệ thống; bộ đếm tích lũy, swap tính theo page |
| psi_memory | PSI memory toàn hệ thống: some/full, avg10/60/300 theo phần trăm, total theo microsecond tích lũy |
| cgroup_v2.ancestors | Mức0 là cgroup của PID child, các mức sau là ancestor tới gốc mount nhìn thấy; không phải cgroup của runner |
| ancestors[].identity | device/inode của thư mục cgroup; dùng kiểm nguồn có đổi giữa các mẫu trước khi tính delta, không ghi đường dẫn/tên nhóm cá nhân |
| ancestors[].files | memory.current/max/high, memory.swap.current/max theo byte hoặc `max` ở trường giới hạn; events/events.local, stat chọn lọc và pressure theo định nghĩa kernel |

`memory.stat` giữ anon/file/shmem/unevictable (byte), pgfault/pgmajfault, các bộ đếm pgscan/pgsteal và workingset_refault. Không quy mọi trường stat thành byte; events là số sự kiện. PSI của cgroup có cùng đơn vị PSI hệ thống. Mỗi probe có status `ok`, `partial` hoặc `unavailable` và reason khi cần; thiếu file/controller, không có cgroup-v2, thiếu quyền hoặc dữ liệu sai không được thay bằng0. Thiếu một trong hai bộ đếm swap làm vmstat `partial`; file cgroup lỗi được ghi riêng. `max` là không có giới hạn tại mức đang đọc, không chứng minh không có giới hạn ẩn. Cgroup scope chỉ gồm các ancestor nhìn thấy trong mount/namespace của observer; cgroup-v1 chưa hỗ trợ.

Membership đọc lại mỗi mẫu, kiểm lại cuối khi lấy các file cgroup; nếu đổi giữa lúc đọc hoặc child thoát thì bỏ attribution mẫu đó và ghi unavailable. Không dùng đường dẫn cgroup, mount hoặc exception chứa đường dẫn trong JSON. Các file đọc nối tiếp, không phải snapshot nguyên tử; sampler này có thêm chi phí đọc/probe, chưa đo overhead trên workload thật. Cadence vẫn wait250ms sau mỗi mẫu, không cam kết chính xác4Hz.

Khi có phép chẩn đoán mới được giao: xem chênh pswpout/PSI total cùng thời gian RSS/VmSwap, rồi đối chiếu delta events/stat và memory.high/max/current của child/ancestor. Chỉ tính delta cgroup khi identity không đổi; không cộng bộ đếm ancestor với child vì có thể bao gồm cùng sự kiện. pswpout tăng chỉ xác định swap-out toàn hệ thống, không tự chỉ ra process hoặc tác nhân; PSI/reclaim counters cũng không tự chứng minh nguyên nhân. Giữ raw, các probe thiếu và lý do watchdog. Telemetry reclaim bổ sung không thay điều kiện dừng dựa trên RSS/MemAvailable/VmSwap hiện hành; thiếu telemetry mới làm hạn chế chẩn đoán, không chứng nhận an toàn.

**Phạm vi đo được giao 05/11, chốt trước chạy:** một lượt tối đa12request theo đúng lệnh và chuỗi ở3.3, lookahead off, cùng cache/backend/threads/context/token limit/guard/watchdog. Câu hỏi bổ sung là delta pswpout/PSI và events/stat/giới hạn child/ancestor quanh lúc RSS/VmSwap đổi; đồng thời xem reported capacity cuối request có giải thích tăngRSS không. Source engine/binary giữ bản đã kiểm, tooling có telemetry mới và nguồn dirty nên exploratory. Không so tốc độ hoặc gán khác biệt với04/11 cho code; nếu dừng thì giữ raw và không lặp. Nếu không có swap/reclaim đáng kể, ghi không tái hiện trong cửa sổ này, không suy đã chữa nguyên nhân cũ.

**Kết quả05/11:** [hồ sơ](experiments/2026-11-05-ram-reclaim-i5/README.md) hoàn tất12/12request/927mẫu, không watchdogviolation; VmSwapchild0 nhưng pswpout hệ thống tăng86023page (336,027MiB với page4096B). Các cgroup nhìn thấy có limit đọc được đều max/events không tăng high/max/OOM, rootpartial; nhóm chứa child có thể có nhiều process nên không quy counter cho riêng engine. RSS requestđầu→cuối tăng288,094MiB khi reported cache/session gần cố định. Không tái hiện sự cố processswap cũ và chưa giải thích phần tăngRSS; không lặp ngay, bước tiếp là accounting allocation/lifetime theo mốc trong PLAN.

### 3.5. Trace capacity theo mốc request (05/11/2026)

Đã có instrumentation chọn tường minh qua `MOE_MEMORY_TRACE=on|off`, mặc định off. RAMrunner có `--memory-trace on|off` (mặc định off), pin vào môi trường child trước load và kiểm status; phải rebuild server khi cập nhật source. Không thay ngưỡng watchdog hoặc thêm hồi tố field vào dữ liệu cũ. Trace đã kiểm MHA/GQA/HTTPfixture và một lượt4request model thật05/11.

Schema `metrics.memory_trace` gồm schema_version1/outcome/samples; trace kết thúc cuối cùng cũng có ở `/experts.last_memory_trace`, kể cả request lỗi. Samples theo các mốc lookupKV/prefill/admission/ra khỏi inner, RSS/VmSwap theo **byte**, khác process snapshot KiB của runner. Retained KV là subset của retained session, không cộng hai lần; active KV sau transfer/drop bằng0. Giới hạn phạm vi và các vùng chưa tính ở [ARCHITECTURE](../docs/ARCHITECTURE.md#trace-capacity-theo-mốc-request-05112026). Thiếu RSS là null, không thay bằng0.

Kiểm tích hợp fixture, preview trước, sau đó chỉ bỏ dry-run nếu được giao thực thi:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --memory-trace on --cache 1 --context 64 --cycles 1 --max-tokens 3 \
  --keep-cache --threads 2 --model-dir tests/fixtures/olmoe/unnormalized-mha \
  --output-root /tmp/moe-memory-stage-fixture --allow-dirty --dry-run
```

Để đọc một request thành công, kiểm outcome completed, thứ tự stage, tổng tracked_host_bytes từ các thành phần được cộng, và active/retained qua admission. So số byte cùng mốc với RSS và sampler reclaim; không cộng peak ở thời điểm khác vào sample hoặc gọi tracked bytes là toàn bộ RAM. Khi lỗi, giữ raw SSE/watchdog, đọc last trace sau khi worker idle nếu có; watchdog kill có thể không cho engine hoàn tất trace nên không dựng stage thiếu. Tắt trace để đo tốc độ độc lập; chưa đo overhead model thật. Chưa mặc định mở thêm một chuỗi12request7B sau thay đổi này.

**Phép chẩn đoán model thật được giao tiếp ngày05/11, chốt trước chạy:** đúng một nhóm4request theo chuỗi3.3, CPU4threads/victim/cache64/context512/max64/keepcache/lookahead off, thêm `--memory-trace on`. Giữ guard/watchdog như3.3 và không chạy thêm nhóm sau kết quả. Câu hỏi chính là delta RSS/capacity tại từng stage và qua request, xem vùng active/retained KV, scratch/prefill và thời gian ngoài phạm vi accounting thay đổi ra sao. Khoảng before_request→after_kv_lookup còn gồm tokenization/encoding/clone tokenizer, không quy delta này cho riêng KV. Không dùng bốn mẫu làm chứng minh plateau/hếtleak hoặc so tốc độ với lượt trước. Nếu dừng thì giữ raw và không lặp ngay.

**Kết quả:** [hồ sơ stage](experiments/2026-11-05-memory-stage-i5/README.md) hoàn tất4/4request/28stage/328samplerow, không watchdogviolation/processswap. Khi mở phiên mới có hai bộKV khoảng128MiB cùng tồn tại trước admission; trackedcapacity giảm khoảng128MiB khi bỏ phiên cũ, RSS không giảm ngay. Không quy mọi tăngRSS cho KV/allocator hoặc loại trừ allocation khác. Bước tiếp chọn candidate KVreuse từ session bị evict, kiểm prefix/LRU/budget/hủy/lỗi trước đối chứng; không lặp cùng phép đo hoặc mở lại lookahead.

### 3.6. Candidate tái sử dụng buffer KV (05/11/2026)

Đã triển khai candidate opt-in `MOE_KV_BUFFER_REUSE=on|off`, mặc định off; RAMrunner có `--kv-buffer-reuse on|off` mặc định off, pin trước load và kiểm status. Không tắt valid prefix reuse khi off. Cần rebuild server sau cập nhật. Điều kiện một donor LRU, budget, capacity và tác động evict sớm khi hủy/lỗi ở [ARCHITECTURE](../docs/ARCHITECTURE.md#candidate-tái-sử-dụng-buffer-kv-05112026).

Đường được kiểm fixture: cùng ID/prefix0 clear và dùng lại buffer; ID mới có thể mượn donor khi một lần evict đủ budget; còn chỗ/sessionless/oversized/donor không vừa thì fallback. Không có pool ngoài budget hoặc ngưỡng RSS mới. Sau handoff ID cũ phải mất cache, không dùng buffer đã ghi đè để phục hồi lịch sử. Giữ callback hủy/lỗi, token/logit parity và retry.

Preview HTTPfixture (nhỏ, không phải số đo7B):

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --memory-trace on --kv-buffer-reuse on --cache 1 --context 64 \
  --cycles 4 --max-tokens 3 --keep-cache --threads 2 \
  --model-dir tests/fixtures/olmoe/unnormalized-mha \
  --output-root /tmp/moe-kv-buffer-reuse-fixture --allow-dirty --dry-run
```

Kiểm metrics.kv_buffer_reuse và status.last_kv_buffer_reuse/mode cùng trace. HTTPfixture này đủ chỗ giữ các session nhỏ trong budget256MiB nên không ép earlyeviction; các unitfixtures Rust riêng giới hạn một slot để thực thi handoff và kiểm buffer thật. Không nhận flagon hoặc HTTPcompleted là bằng chứng đã reuse donor.

Chưa có đối chứng model thật trong bước triển khai. Khi được giao đo, dùng cùng binary/checkpoint/config và chỉ đổi `--kv-buffer-reuse off/on`, giữ guard/watchdog, kiểm source/config/output/prefix và từng stage capacity. Không lấy raw trước bản vá làm đối chứng cùng source hoặc gán lợi ích từ Clippy/pointer reuse. Xét cả việc phiên bị loại sớm mất reuse khi request mới thất bại. Chốt số lượt/thứ tự trước chạy; không tự mở bộ12request/hai máy. RSS/speedup và ổn định dài hạn đều chưa được chứng minh.

**Đối chứng được giao05/11, chốt trước chạy:** đúng một cặp block, thứ tựoff→on; mỗi block process mới/4request theo chuỗi3.3, reset chỉ đầu/keepcache, không warmup riêng. Cùng binary/source/checkpoint vàCPU4threads/victim/cache64/context512/max64/lookahead off/memorytrace on, guard/watchdog không đổi. Tổng tối đa8request, chạy tuần tự, dừng cả cặp nếu block lỗi/watchdog hoặc đầu ra/config không phù hợp; không tự lặp tìm kết quả đẹp. Kiểm source/binary giữa block và cuối, checkpoint/hash ở từng record, RAM/swap-out/PSI trước mỗi block. Chỉ khác --kv-buffer-reuse off/on. Đọc từng sample: text/usage/finish/prefix (HTTP không có tokenIDs/logits), số reuse/evict, active+retainedKV, trackedcapacity, RSS/HWM/swap. Một cặp thứ tự cố định chịu ảnh hưởng OS-cache/tải nền/allocator/process khác, chỉ là khảo sát có giới hạn, không kết luận speedup/ổn định dài hạn hay bật mặc định từ một lần.

**Kết quả:** [hồ sơ cặp chưa hoàn tất](experiments/2026-11-05-kv-reuse-ab-i5/README.md) giữ blockoff có3request hoàn tất rồi watchdog dừng ở64,170s vì processswap134,414MiB. Blockon không chạy theo điều kiện dừng. Chưa có so sánh RSS/speedup off/on; giữ candidateoff, không dùng raw cũ hoặc giả lập giá trị on để dựng cặp. Telemetry có reclaim/swap với events cgroup nhìn thấy không tăng high/max/OOM, chưa xác định tác nhân. Không nâng ngưỡng hoặc lặp ngay.

**Phép hẹp được giao tiếp05/11, chốt trước chạy:** một cặp on→off, mỗi process mới đúng2request, tổng tối đa4request. Workload mới `--workload kv-switch` v1 gồm `Where is Mars` rồi `Explain a Python dictionary.`, mỗi câu session mới/không history; chỉ reset đầu và keepcache. Khác workloadmixed cũ, không ghép sample thành cặp với lượt trước. Giữ CPU4threads/victim/cache64/context512/max64/lookahead off/memorytrace on/guardRSS5GiB/headroom1GiB/watchdogRSS5,5GiB/available1,5GiB/processswap128MiB. Chỉ khác --kv-buffer-reuse on/off trong cặp; cùng binary/source/checkpoint. Dừng cả cặp nếu bất kỳ block lỗi/watchdog/parity/config không phù hợp; không tự lặp. So text/usage/finish (HTTP không có logits), sốreuse/evict, active+retainedKV và RSS/swap tại các stage; một cặp thứ tự cố định chỉ cho tín hiệu hữu hạn, chưa chứng minh speedup/ổn định dài hạn hoặc mặc định bật.

RAMrunner `--workload mixed` mặc định giữ chuỗi cũ; kv-switch luôn tạo session mới từng request. Workloadspec/version/prompts được lưu trong record/preview, kể cả khi chưa có request hoàn tất. Ví dụ preview blockon của phép hẹp:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --memory-trace on --kv-buffer-reuse on --workload kv-switch \
  --cache 64 --context 512 --cycles 2 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty --dry-run
```

Blockoff chỉ đổi --kv-buffer-reuse off. Trước mỗi block kiểm môi trường/pressure và fingerprint, hoàn tất build/tests trước tải; không đổi OS hoặc ngưỡng để vượt failure.

## 4. Kịch bản mở rộng, chỉ dùng khi có câu hỏi cụ thể

| Preset | Mục đích | Request với 5 vòng, một policy |
| --- | --- | --- |
| `topics` | Bốn câu về Mars, tình yêu, Python, Rome; mỗi câu phiên mới, giữ expert cache giữa các câu; reset đầu vòng | 20 |
| `long-context` | Ghi chú cố định dài hơn để quan sát prefill; xem số token thực tế, không tự gọi là sát trần context | 5 |
| `cache-sweep` | Chạy conversation với các ngân sách do người chạy chọn; phục vụ chọn cấu hình | 10 mỗi mức |

Cache-sweep không còn tự chọn 128/192/256 MiB/lớp. Phải truyền danh sách có thứ tự sau khi kiểm dự toán RAM, ví dụ **chỉ preview**:

```bash
python3 tools/run_scenarios.py cache-sweep \
  --testbed i5 --gpu cpu --threads 8 --cache-levels 32 64 \
  --runs 5 --allow-dirty --dry-run
```

Mỗi mức là một process riêng, dừng nếu một mức lỗi. Danh sách không chứng nhận an toàn và không tự đảo thứ tự; nếu đối chứng ngân sách, cần thiết kế cả thứ tự ngược để giảm ảnh hưởng nhiệt/OS cache. Không dùng quét cache làm bằng chứng một thuật toán tốt hơn trong cùng ngân sách.

## 5. Đọc và trình bày kết quả

Một thí nghiệm có một README: **câu hỏi → cấu hình/lệnh → từng mẫu → tóm tắt → kết luận/giới hạn**, kèm raw/log trong cùng thư mục. Không tạo thêm status/handoff trùng nội dung.

Đọc `result.json`, từng `turn-*.json`, warmup riêng và `memory.jsonl`/`watchdog.json`. Báo token thực tế, finish reason, trung vị và khoảng biến thiên; tách lượt hội thoại. Giữ run lỗi nguyên trạng. Khi đối chứng cần kiểm khối lượng token, lịch sử và đầu ra; hội thoại sinh lịch sử khác nhau có thể dẫn đến routing khác nhau.

Số liệu cũ và phạm vi kết luận tra ở [trang benchmark chính](README.md). Lệnh kiểm lại/tái lập từng thí nghiệm kỳ 1 tra ở sổ tay; không sửa hồ sơ cũ chỉ để khớp mặc định runner mới.

Kết quả phép hẹp mục3.6: [hồ sơ model thật i5](experiments/2026-11-05-kv-reuse-narrow-i5/README.md) — on/off đều2/2request hoàn tất; KVcapacity256→128MiB, đầu ra2/2khớp. PeakRSS on2893,664/off2945,574MiB; on có swap43,121MiB/off0 và pressure khác, nên chỉ là tín hiệu RSS trong một cặp. Giữ mặc địnhoff; không coi là nghiệm thu tổngRAM hoặc speedup.

## Chẩn đoán allocator trên i5 (08/11, sau kiểm fixture)

Một process mới, đúng12request mixed/3nhóm4request, CPU trước load/4threads/victim/cache64MiB-lớp/context512/max64/keepcache/lookahead off/KVreuse off/memorytrace on. Chỉ một lượt; dừng nếu runner lỗi/watchdog, không nới ngưỡng hoặc lặp để tìm kết quả mong muốn. Dùng binary/source đã kiểm fixture allocator; đây là instrumentation mới, không so tốc độ với run trước.

Giữ guardRSS5120/headroom1024MiB, watchdogRSS5632/available1536/processswap128MiB. Preflight3mẫu5s, hoãn nếupswpout tăng hoặcavailable dưới1536MiB. Không tests/build/inference đồng thời.

Đọc RSS và allocator (arena, in_use+mmapped, free) ở cùng stage, theo request và endpoint2–12. Ưu tiên bằng chứng giữ trang nếu indicator cuối request chỉ dao động nhỏ (tham khảo12MiB) trong khi RSS tăng lớn (tham khảo150MiB), free tăng sauadmission. Nếu indicator tăng tương ứng đáng kể (tham khảo≥50%deltaRSS khiRSStăng) thì cầncallsite; đó chưa tự làleak. Nếu cảhai dấu hiệu không rõ, telemetrythiếu/interposition hoặcRSSkhôngtáihiện thì kếtluậnchưaphânbiệt, khôngép hai nhánh. Đây làngưỡngchẩnđoánđềxuất, không tolerancefixture/địnhluật; khôngbắtrunmới cójump128MiBởrequest3.

Preview rồi bỏ --dry-run khi preflight đạt:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --memory-trace on --kv-buffer-reuse off --workload mixed --cache 64 \
  --context 512 --cycles 12 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty --dry-run
```

Dirty chỉ exploratory; record/checkpoint/sourceZIP/binary/counterversion phải giữ. Query_ns chỉ đo lời gọi, không toànoverhead; TIDboundary khôngnhậndiệnmọiRayonworker. Khôngsuyexactliveheap/hếtleak/plateau hoặcđónggiaiđoạn1.

## Gate–Up INT8 SIMD A/B trên i5 (gate-up-int8-v1, 08/11)

Câu hỏi: kernel ghép Gate/Up đã có từ Huy có nhanh hơn hai dotSIMD riêng trên cùng dữ liệu INT8 không, và lợi ích còn bao nhiêu khi tính thêm downprojection? Chỉ benchmark kernel, chưa đo model thật hoặc thay mặc định production.

Protocol chốt trước chạy:
- Shape hidden2048/intermediate1024 tương ứng expertOLMoE đang dùng, nhưng **weights/input tổng hợp**, không checkpoint/model7B. Recipe xorshift64v1/seeds17,73,991 nằm trong source snapshot.
- Baseline separate_simd gọi hai dotINT8 SIMD, cùng scales/SiLU/downprojection; candidate paired_simd gọi fused_gate_up_swiglu_seq, full expert dùng production forward_single_threaded. Không dùng scalar làm baseline yếu.
- Hai scope gate_up_silu và full_expert. Mỗi seed/scope warmup10lần/variant,9cặp AB/BA đổi thứ tự,100iterations mỗi sample. Tổng108sample,27cặp mỗi scope.
- Một compute thread; RAYON_NUM_THREADS=1/test-threads1. Yêu cầu AVX2+FMA thực; nếu thiếu thì fail chứ không lặng lẽ đo scalarfallback. Không pin CPU/tần số, phải ghi giới hạn desktop/nhiệt/OS.
- Buffer cấp sẵn và resident, không I/O/allocate trong vùng timing; parityfinite/maxabs<2e-5 kiểm trước timing. Unit test có kích thướcđuôi ngoài shape chính. Không đổi tolerancefixture hiện có.
- Chỉ đúng một lần thực thi protocol. Không model/build/tests khác đồng thời; build release hoàn tất trước subprocess timing. Dirty cho phép khảo sát với sourceZIP/hash/binary, không coi nghiệm thu revision sạch.
- Điều kiện ứng viên đủ đáng thử trên model: median tỷ số separate/paired của full_expert≥1,05 và candidate thắng≥7/9cặp ở từng seed. Nếu không đạt, không tự mở A/B7B hoặc lặp tới khi có lợi. Đây là ngưỡng quyết định thực nghiệm, không bằng chứng tăngtốc toàn inference.

Preview:

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --gate-up-kernel \
  --threads 1 --allow-dirty --dry-run
```

Bỏ --dry-run để thực thi sau checks/preflight. Runner tạo sourceZIP/result/kernel/build/benchmarklogs riêng dưới runs/i5; schema synthetic có checkpointnull. Giữ rawtừngmẫu, thời gian build tách khỏi kernel, validate đủpairs/order/parity, summary côngbố từng seed. Unit test ignored chỉ chạy qua protocol này; không cộng thời gian test harness vào timing từngsample.

## Gate–Up model INT8 A/B (gate-up-model-v1, 08/11)

Câu hỏi sau microbenchmark: trên cùng modelINT8 và cấu hìnhCPU, selector separate/paired có giữ HTTPoutput/workload và giảm expert_compute hoặc thời gian toànrequest không?

Chốt trước tải:
- Một chuỗi4process mới theo separate→paired→paired→separate (ABBA), mỗiprocess2request độc lập: Where is Mars rồi Explain a Python dictionary. Tổng8request,2cặp cho mỗi prompt. Khôngwarmup tách; chỉreset đầu vàkeepcache trongblock. Enginecold không đồng nghĩa OScachecold vì hashcheckpoint trước load.
- CPU trước load/4threads/victim/cache64MiB-lớp/context512/max64, cùngcheckpoint/binary/source. KVFP32/reuseoff/lookaheadoff/memorytraceon, greedy/temp0/top-p1/top-k0/repetition1, hotpin0, khôngroutefile/persistence. Từ chối checkpoint có.moe_usage để tránhstateẩn.
- MOE_GATE_UP_MODE đọc một lần lúcload, mặc địnhpaired; chỉ đổi mode giữa cácprocess. Status/metrics phải đúngmode, expert_quantization=int8, khôngFP8/VRAM/sessiondisk. Fixturelogits/token/prefixparity vàfullchecks trước.
- Guard5120/headroom1024/watchdogRSS5632/available1536/processswap128MiB giữ nguyên. Preflight3mẫu5s, hoãn nếupswpout tăng/availablethấp, khôngnới ngưỡng hoặc chạybuild/test/inference khác đồng thời.
- Dừng toànchuỗi nếu block lỗi/watchdog, source/binary/checkpoint/config drift, text/usage/finish khác hoặc expertcache/fileworkcounters khác cùngprompt. Giữ cảfailure, không lặp tìmkếtquảđẹp.
- Ghi từngpair/timer vàmemory từngblock; chỉ2cặp/prompt nên tín hiệu hữu hạn, không khái quát phầntrăm mọiworkload. Không cộng timers chồnglấn hoặc dùngpayload làmSSDphysicalbytes. HTTP không cho logit/tokenIDparity7B.
- Đây là protocol mới, không ghép trực tiếp thời gian với microbenchmark hoặc lượtRAM trướcmerge. Nếu không thấy lợiíchend-to-end thì ghi giới hạnI/O/scheduling, khôngtựclaimchat nhanh16%.

Preview, rồi bỏ --dry-run khi checks/preflight đạt:

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --gate-up-model-ab \
  --threads 4 --cache 64 --context 512 --max-tokens 64 --lookahead off \
  --allow-dirty --dry-run
```

Runner dùng serverrelease đãbuild trước, khôngbuild giữa4block; dữ liệu riêng dưới runs/i5/run-ID/block-*/i5/child-run-ID, topresult giữmapping/pins vàpairedsummary. Không gọi kếtquảdirty làbaseline nghiệmthu sạch.

Kết quả lượt08/11 được lưu tại [hồ sơ model INT8 i5](experiments/2026-11-08-gate-up-model-int8-i5/README.md):8/8request completed và output/workload khớp; hai cặp trái chiều, I/O/swap khác nhau nên chưa xác nhận speedup toàn request. Giữ nguyên protocol và toàn bộ mẫu, không chạy lại để chọn phần trăm có lợi.

## Gate–Up weights thật resident, lượt cuối (gate-up-resident-v1, 09/11)

Mục đích: khép đánh giá Gate–Up sau HTTP A/B chưa kết luận; kiểm compute với trọng số/scales INT8 thật, không đổi model/precision. Không triển khai kernel mới trong bước này.

Chốt trước đo:
- Một run duy nhất, CPU/Rayon4, AVX2+FMA thực; layers0/7/15, expertIDs0..7 cố định trước. Load24payload trước timing, tổng khoảng144,375MiB, budgetpayload256MiB; không load dense/model forward/GPU. Inputs là synthetic xorshift64 theo seeds17/73/991, không phải hidden states lấy từ hội thoại.
- Hai scope full expert (Gate/Up/SiLU/Down): expert_tasks8 gọi forward_single_threaded trên8expert qua Rayon, giống cách chia task decode; row_parallel gọi forward_reuse trênexpert0, tương ứng đường row-parallel trong engine. Không bao gồm routing/reduction/attention/file-I/O hoặc cả vòng decode.
- Cùng weights/scales/input/scratch/threads; chọn mode ngoài timing. So separate (hai SIMDdot) và paired; không scalar baseline. Tolerance maxabs<2e-5/finite trước timing, không đổi expected.
- Mỗi layer/seed/scope:10warmup/variant,9cặp AB/BA luân phiên,40iteration/sample. Tổng18caseparity/324sample,81cặp/scope. Buffers cấp trước; không claim Rayon không có allocation nội bộ.
- /proc/self/status và io đo trước/sau toàn vùng compute: VmSwap phải0, HWM≤1024MiB và read_bytes không tăng; nếu không đạt thì không dùng kết quả làm bằng chứng compute sạch. Preflight không inference/build khác/available≥1536MiB, pswpout không tăng qua3mẫu5s; không đổi OS/guard/tần số/affinity để lấy kết quả đẹp.
- Quy tắc primary: expert_tasks8 median tỷ số separate/paired≥1,05 và paired thắng≥7/9 ở mọi layer/seed. row_parallel là phụ, không dùng để cứu primary thất bại. Chỉ là tiêu chí chọn kỹ thuật, không kiểm định thống kê hoặc speedup toàn request.
- Nếu fail quy tắc, ghi không chứng minh cải tiến ở workload này và dừng hướng. Nếu pass, chỉ ghi lợi ích compute hẹp của kernel đã có; vẫn chưa có lợi ích end-to-end. Cả hai trường hợp khép Gate–Up và chuyển trọng tâm sang expert I/O; không thêm vòng HTTP để tìm phần trăm có lợi.
- Giữ toàn bộ mẫu/raw kể cả failure; không lặp run. Hashcheckpoint trước timing, build/tests hoàn tất trước testbinary được đo; dirty exploratory công bố rõ.

Lệnh preview: python3 tools/run_benchmark.py --testbed i5 --gpu cpu --gate-up-resident --threads 4 --allow-dirty --dry-run

Sau preview/checks/preflight, bỏ --dry-run để chạy. Runner giữ source ZIP/hash/binary/checkpoint/hardware và từng mẫu; không so thời gian trực tiếp với revision/scope khác.

## Admission cache theo tần suất (cache-admission-v1, 09/11)

Phần mới sau khi dừng Gate–Up primary: giảm nạp lại expert ở cùng cache RAM. Candidate CPU-only opt-in MOE_CACHE_ADMISSION=frequency; mặc định always. Khi miss đã tính xong và cache cần evict, so heat hiện có của newcomer với victim theo thứ tự cũ. Nếu newcomer lạnh hơn và một victim đủ chỗ cho nó, bỏ admission newcomer, giữ victim; ties/hits/room available theo hành vi cũ. Không dự đoán expert, không đổi routing/precision/sampling hoặc tăng cache. Heat vẫn là lịch sử engine, không tần suất tối ưu biết trước tương lai.

Protocol chốt trước chạy:
- Một ABBA duy nhất, always→frequency→frequency→always; mỗi process hai phiên độc lập Mars/Python, reset đầu block rồi keepcache, max32 mỗi request; tổng8request. Không warmup riêng.
- CPU4/victim/cache64MiB-layer/context512; paired Gate–Up cố định ở cả hai phía, FP32KV/reuseoff/lookaheadoff/traceon, greedy/temp0/top-p1/top-k0/repetition1/hotpin0, không usage side state/disk session. Cùng binary/source/checkpoint, hash checkpoint trước mỗi block làm ấm OS cache.
- Guard5120/headroom1024/watchdogRSS5632/available1536/processswap128MiB giữ nguyên. Preflight3mẫu5s/available≥1536MiB/pswpout không tăng; không build/test/inference khác trong timing.
- Trước model: hồi quy cache nóng bị thay trước/sau bản vá, MHA/GQA logits tolerance2e-5 + token/prefix parity dưới budget một expert; full Rust debug/release/Python/web và HTTP fixture nhỏ.
- Dừng nếu lỗi/watchdog/config/source/binary/checkpoint drift hoặc text/usage/finish khác. Counter cache/file được phép thay đổi: đó là mục tiêu của policy; metadata admission và bypass phải đúng mode, baseline bypass0.
- Primary: số file loads VÀ payload đọc giảm ít nhất5% ở từng prompt trong cảhai cặp. Đó là bằng chứng ít công việc nạp hơn trong workload này; không đồng nghĩa giảm5%SSDphysical hoặc speedup5%.
- Ghi từng mẫu timing/RSS/pressure kể cả nghịch chiều. Timing là mô tả trong cửa sổ có OS-cache/pressure khác nhau; không cộng timer chồng lấn, không nghiệm thu ổn định RAM dài hạn. Không bỏ mẫu, không rerun hoặc nới guard nếu thất bại. Không bật mặc định candidate chỉ vì một run có lợi.

Lệnh preview: python3 tools/run_benchmark.py --testbed i5 --gpu cpu --cache-admission-ab --threads 4 --cache 64 --context 512 --max-tokens 32 --lookahead off --allow-dirty --dry-run

Bỏ --dry-run sau checks/preflight. Không so thời gian trực tiếp với Gate–Up max64 hoặc revision cũ.

Kết quả09/11: [Gate–Up resident](experiments/2026-11-09-gate-up-resident-int8-i5/README.md) không đạt primary, đã khép; [cache admission](experiments/2026-11-09-cache-admission-int8-i5/README.md) tăng file loads/payload ở cảhai cặp, candidate đã gỡ khỏi code chính. Lệnh --cache-admission-ab phía trên là protocol lịch sử của source ZIP candidate, không còn option của CLI hiện tại. Chỉ tái lập trên checkout riêng từ snapshot; không giải nén đè workspace hoặc tự bật lại candidate. Quy tắc/mẫu gốc không thay đổi.

## Kernel expert nhiều token (expert-batch-int8-v1, 09/11)

Được người dùng đồng ý sau khi khép Gate–Up/admission. Câu hỏi mới: tính2/4token cùng expert trong một kernel có giảm chi phí compute prefill so với các matvec độc lập hiện hành không? Chỉ tích hợp production sau khi đạt quy tắc compute.

Chốt trước đo:
- Một run CPU/Rayon4/AVX2+FMA thực, real INT8 weights/scales ở layers0/7/15, expert0 cố định; hidden2048/intermediate1024. Ba payload nạp trước timing,18,046875MiB dưới budget256MiB; không load dense/full model/GPU. Activations synthetic xorshift64 seeds17/73/991, không hidden states hội thoại.
- Hai scope batch2/batch4. Baseline gọi production forward_reuse riêng từng token theo thứ tự, chia hàng qua Rayon4, Gate–Up paired cố định. Đã kiểm cfg(not(test)): dispatch_width=1; nhánh nhiều task token song song cũ chỉ cfg(test), không dùng làm baseline. Candidate chia theo hàng, chuyển INT8 sang FP32 trong register rồi dùng cho nhiều token; cả Gate/Up/SiLU/Down và hai lượt đổi layout đều nằm trong timing. Cách chia task thay đổi có chủ đích; không quy toàn bộ gain chỉ cho tái sử dụng trọng số.
- Cùng weights/input/scales/số token; buffer cấp trước. Tổng capacity scratch/input của cảhai biến thể trong mỗi case≤256KiB, báo số thực. Không mở rộng toàn ma trận trọng số FP32 hoặc claim không có allocation nội bộ Rayon.
- Mỗi layer/seed/scope parityfinite/maxabs<2e-5;10warmup/variant,9cặp AB/BA luân phiên,40iteration/sample. Tổng18paritycase/324sample,81cặp/scope. Kiểm unit signed/tails/invalid shape/nonfinite/precision trước.
- Primary: CẢ batch2 VÀ batch4 có median tỷ số separate_tokens/shared_weights≥1,10 và candidate thắng≥7/9 ở mọi layer/seed. Đây là tiêu chí chọn kỹ thuật, không kiểm định thống kê. Không đạt thì khép prototype, không tích hợp hoặc rerun để tuning số đẹp. Đạt mới tích hợp prefill và kiểm fixture/logits/token/RAM trước một phép TTFT model riêng.
- /proc trước/sau compute: VmSwap0,HWM≤1024MiB,read_bytes không tăng; nếu vi phạm thì không dùng như bằng chứng compute sạch. Preflight3mẫu5s/available≥1536MiB/pswpout không tăng, không engine/build khác. Không pin tần số/affinity hoặc đổi OS/guard. Full tests/build kết thúc trước timing testbinary.
- Prototype ban đầu chỉ cfg(test), chưa tác động inference. Một run duy nhất; giữ mọi mẫu/error, không cộng timer hoặc suy giảm decode latency/chat speedup từ kernel resident.

Dùng chung harness resident hiện có; lệnh preview:
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --expert-batch-kernel --threads 4 --allow-dirty --dry-run

Sau checks/preflight, bỏ --dry-run. Raw chỉ mở để kiểm các trường/claim liên quan; tóm tắt được xếp P2 khi kết thúc bước.

## Prefill nhiều token trên model thật (expert-batch-model-v1, 09/11)

Kernel expert-batch-int8-v1 đã đạt ngưỡng compute cho cảbatch2/batch4 trước bước này. Candidate tích hợp CPU/INT8 opt-in MOE_EXPERT_BATCH_PREFILL=on, mặc địnhoff; không đổi model/precision/cache/predictor. Scalar-token fallback cho nhóm lẻ1, các nhóm còn lại4/2, giữ route order khi ghi contributions và Top-K reduction.

Protocol chốt trước tải:
- Một ABBA off→on→on→off,4process mới/mỗiprocess2phiên độc lập Mars/Python, reset đầu block/keepcache. Tổng8request,max1 để cô lập prefill/TTFT; không suy tốc độ decode dài. Không warmup riêng.
- CPU4/victim/cache64MiB-layer/context512, cùng binary/source/checkpoint, Gate–Up per-token paired cố định, FP32KV/reuseoff/lookaheadoff/traceon/greedy/temp0/top-p1/top-k0/repetition1/hotpin0, không usage side state/disk session. Hashcheckpoint làm ấm OS cache, không SSD-cold đồng nhất.
- Scratch batch2+batch4 là48×(hidden+intermediate)byte=147456B/144KiB ở model thật, local trong prefill. Tính vào reserve request và prefill_capacity_bytes; không tăng cache RAM. Mốc after_prefill không giữ capacity local này.
- Unit/fixture trước: SIMD signed/tails, invalid shapes/nonfinite/precision, logits tolerance2e-5 ở MHA/GQA và tailroute lengths, session/token/finish/cachedprefix parity, counter và memoryaccounting; full debug/release/Python/web. HTTP fixture release nhỏ kiểm mode/calls/trace trướcmodel.
- Status/metrics phải đúngflag, candidate phải cóbatchcalls>0, baselinecalls0, max1 quan sát đúng. Text/usage/finish và5cache/fileworkcounters phải khớp mọi block; dừng nếu mismatch/source/binary/checkpoint/config drift/error/watchdog. Calls đếm expert-kernel applications, không là số tokenprompt duy nhất.
- Primary: mỗi prompt trong CẢhai cặp phải có off/on TTFT VÀprefill ratio≥1,05. Ghi từng mẫu kể cảbấtlợi và secondary expert_compute/clientwall/RSS/pressure. Không đổi tiêu chí/tuning/rerun nếu không đạt. Một lượt exploratory nhỏ không chứng minh mọi workload/máy hoặc toàn bộ chat nhanh tương ứng kernel.
- Guard5120/headroom1024/watchdogRSS5632/available1536/processswap128MiB; preflight3mẫu5s/availableđủ/pswpoutkhôngtăng, không build/test/inference khác đồngthời. Không nới guard/đổi OS hoặc loại mẫu để đạt.

Lệnh preview:
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --expert-batch-model-ab --threads 4 --cache 64 --context 512 --max-tokens 1 --lookahead off --allow-dirty --dry-run

Dùng chung wrapper model A/B đã có; bỏ --dry-run sau checks/preflight. Kernel và model là hai source snapshot khác nhau trong cùng hồ sơ, giữ raw riêng và không thay hash hồi tố.

Kết quả hai giai đoạn expert-batch09/11 nằm trong [một hồ sơ P2](experiments/2026-11-09-expert-batch-int8-i5/README.md): kernel đạt tiêu chí, đã tích hợp opt-in/off mặc định; model expert_compute thấp hơn ở4so sánh nhưng TTFT không đạt do một cặp bất lợi. Không sửa rule hoặc bỏ mẫu, không chạy lại để chọn số đẹp.

## Cửa sổ RAM hữu hạn S1 (ram-window-v1, 10/11)

Bước S1 của [PLAN](../docs/PLAN.md#s1-kiểm-ram-hữu-hạn-trước-phép-đo-model-mới). Câu hỏi: với source/binary hiện hành và cấu hình i5 khởi điểm, một process có hoàn tất24request mixed trong gate an toàn không, RSS cuối chuỗi phẳng tới đâu? Đây là kiểm vận hành hữu hạn, không đo tốc độ, không chọn cache chung, không so i5/i7.

Chốt trước chạy:
- Nguồn: engine/tools 56 file hash và binary d12a3939a7405d483a665e8642fa8c3404176e31decb0dad6eece7cd7b90c3fc trùng snapshot model09/11; toolchain rustc1.99.1 không đổi. Tái sử dụng kiểm Rust debug/release133tests/profile của đúng source đó; kiểm docs/tooling mới bằng tools/check.py. Không build/tests/inference khác trong khi chạy.
- Một process mới, workload mixed v1 (Mars, follow-up cùng phiên, Python phiên mới, Love phiên mới) ×6 nhóm =24request, max64, reset chỉ trước request1/keepcache. CPU trước load/4threads/victim/cache64MiB-layer/context512, Gate–Up per-token paired, lookahead off, KVreuse off, batch prefill off, memorytrace on, FP32KV/greedy, không usage side state/disk session.
- Guard RSS5120/headroom1024MiB; watchdog RSS5632/available1536/processswap128MiB, sampler250ms. Không nới ngưỡng hoặc đổi OS/allocator.
- Preflight3mẫu cách5s: không engine/build/benchmark khác, pswpout không tăng, MemAvailable tối thiểu≥5120MiB (dự toán RSS đỉnh khoảng3,5GiB từ RAM08/11 cộng sàn watchdog1536MiB). Không đạt thì hoãn, không chạy.
- Gate an toàn: 24/24request completed; không OutOfMemory, guard từ chối hoặc watchdog violation; sampled RSS và VmHWM≤5120MiB; báo processVmSwap max (khác0 thì không gọi "không swap").
- Chỉ báo bổ sung: RSS sau từng request và spread max−min của request17–24; ≤64MiB phân loại "phẳng trong cửa sổ", vượt thì "chưa phẳng". Không thay gate an toàn, không tự kéo thêm request.
- Đọc capacity/trace cùng mốc với RSS; RSS tăng không tự là leak. Một lượt duy nhất; lỗi/watchdog thì giữ raw và dừng, không lặp ngay.

Lệnh (bỏ --dry-run sau preflight; thêm --allow-dirty chỉ khi tree dirty và ghi exploratory):

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --threads 4 \
  --cache 64 --context 512 --cycles 24 --max-tokens 64 --workload mixed --keep-cache \
  --lookahead off --memory-trace on --kv-buffer-reuse off --expert-batch-prefill off \
  --gate-up-mode paired --ram-limit-mib 5120 --headroom-mib 1024 \
  --stop-rss-mib 5632 --min-available-mib 1536 --stop-swap-mib 128 --dry-run
```

Đạt gate chỉ cho phép viết "vận hành được trong cấu hình/cửa sổ đã thử" và mở S2 có giám sát; không hard cap/hết leak/plateau dài hạn, không tự đóng giai đoạn1.

**Kết quả10/11:** [hồ sơ P1](experiments/2026-11-10-ram-window-i5/README.md) run20261110T043843.817290Z-ram-9d3521ed…, HEAD2f64752 sạch. Gate an toàn ĐẠT: 24/24completed, không violation/OOM, RSS/HWM max3188,621MiB, VmSwap engine0, availablemin6596,8MiB. Spread request17–24 13,980MiB ≤64 → phẳng trong cửa sổ; request1→24 tăng300,887MiB, tracked cố định2909,011MiB. pswpout hệ thống tăng238965page trong run (process khác, tác nhân chưa xác định). Không lặp lượt này.
