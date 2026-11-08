# CPU trước load và kiểm RAM i5 — 04/11/2026

**Bốn request hoàn tất, không vi phạm watchdog.** Model chọn CPU trước load, không tạo dense GPU/pool/pinned buffer; status ghi VRAM 0 và bản đồ thư viện của child không có CUDA/cuBLAS. Đây là kiểm ổn định hữu hạn khoảng 126 s, không chứng minh hết leak/OOM dài hạn hoặc tăng tốc.

**Cập nhật sau đó:** lượt chẩn đoán lookahead một request bị dừng do swap trước khi hoàn tất. Kết quả “bốn request đạt” ở trên chỉ áp dụng lượt kiểm RAM ban đầu; xem mục chẩn đoán bên dưới.

## Thay đổi và điều kiện

`MOE_GPU_MODE=cpu` bỏ bước probe CUDA ngay từ đầu. Runner `--gpu cpu` tự đặt biến này; `auto` giữ hành vi dò GPU/fallback hiện có. API typed GpuMode cho library/examples chọn trước load. Khi đã chọn CPU, toggle/set GPU không khởi tạo muộn; cần restart bằng auto.

Lượt này yêu cầu CPU, 4 Rayon threads, cache 64 MiB/lớp, context 512, max 64 token, victim, lookahead tắt, giữ expert cache giữa request. Guard 5 GiB/headroom 1 GiB; watchdog RSS 5,5 GiB/MemAvailable 1,5 GiB/swap 128 MiB, lấy mẫu 250 ms. Nguồn dirty, snapshot ZIP/hash/binary/checkpoint và hardware nằm trong [record](run/result.json); không nhận là baseline revision sạch.

Sau build release và checks, đã kiểm không có server khác cùng tên rồi chạy:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --cache 64 --context 512 --cycles 4 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty
```

Không tăng ngưỡng so với lượt dừng 03/11. Không chạy build/tests/model khác đồng thời. Bốn đầu vào: `Where is Mars`; `What planet did I just ask about?` giữ lịch sử câu đầu; `Explain a Python dictionary.` phiên mới; `What is love ?` phiên mới. Mỗi request sinh 64 token và kết thúc length. Khác lịch sử/bối cảnh máy so với phép thử topics bị dừng, không tính phần trăm giảm RAM hoặc tăng tốc giữa hai lượt.

## Quan sát

[Status trước request](run/status-start.json): gpu_mode_at_load=cpu, lookahead_enabled=false, GPUenabled=false, allocated_vram_bytes=0, dense_offloaded=false. Record có cpu_start_library_probe.available=true/cuda_libraries=[] qua /proc của chính child.

[Time series](run/memory.jsonl) có 503 mẫu trong 125,73 s: RSS/HWM cao nhất 2,954 GiB; swap tiến trình cao nhất 128 KiB (0,125 MiB), thấp hơn ngưỡng 128 MiB; MemAvailable thấp nhất 4,656 GiB; watchdog_violations rỗng. Không gọi swap bằng 0 trong toàn lượt.

| Request | RSS sau request (GiB) | Swap sau request (KiB) | Lần đủ điều kiện dự đoán ở decode |
| --- | ---: | ---: | ---: |
| 0 | 2,822 | 0 | 0 |
| 1 | 2,853 | 0 | 0 |
| 2 | 2,913 | 0 | 0 |
| 3 | 2,954 | 8 | 0 |

RSS còn tăng trong khoảng quan sát; chưa chứng minh plateau hoặc phân loại toàn bộ tăng là cache/allocator/leak. Status memory capacity và sampled RSS có phạm vi khác nhau.

## Ý nghĩa cho bước tiếp

Mục tiêu bỏ khởi tạo GPU khi chọn CPU đã được kiểm bằng status, bản đồ thư viện trên host và hồi quy logits MHA/GQA/toggle/reset. Lượt RAM này đạt các ngưỡng quan sát đã đặt; chưa xác định nguyên nhân swap của ngày trước hoặc quy mọi khác biệt cho bản vá.

Counter eligible_layers vẫn được tính khi lookahead tắt; cả 4 request đều 0. Ở workload/ngân sách này chưa có cơ hội đủ điều kiện “lớp hiện tại không miss” để dự đoán lớp kế tiếp. Không dùng proposed=0 của chế độ tắt để kết luận vô ích; phát hiện liên quan là eligible_layers=0 độc lập công tắc. Chưa chạy đối chứng 40 request hoặc tự tăng cache/thay gating để tạo kích hoạt. Cần rà điều kiện kích hoạt và mục tiêu phép thử trước khi quyết định đo tiếp; không gọi prefetch nhanh/chậm từ kiểm RAM này.

Payload chuyển nguyên bộ từ runs sang `run/`, giữ một bản duy nhất. 42 hash trong sourceZIP khớp, chín tệp bằng chứng được khóa bằng CHECKSUMS.sha256. Kiểm từ thư mục này:

```bash
sha256sum -c CHECKSUMS.sha256
```


## Bổ sung bộ đếm và lượt chẩn đoán lookahead

Đã thêm histogram số expert còn thiếu ở lớp hiện tại và các lý do chặn/không có ứng viên. [Định nghĩa và lệnh](../../SCENARIOS.md#32-chẩn-đoán-vì-sao-lookahead-không-kích-hoạt) phân biệt lớp cuối, đang thiếu expert, chế độ tắt, chưa có điểm dương và ứng viên dự đoán đã có trong cache. Không đổi gating hoặc xếp hạng; fixture ép các nhánh và kiểm logits trong tolerance cũ. Không điền trường mới vào lượt kiểm RAM ban đầu.

Sau checks và một fixture localhost, preview rồi gọi đúng một request chẩn đoán bằng lệnh đã nêu ở SCENARIOS (bỏ --dry-run, giữ --allow-dirty). Giữ CPU 4 threads/cache 64 MiB-lớp/context 512/max 64, bật lookahead, reset/phiên mới; không tăng guard/watchdog. Checkpoint/source/binary/hash và [record riêng](diagnostic-run/result.json) giữ nguồn đang sửa (exploratory), khác snapshot lượt kiểm RAM ở run/.

Lượt chẩn đoán có status watchdog_stopped, 0 request hoàn tất. Watchdog dừng ở 5,51 s khi VmSwap 132304 KiB = 129,20 MiB, vượt 128 MiB. Có 23 mẫu, RSS mẫu cao nhất 2,225 GiB và HWM cao nhất 2,227 GiB. Mẫu dừng ghi RssAnon 2256552 KiB/RssFile 9168 KiB, MemAvailable 5520372 KiB và swap toàn máy dùng 3515124 KiB (khoảng 3,35 GiB trên tổng 4 GiB). Đây là quan sát hệ thống của lần này, chưa xác định tác nhân reclaim/swap. Không quy khác biệt với lượt kiểm RAM trước cho riêng lookahead hoặc bộ đếm mới.

Status CPU/0 VRAM và probe thư viện CUDA/cuBLAS rỗng vẫn được lưu. Request chưa hoàn tất nên không có histogram/metric generation thực để diễn giải. Từ mã và ý nghĩa eligible_layers của bốn request cũ, có thể xác định rào chặn trực tiếp là vẫn còn ít nhất một miss ở các lớp có lớp kế tiếp; chưa biết phân bố thiếu bao nhiêu expert. Không quy việc chưa gọi predictor thành điểm dự đoán kém. Root error SSE thiếu finish/DONE là hậu quả child bị watchdog dừng; không có kết luận mới về thiếu 1, 2 hay nhiều expert, chất lượng predictor, speedup hoặc hết leak.

Giữ năm file diagnostic-run nguyên trạng, một bản duy nhất chuyển từ runs; 42 source hash trongZIP khớp. Checksum riêng, không sửa CHECKSUMS của lượt kiểm RAM trước:

```bash
sha256sum -c DIAGNOSTIC_CHECKSUMS.sha256
```

Chưa chạy thêm model hoặc thay gating. Bước tiếp cần điều kiện máy ổn định rồi lấy một mẫu chẩn đoán; không bắt đầu phép đối chứng tốc độ khi chưa biết cơ chế có cơ hội hoạt động.

## Lượt chẩn đoán tiếp theo hoàn tất

Sau yêu cầu tiếp tục của người dùng, chạy đúng một request theo [SCENARIOS 3.2](../../SCENARIOS.md#32-chẩn-đoán-vì-sao-lookahead-không-kích-hoạt), bỏ `--dry-run`, giữ các cờ khác. CPU trước load, 4 threads, victim, cache 64 MiB/layer, context 512, tối đa 64 token, lookahead bật; guard/watchdog giữ nguyên. Source và binary khớp hash lượt diagnostic-run trước, không build/test/model khác đồng thời. Trước tải, MemAvailable khoảng 9,6 GiB, swap dùng 0, không tăng pswpout và PSI memory avg10 bằng 0 trong quan sát; các mốc này không bảo đảm an toàn mọi workload.

Lần gọi trong sandbox bị từ chối mở cổng localhost trước khi spawn server, chưa có model/request chạy. Gọi lại với quyền mở localhost được cấp; [record hoàn tất](diagnostic-completed-run/result.json) có run ID `20261104T144232.735540Z-ram-b0a4eb2ffb0e4383ba7843139a65bab9`. Nguồn dirty nên chỉ là khảo sát, không phải baseline nghiệm thu sạch. Checkpoint, source ZIP/42 hash, binary hash và hardware thực được giữ trong record. Status CPU/0 VRAM và probe không có thư viện CUDA/cuBLAS được lưu.

Request `Where is Mars`, phiên mới/reset đầu, có 19 prompt tokens, 64 completion tokens, finish reason `length`. Các bộ đếm trong [turn-0.json](diagnostic-completed-run/turn-0.json) nhất quán: evaluated=1008, no_next=63, blocked=945, eligible=0, prediction_calls=0, proposed/loaded/used/file_bytes speculative đều 0. Đây là 63 bước decode × 16 lớp, trong đó 15 lớp mỗi bước có lớp kế tiếp.

| Số expert còn thiếu trong top-8 | Lượt lớp có lớp kế tiếp |
| --- | ---: |
| 0 | 0 |
| 1 | 4 |
| 2 | 19 |
| 3 | 41 |
| 4 | 156 |
| 5 | 249 |
| 6 | 261 |
| 7 | 144 |
| 8 | 71 |

Tổng 945 lượt; trung bình thiếu 5,477 expert/lượt. Có 881/945 lượt (93,23%) thiếu 4–8 expert, chỉ 23/945 (2,43%) thiếu 1–2 expert. Rào chặn trực tiếp là điều kiện không có miss; predictor chưa được gọi nên chưa thể đánh giá điểm dự đoán hoặc chất lượng ứng viên. Nới nhẹ gate sang 1–2 miss cũng chỉ tạo cơ hội trong phần nhỏ của mẫu này; chưa đo lợi ích hoặc chi phí của thay đổi đó và không sửa gate để thử.

[Sampler](diagnostic-completed-run/memory.jsonl) có 139 mẫu, mốc cuối 34,560 s: RSS/HWM lấy mẫu cao nhất 2,817482 GiB; RSS snapshot sau request 2,817532 GiB, hơi cao hơn do khác thời điểm. VmSwap tiến trình và swap toàn máy bằng 0 trong các mẫu; MemAvailable thấp nhất 6,479691 GiB. Watchdog không vi phạm. Cuối request mỗi lớp giữ 10 expert, tổng cache capacity 962,5 MiB, recycled 0; session capacity khoảng 128,002 MiB. Các capacity này không cộng thành RSS và một request không chứng minh plateau/hết leak/OOM.

Timer request được giữ trong raw (TTFT 4,449548 s, decode 29,097854 s, tổng generate 33,547432 s); không dùng so tốc độ với lượt cũ vì điều kiện máy khác, không có cặp đối chứng. Payload expert file 38.376.898.560 byte khác `/proc` read_bytes delta 19.735.314.432 byte; không nhận payload là lưu lượng SSD vật lý hoặc phép đo băng thông.

Kết luận trong phạm vi mẫu: gate lookahead không phù hợp để tạo hoạt động dự đoán ở workload/ngân sách đã thử. Tạm khép nhánh tối ưu này; đề xuất dùng lookahead **off** trong bước kiểm xu hướng tổng RAM hữu hạn tiếp theo. Không sửa mặc định runtime, tăng cache/ngưỡng, chạy thêm request hay mở bộ đối chứng 40 request. Tổng RAM vẫn là vấn đề chưa đóng.

Chuyển nguyên sáu file từ runs sang `diagnostic-completed-run/`, giữ một payload duy nhất. Đường dẫn gốc ghi trong record/log vẫn giữ nguyên như khi đo. Không sửa raw hoặc checksum hai lượt trước. Kiểm riêng:

```bash
sha256sum -c DIAGNOSTIC_COMPLETED_CHECKSUMS.sha256
```
