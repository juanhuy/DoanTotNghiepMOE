# Khảo sát lookahead i5 — dừng theo watchdog

**Chưa có kết quả đối chứng bật/tắt.** Lượt model lớn dừng trong block đầu (lookahead tắt) vì swap tiến trình vượt 128 MiB. Chỉ một request hoàn tất; không suy ra prefetch nhanh/chậm, không nhận là model OOM hoặc leak đã được xác định.

## Câu hỏi và điều kiện

Kiểm nạp trước expert dự đoán trong cùng ngân sách, theo [lookahead-v1](../../SCENARIOS.md#31-thiết-kế-đối-chứng-prefetch-dự-đoán-03112026). Dự kiến 5 cặp AB/BA xen kẽ, mỗi block 4 câu, tổng 40 request; giữ cùng binary/checkpoint, chuyển riêng lookahead. Cache 64 MiB/lớp, context 512, max 64 token, victim, 4 Rayon threads, yêu cầuCPU. Guard 5 GiB/headroom 1 GiB; watchdog RSS 5,5 GiB/MemAvailable 1,5 GiB/swap 128 MiB, lấy mẫu 250 ms.

Nguồn đang sửa (`git_dirty=true`), HEAD 75d5a62 tại thời điểm đo. Đây là exploratory, không baseline revision sạch. [Record](run/result.json) giữ source.zip, 52 hash nguồn đầu vào, binary/checkpoint hashes và hardware quan sát. Không gọi HEAD là toàn bộ source đã chạy; source.zip là snapshot của lượt này. Không sửa raw để đổi trạng thái failed.

Runner đã yêu cầu tắt GPU sau load; filelog vẫn ghi khởi tạo CUDA/dense lúc load, nên không gọi đây là vận hành hoàn toàn không cấp VRAM. Lượt lỗi chưa lưu status_start riêng: không có snapshot trạng thái GPU sau toggle để mở kiểm độc lập. Bản runner sau lượt này đã thêm status-start/status-end và kiểm GPU thực sự tắt trước khi đo CPU; không bổ sung hồi tố snapshot vào raw cũ.

Lệnh đã gọi từ root với MOE_RAM_LIMIT_BYTES=5368709120 và MOE_RAM_HEADROOM_BYTES=1073741824:

```bash
python3 tools/run_benchmark.py --http-session --http-scenario topics \
  --compare-lookahead --testbed i5 --gpu cpu --threads 4 \
  --cache 64 --context 512 --max-tokens 64 --runs 5 \
  --model-dir models/olmoe-1b-7b-int8 --allow-dirty \
  --tag lookahead-v1-exploratory
```

Không chạy lại lệnh này ngay: cần kiểm ổn định bộ nhớ của cấu hìnhCPU hiện tại trước, không nâng ngưỡng watchdog để cố hoàn tất bộ đo.

## Điều gì đã xảy ra?

[Watchdog](run/block-00-off/watchdog.json) dừng child ở elapsed 74,37 s khi VmSwap 135756 KiB = 132,57 MiB. [Time series](run/block-00-off/memory.jsonl) có 298 mẫu: RSS/HWM quan sát cao nhất 3,242 GiB, MemAvailable thấp nhất 3,715 GiB. RSS chưa vượt ngưỡng 5,5 GiB; lý do dừng là swap. Các giá trị này không xác định nguyên nhân swap hoặc peak tuyệt đối giữa samples.

[Request duy nhất hoàn tất](run/block-00-off/turn-0-0.json): `Where is Mars`, 19 token đầu vào, 64 token sinh, finish length; TTFT 5,762 s, decode 1,462 steps/s, wall 48,865 s. Đây là một mẫu của chế độ tắt, không đại diện cấu hình và không dùng tính speedup. Các counter speculative đều 0, active payload đỉnh 50462720 B; chưa có chế độ bật để so.

HTTP bị ngắt nên root record báo `SSE missing finish or DONE`; nguyên nhân dừng child được ghi ở watchdog. Chưa hoàn tất block đầu, chưa chạy block bật, chưa có cặp. Giữ các phần chưa hoàn tất nguyên trạng; không tạo summary thành công hoặc lấy một mẫu này thay bộ 40 request.

## Kết luận và bước tiếp

Watchdog đã ngăn tiếp tục phép thử khi môi trường không đạt điều kiện ổn định đã đặt. Cần điều tra swap/RAM với workload CPU có giám sát, đồng thời phân biệt weights/dense/cache/KV/driver và tải nền; không kết luận “cache lớn gây OOM” hoặc “prefetch vô ích” từ lượt này. Chỉ tiếp tục đối chứng khi điều kiện bộ nhớ phù hợp; không tự đổi sang GPU hoặc giảm cache rồi gộp chung protocol cũ.

Payload được chuyển nguyên bộ từ runs sang thư mục `run/`, không tạo bản sao raw thứ hai. Kiểm bất biến từ thư mục này:

```bash
sha256sum -c CHECKSUMS.sha256
```

## Điều tra hồi cứu RAM/swap (03/11/2026)

Đọc lại cùng time series, không chạy lại model. Các ô dưới đây dùng MiB (KiB trong raw chia 1024):

| Thời điểm trong block (s) | RSS engine | Swap engine | RSS + swap engine | Swap toàn máy đang dùng |
| --- | ---: | ---: | ---: | ---: |
| 60,09 | 3317,43 | 1,09 | 3318,52 | 320,44 |
| 72,11 | 3319,91 | 1,09 | 3321,00 | 316,64 |
| 74,12 | 3241,92 | 79,40 | 3321,32 | 1093,99 |
| 74,37 | 3188,77 | 132,57 | 3321,34 | 1212,84 |

Trong 2,26 s cuối, swap toàn máy tăng khoảng 896,20 MiB, engine tăng 131,48 MiB. Tổng RSS+swap engine gần như giữ nguyên ở hai mốc 72,11/74,37 s; đây chỉ là chỉ báo thô, không phải tổng allocation hoặc bộ nhớ độc quyền chính xác. Dữ liệu phù hợp với một đợt thu hồi trang sang swap ở cấp hệ thống hơn là việc engine đột ngột cấp thêm hàng GB tại lúc dừng. Chưa xác định tác nhân gây đợt này, chưa loại trừ tăng bộ nhớ ở giai đoạn khác và không chứng minh không leak. Việc MemAvailable tăng khi swap tăng cũng không có nghĩa watchdog sai.

Đọc cấu hình và đường loader cho ước tính dense FP32: embedding/head, các projection/router/norm chiếm 1906843648 B (khoảng 1,776 GiB). Cache 64 MiB × 16 lớp có ngân sách 1 GiB. Hai phần này đã khoảng 2,776 GiB, chưa gồm KV, scratch, tokenizer, driver và allocator. Vì vậy RSS quanh 3,24 GiB tự nó không phải bằng chứng cache mất kiểm soát. Đây là phép tính từ kích thước tensor hiện tại, không phải profiler hồi cứu từng allocation của lượt lỗi.

Rà mã xác nhận `--gpu cpu` được áp sau load: loader đã probe GPU, tạo dense VRAM, pool và pinned buffer trước khi runner toggle. Đây là hạn chế về tính thuần CPU cần sửa cho phép thử tương lai. Không suy 1 GiB VRAM tương đương 1 GiB RAM hoặc nhận đây là nguyên nhân swap đã chứng minh.

Đọc host sau lượt lỗi thấy swappiness 60 và không thấy memory.max/memory.high áp trần ở các cgroup v2 quan sát hiện tại. Đây là trạng thái đọc sau, không phải dữ liệu cgroup của child trong time series cũ và không đủ loại trừ giới hạn/áp lực tại thời điểm thử.

Bổ sung nhẹ cho sampler các lượt sau: RssAnon/RssFile/RssShmem, VmSize/VmData/VmLck/VmPin và MemTotal/MemFree/AnonPages/Cached/SwapCached/Mlocked/Unevictable/Dirty/Writeback/SReclaimable. Đọc thêm trường từ cùng procfile, không thêm profiler smaps mỗi 250 ms. Không điền hồi tố trường mới vào raw cũ; CHECKSUMS vẫn khóa 10 file của lượt cũ.

Trước phép đo tốc độ tiếp: cho phép chọn CPU ngay trước khởi tạo GPU; sau đó kiểm RAM/swap có giám sát trong một lượt ổn định riêng với ghi nhận tài nguyên mới. Nếu vẫn swap, tiếp tục điều tra áp lực toàn máy/cgroup và reclaim trước khi quyết định cấu hình; không tự tăng ngưỡng hay gọi mọi swap là OOM.
