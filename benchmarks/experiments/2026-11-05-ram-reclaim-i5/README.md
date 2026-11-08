# Chẩn đoán RAM/reclaim i5 với telemetry mới, 05/11/2026

## Câu hỏi và cấu hình

Một lượt hữu hạn theo [SCENARIOS3.3–3.4](../../SCENARIOS.md#34-đọc-telemetry-reclaim-mới-04112026), được chốt trước chạy: tối đa12request, giữ cache, lookahead tắt; xem swap-out/PSI và events/stat/giới hạn cgroup khi RSS/VmSwap đổi. Đồng thời kiểm phần tăng RSS có được giải thích bởi cache/session capacity cuối request không. Không đo tăng tốc và không chạy thêm lượt tự động sau kết quả.

CPU trước load, 4Rayonthreads, victim, cache64MiB/layer, context512, tối đa64completion tokens; guardRSS5GiB/headroom1GiB, watchdogRSS5,5GiB/MemAvailable1,5GiB/processswap128MiB giữ nguyên. Mỗi nhóm bốn request hỏi `Where is Mars`, `What planet did I just ask about?` cùng phiên/lịch sử thực, rồi `Explain a Python dictionary.` và `What is love ?` mỗi câu phiên mới. Chỉ reset trước request đầu, không warmup tách, không đổi session budget.

[Preflight](preflight.json) có ba mốc cách5s: MemAvailable9,238/9,339/9,397GiB; swap dùng0, pswpin/pswpout không tăng, PSI memory avg10=0. Kiểm60Python/7Node/catalog40 đạt trước tải,22input engine/Cargo và binary hash khớp lượt04/11, không thấy tiến trình engine/build liên quan. Không build/tests/inference khác đồng thời. Cửa sổ preflight không bảo đảm an toàn mọi workload.

Preview PASS rồi chạy với quyền mở localhost được cấp:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --cache 64 --context 512 --cycles 12 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty
```

[Record](run/result.json) giữ runID `20261105T023906.099843Z-ram-95e2def67276445eb0a12acf4c622b81`, HEAD75d5a62/dirty,44sourcehash/ZIP, binary/checkpoint/hardware. HEAD không đại diện mọi thay đổi chưa commit; đây là exploratory, không baseline nghiệm thu sạch. Hardware quan sát i5-10500H,12logicalCPU,RAM16.571.322.368B; GPU trong inventory nhưng statusCPU/0VRAM và probe không cólibCUDA/cuBLAS. Checkpoint files/hash khớp hồ sơ chuỗi04/11; tooling có thêm telemetry nên không so tốc độ hoặc gán khác biệt hai ngày cho code.

## Hoàn tất12/12request; RSS vẫn tăng

Status `completed`, không watchdog violation. Cả12request sinh64token/finish length. Prompt tokens theo pha19/107/22/20, lượt hội thoại dùng lại11prompt tokens. RSS sau request từ từng `turn-N.json`, đơn vịMiB:

| Request (từ1) | Nhóm/pha | RSS sau request (MiB) |
| --- | --- | ---: |
| 1 | 1/Mars | 2890,859 |
| 2 | 1/hội thoại | 2921,434 |
| 3 | 1/Python | 2983,480 |
| 4 | 1/Love | 3022,461 |
| 5 | 2/Mars | 3060,520 |
| 6 | 2/hội thoại | 3061,824 |
| 7 | 2/Python | 3099,297 |
| 8 | 2/Love | 3133,453 |
| 9 | 3/Mars | 3148,910 |
| 10 | 3/hội thoại | 3156,938 |
| 11 | 3/Python | 3157,605 |
| 12 | 3/Love | 3178,953 |

Cuối nhóm1→2 tăng110,992MiB, nhóm2→3 tăng45,500MiB; request1→12 tăng288,094MiB. Mức tăng giữa nhóm giảm nhưng vẫn dương, không gọi là plateau hoặc hết leak. Các snapshot cuối request đều cache962,5MiB/10expert mỗi lớp, recycled0, sessioncapacity128,001–128,003MiB trong budget256MiB. Các capacity này gần cố định, chưa giải thích phần tăngRSS; không bằng live allocation hoặc RSS toàn tiến trình.

[Time series](run/memory.jsonl) có927mẫu, mốc cuối234,829s. SampledRSS/HWMmax3,104446GiB, VmSwap tiến trình0 trong tất cả mẫu; MemAvailablemin7,129990GiB. Chỉ là cửa sổ hữu hạn12request, không chứng minh hard RAMcap/hếtOOM/ổn định dài hạn. Sampler đọc nối tiếp các file, không phải snapshot nguyên tử, không bảo đảm lấy được mọi peak.

## Reclaim/swap hệ thống có xảy ra, model không bị swap trong mẫu

Các delta dưới đây từ mẫu đầu đến cuối, page_size4096B:

| Bộ đếm toàn hệ thống | Delta |
| --- | ---: |
| pswpout | 86023page, tương đương336,027MiB |
| pswpin | 12943page, tương đương50,559MiB |
| pgscan_kswapd / pgscan_direct | 16283690 / 23148 |
| pgsteal_kswapd / pgsteal_direct | 16183527 / 21289 |
| pgscan_proactive / pgsteal_proactive | 0 / 0 |
| PSI memory some.total | 400543microsecond |
| PSI memory full.total | 396950microsecond |

Swap toàn máy dùng cao nhất297616KiB, trong khi VmSwap của child vẫn0. Có swap-out/reclaim ở hệ thống cùng lúc suy luận, nhưng không tái hiện sự cố processswap>128MiB của04/11. Không suy toàn bộ reclaim là do tác vụ nền: I/O của model cũng có thể tham gia điều kiện reclaim, chưa có đối chứng để tách tác nhân. Bộ đếm page không phải lưu lượngSSD vật lý, không phải dung lượng trang duy nhất, và các bộ đếm phân loại có thể chồng lấn; không cộng chúng thành một tổng.

vmstat và PSI đều statusok trong927mẫu. Cgroup-v2 statuspartial vì root mount thiếu một số file; các mức0–4 có telemetry đọc được, identity device/inode giữ nguyên từng mức. Ở các mức có file giới hạn, memory.max/high/swap.max đều `max`; events/events.local không tăng high/max/oom/oom_kill. Mứcroot thiếu limit/events không được nhận là0 hay không có ràng buộc ẩn.

Cgroup chứa PID child có thể chứa nhiều process; memory.current/stat là số của nhóm, không phải riêng engine. Delta pgscan_kswapd ở mức0 là15603185, pgscan_direct2025; các ancestor cũng có reclaim. Không cộng ancestor với child, không gán số này cho riêng model. Không có bằng chứng event ngưỡng cgroup nhìn thấy kích hoạt trong lượt này; không dùng kết quả mới để loại trừ nguyên nhân cgroup/OS của sự cố cũ.

Lookahead tắt, prediction_calls/speculative load0; tổng eligible5/11340lượt lớp có lớp kế tiếp, disabled_eligible5. Ghi nhận như điều kiện chạy, không mở lại đối chứng tốc độ từ các cơ hội hiếm.

## Kết luận và bước tiếp

Phép kiểm hữu hạn lần này hoàn tất trong các ngưỡng đã đặt. Phần RSS tăng qua request tiếp tục được quan sát dù không có process swap, nên có thể điều tra allocation/lifetime như một câu hỏi riêng. Telemetry mới không xác định được tác nhân sự cố swap04/11 vì sự cố không tái hiện; không quy việc hoàn tất hôm nay cho bản vá telemetry.

Bước tiếp đề xuất: chọn instrumentation hẹp cho live capacity và mốc lookup KV/prefill/session admission/ra khỏi generate_inner để phân biệt vùng còn sống với allocator giữ trang. Chưa có allocationtrace hoặc bằng chứng leak; chưa sửa engine/cachepolicy/OS và chưa chạy thêm model sau12request. Giữ nguyên dữ liệu cũ và ngưỡng bảo vệ; giai đoạn1 về ổn định tổngRAM chưa đóng.

## Bảo toàn payload

Chuyển nguyên17file từ runs vào `run/`, giữ một bản duy nhất; thêm preflight riêng, không chèn hồi tố vào sampler. Khóa18file bằng checksum, README không thuộc payload khóa. Đường dẫn gốc trong record/log giữ như khi đo.

```bash
sha256sum -c CHECKSUMS.sha256
```
