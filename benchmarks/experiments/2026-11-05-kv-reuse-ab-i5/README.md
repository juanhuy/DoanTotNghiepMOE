# Đối chứng KV buffer reuse i5 bị dừng ở blockoff, 05/11/2026

## Kế hoạch và điều kiện

Theo [SCENARIOS3.6](../../SCENARIOS.md#36-candidate-tái-sử-dụng-buffer-kv-05112026), chốt trước chạy đúng một cặp blockoff→on, mỗi block process mới/4request, tổng tối đa8request. Cùng binary/source/checkpoint/config, chỉ đổi --kv-buffer-reuse; dừng cả cặp khi bất kỳ block lỗi/watchdog. Không tự lặp tìm kết quả tốt và không lấy raw cũ trước code làm cùng-source baseline.

CPU trước load/4threads/victim/cache64MiB-lớp/context512/max64/lookahead off/memorytrace on/keepcache. Chuỗi Mars, hội thoại giữ lịch sử thực, Python phiên mới, Love phiên mới; reset chỉ đầu block. GuardRSS5GiB/headroom1GiB, watchdogRSS5,5GiB/MemAvailable1,5GiB/processswap128MiB giữ nguyên. Không warmup riêng, không đổi OS hoặc giải phóng swap cưỡng bức.

Source44hash/binary khớp fixture đã kiểm95Rust/profile/60Python/7Node ở bước triển khai, không build/tests/inference khác đồng thời. [Preflight](preflight.json) ba mốc cách5s ghi RAM khả dụng khoảng9,5GiB, swap cũ khoảng902MiB nhưng pswpout không tăng và PSIavg10=0. Điều kiện rảnh trước tải không bảo đảm lúc nạp/suy luận. Hai lệnh previewoff/on đạt; blockoff được cấp localhost và thực thi:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --memory-trace on --kv-buffer-reuse off --cache 64 --context 512 \
  --cycles 4 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty
```

[Recordoff](off/result.json) runID `20261105T043908.397887Z-ram-c47c685f62734e958b9fe38ed48164af`, HEAD75d5a62/dirty, sourceZIP/hash/checkpoint/binary/hardware giữ đầy đủ. Đây là exploratory, không baseline sạch. CPUmode0VRAM/probe khônglibCUDA. Hash checkpoint trước tải làm ấm OS cache; không nhận engine-cold là OS/SSDcold.

## Kết quả: cặp chưa hoàn tất

Blockoff có status `watchdog_stopped`:3/4request hoàn tất, request4 bị ngắt ở64,170s khi processVmSwap137640KiB=134,414MiB vượt128MiB. Không chạy blockon theo điều kiện dừng đã chốt; không có rawon hoặc samplepair để so sánh.

| Requestoff hoàn tất | Prompt/completion tokens | RSS sau request (MiB) | VmSwap sau request (MiB) |
| --- | --- | ---: | ---: |
| Mars | 19/64 | 2891,082 | 6,977 |
| Hội thoại | 107/64 | 2919,020 | 6,977 |
| Python | 22/64 | 2981,082 | 6,977 |

Ba request finishlength, trace schema1/7stage và gate totals hợp lệ. kv_buffer_reuse.enabled=false, earlyeviction/reusedbytes0, source null; candidate không hoạt động trong block này. Hội thoại dùng lại11prompt tokens theo cơ chế prefix cũ; Python vẫn có retainedKV128MiB+activeKV128MiB tại lookup như đườngoff dự kiến. Không dựng Generation/trace cho request4 chưa hoàn tất. Lỗi SSE thiếu finish/DONE giữ như hậu quả child bịwatchdogdừng, không sửa status thànhcompleted hoặc nhận là hồi quy giao thức mới.

[Sampler](off/memory.jsonl) có254mẫu, mốc cuối64,170s: sampledRSS/HWMmax2,949974GiB, processswapmax134,414MiB, MemAvailablemin7,024029GiB. Watchdog duy nhất là processswap, không phảiRSS/headroom. Không dùng RSS thấp hơn một lượt khác làm bằng chứng ít giữ bộ nhớ vì có chuyển trang ra swap.

## Quan sát reclaim trong lượt dừng

| Mốc (s) | RSS (MiB) | VmSwap (MiB) | RSS+VmSwap (MiB) |
| --- | ---: | ---: | ---: |
| 62,386 | 3020,758 | 6,977 | 3027,734 |
| 63,405 | 2985,816 | 41,949 | 3027,766 |
| 64,170 | 2893,379 | 134,414 | 3027,793 |

Trong1,784s cuối, RSS giảm130436KiB, VmSwap tăng130496KiB, tổng hai trường chỉ tăng60KiB. Phù hợp chuyển trang đang có ra swap ở khoảng này, không chứng minh engine vừa cấp thêm lượng lớn tại mốc dừng hoặc giải thích mọi tăngRSS trước đó. Tổng RSS+VmSwap không phải accounting đầy đủ mọi allocation/shared page.

Trong toàn sampler, pswpout hệ thống tăng73578page, pswpin1937page (page4096B); pgscan_kswapd tăng3787571/direct173532, pgsteal_kswapd3428202/direct55886; proactive scan/steal0. PSI some/full total tăng486068/485213microsecond. Swap toàn máy dùng923800→1227584KiB. Đây là số hệ thống, không lưu lượngSSD hoặc dung lượng trang duy nhất; không cộng các counter phân loại chồng lấn.

Cgroup identities giữ nguyên; ở mức0–4 có file giới hạn đều max, memory.events không tăng high/max/oom/oom_kill. Childgroup pgscan_kswapd tăng3599733/direct105123; các ancestor cũng có reclaim. Root thiếu limit/events được ghiunavailable. Không cộng ancestor vào child hoặc coi số nhóm là của riêng model: cgroup có thể gồm process khác. Không thấy event áp ngưỡng cgroup nhìn thấy trong lượt này; chưa xác định tác nhân reclaim, không loại trừ ràng buộc ẩn hoặc quy mọi tác động cho tải nền thay vì I/O model.

## Kết luận và trạng thái tiếp

Chưa có đối chứng off/on, nên chưa kết luận candidate làm tốt/xấu hơn, giảmRSS hoặc tăng tốc. Sự cố xảy ra ở modeoff không phải bằng chứng đánh giá hiệu quả candidateon. Giữ mặc địnhoff, không chạy tiếp hoặc nâng ngưỡng để hoàn thành cặp bằng mọi giá.

Chỉ cân nhắc phép thử mới khi được giao và có điều kiện bộ nhớ phù hợp; cần chốt lại phạm vi/giới hạn, không ghép lượt khác ngày hoặc source thành một cặp giả. KernelOOM hoặc leak chưa được xác định từ hồ sơ này; guard là hợp tác, chưa hardRSScap hoặc nghiệm thu tổngRAM.

## Payload

Chuyển nguyên8file từ runs vào `off/`, thêm preflight riêng, một payload duy nhất; khóa9file bằngCHECKSUMS. Không tạo thư mục/số đoon để biểu diễn một lượt không chạy. README không thuộc payload khóa, đường dẫn gốc trong record/log giữ nguyên. Raw/checksum các thí nghiệm trước không đổi.

```bash
sha256sum -c CHECKSUMS.sha256
```
