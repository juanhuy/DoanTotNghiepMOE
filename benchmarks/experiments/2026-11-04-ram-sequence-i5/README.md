# Kiểm xu hướng tổng RAM i5, lookahead tắt, 04/11/2026

## Câu hỏi và phạm vi

Ở cùng ngân sách CPU/cache hiện hành, RSS qua chuỗi giữ cache còn tăng ra sao? Protocol chốt trước tại [SCENARIOS 3.3](../../SCENARIOS.md#33-kiểm-xu-hướng-tổng-ram-hữu-hạn-trên-i5-04112026): tối đa12request, ba nhóm bốn câu; không phải benchmark tốc độ hoặc đối chứng lookahead. Sau request đầu reset, giữ cache đến hết lượt. Trong mỗi nhóm: `Where is Mars`, `What planet did I just ask about?` cùng phiên/lịch sử, `Explain a Python dictionary.` phiên mới, `What is love ?` phiên mới. `--cycles 12` đếm12request, không phải12nhóm.

Source/binary khớp hash lượt chẩn đoán hoàn tất trước đó; không build/tests/inference khác đồng thời. Trước tải, ba snapshot cách5s trong [preflight](preflight.json) ghi MemAvailable10,127–10,197GiB, swap dùng0, pswpin/pswpout không tăng, PSI memory avg10=0. Đây là cửa sổ hữu hạn trước tải. Preview PASS, rồi runner được cấp quyền mở localhost ngoài sandbox để thực thi đúng lệnh:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --cache 64 --context 512 --cycles 12 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty
```

Nguồn dirty, HEAD75d5a62 chỉ là HEAD, không phải toàn bộ source đang chạy. [Record](run/result.json) giữ42sourcehash/ZIP, binary/checkpoint hash, cấu hình và hardware quan sát: i5-10500H, RAM16.571.326.464B, GPU có trong inventory nhưng CPU mode không dùng. StatusCPU/0VRAM và probe thư viện CUDA/cuBLAS rỗng. Không coi đây là baseline revision sạch hoặc kiểm CUDA compute.

## Kết quả: watchdog dừng sau10request hoàn tất

Run `20261104T144809.160002Z-ram-e6fa1d0203994a7a8b0174e4abad455f` có status `watchdog_stopped`. Request thứ11 bị ngắt khi swap tiến trình vượt128MiB; không có request thứ12. Giữ lỗi SSE thiếu finish/DONE như hậu quả child bị dừng, không đổi status thành completed.

Trong10request hoàn tất, mỗi request sinh64token/finish length. Prompt tokens theo pha lần lượt19/107/22/20; ba request hội thoại dùng lại11prompt tokens. RSS sau từng request dưới đây lấy từ `turn-N.json`, đơn vịMiB; VmSwap ở tất cả các snapshot sau10request này đều0.

| Request (đếm từ1) | Nhóm/pha | RSS sau request (MiB) |
| --- | --- | ---: |
| 1 | 1/Mars | 2885,289 |
| 2 | 1/hội thoại | 2917,527 |
| 3 | 1/Python | 2979,977 |
| 4 | 1/Love | 3021,922 |
| 5 | 2/Mars | 3066,645 |
| 6 | 2/hội thoại | 3075,277 |
| 7 | 2/Python | 3104,742 |
| 8 | 2/Love | 3140,754 |
| 9 | 3/Mars | 3172,031 |
| 10 | 3/hội thoại | 3136,160 |

Cuối nhóm1→2 tăng118,832MiB; nhóm3 không hoàn tất nên không có mốc cuối nhóm3 để so. Từ request1→9 tăng286,742MiB; request10 giảm trước thời điểm phát hiện process swap, không nhận đây là plateau. Cả10snapshot giữ cache962,5MiB/10expert mỗi lớp, recycled0, session capacity128,001–128,003MiB trong budget256MiB. KV logical theo pha20,5/42,5/21,25/20,75MiB. Capacity/logical bytes không bằng RSS; chưa giải thích được phần RSS tăng bằng các thành phần đã báo.

[Time series](run/memory.jsonl) có830mẫu, mốc cuối207,649s; sampledRSS/HWMmax3,097691GiB, VmSwapmax140632KiB=137,336MiB. MemAvailable thấp nhất7,077408GiB. Watchdog duy nhất là process swap, không phải RSS hoặc system headroom; không có bằng chứng kernel OOM từ record này.

## Đoạn chuyển trang ra swap

| Mốc (s) | RSS (MiB) | VmSwap (MiB) | RSS + VmSwap (MiB) | Swap toàn máy (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 205,895 | 3154,035 | 0 | 3154,035 | 356,359 |
| 206,898 | 3093,738 | 60,297 | 3154,035 | 439,141 |
| 207,649 | 3016,699 | 137,336 | 3154,035 | 520,277 |

Trong1,754s này, RSS giảm140632KiB và VmSwap tăng đúng140632KiB; tổng hai trường không đổi. Swap toàn máy tăng167852KiB. Quan sát phù hợp với việc chuyển các trang đang có ra swap trong khoảng đó, không cho thấy engine vừa cấp thêm một khối lớn tại thời điểm dừng. Không dùng tổng RSS+VmSwap làm accounting đầy đủ mọi allocation hoặc suy nguyên nhân tăng RSS ở các request trước.

[Postflight](postflight.json) là đọc-only sau khi child đã dừng: swappiness60; các ancestor quan sát được của process đọc có memory.max/high=max và không ghi oom/oom_kill. Không lưu đường dẫn cá nhân cgroup; ancestor_level chỉ đếm từ process đọc lên. Đây không phải telemetry cgroup của child tại thời điểm sự cố, không xác định tác nhân reclaim/swap hoặc loại trừ mọi giới hạn khác. Không đổi swappiness, swapoff, drop_caches hay cấu hình OS.

## Ý nghĩa và bước tiếp

Phép kiểm12request chưa đạt điều kiện hoàn tất. Cache budget cố định và CPU trước load chưa đủ chứng minh tổng RAM ổn định. Hai vấn đề còn tách biệt: RSS tăng trước sự cố khi reported capacity gần cố định; và đợt swap cuối khi tổng RSS+VmSwap không đổi. Chưa chứng minh leak hoặc đổ nguyên nhân cho allocator, KV hay cache policy.

Rà tĩnh `take_prefix_kv`/session admission cho thấy KV phiên mới được cấp theo context và session cũ có thể còn sống tới khi xét admission sau generate; status cuối request chỉ đếm session được giữ. Đây là lý do accounting cuối request chưa bao phủ peak tạm, chưa chứng minh đó là nguồn tăng RSS của lượt này. Cần telemetry cấp phát/lifetime phù hợp nếu mở nhánh đó.

Gate đủ điều kiện4/9450lượt lớp có lớp kế tiếp ở10request (request5:1, request8:1, request9:2); modeoff ghi disabled_eligible=4, prediction_calls/speculative load vẫn0. Điều này bổ sung giới hạn cho mẫu chẩn đoán trước: không suy cơ chế luôn bị chặn ở mọi request khi cache được giữ lâu hơn. Chưa mở lại đối chứng tốc độ từ bốn cơ hội hiếm này.

Bước tiếp đề xuất là bổ sung telemetry reclaim/swap cấp hệ thống và cgroup của chính child vào runner (pswpin/pswpout, PSI, memory events/stat và giới hạn hiện hành), kiểm bằng fixture/đọc-only trước tải. Không chạy lại model ngay hoặc nâng ngưỡng để đi qua lỗi. Việc định vị phần tăng allocation/RSS cần giữ riêng với việc tìm tác nhân reclaim.

## Bảo toàn bằng chứng

Chuyển nguyên15file từ runs vào `run/`, một payload duy nhất; đường dẫn gốc trong record/log giữ nguyên. Hai snapshot trước/sau nằm riêng, không chèn hồi tố vào sampler. Khóa17file bằng CHECKSUMS, README không thuộc payload khóa. Chuyển snapshot từ/tmp ban đầu gặp khác filesystem khi rename, đã chuyển bằng copy/unlink có kiểm hash, không đổi nội dung.

```bash
sha256sum -c CHECKSUMS.sha256
```
