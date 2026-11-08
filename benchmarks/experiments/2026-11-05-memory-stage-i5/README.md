# Trace bộ nhớ qua4request OLMoE trên i5, 05/11/2026

## Câu hỏi và điều kiện

Đúng một nhóm4request theo [SCENARIOS3.5](../../SCENARIOS.md#35-trace-capacity-theo-mốc-request-05112026), chốt trước tải: xem RSS/capacity tại lookupKV, prefill, admission và sau inner. Không đo tốc độ hoặc tìm plateau bằng cách lặp tiếp. CPU trước load/4threads/victim/cache64MiB-lớp/context512/max64/keepcache/lookahead off, memory trace on. GuardRSS5GiB/headroom1GiB, watchdogRSS5,5GiB/MemAvailable1,5GiB/processswap128MiB giữ nguyên.

Source44hash và binary khớp fixture trace đã kiểm; không build/test lại hoặc chạy inference khác đồng thời. [Preflight](preflight.json) ba mốc5s có MemAvailable khoảng9,5–9,6GiB, swap cũ khoảng735MiB, pswpout không tăng và PSIavg10=0. Swap cũ không được nhận là swap-out mới hay bằng chứng máy an toàn dài hạn. Preview PASS rồi được cấp localhost và chạy:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --memory-trace on --cache 64 --context 512 --cycles 4 --max-tokens 64 \
  --keep-cache --threads 4 --ram-limit-mib 5120 --headroom-mib 1024 \
  --stop-rss-mib 5632 --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty
```

Chuỗi: `Where is Mars`; `What planet did I just ask about?` cùng phiên/lịch sử chứa output thật; `Explain a Python dictionary.` phiên mới; `What is love ?` phiên mới. Reset chỉ trước request đầu. Hash checkpoint làm ấm OS cache; engine reset không phải SSDcold.

[Record](run/result.json) runID `20261105T033035.713627Z-ram-a7a8546e953a42248e392df56bf2bb4c` có sourceZIP/hash/checkpoint/binary/hardware, HEAD75d5a62/dirty exploratory. Không coi HEAD là source sạch hoặc ghép số thời gian với lượt trước để nhận speedup. CPUmode0VRAM/probe khônglibCUDA; không kiểm CUDAcompute.

## Bốn request hoàn tất; đủ28sample stage

Statuscompleted, không watchdogviolation. Mỗi request sinh64completion/finishlength; prompt19/107/22/20, KVreuse0/11/0/0. Mỗi trace schema1/outcomecompleted/7stage, trùng last_memory_trace trong status sau request; tổng tracked và chuyển ownership hợp lệ. Đơn vị RSS theo trace dưới đây làMiB, khác field byte gốc:

| Stage | Mars | Hội thoại | Python | Love |
| --- | ---: | ---: | ---: | ---: |
| before_request | 1851,293 | 2898,164 | 2926,074 | 2988,781 |
| after_kv_lookup | 1856,063 | 2898,176 | 2928,152 | 2988,785 |
| prefill_buffers_live | 1856,113 | 2898,176 | 2928,152 | 2988,785 |
| after_prefill | 2828,508 | 2914,676 | 2979,422 | 3022,949 |
| before_session_admission | 2898,117 | 2926,020 | 2988,773 | 3030,035 |
| after_session_admission | 2898,117 | 2926,020 | 2988,773 | 3030,035 |
| after_generate_inner | 2898,117 | 2926,020 | 2988,773 | 3030,035 |

Từ mốc cuối request1→4 tăng131,918MiB. Request đầu làm ấm cache expert0→962,5MiB trong prefill; không coi phần tăng này là leak. Ba request sau giữ cache962,5MiB/recycled0. Scratch capacity khoảng0,377MiB; vector prefill chính theo request2,404/12,142/2,784/2,531MiB và không còn tính sau prefill. Capacity bằng0 sau scope không chứng minh allocator trả trang hoặc RSS giảm.

[Sampler](run/memory.jsonl) có328mẫu, mốc cuối82,919s: sampledRSS/HWMmax2,959019GiB, processswap0 trong mọi mẫu, MemAvailablemin6,481834GiB, không watchdogviolation. Snapshot RSS sau HTTP request có thể khác stage cuối vàiKiB vì thời điểm khác; bảng trên chỉ dùng cùng trace. Bộ đếm toàn hệ thống pswpout tăng563page=2,199MiB, pswpin74page=0,289MiB với page4096B; PSI some/full total tăng709619/704461microsecond. Cgroup telemetry partial do file root thiếu, không thay bằng0. Không có processswap incident trong lượt này; không dùng kết quả để giải thích hồi tố toàn bộ sự cố04/11.

## Quan sát giúp chọn một ứng viên tối ưu

1. KV active khoảng128,001MiB gồm backing Vec; logicalKV cuối request chỉ20,5/42,5/21,25/20,75MiB. Ở request đầu, RSS tăng4,770MiB từ before_request tới after_kv_lookup trong khi capacity KV tăng128MiB. Capacity chưa phải số trang RAM vật lý đã dùng; khoảng này còn gồm tokenization/encoding/tokenizerclone, không quy riêng deltaRSS cho KV.
2. Hội thoại dùng lại KV: sau lookup, retained session trở về0 và activeKV128MiB, không có hai bộKV được đếm cùng lúc. RSS chỉ tăng0,012MiB tại khoảng lookup; phần tăng tiếp nằm trong prefill và đoạn sauprefill→trướcadmission.
3. Hai request phiên mới có retainedKV cũ và activeKV mới cùng tồn tại, mỗi bộ khoảng128MiB. Source `take_prefix_kv` cấp KV mới khi không lookup được session; admission/eviction ở cuối generate, nên các session khác vẫn giữ KV trong thời gian request chạy. Đây là overlap allocation/capacity đã xác nhận, không phải suy từ RSS.
4. Với Python, tracked_host_bytes trước admission khoảng3037,388MiB, sau admission2909,386MiB; RSS giữ2988,773MiB. Love có cùng dạng overlap/drop. Không suy phần capacity bỏ đi vẫn còn live trong engine chỉ vì RSS không giảm; cũng chưa xác định allocator/region nào gây mọi phần tăngRSS.

Ở ba request đã ấm, RSS tăng từ prefill_buffers_live→after_prefill lần lượt16,500/51,270/34,164MiB; từ after_prefill→before_session_admission tăng11,344/9,352/7,086MiB. Khoảng sau còn gồm decode và hoàn tất output, không phải timer decode thuần. RSS tăng lớn hơn ở vùng tính toán/ghi dữ liệu so với tại lookup, nhưng KV có thể chỉ chạm trang khi ghi trong các vùng đó; còn expert I/O và allocation sâu chưa được trace đầy đủ. Không quy nguyên nhân cho một buffer chỉ từ các boundary.

## Bước tiếp đề xuất, chưa triển khai trong lượt này

Thử tái sử dụng buffer KV của session được chọn loại theo ngân sách thay vì cấp bộ mới rồi mới evict. Mục tiêu trực tiếp là giảm overlap/cấp phát KV mới khi đổi phiên; chưa hứa giảmRSS hoặc chứng minh KV là nguyên nhân duy nhất. Không tạo pool ngoài ngân sách hoặc phá prefix của session còn được giữ. Nếu phương án cần evict sớm thì phải công bố tác động khi request hủy/lỗi: phiên cũ có thể mất khả năng reuse và cần recompute, tuyệt đối không để KV đã ghi đè vẫn mang sessionID cũ.

Trước đo mới: hồi quy MHA/GQA cho reuse/prefix/LRU/budget, đổi phiên, hủy/lỗi/retry và accounting không đếm trùng; giữ tolerance hiện có. Sau đó chỉ đối chứng hẹp cùng cấu hình/checkpoint/source có hai chế độ nếu cần kết luận lợi ích. Không sửa cacheexpert policy, lookahead hoặc OS từ giả thuyết này. Không chạy thêm request sau bốn mẫu; tổngRAM/giaiđoạn1 chưa đóng.

## Payload

Chuyển nguyên9file từ runs vào `run/`, thêm preflight riêng, một payload duy nhất; khóa10file bằng CHECKSUMS. README không thuộc payload khóa; đường dẫn gốc trong record/log giữ như khi đo. Không sửa raw/checksum các hồ sơ trước.

```bash
sha256sum -c CHECKSUMS.sha256
```
