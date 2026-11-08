# Chẩn đoán allocator và RSS trên i5 — 08/11/2026

## Câu hỏi và protocol

Khi RSS tăng qua nhiều request, bộ nhớ allocator ghi nhận đang dùng có tăng tương ứng không? Chạy đúng một process mới với 12 request mixed, theo protocol đã ghi trước ở [SCENARIOS](../../SCENARIOS.md). Đây là phép chẩn đoán có instrumentation, không dùng để so tốc độ với phiên bản trước.

Cấu hình: CPU trước load, 4 threads, victim, cache64MiB/layer, context512, max64, keepcache/chỉ reset đầu, lookahead off, KVbufferreuse off, memorytrace on. Workload: Mars, follow-up cùng session, Python mới, Love mới, lặp3nhóm.

Lệnh đã preview rồi thực thi:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off   --memory-trace on --kv-buffer-reuse off --workload mixed --cache 64   --context 512 --cycles 12 --max-tokens 64 --keep-cache --threads 4   --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632   --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty
```

[Preflight](preflight.json) có3mẫu cách5s: available khoảng10,7–10,9GiB, swap cũ3616KiB, pswpout912 không tăng, PSIavg10=0. Nguồn44hash/binary khớp fixture đã kiểm96Rust/profile,61Python,7Node. Không build/tests/inference khác đồng thời; được cấp localhost rồi chạy một lượt, không đổi OS/ngưỡng hoặc lặp thêm.

## Nguồn và tình trạng hoàn tất

[Record](run/result.json): run20261108T094427.046258Z-ram-f8bfca077a8a4291827053dccdef49ae; HEAD75d5a62/dirty exploratory. BinarySHA2564b3fa13ffd8526e0234d50caaedb6e756a8eb112ef0bf97019d805cb563bb8c3; sourceZIP44hash/binary khớp fixture và nguồn sau đo. Checkpointfiles/hash và config giống lượt12request trước, nhưng source/binary đã đổi.

Hardware: i5-10500H,12logicalCPU,RAM16571326464B; GPU có trong inventory nhưng CPUmode0VRAM/probe available khônglibCUDA. Chưa kiểm GPUcompute.

**COMPLETED12/12**, không watchdogviolation/guardOOM. Mỗi request64completion/finishlength, prompt19/107/22/20 và cachedprompt0/11/0/0 mỗi nhóm. Đây là smoke/giao thức, không kiểm chất lượng chuẩn hóa hoặc parity với model tham chiếu.

## RSS và allocator ở cùng mốc

Bảng dùng **after_generate_inner**, đơn vịMiB. “In-use+mmap” là uordblks+hblkhd, chỉ báo bookkeeping chứ không exactliveRustpayload. hblkhd cố định1810,258MiB ở các endpoint. Không cộng các cột với RSS.

| Request | RSS | Arena | In-use+mmap | Free trong arena |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2898.047 | 1170.883 | 2919.814 | 61.326 |
| 2 | 2925.918 | 1182.883 | 2919.894 | 73.246 |
| 3 | 2988.668 | 1310.891 | 2919.928 | 201.220 |
| 4 | 3029.953 | 1335.020 | 2919.974 | 225.304 |
| 5 | 3067.449 | 1340.895 | 2919.984 | 231.168 |
| 6 | 3064.047 | 1328.891 | 2920.001 | 219.148 |
| 7 | 3093.492 | 1352.895 | 2920.005 | 243.148 |
| 8 | 3123.508 | 1358.895 | 2920.017 | 249.135 |
| 9 | 3158.859 | 1370.895 | 2920.021 | 261.132 |
| 10 | 3170.180 | 1370.895 | 2920.030 | 261.122 |
| 11 | 3170.215 | 1370.895 | 2920.033 | 261.119 |
| 12 | 3182.594 | 1376.895 | 2920.049 | 267.104 |

Request1→12:
- RSS **+284,547MiB**.
- In-use+mmap **+0,234MiB**; range của indicator ở request2–12 chỉ0,154MiB.
- Arena **+206,012MiB**, free **+205,777MiB**.

Tại request3, trước→sau admission: in_use **giảm128,002MiB**, free **tăng128,002MiB**, arena/mmap không đổi và RSS không giảm. Allocator đã ghi nhận vùng được free, nhưng resident memory chưa giảm tại mốc đó.

84mốc đều có statusglibc_mallinfo2/version2.44, counters hợp lệ và arena=in_use+free. LD_PRELOAD không hiện diện theo envhint; điều này không chứng nhận mọi interposition. Tất cả boundary dùng callerTID23272: không thấy worker gọi trace đổi giữa request trong lượt này; không loại trừ nhiều arena/workerRayon.

Cache+recycled từng layer và session sau admission đúng budget; trace7stage/request trùng status, tổng tracked không đếm retainedKV hai lần. Cache cuối962,5MiB, session khoảng128MiB gần cố định. Lookaheadpredictioncalls0 và gate totals hợp lệ.

## Biên vận hành và chi phí phép đo

684samplerows, mốc cuối173,240s. SampledRSSpeak3182,594MiB; HWM cao nhất từ sampler+turn3186,004MiB. ProcessVmSwapmax **0,117MiB**, minMemAvailable7,752GiB. Không ghi swap0; HWM có thể bắt đỉnh nằm giữa các mẫu RSS.

Globalpswpin+405/pswpout+104627page (4096B/page, swapout khoảng408,699MiB), PSI some.total+485721microsecond. Đây là counter toàn hệ thống, không quy hết cho model hoặc nền và không coi là physicalSSDbytes. Cgroupidentity giữ nguyên; events/events.local đọc được không tăng high/max/oom/oom_kill. 7file thiếu ở rootlevel5 được giữ unavailable; không cộng ancestor với child.

query_ns84mốc: min2067/median11507/max628544ns, tổng5406600ns≈5,407ms. Chỉ đo lời gọi mallinfo2, chưa gồm init/versionclone/TID/proc/serialization/toàn overhead. Không dùng thời gian khác giữa hai lượt làm bằng chứng tăng tốc.

## Kết luận và điểm dừng

**Bằng chứng ủng hộ mạnh việc RSS tăng lớn mà không có tăng tương ứng allocator-accounted in-use ở cuối request.** Indicator gần cố định trong khi RSS tăng>150MiB; allocatorfree tăng, admission chuyển128MiB từ in-use sang free mà RSS không giảm. Dấu hiệu này đáp ứng nhánh retainedmemory đã đề xuất trước chạy.

Không có căn cứ gọi284MiB là leak đối tượng đang dùng. Tuy nhiên, vẫn chưa loại trừ leak nhỏ, allocation tăng/giảm bù nhau, tcache/metadata hoặc mapping ngoài malloc. Chưa định danh từng mapping/trang hoặc quy toàn bộ mức tăng cho KV/expert; VmData−RssAnon không phải số đo chính xác trang chưa chạm. Snapshot các arena là tuần tự và RSS được đọc khác thời điểm.

Lượt mới củng cố lời giải thích allocator giữ vùng đã free và các trang trở thành resident cho hiện tượng271MiB cũ; không điền allocatorstats hồi tố vào raw cũ. Chưa chứng minh plateau, ổn định dài hạn, hardcap tổngRAM hay cấu hình chung i7. Giai đoạn1 chưa đóng.

**Khép nhánh tìm “leak lớn tương ứng271MiB” ở mức chưa thấy tăng in-use tương ứng trong cửa sổ này.** Không tự cài heaptrack hoặc lặp model để đuổi mọi byte. Bước sau là quyết định có cần giảm footprint không, rồi quay về câu hỏi expertcache/I/O. Nếu thử KVreuse hoặc điều chỉnh vòng đời payload, phải là phép đối chứng riêng, thay một yếu tố, cùng budget và kiểm parity; không tự bật mặc định/tăng cache hoặc dùng trim làm bản vá.

## Bảo toàn payload

Chuyển nguyên17file từ runs vào run/, thêm preflight, khóa18file bằngCHECKSUMS.sha256. Một payload duy nhất; README là diễn giải ngoài payload khóa. Đường dẫn gốc trong record/log và mọi raw/checksum cũ giữ nguyên.

```bash
sha256sum -c CHECKSUMS.sha256
```
