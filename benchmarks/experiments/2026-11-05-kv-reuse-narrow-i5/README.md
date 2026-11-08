# Đối chứng hẹp KV buffer reuse trên model thật i5, 05/11/2026

## Câu hỏi và protocol chốt trước chạy

Khi đổi sang một phiên mới phải loại KV cũ theo ngân sách, candidate có tránh cấp thêm một bộKV, giữ đầu ra và giảm bộ nhớ quan sát không? Theo [SCENARIOS3.6](../../SCENARIOS.md#36-candidate-tái-sử-dụng-buffer-kv-05112026): đúng một cặp **on→off**, process riêng, mỗi block2request độc lập `Where is Mars` rồi `Explain a Python dictionary.`. Chỉ reset đầu/keepcache, không history, không warmup tách. Tổng4request, không mở thêm cặp sau kết quả. Khác workloadmixed4request trước đó, không ghép raw cũ làm đối chứng.

RAMrunner bổ sung `--workload kv-switch` v1, giữ mixed mặc định như trước. Workloadspec/version/prompts lưu trong từng record/preview. Hồi quy xác nhận mỗi câu mở session mới/reset đúng, mixed giữ nguyên phase/history/reset. `python3 tools/check.py --web` PASS61Python/7Node/catalog40; HTTPfixture2request tại `/tmp/moe-kv-switch-fixture/i5/20261105T065834.425831Z-ram-6d491265bc9a41e491afb17981be495e` hoàn tất và xác nhận ID khác/messages chỉ1user. Engine/binary không đổi so bản candidate đã kiểm95Rust/profile; tooling2file đổi trước cả hai block, không build/tests/inference khác đồng thời.

Hai block cùng binary/source44hash/checkpoint/config, chỉ khác --kv-buffer-reuse. CPU trước load/4threads/victim/cache64MiB-lớp/context512/max64/lookahead off/memorytrace on; guardRSS5GiB/headroom1GiB, watchdogRSS5,5GiB/MemAvailable1,5GiB/processswap128MiB giữ nguyên. Preview trước, sau đó được cấp localhost chạy on trước:

```bash
python3 tools/check_ram_stability.py --testbed i5 --gpu cpu --lookahead off \
  --memory-trace on --kv-buffer-reuse on --workload kv-switch \
  --cache 64 --context 512 --cycles 2 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --allow-dirty
```

Off chỉ đổi --kv-buffer-reuse off. Nếu block lỗi/watchdog hoặc đầu ra/config không phù hợp thì dừng cả cặp, không tăng ngưỡng/lặp tìm kết quả đẹp. [Preflight ban đầu](preflight.json), [trước on](preflight-before-on.json), [trước off](preflight-before-off.json) giữ các cửa sổ quan sát riêng. Trướcon MemAvailable khoảng7,9–8,1GiB, pswpout không tăng; trướcoff khoảng8,6–8,7GiB, pswpout không tăng10s nhưng PSIavg10 còn0,56→0,20. Điều kiện hệ thống khác nhau, preflight không bảo đảm an toàn lúc tải.

## Provenance và tính tương đương

- [On](on/result.json): `20261105T065959.094659Z-ram-1d301c7f3d93492788f37bf8bc0dcc94`, chạy trước.
- [Off](off/result.json): `20261105T070240.707616Z-ram-aeace2bcae22467b869ebd90389d80aa`, chạy sau.
- Cả hai HEAD75d5a62/dirty exploratory, không baseline revision sạch. SourceZIP/44hash giống nhau và khớp source sau đo; checkpoint files/hash giống nhau; binary SHA256 `c45617f4f626f94af219c1d40ea86d6774735d09d3739bd9e3aa1e97aff4c313` giống nhau và không đổi sau đo. Mọi config ngoài kv_buffer_reuse bằng nhau.
- Hardware quan sát cùng i5-10500H/12logicalCPU/RAM16.571.322.368B; GPU có trong inventory nhưng CPUmode0VRAM/probe khônglibCUDA, không kiểm GPUcompute.
- Cả hai completed2/2, không watchdogviolation. Mỗi request64completion/finishlength; prompt19/22, cachedprompt0 cả hai. Text/messages/usage/finish khớp2/2cặp đầu vào. HTTP không có tokenIDs/logits: đây không phải paritylogits toànmodel7B; fixtureMHA/GQA trước đó kiểm tolerance2e-5.
- Expert workload counters giống nhau từng input: Mars4412hit/6084miss/5924eviction/6084fileloads/38376898560payloadbytes; Python5140hit/5740miss/5740eviction/5740fileloads/36207001600payloadbytes. Payload không phải physicalSSDbytes.

## Từng mẫu bộ nhớ và cơ chế reuse

Đơn vịMiB trừ dòng ghiGiB. RSS lấy mẫu từ memory.jsonl; capacity theo các stage. Không cộng retainedKV hai lần vào retained_session.

| Đại lượng | Off | On |
| --- | ---: | ---: |
| RSS sau request Mars | 2882,469 | 2887,289 |
| RSS sau request Python | 2945,641 | 2893,301 |
| Peak RSS lấy mẫu | 2945,574 | 2893,664 |
| VmHWM cao nhất đọc được (sampler và turn snapshot) | 2945,641 | 2893,664 |
| Max(RSS+VmSwap) cùng mẫu | 2945,574 | 2896,945 |
| VmSwap tiến trình cao nhất lấy mẫu | 0 | 43,121 |
| KV active+retained cao nhất tại stage | 256,001 | 128,001 |
| Tracked host bytes cao nhất tại stage | 3040,171 | 2912,169 |
| MemAvailable hệ thống thấp nhất (GiB) | 5,795 | 5,128 |
| Sampler rows | 176 | 237 |
| Mốc sampler cuối (s) | 44,503 | 59,984 |

On requestPython có reused_capacity_bytes134218496B (128MiB+backingVec768B), early_evicted_sessions1/sourceevicted_session. Cachedprompt0: đây là reuse vùng nhớ, không dùng lại kết quả KV không hợp lệ. Tại after_kv_lookup, activeKV128,001MiB/retainedKV0. Off cùng mốc activeKV128,001MiB+retainedKV128,001MiB; candidate counters0/disabled. Cả hai sau admission chỉ còn một bộKV. Không tạo pool ngoài ngân sách hoặc sửa giới hạn cache/session để ép kết quả.

| Stage requestPython | Off RSS (MiB) | On RSS (MiB) |
| --- | ---: | ---: |
| before_request | 2882,504 | 2887,324 |
| after_kv_lookup | 2882,613 | 2887,637 |
| prefill_buffers_live | 2882,613 | 2887,641 |
| after_prefill | 2934,199 | 2893,645 |
| before_session_admission | 2945,605 | 2893,227 |
| after_session_admission | 2945,605 | 2893,227 |
| after_generate_inner | 2945,605 | 2893,227 |

Trace7stage/request, tổng28stage/4request, trùng last_memory_trace trong status; tổngtracked vàownership hợp lệ. Stage/turn/sampler khác thời điểm nên RSS/HWM có chênh nhỏ; không trộn các peak không đồng thời thành một sample.

## Kết luận có giới hạn

**Đã xác nhận trên workload model thật này:** candidate tránh giữ đồng thời hai bộKV khi đổi phiên, giảm KV capacity cùng tồn tại khoảng128MiB, với text/usage/finish không đổi. Đây là cải thiện cấp phát/capacity của nhánh đã thử, không phải tuyên bố request không còn allocation khác hoặc tổngRAM giảm50%.

**RSS quan sát thấp hơn trong một cặp:** peakRSS lấy mẫu on thấp hơnoff51,910MiB. Max(RSS+VmSwap) cùng mẫu thấp hơn48,629MiB; tổng này chỉ là chỉ báo phụ, không accounting đầy đủ allocation/shared pages. Không lấy nó làm chứng minh đã loại trừ mọi tác động reclaim.

Điều kiện không đủ để gán mức51,910MiB thành lợi ích RSS chắc chắn: một cặp thứ tự cố định, process/allocator/OS-cache/tải nền khác; on có swapmax43,121MiB, off0. Trong sampler, pswpout toàn hệ thống tăng508581page ởon (1986,645MiB) và96058page ởoff (375,227MiB), page4096B; PSI some.total tăng2308025/831171microsecond. Cgroup là nhóm có thể gồm process khác; rootpartial được giữ. Không suy mọi reclaim do riêng model hoặc tải nền. Không dùng chênh thời gian59,984/44,503s làm kết luận candidate nhanh/chậm hơn.

Cả hai ở trong ngưỡng, nhưng chỉ2request/process; chưa chứng minh plateau/hếtleak/OOM/ổn định dài hạn, chưa kiểm lợi ích trên i7 hay workload khác. Giữ candidate opt-in/mặc địnhoff, không tăng cache hoặc nâng ngưỡng theo số capacity tiết kiệm. Tác động evict sớm khi hủy/lỗi vẫn như thiết kế và fixture, không được kiểm bằng một success pair này.

Nhánh KV có bằng chứng hữu hạn để ghi báo cáo: cơ chế reuse chạy thật, tránh một bộKV cấp mới và có tín hiệu RSS thuận lợi. Không kéo dài đo chỉ để tìm phần trăm đẹp hoặc mở lại lookahead. Khi chọn cấu hình vận hành cần cân nhắc thêm workload/memoryguard phù hợp; mục tiêu nghiệm thu tổngRAM/giaiđoạn1 vẫn chưa đóng.

## Payload

Chuyển nguyên7file mỗi block từ runs vào on/off, thêm3preflight riêng, một bộpayload duy nhất; khóa17file bằngCHECKSUMS. README không thuộc payload khóa. Đường dẫn gốc trong record/log giữ như khi đo. Giữ nguyên hồ sơ off→on4request đã thất bại và mọi raw/checksum trước đó; không ghép chúng vào cặp này.

```bash
sha256sum -c CHECKSUMS.sha256
```
