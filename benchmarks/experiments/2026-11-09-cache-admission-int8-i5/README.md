# Admission theo tần suất: không giảm nạp expert, đã loại candidate

Ngày09/11/2026. **Không đạt mục tiêu giảm I/O ở cùng RAM.** Mars tăng số file loads/payload1,53%; Python tăng2,56%, giống nhau ở cảhai cặp ABBA. Candidate đã được gỡ khỏi code chạy chính; raw/source ZIP và kết quả bất lợi được giữ nguyên.

Một số thời gian quan sát ngắn hơn, nhưng điều kiện I/O/OS-cache/swap khác nhau mạnh; chưa quy được speedup cho policy. Không lấy các timer đó để cứu tiêu chí giảm I/O đã đặt trước. Kết quả âm này chỉ loại policy admission thử ở workload đã đo, không chứng minh mọi policy tần suất hoặc mọi tối ưu đều vô ích.

## Candidate đã thử

Mặc định cũ always: expert đã tính xong được đưa vào L2 khi đủ budget; khi đầy thì chọn victim theo thứ tự hiện có (probationary/protected, pin, heat/recency).

Candidate frequency CPU-only opt-in: với một miss đã tính xong, nếu cần evict và newcomer có heat thấp hơn victim, đồng thời một victim đủ chỗ cho newcomer, bỏ admission newcomer để giữ victim. Hits/ties/còn chỗ vẫn theo hành vi cũ. Không bỏ tính expert, không dự đoán routing, không đổi precision/sampling hoặc tăng RAM.

Dùng chính heat lịch sử của engine, không tần suất biết trước tương lai. Bypass nhiều không tự là tốt: candidate bỏ2873/3172admission theo prompt, cachehits lại giảm2947→2894 và3497→3413. Có thể policy giữ nhóm nóng cũ quá lâu; chưa có trace phân định theo thời gian nên đây là suy luận, không kết luận đã chứng minh nguyên nhân.

## Protocol chốt trước chạy

[cache-admission-v1](../../SCENARIOS.md): một ABBA always→frequency→frequency→always, bốn process mới, mỗi process hai phiên độc lập “Where is Mars”/“Explain a Python dictionary.”, max32 mỗi request. Tổng8request; không warmup riêng, reset đầu block rồi keepcache.

CPU4/victim/cache64MiB-layer/context512, paired Gate–Up cố định, FP32KV/reuseoff/lookaheadoff/memorytraceon; greedy/temp0/top-p1/top-k0/repetition1/hotpin0, không usage side state/disk session. Cùng binary/source/checkpoint, chỉ mode admission khác. Hashcheckpoint làm ấm OS cache; không SSD/OS-cache-cold đồng nhất.

Guard5120/headroom1024/watchdogRSS5632/available1536/processswap128MiB giữ nguyên; không build/tests/inference khác đồng thời. [Preflight](preflight.json) ba mẫu cách5s: available ít nhất9864996KiB, swap-used0/pswpout0, không thấy engine/build khác.

Primary phải giảm cả file loads và payload ít nhất5% ở từng prompt trong cảhai cặp; output parity bắt buộc. Counter công việc được phép khác vì đó là mục tiêu của policy. Không đạt, không tuning/rerun để lấy kết quả đẹp.

Lệnh đã chạy trên snapshot candidate (hiện đã gỡ option khỏi CLI):

    python3 tools/run_benchmark.py --testbed i5 --gpu cpu --cache-admission-ab --threads 4 --cache 64 --context 512 --max-tokens 32 --lookahead off --allow-dirty

Không chạy lệnh này trên nguồn hiện tại. Nếu cần tái lập, tạo checkout riêng từ revision trong raw rồi áp source ZIP candidate của một child run; checkpoint nằm ngoài Git. Không giải nén đè workspace đang phát triển.

## Từng mẫu và lượng công việc

| Block | Admission | Prompt | File loads | Payload GiB | Bypass | Cache hits | Total s | Expert compute s | Expert I/O s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | always | Mars | 3453 | 20.285110 | 0 | 2947 | 13.672 | 3.297 | 9.529 |
| 0 | always | Python | 3287 | 19.309921 | 0 | 3497 | 12.386 | 3.493 | 8.256 |
| 1 | frequency | Mars | 3506 | 20.596466 | 2873 | 2894 | 12.265 | 3.088 | 7.735 |
| 1 | frequency | Python | 3371 | 19.803391 | 3172 | 3413 | 9.647 | 3.044 | 4.400 |
| 2 | frequency | Mars | 3506 | 20.596466 | 2873 | 2894 | 7.474 | 2.418 | 3.251 |
| 2 | frequency | Python | 3371 | 19.803391 | 3172 | 3413 | 7.474 | 2.597 | 2.971 |
| 3 | always | Mars | 3453 | 20.285110 | 0 | 2947 | 8.140 | 2.768 | 3.282 |
| 3 | always | Python | 3287 | 19.309921 | 0 | 3497 | 7.568 | 2.727 | 3.035 |

Pair1 là block0(always)→1(frequency); pair2 là block3(always)→2(frequency). Hai pair có cùng số loads/payload theo mỗi prompt:
- Mars3453→3506loads,21.780.971.520→22.115.287.040byte (+1,5349%).
- Python3287→3371loads,20.733.870.080→21.263.728.640byte (+2,5555%).
- Evictions giảm3293→473 ở Mars và3287→199 ở Python, nhưng số lần đọc lại tăng. Ít eviction không đồng nghĩa cache hữu ích hơn.

Payload là byte ứng dụng nạp cho expert, không byte SSD vật lý. Timer compute/I/O có thể chồng lấn; không cộng hai cột thành total. Dùng raw để lấy số không làm tròn; không ghép với thời gian Gate–Up max64/revision khác.

## Bộ nhớ và pressure

| Block | Sampled peak RSS MiB | Max VmSwap MiB | Min available MiB | Δpswpout pages | Δpswpin pages | Δprocess read_bytes GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2935.695 | 0.000 | 7035.234 | 194 | 0 | 13.549 |
| 1 | 2933.750 | 0.000 | 6765.746 | 200742 | 820 | 6.870 |
| 2 | 2937.309 | 0.000 | 6721.777 | 595 | 1011 | 1.550 |
| 3 | 2936.848 | 0.000 | 6694.910 | 10 | 1824 | 1.332 |

Page size4096B. Process swap0 trong các sample, không watchdog violation/OOM trong cửa sổ đo. Swap hệ thống và read_bytes thay đổi rõ; riêng block1 có200742pages swap-out. /proc read_bytes là chỉ báo tiến trình, không counter thiết bị. Không nghiệm thu plateau/hard-cap/no-leak hoặc suy candidate đã cải thiện tổng RAM.

## Kiểm tính đúng và provenance

8/8request completed, mỗi request32completiontokens, Mars19prompttokens/Python22. Cùng prompt khớp text/usage/finish qua cảbốn block; trace completed7mốc/status khớp, status/metrics đúng mode/INT8/FP32/CPU, VRAM0, maps không CUDA/cuBLAS. HTTP không cho logits/tokenIDs model thật; không khái quát bit-exact mọi input.

[Regression trước guard](validation/regression-before.log) FAIL đúng trigger: expert nóng0 bị thay bởi newcomer1. [Sau guard](validation/regression-after.log)2/2PASS, gồm admit newcomer khi nóng hơn và parity MHA/GQA logits tolerance2e-5/token/prefix dưới budget một expert. Không thay expected/tolerance.

[Checks candidate](validation/checks.log): python3 tools/check.py --rust --release --web PASS,129Rust/profile (110lib+6HTTP+7loader+6reference),7ignored/profile,fmt/Clippy,86Python và7Node. [Build server](validation/build.log) PASS. [HTTP fixture frequency](validation/fixture-frequency.json)2request completed/cache1/context64/max3/2threads; kiểm metadata/trace, không hiệu năng7B.

[Record tổng](run/result.json): 20261109T024023.065706Z-admission-f85afca29d7c411b9ca99b6ecdd02dc2; HEAD5663c7ab7c8f905edb2a218a3f524fda342b6179, dirty exploratory trên work/i5-gate-up-ab. Binary SHA-2560712a7cf9321a0e196477ee589baf7c2b637308d9173748b664eeb2768f740ef; cảbốn ZIP có cùng57sourcehash và checkpoint manifest/config9f1312f8… giống nhau. Đã kiểm source/ZIP/binary khớp ngay sau timing, trước khi gỡ candidate.

Hardware: Intel(R) Core(TM) i5-10500H CPU @ 2.50GHz, 12logicalCPU/RAM16571342848B. Inventory GPU nằm trong raw; GPU compute không chạy. Không coi GPU unit test pass là CUDA đã kiểm. Converter Python và stress HTTP riêng không chạy ở bước này.

## Quyết định và vị trí source

Không giữ MOE_CACHE_ADMISSION=frequency, telemetry/Layer fields và CLI/tool thử admission trong code chính. Khôi phục đúng5file đã thay cho candidate từ snapshot Gate–Up resident đã kiểm ngay trước đó, gỡ hai tool candidate do bước này tạo; giữ toàn bộ thay đổi trước và báo cáo/raw.

Những số đo trên mô tả **snapshot candidate đã gỡ**, không phải working tree sau cleanup. Source đầy đủ để xem lại ở source.zip của từng child. Không sửa JSON để đổi revision/hash hoặc giả vờ đây là cấu hình hiện tại.

[CHECKSUMS.sha256](CHECKSUMS.sha256) khóa35payload:29file run,preflight và5validationfile. README diễn giải ngoài payload khóa; source ZIP/JSON giữ nguyên. Các child result:

- Block0/always: [result](run/block-0-always/i5/20261109T024023.072925Z-ram-0d912c190cff4c1dae184e081f1537f6/result.json); source.zip/memory.jsonl/server.log/turn-*.json cùng thư mục.
- Block1/frequency: [result](run/block-1-frequency/i5/20261109T024109.523784Z-ram-d0eb4cdaa8694f79be375f9de9789ce6/result.json); source.zip/memory.jsonl/server.log/turn-*.json cùng thư mục.
- Block2/frequency: [result](run/block-2-frequency/i5/20261109T024150.785069Z-ram-c16f0bf184834f8693f8ff66993ec589/result.json); source.zip/memory.jsonl/server.log/turn-*.json cùng thư mục.
- Block3/always: [result](run/block-3-always/i5/20261109T024224.238253Z-ram-3e1d8abdc58b4c3eb971e1f96ac0102d/result.json); source.zip/memory.jsonl/server.log/turn-*.json cùng thư mục.
