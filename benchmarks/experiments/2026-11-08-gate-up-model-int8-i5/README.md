# Gate–Up trên model INT8 thật: đối chứng i5

Ngày 08/11/2026. **Đã kiểm đường chạy thật và hoàn tất 8/8 request, nhưng chưa xác nhận tăng tốc toàn request do kernel.** Cặp đầu có lợi cho paired, cặp đảo thứ tự có lợi cho separate; áp lực bộ nhớ và I/O thay đổi mạnh giữa các lượt. Không lấy trung bình để công bố một phần trăm tăng tốc.

Kernel paired đã có từ nhánh Huy. Đóng góp của bước này là selector đối chứng, kiểm tính đúng trên model và bằng chứng về giới hạn khi đưa kết quả microbenchmark vào inference thật. Giữ model INT8; không bật INT4/FP8 và không cần model 16B.

## Câu hỏi và cách đo

Theo [protocol gate-up-model-v1](../../SCENARIOS.md), được ghi trước khi chạy: liệu thay hai dot SIMD riêng bằng dot ghép Gate–Up, với cùng trọng số và lượng công việc, có giảm thời gian expert và toàn request không?

- Bốn process mới theo separate → paired → paired → separate (ABBA), mỗi process hai phiên độc lập: “Where is Mars” rồi “Explain a Python dictionary.” Tổng hai cặp cho mỗi prompt; không warmup riêng, reset đầu block và giữ cache giữa hai request.
- CPU trước load, Rayon 4 threads, victim cache 64 MiB/layer, context 512, max 64 token. FP32 KV, KV buffer reuse off, lookahead off, memory trace on; greedy, temperature 0, top-p 1, top-k 0, repetition penalty 1; hot-pin 0, không disk session hoặc usage side state.
- Cùng source, binary và checkpoint trong cả bốn block; chỉ mode khác. Baseline separate gọi hai dot SIMD, không phải scalar. Candidate paired dùng kernel hiện có; mặc định production vẫn paired.
- Guard RSS 5120 MiB/headroom 1024 MiB; watchdog RSS 5632 MiB/available 1536 MiB/process swap 128 MiB. Không nới ngưỡng, không chạy build/tests/inference khác đồng thời.
- Một lượt duy nhất; dừng nếu lỗi, watchdog, đầu ra/counter/cấu hình lệch. Không chạy lại để chọn kết quả đẹp. Các timer expert/I/O có thể chồng lấn, không cộng thành total.

Lệnh thực thi (bỏ qua --dry-run sau preview):

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --gate-up-model-ab \
  --threads 4 --cache 64 --context 512 --max-tokens 64 --lookahead off \
  --allow-dirty
```

[Preflight](preflight.json): ba mẫu cách nhau 5 giây, available ít nhất 9289 MiB, swap đã dùng và pswpout đều 0, không thấy engine/build khác. Điều này không bảo đảm không có pressure trong lúc chạy.

## Provenance và kiểm tính đúng

[Record tổng](run/result.json): run 20261108T150612.721077Z-gate-up-model-bef1e56cc6c3443994fae9a0369fe9b9; HEAD 5663c7ab7c8f905edb2a218a3f524fda342b6179, dirty exploratory trên branch work/i5-gate-up-ab. Không phải kết quả nghiệm thu từ revision sạch.

Hardware quan sát: i5-10500H, 12 logical CPU, RAM 16.571.330.560 byte; inventory thấy GTX 1650 Max-Q 4096 MiB/driver 615.78.08. Cả bốn server chạy CPU, GPU disabled/VRAM logical 0; probe maps không thấy CUDA/cuBLAS. Không kiểm CUDA compute.

Checkpoint có config SHA-256 9f1312f8c9d1f447839e4962ee3d3124c724231cd44003ed81522939922b3d7b. Danh sách file/hash đầy đủ ở từng child result; dtype quan sát int8. Hash checkpoint trước mỗi load làm ấm OS cache, không có điều kiện SSD/OS-cache-cold đồng nhất.

Binary SHA-256 dc38dcb08e4d0b4cfc32d669ff4fc7daa7d3c438b868e31011af08074d516759. Cả bốn snapshot chứa cùng 53 source hash; kiểm source ZIP/hash/source workspace và binary sau đo khớp. Đường dẫn output trong raw giữ nguyên vị trí lúc chạy; archive chỉ di chuyển file, không sửa JSON.

Ở cả bốn block, cùng prompt có text/usage/finish_reason và năm counter cache/file khớp hoàn toàn. Mỗi request sinh 64 token; Mars dùng 19 prompt token, Python dùng 22. Mỗi request có trace completed/7 mốc và khớp snapshot status; mode đúng yêu cầu, cached_prompt_tokens=0. Không có logits/token IDs của model thật qua HTTP, nên không suy ra bit-exact trên mọi input.

## Từng mẫu thời gian

Đơn vị giây, làm tròn ba chữ số; đọc raw để lấy giá trị gốc. Prefill, decode và TTFT ghi riêng; không cộng TTFT vào total.

| Block | Mode | Prompt | Total | Prefill | Decode | TTFT | Expert compute | Expert I/O |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | separate | Mars | 49.030 | 5.243 | 43.779 | 5.251 | 12.166 | 39.330 |
| 0 | separate | Python | 43.678 | 5.002 | 38.669 | 5.009 | 12.312 | 33.859 |
| 1 | paired | Mars | 21.202 | 4.655 | 16.540 | 4.661 | 5.798 | 13.979 |
| 1 | paired | Python | 15.178 | 3.607 | 11.564 | 3.613 | 4.973 | 7.943 |
| 2 | paired | Mars | 18.945 | 3.813 | 15.124 | 3.821 | 6.661 | 9.045 |
| 2 | paired | Python | 18.508 | 4.003 | 14.495 | 4.013 | 6.547 | 8.764 |
| 3 | separate | Mars | 16.130 | 3.149 | 12.975 | 3.155 | 5.507 | 8.016 |
| 3 | separate | Python | 15.681 | 3.097 | 12.561 | 3.120 | 5.722 | 7.563 |

Hai cặp được chốt là (block 0 separate, block 1 paired) và (block 3 separate, block 2 paired). Tỷ số separate/paired >1 có lợi cho paired:

| Cặp | Prompt | Total separate/paired | Expert compute separate/paired |
| --- | --- | ---: | ---: |
| 1 | Mars | 2.313× | 2.098× |
| 1 | Python | 2.878× | 2.476× |
| 2 | Mars | 0.851× | 0.827× |
| 2 | Python | 0.847× | 0.874× |

Cặp đầu cho total 2,313×/2,878×, nhưng cặp đảo cho 0,851×/0,847×. Expert compute cũng đảo chiều. Đây là kết quả chưa đủ để quy lợi ích end-to-end cho kernel; không loại cặp bất lợi, không chỉ chọn các lượt đầu.

## Bộ nhớ và I/O: vì sao chưa kết luận nhân quả

| Block | Sampled peak RSS MiB | Peak VmSwap MiB | Min available MiB | Δpswpout pages | Δpswpin pages | Δprocess read_bytes GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2948.910 | 0.273 | 6373.934 | 54071 | 11094 | 56.244 |
| 1 | 2955.695 | 0.000 | 6538.012 | 48030 | 64 | 13.159 |
| 2 | 2957.023 | 0.000 | 6521.016 | 274 | 673 | 7.320 |
| 3 | 2954.418 | 0.000 | 6495.098 | 0 | 534 | 6.751 |

Page size trong raw là 4096 byte. Block 0/1 có swap hệ thống đáng kể; block 0 còn có VmSwap thoáng qua 0,273 MiB dù snapshot cuối request là 0. Watchdog không vi phạm ngưỡng, không có OOM trong cửa sổ đo. Đây không phải bằng chứng plateau, không leak hoặc total RAM bị hard-cap.

Payload expert cùng prompt không đổi: Mars 6084 file loads/38.376.898.560 byte, Python 5740/36.207.001.600 byte. Ngược lại Δprocess read_bytes qua /proc giảm mạnh giữa các block. Hai đại lượng không đồng nghĩa: payload ứng dụng không phải byte SSD vật lý; /proc là chỉ báo I/O tiến trình trong cửa sổ sampling, không counter thiết bị NVMe. Sự khác nhau phù hợp với điều kiện OS cache/reclaim thay đổi, chưa phân định hết các tác nhân.

Expert compute là thời gian tường bao quanh các đoạn compute, vẫn có thể chịu scheduling/pressure; không phải số chu kỳ CPU thuần. Không đo affinity, tần số hoặc hardware performance counter. Chỉ hai cặp mỗi prompt, không đủ một ước lượng speedup tổng quát.

## Kiểm tra và hướng tiếp

- [Checks trước đo](validation/checks.log): python3 tools/check.py --rust --release --web PASS; 126 Rust tests/profile (107 lib + 6 HTTP + 7 loader + 6 reference), 6 ignored/profile; fmt/Clippy, 76 Python tests và 7 Node tests.
- Fixture MHA/GQA so logits tolerance 2e-5 và token/finish/prefix parity. Hai lượt HTTP release nhỏ [separate](validation/fixture-separate.json)/[paired](validation/fixture-paired.json), mỗi lượt 2 request/context64/max3/cache1/2 threads: completed, output parity và trace 7 mốc. Đây là kiểm chức năng, không phải hiệu năng 7B.
- CUDA compute, converter Python và HTTP stress suite riêng không chạy trong bước này; không suy GPU unit test pass thành compute đã kiểm.
- [Microbenchmark trước đó](../2026-11-08-gate-up-int8-i5/README.md) vẫn là bằng chứng riêng: full expert synthetic/resident/1 thread median pair ratio 1,160×; không chuyển nó thành chat nhanh 16%.
- Bước tiếp theo được thu hẹp: đối chứng compute với weights thật, buffers resident có giới hạn và cách chia task 4 threads như engine, AB/BA trong cùng process. Chốt workload/parity/budget trước chạy; tách đọc file khỏi timing. Chưa triển khai/chưa đo bước này. Không tiếp tục lặp HTTP hiện tại để tìm phần trăm thuận lợi.

## Tra raw

[CHECKSUMS.sha256](CHECKSUMS.sha256) khóa 33 payload: 29 file của phép đo, preflight và 3 file validation. README là diễn giải, không sửa số gốc. Mapping child run ở record tổng:

- Block 0, separate: [result](run/block-0-separate/i5/20261108T150612.728066Z-ram-ca8294eadbe84512b382ff50769f3bff/result.json); source.zip, memory.jsonl, server.log và turn-*.json cùng thư mục.
- Block 1, paired: [result](run/block-1-paired/i5/20261108T150807.278671Z-ram-5f753338b38d4b5c8e897b12f6ba4014/result.json); source.zip, memory.jsonl, server.log và turn-*.json cùng thư mục.
- Block 2, paired: [result](run/block-2-paired/i5/20261108T150906.454118Z-ram-a87f7078df5e4eb6921aec58905c3d4d/result.json); source.zip, memory.jsonl, server.log và turn-*.json cùng thư mục.
- Block 3, separate: [result](run/block-3-separate/i5/20261108T151003.321116Z-ram-82ef2155efe74a81b07d7ae5f36d2ee0/result.json); source.zip, memory.jsonl, server.log và turn-*.json cùng thư mục.
