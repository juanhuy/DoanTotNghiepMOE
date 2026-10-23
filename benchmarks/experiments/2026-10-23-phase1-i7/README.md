# Báo cáo giai đoạn1 — bộ đo i7 phase1-v1

Đây là **bộ số liệu i7 của checkpoint hiện tại**, thực hiện trên cấu hình máy đồng nghiệp với cùng workload và thiết lập đã đo trên máy i5. Không đặt mục tiêu so sánh xếp hạng CPU hay suy diễn tăng tốc giữa hai máy.

## Phiên bản và cấu hình

Source commit `d4adefd9e53627bdc1fd250eabab436c3ac54b0d`, mọi run `git_dirty=false`, cùng source/binary/checkpoint hashes. Testbed i7 do người dùng xác nhận, hardware runner quan sát Intel(R) Core(TM) i7-14650HX (24 threads) / NVIDIA GeForce RTX 5060 Laptop GPU (7.53 GB VRAM); GPU enabled (cuBLAS SGEMV Active), 8 Rayon threads, victim policy, 64MiB cache/layer (~1GiB/16layer), context 512, max 128 output tokens. Guard RSS 5GiB / headroom 1GiB; watchdog RSS 5,5GiB / MemAvailable 1,5GiB / process swap 128MiB / 250ms.

Cold 5 requests; warm 5 requests có 5 warmup riêng; conversation 5 vòng Mars→Love (10 requests). Tổng 20 measured + 5 warmup, từng nhóm 5 mẫu. Reset chỉ xóa cache ứng dụng, không xóa OS cache; checkpoint hashing trước load làm ấm page cache. Warm dùng session mới giữa warmup và mẫu để không reuse KV. Conversation gửi đầy đủ assistant output trước trong history. Một loaded server cho mỗi scenario, không build/tests/inference khác đồng thời.

<details>
<summary>Từng mẫu đo (20 request, không gồm warmup)</summary>

| Nhóm | Cycle | Prompt tokens | Completion tokens | TTFT s | Decode steps/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold Mars | 0 | 16 | 128 | 4.062 | 4.183 | 34.421 |
| cold Mars | 1 | 16 | 128 | 4.163 | 4.121 | 34.979 |
| cold Mars | 2 | 16 | 128 | 3.873 | 4.447 | 32.429 |
| cold Mars | 3 | 16 | 128 | 4.110 | 4.463 | 32.568 |
| cold Mars | 4 | 16 | 128 | 3.615 | 4.261 | 33.420 |
| warm Mars | 0 | 16 | 128 | 2.141 | 4.906 | 28.029 |
| warm Mars | 1 | 16 | 128 | 2.276 | 4.744 | 29.046 |
| warm Mars | 2 | 16 | 128 | 2.016 | 4.862 | 28.135 |
| warm Mars | 3 | 16 | 128 | 2.121 | 4.961 | 27.721 |
| warm Mars | 4 | 16 | 128 | 2.363 | 4.568 | 30.167 |
| conversation Mars | 0 | 16 | 128 | 3.651 | 4.240 | 33.602 |
| conversation Mars | 1 | 16 | 128 | 3.912 | 3.971 | 35.892 |
| conversation Mars | 2 | 16 | 128 | 3.727 | 4.298 | 33.279 |
| conversation Mars | 3 | 16 | 128 | 4.103 | 4.172 | 34.543 |
| conversation Mars | 4 | 16 | 128 | 3.976 | 3.440 | 40.893 |
| conversation Love | 0 | 162 | 128 | 3.234 | 3.717 | 37.404 |
| conversation Love | 1 | 162 | 128 | 2.929 | 3.603 | 38.173 |
| conversation Love | 2 | 162 | 128 | 2.379 | 3.734 | 36.391 |
| conversation Love | 3 | 162 | 128 | 2.389 | 3.223 | 41.799 |
| conversation Love | 4 | 162 | 128 | 3.357 | 3.069 | 44.732 |

</details>

## Kết quả chính

Mỗi ô là trung vị của 5 mẫu. Khoảng wall là min–max, không phải confidence interval. Decode là engine model steps/s sau token đầu, không số chunk SSE. Prompt là `Where is Mars` và `What is love ?` đúng spec trong raw.

| Nhóm | TTFT s | Decode steps/s | Wall s | Wall min–max s |
| --- | ---: | ---: | ---: | ---: |
| cold Mars | 4.062 | 4.261 | 33.420 | 32.429–34.979 |
| warm Mars | 2.141 | 4.862 | 28.135 | 27.721–30.167 |
| conversation Mars | 3.912 | 4.172 | 34.543 | 33.279–40.893 |
| conversation Love | 2.929 | 3.603 | 38.173 | 36.391–44.732 |

## Diễn giải chi phí

Love có 166 prompt tokens so với 19 của Mars, chỉ reuse 11 prompt tokens; cùng session không đồng nghĩa toàn bộ lịch sử đã có KV dùng lại. TTFT cao hơn đi cùng phần prefill dài hơn, không đủ bằng chứng quy hết cho SSD.

Các timer dưới đây là median, có thể chồng lấn, không cộng thành wall. File payload là bytes ứng dụng yêu cầu đọc, có thể đến từ OS page cache; không phải SSD physical I/O.

| Nhóm | Expert load wait s | Expert compute s | Attention s | File payload GB |
| --- | ---: | ---: | ---: | ---: |
| cold Mars | 13.485 | 8.312 | 0.864 | 5.757 |
| warm Mars | 9.573 | 7.664 | 0.855 | 0.000 |
| conversation Mars | 13.745 | 8.990 | 0.910 | 5.757 |
| conversation Love | 13.718 | 9.083 | 1.922 | 0.176 |

## RAM trong bộ đo

| Scenario | Observed HWM max GiB | Process VmSwap max KiB | MemAvailable min GiB | Thời gian quan sát s |
| --- | ---: | ---: | ---: | ---: |
| cold | 3.161 | 0 | 5.463 | 169.1 |
| warm | 3.278 | 0 | 6.347 | 314.1 |
| conversation | 3.164 | 0 | 5.607 | 378.2 |

Ba record status success, watchdog violations rỗng; output text và usage khớp 5 mẫu trong mỗi nhóm, Mars khớp cả ba scenario. Cả 20 mẫu kết thúc reason `length` ở 128 completion tokens; đây là giới hạn sinh chủ động, không chứng minh câu trả lời hoàn chỉnh. HTTP không trả token IDs/logits nên đây không phải bit-exact logits7B. Chỉ chứng minh vận hành trong cấu hình/thời gian đã đo, không bảo đảm mọi workload không OOM/leak. RSS/HWM khác cache capacity, GPU logical khác physical allocation.

## Phạm vi nghiệm thu và phần còn lại

Bộ đo hoàn tất việc thu thập số liệu trên testbed i7 bằng đúng checkpoint, source revision và protocol đã đo trên i5. Kết quả phản ánh năng lực vận hành thực tế của engine trên cấu hình phần cứng i7 (Intel Core i7-14650HX / RTX 5060 Laptop GPU). Không quy trực tiếp mức chênh lệch giữa hai máy cho riêng CPU vì khác biệt kiến trúc GPU, RAM bandwidth và tải nền.

<details>
<summary>Hồ sơ tái lập/audit</summary>

- [Summary machine-readable](summary.json), [script tổng hợp](build_report.py).
- [Cold raw](cold/result.json), [warm raw](warm/result.json), [conversation raw](conversation/result.json).
- Từng folder có build/server log, memory.jsonl/watchdog.json, source snapshots và từng turn; warmup chỉ ở warmup files/result.warmups.
- Kiểm byte bất biến bằng `sha256sum -c CHECKSUMS.sha256` trong thư mục này.
- Protocol/lệnh ở [SCENARIOS](../../SCENARIOS.md), trạng thái ở [PLAN](../../../docs/PLAN.md), lịch sử checks ở [PROGRESS](../../../docs/PROGRESS.md).

</details>
