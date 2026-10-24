# Báo cáo giai đoạn 1 — bộ đo i7 phase1-v1

Đây là **bộ số liệu vận hành trên testbed i7**, thực hiện trên cấu hình máy đồng nghiệp theo cùng giao thức kịch bản (cold, warm, conversation) và ngân sách cache (64 MiB/layer). Bộ đo sử dụng checkpoint `allenai/OLMoE-1B-7B-0125-Instruct` (khác với checkpoint `0924-Instruct` trên máy i5, chi tiết xem thư mục [provenance/](provenance/)). Không đặt mục tiêu so sánh xếp hạng hai máy hay suy diễn tăng tốc giữa hai cấu hình phần cứng.

## Phiên bản và cấu hình

Source commit `d4adefd9e53627bdc1fd250eabab436c3ac54b0d`, mọi run `git_dirty=false`, cùng source/binary/checkpoint hashes. Testbed i7 do người dùng xác nhận; hardware runner quan sát được: Intel(R) Core(TM) i7-14650HX (24 logical threads, 16 physical cores), 15,910,060,032 bytes RAM vật lý (≈ 14.817 GiB / 15.910 GB), NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB VRAM / 7.53 GB khả dụng); GPU runtime enabled thực (`cuBLAS SGEMV Active`, offload 1.38 GB Dense weights), 8 Rayon threads, expert cache policy `victim`, 64 MiB cache/layer (~1 GiB/16 layers), context 512, max 128 output tokens. Guard RSS 5 GiB / headroom 1 GiB; watchdog RSS 5.5 GiB / MemAvailable 1.5 GiB / process swap 128 MiB / lấy mẫu 250 ms.

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

Trên i7, câu Love có 162 prompt tokens so với 16 của Mars, và quan sát thấy cached prompt tokens là 143 tokens (nhờ đó TTFT prefill chỉ 2.929s). Mức reuse 143 tokens trên i7 so với 11 tokens trên i5 là một quan sát thực nghiệm; giả thuyết sơ bộ là định dạng chat template của bản 0125-Instruct cho phép khớp tiền tố chuỗi tốt hơn, nhưng chưa khẳng định khi chưa có đối chiếu chuỗi token prefix độc lập.

Các timer dưới đây là median, có thể chồng lấn, không cộng thành wall. File payload là bytes ứng dụng yêu cầu đọc qua pread, có thể đến từ OS page cache hoặc prefetch chưa dùng; không phải lượng dữ liệu SSD đọc vật lý.

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

Ba record status success, watchdog violations rỗng; output text và usage khớp 5 mẫu trong mỗi nhóm, Mars khớp cả ba scenario. Cả 20 mẫu kết thúc reason `length` ở 128 completion tokens; đây là giới hạn sinh chủ động, không chứng minh câu trả lời hoàn chỉnh và không mô tả như đã dừng bằng EOS. HTTP không trả token IDs/logits nên đây không phải bit-exact logits7B. Kết quả chỉ chứng minh vận hành trong cấu hình/thời gian đã đo, không bảo đảm mọi workload không OOM/leak. RSS/HWM khác cache capacity, GPU logical khác physical allocation.

## Phạm vi nghiệm thu và phần còn lại

Bộ đo thu thập số liệu vận hành độc lập trên testbed i7 với source revision `d4adefd9e53627bdc1fd250eabab436c3ac54b0d` và checkpoint `0125-Instruct`. Kết quả phản ánh năng lực vận hành riêng của engine trên cấu hình phần cứng i7 (Intel Core i7-14650HX / RTX 5060 Laptop GPU). Không xếp hạng hai máy và không quy trực tiếp mức chênh lệch giữa hai máy cho riêng CPU hay GPU khi chưa có thí nghiệm đối chứng có kiểm soát. Hồ sơ nguồn gốc checkpoint được lưu tại thư mục [provenance/](provenance/).

<details>
<summary>Hồ sơ tái lập/audit</summary>

- [Summary machine-readable](summary.json), [script tổng hợp](build_report.py).
- [Cold raw](cold/result.json), [warm raw](warm/result.json), [conversation raw](conversation/result.json).
- Từng folder có build/server log, memory.jsonl/watchdog.json, source snapshots và từng turn; warmup chỉ ở warmup files/result.warmups.
- Hồ sơ nguồn gốc checkpoint tại [provenance/](provenance/).
- Kiểm byte bất biến bằng `sha256sum -c CHECKSUMS.sha256` trong thư mục này.
- Protocol/lệnh ở [SCENARIOS](../../SCENARIOS.md), trạng thái ở [PLAN](../../../docs/PLAN.md), lịch sử checks ở [PROGRESS](../../../docs/PROGRESS.md).

</details>
