# Báo cáo giai đoạn1 — bộ đo i5 phase1-v1

Đây là **bộ số liệu i5 của checkpoint hiện tại**, tách khỏi các thí nghiệm tối ưu trước đó. Đọc trang này để review; raw/log/checksum chỉ cần khi audit. Không có phép đo i7 mới, không xếp hạng CPU hoặc suy speedup so với cấu hình khác.

## Phiên bản và cấu hình

Source commit `d4adefd9e53627bdc1fd250eabab436c3ac54b0d`, mọi run `git_dirty=false`, cùng source/binary/checkpoint hashes. i5 do người dùng xác nhận, hardware runner quan sát Core i5-10500H/GTX1650 Max-Q; GPU enabled,8 Rayon threads, victim,64MiB cache/layer (~1GiB/16layer), context512, max128 output tokens. Guard RSS5GiB/headroom1GiB; watchdog RSS5,5GiB/MemAvailable1,5GiB/process swap128MiB/250ms.

Cold5 requests; warm5 requests có5 warmup riêng; conversation5 vòng Mars→Love (10 requests). Tổng20 measured+5 warmup, từng nhóm5 mẫu. Reset chỉ xóa cache ứng dụng, không xóa OS cache; checkpoint hashing trước load làm ấm page cache. Warm dùng session mới giữa warmup và mẫu để không reuse KV. Conversation gửi đầy đủ assistant output trước trong history. Một loaded server cho mỗi scenario, không build/tests/inference khác đồng thời.

<details>
<summary>Từng mẫu đo (20 request, không gồm warmup)</summary>

| Nhóm | Cycle | Prompt tokens | Completion tokens | TTFT s | Decode steps/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold Mars | 0 | 19 | 128 | 2.757 | 3.655 | 37.506 |
| cold Mars | 1 | 19 | 128 | 2.121 | 3.777 | 35.743 |
| cold Mars | 2 | 19 | 128 | 1.791 | 3.783 | 35.364 |
| cold Mars | 3 | 19 | 128 | 1.795 | 3.663 | 36.468 |
| cold Mars | 4 | 19 | 128 | 2.104 | 3.651 | 36.886 |
| warm Mars | 0 | 19 | 128 | 2.039 | 3.980 | 33.950 |
| warm Mars | 1 | 19 | 128 | 2.044 | 3.987 | 33.899 |
| warm Mars | 2 | 19 | 128 | 2.023 | 3.989 | 33.863 |
| warm Mars | 3 | 19 | 128 | 2.005 | 3.995 | 33.791 |
| warm Mars | 4 | 19 | 128 | 2.019 | 3.963 | 34.065 |
| conversation Mars | 0 | 19 | 128 | 2.181 | 3.685 | 36.648 |
| conversation Mars | 1 | 19 | 128 | 2.005 | 3.673 | 36.586 |
| conversation Mars | 2 | 19 | 128 | 2.014 | 3.681 | 36.519 |
| conversation Mars | 3 | 19 | 128 | 1.984 | 3.687 | 36.425 |
| conversation Mars | 4 | 19 | 128 | 2.009 | 3.674 | 36.575 |
| conversation Love | 0 | 166 | 128 | 9.477 | 3.207 | 49.080 |
| conversation Love | 1 | 166 | 128 | 9.099 | 3.228 | 48.446 |
| conversation Love | 2 | 166 | 128 | 8.759 | 3.250 | 47.836 |
| conversation Love | 3 | 166 | 128 | 9.109 | 3.234 | 48.374 |
| conversation Love | 4 | 166 | 128 | 9.086 | 3.225 | 48.461 |

</details>

## Kết quả chính

Mỗi ô là trung vị của5 mẫu. Khoảng wall là min–max, không phải confidence interval. Decode là engine model steps/s sau token đầu, không số chunk SSE. Prompt là `Where is Mars` và `What is love ?` đúng spec trong raw.

| Nhóm | TTFT s | Decode steps/s | Wall s | Wall min–max s |
| --- | ---: | ---: | ---: | ---: |
| cold Mars | 2.104 | 3.663 | 36.468 | 35.364–37.506 |
| warm Mars | 2.023 | 3.987 | 33.899 | 33.791–34.065 |
| conversation Mars | 2.009 | 3.681 | 36.575 | 36.425–36.648 |
| conversation Love | 9.099 | 3.228 | 48.446 | 47.836–49.080 |

## Diễn giải chi phí

Love có166 prompt tokens so với19 của Mars, chỉ reuse11 prompt tokens; cùng session không đồng nghĩa toàn bộ lịch sử đã có KV dùng lại. TTFT cao hơn đi cùng phần prefill dài hơn, không đủ bằng chứng quy hết cho SSD.

Các timer dưới đây là median, có thể chồng lấn, không cộng thành wall. File payload là bytes ứng dụng yêu cầu đọc, có thể đến từ OS page cache; không phải SSD physical I/O.

| Nhóm | Expert load wait s | Expert compute s | Attention s | File payload GB |
| --- | ---: | ---: | ---: | ---: |
| cold Mars | 12.075 | 8.852 | 1.864 | 36.705 |
| warm Mars | 10.841 | 9.133 | 1.893 | 34.176 |
| conversation Mars | 11.947 | 9.289 | 1.910 | 36.705 |
| conversation Love | 13.911 | 14.309 | 4.179 | 38.156 |

I/O và expert compute đều còn đáng kể; bộ này chưa đo bandwidth/latency thiết bị hoặc A/B nên chưa xác nhận đã chạm trần phần cứng hay tìm được tối ưu mới.

## RAM trong bộ đo

| Scenario | Observed HWM max GiB | Process VmSwap max KiB | MemAvailable min GiB | Thời gian quan sát s |
| --- | ---: | ---: | ---: | ---: |
| cold | 3.340 | 0 | 6.845 | 183.3 |
| warm | 3.494 | 0 | 6.730 | 352.9 |
| conversation | 3.348 | 0 | 7.085 | 426.3 |

Ba record status success, watchdog violations rỗng; output text và usage khớp5 mẫu trong mỗi nhóm, Mars khớp cả ba scenario. Cả20 mẫu kết thúc reason `length` ở128 completion tokens; đây là giới hạn sinh chủ động, không chứng minh câu trả lời hoàn chỉnh. HTTP không trả token IDs/logits nên đây không phải bit-exact logits7B. Chỉ chứng minh vận hành trong cấu hình/thời gian đã đo, không bảo đảm mọi workload không OOM/leak. RSS/HWM khác cache capacity, GPU logical khác physical allocation.

## Phạm vi nghiệm thu và phần còn lại

Engine INT8, cache phân tầng, streaming/Stop/reset và guard đã có fixture/host CUDA/manual web và kiểm RAM trước bộ này (xem PROGRESS). Bộ này cung cấp số liệu end-to-end lặp lại trên source sạch. Quality12-case smoke trước checkpoint đạt10/12, có hai câu sai; không đo lại chất lượng chuẩn hóa ở đây và không coi tốc độ là chất lượng.

Không so trực tiếp với UI9,44tok/s hoặc thí nghiệm cache128/256 cũ. Guard là hợp tác, có thể overshoot; OS cache, nhiệt/tần số và tải desktop không cố định. Năm mẫu không chứng minh tối ưu toàn cục. i7 phải tự chạy cùng source/config/protocol sau preflight và giữ raw riêng; không mượn kết quả i5. INT4/backend mới/quality chuẩn hóa/cgroup hard cap ngoài phạm vi triển khai giai đoạn1 hiện tại; các giới hạn phải tiếp tục được ghi rõ.



<details>
<summary>Hồ sơ tái lập/audit</summary>

- [Summary machine-readable](summary.json), [script tổng hợp](build_report.py).
- [Cold raw](cold/result.json), [warm raw](warm/result.json), [conversation raw](conversation/result.json).
- Từng folder có build/server log, memory.jsonl/watchdog.json, source snapshots và từng turn; warmup chỉ ở warmup files/result.warmups.
- Kiểm byte bất biến bằng `sha256sum -c CHECKSUMS.sha256` trong thư mục này. Corpus40 lịch sử không đổi; payload đã copy giữ SHA khớp.
- Protocol/lệnh ở [SCENARIOS](../../SCENARIOS.md), trạng thái ở [PLAN](../../../docs/PLAN.md), lịch sử checks ở [PROGRESS](../../../docs/PROGRESS.md).

</details>
