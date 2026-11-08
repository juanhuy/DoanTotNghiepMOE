# Benchmark của MoE-TierEngine

Kết quả mới05/11: [KVreuse hẹp trên model thật i5](experiments/2026-11-05-kv-reuse-narrow-i5/README.md): một cặp on→off hoàn tất, KVcapacity256→128MiB và đầu ra2/2khớp. PeakRSS quan sát thấp hơn51,910MiB; pressure khác/on có swap, chưa chứng minh mức giảm tổng quát hoặc hếtOOM.

Câu hỏi nghiên cứu hiện hành: **Trong ngân sách bộ nhớ cho trước, cách quản lý và di chuyển expert ảnh hưởng thế nào đến thời gian phản hồi, tốc độ sinh và lượng dữ liệu phải truyền?** GPU là tài nguyên tùy chọn; ngân sách RAM và VRAM phải ghi riêng.

## Bắt đầu từ đâu?

| Bạn cần làm gì? | Đọc ở đâu? |
| --- | --- |
| Chọn phép thử, hiểu đầu vào và cách chạy | [SCENARIOS.md](SCENARIOS.md) |
| Đọc kết quả và lệnh của báo cáo lần 1 | [Sổ tay thực nghiệm](../baocao/HUONG_DAN_LENH_THUC_NGHIEM.md) |
| Biết công việc tiếp theo | [PLAN.md](../docs/PLAN.md) |
| Kiểm nguồn một kết luận cũ | README trong thư mục thí nghiệm tương ứng, liên kết bên dưới |

Không cần đọc toàn bộ JSON hoặc chạy lại mọi thí nghiệm. Một đợt đánh giá vận hành dùng bộ chính; một cải tiến chỉ cần phép đối chứng liên quan và các kiểm tính đúng/ổn định cần thiết.

Kết quả allocator08/11: [12request có chỉ báo allocator](experiments/2026-11-08-allocator-i5/README.md) hoàn tất; RSS cuối request tăng284,547MiB nhưng in-use+mmap tăng0,234MiB, free tăng205,777MiB. Ủng hộ allocator giữ vùng đã free, chưa chứng minh hếtleak/plateau.

## Kết quả đã có

| Hồ sơ | Vai trò |
| --- | --- |
| [KVreuse off/on i5 05/11, chưa thành cặp](experiments/2026-11-05-kv-reuse-ab-i5/README.md) | Blockoff dừng do swap sau3/4request, blockon chưa chạy; chưa có kết luận RSS/speedup của candidate |
| [Trace stage i5 05/11](experiments/2026-11-05-memory-stage-i5/README.md) |4request có trace: xác nhận hai bộKV khoảng128MiB cùng tồn tại khi đổi phiên; chưa chứng minh nguyên nhân toàn bộ tăngRSS hoặc lợi ích candidate reuse |
| [RAM/reclaim i5 05/11](experiments/2026-11-05-ram-reclaim-i5/README.md) | Telemetry mới trên model thật:12/12request hoàn tất/processswap0, vẫn tăngRSS288MiB; có swap/reclaim hệ thống, chưa xác định tác nhân sự cố cũ |
| [Chuỗi RAM i5 04/11](experiments/2026-11-04-ram-sequence-i5/README.md) | Lookahead tắt,10/12request hoàn tất rồi watchdog dừng do swap; RSS tăng khi capacity cuối request gần cố định, chưa chứng minh ổn định |
| [CPU trước load i5 04/11](experiments/2026-11-04-cpu-load-i5/README.md) | Bỏ CUDA khi chọnCPU;4request kiểm RAM và1request chẩn đoán hoàn tất; gate chặn lookahead ở945/945lượt lớp có lớp kế tiếp, chưa đo speedup |
| [Lookahead i5 03/11](experiments/2026-11-03-lookahead-i5/README.md) | Lượt khảo sát dừng do swap trước cặp đầu; chưa có kết luận hiệu năng |
| [Vận hành i5](experiments/2026-10-22-phase1-i5/README.md) | Cold, warm, hội thoại trên cấu hình i5 |
| [Vận hành i7](experiments/2026-10-23-phase1-i7/README.md) | Cùng loại kịch bản trên cấu hình/model i7; không xếp hạng hai máy |
| [Inclusive/victim](experiments/2026-10-22-web-session-i5-081202/README.md) | Đối chứng bố trí cache hai tầng, số mẫu nhỏ |
| [KV và dispatch](experiments/2026-10-22-prefill-i5/README.md) | Lợi ích tái sử dụng tiền tố trong workload đã thử; dispatch nhóm bị loại |
| [RAM i5](experiments/2026-10-22-ram-guard-i5/README.md) | Quan sát bộ nhớ hữu hạn và kiểm từ chối cấu hình quá dự toán |
| [Kiểm chức năng model thật](experiments/2026-10-22-stage1-validation/README.md) | Smoke và tiêu chí nội dung sơ bộ |

Các hồ sơ này khác cấu hình, thời điểm và mục tiêu. Không ghép thành một chuỗi tăng tốc. Hai bộ i5/i7 dùng checkpoint khác nhau theo báo cáo kỳ 1; nhận định “cùng checkpoint hai máy” trong hướng dẫn cũ là mục tiêu đối chứng, không phải sự kiện đã được chứng minh.

Đính chính khi đọc hồ sơ i7: đoạn diễn giải Love 166/Mars 19/reuse 11 mang số của i5; bảng và raw i7 có đầu vào 162/16, reuse lượt tiếp 143. Xem báo cáo kỳ 1 và raw để diễn giải đúng. Không sửa README đã khóa checksum để tránh phá hồ sơ cũ.

## Quy tắc đo và đọc kết quả

- Chọn `--testbed i5` hoặc `i7` tường minh; [machines.json](machines.json) chỉ là nhãn và cấu hình khởi đầu, không phải xác nhận phần cứng hiện tại.
- Đối chứng một cơ chế trên cùng máy/checkpoint/workload. Giữ ngân sách, threads, backend, context và giới hạn sinh tương đương; công bố bộ đệm tạm/nạp trước trong ngân sách.
- Hoàn tất build/tests trước đo; chỉ chạy một model mỗi lúc. Dùng guard/watchdog và kiểm RAM tại máy thử. Mặc định runner không phải chứng nhận an toàn.
- Preview bằng `--dry-run`. Đo có kiểm soát cần source sạch; `--allow-dirty` cho phép khảo sát nhưng phải ghi giới hạn. Không tự commit thay đổi người dùng để đo.
- Cache rỗng của ứng dụng không phải SSD/OS cache lạnh. Hash checkpoint trước chạy có thể làm ấm page cache.
- Ghi token thực tế, từng mẫu và độ phân tán. Hội thoại có câu trả lời sinh ra làm lịch sử nên workload có thể khác giữa model hoặc biến thể.
- RAM cache budget khác RSS. VRAM logical khác cấp phát vật lý. Từ bản vá 04/11/2026, `MOE_GPU_MODE=cpu`/runner `--gpu cpu` bỏ qua CUDA ngay lúc load; record cũ CPU sau load vẫn có thể đã cấp VRAM. Kiểm `gpu_mode_at_load` và trạng thái thực.
- Byte đọc file là payload ứng dụng, không phải byte SSD vật lý. Timer có thể chồng lấn, không cộng thành wall time. Decode hiện đếm model steps sau token đầu.
- GPU được yêu cầu/enabled không chứng minh mọi phép tính chạy GPU. Kiểm tính đúng bằng fixture và smoke nội dung không thay thế đánh giá chất lượng chuẩn hóa.

## Lưu dữ liệu

`tools/run_benchmark.py` là entry point chung; `tools/run_scenarios.py` chọn kịch bản HTTP. Run mới nằm ở `runs/<testbed>/<run-id>/`, không ghi đè. Chỉ đưa hồ sơ đã review vào Git, không cần lưu mọi preview/lần thử.

`experiments/` giữ hồ sơ từng thí nghiệm. `results/` và [manifest.json](manifest.json) giữ corpus lịch sử; `historical_unverified` không phải số đo mới. [Catalog lịch sử](../docs/BENCHMARKS.md) dùng tra cứu, không phải bảng so máy.

Giữ nguyên payload/checksum và cả kết quả thất bại. Đính chính cách hiểu tại tài liệu tổng hợp, không sửa số gốc.

```bash
python3 tools/check.py --web
```

Lệnh kiểm tài liệu, catalog và tooling; không nạp model, không tạo benchmark mới. Nhật ký thay đổi ở [PROGRESS.md](../docs/PROGRESS.md).

Kiểm ngân sách08/11: [12request i5](experiments/2026-11-08-ram-budget-i5/README.md) hoàn tất, sampledRSSpeak3169,426MiB/processswap0 nhưng RSSrequest1→12 tăng271,168MiB; chưa chứng minh plateau hoặc ổn định dài hạn.
