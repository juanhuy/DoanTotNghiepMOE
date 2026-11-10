# [BÁO CÁO KỲ 1] Báo cáo định kỳ lần 1

- Nội dung: [Báo cáo Markdown](BAO_CAO_DINH_KY_LAN_1.md).
- Bản dàn trang: [LaTeX](latex/main.tex), [PDF](latex/main.pdf).
- Học cách kiểm và tái lập: [Sổ tay lệnh thực nghiệm](HUONG_DAN_LENH_THUC_NGHIEM.md) — đầu vào, lệnh, file đầu ra và cách đọc số liệu.
- Phạm vi: khái niệm, kiến trúc nguyên mẫu, thuật toán và benchmark sơ bộ; đánh giá riêng từng cấu hình/model, không xếp hạng hai máy.

## Hồ sơ số liệu

- [i5 — 22/10/2026](../benchmarks/experiments/2026-10-22-phase1-i5/README.md): metadata nguồn phù hợp 0924-Instruct.
- [i7 — 23/10/2026](../benchmarks/experiments/2026-10-23-phase1-i7/README.md): thành viên xác nhận 0125-Instruct, revision nguồn chưa có metadata kèm bộ đo.

Mỗi bộ có 20 request đo và 5 warmup. Cùng câu hỏi không bảo đảm cùng chuỗi token, lịch sử hay routing. Số liệu mô tả điều kiện đã thử; chưa là đánh giá chất lượng chuẩn hóa hoặc bảo đảm an toàn bộ nhớ dài hạn. Không cần chạy thêm benchmark để hoàn thiện bản kỳ 1.

Raw và checksum giữ nguyên. Không dùng build_report.py để ghi đè artifact; chỉ sửa phần trình bày và kiểm lại số liệu từ file đã lưu. Mã revision và hash được giữ trong hồ sơ kỹ thuật, không bắt buộc đưa vào phần diễn giải của báo cáo.

## Đính chính tài liệu lưu trữ i7

README trong artifact i7 là một phần bộ checksum, giữ nguyên để bảo toàn hồ sơ. Các câu cũ nói cùng checkpoint/workload với i5, Love166/reuse11, đơn vịGB cho sốGiB và suy nguyên nhân từ phần cứng không được dùng trong báo cáo. Giá trị đúng trong raw i7 là Mars16token, Love162token/reuse143. Bản diễn giải đã hiệu chỉnh nằm trong chương 5 báo cáo chính.

## Kiểm chứng các cách khảo sát trong Bảng 1.1

Bảng 1.1 nêu cách khảo sát và câu hỏi nghiên cứu; kết luận và giới hạn của từng phép thử nằm ở chương 5. Khi trình bày, mở một request hoặc cặp thử cụ thể, chỉ đầu vào, cấu hình, kết quả rồi mới giải thích số tổng hợp.

| Dòng trong Bảng 1.1 | Mở hồ sơ nào? | Trường cần xem và phạm vi kiểm chứng |
| --- | --- | --- |
| Suy luận khi chỉ giữ một phần expert | [Một request Mars i5](../benchmarks/experiments/2026-10-22-phase1-i5/warm/turn-0-0.json), [record cả kịch bản](../benchmarks/experiments/2026-10-22-phase1-i5/warm/result.json) | messages, response.text, usage, finish_reason; config.cache_mib và trạng thái cache ở status_start/status_end trong result. Chứng minh vận hành với cache expert giới hạn; không tự chứng minh tổng trọng số vượt RAM khả dụng. |
| Đo các phần chi phí | [Cùng request Mars](../benchmarks/experiments/2026-10-22-phase1-i5/warm/turn-0-0.json) | response.metrics: prefill_ms, decode_ms, expert_io_ms, expert_compute_ms, attention_ms, lm_head_ms, total_ms. Đây là timer do engine ghi nhận, không phải profiler độc lập của GPU/SSD. |
| Hai cách phối hợp cache | [AB/BA raw](../benchmarks/experiments/2026-10-22-web-session-i5-081202/i5/20261022T092416.839618Z-web-paired-abba-443258fd6bdd4e2c84bde18575709b5e/result.json) | result.runs: policy, messages, response.metrics và usage. Có 8 request: 4 vòng × 2 prompt, thứ tự inclusive/victim/victim/inclusive. Chỉ 2 mẫu mỗi policy/prompt; xem mục 5.5. |
| Tái sử dụng tiền tố KV | [5 cặp KV prefix](../benchmarks/experiments/2026-10-22-prefill-i5/paired-session.json) | samples: pair, variant before/after, response.metrics.cached_prompt_tokens, prefill_ms và response.token_ids. Có 10 mẫu = 5 cặp; output token đầu khớp 5/5 cặp. max_tokens=1, nên không chứng minh tốc độ decode 128 token; xem mục 5.6. |
| Tuần tự và bổ sung song song | [9 cặp dispatch](../benchmarks/experiments/2026-10-22-prefill-i5/paired-prefill.json) | samples: pair, variant, prefill_ms và token_ids. Có 18 mẫu = 9 cặp, output token đầu khớp 9/9; candidate chậm hơn 7/9 cặp nên không được chọn. Xem mục 5.7. |
| Giới hạn và quan sát RAM | [Reset32](../benchmarks/experiments/2026-10-22-ram-guard-i5/reset-32/result.json), [Keep128](../benchmarks/experiments/2026-10-22-ram-guard-i5/keep-128/result.json), [log cấu hình bị chặn](../benchmarks/experiments/2026-10-22-ram-guard-i5/preflight-256/server.log) | Hai lượt 12 request có memory.jsonl/watchdog.json trong cùng thư mục. Log kiểm âm ghi reserve 7288299520 B vượt limit 5368709120 B và OutOfMemory từ guard. Đây là từ chối theo dự toán, không phải kernel OOM. Xem mục 5.8. |

### Trình tự trình bày một bằng chứng

1. Đọc câu hỏi và điều kiện thử trong mục tương ứng của chương 5.
2. Mở JSON, chỉ messages hoặc thông tin workload; chỉ config/cache/context/token limit và biến thể được thử.
3. Mở response.metrics và output; giải thích một mẫu trước khi nói trung vị. Với cặp thử, chỉ hai biến thể có cùng pair.
4. Nêu cỡ mẫu và giới hạn. Ví dụ: lợi ích KV prefix đo ở prefill với max_tokens 1; policy chỉ n=2; RAM chỉ được quan sát trong thời gian đã ghi.

Các fixture số học và log thực nghiệm có vai trò khác nhau. Outputtoken khớp ở các cặp max_tokens 1 chỉ xác nhận token đầu trong phép thử đó; việc so logits trong sai số cho phép thuộc các fixture kiểm thử, không phải đối chứng logits đầy đủ trên model 7B. Có thể mở source/harness được liên kết tại README từng thí nghiệm để giải thích cách tạo số đo.

Lệnh checksum chỉ kiểm tệp bằng chứng có bị thay đổi hay không, không chứng minh phương pháp thử đúng hoặc kết luận có thể áp dụng cho mọi trường hợp. Khi chưa cần tái đo, dùng hồ sơ đã lưu để trình bày; không cần chạy lại toàn bộ benchmark trong buổi phản biện.

## Đọc và tái lập thực nghiệm

[Sổ tay lệnh](HUONG_DAN_LENH_THUC_NGHIEM.md) giải thích đầu vào, lệnh chạy, cách JSON được ghi và cách đọc lại dữ liệu thành bảng. Các lệnh hồi cứu không phải bản chép đầy đủ lịch sử terminal; chạy lại không bảo đảm thời gian giống từng mẫu cũ.

Để thiết kế phép đo giai đoạn tiếp, dùng [SCENARIOS](../benchmarks/SCENARIOS.md), không chạy lại toàn bộ bộ kỳ 1 theo mặc định.

## Bản nộp và biên dịch

Bản PDF là báo cáo định kỳ lần 1, không kèm phụ lục. Lệnh và bản đồ bằng chứng được giữ tại [sổ tay thực nghiệm](HUONG_DAN_LENH_THUC_NGHIEM.md). Header có tên đồ án bên trái và tên chương bên phải, kể cả trang đầu chương; footer giữ tên hai sinh viên, lớp và số trang.

Phần đầu dùng số La Mã; số Ả Rập bắt đầu từ Mở đầu theo mục II.5 của hướng dẫn. Mẫu trang nội dung có câu khác về điểm bắt đầu số trang; bản này chọn quy định chính nêu rõ thứ tự các phần. Giữ một bìa và không có trang giao nhiệm vụ theo phạm vi báo cáo kỳ 1 đã thống nhất.

Biên dịch bằng `bash baocao/latex/compile.sh`. Cần Times New Roman cài trên máy, hoặc bốn file Times.TTF, Timesbd.TTF, Timesi.TTF, Timesbi.TTF trong `baocao/latex/fonts/`. Bộ font cục bộ được lấy từ [gói corefonts times32](https://sourceforge.net/projects/corefonts/files/the%20fonts/final/times32.exe/); tệp tải có SHA-1 `20b79e65cdef4e2d7195f84da202499e3aa83060`, chỉ giải nén font, không chạy EXE. Font không được đưa vào Git. Nếu thiếu, cấu hình sẽ báo lỗi để tránh xuất nhầm font thay thế. PDF đã xuất nhúng font, người đọc không cần cài thêm.

Khoảng cách sau dấu kết câu được đặt bằng khoảng cách thông thường giữa các từ; tắt giãn câu mặc định của LaTeX. Thân bài vẫn căn đều hai lề, nên khoảng cách giữa từ có thể co giãn theo dòng. Không thay đổi chuỗi câu hỏi thử nghiệm, công thức, mã lệnh và số liệu khi chuẩn hóa dấu câu.

## Tài liệu nền cho báo cáo tiếp theo

Bắt đầu từ [Hướng dẫn hệ thống](../docs/SYSTEM_GUIDE.md), sau đó đối chiếu [ma trận bằng chứng](../docs/thesis/EVIDENCE_MATRIX.md) và README của từng thí nghiệm. Hướng dẫn hệ thống mô tả trạng thái hiện tại; báo cáo kỳ 1 là snapshot lịch sử.

## Bổ sung provenance từ nhánh Huy

[Hồ sơ provenance i7](../benchmarks/experiments/2026-10-23-phase1-i7/provenance/README.md) và3JSON được giữ từ nhánh Huy. README gốc trong artifact i7 giữ nguyên byte đã khóa checksum; không sửa diễn giải cũ ngay trong payload. Báo cáo kỳ1 và PDF đã nộp giữ phiên bản hoàn chỉnh trên main; phần mới INT4/FP8 là nghiên cứu sau kỳ1, chưa được nhập như kết quả đã nghiệm thu vào bản nộp.
