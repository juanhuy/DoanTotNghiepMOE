# Báo cáo định kỳ lần 1

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

## Các JSON được tạo từ đâu?

### Bộ cold/warm/conversation

Luồng tạo dữ liệu là: run_scenarios.py → run_benchmark.py → benchmark_http_session.py → server Rust → phản hồi HTTP/SSE → ghi file JSON. Đây là luồng phần mềm tạo bộ kết quả; các con số trong response.metrics do engine đo bằng timer và bộ đếm, không phải văn bản do model sinh.

- Engine sinh văn bản, token usage và metrics. HTTP handler gửi các trường đó trong sự kiện kết thúc stream.
- Client Python thu stream, ghép văn bản và giữ metrics/usage. Runner bổ sung messages đã gửi, session, policy và thông tin tiến trình trước/sau request.
- run_cycle trong [benchmark_http_session.py](../tools/benchmark_http_session.py) gọi hàm ghi JSON để tạo turn-{cycle}-{turn}.json hoặc warmup-{cycle}-{index}.json.
- execute trong cùng file tạo result.json, thêm cấu hình, thông tin model/hardware, hashes và tập request. MemoryWatchdog ghi memory.jsonl và watchdog.json.
- summary.json là bản tổng hợp sau phép đo, không phải một request mới.

Lời gọi tương ứng với bộ warm i5 có cấu hình như sau, chạy từ thư mục gốc dự án:

```bash
python3 tools/run_scenarios.py warm \
  --testbed i5 --gpu auto --threads 8 \
  --model-dir models/olmoe-1b-7b-int8 \
  --cache 64 --context 512 --max-tokens 128 --runs 5 \
  --single-policy --expert-cache-policy victim \
  --output-root benchmarks/runs/recheck
```

Lệnh này tổ chức năm vòng warm, mỗi vòng gồm một request làm ấm và một request đo. Nó tạo bộ kết quả mới ở thư mục riêng, không viết lại artifact ngày22/10. Đổi warm thành cold hoặc conversation để chọn kịch bản tương ứng; đổi testbed/model-dir theo máy và model thực tế. Runner tự khởi tạo server, không cần mở server thủ công trước. Không chạy đồng thời với một phiên inference khác.

Có thể thêm --dry-run để chỉ xem kế hoạch mà không suy luận. Workspace đang có thay đổi chưa commit thì preview cần thêm --allow-dirty. Đây là hướng dẫn tái hiện phương pháp; muốn đối chứng một thay đổi cần khớp phiên bản code, model và các điều kiện liên quan. Chạy lại không bảo đảm từng giá trị thời gian giống hệt vì cache hệ điều hành, tải nền và nhiệt độ có thể khác.

### Các JSON khác

- paired-session.json và paired-prefill.json do hai bài thử Rust benchmark_session_prefix_paired và benchmark_prefill_dispatch_paired trong [olmoe.rs](../src/olmoe.rs) tạo. Chúng chạy hai biến thể trong cùng tiến trình và ghi từng cặp mẫu, không đi qua HTTP. Lệnh chạy và điều kiện nằm trong [README prefill](../benchmarks/experiments/2026-10-22-prefill-i5/README.md).
- Bộ RAM do [check_ram_stability.py](../tools/check_ram_stability.py) tạo bằng request HTTP và quan sát tiến trình; không phải benchmark tốc độ cùng loại với paired prefill.
- Các file source snapshots, log và hashes hỗ trợ truy lại cách chạy. JSON có dữ liệu không tự chứng minh phép đo đúng; cần đọc cùng cấu hình và code ghi metric như bản đồ kiểm chứng ở trên.

## Xem lại dữ liệu thành bảng, không chạy model

Đoạn sau chỉ đọc năm file đã lưu của warm i5 rồi in bảng; không nạp model, gửi request hoặc tạo số đo mới:

```python
import json
from pathlib import Path

folder = Path("benchmarks/experiments/2026-10-22-phase1-i5/warm")
print("Luot | TTFT(s) | Decode(steps/s) | WALL(s)")
for path in sorted(folder.glob("turn-*-0.json")):
    sample = json.loads(path.read_text())
    metric = sample["response"]["metrics"]
    print(sample["cycle"] + 1,
          round(metric["time_to_first_token_ms"] / 1000, 3),
          round(metric["decode_tokens_per_second"], 3),
          round(metric["total_ms"] / 1000, 3), sep=" | ")
```

Có thể chạy đoạn này trong Python terminal hoặc lưu thành script tạm rồi gọi python3. Đây cũng là dữ liệu có thể đưa vào bảng tính hoặc vẽ biểu đồ; hình/bảng sinh lại từ raw phải được ghi là biểu diễn dữ liệu đã đo, không phải lần đo mới.

## Bản nộp và biên dịch

Bản PDF là báo cáo định kỳ lần 1, không kèm phụ lục. Lệnh và bản đồ bằng chứng được giữ tại [sổ tay thực nghiệm](HUONG_DAN_LENH_THUC_NGHIEM.md). Header có tên đồ án bên trái và tên chương bên phải, kể cả trang đầu chương; footer giữ tên hai sinh viên, lớp và số trang.

Phần đầu dùng số La Mã; số Ả Rập bắt đầu từ Mở đầu theo mục II.5 của hướng dẫn. Mẫu trang nội dung có câu khác về điểm bắt đầu số trang; bản này chọn quy định chính nêu rõ thứ tự các phần. Giữ một bìa và không có trang giao nhiệm vụ theo phạm vi báo cáo kỳ 1 đã thống nhất.

Biên dịch bằng `bash baocao/latex/compile.sh`. Cần Times New Roman cài trên máy, hoặc bốn file Times.TTF, Timesbd.TTF, Timesi.TTF, Timesbi.TTF trong `baocao/latex/fonts/`. Bộ font cục bộ được lấy từ [gói corefonts times32](https://sourceforge.net/projects/corefonts/files/the%20fonts/final/times32.exe/); tệp tải có SHA-1 `20b79e65cdef4e2d7195f84da202499e3aa83060`, chỉ giải nén font, không chạy EXE. Font không được đưa vào Git. Nếu thiếu, cấu hình sẽ báo lỗi để tránh xuất nhầm font thay thế. PDF đã xuất nhúng font, người đọc không cần cài thêm.

Khoảng cách sau dấu kết câu được đặt bằng khoảng cách thông thường giữa các từ; tắt giãn câu mặc định của LaTeX. Thân bài vẫn căn đều hai lề, nên khoảng cách giữa từ có thể co giãn theo dòng. Không thay đổi chuỗi câu hỏi thử nghiệm, công thức, mã lệnh và số liệu khi chuẩn hóa dấu câu.
