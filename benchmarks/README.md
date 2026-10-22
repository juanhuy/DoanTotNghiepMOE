# Giao thức benchmark hai testbed

Hai máy là `i5` (máy người dùng) và `i7` (máy đồng nghiệp), xem [machines.json](machines.json). Đây là nhãn khai báo, không phải suy luận từ số luồng. 4/8 Rayon threads là cấu hình khởi đầu ghi rõ, **chưa chứng minh tối ưu**. Không ghi thông số RAM bandwidth/kiến trúc GPU từ phỏng đoán.

Agent nhận workspace máy i7 làm theo mục **Tiếp nhận trên testbed i7** trong [WORKFLOWS](../docs/WORKFLOWS.md): kiểm source/checkpoint/môi trường/RAM trước khi đo và giữ raw theo testbed; OOM i5 hiện chưa được giải quyết không cho phép mặc định baseline ổn định trên i7.

## Mục tiêu bộ ghi báo cáo i5/i7

Chạy cùng bộ test và cùng ngân sách cache RAM trên từng cấu hình để mô tả khả năng vận hành của engine. Giữ revision/checkpoint, preset/prompt/history, context/token limit, số mẫu và reset/warmup thống nhất; ghi hardware/backend/threads thực của từng máy. Không đặt mục tiêu xếp hạng i5/i7 hoặc quy chênh lệch cho riêng CPU. Nếu sau này nghiên cứu causal hardware comparison, đó là thí nghiệm riêng.

Ngân sách cache chung chưa chốt vì OOM i5 còn mở. Sau kiểm RAM, dùng `--cache` tường minh ở cả hai máy; cache budget khác total RSS. Bộ báo cáo runtime mặc định victim dùng `--single-policy --expert-cache-policy victim` khi gọi `run_scenarios.py`; mặc định AB/BA của wrapper dành cho nghiên cứu policy, không tự coi là protocol báo cáo đã chốt. Các con số cache/threads trong ví dụ dưới là minh họa, không phải cấu hình nghiệm thu an toàn.

## Dữ liệu và nguồn gốc

- `results/`: 40 JSON lịch sử duy nhất, giữ nguyên payload. [manifest.json](manifest.json) chứa SHA-256, kind, nhãn lịch sử, status và alias bản sao đã bỏ. `historical_unverified` nghĩa chưa xác minh lại đầy đủ provenance; checksum không chứng minh tác giả/điều kiện phép đo.
- `runs/<i5|i7>/<UTC-uuid>/`: run mới có `result.json` và `stderr.log`, tạo riêng không ghi đè. Các run mới được bỏ qua bởi Git mặc định; chọn artifact đã review để đưa vào báo cáo/PR, không commit mọi lần chạy thử.
- `suites/`: kịch bản version control. Suite hash và nội dung nằm trong record mới khi áp dụng.
- [docs/BENCHMARKS.md](../docs/BENCHMARKS.md): catalog lịch sử sinh quyết định; không tạo bảng đối đầu hai máy từ các commit/config khác nhau.

```bash
python3 tools/export_benchmark_evidence.py --check
```

## Chạy phép đo mới

Hoàn tất checks/build trước, để máy rảnh và cắm nguồn; ghi chế độ nguồn/tải nền nếu muốn kết luận hiệu năng. Không chỉnh source, suite, checkpoint hoặc chạy build/tests/inference khác trong suốt phép đo. Runner build đúng example trước khi timing model, hash checkpoint, thu hardware quan sát được và giữ logs.

```bash
# Preview không đo, không cần model:
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --allow-dirty --dry-run

# Sau khi commit một revision sạch, trên máy i5:
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --cache 128 --threads 4 --runs 3

# Máy i7: chạy cùng bộ test/ngân sách cache để ghi báo cáo riêng từng máy;
# ghi rõ threads/backend thực, không quy chênh lệch cho riêng CPU.
python3 tools/run_benchmark.py --testbed i7 --gpu cpu --cache 128 --threads 4 --runs 3

# Suite, giá trị cache/context thực được truyền xuống binary:
python3 tools/run_benchmark.py --testbed i7 --gpu auto --threads 8 --cache 128 --suite benchmarks/suites/suite-v1.json
```

`--gpu` và `--testbed` bắt buộc. `--model-dir` chọn checkpoint khác; mặc định context512/cache128MiB/runs3/max_tokens16 cho single prompt. Suite dùng `max_tokens` của từng case, không dùng max token single prompt. `--case` chọn case của suite. Mẫu 3 lượt trong cùng process gồm 1 engine-cold + 2 warm, không đủ để gọi là thí nghiệm cold-start độc lập nhiều mẫu; muốn đo cold nhiều mẫu cần nhiều process/run ID.

Runner từ chối dirty tree trừ khi `--allow-dirty`: dữ liệu đó là exploratory, không dùng cho đối chiếu có kiểm soát vì bool dirty không tái dựng được patch. Cơ chế publish dùng hard-link nguyên tử cùng filesystem; nếu filesystem không hỗ trợ, run lỗi rõ thay vì ghi đè.

## Giới hạn phải giữ khi báo cáo

- Hash nội dung model **đọc file trước khi đo**, ảnh hưởng page cache. Cache OS không kiểm soát; engine-cold chỉ nghĩa cache expert của process bắt đầu rỗng, không có chứng minh SSD-cold.
- Chế độ `cpu` tắt GPU compute **sau load**. Load có thể đã khởi tạo GPU/cấp VRAM; load_ms gồm phần đó. Không gọi đây là baseline load thuần CPU. `auto` có thể fallback CPU; báo trạng thái quan sát được, không chỉ request GPU.
- GPU status hiện là logical telemetry có giới hạn R7/circuit breaker; nhãn enabled không chứng minh mọi kernel chạy GPU. Test GPU cần probe và kiểm tra thiết bị thật riêng.
- `expert_bytes_read` hiện trộn file/VRAM; không gọi là số byte SSD. I/O/compute timer chồng lấn, tổng timer không phải wall time. RSS khác allocator capacity và page cache.
- Load, prefill, TTFT, decode và tổng generate đo riêng. Decode throughput hiện đếm model steps sau token đầu; EOS/token count cần đọc cùng kết quả.
- So tối ưu code: cùng hardware/config, chỉ revision thay đổi và mô tả diff. So hai máy: cùng revision/hash/workload/config, ghi mọi khác biệt hardware/threads/backend. Không trộn hai câu hỏi để suy ra nguyên nhân tăng tốc.
- Đưa từng mẫu, sample size và phân tán trước khi nêu trung vị/tỷ lệ. Không lựa lượt tốt nhất rồi gọi đại diện. Quality exact/substring là smoke heuristic, không phải benchmark chất lượng chuẩn hóa.

## Kịch bản HTTP/SSE hai lượt cùng session

Các runner cold/warm/conversation/topics/long-context/cache-sweep có sẵn tại [hướng dẫn kịch bản](SCENARIOS.md), gồm preview, số request và giới hạn diễn giải.

Chạy qua cùng API mà web dùng: reset cache, history/session mới, `Where is Mars`, rồi `What is love ?` với câu trả lời trước trong history. `--runs` là số vòng; với `--compare-cache-policies` là số cặp, thứ tự AB/BA trên cùng một server đã nạp model. Mỗi policy đều reset trước Mars, Love giữ session của Mars. Mặc định runtime/runner hiện dùng `victim`; chỉ định `inclusive` để đo reference cũ. `auto` ở mode này từ chối fallback CPU để không tạo so sánh GPU giả.

```bash
python3 tools/run_benchmark.py --http-session --compare-cache-policies \
  --testbed i5 --gpu auto --threads 8 --cache 128 --max-tokens 128 \
  --runs 2 --allow-dirty --tag web-abba --dry-run
# Bỏ --dry-run để đo sau khi hoàn tất checks và dừng inference khác.
```

Dirty run là exploratory; giữ source/hash/binary/checkpoint/config/server log, từng turn và result thành công. Parser SSE từ chối generation_error, thiếu finish/DONE và metrics không hợp lệ. Client first-content sau headers khác TTFT engine; process read_bytes delta khác file payload. Tham khảo [bằng chứng i5 và giới hạn](experiments/2026-10-22-web-session-i5-081202/README.md). Không dùng số TTFT của workload sửa history/1 output token để thay kịch bản hai lượt128 token.

## Đưa run mới thành bằng chứng

Reviewer kiểm metadata/log/status/output và điều kiện trước khi chọn artifact. Giữ run ID, result + log cùng nhau, ghi lý do/scope claim vào EVIDENCE_MATRIX/PROGRESS. Có thể force-add riêng thư mục run đã chấp nhận vào PR; không sửa số liệu để khớp claim. Manifest hiện khóa corpus lịch sử; không nhét schema v2 vào corpus cũ hoặc giả status historical. Chưa có benchmark mới được nghiệm thu trong lần chuẩn hóa này.
