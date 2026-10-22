# Workflow dùng chung

## Thay đổi engine hoặc sửa lỗi

1. Đọc AGENTS, PLAN/PROGRESS và module trong ARCHITECTURE. Nêu trigger, hành vi mong muốn, phạm vi file và invariant rủi ro.
2. Lỗi logic dùng hồi quy fixture nhỏ; hiệu năng chốt baseline/config trước. Không đổi expected để che khác biệt.
3. Diff tập trung, chạy `python3 tools/check.py --rust`; thêm release/HTTP/converter/CUDA theo loại thay đổi.
4. Review diff/bằng chứng; ghi PROGRESS cả fail/skipped và cập nhật PLAN; PR nêu vấn đề/hành vi/kiểm chứng/giới hạn.

## Review và multi-agent

Một owner tích hợp. Có thể dùng reviewer và explorer/evidence reviewer cho tác vụ đáng kể. Giao việc theo mẫu ngắn:

```text
Mục tiêu/phạm vi:
File được sửa (hoặc chỉ đọc):
Invariant/điều kiện nghiệm thu:
Lệnh kiểm chứng:
Đầu ra: findings hoặc diff + kết quả + giới hạn; không ghi PROGRESS.
```

Không bắt mọi việc qua năm persona. Role đọc-only là chỉ dẫn, quyền/sandbox vẫn kế thừa phiên cha. Hai người cần sửa cùng module thì dùng branch/worktree riêng rồi tích hợp tuần tự. Không dùng nhiều worktree để benchmark đồng thời trên cùng hardware.

## Benchmark và bằng chứng luận văn

Theo [protocol](../benchmarks/README.md). Revision sạch, chỉ định testbed/config, runner tạo run ID. Kết quả dirty/lỗi là thí nghiệm có giới hạn, không đưa vào bảng nghiệm thu. Không chạy benchmark hardware trong shared CI. Metadata đầy đủ không tự chứng minh tăng tốc: reviewer phải kiểm điều kiện tương đương, nội dung output và giới hạn trước khi cập nhật ma trận luận điểm.

## Handoff

### Tiếp nhận trên testbed i7

Mục tiêu hiện hành là ghi báo cáo vận hành trên hai cấu hình bằng cùng bộ test và cùng ngân sách cache RAM, không xếp hạng CPU i5/i7. Nên giữ revision/checkpoint/prompt/history/context/token limit/số lượt/reset-warmup thống nhất, ghi backend/threads thực nếu khác. Ngân sách chung chưa chốt trước khi xử lý OOM. Nhãn i7 là máy đồng nghiệp do người dùng khai báo, không xác nhận model CPU/GPU/RAM. Đây là hướng dẫn chạy mới trên máy đó; chưa có baseline i7 mới được nghiệm thu. Trạng thái runtime mới nhất và incident OOM i5 ở [PLAN](PLAN.md)/[PROGRESS](PROGRESS.md) phải được đọc trước khi chạy tải thật.

1. **Nhận đúng source.** Đọc AGENTS, PLAN/PROGRESS, [protocol](../benchmarks/README.md), [machines](../benchmarks/machines.json), [presets](../benchmarks/SCENARIOS.md) và skill `moe-benchmark`. Xác nhận commit với người dùng, tạo branch riêng từ commit đó. `git rev-parse HEAD` và `git status --short` phải được ghi; clone không nhận thay đổi chưa commit/push trên i5. Nếu checkout thiếu runner/bản vá được mô tả, báo thiếu revision thay vì tự nhận đã có. Không commit/reset thay đổi người dùng chỉ để làm sạch phép đo.
2. **Kiểm môi trường và checkpoint.** Theo [CONTRIBUTING](../CONTRIBUTING.md), Linux là môi trường kiểm chứng chính; máy Windows dùng workspace Linux/WSL2 phù hợp, không tuyên bố native Windows CUDA đã hỗ trợ. Model weights không nằm trong Git: đặt checkpoint đúng format, chọn `--model-dir` theo máy, đối chiếu hash nội dung trong record với i5. Không tái sử dụng symlink/đường dẫn model cá nhân của i5. Quan sát CPU/RAM/swap/GPU thực, ghi thiếu nếu không thu được; không suy hardware từ thread count hoặc tên folder lịch sử.
3. **Checks trước tải.** Chạy `python3 tools/check.py --rust --release --http`; thêm `--web` nếu có Node.js, báo skipped nếu không có. Chạy GPU probe/targeted CUDA tests khi cần GPU và phân biệt EXECUTED với skipped. Hoàn tất mọi build/tests rồi mới đo. Không có inference server/benchmark khác đồng thời trên máy i7.
4. **Kiểm RAM trước baseline.** OOM i5 đã được xác nhận; chưa có total-RSS governor được nghiệm thu. Không mặc định CPU i7 đồng nghĩa nhiều RAM hoặc không OOM, không dựa vào adaptive cache khi đối chiếu. Chọn cache/context/token limit tường minh, quan sát RSS/VmSwap/MemAvailable qua các request và chừa RAM cho OS/IDE/browser. Budget nhỏ vẫn chưa bảo đảm an toàn; dừng tải nếu memory pressure tăng, không cố tái hiện OOM. Khi incident RAM chưa được giải quyết/kiểm stability trên máy thử, chỉ làm preflight/smoke có giám sát, chưa gọi là baseline nghiệm thu.
5. **Preview rồi đo trong phạm vi được giao.** Ví dụ dưới là preview i7, cache64MiB/layer để xem cấu hình; không phải cấu hình tối ưu hoặc cấu hình đối chiếu i5 đã chốt. `auto` HTTP từ chối fallback CPU; nếu không có GPU, dùng `--gpu cpu` và báo đó là thí nghiệm backend khác. Dirty tree chỉ dùng `--allow-dirty` cho exploratory, không thêm flag này để lách điều kiện baseline sạch. Chỉ bỏ `--dry-run` khi revision/checkpoint/config và điều kiện RAM phù hợp đã được xác nhận bằng bằng chứng.

```bash
python3 tools/run_scenarios.py conversation \
  --testbed i7 --gpu auto --threads 8 --cache 64 \
  --context 512 --max-tokens 128 --runs 5 \
  --single-policy --expert-cache-policy victim --dry-run
```

`--model-dir` có thể bổ sung để chọn checkpoint trên máy i7. Lệnh trên chọn riêng victim cho bộ ghi báo cáo; dùng cùng lệnh trên i5 và chỉ đổi nhãn testbed sau khi chốt cấu hình chung an toàn. Wrapper mặc định AB/BA nếu không có --single-policy: đó là nghiên cứu policy riêng, chưa phải yêu cầu của bộ báo cáo. Khác GPU/RAM/OS/power mode/backend/threads phải ghi rõ; không tự mở mục tiêu so hardware. Không so trực tiếp với số UI/dirty runs i5 hoặc đổi đồng thời cache và code rồi gán lợi ích cho CPU.

6. **Gửi bằng chứng.** Runner lưu run ID riêng tại `benchmarks/runs/i7/`, mặc định Git-ignore. Giữ nguyên cả thư mục result/log/từng turn và warmup (HTTP), bao gồm run lỗi; gửi qua kênh handoff được người dùng giao hoặc đưa artifact đã review vào PR. Không chỉ gửi ảnh/bảng trung vị, không sửa raw i5, không đổi nhãn lịch sử trong manifest. Owner i7 cập nhật PROGRESS trên branch của mình: commit, checkpoint hash, command/config, hardware quan sát, samples, pass/fail/skipped, RAM/swap/OS-cache/tải nền và giới hạn. Hai owner tích hợp PROGRESS tuần tự khi merge, không ghi đè mục của nhau. Không tự gửi qua email/Slack hoặc push/merge khi chưa được giao.

Sau khi đọc, agent i7 phải phân biệt được ba câu hỏi: kết quả được ghi đúng nguồn; cấu hình có chạy ổn định trên máy này; và bộ test/ngân sách cache có thống nhất với protocol báo cáo trên i5 hay không. Đạt một điều không tự chứng minh hai điều còn lại.

### Nội dung bàn giao

PROGRESS giữ kết quả, PLAN giữ việc tiếp, PR giữ diff. Tin handoff chỉ cần branch/commit, mục tiêu/file, lệnh + pass/fail/skipped, artifact, giới hạn và bước tiếp theo cụ thể. Không commit transcript/token/đường dẫn model cá nhân hoặc status snapshot trùng nhau. Công việc routine trong phạm vi đã giao không cần xin xác nhận lại; giữ cơ chế approval host cho tài khoản/sandbox/hành động ngoài phạm vi.

## Kiểm RAM trước bộ đo báo cáo

Bản bảo vệ RAM hiện là guard hợp tác, chưa hard RSS cap hoặc bằng chứng hết leak. Server mặc định64MiB/layer cố định; không dùng giá trị này làm ngân sách nghiệm thu chung trước khi kiểm trên từng máy. Guard mặc định RSS=min(6GiB,RAM/2), system headroom=min(1GiB,RAM/8), override byte dương bằng MOE_RAM_LIMIT_BYTES/MOE_RAM_HEADROOM_BYTES. Không tăng guard để bỏ qua memory pressure; giảm cache/context hoặc giải phóng tài nguyên và kiểm lại.

Hoàn tất checks và build, dừng server khác, rồi preview (i7 đổi đúng testbed):

```bash
cargo build --release --locked --bin moe-tier-engine
python3 tools/check_ram_stability.py --testbed i5 --gpu auto --cache 64 \
  --cycles 12 --max-tokens 128 --keep-cache --allow-dirty --dry-run
```

Bỏ --dry-run chỉ khi máy không có inference/build/tests khác. Dirty chỉ exploratory; bỏ --allow-dirty trên revision sạch. Runner này **không phải benchmark tốc độ**: dùng prebuilt binary, giữ nguồn/hash/config/từng input/output/status và sampler250ms `memory.jsonl`. RSS guard5GiB và watchdog5,5GiB/MemAvailable1,5GiB/process swap128MiB là default riêng của phép thử; có CLI options ghi vào record. Watchdog dừng child của nó khi quá ngưỡng/mất telemetry, không kill server của người dùng. `--keep-cache` chỉ reset đầu; bỏ flag để xen kẽ reset. --cache bắt buộc và theo MiB/layer, không phải total RSS.

Kiểm result status, watchdog_violations và error/server.log trước khi diễn giải; status failed do preflight từ chối ngân sách quá lớn là kiểm âm tính khi được thiết kế như vậy, không đổi raw thành completed. Theo dõi RSS/HWM/VmSwap, memory available, session capacity và cache/recycled giữa các request; giữ giới hạn số mẫu/thời gian. Không gọi một lượt hữu hạn là chứng minh không leak hoặc ổn định mọi workload. Sau thay đổi ảnh hưởng hiệu năng, đo tốc độ độc lập sau kiểm RAM, dùng protocol chung thay vì so số với cache/budget khác.
