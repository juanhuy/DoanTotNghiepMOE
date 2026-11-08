# Workflow dùng chung

## Thay đổi engine và kiểm chứng

1. Đọc AGENTS, PLAN/PROGRESS và module liên quan trong ARCHITECTURE. Xác định trigger, hành vi mong muốn, phạm vi sửa và invariant.
2. Với lỗi logic, dùng hồi quy fixture; không sửa expected/tolerance để che regression. Với tối ưu, chốt câu hỏi và điều kiện đối chứng trước.
3. Chạy `python3 tools/check.py --rust`; thêm release/HTTP/web/converter/CUDA theo thay đổi. Hoàn tất checks/build trước benchmark.
4. Ghi PROGRESS với thay đổi, lệnh, pass/fail/skipped, bằng chứng và giới hạn. Review diff trước commit; PR mô tả vấn đề, hành vi và kiểm chứng.

Một owner tích hợp. Hai người dùng branch/worktree riêng; không sửa đồng thời cùng file hoặc chạy benchmark song song trên cùng máy. Agent đọc AGENTS và hai skill trong .agents/skills/: moe-engine cho Rust/cache/FFI, moe-benchmark cho đo và diễn giải bằng chứng. Không cần MCP/plugin/tài khoản API để build engine. Không đưa token hoặc cấu hình cá nhân vào repo.

## Tiếp nhận trên testbed i7

### Nhận phiên bản bàn giao

Branch bàn giao lần này: **handoff/i5-memory-20261108**. Commit chính xác lấy từ remote sau khi owner push thành công; không dùng HEAD75d5a62 của raw cũ làm revision của bản vá hiện tại. Đọc thông báo bàn giao để đối chiếu commit. Branch Huy là công việc riêng của đồng nghiệp, không tự reset hoặc merge đè.

Trong checkout riêng, kiểm working tree trước:

```bash
git status --short
git fetch origin
git switch -c i7/verify-i5-memory-20261108 --track origin/handoff/i5-memory-20261108
git rev-parse HEAD
```

Lệnh switch dùng khi branch local mới chưa tồn tại. Nếu đã có branch đó, switch về branch local rồi cập nhật bằng `git pull --ff-only` khi working tree phù hợp. Không reset/clean để xóa thay đổi. Nếu đang có thay đổi trên Huy hoặc branch khác, giữ chúng và dùng checkout/worktree riêng; không mang diff chưa commit sang phép kiểm rồi gọi đó là cùng revision.

### Đọc đúng kết quả và phạm vi

1. Đọc [SYSTEM_GUIDE](SYSTEM_GUIDE.md), [PLAN](PLAN.md), các mục mới nhất trong [PROGRESS](PROGRESS.md), [benchmark guide](../benchmarks/README.md), [machines](../benchmarks/machines.json) và skill moe-benchmark.
2. Bản mới có CPU trước load, lookahead telemetry, memorytrace/allocator/TID, KVbufferreuse opt-in và runner có provenance. KVbufferreuse mặc định off; memorytrace off; lookahead mặc định on nhưng các protocol RAM hiện dùng off tường minh.
3. i5 đã có [phép allocator12request](../benchmarks/experiments/2026-11-08-allocator-i5/README.md): RSS tăng284,547MiB nhưng allocator indicator tăng0,234MiB; free tăng205,777MiB. Đây là bằng chứng giữ vùng đã giải phóng trong cửa sổ đó, chưa chứng minh hết leak/plateau/hardcap hoặc điều gì trên i7. **Không cần lặp cuộc điều tra allocator trên i7 như thể chưa làm.**
4. Bộ kỳ1 i5/i7 dùng checkpoint khác nhau theo báo cáo; không xếp hạng CPU. Với phép đối chiếu mới, cần cùng revision/checkpoint/suite/prompt/history/context/tokenlimit/cachebudget/số lượt, ghi backend/threads/hardware riêng.

### Kiểm môi trường và fixture trước

Linux là môi trường kiểm chứng chính; trên Windows dùng workspace Linux/WSL2 phù hợp, không coi native Windows CUDA đã được nghiệm thu. Theo [CONTRIBUTING](../CONTRIBUTING.md), chuẩn bị Rust stable/rustfmt/Clippy và Python3.11+; Node phục vụ web tests.

```bash
python3 tools/check.py --rust --release --http --web
cargo build --release --locked --bin moe-tier-engine
```

Nếu thiếu Node, bỏ --web và báo skipped. --http cần localhost; dùng quyền socket phù hợp môi trường. Fixture không cần checkpoint7B hoặc GPU. GPU unit tests pass có thể chưa chạy CUDA: báo probe/EXECUTED/skipped đúng thực tế. Ghi OS/kernel/WSL, CPU/RAM/swap/GPU quan sát, Rustversion, commit/dirty và kết quả checks. Không kế thừa số i5 thành kết quả i7.

### Smoke HTTP CPU có giám sát

Sau checks/build và khi không có inference/build khác, preview trên fixture nhỏ:

```bash
python3 tools/check_ram_stability.py --testbed i7 --gpu cpu --lookahead off \
  --memory-trace on --kv-buffer-reuse off --workload mixed --cache 1 \
  --context 64 --cycles 2 --max-tokens 3 --keep-cache --threads 2 \
  --model-dir tests/fixtures/olmoe/unnormalized-mha --dry-run
```

Sau khi preview đúng và máy đủ RAM cho guard, bỏ --dry-run để chạy đúng2request. Đây là smoke fixture, không số đo tốc độ/RAM7B hoặc chứng minh donorKVreuse đã kích hoạt. Không thêm --allow-dirty để lách baseline sạch; nếu phải khảo sát dirty thì công bố exploratory.

Kiểm result.status/watchdog_violations/server.log, trace7stage mỗi request trùng last_memory_trace. Allocator status trên LinuxGNU có thể là glibc_mallinfo2 hoặc symbol_unavailable; trên nền tảng khác có thể unsupported_platform. Null là chưa biết, không thay bằng0. in_use+mmap là indicator, không exactliveheap; query_ns chỉ đo lời gọi allocator.

### Model thật: chỉ theo protocol đã được giao

Weights không nằm trong Git. Dùng checkpoint đúng format trên máy i7, chọn --model-dir và đối chiếu hash với hồ sơ mục tiêu. Không convert lại checkpoint merged đã hợp lệ; không kế thừa symlink/đường dẫn model cá nhân của i5. `machines.json` là nhãn/cấu hình khởi đầu, hardware thực phải được thu riêng.

Chưa chốt cache chung i5/i7 và chưa giao một cuộc tối ưu riêng trên i7. Nếu được giao cửa sổ RAM model thật, xem [RAM_BUDGET](RAM_BUDGET.md), dự toán/preflight và preview sau:

```bash
python3 tools/check_ram_stability.py --testbed i7 --gpu cpu --lookahead off \
  --memory-trace on --kv-buffer-reuse off --workload mixed --cache 64 \
  --context 512 --cycles 12 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --dry-run
```

Đây chỉ là preview, không yêu cầu chạy12request ngay. 4threads giữ cấu hình chẩn đoán, không được đo là tối ưu cho i7. Thu3mẫu5s trước tải; hoãn nếu pswpout tăng hoặc MemAvailable dưới1536MiB. Không nới guard/đổi OS để vượt lỗi, không chạy CPU và GPU cùng lúc. CPU đã chọn trước load không khởi tạo CUDA; nếu chọn auto phải ghi trạng thái GPU thực và ngân sách VRAM, không ghép với CPU như cùng điều kiện.

## Bàn giao điều tra prefill/RAM (08/11/2026)

Mục này giữ anchor cho các hồ sơ trước. Điều tra hiện đã có kết quả allocator nêu trên; hướng i5 tiếp theo ở PLAN. Nếu cần mở lại, câu hỏi phải cụ thể: indicator cuối request có tăng không, hay chủ yếu là allocator giữ vùng đã free? Không tự bật trim/đổi allocator/tăng cache hoặc cài profiler và lặp7B chỉ vì RSS chưa phẳng.

Các raw/source snapshot cũ mang HEAD trước khi publish và dirty flags là sự kiện lịch sử, không sửa chúng thành commit mới. Source ZIP và hashes dùng để truy nguyên đúng lần đo. Source hiện hành có thể khác snapshot trong các experiment trước.

## Bằng chứng và phối hợp phát triển

- Runner lưu riêng vào benchmarks/runs/i7/, mặc định Git-ignore. Giữ cả result/log/từng turn/memory/source và kết quả lỗi. Không chỉ gửi ảnh hoặc bảng median.
- Ghi config/checkpoint/revision/source/binary/hardware, từng mẫu, pass/fail/skipped, RAM/swap/pressure và giới hạn. Hoàn tất build/tests rồi mới đo; một benchmark trên một máy.
- Đưa artifact đã review vào experiment mới, giữ một payload và checksum; không sửa raw i5, đổi nhãn manifest hoặc tạo nhiều statusreport trùng nội dung.
- Owner i7 ghi PROGRESS trên branch riêng; tích hợp tuần tự, không ghi đè mục i5. Thay đổi engine phải có mục tiêu/file/invariant/kiểm chứng rõ, tránh hai người sửa cùng module.
- Push/merge/gửi ra dịch vụ theo yêu cầu người dùng. Việc push bản bàn giao hiện tại không cấp phép tự merge công việc khác của đồng nghiệp.
- Khi đối chứng hiệu năng, timers có thể chồng lấn; payload không bằng physicalSSDbytes, enginecold không đồng nghĩa OScachecold, cachebudget không bằng RSS.

## Kiểm RAM và giới hạn vận hành

Server không override dùng cache64MiB/layer cố định. Guard mặc định RSS=min(6GiB,RAM/2), headroom=min(1GiB,RAM/8); biến MOE_RAM_LIMIT_BYTES/MOE_RAM_HEADROOM_BYTES đặt byte dương. RunnerRAM dùng guard5GiB và watchdog5,5GiB/available1,5GiB/processswap128MiB tường minh trong protocol hiện tại.

Guard là kiểm hợp tác, không hard RSS cap; drop buffer không bảo đảm OS thu hồi trang ngay. Status khi model bận có thể là snapshot cũ, nên phân tích peak dùng sampler/HWM, ghi rõ thời điểm. Hoàn tất một cửa sổ hữu hạn không chứng minh ổn định dài hạn hoặc không leak.

Chỉ tái lập phase1-v1 khi được giao rõ mới dùng revision lịch sử d4adefd9e53627bdc1fd250eabab436c3ac54b0d. Không quay engine mới về revision cũ để chạy mặc định hoặc ghép các phiên bản thành chuỗi tăng tốc.
