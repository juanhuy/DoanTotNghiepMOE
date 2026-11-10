# Workflow dùng chung

## Thay đổi engine và kiểm chứng

1. Đọc AGENTS, PLAN/PROGRESS và module liên quan trong ARCHITECTURE. Xác định trigger, hành vi mong muốn, phạm vi sửa và invariant.
2. Với lỗi logic, dùng hồi quy fixture; không sửa expected/tolerance để che regression. Với tối ưu, chốt câu hỏi và điều kiện đối chứng trước.
3. Chạy `python3 tools/check.py --rust`; thêm release/HTTP/web/converter/CUDA theo thay đổi. Hoàn tất checks/build trước benchmark.
4. Ghi PROGRESS với thay đổi, lệnh, pass/fail/skipped, bằng chứng và giới hạn. Review diff trước commit; PR mô tả vấn đề, hành vi và kiểm chứng.

Một owner tích hợp. Hai người dùng branch/worktree riêng; không sửa đồng thời cùng file hoặc chạy benchmark song song trên cùng máy. Agent đọc AGENTS và hai skill trong .agents/skills/: moe-engine cho Rust/cache/FFI, moe-benchmark cho đo và diễn giải bằng chứng. Không cần MCP/plugin/tài khoản API để build engine. Không đưa token hoặc cấu hình cá nhân vào repo.

## Tiếp tục ở chat mới trên cùng workspace i5

Điểm bàn giao duy nhất là [PLAN hiện hành](PLAN.md), gồm sáu mục tiêu, S0–S6, điều kiện kiểm chứng và prompt sao chép. S0–S5 đã có bằng chứng ngày10/11; tiếp tục đúng bước PLAN ghi là đang mở, không lặp các phép đã chốt. Không đọc toàn bộ raw/PROGRESS hoặc tự chạy tất cả nhánh; protocol mới phải chốt trước đo.

Code/tools/bằng chứng ngày10/11 được commit vào main; nhánh work/i5-gate-up-ab giữ lại. Kiểm git status/HEAD khi bắt đầu, giữ mọi thay đổi phát sinh mới. Trạng thái push xem PROGRESS; máy i7/checkout khác chỉ có phần mới sau khi owner i5 publish. Các lệnh switch/pull ở mục i7 bên dưới chỉ áp dụng khi tiếp nhận revision thực đã publish trên checkout sạch, không dùng để bỏ thay đổi local.

Kernel nhiều token là CPU/INT8 opt-in/off mặc định; TTFT không đạt ở prompt ngắn09/11, đạt ở prompt64/128token10/11. Gate–Up/heat-admission đã khép; điều tra allocator đã có bằng chứng. Trạng thái và điều kiện tiếp tục ở PLAN, không tạo bản status thứ hai.

## Tiếp nhận trên testbed i7

### Nhận gốc phát triển chung sau hợp nhất

Gốc chung là **origin/main**, chứa lịch sử bản i5 và nhánh Huy. Branch handoff/i5-memory-20261108 giữ snapshot trước tích hợp; không dùng branch đó làm nền mới mặc định. Commit chung chính xác được owner thông báo sau push.

Trong checkout có workingtree sạch:

```bash
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
git rev-parse HEAD
```

Nếu chưa có main local, tạo trackingbranch từorigin/main. Không reset/clean để xóa thay đổi. Sau khi hai máy xác nhận cùng commit, mỗi người tạo branch riêng từmain để phát triển, ví dụ work/i5-next và work/i7-next; tránh sửa/push cùng main đồng thời.

Nhánh Huy cũ đã được tích hợp, không bị force-push. Nếu đồng nghiệp có commit cục bộ mới, giữ chúng trên branch đó rồi merge/rebase theo kế hoạch, không ép main ghi đè. Nếu Huy local vẫn ở tip đã merge và không có thay đổi, có thể cập nhật bằng gitmerge --ff-only origin/main, nhưng main vẫn là điểm đồng bộ chung.

### Đọc đúng kết quả và phạm vi

1. Đọc [SYSTEM_GUIDE](SYSTEM_GUIDE.md), [PLAN](PLAN.md), các mục mới nhất trong [PROGRESS](PROGRESS.md), [benchmark guide](../benchmarks/README.md), [machines](../benchmarks/machines.json) và skill moe-benchmark.
2. Bản hợp nhất có CPU trước load, lookahead telemetry, memorytrace/allocator/TID, KVbufferreuse và các tính năng Huy: INT4/evaluation/routes/sampling/FP8/persistence. KVbufferreuse mặc định off; memorytrace off; lookahead mặc định on nhưng các protocol RAM hiện dùng off tường minh.
3. i5 đã có [phép allocator12request](../benchmarks/experiments/2026-11-08-allocator-i5/README.md): RSS tăng284,547MiB nhưng allocator indicator tăng0,234MiB; free tăng205,777MiB. Đây là bằng chứng giữ vùng đã giải phóng trong cửa sổ đó, chưa chứng minh hết leak/plateau/hardcap hoặc điều gì trên i7. **Không cần lặp cuộc điều tra allocator trên i7 như thể chưa làm.** Đây là số đo trước merge; chưa có benchmark7B/FP8/INT4 được nghiệm thu trên bản hợp nhất.
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

Chưa chốt cache chung i5/i7. Việc model thật đang giao cho i7 là [nhiệm vụ tầng VRAM expert](#nhiệm-vụ-i7-tầng-vram-expert-bậttắt-giao-10112026); ngoài ra chưa giao cuộc tối ưu riêng. Nếu sau này được giao cửa sổ RAM model thật, xem [RAM_BUDGET](RAM_BUDGET.md), dự toán/preflight và preview sau:

```bash
python3 tools/check_ram_stability.py --testbed i7 --gpu cpu --lookahead off \
  --memory-trace on --kv-buffer-reuse off --workload mixed --cache 64 \
  --context 512 --cycles 12 --max-tokens 64 --keep-cache --threads 4 \
  --ram-limit-mib 5120 --headroom-mib 1024 --stop-rss-mib 5632 \
  --min-available-mib 1536 --stop-swap-mib 128 --dry-run
```

Đây chỉ là preview, không yêu cầu chạy12request ngay. 4threads giữ cấu hình chẩn đoán, không được đo là tối ưu cho i7. Thu3mẫu5s trước tải; hoãn nếu pswpout tăng hoặc MemAvailable dưới1536MiB. Không nới guard/đổi OS để vượt lỗi, không chạy CPU và GPU cùng lúc. CPU đã chọn trước load không khởi tạo CUDA; nếu chọn auto phải ghi trạng thái GPU thực và ngân sách VRAM, không ghép với CPU như cùng điều kiện.

## Nhiệm vụ i7: tầng VRAM expert bật/tắt (giao 10/11/2026)

**Vì sao giao:** trên i5 (GTX1650 Max-Q 4GB), khi dense chạy GPU, tắt tầng VRAM expert nhanh hơn ở prefill ([P1](../benchmarks/experiments/2026-11-10-gpu-expert-tier-i5/README.md), TTFT on/off1,21–1,45) và chat/decode ([P1](../benchmarks/experiments/2026-11-10-gpu-tier-chat-i5/README.md), total1,39–1,87×, decode trung vị3,13→5,53tok/s). Nguyên nhân quan sát: pool VRAM i5 chỉ chứa298/1024 expert (~29%) nên xoay vòng, upload5–41GiB/request. i7 (RTX5060 Laptop ~7,5GB theo hồ sơ kỳ1) có thể chứa phần lớn expert, nên kết quả có thể **ngược lại** — đây là câu hỏi cần đo, không suy từ i5. Mặc định `MOE_GPU_EXPERT_TIER` (hiện on) chỉ được quyết định sau khi có kết quả i7. Đọc [ARCHITECTURE: công tắc tầng VRAM](ARCHITECTURE.md#công-tắc-tầng-vram-expert-moe_gpu_expert_tier-1011) trước khi chạy.

**1. Nhận đúng source.** Cần main từ commit 1d06fca trở đi (chứa protocol, preflight và bản sửa CRLF để clone sạch không bị coi là dirty); owner i5 thông báo commit đã publish. Sau `git pull`, `git status --short` phải rỗng trước khi đo. Trên checkout sạch làm theo mục "Nhận gốc phát triển chung" ở trên, rồi tạo branch riêng, ví dụ `git switch -c work/i7-gpu-tier`. Ghi `git rev-parse HEAD`, `git status --short` (phải rỗng), `rustc -V`, OS/kernel/WSL.

**2. Kiểm môi trường và GPU thật (trước mọi phép đo, không chạy song song):**

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
python3 tools/check.py --rust --release --web
cargo test --locked --lib gpu -- --nocapture --test-threads=1 2>&1 | grep -E "EXECUTED|SKIPPED|Available|Total VRAM|Dense Offloaded|test result"
cargo build --release --locked --bin moe-tier-engine && sha256sum target/release/moe-tier-engine
```

Phải thấy `Available: true`, `Dense Offloaded: true` và các dòng `EXECUTED ... CUDA`, gồm `gpu_expert_tier_off_keeps_dense_on_gpu_without_vram_uploads`. Nếu là SKIPPED thì CUDA chưa chạy: dừng, ghi lý do, không đo. Kiểm fixture server hai chế độ:

```bash
for m in on off; do python3 tools/check_ram_stability.py --testbed i7 --gpu auto --gpu-expert-tier $m \
  --lookahead off --memory-trace on --cache 1 --context 64 --cycles 2 --max-tokens 3 --keep-cache \
  --threads 2 --model-dir tests/fixtures/olmoe/unnormalized-mha --output-root /tmp/moe-tier-fixture-$m; done
```

Cả hai `completed`, text giống nhau, status `gpu_expert_tier_enabled` đúng chế độ.

**3. Checkpoint.** Dùng checkpoint INT8 merged trên máy i7 qua `--model-dir <đường-dẫn>`; không có file `.moe_usage` trong thư mục. Runner tự hash và lưu manifest. Hai phép dưới so sánh bật/tắt **trong nội bộ i7**, không cần cùng checkpoint với i5; vẫn ghi hash để đối chiếu.

**4. Hai phép đo, đúng thứ tự, mỗi phép một lượt.** Protocol giữ nguyên như i5 ([SCENARIOS](../benchmarks/SCENARIOS.md): `gpu-tier-long-v1`, `gpu-tier-chat-v1`). Harness cố định CPU4 threads/cache64/context512 để giữ protocol; 4 threads không phải cấu hình tối ưu của i7 và phải ghi rõ. Trước mỗi phép: đóng ứng dụng nặng, không build/test/inference khác, rồi:

```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv > /tmp/i7-nvidia-pre.csv
python3 tools/preflight_memory.py --output /tmp/i7-preflight.json --min-available-mib 5120
```

Chỉ chạy khi preflight in `"eligible": true` (exit 0); không đạt thì chờ hoặc ghi hoãn, không nới ngưỡng. Phép A — prefill (16 request, max1):

```bash
python3 tools/run_benchmark.py --testbed i7 --gpu auto --gpu-tier-model-ab \
  --expert-batch-workload long-prefill --threads 4 --cache 64 --context 512 \
  --max-tokens 1 --lookahead off --model-dir <đường-dẫn-checkpoint>
```

Phép B — chat nhiều lượt/decode (32 request, max64), sau preflight mới:

```bash
python3 tools/run_benchmark.py --testbed i7 --gpu auto --gpu-tier-model-ab \
  --expert-batch-workload mixed --threads 4 --cache 64 --context 512 \
  --max-tokens 64 --lookahead off --model-dir <đường-dẫn-checkpoint>
```

Chạy `--dry-run` trước mỗi lệnh để xem protocol. Harness tự dừng nếu text/usage lệch, GPU/dense không bật, tier off mà vẫn có VRAM traffic, hoặc watchdog. Lỗi thì giữ raw và báo, **không chạy lại để lấy số đẹp, không đổi ngưỡng/threads/cache**.

**5. Đọc và lưu kết quả.** Trong `result.json` của mỗi phép: `off_faster`/`on_faster`, `pairs[].timers` (ratio on/off), `pairs[].work` (expert_vram_loads, expert_vram_upload_bytes/ms, expert_file_loads, decode_tokens_per_second ở phép B), `gpu_status_start` và `vram_experts_resident` trong status (số slot pool VRAM thực trên i7). Chuyển nguyên thư mục run từ `benchmarks/runs/i7/` vào `benchmarks/experiments/<ngày>-gpu-tier-i7/` cùng preflight/nvidia-smi, tạo `CHECKSUMS.sha256` (`sha256sum`), viết README ngắn theo mẫu hồ sơ i5, thêm vào mục lục benchmark và ghi một mục PROGRESS trên branch i7 (lệnh, commit, binary hash, checkpoint hash, hardware, pass/fail, giới hạn). Không sửa hồ sơ i5, không so số tuyệt đối với i5 (khác GPU/CPU/checkpoint).

**6. Quyết định sau khi có kết quả (owner hai máy thống nhất):**

| Kết quả i7 | Quyết định |
| --- | --- |
| `off_faster` ở cả A và B | Đổi mặc định MOE_GPU_EXPERT_TIER sang off cho cả hai máy (sửa engine/runner/test/tài liệu, kiểm đầy đủ) |
| `on_faster` ở A và/hoặc B | Giữ mặc định on; ghi cấu hình riêng: i5 off, i7 on. Báo cáo nêu tầng VRAM có lợi phụ thuộc tỉ lệ pool/working set và PCIe |
| Không nhất quán | Giữ mặc định on; i5 vẫn khuyến nghị off; ghi i7 chưa kết luận |

**Prompt sao chép cho agent trên máy i7:**

> Đây là máy i7 (testbed i7). Đọc AGENTS.md, rồi làm đúng mục "Nhiệm vụ i7: tầng VRAM expert bật/tắt" trong docs/WORKFLOWS.md trên branch riêng work/i7-gpu-tier. Dùng `--testbed i7`, không kế thừa số i5. Kiểm GPU thật trước (EXECUTED, không SKIPPED); chạy tools/preflight_memory.py trước mỗi phép; chạy phép A (long-prefill, max1) rồi phép B (mixed, max64), mỗi phép đúng một lượt, không chạy gì song song, không rerun/đổi threads/cache/ngưỡng. Lưu raw vào benchmarks/experiments/<ngày>-gpu-tier-i7 với CHECKSUMS, README, mục lục và một mục PROGRESS; báo off_faster/on_faster của từng phép, số slot VRAM thực, upload VRAM và giới hạn. Không đổi mặc định MOE_GPU_EXPERT_TIER, không push main.

**Cấu hình khuyến nghị hiện tại cho i5 (chưa phải mặc định code):** `MOE_GPU_MODE=auto MOE_GPU_EXPERT_TIER=off`, cache64MiB/layer, CPU4, victim, batch prefill off. Áp dụng cho demo/vận hành i5 trong phạm vi đã đo; không áp sang i7 trước khi có kết quả trên.

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


## Phân công sau khi nhận chung source

Cập nhật10/11: i5 làm S6 theo PLAN; i7 làm [nhiệm vụ tầng VRAM expert](#nhiệm-vụ-i7-tầng-vram-expert-bậttắt-giao-10112026) trước, sau đó mới tới các việc dưới đây. Ghi chú cũ: i5 tiếp tục câu hỏi expertcache/I/O theoPLAN. i7 trước mắt xác nhậnchecks/fixture và đánh giá độc lập công cụ chất lượng/routing; chưa tự đổi cùng moduleolmoe.rs mà không thống nhất owner. Nhánh nghiên cứuINT4/FP8 dùng baseline/corpus/config riêng, không gộp số liệu khảo sát cũ thành kết luận bản merge đã nhanh hơn.

CI converter đã bao gồm cả testconverterINT8 vàINT4; local cần môi trường Python cótorch/numpy/safetensors để chạy --converter. Nếu thiếu, báo skipped, không gọi toànbộconverterPASS. Các trường API chưa thực thi đầy đủ được nêu trongARCHITECTURE.
