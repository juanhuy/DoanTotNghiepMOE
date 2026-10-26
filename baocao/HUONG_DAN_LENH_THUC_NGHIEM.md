# Sổ tay lệnh kiểm thử và thực nghiệm của báo cáo lần 1

Tài liệu này giúp đi từ **câu hỏi cần kiểm tra → lệnh → dữ liệu đầu ra → cách giải thích**. Đọc cùng [chương 5](BAO_CAO_DINH_KY_LAN_1.md#chương-5-thiết-kế-thực-nghiệm-và-kết-quả-bước-đầu). Các lệnh chạy từ thư mục gốc dự án, trừ khi có ghi khác.

Đây là tài liệu hồi cứu lập ngày 25/10/2026, không phải bản chép nguyên lịch sử terminal. Những lượt thực nghiệm được nêu dưới đây đã có hồ sơ; **đa số dòng lệnh được dựng lại từ cấu hình, mã công cụ và dữ liệu đã lưu**, vì hồ sơ không lưu đầy đủ argv của mọi lần chạy. Không gắn một dòng lệnh dựng lại với thời điểm chạy cũ như thể đã có shell transcript.

Chạy lại có thể tạo kết quả mới cùng phương pháp, **không bảo đảm số giây giống từng mẫu cũ**. Muốn kiểm chứng một con số trong báo cáo, trước hết đọc dữ liệu cũ; muốn đối chứng một thay đổi, còn phải khớp phiên bản mã, trọng số, đầu vào, ngân sách và điều kiện đo.

## 1. Bản đồ các phép kiểm có thật

| Phần báo cáo | Câu hỏi kiểm tra | Bằng chứng đã lưu | Lệnh/công cụ tương ứng |
| --- | --- | --- | --- |
| 5.2–5.4 | i5 vận hành với model và cấu hình đã chọn thế nào? | [Bộ i5](../benchmarks/experiments/2026-10-22-phase1-i5/README.md), 20 mẫu đo + 5 làm ấm | `run_scenarios.py`, mục 5 dưới đây |
| 5.10 | i7 vận hành với model và cấu hình của i7 thế nào? | [Bộ i7](../benchmarks/experiments/2026-10-23-phase1-i7/README.md), 20 mẫu đo + 5 làm ấm | Cùng công cụ, chọn nhãn/model thực tế i7 |
| 5.5 | Đổi inclusive/victim có lợi trong cấu hình khảo sát không? | [8 yêu cầu AB/BA](../benchmarks/experiments/2026-10-22-web-session-i5-081202/i5/20261022T092416.839618Z-web-paired-abba-443258fd6bdd4e2c84bde18575709b5e/result.json) | `run_benchmark.py --http-session --compare-cache-policies`, mục 6 |
| 5.6 | Dùng lại tiền tố KV giảm phần tính lại thế nào? | [10 mẫu = 5 cặp](../benchmarks/experiments/2026-10-22-prefill-i5/paired-session.json) | Rust test `benchmark_session_prefix_paired`, mục 7 |
| 5.7 | Tính nhóm token song song có nhanh hơn cách lần lượt không? | [18 mẫu = 9 cặp](../benchmarks/experiments/2026-10-22-prefill-i5/paired-prefill.json) | Rust test `benchmark_prefill_dispatch_paired`, mục 8 |
| 5.8 | RAM diễn biến ra sao, cấu hình quá dự toán có bị từ chối không? | [Hai lượt 12 yêu cầu và kiểm âm](../benchmarks/experiments/2026-10-22-ram-guard-i5/README.md) | `check_ram_stability.py`, mục 9 |
| 5.9 | Phép tính, giao thức, dừng/gửi lại và nội dung sơ bộ có vấn đề gì? | [Smoke model thật](../benchmarks/experiments/2026-10-22-stage1-validation/README.md), mã kiểm thử và [nhật ký](../docs/PROGRESS.md) | `check.py`, test CUDA, harness smoke và kiểm web thủ công, mục 3 và 10 |

Các hàng này không cùng một cấu hình. Ví dụ, bộ vận hành dùng cache 64 MiB/lớp, còn thí nghiệm policy và KV dùng 128 MiB/lớp. Không gộp chúng thành chuỗi trước/sau hoặc cộng phần trăm tăng tốc.

## 2. Trước khi dùng lệnh

Chọn đúng máy và model. Các biến sau là biến tiện dụng của terminal; tên nhãn không tự phát hiện phần cứng:

```bash
export BENCH_TESTBED=i5
export BENCH_MODEL_DIR=models/olmoe-1b-7b-int8
export MOE_RAM_LIMIT_BYTES=5368709120
export MOE_RAM_HEADROOM_BYTES=1073741824
```

- `BENCH_TESTBED`: máy Sơn là `i5`; máy Huy đặt `i7`.
- `BENCH_MODEL_DIR`: sửa thành thư mục model INT8 thực tế. Cùng tên thư mục không chứng minh cùng trọng số. Bộ cũ i5/i7 dùng hai checkpoint khác nhau và được báo cáo riêng.
- Hai biến `MOE_RAM_*` lần lượt đặt ngưỡng tiến trình 5 GiB và RAM cần chừa 1 GiB. Đây là ngưỡng kiểm tra hợp tác, không phải giới hạn cứng do hệ điều hành áp đặt.
- Một máy chỉ chạy một phép thử model tại một thời điểm. Dừng server web của mình trước khi dùng runner tự mở server; không chạy build/test khác trong lúc lấy số đo.

Kiểm tra phiên bản hiện tại trước khi đo:

```bash
git status --short
git rev-parse HEAD
```

Các runner thông thường từ chối cây làm việc chưa sạch. Trong tài liệu này, `--allow-dirty` đi cùng `--dry-run` để xem kế hoạch. Dùng nó cho phép đo thật chỉ tạo hồ sơ khảo sát có thay đổi chưa commit, không tạo đối chứng revision sạch. Không cần commit hoặc đo lại chỉ để đọc các kết quả đã có.

## 3. Lệnh kiểm đúng phần mềm, không phải benchmark model 7B

### 3.1. Công cụ và giao thức phía giao diện

```bash
python3 tools/check.py --web
```

Lệnh kiểm catalog bằng chứng, công cụ Python và bộ phân tích luồng SSE bằng Node.js. Nó kiểm các tình huống như UTF-8 bị chia mảnh, thiếu sự kiện kết thúc, lỗi giữa phản hồi và hủy. Nó **không mở Chrome, không kiểm bố cục web và không nạp model 7B**. Trên môi trường đã ghi, nhóm kiểm công cụ hiện có 40 Python tests và 7 Node tests; số lượng có thể thay đổi khi mã được bổ sung.

### 3.2. Bộ kiểm phần mềm rộng hơn

```bash
python3 tools/check.py --rust --release --http --web
```

Lệnh chạy tuần tự:

1. Các kiểm công cụ như trên.
2. Định dạng Rust, Clippy và kiểm thử Rust ở cấu hình debug/release.
3. Kiểm parser SSE nếu có Node.
4. Build server và chạy 30 vòng HTTP trên **model fixture nhỏ**, gồm các lượt hoàn tất và cố ý hủy.

Đợt kiểm trước bộ vận hành ghi 76 Rust tests đạt, 5 ignored ở mỗi profile; nhật ký trước đó ghi HTTP 20 hoàn tất/10 hủy. Đây là kết quả lịch sử trong PROGRESS, không phải số lượng được hứa cho mọi phiên bản sau này. `ignored` nghĩa là không chạy trong lệnh tổng; phải gọi tường minh nếu muốn thực hiện bài thử đó.

`check.py --http` đặt kết quả chi tiết trong thư mục tạm và dọn sau khi xong. Nếu muốn tự giữ file HTTP stress để học, dùng hai lệnh sau riêng sau khi đã dừng mọi phép đo:

```bash
cargo build --release --locked --bin moe-tier-engine
HTTP_TEST_DIR=$(mktemp -d /tmp/moe-http-fixture-XXXXXX)
python3 tools/stress_http.py \
  tests/fixtures/olmoe/unnormalized-mha \
  "$HTTP_TEST_DIR/stress.json" 30
```

Công cụ này tự mở server dùng fixture, ghi số lượt hoàn tất/hủy và mẫu RSS vào `stress.json`. RSS/tốc độ của fixture không đại diện cho model 7B. Việc lệnh vượt qua kiểm thử cũng không chứng minh mọi câu trả lời đều đúng về nội dung.

### 3.3. Kiểm riêng GPU

```bash
cargo test --release --locked --lib test_gpu_probe -- --nocapture --test-threads=1
cargo test --release --locked --lib test_gpu_pinned_buffer -- --nocapture --test-threads=1
cargo test --release --locked --lib undersized_pinned_staging_falls_back_and_preserves_logits -- --nocapture --test-threads=1
cargo test --release --locked --lib victim_cache_gpu_keeps_tiers_disjoint_and_matches_cpu_logits -- --nocapture --test-threads=1
```

Các bài này lần lượt kiểm nhận diện GPU, vùng RAM được ghim, đường thay thế khi vùng truyền quá nhỏ và tính nhất quán bộ đệm victim/đầu ra số học. Một số test có thể bỏ qua công việc CUDA khi không có GPU nhưng vẫn trả thành công. Cần đọc probe và dòng `EXECUTED` của bài tương ứng, không chỉ nhìn chữ `ok`. Các fixture dùng sai số đã quy định; chúng không so toàn bộ model 7B với một bản FP32 chuẩn.

## 4. Mở web để hiểu một yêu cầu đi qua hệ thống

Đây là lệnh tương ứng để mở demo với cache 64 MiB/lớp, không tự tạo bộ số liệu năm mẫu:

```bash
RAYON_NUM_THREADS=8 \
MOE_MODEL_DIR="$BENCH_MODEL_DIR" \
MOE_CACHE_BYTES_PER_LAYER=67108864 \
MOE_CONTEXT=512 \
MOE_BIND=127.0.0.1:8080 \
cargo run --release --locked --bin moe-tier-engine
```

Các biến bảo vệ RAM ở mục 2 được kế thừa. Xem địa chỉ máy chủ trong log; cấu hình mặc định hiện tại là `127.0.0.1:8080`. Khi server đang mở, ở terminal khác có thể kiểm:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/experts
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"olmoe","messages":[{"role":"user","content":"Where is Mars"}],"session_id":"learn-mars","max_tokens":32,"stream":true}'
```

- `/health` cho biết tiến trình phục vụ được yêu cầu, không chứng minh chất lượng model.
- `/experts` cho biết cấu hình và trạng thái; khi mô hình đang bận có thể trả bản ghi cũ.
- Lệnh POST đi cùng đường suy luận HTTP/SSE với web; `-N` giúp curl không giữ nội dung đợi đầy bộ đệm mới in ra.
- Ví dụ này giới hạn **32 token để quan sát**, khác giới hạn 128 của bộ vận hành. Không dùng thời gian ví dụ làm số đo của bảng chương 5.
- Gửi cùng `session_id` mà chỉ giữ một câu hỏi không tự tạo hội thoại hai lượt. Lượt sau phải gửi đúng lịch sử, gồm câu trả lời thực tế trước đó.

Đường xử lý: câu hỏi → HTTP → kiểm đầu vào và quyền dùng mô hình → mã hóa token → dùng lại KV nếu phù hợp → xử lý đầu vào → sinh tiếp → SSE trả nội dung và kết quả cuối. CPU tính expert; GPU hỗ trợ các phép chiếu và giữ một phần expert trong VRAM.

## 5. Bộ vận hành cold, warm, conversation của i5/i7

### 5.1. Xem kế hoạch trước, không chạy model

```bash
python3 tools/run_scenarios.py warm \
  --testbed "$BENCH_TESTBED" --gpu auto --threads 8 \
  --model-dir "$BENCH_MODEL_DIR" \
  --cache 64 --context 512 --max-tokens 128 --runs 5 \
  --single-policy --expert-cache-policy victim \
  --output-root benchmarks/runs/learning \
  --dry-run --allow-dirty
```

Đầu ra preview phải nêu 5 yêu cầu đo và 5 làm ấm. Đổi `warm` thành `cold` thì có 5 yêu cầu đo/0 làm ấm; thành `conversation` thì có 10 yêu cầu đo/0 làm ấm. Preview không nạp trọng số, không làm nóng GPU và không tạo số đo hiệu năng.

### 5.2. Lệnh tái lập bộ vận hành theo cấu hình đã lưu

Chỉ chạy khi thật sự muốn đo mới và cây làm việc đáp ứng điều kiện ghi nhận:

```bash
for scenario in cold warm conversation; do
  python3 tools/run_scenarios.py "$scenario" \
    --testbed "$BENCH_TESTBED" --gpu auto --threads 8 \
    --model-dir "$BENCH_MODEL_DIR" \
    --cache 64 --context 512 --max-tokens 128 --runs 5 \
    --single-policy --expert-cache-policy victim \
    --output-root benchmarks/runs/learning || break
done
```

Runner tự build và mở server riêng cho từng kịch bản; không cần mở server bằng cargo run trước. Nó giữ một server cho các vòng trong cùng kịch bản và dừng khi xong. `|| break` dừng vòng nếu một kịch bản lỗi.

| Lựa chọn | Ý nghĩa |
| --- | --- |
| `--gpu auto` | Trong runner HTTP này, yêu cầu GPU và kiểm GPU thực bật; không bật được thì từ chối phép đo |
| `--threads 8` | 8 luồng Rayon cho công việc CPU; không phải 8 yêu cầu cùng chạy |
| `--cache 64` | 64 MiB mỗi lớp, tổng ngân sách expert 1 GiB trên 16 lớp; không phải tổng RSS |
| `--context 512` | Tổng token đầu vào và giới hạn sinh phải nằm trong cửa sổ 512 |
| `--max-tokens 128` | Sinh tối đa 128 token; EOS có thể làm dừng sớm |
| `--runs 5` | 5 vòng; số request mỗi vòng phụ thuộc kịch bản |
| `--single-policy` | Chỉ dùng victim, không xen kẽ hai chính sách |
| `--output-root` | Tạo run mới riêng dưới nhãn máy; không ghi lại vào hồ sơ thực nghiệm đã chốt |

**Cold:** đặt lại bộ đệm ứng dụng trước mỗi vòng rồi hỏi `Where is Mars`. Không khởi động lại server mỗi mẫu và không xóa bộ đệm tệp của hệ điều hành.

**Warm:** đặt lại ở đầu vòng, hỏi Mars một lần làm ấm rồi đo lần hỏi Mars với phiên mới. Không đưa lần làm ấm vào năm mẫu đo; phiên mới giúp tách lợi ích expert cache khỏi KV hội thoại.

**Conversation:** đặt lại đầu vòng, hỏi `Where is Mars`, sau đó `What is love ?` với cùng phiên và đầy đủ câu trả lời Mars. Cả hai lượt được đo. Câu Mars của bộ này không có dấu hỏi ở cuối.

Một bộ có **20 yêu cầu đo + 5 làm ấm**, chia bốn nhóm có năm mẫu: cold Mars, warm Mars, conversation Mars, conversation Love. Các hồ sơ cũ i5/i7 là hai cấu hình/model khác nhau; chạy cùng công cụ không biến chúng thành phép so riêng sức mạnh CPU.

Chương 5 dùng mã K1 cho cold, K2 cho warm, K3.1 cho câu hỏi đầu và K3.2 cho câu hỏi tiếp của conversation; mỗi mã đều có trình tự và câu hỏi nguyên văn ở mục 5.3.2. Các tên nhóm trong JSON lưu trữ được giữ nguyên.

### 5.3. Đọc các file mà runner tạo

| File | Dữ liệu và cách dùng |
| --- | --- |
| `result.json` | Cấu hình, model/hardware/hash, trạng thái và tập request; xem điều kiện trước khi đọc số |
| `turn-0-0.json` | Yêu cầu đo đầu vòng 0; chứa lịch sử thực gửi, kết quả, metric và quan sát tiến trình |
| `turn-0-1.json` | Yêu cầu thứ hai, nếu kịch bản có; ở conversation là Love |
| `warmup-0-0.json` | Lượt làm ấm, không trộn vào thống kê chính |
| `memory.jsonl` | Mỗi dòng là một mẫu bộ nhớ; không phải mỗi dòng ứng với một token |
| `watchdog.json` | Ngưỡng giám sát và vi phạm; rỗng chỉ nghĩa là không phát hiện vi phạm trong lượt đó |
| `build.log`, `server.log` | Lỗi build/nạp model, cấu hình và diễn biến phía server |

Các bộ đã lưu còn có `summary.json` được tổng hợp sau đo. Runner không có nghĩa tự sinh đúng bản summary báo cáo đó. Script `build_report.py` trong artifact là mã tổng hợp đã lưu; **không chạy ngay tại artifact bất biến**, vì có thể ghi lại summary/README và làm sai checksum. Có thể tính lại trực tiếp bằng lệnh đọc ở mục 11.

## 6. Đối chứng inclusive/victim, không phải bộ vận hành chính

Lệnh dựng lại phương pháp AB/BA của mục 5.5:

```bash
python3 tools/run_benchmark.py \
  --http-session --http-scenario conversation \
  --testbed "$BENCH_TESTBED" --gpu auto --threads 8 \
  --model-dir "$BENCH_MODEL_DIR" \
  --cache 128 --context 512 --max-tokens 128 --runs 2 \
  --compare-cache-policies --expert-cache-policy inclusive \
  --tag policy-abba --output-root benchmarks/runs/learning
```

`--runs 2` ở chế độ so policy tạo hai cặp thứ tự AB rồi BA, tức bốn vòng: inclusive, victim, victim, inclusive. Mỗi vòng có Mars/Love nên tổng **8 request**, mỗi policy/prompt chỉ có hai mẫu.

Có thể thêm `--dry-run --allow-dirty` để xem bốn policy mà không nạp model. Bộ cũ không lưu đầy đủ biến RAM như record đời sau; ngưỡng RAM hiện kế thừa mục 2 là lựa chọn cho lần đo mới, không được gán ngược thành cấu hình đã xác minh của lượt cũ.

Câu hỏi cần trả lời là đổi nơi giữ expert có thay số lần đọc tệp, truyền VRAM/RAM và tổng thời gian không. Không chỉ nhìn cache hit: victim có thể giảm đọc tệp nhưng tăng truyền từ GPU về RAM. Số giảm 20,77%/13,42% trong báo cáo thuộc hai trung vị của khảo sát nhỏ đã lưu, không phải cam kết cho lần chạy sau.

## 7. Đối chứng tái sử dụng KV: năm cặp trong một tiến trình

Đây là bài Rust bị đánh dấu ignored, phải gọi tường minh. Dừng server web trước; bài thử tự nạp model. Lệnh tương ứng với harness còn trong mã hiện tại:

```bash
PAIR_DIR=$(mktemp -d /tmp/moe-paired-XXXXXX)
RAYON_NUM_THREADS=4 \
MOE_PREFILL_TESTBED="$BENCH_TESTBED" \
MOE_PREFILL_MODEL="$BENCH_MODEL_DIR" \
MOE_PREFILL_REPORT="$PAIR_DIR/session.json" \
cargo test --release --locked --lib benchmark_session_prefix_paired \
  -- --ignored --nocapture --test-threads=1
```

- Bài thử dựng đoạn ngữ cảnh lặp về Mars, tạo KV cho câu hỏi cuối ban đầu rồi thay câu hỏi cuối thành Love.
- Hai phương án chạy cùng model/tiến trình: cách tham chiếu không dùng lại KV cho đầu vào sửa trong phép thử, và cách lấy tiền tố chung hợp lệ.
- Việc chuẩn bị trạng thái KV được thực hiện ngoài khoảng đo. Mỗi phương án được làm ấm rồi có năm cặp xen kẽ.
- Cache 128 MiB/lớp, context 512, giới hạn sinh 1 token được đặt trong mã test; không đổi bằng `--cache` của runner HTTP.
- GPU tự thăm dò khi nạp; đọc `gpu_start`/`gpu_end` để biết thực tế, không suy GPU bật từ tên test.
- File đầu ra dùng cơ chế tạo mới, từ chối ghi đè file đã tồn tại.

Bộ lịch sử có 116 token đầu vào, dùng lại 104 token ở phương án mới, còn 12 token phải xử lý; token đầu khớp ở cả năm cặp. Thời gian prefill giảm 67,56% khi tính từ hai trung vị. **Đây không phải tốc độ sinh 128 token tăng 67,56%.**

Bài Rust này kế thừa bảo vệ RAM của engine nhưng không có watchdog ngoài như runner HTTP. Nó dùng cache lớn hơn bộ vận hành chính. Không cần chạy chỉ để xem lại kết quả; muốn tái đo cần kiểm riêng ngân sách và không nạp hai model.

## 8. Đối chứng phân công token: chín cặp

Dùng một thư mục đầu ra mới:

```bash
PAIR_DIR=$(mktemp -d /tmp/moe-paired-XXXXXX)
RAYON_NUM_THREADS=4 \
MOE_PREFILL_TESTBED="$BENCH_TESTBED" \
MOE_PREFILL_MODEL="$BENCH_MODEL_DIR" \
MOE_PREFILL_REPORT="$PAIR_DIR/dispatch.json" \
cargo test --release --locked --lib benchmark_prefill_dispatch_paired \
  -- --ignored --nocapture --test-threads=1
```

Bài này dùng câu **`Where is Mars ?`**, có dấu hỏi và khoảng trắng, khác chuỗi Mars của bộ vận hành. Nó làm ấm bốn lượt, rồi đo chín cặp cách lần lượt/theo nhóm với thứ tự xen kẽ. Cache 128 MiB/lớp, context 512, sinh tối đa một token; mô hình chỉ nạp một lần.

Trong file, `before` tương ứng cách lần lượt; `after` là phương án theo nhóm. Không được hiểu after tự động là phương án tốt hơn. Dữ liệu cũ cho thấy theo nhóm chậm hơn 7/9 cặp; trung vị thay đổi từng cặp tăng 4,55%, nên phương án này không được bật trong đường chạy chính. Hạn chế RAM/GPU và cách ghi file tương tự mục 7.

## 9. Kiểm RAM có giám sát và cấu hình bị từ chối

Các dòng sau dựng lại cấu hình của ba hồ sơ RAM. Runner tự mở server, đọc bộ nhớ mỗi 250 ms và chỉ quản lý tiến trình do nó tạo.

### 9.1. Mười hai yêu cầu, tối đa 32 token, có đặt lại bộ đệm

```bash
python3 tools/check_ram_stability.py \
  --testbed "$BENCH_TESTBED" --gpu auto --model-dir "$BENCH_MODEL_DIR" \
  --threads 8 --cache 64 --context 512 --cycles 12 --max-tokens 32 \
  --ram-limit-mib 5120 --headroom-mib 1024 \
  --stop-rss-mib 5632 --min-available-mib 1536 --stop-swap-mib 128 \
  --output-root benchmarks/runs/learning
```

`--cycles 12` ở công cụ này là 12 request, khác `--runs` của wrapper kịch bản. Runner dùng một chu kỳ bốn câu hỏi; có lượt tiếp nối dùng history. Đặt lại trước các pha 0/3, không phải reset trước cả 12 request. Xem các `turn-*.json` để biết đúng câu hỏi và lịch sử từng lượt.

### 9.2. Mười hai yêu cầu, tối đa 128 token, giữ bộ đệm

```bash
python3 tools/check_ram_stability.py \
  --testbed "$BENCH_TESTBED" --gpu auto --model-dir "$BENCH_MODEL_DIR" \
  --threads 8 --cache 64 --context 512 --cycles 12 --max-tokens 128 \
  --keep-cache \
  --ram-limit-mib 5120 --headroom-mib 1024 \
  --stop-rss-mib 5632 --min-available-mib 1536 --stop-swap-mib 128 \
  --output-root benchmarks/runs/learning
```

`--keep-cache` chỉ đặt lại lúc đầu. Một số câu có thể dừng EOS trước 128 token. Bộ cũ quan sát khoảng 212,1 giây/848 mẫu và 488,7 giây/1952 mẫu; không dùng các khoảng này làm bằng chứng chạy ổn định nhiều giờ hoặc khẳng định không rò rỉ bộ nhớ.

### 9.3. Kiểm âm ngân sách 256 MiB/lớp

```bash
python3 tools/check_ram_stability.py \
  --testbed "$BENCH_TESTBED" --gpu auto --model-dir "$BENCH_MODEL_DIR" \
  --threads 8 --cache 256 --context 512 --cycles 1 --max-tokens 1 \
  --ram-limit-mib 5120 --headroom-mib 1024 \
  --stop-rss-mib 5632 --min-available-mib 1536 --stop-swap-mib 128 \
  --output-root benchmarks/runs/learning
```

Đây là **cấu hình để kiểm sự từ chối**, không phải đề nghị tăng cache khi dùng bình thường. Bộ cũ ghi dự toán 7.288.299.520 byte vượt ngưỡng 5 GiB, server dừng trước khi nạp trọng số lớn. Record giữ status failed và không có request suy luận hoàn tất; đó là kết quả mong đợi. Không đổi failed thành success để làm đẹp hồ sơ. Giữ watchdog khi tái hiện, không chạy server cache 256 đơn độc để cố gây OOM.

Bộ cũ là phép khảo sát trên cây mã chưa sạch. Lệnh trên cố ý không tự thêm `--allow-dirty` cho một lần đo mới. Preview có thể thêm `--dry-run --allow-dirty`; nếu chủ động ghi khảo sát dirty thì phải công bố rõ và giữ mã tương ứng.

## 10. Smoke chất lượng, Stop và kiểm web thủ công

Hồ sơ smoke lưu [harness thực tế](../benchmarks/experiments/2026-10-22-stage1-validation/model-smoke/smoke_runner.py), các JSON từng trường hợp và source snapshot. Đường vào của harness lưu trữ là:

```bash
python3 benchmarks/experiments/2026-10-22-stage1-validation/model-smoke/smoke_runner.py
```

Đây là **harness lịch sử để đọc**, không phải runner chuẩn khuyên chạy lại nguyên trạng: nó gắn cứng nhãn i5, thư mục model và cache 128 MiB/lớp, không có watchdog độc lập như bộ RAM mới. Nó dùng binary release đã build sẵn, kiểm Mars→Love, hủy stream rồi reset/gửi lại, đầu vào dài và 12 câu kiểm nội dung. Nếu muốn thực nghiệm mới, cần chuẩn bị lại điều kiện giám sát phù hợp thay vì chỉ sao chép lệnh này.

Hồ sơ ghi 16 request hoàn tất, thêm một request cố ý hủy. Trong 12 câu nội dung, tiêu chí exact/substring đạt 10, trượt 2. Đây là kiểm sơ bộ bằng quy tắc, không phải độ chính xác tổng quát 83,33% và không chứng minh lượng tử hóa gây hai lỗi.

Có sai lệch đã công bố: aggregate `result.json` của smoke bị sửa metadata history của case Mars do dùng chung danh sách trong harness; [mars.json](../benchmarks/experiments/2026-10-22-stage1-validation/model-smoke/mars.json) giữ đúng đầu vào tại lượt đó. Khi tái dựng, ưu tiên file từng lượt. Không sửa raw để che sai lệch.

Bảy thao tác web trong báo cáo là **người dùng kiểm thủ công**, không có câu lệnh terminal nào tự chứng minh chúng: chat, hỏi theo history, Stop, gửi lại, lỗi quá context, reset và reload. Node tests kiểm parser, còn HTTP fixture kiểm giao thức; hai loại này không thay cho quan sát trình duyệt.

## 11. Xem lại số liệu đã có, không chạy model

### 11.1. In năm mẫu warm và tính trung vị

Lệnh sau chỉ đọc JSON đã lưu, không ghi file hoặc gửi request:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from statistics import median

folder = Path('benchmarks/experiments/2026-10-22-phase1-i5/warm')
rows = []
for path in sorted(folder.glob('turn-*-0.json')):
    record = json.loads(path.read_text())
    response = record['response']
    metric = response['metrics']
    row = (record['cycle'] + 1,
           metric['time_to_first_token_ms'] / 1000,
           metric['decode_tokens_per_second'],
           metric['total_ms'] / 1000)
    rows.append(row)
    print('Lượt %d: TTFT %.3fs; decode %.3f steps/s; WALL %.3fs' % row)
assert len(rows) == 5
print('Trung vị: TTFT %.3fs; decode %.3f steps/s; WALL %.3fs' %
      tuple(median(row[column] for row in rows) for column in (1, 2, 3)))
PY
```

Trung vị phải hiển thị khoảng **2,023 s; 3,987 steps/s; 33,899 s**. Đây là biểu diễn lại dữ liệu, không phải phép đo mới. Đổi đường dẫn i5 thành `2026-10-23-phase1-i7/warm` để xem bộ i7; không đổi nhãn của số đo cũ.

Khi đọc một request:

1. Xem `messages`: câu hỏi, lịch sử có đúng không?
2. Xem `usage`: số token đầu vào/đầu ra thực tế.
3. Xem số token tái sử dụng KV, lý do dừng và văn bản trả lời.
4. Đọc TTFT, prefill, decode, tổng thời gian; không cộng các timer thành phần chồng lấp để suy tổng.
5. Đọc dữ liệu tệp/VRAM và bộ nhớ; không đồng nhất payload với SSD vật lý, cache budget với RSS hoặc VRAM logic với tổng cấp phát GPU.

### 11.2. Vì sao đọc nhiều GB dù model chỉ có một bộ trọng số?

Lượng dữ liệu đọc tệp là tổng của nhiều lần nạp, không phải kích thước các tệp khác nhau. Một expert có thể bị đẩy khỏi cache rồi phải nạp lại; cùng một phần trọng số vì thế được tính nhiều lần. Bộ đệm tệp của hệ điều hành còn có thể phục vụ các lần đọc đó mà SSD không phải đọc lại tương ứng.

Lấy mẫu warm Mars đầu tiên để nối các con số:

- 19 token đầu vào, không dùng lại KV; 128 token đầu ra cần 127 bước model sau prefill.
- Mỗi token được xử lý qua 16 lớp, mỗi lớp chọn 8 expert. Tổng số lượt dùng expert logic là `(19 + 127) × 16 × 8 = 18.688`.
- Bộ đếm ghi 13.842 hit và 4.846 miss, tổng đúng 18.688; tỷ lệ hit khoảng 74,07%.
- Đây là lượt dùng logic. Một expert nạp một lần trong prefill có thể phục vụ nhiều token; số lần nạp tệp có thể khác số miss. Nạp trước chưa dùng còn có thể góp vào lượng đọc tệp.
- Vì vậy, hit rate, lượng đọc tệp, thời gian chờ expert và SSD thực đọc phải được xem riêng. Hit trong VRAM vẫn có chi phí truyền về RAM để CPU tính.

### 11.3. Kiểm checksum hồ sơ

```bash
(cd benchmarks/experiments/2026-10-22-phase1-i5 && sha256sum -c CHECKSUMS.sha256)
(cd benchmarks/experiments/2026-10-23-phase1-i7 && sha256sum -c CHECKSUMS.sha256)
(cd benchmarks/experiments/2026-10-22-ram-guard-i5 && sha256sum -c CHECKSUMS.sha256)
(cd benchmarks/experiments/2026-10-22-stage1-validation && sha256sum -c CHECKSUMS.sha256)
```

Các lệnh chỉ kiểm tính toàn vẹn: tệp còn đúng byte đã khóa hay không. Chúng không xác nhận phương pháp đo đúng, không chứng minh mọi phép tính đúng và không xác minh model nguồn chỉ từ tên file.

Bộ prefill dùng `checksums.json` thay vì định dạng sha256sum:

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
folder = Path('benchmarks/experiments/2026-10-22-prefill-i5')
items = json.loads((folder / 'checksums.json').read_text())
for name, expected in items.items():
    actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    assert actual == expected, name
print('PASS:', len(items), 'tệp prefill khớp checksum')
PY
```

## 12. Tự giải thích trước khi gọi một kết quả là bằng chứng

Với mỗi phép thử, thử trả lời sáu câu:

1. **Đang hỏi điều gì?** Vận hành, đúng phép tính, chất lượng nội dung hay hiệu quả một thay đổi?
2. **Đầu vào thật là gì?** Model nào, chuỗi câu hỏi nào, có history không, bao nhiêu token?
3. **Đang đổi gì và giữ gì?** Policy, cách dùng KV hay phân công luồng? Có đổi cache hoặc model cùng lúc không?
4. **Số được đo ở đâu?** Bên trong engine, phía HTTP, hệ điều hành hay kiểm bằng quy tắc nội dung?
5. **Có bao nhiêu mẫu?** Warmup có bị tính lẫn không; dùng trung vị gì; đơn vị gì?
6. **Kết luận tới đâu?** Một cấu hình, một đầu vào, một khoảng quan sát; chưa đo được điều gì?

Nếu giải thích được sáu câu bằng một request và một bảng tổng hợp, bạn đã nắm được cơ sở của phép thử đó. Để học sâu từng khối tính toán, xem thêm [lộ trình học](../docs/research/LEARNING_PLAN.md); tài liệu lệnh không thay thế toàn bộ lý thuyết Transformer/MoE hoặc việc đọc từng hạt nhân CPU/CUDA.

## 13. Kết quả rà soát chương 5 ngày 25/10/2026

Rà soát chỉ đọc hồ sơ, không nạp model hoặc tạo số đo mới:

- 201 tệp trong bốn danh sách SHA-256 và danh sách JSON prefill khớp checksum.
- 40 dòng request của hai máy khớp JSON từng lượt và các tổng hợp tương ứng.
- 264 phép đối chiếu số ở các bảng request, trung vị, khoảng dao động, chi phí, bộ nhớ và cặp thử đạt trong mức làm tròn được trình bày.
- Đối chiếu số mẫu policy, RAM và quality với hồ sơ gốc; giữ các trường hợp lỗi và giới hạn như đã công bố.
- Kiểm số liệu phù hợp không xác nhận đầy đủ shell history, bộ câu hỏi có tính đại diện hay mọi trường hợp hệ thống đều đúng. Hồ sơ kiểm thủ công và các lần kiểm phần mềm cũ có mức lưu trữ khác với raw benchmark.

Lộ trình đọc ngắn: mục 4 để hiểu một request → mục 11 để hiểu một số đo → mục 5 để hiểu bộ vận hành → mục 6–9 để hiểu phép đối chứng và RAM. Chưa cần chạy lại tất cả các lệnh.
