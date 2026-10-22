# Runner kịch bản HTTP/SSE

Các preset gọi `tools/run_benchmark.py`, tự mở server riêng và đo tuần tự. Mặc định 5 mẫu **mỗi policy**, 128 output tokens tối đa, context512/cache128MiB mỗi layer; so sánh `inclusive` với `victim` theo thứ tự AB/BA. Đây là so sánh policy trong cùng binary, không phải A/B hai revision của code.

| Preset | Trạng thái trước đo | Câu hỏi cần kiểm chứng | Requests mặc định, gồm warmup |
| --- | --- | --- | --- |
| `cold` | Reset expert/KV trước mỗi mẫu, một chat mới | TTFT/wall khi cache ứng dụng rỗng | 10 |
| `warm` | Reset, chạy Mars một lần warmup, rồi Mars trong chat/session mới | Expert cache ấm giúp bao nhiêu khi không reuse KV? | 20, trong đó 10 warmup |
| `conversation` | Reset trước Mars, Love dùng cùng history/session | Hiệu năng đúng chuỗi web Mars→Love, prefix reuse thực tế | 20 |
| `topics` | Reset trước chuỗi Mars→love→Python→Rome; mỗi câu chat/session riêng, giữ expert cache | Cache có ích khi đổi chủ đề, không trộn chi phí history dài? | 40 |
| `long-context` | Reset, một research note cố định rồi yêu cầu tóm tắt | Prefill/attention với input dài hơn | 10 |
| `cache-sweep` | Chạy `conversation` lần lượt cache128/192/256MiB mỗi layer | Đổi ngân sách RAM có giảm file load/wall, đổi lại RSS/swap thế nào? | 60 |

`cold` nghĩa **engine-cache-cold**, không phải OS/SSD-cold hay khởi động process mới. Checkpoint hashing đọc weights trước phép đo. `warm` có đúng một warmup mỗi mẫu, không có nghĩa cache đã đạt steady state. `topics` chủ ý tách session để tránh tăng context. `long-context` là prompt cố định dài hơn các prompt còn lại, không phải near-context benchmark chuẩn hóa theo token; xem `usage.prompt_tokens` thực tế. Input + output budget phải vừa context; runner giữ lỗi API nếu không vừa.

## Bộ ghi báo cáo giai đoạn1: protocol phase1-v1

Cấu hình ứng viên đã qua kiểm RAM i5: victim,64MiB cache/layer, context512,8 Rayon threads, tối đa128 output tokens/request, RAM guard5GiB/system headroom1GiB. Chạy cùng revision/checkpoint và bộ này trên i5/i7 để ghi kết quả riêng, không xếp hạng CPU. i7 chỉ đo sau preflight RAM/GPU tại máy đó; thiếu GPU phải ghi thành backend CPU khác, không âm thầm fallback.

| Kịch bản | Mẫu đo | Warmup | Trạng thái |
| --- | --- | --- | --- |
| cold | 5 Mars | 0 | Mỗi mẫu reset ứng dụng, session mới |
| warm | 5 Mars | 5 Mars | Mỗi cycle reset→1 warmup→1 mẫu, hai session độc lập; không reuse KV |
| conversation | 5 Mars +5 Love | 0 | Reset trước Mars, Love giữ history/session |

Tổng20 request đo +5 warmup, mỗi nhóm kết quả5 mẫu; warmup không tính trung vị. Không bắt chạy thêm topics/long-context/cache-sweep cho bộ báo cáo này. Single-policy victim, không AB/BA. Chưa gọi là baseline nghiệm thu trước khi source sạch, config/checkpoint được đóng và run hoàn tất; ngưỡng RAM khác hoặc cache khác là protocol khác phải ghi rõ.

Preview từ root (trên dirty tree chỉ preview thêm --allow-dirty):

```bash
for scenario in cold warm conversation; do
  MOE_RAM_LIMIT_BYTES=5368709120 MOE_RAM_HEADROOM_BYTES=1073741824 \
  python3 tools/run_scenarios.py "$scenario" --testbed i5 --gpu auto \
    --threads 8 --cache 64 --context 512 --max-tokens 128 --runs 5 \
    --single-policy --expert-cache-policy victim --allow-dirty --dry-run || break
done
```

Đo thật sau checks/commit nguồn sạch: bỏ **cả --allow-dirty và --dry-run**, dừng inference khác; i7 đổi testbed, giữ settings đã chốt, ghi hardware thực. Không sửa source/checkpoint/config trong suốt bộ đo, không build/tests/inference khác đồng thời. Runner build trước từng phép đo, timing chỉ sau load; watchdog chạy cùng protocol trên cả hai máy.

HTTP runner giữ memory.jsonl250ms/watchdog.json, dừng child riêng khi RSS>5,5GiB/MemAvailable<1,5GiB/VmSwap>128MiB hoặc sampler lỗi. Limit/headroom engine được giữ trong record cùng status thực; không gọi là hard RSS cap. Run lỗi/negative giữ status gốc, không được đưa vào bảng completed. Các record cuối phải cùng commit, git_dirty=false, checkpoint/source/binary hashes và cache/threads/context/token/guard/policy đúng. Nếu có source drift, dừng bộ đo thay vì trộn.

Bảng báo cáo chỉ cần4 dòng: cold Mars, warm Mars, conversation Mars, conversation Love; mỗi dòng có từng mẫu và median/min/max TTFT, decode, wall, cache/file/VRAM counters, peak RSS/HWM/swap từ sampler. Kiểm output/usage/finish reason của các lượt lặp; không gọi text parity là logits7B parity. Không gộp Mars/Love thành một tốc độ; không suy speedup so với các lượt cũ khác config. User review một README tổng hợp cuối; JSON/log/source/checksum là hồ sơ audit, không tạo thêm status/handoff file trùng.

## Preview và chạy

Từ thư mục gốc, preview không build, hash checkpoint hoặc khởi động model:

```bash
python3 tools/run_scenarios.py conversation \
  --testbed i5 --gpu auto --threads 8 --runs 5 \
  --allow-dirty --dry-run
```

Đổi `conversation` thành preset trong bảng. Bỏ `--dry-run` để đo. Hoàn tất build/tests và dừng inference/benchmark khác trước khi chạy. `--gpu auto` trong mode HTTP từ chối fallback CPU. Threads8 giữ cấu hình thí nghiệm i5 trước đó, chưa chứng minh là tối ưu. `--allow-dirty` chỉ dành cho exploratory; bỏ flag trên revision sạch để lập baseline.

```bash
# Một policy để kiểm tra runtime hiện hành; không tạo so sánh speedup:
python3 tools/run_scenarios.py warm --testbed i5 --gpu auto \
  --threads 8 --runs 5 --single-policy --expert-cache-policy victim --dry-run

# Quét ngân sách RAM tuần tự; dừng ngay nếu một run lỗi:
python3 tools/run_scenarios.py cache-sweep --testbed i5 --gpu auto \
  --threads 8 --runs 5 --allow-dirty --dry-run

# Kiểm fixture nhỏ, kiểm protocol, không đánh giá tốc độ 7B:
python3 tools/run_scenarios.py conversation --testbed i5 --gpu cpu \
  --model-dir tests/fixtures/olmoe/unnormalized-mha \
  --threads 2 --cache 1 --context 64 --max-tokens 3 --runs 2 \
  --allow-dirty
```

`--runs` là số mẫu mỗi policy, không phải số request; `--max-tokens`, `--context`, `--model-dir`, `--output-root` có thể đổi. `cache-sweep` cố định ba mức cache trong bảng. Mỗi mức là process riêng, thứ tự ngân sách tăng dần nên vẫn có confound OS cache/nhiệt; muốn kết luận về ngân sách cần đo lại đảo thứ tự bằng entry point chung:

```bash
python3 tools/run_benchmark.py --http-session --http-scenario conversation \
  --compare-cache-policies --testbed i5 --gpu auto --threads 8 \
  --cache 256 --context 512 --max-tokens 128 --runs 5 --dry-run
```

## Đọc kết quả

Mỗi run ID giữ `result.json`, server/build log, checkpoint/source/binary hashes, spec workload và hash spec. `result.runs` chỉ chứa mẫu đo; `result.warmups` giữ warmup riêng. Từng request lưu file ngay khi hoàn tất, gồm history thực gửi và session ID; request lỗi không biến thành mẫu thành công. Không overwrite dữ liệu cũ. Fixture/raw tạm dùng để kiểm runner không đưa vào corpus luận văn.

So sánh theo **cùng preset, turn/prompt, budget và cấu hình**. Trước tỷ lệ cải thiện, xem từng mẫu và phân tán của wall, TTFT, decode, prompt/completion tokens, cached tokens, file payload/load count, D2H/H2D và RSS/swap. Kiểm text/usage giữa hai policy trước khi quy kết hiệu năng; output khác có thể làm thay đổi routing/history. Không gộp Mars và Love thành một tốc độ đại diện; không cộng các timer chồng lấn thành wall hoặc gọi file payload là SSD bytes. Chưa có kết quả 7B mới chỉ vì runner đã sẵn sàng.
