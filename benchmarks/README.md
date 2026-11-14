# Benchmark của MoE-TierEngine

Kết quả mới 12/11 (A3): [P2 — số thread i5](experiments/2026-11-12-threads-i5/README.md) với GPU dense, tier off, chat max64: 6/8/12 thread không nhanh hơn 4 thread một cách nhất quán (total 4/N 0,890–1,021) → giữ 4 thread trên i5. Lượt đầu dừng do RAM bị app khác chiếm, giữ làm hồ sơ.

Kết quả mới 11/11 (i7, tầng VRAM trên checkpoint chung): [P1 — tầng VRAM bật/tắt trên i7](experiments/2026-11-11-gpu-tier-i7-r2/README.md) RTX 5060 Laptop (7,53 GiB VRAM, pool 845–858/1024 expert ≈ 83%): Phép B chat/decode max 64 tắt nhanh hơn 8/8 (`off_faster: True`, total 1,78–2,00×, decode trung vị 4,92→9,75 tok/s; upload 33–41 GiB/request); Phép A prefill max 1 không nhất quán (`off_faster: False, on_faster: False`, on thắng 1/4 ở cặp 0 turn 2 `0,857×`, off thắng 3/4 `1,186–1,330×`). Mặc định chưa đổi.

Kết quả mới10/11 (S5): [P1 — chat/decode tầng VRAM](experiments/2026-11-10-gpu-tier-chat-i5/README.md) GPU dense hai phía, tắt tầng VRAM expert nhanh hơn8/8 (total1,39–1,87×; decode trung vị3,13→5,53tok/s). Tiêu chí cân nhắc đổi mặc định đạt; mặc định chưa đổi.

Kết quả mới10/11 (S4): [P1 — prefetch theo router](experiments/2026-11-10-prefill-prefetch-i5/README.md) prefill prefetch on/off cho TTFT on/off1,240/1,172 (cặp0) và0,959/0,928 (cặp1): không nhất quán; lookahead decode gate mở0,066%. Chưa chứng minh lợi ích prefetch.

Kết quả mới10/11 (S3, tầng VRAM): [P1 — tier bật/tắt](experiments/2026-11-10-gpu-expert-tier-i5/README.md) với dense GPU hai phía, tắt tầng VRAM expert nhanh hơn ở4/4 (TTFT on/off1,211–1,446). Lợi GPU đến từ dense; tầng VRAM theo thiết kế hiện tại là chi phí ròng trên workload này.

Kết quả mới10/11 (S3): [P1 — backend CPU/GPU](experiments/2026-11-10-backend-cpu-gpu-i5/README.md) GPU auto đạt rule4/4, TTFT ratio cpu/auto1,235–1,618, chủ yếu attention/dense cuBLAS; tầng VRAM expert lợi ròng chưa chứng minh (upload đồng bộ ~5,3GiB/request). RSS auto +~380MiB.

Kết quả mới10/11 (S2): [P1 — batch prefill prompt dài](experiments/2026-11-10-expert-batch-long-i5/README.md) đạt tiêu chí chốt trước: TTFT/prefill off/on ≥1,05 ở cả4 so sánh (1,065–1,094), expert_compute giảm29,4–33,6%. Một ABBA, max1; warmup bakery cặp1 chỉ1,034. Giữ opt-in.

Kết quả mới10/11: [P1 — cửa sổ RAM S1](experiments/2026-11-10-ram-window-i5/README.md) hoàn tất24/24request mixed trên tree sạch, RSS/HWM max3188,6MiB, VmSwap engine0, spread request17–24 14MiB; swap-out hệ thống tăng do process khác. Chỉ là cửa sổ hữu hạn, không hard-cap.

Kết quả mới09/11: [P2 — kernel expert nhiều token](experiments/2026-11-09-expert-batch-int8-i5/README.md) đạt1,410×/1,381× trên weights thật resident. Model có expert_compute thấp hơn17,5–29,2% trong4so sánh, nhưng TTFT chỉ tốt hơn3/4 nên chưa đạt tiêu chí toànrequest. Tính năng CPU/INT8 opt-in, mặc địnhoff; scratch bổ sung144KiB được tính vào budget.

Câu hỏi nghiên cứu hiện hành: **Trong ngân sách bộ nhớ cho trước, cách quản lý và di chuyển expert ảnh hưởng thế nào đến thời gian phản hồi, tốc độ sinh và lượng dữ liệu phải truyền?** GPU là tài nguyên tùy chọn; ngân sách RAM và VRAM phải ghi riêng.

## Phạm vi báo cáo và mô hình chung (từ 11/11/2026)

Từ 11/11, mọi benchmark mới trên **cả i5 và i7** dùng một checkpoint chung: checkpoint INT8 của i5 (config.json `9f1312f8…`, tokenizer.json `b1fb1517…`, danh sách hash 16 file ở [checkpoints/olmoe-int8-i5.sha256](checkpoints/olmoe-int8-i5.sha256)). Rà manifest checkpoint trong raw ngày 11/11:

| Nhóm | Hồ sơ | Dùng cho báo cáo mới |
| --- | --- | --- |
| Đã đo trên checkpoint chung | 24/27 hồ sơ i5 (mọi P1, P2 và P3 i5) | Dùng được nguyên trạng; không cần đo lại chỉ vì đổi quy ước model |
| Không dùng model (kernel synthetic) | Gate–Up synthetic 08/11 | Chỉ tín hiệu kernel, như trước |
| **Khác model** | Vận hành i7 kỳ 1 (config `272998dd…`), lượt i7 11/11 (trên branch i7), INT8/INT4 từ Huy (không có manifest) | **Không trích làm số đo của báo cáo mới.** Giữ làm lịch sử; số i7 cho báo cáo lấy từ lượt đo trên checkpoint chung |

Không xóa hồ sơ cũ: Git history vẫn giữ payload nên xóa không giảm dung lượng repo, còn liên kết từ PLAN/EVIDENCE_MATRIX/PROGRESS sẽ gãy. Kỳ 1 chủ yếu là lý thuyết, nên kỳ 2 **không tái lập bộ đo vận hành kỳ 1** (i5 chuyển P2, i7 chuyển P3 ngày 11/11). Báo cáo chỉ cần trích P1 (và kết luận ngắn từ P2); P3 là lưu trữ, không mở mặc định.

## ĐỌC TRƯỚC — chọn tài liệu theo việc cần làm

Không đọc toàn bộ JSON/trace để hiểu tiến độ. Chọn đúng câu hỏi, đọc tài liệu tóm tắt và README của hồ sơ liên quan; chỉ mở raw khi cần kiểm số liệu hoặc điều tra lỗi. Không cần đọc tất cả P1 mỗi lượt.

| Nhãn | Tài liệu | Dùng khi nào? |
| --- | --- | --- |
| **ĐỌC TRƯỚC** | [SYSTEM_GUIDE](../docs/SYSTEM_GUIDE.md) | Hiểu luồng hệ thống và thuật ngữ |
| **HIỆN HÀNH** | [PLAN](../docs/PLAN.md) | Biết hướng đã dừng và việc tiếp theo |
| **BẰNG CHỨNG CHÍNH** | [EVIDENCE_MATRIX](../docs/thesis/EVIDENCE_MATRIX.md) | Chọn luận điểm được phép viết; ưu tiên bảng “Đóng góp trực tiếp” |
| **BÁO CÁO KỲ 1** | [baocao/README](../baocao/README.md) | Tìm bản báo cáo và đính chính số liệu i7 |
| **TRA CỨU** | [PROGRESS](../docs/PROGRESS.md) | Đọc quy tắc và mục mới nhất; chỉ tìm mốc cũ khi cần |
| **THEO TÁC VỤ** | [ARCHITECTURE](../docs/ARCHITECTURE.md), [WORKFLOWS](../docs/WORKFLOWS.md), [SCENARIOS](SCENARIOS.md) | Sửa engine, bàn giao hoặc chuẩn bị đúng phép đo |

## Nhãn ưu tiên của 29 hồ sơ thí nghiệm

Nhãn do mục lục này quản lý, không chèn vào README/raw đã khóa checksum. **Ưu tiên đọc không đồng nghĩa kết quả tích cực, đã nghiệm thu hoặc đủ điều kiện so tốc độ hai máy.** Giữ giới hạn của từng hồ sơ khi trích dẫn.

### P1 — QUAN TRỌNG: bằng chứng chính để viết báo cáo

Đọc hồ sơ liên quan đến luận điểm đang viết. Đây là bộ nguồn chính, không phải danh sách tối ưu đã thành công.

| Hồ sơ | Dùng cho câu hỏi nào? |
| --- | --- |
| [Tầng VRAM expert bật/tắt trên i7 11/11](experiments/2026-11-11-gpu-tier-i7-r2/README.md) | Cùng checkpoint chung với i5 trên RTX 5060 Laptop (pool 845–858/1024 expert): chat/decode max64 tắt nhanh hơn 8/8 (`off_faster: True`, total 1,78–2,00×, decode trung vị 4,92→9,75 tok/s); prefill max1 không nhất quán (`on` thắng 1/4, `off` thắng 3/4) |
| [Inclusive/victim](experiments/2026-10-22-web-session-i5-081202/README.md) | Đánh đổi cách bố trí cache; số mẫu nhỏ |
| [KV prefix và dispatch](experiments/2026-10-22-prefill-i5/README.md) | Lợi ích tái sử dụng tiền tố trong workload hẹp; dispatch nhóm bị loại |
| [Kiểm chức năng model thật](experiments/2026-10-22-stage1-validation/README.md) | Smoke chức năng và giới hạn kiểm chất lượng; chưa nghiệm thu baseline |
| [RAM i5 08/11](experiments/2026-11-08-ram-budget-i5/README.md) | RSS thực khác cache budget; chưa bảo đảm plateau/hard-cap |
| [Allocator i5 08/11](experiments/2026-11-08-allocator-i5/README.md) | Bằng chứng giữ vùng đã free, thu hẹp giả thuyết leak lớn |
| [Chat/decode tầng VRAM S5 10/11](experiments/2026-11-10-gpu-tier-chat-i5/README.md) | Chat nhiều lượt max64, GPU dense: tắt tầng VRAM expert total1,39–1,87× ở8/8, decode trung vị3,13→5,53tok/s; upload33–41GiB/request |
| [Prefetch theo router S4 10/11](experiments/2026-11-10-prefill-prefetch-i5/README.md) | Prefill prefetch on/off không nhất quán (phủ13–15% lần nạp); lookahead decode gate đủ điều kiện0,066% ở cache64; cơ chế đúng chức năng, lợi ích chưa chứng minh |
| [Tầng VRAM expert bật/tắt 10/11](experiments/2026-11-10-gpu-expert-tier-i5/README.md) | GPU dense bật hai phía: tắt tầng VRAM expert giảm TTFT17–31% ở4/4; upload ~5,3GiB/request là chi phí chính; mặc định chưa đổi |
| [Backend CPU/GPU S3 10/11](experiments/2026-11-10-backend-cpu-gpu-i5/README.md) | GPU auto giảm TTFT19–38% nhờ dense/attention cuBLAS; tầng VRAM expert phục vụ ~8% miss nhưng upload ~5,3GiB/request; expert vẫn CPU |
| [Batch prefill prompt dài S2 10/11](experiments/2026-11-10-expert-batch-long-i5/README.md) | Primary đạt 4/4: TTFT giảm 6,1–8,6%, compute 29,4–33,6% ở prompt 64/128 token, max1/CPU4; opt-in, không suy decode/i7 |
| [Cửa sổ RAM S1 10/11](experiments/2026-11-10-ram-window-i5/README.md) | Gate an toàn S1 đạt 24/24 request trên tree sạch; spread 8 request cuối 14MiB; chưa hard-cap/plateau dài hạn |

### P2 — BỔ TRỢ: giới hạn và quyết định giữ/bỏ hướng

Khi viết báo cáo, thường chỉ cần kết luận ngắn và liên kết nguồn; mở mẫu chi tiết nếu bảo vệ quyết định đó.

| Hồ sơ | Vì sao giữ? |
| --- | --- |
| [Số thread CPU A3 i5 12/11](experiments/2026-11-12-threads-i5/README.md) | Chọn cấu hình vận hành: 6/8/12 thread không hơn 4 thread nhất quán; expert_compute không tăng theo số thread |
| [Expert INT8 nhiều token](experiments/2026-11-09-expert-batch-int8-i5/README.md) | Compute có lợi ở kernel và các so sánh model; TTFT chưa ổn định, opt-in/off mặc định |
| [Vận hành i5 kỳ 1](experiments/2026-10-22-phase1-i5/README.md) | Bối cảnh kỳ 1 (cold/warm/hội thoại, checkpoint chung, revision cũ d4adefd). Kỳ 2 không tái lập bộ này; vận hành hiện hành dùng S1 |
| [RAM guard 22/10](experiments/2026-10-22-ram-guard-i5/README.md) | Kiểm từ chối cấu hình và giới hạn vận hành hữu hạn |
| [CPU trước load](experiments/2026-11-04-cpu-load-i5/README.md) | Xác nhận chọn CPU bỏ CUDA ngay lúc load; chẩn đoán gate lookahead |
| [KV buffer reuse hẹp](experiments/2026-11-05-kv-reuse-narrow-i5/README.md) | Giảm KV capacity đồng thời trong một cặp; chưa kết luận RSS tổng quát |
| [Gate–Up resident weights thật](experiments/2026-11-09-gate-up-resident-int8-i5/README.md) | Không đạt tiêu chí chính 4 luồng; căn cứ khép hướng |
| [Admission cache theo tần suất](experiments/2026-11-09-cache-admission-int8-i5/README.md) | Tăng nạp lại trong workload thử; candidate đã gỡ |

### P3 — TRA CỨU: chẩn đoán, bước trung gian và dữ liệu lịch sử

Không đọc mặc định khi hỏi tiến độ hoặc viết phần kết quả chính. Chỉ mở khi cần truy nguyên một kết luận hay tái lập phép thử cụ thể.

| Hồ sơ | Vai trò lưu trữ |
| --- | --- |
| [Vận hành i7 kỳ 1](experiments/2026-10-23-phase1-i7/README.md) | Lịch sử kỳ 1 trên **model khác** (config `272998dd…`); không trích cho báo cáo mới. Đọc [đính chính](../baocao/README.md) nếu cần |
| [GPU expert compute A1 i5, hai lượt hỏng 12–14/11](experiments/2026-11-14-gpu-compute-i5/README.md) | Lượt 1 engine bị swap; lượt 2 chạy bằng pin, GPU bị bóp giữa chừng nên rule không dùng được. Căn cứ cho power guard (AC + GPU không throttle) trong preflight; chưa là kết quả A1 |
| [Lookahead 03/11](experiments/2026-11-03-lookahead-i5/README.md) | Lượt dừng sớm, chưa có đối chứng hiệu năng |
| [INT8/INT4 và routing từ Huy](experiments/2026-11-03-quantization-tradeoff/README.md) | Dữ liệu tiếp nhận thiếu provenance; chưa là bằng chứng chuẩn hóa |
| [Chuỗi RAM 04/11](experiments/2026-11-04-ram-sequence-i5/README.md) | Tái hiện pressure trước các bước chẩn đoán sau |
| [KV reuse A/B chưa thành cặp](experiments/2026-11-05-kv-reuse-ab-i5/README.md) | Giữ failure, không dùng làm kết luận lợi ích candidate |
| [Trace từng stage](experiments/2026-11-05-memory-stage-i5/README.md) | Phát hiện hai bộ KV cùng tồn tại |
| [RAM/reclaim 05/11](experiments/2026-11-05-ram-reclaim-i5/README.md) | Bước khoanh vùng trước phép allocator |
| [Gate–Up synthetic 1 luồng](experiments/2026-11-08-gate-up-int8-i5/README.md) | Tín hiệu kernel trung gian; không suy thành chat speedup |
| [Gate–Up HTTP A/B](experiments/2026-11-08-gate-up-model-int8-i5/README.md) | Kết quả trái chiều, tiền đề cho quyết định ở hồ sơ resident |

## Chỉ mở dữ liệu thô khi cần kiểm chứng

| Loại file | Chức năng và cách đọc |
| --- | --- |
| README của thí nghiệm | Tóm tắt protocol, kết quả và giới hạn; đọc trước raw |
| result.json / turn-*.json | Số đo gốc. Chọn run/request/trường cần kiểm; nếu result chứa turns thì không cần đọc thêm các bản turn tương ứng |
| memory.jsonl / trace | Chuỗi chẩn đoán. Lọc đúng khoảng thời gian và trường liên quan, không đổ toàn bộ vào ngữ cảnh |
| source.zip / logs | Truy source hoặc lỗi build/runtime; không phải nội dung báo cáo chính |
| CHECKSUMS.sha256 | Kiểm bất biến bằng công cụ; không cần đọc payload để tính checksum |
| [manifest.json](manifest.json), [results/](results/), [catalog lịch sử](../docs/BENCHMARKS.md) | Tra đúng entry cần trích; nhãn historical_unverified không phải số đo mới |
| runs/ | Lượt local chưa được xếp hồ sơ ở mục lục; không tự coi là bằng chứng báo cáo hoặc tự xóa |

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
