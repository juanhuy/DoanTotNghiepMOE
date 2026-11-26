# Module 14 — Dự án thực hành và chuẩn bị bảo vệ

Module này có hai phần:

- **Phần A:** sáu dự án nhỏ, mỗi dự án tự làm lại một phần cốt lõi của đồ án từ đầu. Làm xong cả sáu là bạn có thể tự viết lại một phiên bản thu nhỏ của MoE-TierEngine, và có sáu sản phẩm để đưa vào CV.
- **Phần B:** bộ câu hỏi hội đồng hay hỏi, kèm câu trả lời mẫu dựa trên số liệu thật của đồ án.

---

## Phần A — Dự án thực hành

Mỗi dự án nên là một repo riêng trên GitHub, có README nêu: mục tiêu, cách chạy, kết quả đo, giới hạn. Đó là cách trình bày nhà tuyển dụng đánh giá cao.

### Dự án 1 — Mini LLM inference bằng numpy (module 04, 05, 06)

**Mục tiêu:** cài forward của một mô hình MoE tí hon, có KV cache, sinh văn bản bằng greedy, và chứng minh khớp Transformers.

Yêu cầu:

1. Dùng Transformers tạo `OlmoeForCausalLM` tí hon có trọng số ngẫu nhiên (xem `tools/make_olmoe_reference.py`), lưu ra safetensors.
2. Tự cài bằng numpy: embedding, RMSNorm, QK-norm, RoPE (rotate-half), attention có mặt nạ nhân quả và GQA, router top-k (có cờ chuẩn hóa), SwiGLU, LM head.
3. Hai chế độ: `forward_full(tokens)` và `forward_step(token, kv_cache)`.
4. Test: logits khớp Transformers dưới 1e-5 ở mọi vị trí; sinh 30 token bằng `forward_step` cho đúng chuỗi như `model.generate(do_sample=False)`.
5. Bonus: thêm lượng tử hóa INT8 theo hàng cho expert, đo top-1 và KL so với bản FP32.

**Kỹ năng thể hiện:** hiểu kiến trúc Transformer/MoE đến từng phép tính, kiểm thử số học.

### Dự án 2 — Bộ mô phỏng chính sách cache cho expert (module 09)

**Mục tiêu:** so sánh chính sách cache trên vết định tuyến thật.

Yêu cầu:

1. Thu vết định tuyến từ engine (`MOE_ROUTE_TRACE`) cho 3 loại hội thoại khác nhau (hỏi đáp, viết code, dịch).
2. Cài FIFO, LRU, LFU có suy giảm, SLRU, chính sách của engine (heat + hai đoạn 70%), OPT.
3. Với dung lượng 5, 10, 20, 32 expert mỗi lớp: vẽ hit rate theo dung lượng cho từng chính sách.
4. Thêm mô hình chi phí (đọc SSD 2,4 ms, page cache 0,5 ms) và báo thời gian ước lượng, không chỉ hit rate.
5. Viết kết luận: chính sách hiện tại cách OPT bao xa? Có đáng cải tiến không?

**Kỹ năng thể hiện:** cấu trúc dữ liệu, phân tích dữ liệu, ra quyết định dựa trên số liệu.

### Dự án 3 — Kernel INT8 trong Rust (module 02, 08)

**Mục tiêu:** viết kernel matvec INT8 nhanh nhất có thể và giải thích được vì sao.

Yêu cầu:

1. Bốn phiên bản: vòng lặp thường; 8 bộ cộng; AVX2 intrinsics; AVX2 + Rayon theo hàng.
2. Bản nhiều token (2 và 4 vector đầu vào cùng lúc).
3. Đo bằng `criterion` với ma trận `768 × 2048` và `1024 × 2048`.
4. Tính băng thông đạt được, vẽ lên biểu đồ roofline của máy bạn.
5. Test: mọi phiên bản khớp bản thường (dung sai tương đối 1e-5), kể cả kích thước không chia hết cho 32.

**Kỹ năng thể hiện:** tối ưu hiệu năng mức thấp, SIMD, đo đạc, hiểu giới hạn phần cứng.

### Dự án 4 — Server chat streaming (module 03)

**Mục tiêu:** một server tương thích API OpenAI, stream bằng SSE, bọc quanh Dự án 1 hoặc một mô hình nhỏ.

Yêu cầu:

1. Axum, `POST /v1/chat/completions` hỗ trợ `stream: true/false`, `GET /v1/models`.
2. Một request tại một thời điểm (429 khi bận); channel có giới hạn; hủy khi client ngắt; hạn mức thời gian tổng.
3. Test bằng client chính thức của OpenAI (đặt `base_url` về server của bạn).
4. Bonus: hàng đợi có giới hạn thay vì từ chối ngay; endpoint `/metrics` theo định dạng Prometheus.

**Kỹ năng thể hiện:** backend, async, giao thức streaming, xử lý lỗi và hủy.

### Dự án 5 — Kernel GPU và mô hình chi phí offload (module 10)

**Mục tiêu:** chứng minh bằng số đo nguyên tắc "di chuyển phép tính đến dữ liệu".

Yêu cầu:

1. Cài kernel INT8 SwiGLU bằng CuPy `RawKernel` (có thể bắt đầu từ `KERNEL_SOURCE` của đồ án), kiểm khớp numpy.
2. Đo ba cách tính 1 expert: CPU numpy; GPU khi trọng số phải chép lên mỗi lần; GPU khi trọng số đã ở VRAM.
3. Đo băng thông PCIe (pageable và pinned) của máy.
4. Xây mô hình chi phí từ các số đo và dự đoán thời gian cho "expert dùng n lần"; kiểm dự đoán bằng đo thật với n = 1, 5, 50.

**Kỹ năng thể hiện:** CUDA, đo GPU, mô hình hóa hiệu năng.

### Dự án 6 — Một nghiên cứu hiệu năng hoàn chỉnh (module 11)

**Mục tiêu:** tự thực hiện một thí nghiệm theo đúng quy trình của đồ án, từ câu hỏi đến báo cáo.

Gợi ý câu hỏi: "Với OLMoE chạy CPU, cache 32 MiB/lớp so với 64 MiB/lớp ảnh hưởng thế nào đến tốc độ sinh khi checkpoint nằm trong page cache?"

Yêu cầu:

1. Viết protocol (câu hỏi, cấu hình, ABBA, workload, quy tắc kết luận) và commit **trước** khi chạy.
2. Kiểm điều kiện máy trước khi đo; ghi commit, mã băm binary và checkpoint.
3. Chạy bằng `tools/demo.sh` hoặc `tools/run_benchmark.py`; lưu dữ liệu thô và tạo `CHECKSUMS.sha256`.
4. Viết README: kết quả theo quy tắc, bảng tỉ số từng cặp, chỉ số công việc (hit, byte đọc), diễn giải, giới hạn.
5. Nếu kết quả không nhất quán, báo đúng như vậy.

**Kỹ năng thể hiện:** tư duy thực nghiệm, tính trung thực khoa học, viết tài liệu kỹ thuật.

### Kiến thức của đồ án dùng được ở đâu khi đi làm

| Vị trí | Module quan trọng nhất | Trong đồ án đã có |
| --- | --- | --- |
| Backend engineer | 01, 03, 08, 11 | server Axum, SSE, hủy request, giới hạn tài nguyên, xử lý lỗi |
| Systems / performance engineer | 02, 08, 09, 11 | SIMD, roofline, cache, page cache, cgroup, đo ABBA |
| ML / inference engineer | 05, 06, 07, 10, 12 | Transformer, MoE, lượng tử hóa, kernel GPU, so logits |
| MLOps / platform | 08, 11, 13 | cgroup (Kubernetes), giám sát bộ nhớ, tái lập kết quả |
| Nghiên cứu (học lên cao học) | 04, 06, 09, 11, 12 | thiết kế thí nghiệm, đối chứng, báo cáo trung thực |

---

## Phần B — Chuẩn bị bảo vệ đồ án

Nguyên tắc trả lời:

- **Trả lời bằng số liệu cụ thể** và nói số đó từ thí nghiệm nào.
- **Phân biệt đo được với suy ra.** "Đo được 58–76%" khác "theo Amdahl, tối đa 1,3 lần".
- **Nói rõ giới hạn trước khi bị hỏi.** Hội đồng đánh giá cao việc tự biết giới hạn của mình.
- **Không nhận điều chưa đo.**

### B.1 Tổng quan

**Hỏi: Đồ án giải quyết vấn đề gì?**

Chạy mô hình ngôn ngữ MoE lớn hơn RAM trên máy tính cá nhân. Ví dụ Qwen3-30B-A3B: checkpoint INT8 khoảng 30 GB, trong khi máy thử nghiệm có 16 GB RAM. MoE chỉ dùng một phần nhỏ expert cho mỗi token, nên có thể giữ expert hay dùng ở bộ nhớ nhanh và đọc phần còn lại khi cần. Đồ án xây engine làm việc đó, phân tầng VRAM, RAM, SSD, và đo chi phí của từng cách tổ chức.

**Hỏi: Đóng góp chính là gì?**

Ba điểm, đều có số đo:
1. Engine tính đúng: trùng top-1 với Transformers ở 100% vị trí trên cả hai mô hình thật, KL cỡ 10⁻¹¹.
2. Kết luận về cách dùng VRAM: chỉ chứa expert mà tính trên CPU thì chậm hơn tắt tầng VRAM (8/8 so sánh trên cả hai máy); tính expert ngay trên GPU thì nhanh hơn (8/8 trên cả hai máy). Nơi tính quan trọng hơn nơi chứa.
3. Chạy được Qwen3-30B-A3B dưới trần RAM 8 GB, bộ nhớ ẩn danh khoảng 5,7 GiB, không lần nào bị dừng do thiếu bộ nhớ, trên hai máy.

**Hỏi: Vì sao chọn Rust mà không phải Python hay C++?**

Python quá chậm cho vòng nhân ma trận và khó kiểm soát chính xác bộ nhớ, trong khi đồ án cần biết từng MiB. C++ nhanh nhưng dễ lỗi bộ nhớ với nhiều bộ đệm sống lâu và nhiều luồng. Rust nhanh ngang C++, kiểm tra an toàn bộ nhớ lúc biên dịch, và vẫn cho phép SIMD, FFI tới CUDA khi cần. Phần `unsafe` được giới hạn ở FFI GPU và intrinsics, có ghi bất biến.

**Hỏi: Đồ án khác llama.cpp thế nào?**

llama.cpp là dự án lớn, hỗ trợ rất nhiều mô hình và định dạng; cách offload phổ biến của nó là đưa N lớp lên GPU, hoặc chỉ định tensor nào (ví dụ toàn bộ expert) nằm ở CPU, và dựa vào mmap cùng page cache của hệ điều hành. Đồ án tập trung vào **quản lý từng expert** như một đơn vị cache: cache theo lớp có ngân sách byte, chính sách loại bỏ theo heat, pool expert trên VRAM tính tại chỗ, và đặc biệt là đo có đối chứng từng quyết định thiết kế. Đồ án không nhằm thay llama.cpp mà nhằm trả lời câu hỏi nghiên cứu về chi phí di chuyển expert.

### B.2 Lý thuyết

**Hỏi: MoE là gì, khác mô hình thường ở đâu?**

Lớp feed-forward được thay bằng nhiều expert và một router. Mỗi token, router chọn k expert (k = 8 trong cả hai mô hình) và chỉ tính các expert đó. Qwen3-30B-A3B có khoảng 30 tỷ tham số nhưng mỗi token chỉ dùng khoảng 3 tỷ. Tốc độ tính như mô hình 3 tỷ, nhưng mọi expert vẫn phải truy cập được vì token sau có thể chọn expert khác.

**Hỏi: OLMoE và Qwen3 khác nhau thế nào về mặt cài đặt?**

Qwen3 có 48 lớp và 128 expert mỗi lớp (OLMoE: 16 và 64); dùng GQA 32 đầu query với 4 đầu KV (OLMoE: 16 và 16); QK-norm theo từng đầu (OLMoE: trên toàn vector); chuẩn hóa lại trọng số top-k (OLMoE: không). Engine đọc `model_type` và các tham số cấu hình để chọn đúng cách tính. Các khác biệt nhỏ này đều được kiểm bằng so logits với Transformers.

**Hỏi: Vì sao decode chậm hơn prefill tính trên mỗi token?**

Decode tính từng token, mỗi trọng số đọc từ bộ nhớ chỉ dùng cho 1 phép nhân–cộng (cường độ tính toán khoảng 2 phép/byte với INT8), nên bị giới hạn bởi băng thông bộ nhớ. Prefill xử lý nhiều token cùng lúc, mỗi trọng số được dùng cho nhiều token. Engine còn gom token theo expert khi prefill để mỗi expert chỉ nạp một lần.

**Hỏi: KV cache là gì? Dung lượng bao nhiêu?**

Lưu key và value của các token đã xử lý để không tính lại mỗi bước. Dung lượng `2 × L × T × h_kv × d_h × 4` byte với FP32: OLMoE 256 KiB mỗi token (128 MiB cho 512 token), Qwen3 192 KiB mỗi token nhờ GQA.

**Hỏi: Lượng tử hóa INT8 làm mô hình kém đi bao nhiêu?**

Đo ở mức logits so với BF16 gốc: top-1 trùng 98,7% (Qwen3) và 98,9% (OLMoE), KL trung bình khoảng 0,008 nat. Cả 4 vị trí đổi top-1 đều là vị trí gần hòa, hai token đứng đầu chênh 0,004 đến 0,19 logit. Giới hạn: đây là 6 chuỗi ngắn; đồ án chưa đo perplexity trên tập lớn hay benchmark tác vụ.

**Hỏi: Sao không dùng INT4 cho nhanh hơn?**

Đồ án hướng tới chạy mô hình lớn **với chất lượng giữ được** trên phần cứng hạn chế. INT8 đã được kiểm gần như không đổi đầu ra (top-1 trùng 98,7–98,9% so với BF16). INT4 giảm một nửa số byte phải đọc, nhưng làm lệch đầu ra nhiều hơn hẳn và cần một vòng đánh giá chất lượng riêng. Nhóm chọn giữ INT8 và giảm I/O bằng cách khác không đổi đầu ra: dành VRAM cho các expert hay dùng (văn bản khớp 16/16 với CPU). Engine có đường INT4 nhưng đồ án không công bố số liệu INT4.

### B.3 Thiết kế

**Hỏi: Engine chọn expert nào để loại khỏi cache?**

Cache mỗi lớp có ngân sách byte (mặc định 64 MiB). Mỗi expert có bộ đếm heat (số lần được router chọn, giảm một nửa sau mỗi 65 536 lượt). Cache chia hai đoạn: thử thách (expert mới vào) và bảo vệ (expert đã được dùng lại khi đang trong cache, tối đa 70% ngân sách). Khi cần chỗ, loại expert có heat thấp nhất trong đoạn thử thách, hòa thì loại expert dùng lâu nhất. Đây là biến thể của SLRU với tiêu chí LFU có suy giảm, không phải LRU thuần.

**Hỏi: Dense của Qwen3 không vừa GPU 4 GB, vậy Qwen3 có dùng được GPU trên máy i5 không?**

Có. Dense FP32 của Qwen3 khoảng 4,9 GB, lớn hơn GTX 1650 4 GB, nên với cấu hình mặc định engine chạy CPU. Nhưng đưa dense lên GPU vốn lợi ích nhỏ: dense chỉ chiếm 12–23% thời gian, theo Amdahl tối đa 1,13–1,30 lần; còn chờ đọc expert chiếm 58–76%. Vì vậy nhóm làm ngược lại: để dense ở CPU (`MOE_GPU_DENSE=off`) và dành hết VRAM cho 713 expert hay dùng nhất (học từ một workload khác), tính chúng bằng kernel GPU. Đối chứng ABBA dưới trần 10 GiB: nhanh hơn CPU ở 16/16 so sánh, tốc độ sinh 0,94 → 1,33 tok/s, văn bản khớp 16/16. Máy i7 lặp lại độc lập: 1 594 expert, 16/16, 1,92 → 3,74 tok/s.

**Hỏi: Chỉ 11% expert nằm trên VRAM, sao lại có ích?**

Router không chọn đều: một số expert được chọn nhiều hơn hẳn. Mô phỏng trên routing thật cho thấy nếu chọn expert theo thứ tự nạp thì chỉ giảm 19% số lần đọc SSD, còn chọn theo tần suất thì giảm 28–31%. Ngoài ra các expert trong pool được tính trên GPU nên CPU cũng bớt việc (thời gian tính expert trên CPU giảm từ khoảng 24 xuống 15–18 giây mỗi request).

**Hỏi: Hồ sơ tần suất học từ chính bộ prompt đo thì có gian lận không?**

Không học từ bộ đo. Hồ sơ được học từ workload `long-prefill` (hai đoạn văn dài về thư viện), còn phép đo dùng workload `mixed` (Mars, Python, love). Tệp hồ sơ được commit cùng protocol trước khi đo, kèm mã băm. Giới hạn còn lại: mới thử một workload đo; workload khác xa (ngôn ngữ khác, mã nguồn) chưa được kiểm.

**Hỏi: RAM guard có đảm bảo không bao giờ hết RAM không?**

Không. RAM guard là kiểm tra hợp tác: engine ước lượng trước khi nạp và kiểm RSS ở các điểm định sẵn khi chạy. Bộ nhớ có thể tăng giữa hai lần kiểm. Giới hạn cứng trong thực nghiệm do cgroup v2 của kernel Linux đảm nhận (`memory.max`, cấm swap); RAM guard được đặt thấp hơn trần cgroup 1 GiB để engine từ chối trước khi kernel phải can thiệp.

**Hỏi: Làm sao biết engine tính đúng trên mô hình 30 tỷ tham số khi máy không chạy nổi Transformers?**

Viết bộ tham chiếu chạy Transformers **từng lớp**: mỗi lớp được dựng, nạp trọng số riêng của lớp đó, chạy, rồi giải phóng. RAM đỉnh khoảng 2,8 GB. Công cụ này được kiểm trước trên mô hình nhỏ (khớp chạy cả mô hình đến 10⁻⁵). Rồi so ba bộ logits: engine, Transformers dùng cùng expert INT8, Transformers BF16 gốc, để tách lỗi của engine khỏi ảnh hưởng của lượng tử hóa.

### B.4 Kết quả và phương pháp

**Hỏi: Đo bao nhiêu lần? Có ý nghĩa thống kê không?**

Mỗi đối chứng là một lượt ABBA gồm 4 tiến trình mới, 8 so sánh ghép cặp. Quy tắc chốt trước: chỉ kết luận nhanh hơn khi **cả 8** tỉ số cùng chiều và vượt 5%. Nếu thực sự không có khác biệt, xác suất cả 8 cặp cùng nghiêng một phía một cách ngẫu nhiên là khoảng 0,8% (kiểm định dấu). Khi không đạt, kết quả được ghi là không nhất quán, như với prefetch và cache 96 MiB. Các đối chứng Qwen3 (cache 96 MiB, pool VRAM) ghép hai cặp khối × 8 request nên có 16 so sánh; xác suất cả 16 cùng chiều do ngẫu nhiên là khoảng 0,003%.

**Hỏi: Vì sao dùng ABBA mà không đo A rồi đo B?**

Máy thay đổi theo thời gian (nóng lên, page cache đầy dần). Đo A rồi B thì độ trôi bị tính nhầm vào khác biệt. ABBA triệt tiêu độ trôi tuyến tính vì trung bình hai phía có cùng "thời điểm trung bình".

**Hỏi: Có so tốc độ giữa i5 và i7 không?**

Không. Hai máy khác CPU, GPU, SSD. Mỗi máy được dùng để kiểm kết luận có lặp lại không (cùng chiều 8/8), không để xếp hạng. Hai máy dùng chung checkpoint đã kiểm mã băm, và sinh văn bản giống hệt nhau.

**Hỏi: Tại sao tăng cache cho Qwen3 lại không nhanh hơn?**

Dưới trần cgroup cố định, cache expert và page cache dùng chung ngân sách. Tăng cache từ 64 lên 96 MiB/lớp làm hit rate tăng từ 50% lên 60% và dữ liệu đọc giảm 20%, nhưng page cache bớt 1,5 GiB nên các lần đọc còn lại phải xuống SSD nhiều hơn. Hai hiệu ứng gần như triệt tiêu, kết quả không nhất quán. Muốn nhanh hơn cần bộ nhớ ngoài trần: VRAM là bộ nhớ như vậy, và dành VRAM cho expert đã làm Qwen3 nhanh hơn 16/16.

**Hỏi: Số liệu này lấy ở đâu, kiểm lại được không?**

Mỗi thí nghiệm có một thư mục hồ sơ trong `benchmarks/experiments/`: README nêu câu hỏi, protocol, kết quả và giới hạn; dữ liệu thô từng request; mã băm commit, binary, checkpoint; tệp checksum. Protocol được commit trước khi đo. Lượt hỏng (swap, chạy pin) vẫn được giữ kèm nguyên nhân.

### B.5 Câu hỏi khó

**Hỏi: 1,1–1,8 tok/s thì dùng được vào việc gì?**

Với Qwen3 trên CPU, tốc độ này đủ cho việc không cần tương tác nhanh (xử lý tài liệu, chạy nền), và quan trọng hơn là chứng minh một mô hình gấp đôi RAM chạy được, ổn định, kết quả đúng. Phân tích đã chỉ ra nút thắt (I/O chiếm 58–76%), và hướng giải quyết đã được đo: dành VRAM cho expert đưa Qwen3 lên 1,33 tok/s trên GPU 4 GB và 3,74 tok/s trên GPU 8 GB. Với OLMoE, engine đạt khoảng 10 tok/s trên i5 và khoảng 50 tok/s trên i7 khi tính expert trên GPU, đủ cho hội thoại.

**Hỏi: Có phải em chỉ chọn kết quả đẹp để báo cáo?**

Không. Báo cáo có riêng mục các hướng không mang lại lợi ích: prefetch không nhất quán, admission làm tệ hơn 2,56%, tăng luồng không giúp, cache lớn hơn không nhanh hơn, tầng VRAM khi tính CPU thì chậm hơn. Quy tắc kết luận được commit trước khi đo, nên không thể chỉnh sau khi thấy số.

**Hỏi: Kết quả "tầng VRAM nhanh hơn 50 lần" trên i7 có đáng tin không?**

Kết luận "nhanh hơn ở 8/8 so sánh" đáng tin. Độ lớn của tỉ số thì không nên trích làm mức điển hình: trong lượt đó, phía đối chứng đọc tệp chỉ 3,1–3,8 GB/s, thấp hơn mức 6,1–12,1 GB/s ở các lượt i7 khác, nên mốc so sánh chậm hơn bình thường. Báo cáo ghi rõ điều này.

**Hỏi: Nếu có thêm ba tháng, em sẽ làm gì?**

Theo thứ tự lợi ích dự kiến:
1. Cập nhật hồ sơ đặt expert theo workload (hiện học một lần, giữ cố định), đánh giá trước bằng chính công cụ mô phỏng đã khớp engine.
2. Đối chứng có kiểm soát cách chia VRAM giữa dense và expert (hiện mới có một lượt khám phá trên i7).
3. Định tuyến có tính đến cache khi router gần hòa, kèm đánh giá chất lượng.
4. Đánh giá chất lượng trên benchmark chuẩn và perplexity.

Mỗi hướng đi kèm một thí nghiệm ABBA với protocol chốt trước.

### B.6 Những câu không nên nói

- "Qwen3 không dùng được GPU 4 GB": sai từ 24/11, dành VRAM cho expert đã đo nhanh hơn 16/16.
- "Pool VRAM nhanh hơn 1,42 lần trong mọi trường hợp": đó là trung vị tốc độ sinh trên một workload; khoảng đo được của tổng thời gian là 0,69–0,87 lần (i5).
- "Engine không bao giờ hết RAM": RAM guard không phải giới hạn cứng.
- "Nhanh hơn X lần so với llama.cpp": chưa đo đối chứng với llama.cpp.
- "Chính sách cache là LRU": sai, là heat có suy giảm với hai đoạn.
- "Tổng cải thiện là A% + B%": các thí nghiệm khác điều kiện, không cộng được.
- "Tốc độ đọc SSD là X GB/s": đó là tốc độ của engine (byte yêu cầu chia thời gian đọc), không phải băng thông thiết bị đo độc lập.

### Luyện tập

1. Nhờ một người bạn đóng vai hội đồng, hỏi ngẫu nhiên 10 câu trong phần B, trả lời không nhìn tài liệu, mỗi câu dưới 1 phút.
2. Với mỗi con số bạn nói ra, chỉ được tên hồ sơ chứa nó (bài 6 module 11).
3. Mở web demo (`tools/demo.sh olmoe-gpu`) và giải thích từng ô số liệu trên màn hình trong 3 phút.
