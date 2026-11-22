# Module 11 — Đo lường hiệu năng như một nhà khoa học

## Mục tiêu

Biết thiết kế một phép đo hiệu năng cho ra kết luận đáng tin: kiểm soát nhiễu, chọn chỉ số, thiết kế so sánh có đối chứng, chốt quy tắc kết luận trước khi đo, lưu dấu nguồn gốc dữ liệu, và trình bày kết quả trung thực, kể cả kết quả âm. Đây là phần làm nên chất lượng khoa học của đồ án, và là phần hội đồng hay hỏi vặn nhất.

**Dùng khi đi làm:** mọi quyết định kỹ thuật dựa trên số liệu: tối ưu có thực sự nhanh hơn không, có hồi quy hiệu năng sau khi deploy không, A/B test sản phẩm, so sánh cloud instance, benchmark mô hình AI. Người biết đo đúng hiếm hơn người biết tối ưu, và kết luận sai từ số đo sai còn tệ hơn không đo.

## 1. Vì sao đo hiệu năng khó

Cùng một chương trình, chạy hai lần, cho hai thời gian khác nhau. Nguồn nhiễu thường gặp:

| Nguồn nhiễu | Ví dụ trong đồ án |
| --- | --- |
| Ứng dụng khác chiếm CPU, RAM | Chrome, Slack mở lại làm engine bị swap, lượt đo A1 lần 1 bị hủy |
| Nhiệt độ và xung nhịp | CPU giảm xung khi nóng; turbo boost thay đổi theo tải |
| Nguồn điện | Lượt A1 lần 2 chạy bằng pin, GPU bị giảm xung 5–8 lần giữa chừng |
| Page cache | Lần đầu đọc tệp từ SSD, lần sau từ RAM |
| Trạng thái cache của chương trình | Request đầu tiên cache rỗng (lạnh), các request sau cache đã đầy (ấm) |
| Khởi động lần đầu | Biên dịch JIT, cấp phát lần đầu, nạp thư viện |
| Trôi theo thời gian | Máy nóng dần, page cache đầy dần, bộ nhớ phân mảnh dần |

Nguyên tắc: **không thể loại hết nhiễu, nhưng có thể thiết kế phép đo sao cho nhiễu không làm sai kết luận**.

## 2. Chọn chỉ số

Đo cái gì quan trọng ngang đo thế nào. Với inference LLM:

| Chỉ số | Định nghĩa | Phản ánh |
| --- | --- | --- |
| TTFT | thời gian đến token đầu tiên | độ trễ người dùng cảm nhận khi bấm gửi; chi phối bởi prefill |
| Tốc độ sinh (tok/s) | số token sinh sau token đầu / thời gian decode | tốc độ chữ hiện ra; chi phối bởi decode |
| Tổng thời gian | từ đầu đến cuối request | tổng chi phí |
| Thông lượng | token/s của cả hệ thống khi phục vụ nhiều request | năng lực server |
| p50, p99 | trung vị, phân vị 99 của độ trễ | trải nghiệm điển hình và trường hợp xấu |

Đồ án còn đo **công việc** bên cạnh thời gian: số lần cache hit/miss, số byte đọc tệp, số byte chép qua PCIe, số lần tính expert trên GPU. Đây là thói quen rất tốt: thời gian dao động theo nhiễu, nhưng số byte đọc thì tất định. Khi thời gian thay đổi, xem chỉ số công việc để biết vì sao. Ví dụ, ở thí nghiệm trần RAM, số lần đọc tệp giống hệt nhau giữa hai mức trần; chỉ tốc độ mỗi lần đọc thay đổi. Điều đó cho phép kết luận chắc chắn nguyên nhân là page cache nhỏ đi, không phải engine làm khác đi.

**Bộ đếm thời gian thành phần có thể chồng lấp.** Engine ghi `expert_io_ms`, `expert_compute_ms`, `attention_ms`, `lm_head_ms`, nhưng đọc tệp và tính có thể chạy song song, nên tổng các thành phần không bằng `total_ms`. Dùng tỉ lệ thành phần để định hướng, không cộng chúng lại.

## 3. Thiết kế so sánh có đối chứng

### 3.1 Vấn đề của "trước và sau"

Đo cấu hình A hôm qua, cấu hình B hôm nay, thấy B nhanh hơn 10%. Kết luận B tốt hơn? Không được: hôm nay máy có thể mát hơn, ít ứng dụng chạy hơn, phiên bản driver khác. Thay đổi môi trường lẫn với thay đổi cấu hình, không tách được (confounding).

**Quy tắc:** chỉ so sánh hai cấu hình đo xen kẽ trong cùng một lượt, cùng binary, cùng checkpoint, cùng workload, chỉ khác đúng một yếu tố.

### 3.2 Xen kẽ ABBA

Chạy theo thứ tự A, B, B, A. Giả sử môi trường trôi tuyến tính theo thời gian (máy nóng dần làm mọi thứ chậm thêm δ mỗi khối):

```
khối:     1      2       3       4
cấu hình: A      B       B       A
thời gian: a     b+δ     b+2δ    a+3δ

trung bình A = a + 1,5δ
trung bình B = b + 1,5δ          → hiệu B − A = b − a, độ trôi bị triệt tiêu
```

Với thứ tự A, B, A, B, trung bình A là `a + δ`, trung bình B là `b + 2δ`: độ trôi δ bị cộng nhầm vào B. ABBA triệt tiêu trôi tuyến tính; đây là lý do nó được dùng trong thử nghiệm lâm sàng và trong mọi phép đo của đồ án.

Trong đồ án, mỗi khối là một **tiến trình mới** (khởi động engine lại), để trạng thái của khối trước (cache, page cache, phân mảnh) không lây sang khối sau. Mỗi khối chạy 8 request: 4 đầu khởi động không tính, 4 sau được đo. Ghép khối 1 với 2 và khối 4 với 3, được 2 cặp × 4 prompt = **8 so sánh**.

### 3.3 Quy tắc kết luận chốt trước

Rủi ro lớn nhất khi tự đo kết quả của chính mình là **tìm cách đọc số liệu cho hợp ý muốn**: thử nhiều chỉ số rồi chỉ báo chỉ số đẹp, bỏ lượt "bất thường", đổi ngưỡng sau khi thấy số. Cách chống: viết protocol (câu hỏi, cấu hình hai phía, thứ tự chạy, chỉ số, quy tắc kết luận) và **commit vào git trước khi đo**. Commit có thời điểm, ai cũng kiểm được protocol có trước dữ liệu. Khoa học gọi đây là đăng ký trước (pre-registration).

Quy tắc của đồ án, ghi trong `benchmarks/SCENARIOS.md`:

> B nhanh hơn nếu **cả 8** tỉ số `t_B / t_A` (cho tổng thời gian và thời gian decode) đều nhỏ hơn `1/1,05`. B chậm hơn nếu cả 8 đều lớn hơn 1,05. Còn lại là **không nhất quán**.

Quy tắc này khắt khe có chủ đích: chỉ kết luận khi hiệu ứng lặp lại ở mọi cặp và đủ lớn để vượt nhiễu thông thường (5%). Kết quả "không nhất quán" là kết quả hợp lệ và được báo cáo như vậy (prefetch, cache 96 MiB, trần 8192 MiB trên i5).

### 3.4 Thống kê

Với số mẫu nhỏ (8 so sánh), đồ án không tính p-value hay khoảng tin cậy, mà dùng quy tắc "mọi cặp cùng chiều". Vì sao vẫn có sức nặng: nếu thực sự không có khác biệt, mỗi cặp có khoảng 50% khả năng nghiêng về mỗi bên, xác suất cả 8 cặp cùng nghiêng một phía một cách ngẫu nhiên là `2 × 0,5⁸ ≈ 0,8%`. Đây chính là phép kiểm định dấu (sign test). Thêm điều kiện vượt 5% làm nó còn chặt hơn.

Khi có nhiều mẫu hơn, các công cụ chuẩn là: trung vị và phân vị; **bootstrap** để có khoảng tin cậy cho trung vị hoặc tỉ số; kiểm định Wilcoxon cho dữ liệu ghép cặp; Mann–Whitney cho hai nhóm độc lập. Tránh dùng trung bình và độ lệch chuẩn với thời gian, vì phân phối thời gian thường lệch phải (có đuôi dài các lần chậm bất thường).

## 4. Kiểm soát môi trường

Danh sách kiểm tra trước khi đo, đúng như đồ án làm (`tools/preflight_memory.py`, `tools/run_benchmark.py`):

- RAM khả dụng đủ lớn; không có swap mới phát sinh.
- Không có build, test hay benchmark khác đang chạy.
- Máy cắm sạc; GPU không có lý do giảm xung (`nvidia-smi`).
- Tắt tự khóa, tự mờ màn hình, chế độ ngủ (`systemd-inhibit`).
- Nếu cần đo SSD thật: xóa tệp khỏi page cache và xác minh.
- Trong khi đo: bộ giám sát lấy mẫu bộ nhớ mỗi 250 ms, dừng lượt đo nếu engine bị swap.

Hai lượt hỏng của thí nghiệm A1 là bài học tốt. Lượt 2 có đủ số liệu và quy tắc cho ra "không nhất quán". Nếu không có nhật ký nguồn điện, có thể đã kết luận sai rằng tính expert trên GPU không có lợi. Nhật ký cho thấy GPU bị giảm xung giữa chừng vì máy chạy pin, nên lượt đó được ghi là hỏng do môi trường, quy trình được bổ sung kiểm tra nguồn điện, rồi đo lại.

## 5. Nguồn gốc dữ liệu (provenance)

Một con số không có nguồn gốc là con số không kiểm chứng được. Mỗi hồ sơ đo của đồ án ghi:

| Thông tin | Lý do |
| --- | --- |
| Commit git và cờ `git_dirty` | biết chính xác code nào; `dirty = true` nghĩa là có thay đổi chưa commit, số đo không tái lập được |
| Mã băm SHA-256 của binary | cùng commit nhưng build khác vẫn có thể khác |
| Mã băm checkpoint | đảm bảo đúng trọng số; hai máy kiểm mã băm để chắc dùng chung mô hình |
| Phần cứng, driver, nguồn điện | bối cảnh để diễn giải |
| Cấu hình đầy đủ | cache, context, max_tokens, threads, chế độ GPU, workload |
| Kết quả thô từng request | tính lại được mọi con số tổng hợp |
| `CHECKSUMS.sha256` | phát hiện dữ liệu bị sửa sau khi đo |

**Dữ liệu thô bất biến.** Phát hiện sai sót trong phần diễn giải thì viết **đính chính** trong README, không sửa tệp kết quả gốc. Đồ án đã làm vậy khi review hồ sơ của máy i7: trung vị bị tính sai cách và đơn vị GiB/GB bị lẫn, nên README được thêm mục "Đính chính khi review", còn dữ liệu thô giữ nguyên.

**Kết quả âm và lượt hỏng được giữ.** Prefetch không nhất quán, admission làm tệ hơn, thêm luồng không giúp, cache lớn hơn không nhanh hơn: tất cả đều có hồ sơ và được báo cáo. Chỉ báo kết quả đẹp là một dạng thiên lệch xuất bản, làm người đọc tin rằng mọi thứ đều có ích.

## 6. Định luật Amdahl: biết trước giới hạn

Nếu phần cần tăng tốc chiếm tỉ lệ f của thời gian và được tăng tốc s lần:

```
S = 1 / ((1 − f) + f/s)  ≤  1 / (1 − f)
```

Áp dụng trước khi bỏ công tối ưu. Với Qwen3 trên i5, attention và LM head chiếm 12–23% thời gian request. Đưa toàn bộ phần đó lên GPU, kể cả nhanh vô hạn, request chỉ nhanh hơn tối đa `1/(1−0,12) ≈ 1,13` đến `1/(1−0,23) ≈ 1,30` lần. Trong khi đó, chờ đọc expert chiếm 58–76%. Kết luận rõ ràng: muốn nhanh hơn phải giảm I/O, không phải tăng tốc dense.

## 7. Profiling

Đo tổng thời gian cho biết **có** chậm hay không; profiling cho biết **chậm ở đâu**.

- **Đo bằng timer trong code** (như engine làm): rẻ, luôn bật, nhưng chỉ thấy những gì đã đặt timer.
- **Profiler lấy mẫu** (`perf record -g`, flamegraph): thấy mọi hàm, không cần sửa code.
- **Bộ đếm phần cứng** (`perf stat -e cache-misses,instructions,cycles`): IPC thấp và nhiều cache miss nghĩa là bị giới hạn bởi bộ nhớ.
- **Theo dõi I/O** (`iostat -x 1`, `blktrace`): thấy byte SSD thực sự đọc, phân biệt với byte ứng dụng yêu cầu.
- **GPU:** Nsight Systems, Nsight Compute.

Đồ án ghi rõ những gì **chưa** đo được: chưa dùng `iostat` nên tốc độ đọc là "payload engine chia thời gian đọc của engine", không phải băng thông SSD; giả thuyết "thêm luồng không giúp vì băng thông RAM bão hòa" chưa có profiler xác nhận. Nói rõ giới hạn của bằng chứng là một phần của trình bày trung thực.

## 8. Các lỗi kinh điển

1. Đo bản debug.
2. Đo một lần duy nhất.
3. So hai số đo ở hai ngày hoặc hai máy khác nhau rồi kết luận về code.
4. Tính lần khởi động đầu tiên (lạnh) chung với các lần sau.
5. Để trình biên dịch tối ưu bỏ đoạn code đang đo (thiếu `black_box`).
6. Dùng trung bình khi có ngoại lai.
7. Cộng phần trăm cải thiện của nhiều thí nghiệm khác nhau thành "tổng cải thiện".
8. Nhầm GB với GiB (sai 7,4%), MB/s với Mb/s (sai 8 lần).
9. Báo "nhanh hơn 50%" mà không nói là giảm 50% thời gian (nhanh gấp 2) hay tăng 50% tốc độ (nhanh gấp 1,5).
10. Nhầm "byte ứng dụng đọc" với "byte SSD đọc" khi có page cache.

---

## Bài tập

### Bài 1 ★ — Amdahl

Một request gồm: đọc expert 60%, tính expert 25%, attention 10%, khác 5%.

1. Tăng tốc tính expert 4 lần (GPU): request nhanh hơn bao nhiêu?
2. Giảm byte đọc một nửa (INT4) làm phần đọc nhanh gấp 2: nhanh hơn bao nhiêu?
3. Làm cả hai?

<details><summary>Lời giải</summary>

1. `1 / (0,75 + 0,25/4) = 1/0,8125 ≈ 1,23`.
2. `1 / (0,40 + 0,60/2) = 1/0,70 ≈ 1,43`.
3. `1 / (0,15 + 0,30 + 0,0625) = 1/0,5125 ≈ 1,95`.

Tối ưu phần lớn nhất trước.
</details>

### Bài 2 ★★ — Mô phỏng ABBA

Viết mô phỏng Python: cấu hình A có thời gian thật 10,0 s, B là 9,6 s (nhanh hơn 4%). Mỗi khối có độ trôi tuyến tính +0,3 s mỗi khối và nhiễu ngẫu nhiên `N(0, 0,15)`. Lặp 10 000 lần:

1. Thiết kế AB (2 khối), ABAB (4 khối), ABBA (4 khối). Mỗi thiết kế kết luận "B nhanh hơn" sai (hoặc đúng) bao nhiêu phần trăm?
2. Áp dụng quy tắc của đồ án (mọi cặp tỉ số < 1/1,05) với ABBA, 4 prompt mỗi khối: tỉ lệ kết luận "B nhanh hơn" là bao nhiêu? Vì sao thấp? Điều đó tốt hay xấu?

<details><summary>Gợi ý</summary>

AB và ABAB thường thiên lệch theo hướng của độ trôi; ABBA thì không. Với quy tắc 5% và hiệu ứng thật chỉ 4%, ABBA gần như không bao giờ kết luận "nhanh hơn", mà ra "không nhất quán". Đó là chủ đích: quy tắc chỉ khẳng định những hiệu ứng đủ lớn và ổn định. Cái giá là bỏ sót hiệu ứng nhỏ (âm tính giả); cái được là gần như không bao giờ khẳng định sai (dương tính giả). Với một báo cáo khoa học, đánh đổi này hợp lý.
</details>

### Bài 3 ★★ — Đọc kết quả thô

Mở `benchmarks/experiments/2026-11-14-gpu-compute-i5/run/result.json`. Mỗi phần tử của `pairs` có `timers.total_ms.on`, `timers.total_ms.off` và `work.expert_file_loads`. Viết script:

1. In 8 tỉ số `on/off` của `total_ms`, kiểm khớp với khoảng 0,743–0,813 ghi trong README.
2. Tính trung vị `expert_file_loads` của mỗi phía.
3. Áp quy tắc của đồ án và in kết luận.
4. Kiểm checksum của thư mục: `sha256sum -c CHECKSUMS.sha256`.

<details><summary>Lời giải</summary>

```python
import json, statistics
d = json.load(open("benchmarks/experiments/2026-11-14-gpu-compute-i5/run/result.json"))
r = [p["timers"]["total_ms"]["on"] / p["timers"]["total_ms"]["off"] for p in d["pairs"]]
print([round(x, 3) for x in r], min(r), max(r))
for side in ("on", "off"):
    print(side, statistics.median(p["work"]["expert_file_loads"][side] for p in d["pairs"]))
print("on nhanh hơn" if all(x < 1/1.05 for x in r) else "không nhất quán / khác")
```

Tệp còn có trường `decision_rule` và `compute_faster` do harness tự tính: hãy so kết luận của bạn với trường đó. Đây là cách một người ngoài kiểm lại số liệu của báo cáo.
</details>

### Bài 4 ★★ — Bắt lỗi một báo cáo

Một bạn viết: *"Sau khi tối ưu kernel, tốc độ tăng từ 8,24 lên 10,73 tok/s (+30%). Thêm tối ưu prefetch tăng thêm 15%, nên tổng cải thiện là 45%. Số liệu đo hôm qua (trước) và hôm nay (sau) trên laptop, mỗi bên chạy một lần."* Liệt kê mọi vấn đề phương pháp.

<details><summary>Lời giải</summary>

1. Trước và sau đo khác ngày: lẫn yếu tố môi trường.
2. Mỗi bên chỉ chạy một lần: không biết nhiễu lớn cỡ nào.
3. Không nói workload, cấu hình, phiên bản code, cache lạnh hay ấm.
4. Cộng phần trăm sai cách: hai cải thiện nhân với nhau (`1,30 × 1,15 = 1,495`), và quan trọng hơn là hai thí nghiệm khác điều kiện thì không cộng hay nhân được.
5. Không có quy tắc kết luận chốt trước.
6. Không nói máy có cắm sạc không (laptop).
7. Không có chỉ số công việc để giải thích vì sao nhanh hơn.
</details>

### Bài 5 ★★★ — Viết một protocol

Bạn muốn kiểm tra: "Kernel expert nhiều token có làm TTFT của Qwen3 nhanh hơn với prompt 128 token không?". Viết protocol hoàn chỉnh theo mẫu các mục trong `benchmarks/SCENARIOS.md`: câu hỏi, cấu hình cố định, biến thay đổi, workload, thứ tự khối, số request, chỉ số chính, quy tắc kết luận, điều kiện dừng do môi trường, cách lưu dữ liệu.

<details><summary>Gợi ý</summary>

Đọc mục S2 (`expert-batch-long`) trong `benchmarks/SCENARIOS.md`, vốn đã làm đúng câu hỏi này cho OLMoE. Những điểm cần nghĩ thêm cho Qwen3: phải chạy dưới trần cgroup (nếu không thì page cache lớn hơn RAM?); có xóa page cache trước mỗi khối không (nếu không, khối sau hưởng page cache của khối trước, làm lệch ABBA); kernel chỉ ảnh hưởng prefill nên chỉ số chính là TTFT, không phải tốc độ decode; `max_tokens = 1` để cô lập prefill.
</details>

### Bài 6 ★★★ — Kiểm chứng chéo

Chọn một con số trong chương 5 của báo cáo (`baocao/lan2/latex/chapters/ch5_thucnghiem.tex`), ví dụ "Bộ nhớ ẩn danh đỉnh 5,62 GiB" của Qwen3 trên i5 ở trần 10240. Lần ngược: README hồ sơ nào, tệp thô nào, trường nào; tự tính lại từ dữ liệu thô và đổi đơn vị. Con số có khớp không?

<details><summary>Gợi ý</summary>

README `2026-11-17-qwen3-ram-cap-i5` ghi đỉnh anon 5755,7 MiB. `5755,7 / 1024 = 5,62 GiB`. Tìm trường tương ứng trong `limit-10240/` (tệp `result.json` hoặc chuỗi giám sát bộ nhớ). Làm bài này với ba con số bất kỳ là cách tốt nhất để tự tin trả lời khi hội đồng hỏi "số này lấy ở đâu".
</details>

## Tài liệu đọc thêm

- Brendan Gregg, *Systems Performance* (2nd ed.), chương về phương pháp.
- Georges, Buytaert, Eeckhout, *Statistically Rigorous Java Performance Evaluation* (2007).
- Mytkowicz et al., *Producing Wrong Data Without Doing Anything Obviously Wrong!* (2009).
- Raj Jain, *The Art of Computer Systems Performance Analysis*.
- Tài liệu của crate `criterion` (Rust) về thống kê trong benchmark.
