# Module 08 — Bộ nhớ và hệ điều hành

## Mục tiêu

Hiểu dữ liệu đi từ SSD lên RAM, lên cache CPU, lên GPU như thế nào và mỗi chặng tốn bao lâu; đọc đúng các con số bộ nhớ của Linux (RSS, anon, file, swap, page cache); biết cách giới hạn bộ nhớ của một tiến trình bằng cgroup. Đây là nền tảng để hiểu mọi kết quả đo của đồ án.

**Dùng khi đi làm:** vận hành server và container (Kubernetes đặt giới hạn bộ nhớ bằng cgroup; pod bị "OOMKilled" là đúng cơ chế trong module này), tối ưu cơ sở dữ liệu (buffer pool, page cache), chẩn đoán rò rỉ bộ nhớ, thiết kế hệ thống lưu trữ, tối ưu hiệu năng mọi loại phần mềm.

## 1. Phân cấp bộ nhớ

Bộ nhớ càng nhanh thì càng nhỏ và càng đắt. Các con số sau là cỡ độ lớn điển hình cho laptop, đủ để ước lượng:

| Tầng | Dung lượng | Độ trễ truy cập | Băng thông |
| --- | --- | --- | --- |
| Thanh ghi CPU | vài trăm byte | < 1 ns | — |
| Cache L1 | 32–48 KB mỗi nhân | ~1 ns | hàng trăm GB/s |
| Cache L2 | 256 KB–2 MB mỗi nhân | ~4 ns | |
| Cache L3 | 8–36 MB dùng chung | ~10–20 ns | |
| RAM (DDR4/DDR5) | 8–64 GB | ~80–100 ns | 20–80 GB/s |
| VRAM của GPU (GDDR6) | 4–24 GB | | 100–1000 GB/s (cho GPU) |
| PCIe giữa RAM và GPU | — | ~vài µs mỗi lần chép | 3.0 x16: ~12 GB/s; 4.0 x16: ~25 GB/s |
| SSD NVMe | 0,5–4 TB | ~50–100 µs | 2–7 GB/s đọc tuần tự |
| HDD | vài TB | ~5–10 ms | 100–200 MB/s |

Hai bài học:

- Khoảng cách giữa các tầng là **hàng trăm đến hàng nghìn lần**. Một lần đọc SSD bằng khoảng 1000 lần đọc RAM.
- Băng thông khác độ trễ. Đọc một khối lớn liên tục thì tốc độ gần với băng thông; đọc nhiều mẩu nhỏ rải rác thì bị độ trễ chi phối.

Ánh xạ sang đồ án:

```
SSD (tệp checkpoint, 7–30 GB)  ──đọc tệp──►  RAM (cache expert, 64 MiB/lớp)  ──PCIe──►  VRAM (pool expert)
       ~2 GB/s                                      ~20–40 GB/s                           GPU đọc ~100+ GB/s
```

Mỗi chặng đi lên chậm hơn chặng trên nó cả chục lần. Toàn bộ thiết kế phân tầng nhằm giữ dữ liệu hay dùng ở tầng nhanh nhất có thể.

## 2. Cache CPU và tính cục bộ

CPU không đọc từng byte từ RAM mà đọc theo **cache line** 64 byte. Hai loại cục bộ giúp cache hiệu quả:

- **Cục bộ không gian:** dùng một byte thì có khả năng sắp dùng các byte cạnh nó. Duyệt mảng tuần tự tận dụng trọn mỗi cache line, và bộ nạp trước phần cứng (hardware prefetcher) tự đoán và nạp các line tiếp theo.
- **Cục bộ thời gian:** dùng một dữ liệu thì có khả năng sắp dùng lại nó.

Hệ quả cho code:

- Duyệt ma trận row-major theo hàng nhanh hơn duyệt theo cột rất nhiều lần.
- Trọng số expert lưu theo hàng nên mỗi tích vô hướng là một lần đọc tuần tự, tận dụng tối đa prefetcher.
- Vector đầu vào `x` (8 KB) được dùng lại cho mọi hàng của ma trận, nên nằm sẵn trong L1/L2. Chỉ trọng số là phải kéo từ RAM.

## 3. Mô hình roofline: tính toán hay bộ nhớ là giới hạn

**Cường độ tính toán** (arithmetic intensity) của một thuật toán = số phép tính / số byte phải đọc từ bộ nhớ.

Máy có tốc độ tính tối đa P (phép/giây) và băng thông B (byte/giây). Thuật toán có cường độ I thì đạt tối đa:

```
tốc độ = min(P, I × B)
```

Điểm gãy `I* = P / B`. Thuật toán có `I < I*` bị giới hạn bởi bộ nhớ (memory-bound); `I > I*` bị giới hạn bởi tính toán (compute-bound).

Ví dụ CPU laptop: 4 nhân × 2 FMA × 8 làn × 2 phép × 3 GHz ≈ 384 GFLOP/s, băng thông khoảng 30 GB/s, nên `I* ≈ 13` phép mỗi byte.

| Phép tính | Cường độ | Kết luận |
| --- | --- | --- |
| Matvec INT8 (decode, 1 token) | mỗi byte trọng số: 1 nhân + 1 cộng = **2 phép/byte** | bị giới hạn bởi bộ nhớ, rất xa điểm gãy |
| Matvec FP32 | 2 phép / 4 byte = **0,5 phép/byte** | còn tệ hơn |
| GEMM với n token (prefill) | khoảng **2n phép/byte** INT8 | n = 64 đã vượt điểm gãy |

Đây là lời giải thích định lượng cho nhiều kết quả của đồ án:

- Decode chậm hơn prefill tính trên mỗi token.
- Thêm luồng CPU không giúp: tính toán vốn không phải giới hạn.
- Kernel nhiều token nhanh hơn vì tăng cường độ tính toán từ 2 lên 4–8 phép/byte.
- Lượng tử hóa làm decode nhanh lên: cùng số phép, ít byte hơn.

## 4. Bộ nhớ ảo

Mỗi tiến trình thấy một không gian địa chỉ riêng (bộ nhớ ảo). Hệ điều hành ánh xạ địa chỉ ảo sang RAM vật lý theo đơn vị **trang** (page) 4 KB, qua bảng trang. CPU có bộ đệm TLB để tra nhanh.

- Khi cấp phát (`malloc`), hệ điều hành chỉ ghi nhận vùng địa chỉ; trang vật lý chỉ được cấp khi chương trình **chạm** vào lần đầu (page fault, rồi cấp trang). Vì vậy cấp 10 GB mà không dùng thì RSS không tăng.
- **mmap một tệp:** ánh xạ tệp vào không gian địa chỉ; đọc vào vùng đó gây page fault, hệ điều hành đọc trang tương ứng từ đĩa vào page cache.

## 5. Page cache

Khi chương trình đọc một tệp, Linux không chỉ trả dữ liệu mà còn **giữ bản sao trong RAM** (page cache). Lần đọc sau cùng chỗ đó lấy từ RAM, không chạm đĩa. Page cache dùng RAM còn trống và được thu hồi tự động khi tiến trình cần bộ nhớ.

Điều này cực kỳ quan trọng khi đọc số liệu của đồ án:

- Checkpoint OLMoE 7 GB vừa RAM 16 GB, nên sau lần đọc đầu, toàn bộ tệp nằm trong page cache. "Đọc expert từ tệp" thực chất là chép từ page cache với tốc độ 9–12 GB/s, không phải đọc SSD.
- Muốn đo đọc SSD thật, phải **xóa tệp khỏi page cache** trước khi đo. Đồ án dùng `posix_fadvise(..., POSIX_FADV_DONTNEED)` cho từng tệp checkpoint, rồi kiểm bằng `fincore` rằng còn 0 byte trong cache.
- Chỉ số `expert_file_bytes_read` là số byte engine **yêu cầu đọc**, không phải số byte SSD thực sự đọc. Hồ sơ đo ghi rõ phân biệt này.

Các cách đọc tệp:

| Cách | Đặc điểm |
| --- | --- |
| `read` | đọc tuần tự theo con trỏ tệp, qua page cache |
| `pread` | đọc tại một vị trí cho trước, không đổi con trỏ tệp; nhiều luồng dùng chung một tệp an toàn |
| `mmap` | ánh xạ tệp, truy cập như mảng; tiện nhưng khó kiểm soát lúc nào đọc đĩa và khó tính bộ nhớ |
| `O_DIRECT` | bỏ qua page cache, đọc thẳng vào bộ đệm của ứng dụng (phải căn lề); dùng khi ứng dụng tự quản cache |

**Trong đồ án:** `src/safetensors.rs::read_raw_into` dùng `read_exact_at` (tức `pread`) tại offset đã tính sẵn của tensor. Một expert là một lần đọc liên tục 4,7–6 MB, gần với điều kiện tốt nhất của SSD.

## 6. Đọc đúng các con số bộ nhớ của Linux

`/proc/<pid>/status` và `/proc/meminfo` cung cấp:

| Trường | Ý nghĩa |
| --- | --- |
| `VmRSS` | Resident Set Size: bộ nhớ vật lý tiến trình đang chiếm tại thời điểm đọc |
| `VmHWM` | High Water Mark: RSS lớn nhất từng đạt |
| `RssAnon` | phần bộ nhớ ẩn danh (heap, stack): dữ liệu chương trình tự tạo |
| `RssFile` | phần ánh xạ từ tệp (thư viện, tệp mmap) |
| `VmSwap` | phần bộ nhớ của tiến trình đã bị đẩy ra swap |
| `MemTotal` | tổng RAM |
| `MemAvailable` | ước lượng RAM còn cấp được mà không cần swap (gồm cả page cache thu hồi được) |

Lưu ý: "RAM trống" (`MemFree`) thường rất nhỏ trên Linux vì page cache dùng hết chỗ trống. Đó không phải dấu hiệu thiếu RAM. Chỉ số đúng để xem là `MemAvailable`.

**Trong đồ án:** `src/ram.rs::snapshot` đọc `VmRSS` từ `/proc/self/status`, `MemAvailable` và `MemTotal` từ `/proc/meminfo`, và nếu chạy trong cgroup thì đọc thêm `/sys/fs/cgroup/memory.max` và `memory.current` để biết trần thật.

## 7. Swap và OOM killer

Khi RAM gần hết:

1. Linux thu hồi page cache trước (rẻ: chỉ cần bỏ, tệp vẫn còn trên đĩa).
2. Sau đó đẩy trang ẩn danh ít dùng ra **swap** trên đĩa. Truy cập lại trang đã swap rất chậm; chương trình có thể chậm đi hàng trăm lần.
3. Nếu vẫn không đủ, **OOM killer** chọn một tiến trình (thường là tiến trình chiếm nhiều nhất) và giết nó (SIGKILL, không kịp dọn dẹp).

Đồ án từng gặp đúng điều này ở giai đoạn đầu: kernel log ghi engine bị OOM killer dừng khi bộ nhớ ẩn danh khoảng 9,93 GiB. Từ đó có hai biện pháp:

- **RAM guard** trong engine (`src/ram.rs`): ước lượng bộ nhớ cần trước khi nạp và kiểm RSS trong khi chạy; vượt ngưỡng thì trả lỗi có kiểm soát (HTTP 503) thay vì để hệ điều hành giết.
- **Bộ giám sát độc lập** khi đo: lấy mẫu mỗi 250 ms và dừng lượt đo nếu engine bắt đầu bị swap. Kết quả đo trong lúc bị swap không phản ánh hiệu năng thật.

RAM guard là **kiểm tra hợp tác**: engine tự kiểm ở các điểm định sẵn, bộ nhớ vẫn có thể tăng giữa hai lần kiểm. Nó không phải giới hạn cứng.

## 8. cgroup v2: giới hạn cứng do kernel áp đặt

cgroup (control group) là cơ chế của Linux để nhóm tiến trình và giới hạn tài nguyên cho cả nhóm. Đây là nền tảng của Docker và Kubernetes.

Các tệp quan trọng trong thư mục cgroup:

| Tệp | Ý nghĩa |
| --- | --- |
| `memory.max` | trần bộ nhớ cứng; vượt thì kernel thu hồi, không được thì OOM kill trong nhóm |
| `memory.high` | ngưỡng mềm; vượt thì kernel bóp tiến trình (thu hồi mạnh, chạy chậm lại) |
| `memory.swap.max` | trần swap; đặt 0 để cấm swap |
| `memory.current` | bộ nhớ nhóm đang dùng |
| `memory.stat` | chi tiết, gồm `anon` (ẩn danh) và `file` (page cache) |
| `memory.events` | bộ đếm sự kiện, gồm `oom_kill` |

**Điểm mấu chốt:** `memory.max` tính **cả page cache** do tiến trình trong nhóm tạo ra. Vì vậy đặt engine trong cgroup 8 GiB nghĩa là engine chỉ có 8 GiB cho cả dữ liệu của nó lẫn phần checkpoint được cache. Đây là cách mô phỏng "mô hình lớn hơn RAM" trên một máy có RAM nhiều hơn.

Cách chạy một lệnh trong cgroup riêng bằng systemd:

```bash
systemd-run --user --scope -p MemoryMax=8192M -p MemorySwapMax=0 ./target/release/moe-tier-engine
```

**Trong đồ án:** các thí nghiệm trần RAM chạy engine như trên, xóa page cache của checkpoint trước mỗi khối, rồi đọc `memory.stat` và `memory.events` để báo cáo đỉnh `anon`, đỉnh `file` và số lần `oom_kill` (hồ sơ `2026-11-16-ram-limit-*`, `2026-11-17-qwen3-ram-cap-i5`).

Đọc kết quả Qwen3 dưới trần 8192 MiB: đỉnh scope 8192 MiB (chạm trần), anon khoảng 5740 MiB, file khoảng 4518 MiB. Tổng anon và file vượt 8192 vì đây là đỉnh của từng thành phần tại những thời điểm khác nhau. Khi chạm trần, kernel thu hồi page cache (file) để nhường cho anon. Engine không bị giết, chỉ chạy chậm hơn vì phải đọc SSD nhiều hơn.

## 9. Định dạng tệp mô hình

- **Pickle** (định dạng cũ của PyTorch `.bin`): có thể chứa code tùy ý, nạp một tệp lạ là chạy code lạ. Không an toàn.
- **Safetensors:** 8 byte độ dài header, header JSON mô tả từng tensor (dtype, shape, offset), rồi dữ liệu thô liền nhau. Không chứa code, đọc được bất kỳ tensor nào bằng một lần `pread` mà không phải đọc cả tệp. Engine dựa hoàn toàn vào tính chất này để nạp từng expert riêng lẻ.
- **GGUF** (llama.cpp): tương tự nhưng gói cả tokenizer, cấu hình và hỗ trợ nhiều kiểu lượng tử hóa.

---

## Bài tập

### Bài 1 ★ — Ước lượng nhanh

Không đo, chỉ dùng bảng ở mục 1, ước lượng thời gian:

1. Đọc một expert Qwen3 (4,7 MB) từ SSD 2 GB/s; từ page cache trong RAM (10 GB/s); chép từ RAM lên VRAM qua PCIe 3.0 (12 GB/s).
2. Một token decode Qwen3 với hit rate 50%: số byte phải đọc từ SSD, và thời gian tương ứng.
3. So với tốc độ đo được (khoảng 1,1–1,3 tok/s trên i5), ước lượng của bạn có hợp lý không?

<details><summary>Lời giải</summary>

1. SSD: 2,4 ms; page cache: 0,47 ms; PCIe: 0,39 ms.
2. Mỗi token cần `48 × 8 = 384` lượt expert, 50% miss là 192 expert × 4,7 MB ≈ 0,9 GB, mất khoảng 0,45 s ở 2 GB/s.
3. Riêng đọc đã khoảng 0,45 s/token, cộng tính toán và attention, ra khoảng 1,5–2 tok/s ở trường hợp tốt. Đo được 1,1–1,3 tok/s, cùng cỡ độ lớn: hợp lý. Phép ước lượng này cũng cho thấy ngay I/O là nút thắt, đúng với đo đạc (58–76% thời gian là chờ đọc).
</details>

### Bài 2 ★ — Đọc /proc

Viết chương trình (Rust hoặc Python) in `VmRSS`, `VmHWM`, `RssAnon`, `RssFile` của chính nó: (a) lúc khởi động, (b) sau khi cấp 1 GB nhưng không ghi, (c) sau khi ghi vào toàn bộ 1 GB, (d) sau khi giải phóng. Giải thích từng bước.

<details><summary>Lời giải</summary>

(b) RSS gần như không đổi: cấp phát chỉ dành địa chỉ ảo. (c) RSS tăng khoảng 1 GB, tất cả ở `RssAnon`. (d) Với khối 1 GB, malloc dùng `mmap` riêng nên `free` trả ngay cho hệ điều hành và RSS giảm; còn `VmHWM` vẫn giữ đỉnh. Với nhiều khối nhỏ, RSS thường không giảm (module 02).
</details>

### Bài 3 ★★ — Thấy page cache bằng mắt

1. Tạo tệp 2 GB: `dd if=/dev/urandom of=/tmp/big bs=1M count=2048`.
2. Xóa tệp khỏi page cache: `python3 -c "import os; f=os.open('/tmp/big', os.O_RDONLY); os.posix_fadvise(f, 0, 0, os.POSIX_FADV_DONTNEED)"`. Kiểm bằng `fincore /tmp/big`.
3. Đo thời gian `cat /tmp/big > /dev/null` hai lần liên tiếp. Tốc độ mỗi lần?
4. Theo dõi cột `buff/cache` của `free -m` trước và sau.

<details><summary>Lời giải</summary>

Lần đầu đọc từ SSD (cỡ 1–3 GB/s tùy ổ), lần hai đọc từ RAM (cỡ 5–15 GB/s). `fincore` cho thấy tệp đi từ 0 lên 2 GB trong cache, và `buff/cache` tăng tương ứng. Đây đúng là điều đồ án phải loại trừ để đo đọc SSD thật: nếu không xóa page cache, mọi lần "đọc tệp" đều là đọc RAM.
</details>

### Bài 4 ★★ — Cgroup

Viết chương trình Python cấp phát dần từng khối 100 MB (và ghi vào) cho đến 3 GB, in tổng sau mỗi khối. Chạy nó với:

```bash
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 python3 alloc.py
```

Điều gì xảy ra? Kiểm `journalctl --user -n 20` hoặc `dmesg`. Lặp lại với `MemoryHigh=1G` thay vì `MemoryMax`.

<details><summary>Lời giải</summary>

Với `MemoryMax=1G`: chương trình bị giết (OOM kill trong cgroup) khi vượt khoảng 1 GB; thông báo dừng ở khoảng 900 MB. Phần còn lại của máy không bị ảnh hưởng. Với `MemoryHigh=1G`: chương trình không bị giết mà chạy chậm dần vì bị kernel thu hồi mạnh (và nếu cho phép swap thì bị đẩy ra swap). Kubernetes đặt `memory.max` theo `resources.limits.memory`; pod bị "OOMKilled" chính là trường hợp đầu.
</details>

### Bài 5 ★★ — Đọc rải rác so với tuần tự

Với tệp 2 GB ở bài 3 (đã xóa khỏi page cache), đo thông lượng khi đọc:

1. tuần tự cả tệp theo khối 4,7 MB,
2. 400 khối 4,7 MB ở vị trí ngẫu nhiên (dùng `os.pread`),
3. 400 000 khối 4 KB ở vị trí ngẫu nhiên.

Nhận xét. Kích thước expert (4,7–6 MB) nằm ở vùng nào?

<details><summary>Gợi ý</summary>

Khối 4,7 MB ngẫu nhiên gần như nhanh bằng đọc tuần tự, vì mỗi khối đủ lớn để che độ trễ truy cập. Khối 4 KB ngẫu nhiên chậm hơn nhiều lần nếu đọc tuần tự từng khối một (độ trễ chiếm chủ đạo); song song nhiều luồng (tăng queue depth) thì nhanh lên. Expert lớn là điều may mắn cho thiết kế: mỗi lần đọc một expert là một lần đọc "đủ lớn".
</details>

### Bài 6 ★★ — Roofline

Với máy của bạn, tra hoặc đo: số nhân, xung, hỗ trợ AVX2 hay AVX-512, loại RAM và số kênh. Tính đỉnh GFLOP/s và băng thông lý thuyết, rồi điểm gãy `I*`. Áp dụng: matvec INT8 `768 × 2048` (một ma trận expert Qwen3) mất tối thiểu bao lâu? Kernel nhiều token với 4 token có cường độ bao nhiêu?

<details><summary>Gợi ý</summary>

Matvec `768 × 2048` INT8 là 1,5 MB, ở 25 GB/s mất khoảng 63 µs. Với 4 token, cùng 1,5 MB phục vụ 4 lần nhân–cộng mỗi byte, cường độ khoảng 8 phép/byte, vẫn dưới điểm gãy của hầu hết CPU, nên tốc độ tăng gần tỉ lệ với số token. Đồ án đo được kernel 4 token nhanh hơn 1,38 lần chứ không phải 4 lần, vì còn chi phí đọc 4 vector đầu vào, ghi 4 đầu ra và vì một phần thời gian vốn không thuộc phần đọc trọng số.
</details>

### Bài 7 ★★★ — Giải thích một kết quả thật

Đọc `benchmarks/experiments/2026-11-17-qwen3-ram-cap-i5/README.md`. Giải thích bằng kiến thức của module này:

1. Vì sao siết trần từ 10240 xuống 8192 MiB không làm thay đổi hit rate hay số byte đọc, mà tốc độ sinh vẫn giảm?
2. Vì sao đỉnh `anon` gần như giống nhau ở hai mức trần?
3. Vì sao tổng `anon + file` vượt trần?

<details><summary>Lời giải</summary>

1. Hit rate và số byte đọc phụ thuộc vào định tuyến và cache expert của engine, cả hai đều không đổi. Trần nhỏ hơn chỉ làm page cache nhỏ đi, nên tỉ lệ lần đọc phải xuống SSD (thay vì lấy từ page cache) cao hơn. Tốc độ đọc trung bình giảm từ 1,6–2,2 xuống 1,4–1,7 GB/s.
2. Anon là dữ liệu engine tự quản (dense 2,29 GiB, cache 2,96 GiB...), có kích thước cố định theo cấu hình, không phụ thuộc trần.
3. Hai đỉnh được ghi ở những thời điểm khác nhau. Tại mọi thời điểm, `current = anon + file + phần nhỏ khác` ≤ trần; nhưng `max(anon) + max(file)` có thể vượt trần.
</details>

## Tài liệu đọc thêm

- Ulrich Drepper, *What Every Programmer Should Know About Memory* (2007).
- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model* (2009).
- Tài liệu kernel Linux, *Control Group v2*: https://docs.kernel.org/admin-guide/cgroup-v2.html
- `man 5 proc` (mục `/proc/pid/status`, `/proc/meminfo`), `man 2 pread`, `man 2 posix_fadvise`.
- Brendan Gregg, *Systems Performance* (2nd ed.).
- "Latency numbers every programmer should know".
