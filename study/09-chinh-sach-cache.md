# Module 09 — Chính sách cache

## Mục tiêu

Hiểu các chính sách cache kinh điển (LRU, LFU, FIFO, CLOCK, SLRU, ARC, TinyLFU), cận tối ưu Belady, các khái niệm admission và prefetch, cách phối hợp cache nhiều tầng (inclusive, exclusive/victim). Biết đánh giá chính sách bằng mô phỏng trên vết truy cập, và đọc được chính xác chính sách mà engine đang dùng.

**Dùng khi đi làm:** cache có mặt ở mọi nơi: Redis/Memcached (chính sách `allkeys-lru`, `allkeys-lfu`), CDN, buffer pool của cơ sở dữ liệu (PostgreSQL dùng clock-sweep, MySQL InnoDB dùng LRU chia hai đoạn), cache trong ứng dụng Java (Caffeine dùng W-TinyLFU), cache CPU, cache KV của LLM server. Phỏng vấn hay yêu cầu cài LRU O(1).

## 1. Khái niệm

- **Hit:** dữ liệu cần đã có trong cache. **Miss:** chưa có, phải lấy từ tầng dưới chậm hơn.
- **Hit rate** = hits / (hits + misses).
- **Eviction (loại bỏ):** cache đầy, phải bỏ một phần tử để có chỗ. **Chính sách thay thế** quyết định bỏ phần tử nào.
- **Admission (nạp vào):** có nên đưa phần tử mới vào cache hay không. Mặc định là luôn nạp; một số chính sách từ chối phần tử "có vẻ không đáng".
- **Prefetch (nạp trước):** đưa dữ liệu vào cache trước khi được yêu cầu.

Ba nguyên nhân gây miss ("3C"):

1. **Compulsory:** lần đầu truy cập, không thể tránh (trừ khi prefetch).
2. **Capacity:** cache quá nhỏ cho tập đang dùng.
3. **Conflict:** do cách tổ chức cache (với cache phần cứng); với cache phần mềm thì do chính sách loại bỏ chọn sai.

**Hit rate không phải mục tiêu cuối cùng.** Mục tiêu là thời gian hoặc chi phí. Hai cache có cùng hit rate có thể cho tốc độ khác nhau nếu chi phí của miss khác nhau (miss lấy từ RAM rẻ hơn nhiều so với miss phải đọc SSD).

## 2. Cận tối ưu: Belady (OPT)

Nếu biết trước toàn bộ tương lai, chính sách tối ưu là: **loại phần tử sẽ được dùng lại muộn nhất** (hoặc không bao giờ). Belady chứng minh điều này năm 1966.

OPT không cài được trong thực tế vì không biết tương lai, nhưng rất hữu ích để **đánh giá**: chạy OPT trên vết truy cập đã ghi để biết hit rate tối đa có thể đạt. Nếu LRU đạt 50% còn OPT chỉ 55%, cố cải thiện chính sách không đáng; phải tăng dung lượng. Nếu OPT đạt 80%, còn nhiều chỗ để cải thiện.

## 3. Các chính sách cổ điển

### FIFO

Loại phần tử vào sớm nhất. Đơn giản, không cần cập nhật khi hit. Kém khi có phần tử dùng thường xuyên nhưng vào sớm.

### LRU (Least Recently Used)

Loại phần tử **lâu nhất không được dùng**. Dựa trên cục bộ thời gian: vừa dùng thì sắp dùng lại.

Cài O(1) cho cả get và put bằng **bảng băm + danh sách liên kết đôi**: bảng băm tìm nút trong O(1); danh sách giữ thứ tự dùng, khi hit thì chuyển nút lên đầu, khi loại thì bỏ nút ở cuối.

Điểm yếu: một lần quét lớn (đọc nhiều phần tử chỉ một lần) đẩy hết các phần tử hay dùng ra khỏi cache (cache pollution).

### LFU (Least Frequently Used)

Loại phần tử **được dùng ít lần nhất**. Giữ tốt các phần tử nóng lâu dài.

Điểm yếu:
- Phần tử từng rất nóng nhưng đã hết dùng vẫn chiếm chỗ mãi. Cách sửa: **suy giảm (aging/decay)**: định kỳ chia đôi mọi bộ đếm.
- Phần tử mới vào có bộ đếm 1, dễ bị loại ngay trước khi kịp chứng tỏ giá trị.

### CLOCK

Xấp xỉ LRU với chi phí thấp: các phần tử xếp vòng tròn, mỗi phần tử có một bit "vừa dùng". Kim đồng hồ quay: gặp bit 1 thì xóa thành 0 và đi tiếp, gặp bit 0 thì loại. Dùng trong quản lý trang của hệ điều hành và PostgreSQL.

### SLRU (Segmented LRU) và 2Q

Chia cache làm hai đoạn:
- **Đoạn thử thách (probationary):** phần tử mới vào đây.
- **Đoạn bảo vệ (protected):** phần tử được dùng lại khi đang ở đoạn thử thách thì được thăng lên đây.

Khi cần chỗ, loại từ đoạn thử thách trước. Đoạn bảo vệ có giới hạn dung lượng; khi đầy, phần tử kém nhất bị hạ về đoạn thử thách. Kết quả: phần tử chỉ dùng một lần không bao giờ đẩy được phần tử nóng ra ngoài. Chống được cache pollution do quét.

### ARC (Adaptive Replacement Cache)

Tự điều chỉnh kích thước giữa phần "gần đây" và phần "thường xuyên" dựa trên các "bóng ma" (ghost list): danh sách id các phần tử vừa bị loại. Hit vào bóng ma nghĩa là phần đó cần lớn hơn. Dùng trong ZFS.

### TinyLFU và W-TinyLFU

Admission thông minh: khi phần tử mới muốn vào và phải đẩy một phần tử ra, so **tần suất ước lượng** của hai bên (đếm bằng Count-Min Sketch, tốn rất ít bộ nhớ); chỉ cho vào nếu phần tử mới thường xuyên hơn. W-TinyLFU thêm một cửa sổ LRU nhỏ cho phần tử mới để không bị từ chối oan. Dùng trong thư viện Caffeine của Java, đạt hit rate gần OPT trên nhiều workload.

## 4. Prefetch

Đánh giá một cơ chế prefetch bằng ba chỉ số:

- **Accuracy (độ chính xác):** bao nhiêu phần dữ liệu nạp trước thực sự được dùng.
- **Coverage (độ phủ):** bao nhiêu phần miss được loại bỏ nhờ nạp trước.
- **Timeliness (đúng lúc):** nạp xong trước khi cần, nhưng không quá sớm để bị loại trước khi dùng.

Prefetch sai gây hại kép: tốn băng thông và đẩy phần tử có ích ra khỏi cache.

Hai loại:

- **Theo nhu cầu đã biết:** biết chắc sẽ cần nhưng chưa đến lúc. Ví dụ: khi prefill, router của lớp đã chọn xong cho mọi token, nên biết chính xác các expert cần nạp.
- **Dự đoán:** đoán từ lịch sử. Ví dụ: đoán expert của lớp sau dựa trên expert của lớp hiện tại.

## 5. Cache nhiều tầng

Khi có hai tầng cache (RAM và VRAM), câu hỏi là một phần tử có được nằm ở **cả hai** tầng không:

- **Inclusive:** tầng lớn chứa mọi thứ tầng nhỏ có. Đơn giản, nhưng dung lượng thực bằng tầng lớn nhất (bị trùng lặp).
- **Exclusive / victim:** một phần tử chỉ ở một tầng. Phần tử bị loại khỏi tầng trên rơi xuống tầng dưới (tầng dưới là "victim cache"). Dung lượng thực bằng tổng hai tầng, nhưng phải di chuyển dữ liệu nhiều hơn.

Cache CPU AMD dùng L3 dạng victim, Intel từng dùng L3 inclusive. Trong đồ án, đây chính là lựa chọn inclusive/victim giữa RAM và VRAM.

## 6. Chính sách của MoE-TierEngine

### 6.1 Cache RAM theo lớp

Mỗi lớp có một cache expert riêng, ngân sách tính bằng byte (mặc định 64 MiB). Chính sách nằm trong `src/olmoe.rs`, hàm `admit_expert_l2`. Đây là một biến thể của **SLRU với tiêu chí LFU có suy giảm**:

1. **Bộ đếm heat:** mỗi lần router chọn expert e ở một lớp, `heat[e] += 1`. Đồng hồ của lớp tăng 1 mỗi lượt; cứ 65 536 lượt thì mọi heat của lớp bị chia đôi (suy giảm):

   ```rust
   layer.clock = layer.clock.wrapping_add(1);
   if layer.clock % 65_536 == 0 {
       layer.heat.iter_mut().for_each(|heat| *heat /= 2);
   }
   layer.heat[expert_id] = layer.heat[expert_id].saturating_add(1);
   ```

2. **Hai đoạn:** expert mới vào cache thuộc đoạn thử thách. Expert được dùng lại khi đang nằm trong cache (hit) thì thăng lên đoạn bảo vệ. Đoạn bảo vệ giới hạn ở 70% ngân sách; vượt thì expert có `(heat, last_used)` nhỏ nhất trong đoạn bảo vệ bị hạ về thử thách.

3. **Chọn expert bị loại:** ưu tiên trong đoạn thử thách, chọn expert có `(heat, last_used)` nhỏ nhất, tức heat thấp nhất, hòa thì dùng lâu nhất. Không còn ứng viên thử thách thì xét cả đoạn bảo vệ.

   ```rust
   .filter(|(_, e)| !e.is_pinned && !e.is_protected)
   .min_by_key(|(_, e)| (layer.heat[e.id], e.last_used))
   ```

4. **Ghim (pinning):** tùy chọn `MOE_HOT_PIN_COUNT` ghim N expert nóng nhất của mỗi lớp, không bao giờ bị loại (trừ khi mọi phần tử đều bị ghim). Mặc định tắt.

5. **Tái sử dụng bộ nhớ:** expert bị loại không bị giải phóng mà vùng nhớ của nó được đưa vào `recycled_buffers` để nạp expert tiếp theo.

Vì tiêu chí chính là heat (tần suất có suy giảm), chính sách này gần LFU hơn LRU; thời gian dùng cuối chỉ là tiêu chí phụ khi bằng heat. Vì vậy giao diện web gọi bộ đếm là "Evictions (bỏ expert heat thấp nhất)" chứ không gọi là "LRU evictions".

### 6.2 Phối hợp RAM và VRAM

- **Inclusive** (đối chứng): expert vừa đọc từ tệp cũng được đưa lên VRAM.
- **Victim** (mặc định khi bật tầng VRAM mà expert tính trên CPU): expert bị loại khỏi RAM được đẩy lên VRAM; khi lấy lại từ VRAM về RAM thì bản VRAM bị xóa. Hai tầng không trùng nhau.
- **Lấp đầy** (fill-only, khi tính expert trên GPU): VRAM nhận các expert được nạp đầu tiên đến khi đầy rồi giữ cố định, không trao đổi với RAM. Đơn giản, không tốn băng thông PCIe sau lần lấp đầu. Nhược điểm: pool không thích nghi; với OLMoE trên i5, pool dồn vào các lớp đầu (vì lớp đầu được nạp trước).
- **Đặt theo hồ sơ tần suất** (`MOE_GPU_EXPERT_PLACEMENT`, thêm 24/11): một lượt chạy CPU ghi số lần router chọn mỗi cặp (lớp, expert) bằng `MOE_USAGE_PATH`; lúc khởi động engine nạp các cặp có số lần lớn nhất lên VRAM tới khi đầy, rồi giữ cố định như lấp đầy. Lệnh reset chỉ xóa các expert không thuộc hồ sơ.

### 6.2b Bài học từ Qwen3: chọn đúng expert quan trọng hơn chọn đúng lớp

Với Qwen3, VRAM 4 GB chỉ chứa 713 trong 6 144 expert (11,6%). Mô phỏng trên routing thật (`tools/sim_vram_expert_pool.py`, khớp tuyệt đối bộ đếm hit/miss/eviction của engine) cho số lần đọc SSD còn lại, so với không có pool:

| Cách đặt (≈700 slot) | Đọc SSD còn lại |
| --- | ---: |
| Lấp đầy theo thứ tự nạp | 0,81 |
| Lấp đầy, chia đều slot cho các lớp | 0,80 |
| Theo tần suất học từ prompt khác | 0,69 |
| Theo tần suất của chính chuỗi đo (cận trên, nhìn trước tương lai) | 0,61 |

Chia đều theo lớp gần như không giúp: vấn đề là chọn sai expert, không phải dồn sai lớp. Đây chính là bài học của Belady (mục 3): biết trước tương lai thì tốt nhất; hồ sơ tần suất là cách xấp xỉ tương lai bằng quá khứ. Đo thật (ABBA, trần 10 GiB) với hồ sơ học từ workload khác: Qwen3 nhanh hơn CPU 16/16 trên cả i5 (tốc độ sinh 0,94 → 1,33 tok/s) và i7 (1 594 slot, 1,92 → 3,74 tok/s). Hồ sơ: `benchmarks/experiments/2026-11-24-vram-pool-sim-i5`, `2026-11-24-qwen3-vram-pool-ab-i5`, `2026-11-26-qwen3-vram-pool-ab-i7`.

### 6.3 Prefetch trong engine

- **Prefill prefetch:** khi xử lý đầu vào, nạp trước tối đa k expert còn thiếu của lớp. Đối chứng bật/tắt cho kết quả không nhất quán (TTFT 1,240 và 1,172 ở cặp đầu, 0,959 và 0,928 ở cặp sau). Không bật.
- **Dự đoán liên lớp khi decode:** kết hợp hai nguồn: bảng đếm số lần cặp (expert lớp l, expert lớp l+1) cùng được chọn, và điểm thu được khi áp router của lớp l+1 lên trạng thái ẩn hiện tại ("router hoa tiêu"). Dự đoán tối đa 2 expert chưa có trong bộ nhớ (`predict_next_layer_experts` trong `src/olmoe.rs`). Chỉ được kích hoạt khi lớp hiện tại không còn gì để nạp, điều này hiếm khi xảy ra với cache nhỏ: 15 trên 22 680 lượt xét (0,066%). Không bật.

### 6.4 Admission đã thử và bỏ

Đồ án thử thêm admission: expert mới có heat thấp hơn expert sắp bị loại thì không được nạp vào cache (ý tưởng giống TinyLFU). Kết quả: số lần nạp tăng 1,53% và dữ liệu đọc tăng 2,56%, tức làm cache kém hơn. Lý do có thể là khi decode, expert mới thường sẽ được dùng lại ngay ở các token tiếp theo (cục bộ thời gian) dù heat lịch sử còn thấp. Phương án bị gỡ khỏi code.

### 6.5 Cache lớn hơn dưới trần RAM cố định

Với Qwen3 dưới trần cgroup 10 GiB, tăng cache từ 64 lên 96 MiB/lớp: hit rate tăng từ 50% lên 60%, dữ liệu đọc giảm 20%, nhưng thời gian **không** cải thiện nhất quán. Vì cache expert và page cache của hệ điều hành dùng chung một trần: thêm 1,5 GiB cho cache expert là bớt 1,5 GiB page cache, nên các lần miss còn lại phải xuống SSD thật nhiều hơn. Bài học áp dụng rộng: **khi nhiều tầng cache chia chung một ngân sách, tối ưu một tầng có thể làm hại tầng kia**. Cơ sở dữ liệu cũng gặp đúng vấn đề này giữa buffer pool của nó và page cache của hệ điều hành.

---

## Bài tập

### Bài 1 ★ — Chạy tay

Cache chứa 3 phần tử. Chuỗi truy cập: `A B C A D B E A B C D`. Chạy tay FIFO, LRU, LFU và OPT. Với LFU, bộ đếm của mỗi phần tử giữ nguyên cả khi nó bị loại (đếm toàn bộ lịch sử); hòa thì loại phần tử vào cache sớm hơn. Đếm số miss của mỗi chính sách.

<details><summary>Lời giải</summary>

Ký hiệu M là miss, h là hit, theo thứ tự 11 lượt truy cập:

| Chính sách | Diễn biến | Số miss |
| --- | --- | --- |
| FIFO | M M M h M h M M M M M | 9 |
| LRU | M M M h M M M M h M M | 9 |
| LFU | M M M h M M M h h M M | 8 |
| OPT | M M M h M h M h h M M | 7 |

Ví dụ OPT ở lượt 5 (D vào, cache {A, B, C}): lần dùng kế tiếp của A là lượt 8, B là lượt 6, C là lượt 10, nên loại C. Ở lượt 7 (E vào, cache {A, B, D}): A dùng ở lượt 8, B ở lượt 9, D ở lượt 11, nên loại D.

LFU tốt hơn LRU ở đây vì A và B được dùng nhiều lần. OPT là giới hạn dưới: không chính sách nào ít miss hơn 7 trên chuỗi này. Chương trình ở bài 3 phải cho đúng các con số trên; đây là test đầu tiên nên viết cho bộ mô phỏng.
</details>

### Bài 2 ★★ — LRU O(1)

Cài `LruCache<K, V>` trong Rust với `get` và `put` O(1). Gợi ý: `HashMap<K, usize>` trỏ vào chỉ số trong một `Vec` các nút có `prev`/`next` (tránh con trỏ thô). Viết test với 10 000 thao tác ngẫu nhiên so với một bản cài đơn giản O(n) dùng `VecDeque`.

<details><summary>Gợi ý</summary>

Cài danh sách liên kết đôi trên mảng (arena): mỗi nút `{key, value, prev: usize, next: usize}`, hai nút giả `head` và `tail`. `get`: tra map, tách nút ra khỏi vị trí hiện tại, chèn sau `head`. `put` khi đầy: lấy nút trước `tail`, xóa khỏi map, tái sử dụng chỗ đó cho phần tử mới. Kiểm thử ngẫu nhiên so với bản đơn giản là kỹ thuật **differential testing**, rất hiệu quả để bắt lỗi cấu trúc dữ liệu (module 12).

So sánh: cache của engine dùng `VecDeque` và tìm tuyến tính (`position`). Với tối đa 14 expert mỗi lớp, tìm tuyến tính nhanh hơn bảng băm, vì dữ liệu nhỏ nằm gọn trong cache CPU. Không phải lúc nào O(1) cũng nhanh hơn O(n): với n nhỏ, hằng số quyết định.
</details>

### Bài 3 ★★ — Bộ mô phỏng cache

Viết bằng Python một bộ mô phỏng nhận vết truy cập (danh sách id) và dung lượng C, trả về hit rate cho: FIFO, LRU, LFU, LFU có suy giảm, SLRU (70% bảo vệ), OPT. Sinh ba loại vết:

1. đều ngẫu nhiên trên 128 id,
2. Zipf với tham số 0,8,
3. Zipf nhưng tập "nóng" đổi sau mỗi 2 000 truy cập (mô phỏng đổi chủ đề hội thoại).

Với C = 14, 32, 64, lập bảng. Chính sách nào tốt nhất ở từng loại vết? Khoảng cách tới OPT bao nhiêu?

<details><summary>Gợi ý</summary>

OPT cài bằng cách tính trước, cho mỗi vị trí, lần truy cập kế tiếp của cùng id; khi cần loại, chọn phần tử có lần kế tiếp xa nhất (dùng heap). Kết quả thường thấy:

- Vết đều: mọi chính sách xấp xỉ C/128; OPT cao hơn chút nhờ biết tương lai.
- Zipf tĩnh: LFU và SLRU tốt hơn LRU.
- Zipf thay đổi: LFU không suy giảm tệ đi rõ (giữ expert của chủ đề cũ); LFU có suy giảm và SLRU thích nghi tốt.

Đây là lý do engine dùng heat **có suy giảm** và chia đoạn.
</details>

### Bài 4 ★★ — Inclusive hay victim

RAM chứa được 14 expert, VRAM chứa được 6 expert (cùng một lớp). Truy cập theo vết Zipf ở bài 3. Mô phỏng hai chính sách:

- Inclusive: một expert có thể ở cả hai tầng; VRAM giữ 6 expert được dùng gần nhất.
- Victim: expert bị loại khỏi RAM rơi xuống VRAM; lấy từ VRAM thì chuyển lên RAM và xóa khỏi VRAM.

Đếm: số lần phải đọc tệp, số lần chép VRAM → RAM, số lần chép RAM → VRAM. Nếu đọc tệp tốn 2,4 ms, mỗi lần chép tốn 0,4 ms, chính sách nào nhanh hơn? Nếu đọc tệp chỉ tốn 0,5 ms (page cache)?

<details><summary>Gợi ý</summary>

Victim có dung lượng hiệu dụng 20 so với 14 của inclusive, nên ít đọc tệp hơn, nhưng mỗi lần lấy từ VRAM đều phải chép hai chiều. Khi đọc tệp rẻ (page cache), chi phí chép có thể vượt phần tiết kiệm. Đây là phiên bản thu nhỏ của kết quả đồ án: tầng VRAM với expert tính trên CPU chậm hơn tắt tầng, vì checkpoint nằm trong page cache nên "đọc tệp" vốn đã rẻ, trong khi mỗi request phải chép 33–41 GiB qua PCIe.
</details>

### Bài 5 ★★★ — Đánh giá trên vết thật

Bật ghi vết định tuyến của engine bằng cách đặt `MOE_ROUTE_TRACE=/đường/dẫn/route.trace` trước khi chạy (xem `RouteTraceLogger` trong `src/olmoe.rs` để biết định dạng từng dòng), chạy một hội thoại vài lượt với OLMoE. Chuyển vết thành chuỗi truy cập theo từng lớp, rồi chạy bộ mô phỏng của bài 3 với C = 10 (đúng 64 MiB/lớp của OLMoE). So hit rate của mô phỏng LRU, SLRU, OPT với hit rate mà engine báo cáo cho cùng hội thoại.

<details><summary>Gợi ý</summary>

Hit rate của engine không hoàn toàn trùng với mô phỏng vì engine tính hit theo cả prefill (nhiều token dùng chung expert được tính là hit) và có pool VRAM. Hãy mô phỏng riêng phần decode trước. Nếu OPT chỉ cao hơn chính sách hiện tại vài phần trăm, cải tiến chính sách không đáng; cần tăng dung lượng hoặc giảm kích thước expert (INT4). Đây là cách lập luận dựa trên dữ liệu trước khi bỏ công tối ưu.
</details>

### Bài 6 ★★★ — Chia ngân sách

Một máy có trần 10 GiB cho engine. Dense và phần cố định chiếm 2,7 GiB. Phần còn lại chia giữa cache expert (C) và page cache (P). Giả sử:

- hit rate cache expert `h(C)` tăng theo C nhưng chậm dần (ví dụ `h = 1 − exp(−C/3)` với C tính bằng GiB),
- lần miss được page cache phục vụ với xác suất `P/30` (checkpoint 30 GB), tốn 0,5 ms; còn lại đọc SSD tốn 2,4 ms.

Viết hàm thời gian trung bình mỗi lượt expert theo C, tìm C tối ưu. Kết quả có giải thích được vì sao 64 và 96 MiB/lớp cho tốc độ gần như nhau không?

<details><summary>Gợi ý</summary>

Vẽ đồ thị thời gian theo C. Thường có một vùng khá phẳng quanh điểm tối ưu: tăng C thì giảm miss nhưng mỗi miss đắt hơn, hai hiệu ứng gần như bù nhau. Với 48 lớp, 64 MiB/lớp là 3 GiB, 96 MiB/lớp là 4,5 GiB; nếu cả hai nằm trong vùng phẳng, đo sẽ thấy "không nhất quán", đúng như hồ sơ `2026-11-18-qwen3-cache-ab-i5`. Mô hình đơn giản này cũng cho thấy cách thoát: thêm bộ nhớ ngoài trần (VRAM) hoặc giảm kích thước miss (INT4).
</details>

## Tài liệu đọc thêm

- L. A. Belady, *A study of replacement algorithms for a virtual-storage computer* (1966).
- Megiddo & Modha, *ARC: A Self-Tuning, Low Overhead Replacement Cache* (2003).
- Einziger, Friedman, Manes, *TinyLFU: A Highly Efficient Cache Admission Policy* (2017).
- Tài liệu chính sách eviction của Redis.
- Remzi & Andrea Arpaci-Dusseau, *Operating Systems: Three Easy Pieces*, chương "Beyond Physical Memory: Policies".
