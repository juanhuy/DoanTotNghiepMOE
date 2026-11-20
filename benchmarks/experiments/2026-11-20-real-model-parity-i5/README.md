# Parity logits trên model thật: Qwen3-30B-A3B và OLMoE-1B-7B, i5, 20/11/2026

## Câu hỏi và protocol

Trên checkpoint thật, engine có tính đúng như Transformers không, và lượng tử hóa expert INT8 làm đầu ra lệch bao nhiêu so với model BF16 gốc? Trước đây parity chỉ được chứng minh trên fixture nhỏ ngẫu nhiên. Protocol [`real-model-parity-v1`](../../SCENARIOS.md#parity-logits-trên-model-thật-real-model-parity-v1-2011) được commit trước khi đo (b6f6263).

- **Bản tham chiếu chạy từng layer** (`tools/stream_reference.py`): mỗi decoder layer được dựng bằng class Transformers 4.51.3 (eager, F32), nạp trọng số riêng layer đó, chạy mọi chuỗi rồi giải phóng. RAM đỉnh khoảng 2,8GB (Qwen3) và 2,0GB (OLMoE) thay vì 57GB/14GB.
- **Kiểm công cụ trước khi dùng:** ra đúng từng bit logits của Transformers chạy cả model trên 5 fixture, và khớp trong 1e-5 trên model ngẫu nhiên nhỏ OLMoE/Qwen3 với expert BF16 lẫn INT8 (`tools/test_stream_reference.py`).
- **Đầu vào:** 6 chuỗi cố định (`tools/make_parity_ids.py`): 4 prompt chat theo template của checkpoint (Qwen3 thinking off), một đoạn tiếng Anh và một đoạn code. Qwen3 có 159 vị trí, OLMoE 186 vị trí; so logits ở mọi vị trí trên toàn vocab (151.936 / 50.304).
- **Ba bộ logits:**
  - `engine`: `examples/dump_logits`, CPU, cache 64MiB/layer. Qwen3 dùng dense BF16 và embedding từ đĩa như benchmark; OLMoE dùng dense F32 mặc định.
  - `ref-int8`: Transformers, expert giải nén từ đúng checkpoint INT8 mà engine dùng.
  - `ref-bf16`: Transformers, toàn bộ trọng số BF16 gốc.
- **Rule phép so 1 (engine với ref-int8):** đạt nếu top-1 trùng ở mọi vị trí có biên ≥ 0,01 và KL trung bình ≤ 1e-4. Phép so 2 (ref-bf16 với engine) chỉ mô tả.

HEAD 2a50149 (b6f6263 cộng bản sửa bộ nhớ của script tham chiếu, xem bên dưới). Binary `dump_logits` sha256 `89160516b45f65b6…`. i5-10500H, 15,4GiB RAM, Linux 7.2.11, CPU-only. torch 2.15.0+cpu, Python 3.14.

## Kiểm nguồn checkpoint

- **Qwen3:** dùng lại checkpoint của `2026-11-17-qwen3-ram-cap-i5`: revision `ad44e777…`, SHA 16 shard khớp Hub, i7 convert độc lập trùng 478/478 file.
- **OLMoE:**
  - Tải `allenai/OLMoE-1B-7B-0924-Instruct` revision `7f1c97f440f06ce36705e4f2b843edb5925f4498`; SHA256 3 shard khớp LFS của Hub ([source-checkpoint-sha256.json](olmoe/source-checkpoint-sha256.json)). `tokenizer.json` trùng hash `b1fb1517…` của checkpoint i5.
  - Convert lại bằng `tools/prepare_olmoe.py` rồi so từng tensor với `../Models/olmoe_merged`, checkpoint của toàn bộ thí nghiệm OLMoE: **2195/2195 tensor trùng dtype, shape và byte** ([checkpoint-tensor-compare.json](olmoe/checkpoint-tensor-compare.json)).
  - Như vậy mọi hồ sơ OLMoE đã dùng đúng checkpoint này, khép lại điểm "chưa xác minh toàn tensor" ghi trong PROGRESS 24/10.

## Kết quả

### Phép so 1: engine tính đúng (engine với ref-int8): ĐẠT cả hai model

| Model | Vị trí | Top-1 trùng | Top-5 overlap | max \|Δlogit\| | mean \|Δlogit\| | KL trung bình / max |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-30B-A3B (48 layer, 128 expert) | 159 | 159/159 | 1,000 | 3,0e-4 | 3,9e-6 | 1,2e-11 / 1,2e-9 |
| OLMoE-1B-7B (16 layer, 64 expert) | 186 | 186/186 | 1,000 | 3,5e-5 | 2,3e-6 | 3,7e-12 / 2,1e-10 |

- Sai khác chỉ còn ở mức làm tròn F32 do thứ tự cộng khác nhau, tích lũy qua 48 hoặc 16 layer. Không vị trí nào có \|Δ\| > 0,1, và không có router nào đổi lựa chọn expert đủ để thấy trên logits.
- Engine tất định: binary build lại từ HEAD cho logits Qwen3 giống từng byte với lượt đầu (`engine/` và `engine-rerun/` trong `qwen3/logits.sha256`).

### Phép so 2: ảnh hưởng của INT8 (ref-bf16 với engine), chỉ mô tả

| Model | Top-1 trùng | Top-5 overlap | KL trung bình | KL max | max \|Δlogit\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B | 157/159 (98,7%) | 0,945 | 0,0089 | 0,381 | 4,50 |
| OLMoE-1B-7B | 184/186 (98,9%) | 0,972 | 0,0077 | 0,841 | 3,15 |

- **Bốn vị trí top-1 đổi đều là trường hợp gần hòa** (biên của BF16 từ 0,004 đến 0,19 logit):
  - Qwen3, "What is love ?": token mở đầu câu trả lời là "Love" (BF16) hay "The" (INT8), biên 0,19.
  - Qwen3, đoạn Apollo: "196**9**" hay "196**3**", biên 0,15.
  - OLMoE, giữa template chat "<|": "ans" hay "•", biên 0,004. Đây không phải vị trí model sinh câu trả lời.
  - OLMoE: "first **person**" hay "first **human**", biên 0,011.
- KL trung bình khoảng 0,008–0,009 nat/token, nghĩa là phân phối gần như giữ nguyên. Lệch lớn nhất rơi vào các vị trí đầu câu trả lời, nơi nhiều token gần nhau.
- Lệch này đến từ lượng tử hóa, không phải từ engine: `ref-int8` so với `ref-bf16` (thuần Transformers) cho đúng các con số trên ([ref-int8-vs-ref-bf16.json](qwen3/ref-int8-vs-ref-bf16.json)).

## Diễn giải

- **Đóng giới hạn "chưa có parity trên model thật"** của các hồ sơ Qwen3 và OLMoE. Engine Rust cho cùng logits với Transformers trên hai checkpoint thật, đến sai số làm tròn F32, ở mọi vị trí của bộ prompt.
- Chất lượng giữ được ở mức logits: INT8 theo hàng chỉ đổi top-1 ở các vị trí gần hòa (khoảng 1,2%), KL khoảng 0,008.
- Phương pháp chạy tham chiếu từng layer cho phép kiểm model lớn hơn RAM nhiều lần (57GB trên máy 15GB) mà không cần GPU hay máy chủ.

## Giới hạn

- Chỉ 6 chuỗi ngắn (159/186 vị trí), một lượt, CPU-only. Không đo perplexity hay benchmark chất lượng chuẩn; top-1/KL là độ lệch so với BF16, không phải độ chính xác tác vụ.
- Tham chiếu tính bằng F32 trên trọng số BF16, chưa so với Transformers chạy BF16 thuần (cách chạy thông thường trên GPU).
- Engine dùng đường `model.logits` (từng token, cùng đường với test fixture). Đường prefill theo lô và GPU expert compute không nằm trong phép so này. GPU đã có test riêng: logits < 1e-4 so với CPU.
- **Script tham chiếu sửa sau khi commit protocol:** lượt Qwen3 đầu tiên bị OOM killer dừng ở layer 22, vì handle `safe_open` giữ lại tensor đã đọc (+614MiB/layer). Bản sửa (2a50149) chỉ đổi cách mở file, không đổi phép tính; `test_stream_reference.py` vẫn đạt. Mọi kết quả trên đều chạy bằng bản đã sửa.

## Bảo toàn payload

Logits thô (khoảng 480MB) không commit; nằm ngoài repo tại `../parity-logits/`, kèm sha256 trong `qwen3/logits.sha256` và `olmoe/logits.sha256`. Repo giữ ids, metadata engine/tham chiếu, log và kết quả `compare_logits.py`. Toàn bộ khóa bằng CHECKSUMS.sha256 (không gồm README).

```bash
sha256sum -c CHECKSUMS.sha256
```
