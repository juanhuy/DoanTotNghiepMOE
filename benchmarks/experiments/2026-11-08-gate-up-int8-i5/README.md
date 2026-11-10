# Gate–Up INT8: separate SIMD và paired SIMD trên i5

Ngày08/11/2026. Đây là phép đo **kernel tổng hợp**, không inference model7B, không INT4/FP8 và không đổi checkpoint. Candidate ghép cặp đã có từ nhánh Huy; bước này bổ sung đối chứng để đánh giá nó.

## Protocol chốt trước chạy

Theo [SCENARIOS gate-up-int8-v1](../../SCENARIOS.md):
- Shape hidden2048/intermediate1024; dữ liệu tổng hợp xorshift64v1, seeds17/73/991 theo source snapshot. Không đọc checkpoint, checkpoint=null.
- Baseline: hai dotINT8 SIMD riêng choGate vàUp; candidate: production fused_gate_up_swiglu_seq. Cùng scales, SiLU và downprojection; full_expert candidate gọi production forward_single_threaded. Baseline không phải scalar.
- Hai scope gate_up_silu/full_expert. Mỗi seed/scope10warmup/variant,9cặpAB/BA,100iterations/sample; tổng108sample=27cặp/scope.
- Một compute thread, test-threads1/RAYON_NUM_THREADS1; xác nhậnAVX2+FMA thực. Buffer cấp trước/resident; không fileI/O hoặc allocation trong timing. Không pinCPU/tần số, có thể có tải desktop/nhiệt/migration.
- Parityfinite/maxabs<2e-5 trước đo; unit test có đuôi7/32/33 vàshape2048. Không đổi tolerancefixture.
- Chỉ một lượt. Đáng thử modelAB nếu full_expert có median tỷ sốseparate/paired≥1,05 và paired thắng≥7/9cặp ở từngseed. Không lặp để tìm kết quả thuận lợi.

Lệnh preview rồi thực thi:

```bash
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --gate-up-kernel   --threads 1 --allow-dirty --dry-run
python3 tools/run_benchmark.py --testbed i5 --gpu cpu --gate-up-kernel   --threads 1 --allow-dirty
```

[Preflight](preflight.json): không có engine/build/tests liên quan; available khoảng9,61GiB/swap0. Hoàn tất checks trước đo. Runner buildrelease với --no-run hoàn tất trước khi chạy trực tiếp testbinary, không Cargo/build trong vùng timing.

## Provenance

[Record](run/result.json) run20261108T145119.570507Z-gate-up-50c0f890965b4a42a063547647aad317; HEAD5663c7a/source dirty exploratory trên branchwork/i5-gate-up-ab. CPU i5-10500H/12logicalCPU; một expert task tuần tự, không phải mô hình chạy1thread tối ưu. Hardware/rustc/Rustflags/affinity/sourceZIP/hash/binary ghi trong record. GPUinventory không khả dụng ở lượt này (record giữ nguyên trạng thái); benchmark chỉ gọi CPU và xác nhậnAVX2+FMA.

SourceZIP và testbinary khớp nguồn/binary sau đo; thay đổi tài liệu sau đo không làm đổi các sourcehash. Run này không dùng tensor thực của checkpoint, không kết luận chất lượng mô hình hoặc tính tương đương bitwise trên mọi input từ maxerror0 của bộ dữ liệu thử.

## Kết quả từng seed

Tỷ số là **trung vịseparate_elapsed/paired_elapsed tính từng cặp**; >1 là paired nhanh hơn. Wins tính từng cặp, không phải sốrequestchat.

| Seed | Gate–Up + SiLU | Toàn expert |
| --- | ---: | ---: |
| 17 | 1.236194×; 9/9 | 1.146115×; 9/9 |
| 73 | 1.251794×; 9/9 | 1.163976×; 9/9 |
| 991 | 1.242092×; 9/9 | 1.198537×; 7/9 |

Tổng Gate–Up:27/27cặp thắng, medianpairedratio1,246329×. Toàn expert:25/27cặp thắng, medianpairedratio1,160487×. Hai cặp thua nằm ởseed991; không bỏ chúng khỏi raw hay summary.

Trung vị thời gian mỗi lần gọi (trung vị riêng từng biến thể, khác phép lấytrungvịtỷsố):
| Scope | Separate SIMD (µs) | Paired SIMD (µs) |
| --- | ---: | ---: |
| Gate–Up + SiLU | 253,416 | 205,288 |
| Full expert | 390,263 | 334,648 |

[Rawkernel.json](run/kernel.json) giữ108sample và6paritycase. Maxabs error ở cả6case=0, trongtolerance2e-5. Chỉ nói maxerror0 trên dữ liệu đã thử, không chứng minh mọi đầu vào bit-exact.

## Kết luận có giới hạn

Đạt tiêu chí đã chốt: full_expert medianratio≥1,05; thắng9/9,9/9,7/9 theo seed. Candidate **đủ đáng thử đối chứng trên modelINT8 thật**. Trong phép này, ghépGate/Up có lợi ngay khi baseline cũng dùngSIMD; lợi ích giảm khi tính thêm downprojection.

Một lượt ngắn, tổngprocesswall≈3,471s, không kiểm cố định nhiệt/tần số/affinity. Gate–Up seed991 có outlier ratio tới2,134×; full_expert có cả cặp thua0,941×/0,865×. Vì vậy chỉ báo trung vị và từngseed, không khái quát tỷ lệ cố định mọi lần/máy. Dữ liệu được dùng lặp100lần trongRAM và có thể ởcacheCPU, khác tải expert từfile/mixedinference.

Không suy1,16× thành chat nhanh16%, giảmTTFT16%, giảmRAM hoặc cắt50%L1traffic: chưa đo các đại lượng đó. Chưa đánh giá4threads/expert song song, tokenrouting/I/O hay GPU. Testharness không đổi mặc địnhproduction hoặc lựa chọnINT4/FP8.

Bước tiếp: nếu làm modelAB, thêm selectorbaseline/candidate được kiểm trên cùngbinary/checkpointINT8/FP32KV/greedy/cache/backend/threads/prompt/history/tokenlimit, chỉ đổi kernel. Không so nguyên bản trướcmerge với saumerge vì có nhiều thay đổi. Chưa chạy bướcmodel đó trong lượt này.

## Payload

Chuyển nguyên5file run vào run/, thêm preflight, khóa6file bằngCHECKSUMS. Một payload duy nhất, giữ đường dẫn gốc trongrecord/log. README ngoài payload khóa; mọi raw/checksum cũ giữ nguyên.

```bash
sha256sum -c CHECKSUMS.sha256
```
