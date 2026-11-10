# Gate–Up với trọng số INT8 thật, resident và 4 luồng

Ngày 09/11/2026. **Không đạt tiêu chí cải tiến đã đặt; khép hướng Gate–Up.** Không mở thêm lượt HTTP để tìm phần trăm tăng tốc.

Phép đo chính dùng 8 expert chia trên 4 worker như nhánh compute decode của engine: paired thắng 22/81 cặp, median tỷ số separate/paired = 0,959657×. Tức thời gian paired dài hơn khoảng4,2% theo nghịch đảo tỷ số trung vị; đây là quan sát workload hẹp, không phải kết luận mọi máy/model đều chậm hơn.

Scope phụ chia theo hàng có lợi: thắng79/81, medianratio1,136701×. Theo protocol, scope phụ không cứu kết quả chính thất bại. Không gọi đây là tăng tốc hệ thống hoặc bỏ những mẫu bất lợi.

## Protocol chốt trước đo

[gate-up-resident-v1](../../SCENARIOS.md): đúng một run; cùng weights/scales INT8 thật ở layers0/7/15, expertIDs0..7; hidden2048/intermediate1024. Cả24payload nạp trước timing,144,375MiB dưới budget256MiB; không load dense/full model/GPU.

Activations là synthetic xorshift64 seeds17/73/991, không phải hidden states từ hội thoại. Hai scope full expert gồm Gate/Up/SiLU/Down:
- expert_tasks8: tám forward_single_threaded qua Rayon4, cùng cách chia task ở đường decode; không gồm routing/reduction/attention/I/O.
- row_parallel: forward_reuse trênexpert0, chia hàng qua cùng pool4worker.

Mỗi layer/seed/scope: parityfinite/maxabs<2e-5,10warmup/variant,9cặp AB/BA luân phiên,40iteration/sample. Mode chọn ngoài timing. Tổng18caseparity/324sample,81cặp/scope. Cùng weights/input/scales/scratch, baseline hai SIMDdot chứ không scalar. Không sửa tolerance hoặc expected.

Primary phải medianratio≥1,05 và thắng≥7/9 ở mọi layer/seed; kết quả không đạt. Không pin CPU/tần số hoặc loại outlier; không coi tiêu chí kỹ thuật này là kiểm định thống kê.

Lệnh thực thi:

    python3 tools/run_benchmark.py --testbed i5 --gpu cpu --gate-up-resident --threads 4 --allow-dirty

## Từng nhóm mẫu

Tỷ số là separate/paired, >1 có lợi cho paired. Mỗi hàng có9cặp; mọi thời gian gốc và thứ tự nằm trong [kernel.json](run/kernel.json).

| Scope | Layer | Seed | Paired thắng | Median pair ratio |
| --- | ---: | ---: | ---: | ---: |
| expert_tasks8 | 0 | 17 | 4/9 | 0.992632× |
| expert_tasks8 | 0 | 73 | 3/9 | 0.961559× |
| expert_tasks8 | 0 | 991 | 4/9 | 0.996867× |
| expert_tasks8 | 7 | 17 | 0/9 | 0.958350× |
| expert_tasks8 | 7 | 73 | 1/9 | 0.922510× |
| expert_tasks8 | 7 | 991 | 5/9 | 1.009633× |
| expert_tasks8 | 15 | 17 | 2/9 | 0.959592× |
| expert_tasks8 | 15 | 73 | 3/9 | 0.974058× |
| expert_tasks8 | 15 | 991 | 0/9 | 0.956006× |
| row_parallel | 0 | 17 | 9/9 | 1.133427× |
| row_parallel | 0 | 73 | 9/9 | 1.146115× |
| row_parallel | 0 | 991 | 9/9 | 1.130390× |
| row_parallel | 7 | 17 | 9/9 | 1.174548× |
| row_parallel | 7 | 73 | 9/9 | 1.120688× |
| row_parallel | 7 | 991 | 9/9 | 1.170936× |
| row_parallel | 15 | 17 | 9/9 | 1.125780× |
| row_parallel | 15 | 73 | 8/9 | 1.105780× |
| row_parallel | 15 | 991 | 8/9 | 1.143214× |

## Tính đúng và điều kiện đo

18caseparity có maxabs0 trên dữ liệu đã thử; không chứng minh bit-exact mọi input. Dùng các phương thức production với hai mode; không thay cách chia task giữa hai variant.

[Preflight](preflight.json) ba mẫu cách5s: available ít nhất9665000KiB, swap-used0/pswpout0 và không thấy build/inference khác. [Checks trước đo](checks.log): python3 tools/check.py --rust --release --web PASS,127Rust tests/profile (108lib+6HTTP+7loader+6reference),7ignored/profile, fmt/Clippy,81Python và7Node. Có unit test gọi cả hai đường batch với hai mode trong pool4worker. Build runner hoàn tất trước chạy trực tiếp testbinary; không build/tests đồng thời timing.

Snapshot /proc trước/sau compute: VmRSS/VmHWM156180→156684KiB, VmSwap0 cảhai,6thread toànprocess (4Rayon worker, không phải engine6worker). read_bytes52277248→52277248, không phát sinh đọc storage trong vùng compute; rchar/syscr có tăng do đọc /proc. Buffers cấp trước nhưng không chứng minh Rayon không cấp phát nội bộ. Không có sampler liên tục trong run ngắn; snapshot/HWM không chứng nhận total RAM hard-cap.

## Provenance và giới hạn

[Run record](run/result.json): 20261109T023336.846352Z-gate-up-resident-7a206b80e9bf4b879479c300b97698fa; HEAD5663c7ab7c8f905edb2a218a3f524fda342b6179, dirty exploratory trên work/i5-gate-up-ab. i5-10500H/12logicalCPU/RAM16571342848B; GPUinventory unavailable, chỉ CPU/AVX2+FMA đã xác nhận.

Binary SHA-256 d3eac5baf782f890e249e3c02611b06bd1b0bd68c5f487eb83c8572cc7c4182a; 52sourcehash/ZIP và binary khớp ngay sau đo, trước khi bắt đầu sửa policy admission của phần mới. Checkpoint manifest đầy đủ trong record, config9f1312f8… trùng checkpoint INT8 của phép HTTP trước; không so thời gian giữa hai revision/scope.

Testbinary wall12,93s gồm load/chuẩn bị/harness, không phải tổng thời gian kernel. Chỉ một process, chọn ba lớp/tám expert cố định, synthetic activations; không đại diện toàn bộ expert hoặc nội dung chat.

[CHECKSUMS.sha256](CHECKSUMS.sha256) khóa7payload:5runfile,preflight vàchecks.log. Không sửa raw/source ZIP của hai phép Gate–Up08/11; kết quả mới thu hẹp điều kiện mà tín hiệu microbenchmark1thread còn có ích.

## Quyết định

Không ghi nhận Gate–Up là cải tiến tốc độ hệ thống trên i5 trong bước này. Giữ code paired mặc định đã tiếp nhận, selector vẫn phục vụ tái lập; không tự thay default theo một phép đo hẹp. Chuyển công việc sang admission cache để giảm nạp lại expert ở cùng RAM, kiểm bằng output parity và số file loads/payload. Đây là một thay đổi mới, phải có bằng chứng riêng; chưa lấy kết quả Gate–Up làm thành tích của policy.
