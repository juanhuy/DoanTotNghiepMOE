# Module 10 — GPU và CUDA

## Mục tiêu

Hiểu GPU khác CPU ở đâu, mô hình lập trình CUDA (grid, block, thread, warp), các tầng bộ nhớ của GPU, vì sao truyền dữ liệu qua PCIe thường là nút thắt, cách dùng cuBLAS và cách viết một kernel. Đọc hiểu được kernel INT8 SwiGLU của đồ án và giải thích được kết quả quan trọng nhất của đồ án: **tính tại chỗ quan trọng hơn chứa**.

**Dùng khi đi làm:** ML engineer, inference engineer, kỹ sư HPC, đồ họa, xử lý ảnh y tế, xe tự hành. Viết kernel CUDA/Triton là kỹ năng được trả lương rất cao. Ngay cả khi chỉ dùng PyTorch, hiểu chi phí chép CPU–GPU và cách GPU đọc bộ nhớ giúp tránh code chậm gấp 10 lần mà không biết vì sao.

## 1. CPU và GPU: hai triết lý

| | CPU | GPU |
| --- | --- | --- |
| Số nhân | 4–24 nhân mạnh | hàng nghìn nhân đơn giản |
| Tối ưu cho | độ trễ thấp của một luồng | thông lượng của hàng nghìn luồng |
| Cách che độ trễ bộ nhớ | cache lớn, dự đoán rẽ nhánh, thực thi không theo thứ tự | chuyển ngay sang nhóm luồng khác khi một nhóm phải chờ |
| Băng thông bộ nhớ | 20–80 GB/s | 100–3000 GB/s |

GPU thắng khi có rất nhiều việc giống nhau, độc lập: nhân ma trận, xử lý ảnh, attention. CPU thắng khi việc nhỏ, nhiều rẽ nhánh, hoặc dữ liệu đang nằm sẵn ở RAM.

Hai GPU trong đồ án:

| | GTX 1650 Max-Q (i5) | RTX 5060 Laptop (i7) |
| --- | --- | --- |
| VRAM | 4 GB | 8 GB |
| Kiến trúc (compute capability) | Turing (7.5) | Blackwell (12.0) |
| Tensor Core | không | có |

## 2. Mô hình lập trình CUDA

Một **kernel** là hàm chạy trên GPU, được gọi đồng thời bởi rất nhiều luồng. Các luồng được tổ chức theo hai cấp:

```
Grid (toàn bộ lần gọi kernel)
 ├── Block 0  (tối đa 1024 luồng, chạy trên một SM, có shared memory chung)
 │    ├── Warp 0: luồng 0–31   (32 luồng thực thi cùng một lệnh tại một thời điểm)
 │    ├── Warp 1: luồng 32–63
 │    └── ...
 ├── Block 1
 └── ...
```

- Mỗi luồng biết vị trí của nó qua `blockIdx.x`, `threadIdx.x`, `blockDim.x`.
- **SM** (Streaming Multiprocessor) là một cụm nhân. Mỗi block chạy trọn trên một SM.
- **Warp** 32 luồng chạy cùng một lệnh (SIMT). Nếu các luồng trong warp rẽ nhánh khác nhau (`if`), hai nhánh chạy lần lượt, lãng phí (warp divergence).

Ví dụ kernel cộng vector:

```cuda
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// gọi: add<<<(n + 255) / 256, 256>>>(a, b, c, n);
```

## 3. Các tầng bộ nhớ của GPU

| Bộ nhớ | Phạm vi | Tốc độ | Ghi chú |
| --- | --- | --- | --- |
| Thanh ghi | một luồng | nhanh nhất | dùng quá nhiều làm giảm số luồng chạy cùng lúc |
| Shared memory | một block | gần thanh ghi | các luồng trong block trao đổi dữ liệu; phải `__syncthreads()` |
| L2 cache | toàn GPU | | vài MB |
| Global memory (VRAM) | toàn GPU, và host chép vào | 100–1000 GB/s | độ trễ hàng trăm chu kỳ |

### Coalescing: quy tắc số một

Khi 32 luồng của một warp đọc 32 địa chỉ **liền nhau**, phần cứng gộp lại thành ít giao dịch bộ nhớ lớn (coalesced). Nếu đọc rải rác, mỗi luồng tốn một giao dịch riêng, băng thông hiệu dụng giảm nhiều lần.

Ví dụ với ma trận row-major `W[rows][cols]`:

- Luồng t đọc `W[row][t]`, `W[row][t + 256]`, ... (cùng hàng, cột liền nhau giữa các luồng): **coalesced**, tốt.
- Luồng t đọc `W[t][col]` (mỗi luồng một hàng): các địa chỉ cách nhau `cols` phần tử, **không coalesced**, tệ.

## 4. Host, device và PCIe

GPU rời có bộ nhớ riêng (VRAM). Dữ liệu phải được **chép** từ RAM sang VRAM qua PCIe trước khi GPU dùng được, và chép ngược lại để CPU đọc kết quả.

- PCIe 3.0 x16 khoảng 12 GB/s thực tế, PCIe 4.0 x16 khoảng 25 GB/s. Laptop thường chạy x8 hoặc thấp hơn.
- So sánh: VRAM của GPU tầm trung 100–300 GB/s. **PCIe chậm hơn VRAM 10–20 lần.**
- **Bộ nhớ ghim (pinned / page-locked):** chép từ RAM thường phải qua một bộ đệm trung gian; chép từ vùng RAM đã ghim thì GPU đọc trực tiếp (DMA), nhanh hơn và cho phép chép bất đồng bộ. Engine có `PinnedBuffer` trong `src/gpu.rs`.

**Hệ quả quan trọng nhất:** đưa một phép tính lên GPU chỉ có lợi nếu thời gian tiết kiệm được khi tính lớn hơn thời gian chép dữ liệu đi và về.

```
lợi ích = t_tính_CPU − (t_chép_lên + t_tính_GPU + t_chép_về)
```

Với matvec (decode), CPU và GPU đều bị giới hạn bởi băng thông. Nếu ma trận phải chép qua PCIe mỗi lần, GPU đọc ma trận qua PCIe (12 GB/s) còn chậm hơn CPU đọc từ RAM (20–40 GB/s). **Chỉ khi ma trận đã nằm sẵn trên VRAM**, GPU mới thắng.

## 5. Kết quả trung tâm của đồ án qua lăng kính này

Đồ án thử ba cách dùng GPU cho expert (module 06, chương 5 báo cáo):

1. **Dense trên GPU:** ma trận attention, router, LM head nằm cố định trên VRAM. Mỗi token chỉ chép vector (vài KB) lên và về. Kết quả: TTFT nhanh hơn 1,24–1,62 lần, attention nhanh hơn 9–17 lần. Đúng như lý thuyết: dữ liệu lớn đứng yên, dữ liệu nhỏ di chuyển.

2. **Tầng VRAM, expert tính trên CPU:** VRAM chứa expert, nhưng CPU tính, nên mỗi lần dùng expert ở VRAM phải chép 6 MB về RAM; theo chính sách victim, expert bị loại khỏi RAM lại được chép lên. Tổng cộng 33–41 GiB qua PCIe mỗi request 64 token. Kết quả: **chậm hơn** tắt tầng VRAM ở 8/8 so sánh, trên cả hai máy. Dữ liệu lớn bị di chuyển liên tục: sai nguyên tắc.

3. **Tính expert trên GPU:** expert có trong VRAM được tính ngay tại đó; chỉ chép vector đầu vào 8 KB lên và vector đầu ra 8 KB về. Lấp pool một lần (1,8 GB), sau đó không chép expert nữa. Kết quả: **nhanh hơn** tắt tầng ở 8/8 so sánh, trên cả hai máy.

Một câu tóm tắt để trả lời hội đồng: **"Di chuyển phép tính đến dữ liệu, đừng di chuyển dữ liệu đến phép tính."** Đây là nguyên tắc chung của hệ thống phân tán và xử lý dữ liệu lớn (MapReduce, cơ sở dữ liệu đẩy điều kiện lọc xuống tầng lưu trữ), không chỉ riêng GPU.

## 6. cuBLAS

cuBLAS là thư viện đại số tuyến tính của NVIDIA, cài sẵn các phép như GEMV (ma trận × vector) và GEMM (ma trận × ma trận) đã tối ưu cho từng kiến trúc GPU. Gần như không thể tự viết nhanh hơn cho các phép chuẩn.

Điểm hay vấp: cuBLAS theo quy ước **column-major** (theo cột, như Fortran). Ma trận row-major `A (m × n)` khi đưa cho cuBLAS được hiểu là ma trận column-major `Aᵀ (n × m)`. Muốn tính `A x` thì gọi GEMV với cờ chuyển vị `CUBLAS_OP_T`. Engine khai báo hằng `CUBLAS_OP_T` trong `src/gpu.rs` và gọi `cublasSgemv_v2` cho phần dense.

`Sgemv` là phiên bản FP32 ("S" là single precision). Đây là lý do engine phải nới dense sang FP32 trên VRAM, và với Qwen3 dense FP32 cần khoảng 4,9 GB, không vừa GTX 1650 4 GB. Dùng GEMM nửa độ chính xác (`cublasGemmEx` với BF16/FP16) sẽ giữ dense ở khoảng 2,5 GB: đó là một hướng phát triển đã nêu.

## 7. Driver API, Runtime API và NVRTC

- **Runtime API** (`cudaMalloc`, `cudaMemcpy`, cú pháp `<<<...>>>`): tiện, cần biên dịch bằng `nvcc`.
- **Driver API** (`cuInit`, `cuCtxCreate`, `cuMemAlloc`, `cuLaunchKernel`): thấp hơn, kiểm soát nhiều hơn, gọi được từ mọi ngôn ngữ có FFI.
- **NVRTC:** thư viện biên dịch mã nguồn CUDA C++ thành PTX (mã trung gian của GPU) **lúc chương trình đang chạy**.

Engine dùng Driver API và NVRTC, nạp động bằng `dlopen` (module 02). Lợi ích: build Rust không cần cài bộ công cụ CUDA, và kernel được biên dịch đúng cho GPU đang có trên máy (`compute_75` cho GTX 1650, kiến trúc khác cho RTX 5060), có bản dự phòng nếu biên dịch theo kiến trúc mới nhất thất bại. CuPy (`RawKernel`) và PyTorch (với một số tính năng JIT) dùng cùng cơ chế.

**Context:** mọi lệnh Driver API chạy trong một context gắn với luồng gọi. Bất biến đầu tiên ghi trong `src/gpu_expert.rs`: "mọi lời gọi đặt context chính của engine làm context hiện tại cho luồng gọi". Quên bước này là lỗi khó hiểu khi gọi từ luồng Rayon khác với luồng đã khởi tạo.

## 8. Đọc kernel INT8 SwiGLU của đồ án

Mã nguồn kernel nằm trong chuỗi `KERNEL_SOURCE` của `src/gpu_expert.rs`. Cấu trúc:

**Kernel 1: `moe_gate_up`**, lưới `m` block (mỗi block một hàng), 256 luồng mỗi block.

```cuda
extern "C" __global__ void moe_gate_up(const signed char* w, const float* s, const float* x,
                                       float* act, int H, int I, int vec4) {
    __shared__ float sh_g[32];
    __shared__ float sh_u[32];
    int row = blockIdx.x;
    float g = block_sum(dot_row(w + (size_t)row * H, x, H, vec4), sh_g);          // hàng gate
    float u = block_sum(dot_row(w + (size_t)(I + row) * H, x, H, vec4), sh_u);    // hàng up
    if (threadIdx.x == 0) {
        float gg = g * s[row];             // nhân hệ số tỉ lệ một lần mỗi hàng
        float uu = u * s[I + row];
        act[row] = gg / (1.0f + expf(-gg)) * uu;    // SiLU(gate) * up
    }
}
```

**Tích vô hướng của một hàng (`dot_row`):** luồng t xử lý các phần tử t, t+256, t+512, ... Nhờ vậy 32 luồng của một warp đọc 32 vị trí liền nhau: **coalesced**. Khi kích thước chia hết cho 4 (`vec4`), mỗi luồng đọc 4 byte INT8 (`char4`) và 4 số FP32 (`float4`) một lần, giảm số lệnh đọc.

**Cộng dồn trong block (`block_sum`)** hai bước:

```cuda
__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;                         // luồng 0 của warp có tổng của 32 luồng
}
```

1. Trong mỗi warp: `__shfl_down_sync` cho luồng đọc thanh ghi của luồng khác trong cùng warp, không qua bộ nhớ. 5 bước (16, 8, 4, 2, 1) gộp 32 giá trị thành 1.
2. Giữa các warp: luồng 0 của mỗi warp ghi tổng vào shared memory (`sh[wid]`), `__syncthreads()`, rồi warp 0 cộng 8 giá trị đó (256 luồng / 32 = 8 warp).

**Kernel 2: `moe_down`**, lưới `d` block, block j tính `out[j] = (W_d[j,:] · act) · s_d[j]`, cùng cấu trúc.

Sau hai kernel, chép `out` (d số FP32 = 8 KB) về RAM. Trọng số không bao giờ đi qua PCIe trên đường này.

Kiểm chứng: test `gpu_expert_compute_matches_cpu_and_skips_vram_downloads` (`src/olmoe.rs`) chạy trên GPU thật, kiểm logits khớp đường CPU dưới 10⁻⁴ và không có lượt chép VRAM → RAM nào.

## 9. Xử lý lỗi GPU

Lỗi GPU (hết VRAM, driver lỗi) không được làm sập engine. Engine có **circuit breaker** (`trip_circuit_breaker` trong `src/gpu.rs`): gặp lỗi runtime thì bật cờ, các lần sau đi đường CPU. Lúc khởi động, nếu VRAM không đủ cho dense thì tự chuyển chế độ CPU và ghi log thay vì báo lỗi. Đây là mẫu **suy giảm êm** (graceful degradation), dùng rộng rãi trong hệ thống phân tán.

## 10. Công cụ

- `nvidia-smi`: VRAM đang dùng, nhiệt độ, công suất, xung, lý do giảm xung (`nvidia-smi -q -d PERFORMANCE`). Đồ án kiểm lý do giảm xung trước mỗi lượt đo sau khi một lượt hỏng vì chạy pin.
- Nsight Systems: dòng thời gian CPU–GPU, thấy rõ lúc GPU chờ chép dữ liệu.
- Nsight Compute: phân tích chi tiết một kernel (băng thông đạt được, coalescing, occupancy).

---

## Bài tập

Các bài tập lập trình dùng CuPy (`pip install cupy-cuda12x`) trên máy có GPU NVIDIA. `cupy.RawKernel` biên dịch mã CUDA bằng NVRTC, đúng cơ chế engine dùng.

### Bài 1 ★ — Chép hay không chép

Một ma trận `768 × 2048` INT8 (1,5 MB). CPU tính matvec với băng thông RAM 25 GB/s. GPU có VRAM 150 GB/s, PCIe 10 GB/s. Bỏ qua độ trễ khởi chạy. Tính thời gian cho ba trường hợp: (a) CPU, (b) GPU khi ma trận phải chép lên mỗi lần, (c) GPU khi ma trận đã có trên VRAM. Kết luận?

<details><summary>Lời giải</summary>

(a) 1,5 MB / 25 GB/s = 60 µs. (b) chép 150 µs + tính 10 µs = 160 µs, chậm hơn CPU gần 3 lần. (c) 10 µs (cộng vài µs chép vector), nhanh hơn CPU 6 lần. Đây là toàn bộ kết luận của đồ án về tầng VRAM gói trong một phép tính.
</details>

### Bài 2 ★ — Pool VRAM

GTX 1650 có 4 GB, trong đó dense FP32 của OLMoE và vùng tính tạm chiếm phần lớn; pool chứa được 299 expert OLMoE (6,02 MiB mỗi expert). Pool chiếm bao nhiêu? Với RTX 5060 8 GB và pool 960 expert thì sao? Với Qwen3 (4,51 MiB mỗi expert), 4 GB VRAM nếu chỉ dùng cho pool chứa được khoảng bao nhiêu expert, bằng bao nhiêu phần trăm tổng số?

<details><summary>Lời giải</summary>

299 × 6,02 ≈ 1 800 MiB (đúng con số 1799 MiB lấp pool trong hồ sơ). 960 × 6,02 ≈ 5 780 MiB. Qwen3: khoảng 3 200 MiB khả dụng / 4,51 ≈ 700 expert trên 6 144, khoảng 11%. Con số 11% này là lý do báo cáo đánh giá hướng "pool VRAM cho Qwen3 trên i5" là khả thi nhưng lợi ích có giới hạn.
</details>

### Bài 3 ★★ — Băng thông PCIe

Dùng CuPy đo băng thông chép host → device với mảng 256 MB: (a) từ mảng numpy thường, (b) từ bộ nhớ ghim (`cupy.cuda.alloc_pinned_memory`). So với băng thông lý thuyết PCIe của máy (xem `nvidia-smi -q | grep -A3 "Link"`).

<details><summary>Gợi ý</summary>

```python
import cupy as cp, numpy as np, time
n = 256 << 20
a = np.ones(n, np.uint8); d = cp.empty(n, cp.uint8)
cp.cuda.Device().synchronize(); t = time.perf_counter()
d.set(a); cp.cuda.Device().synchronize()
print("pageable", n / (time.perf_counter() - t) / 1e9, "GB/s")
mem = cp.cuda.alloc_pinned_memory(n); p = np.frombuffer(mem, np.uint8, n); p[:] = 1
t = time.perf_counter(); d.set(p); cp.cuda.Device().synchronize()
print("pinned", n / (time.perf_counter() - t) / 1e9, "GB/s")
```

Chạy vài lần, bỏ lần đầu (khởi tạo). Pinned thường nhanh hơn rõ.
</details>

### Bài 4 ★★ — Kernel đầu tiên và coalescing

Viết hai kernel cùng tính tổng mỗi hàng của ma trận `4096 × 4096` float32:

- (A) mỗi **luồng** tính một hàng (đọc theo hàng, các luồng cạnh nhau đọc các hàng khác nhau);
- (B) mỗi **block** 256 luồng tính một hàng, luồng t đọc cột t, t+256, ..., rồi cộng dồn bằng `warp_sum` và shared memory như `block_sum` của engine.

Đo thời gian và băng thông đạt được của mỗi kernel. Vì sao (B) nhanh hơn nhiều?

<details><summary>Gợi ý</summary>

(A) không coalesced: 32 luồng của một warp đọc 32 địa chỉ cách nhau 16 KB. (B) coalesced: 32 luồng đọc 128 byte liền nhau. Thường thấy (B) đạt gần băng thông VRAM, (A) chậm hơn vài lần. Kernel `moe_gate_up` của engine dùng đúng cách (B).
</details>

### Bài 5 ★★★ — Tái tạo kernel của engine

Chép chuỗi `KERNEL_SOURCE` từ `src/gpu_expert.rs` vào một `cupy.RawModule`. Tạo expert ngẫu nhiên: trọng số INT8 `[gate (m×d); up (m×d); down (d×m)]`, hệ số tỉ lệ `[s_g (m); s_u (m); s_d (d)]` với `d = 2048, m = 1024`. Gọi `moe_gate_up` rồi `moe_down` với đúng kích thước lưới. So với bản numpy:

```
a = silu((Wg @ x) * s_g) * ((Wu @ x) * s_u);   out = (Wd @ a) * s_d
```

Sai số tương đối lớn nhất? Đo thời gian một lần gọi và tính băng thông đạt được (6 MB / thời gian).

<details><summary>Gợi ý</summary>

Lưới: `moe_gate_up` gọi với `grid=(m,), block=(256,)`; `moe_down` với `grid=(d,), block=(256,)`. Tham số `vec4 = 1` khi d và m chia hết cho 4. Sai số kỳ vọng cỡ 1e-6 tương đối (chỉ khác thứ tự cộng). Nếu sai số lớn, lỗi thường ở offset: ma trận down bắt đầu ở `2*I*H`, hệ số `s_d` bắt đầu ở `2*I`.

Bài này cho bạn nắm trọn phần GPU compute của đồ án: từ đây có thể tự thử cải tiến (ví dụ một block tính nhiều hàng, hay hợp nhất gate/up thành một lần đọc x) và đo.
</details>

### Bài 6 ★★★ — Khi nào nên đưa lên GPU

Viết một hàm Python `should_offload(bytes_weight, bytes_io, cpu_bw, gpu_bw, pcie_bw, resident: bool, uses: int)` trả về thời gian CPU, thời gian GPU và lựa chọn. `uses` là số lần trọng số được dùng sau khi chép lên (nếu không thường trú). Áp dụng cho: dense OLMoE (dùng mọi token), một expert dùng 1 lần, một expert dùng 50 lần trong một request. Kết quả có khớp ba thí nghiệm của mục 5 không?

<details><summary>Gợi ý</summary>

Thời gian GPU khi không thường trú: `bytes_weight/pcie_bw + uses × (bytes_weight/gpu_bw + bytes_io/pcie_bw)`; khi thường trú bỏ số hạng đầu. Thời gian CPU: `uses × bytes_weight/cpu_bw`. Expert dùng 1 lần: CPU thắng. Thường trú: GPU thắng. Dùng nhiều lần: chi phí chép được chia đều, GPU thắng từ một ngưỡng nào đó. Đây là mô hình chi phí mà mọi hệ thống offloading (llama.cpp, DeepSpeed, vLLM) đều phải giải.
</details>

## Tài liệu đọc thêm

- NVIDIA, *CUDA C++ Programming Guide* và *CUDA C++ Best Practices Guide*.
- Kirk & Hwu, *Programming Massively Parallel Processors* (4th ed.).
- NVIDIA blog, *Using Shared Memory in CUDA C/C++*, *Faster Parallel Reductions on Kepler* (warp shuffle).
- Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*.
- Tài liệu OpenAI Triton (viết kernel GPU bằng Python).
