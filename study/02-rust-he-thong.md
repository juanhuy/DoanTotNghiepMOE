# Module 02 — Rust cho lập trình hệ thống và hiệu năng

## Mục tiêu

Hiểu dữ liệu nằm trong bộ nhớ ra sao, CPU thực thi vòng lặp như thế nào, vì sao một vòng lặp nhanh hơn vòng khác gấp 5 lần dù cùng số phép tính, cách chạy song song an toàn, và cách Rust gọi thư viện C (CUDA) qua FFI.

**Dùng khi đi làm:** tối ưu backend có độ trễ thấp, cơ sở dữ liệu, game engine, xử lý tín hiệu, hạ tầng AI (viết kernel, inference server), hệ thống nhúng. Phỏng vấn vị trí systems/performance engineer hỏi rất nhiều về cache CPU, SIMD và đồng bộ luồng.

## 1. Stack, heap và bố cục bộ nhớ

- **Stack:** vùng nhớ cho biến cục bộ, cấp và trả cực nhanh (chỉ dịch một con trỏ), kích thước phải biết lúc biên dịch, giới hạn khoảng 8 MB mỗi luồng.
- **Heap:** vùng nhớ cấp phát động (`Vec`, `Box`, `String`), kích thước tùy ý, nhưng cấp và trả tốn hơn vì phải qua bộ cấp phát (allocator).

Một `Vec<f32>` gồm ba trường nằm trên stack, mỗi trường 8 byte, và một vùng dữ liệu trên heap:

```
stack:  [ ptr | len | capacity ]
            │
heap:       └──► [ f32 f32 f32 f32 ... (capacity phần tử) ]
```

- `len`: số phần tử đang dùng; `capacity`: số phần tử đã cấp chỗ.
- Khi `push` vượt `capacity`, Vec cấp vùng mới gấp đôi, chép dữ liệu sang, rồi trả vùng cũ. Vì vậy nên dùng `Vec::with_capacity(n)` khi biết trước kích thước.
- `&[f32]` (slice) là "con trỏ béo": gồm `ptr` và `len`. Lát cắt `&v[10..20]` không sao chép gì, chỉ tạo một cặp (ptr, len) mới.

**Trong đồ án:** `Int8Expert` giữ trọng số trong `Vec<u8>` và hệ số tỉ lệ trong `Vec<f32>`. Các phép tính nhận lát cắt `&self.weights[..n]`, `&self.weights[n..2*n]` cho từng ma trận gate/up/down mà không sao chép (`src/int8_expert.rs`, hàm `forward_reuse`).

## 2. Bộ cấp phát và vì sao RSS không giảm

Khi chương trình `free` một vùng nhớ, bộ cấp phát (glibc malloc trên Linux) thường **giữ lại** vùng đó để dùng cho lần cấp sau, thay vì trả ngay cho hệ điều hành. Hệ quả:

- Bộ nhớ thường trú (RSS, xem module 08) của tiến trình có thể không giảm sau khi giải phóng nhiều dữ liệu.
- Cấp rồi trả liên tục các khối lớn với kích thước khác nhau gây **phân mảnh**: còn nhiều chỗ trống nhỏ nhưng không có chỗ đủ lớn liền nhau, nên RSS tăng dần.

Hai cách xử lý trong code hiệu năng cao:

1. **Tái sử dụng bộ đệm.** Engine giữ danh sách `recycled_buffers` theo từng lớp; khi một expert bị loại khỏi cache, vùng nhớ của nó được giữ lại để nạp expert tiếp theo (`acquire_expert_tiered` trong `src/olmoe.rs` gọi `layer.recycled_buffers.pop()`). Mỗi expert cùng kích thước nên vùng nhớ khớp hoàn hảo.
2. **Bộ đệm tạm dùng lại (scratch).** `ExpertScratch` và `StepScratch` được cấp một lần và dùng cho mọi token.

Đồ án từng có hồ sơ đo riêng về hiện tượng này (`benchmarks/experiments/2026-11-08-allocator-i5/`): RSS tăng khoảng 284 MiB qua 12 request dù bộ đếm của allocator chỉ tăng 0,23 MiB. Phần tăng là bộ nhớ allocator giữ lại, không phải rò rỉ (leak).

## 3. CPU thực thi vòng lặp như thế nào

### 3.1 Pipeline và chuỗi phụ thuộc

CPU hiện đại thực hiện nhiều lệnh chồng lên nhau (pipeline) và nhiều lệnh cùng lúc (superscalar). Một phép nhân–cộng số thực (FMA) mất khoảng 4 chu kỳ mới có kết quả (độ trễ), nhưng CPU có thể **bắt đầu** 2 phép FMA mới mỗi chu kỳ (thông lượng), miễn là chúng không phụ thuộc nhau.

Xét tích vô hướng viết ngây thơ:

```rust
let mut sum = 0.0;
for i in 0..n { sum += a[i] * b[i]; }
```

Mỗi vòng cần kết quả `sum` của vòng trước. Đây là một **chuỗi phụ thuộc** dài n bước, nên mỗi bước phải chờ 4 chu kỳ. CPU gần như rảnh.

Cách sửa: dùng nhiều bộ cộng độc lập.

```rust
let mut s = [0.0f32; 8];
for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
    for lane in 0..8 { s[lane] += x[lane] * y[lane]; }
}
let sum: f32 = s.iter().sum();
```

Tám bộ cộng chạy song song, che được độ trễ. Đây chính là `src/backend.rs::dot_product`. Comment trong code giải thích: *"Independent accumulators shorten the dependency chain and allow LLVM to vectorize"*.

**Lưu ý về kết quả:** cộng số thực không có tính kết hợp, `(a+b)+c` có thể khác `a+(b+c)` ở bit cuối. Đổi thứ tự cộng làm kết quả lệch rất nhỏ. Đây là lý do engine và Transformers lệch nhau cỡ 10⁻⁵ dù cùng thuật toán (module 12).

### 3.2 SIMD

SIMD (Single Instruction, Multiple Data) là lệnh xử lý nhiều số cùng lúc trong một thanh ghi rộng:

| Tập lệnh | Độ rộng thanh ghi | Số f32 mỗi lệnh |
| --- | --- | --- |
| SSE | 128 bit | 4 |
| AVX2 | 256 bit | 8 |
| AVX-512 | 512 bit | 16 |

Có hai cách để có SIMD:

1. **Tự vector hóa:** viết vòng lặp đơn giản, độ dài cố định (như `chunks_exact(8)` ở trên) để trình biên dịch LLVM tự sinh lệnh SIMD. Ưu điểm là code di động, chạy được trên mọi CPU.
2. **Intrinsics:** gọi trực tiếp lệnh SIMD. Nhanh và chắc chắn hơn, nhưng gắn với một kiến trúc và phải dùng `unsafe`.

**Trong đồ án:** `src/int8_expert.rs` có hàm `dot_i8_f32_avx2` dùng intrinsics AVX2 để tính tích vô hướng giữa một hàng INT8 và vector FP32. Với mỗi khối 8 phần tử:

```rust
let val0 = std::ptr::read_unaligned(q_ptr as *const i64); // đọc 8 byte INT8
let q0_i8 = _mm_cvtsi64_si128(val0);         // đưa vào thanh ghi 128 bit
let q0_i32 = _mm256_cvtepi8_epi32(q0_i8);    // mở rộng 8 số i8 → 8 số i32 (256 bit)
let q0_f32 = _mm256_cvtepi32_ps(q0_i32);     // đổi sang 8 số f32
let x0 = _mm256_loadu_ps(x_ptr);             // đọc 8 số f32 của đầu vào
sum0 = _mm256_fmadd_ps(q0_f32, x0, sum0);    // sum0 += q * x, 8 làn cùng lúc
```

Hàm xử lý 32 phần tử mỗi vòng bằng 4 thanh ghi tổng (`sum0..sum3`), kết hợp cả hai kỹ thuật: SIMD và nhiều bộ cộng độc lập. Cuối cùng là bước **giảm ngang** (horizontal reduction) cộng 8 làn của thanh ghi thành 1 số.

Hàm được đánh dấu `#[target_feature(enable = "avx2,fma")]` và chỉ được gọi sau khi kiểm tra CPU lúc chạy:

```rust
if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
    return unsafe { dot_i8_f32_avx2(row, input) };
}
dot_i8_f32_scalar(row, input)   // CPU cũ: dùng bản thường
```

Mẫu "phát hiện tính năng lúc chạy, có bản dự phòng" là cách chuẩn để một binary chạy được trên mọi máy x86.

### 3.3 Tính toán hay bộ nhớ là giới hạn?

Nhân ma trận với vector (matvec) INT8 đọc mỗi byte trọng số **đúng một lần** và chỉ làm 1 phép nhân–cộng với nó. CPU có thể làm hàng chục phép tính trong thời gian đọc 1 byte từ RAM, nên phần lớn thời gian là chờ dữ liệu. Ta nói bài toán **bị giới hạn bởi băng thông bộ nhớ** (memory-bound). Module 08 trình bày mô hình roofline để tính chính xác điều này.

Hệ quả thực tế đã đo trong đồ án:

- Tăng số luồng từ 4 lên 8, 12, 24 **không** làm tính expert nhanh hơn trên cả hai máy (hồ sơ `2026-11-12-threads-i5`, `2026-11-13-threads-i7`), vì băng thông RAM đã bão hòa.
- Kernel nhiều token (`src/int8_expert/batch_prototype.rs`) đọc mỗi hàng trọng số một lần cho 2–4 token, nên nhanh hơn khoảng 1,4 lần. Số phép tính không đổi, chỉ số byte phải đọc giảm.

## 4. Đồng thời và song song

### 4.1 Send và Sync

- `Send`: giá trị được phép **chuyển** sang luồng khác.
- `Sync`: giá trị được phép **chia sẻ tham chiếu** giữa các luồng.

Trình biên dịch tự suy ra hai tính chất này. Con trỏ thô (`*mut T`) không có cả hai, nên struct chứa con trỏ thô (ví dụ con trỏ tới bộ nhớ GPU) phải tự khai báo `unsafe impl Send` và `unsafe impl Sync`. Lập trình viên chịu trách nhiệm đảm bảo điều đó đúng. Xem `src/gpu.rs`: `unsafe impl Send for GpuMatrix {}`.

### 4.2 Chia sẻ dữ liệu giữa các luồng

| Công cụ | Khi nào dùng |
| --- | --- |
| `Arc<T>` | Nhiều luồng cùng **đọc** một giá trị. Đếm tham chiếu nguyên tử, giải phóng khi người cuối buông. |
| `Mutex<T>` | Nhiều luồng cần **ghi**; mỗi lúc chỉ một luồng giữ khóa. |
| `RwLock<T>` | Đọc nhiều, ghi ít: nhiều người đọc cùng lúc hoặc một người ghi. |
| Atomic (`AtomicBool`, `AtomicU64`...) | Cờ hoặc bộ đếm đơn giản, không cần khóa. |
| Channel (`mpsc`) | Truyền thông điệp giữa luồng thay vì chia sẻ bộ nhớ. |

**Trong đồ án** (`src/engine.rs`):

- Mô hình nằm trong `Arc<Mutex<OlmoeModel>>`: luồng suy luận khóa để dùng.
- Trạng thái cho giao diện nằm trong `Arc<RwLock<Option<ExpertStatusReport>>>`, để trang web đọc trạng thái mà không phải chờ khóa mô hình trong lúc suy luận đang chạy.
- Kết quả từng token được gửi qua channel có giới hạn sang luồng HTTP (module 03).

### 4.3 Rayon: song song dữ liệu

Rayon biến iterator thường thành song song bằng cách thêm `par_`:

```rust
use rayon::prelude::*;

// nhân ma trận (rows × cols) với vector, mỗi hàng một tác vụ
output.par_iter_mut()
    .zip(matrix.par_chunks_exact(cols))
    .for_each(|(out, row)| *out = dot(row, input));
```

Rayon dùng một **thread pool** với cơ chế **work stealing**: mỗi luồng có hàng đợi tác vụ riêng; luồng rảnh "lấy trộm" tác vụ từ luồng bận. Nhờ vậy tải tự cân bằng.

Có hai điều cần nhớ:

- Song song có chi phí (chia việc, đồng bộ). Với việc nhỏ, chạy tuần tự nhanh hơn. Attention trong `src/backend.rs` chỉ chạy song song khi `num_attention_heads * positions >= 4096`.
- Song song lồng nhau (một tác vụ Rayon lại gọi Rayon bên trong) dễ làm chậm. Đồ án đã thử "dispatch song song theo nhóm" khi prefill và thấy chậm hơn 7/9 cặp, nên phương án này bị loại.

Số luồng của engine đặt bằng biến `RAYON_NUM_THREADS` (`src/backend.rs::configure_parallelism`). Kết quả đo cho thấy 4 luồng là đủ trên cả hai máy.

## 5. `unsafe`

`unsafe` không tắt kiểm tra của Rust. Nó chỉ cho phép thêm 5 việc:

1. Dereference con trỏ thô.
2. Gọi hàm `unsafe` (bao gồm hàm C qua FFI và hàm intrinsics).
3. Truy cập biến `static mut`.
4. Cài trait `unsafe` (như `Send`, `Sync`).
5. Truy cập union.

Nguyên tắc dùng `unsafe` cho đúng:

- **Giữ khối `unsafe` nhỏ nhất có thể** và bọc trong một hàm an toàn.
- **Viết comment `// SAFETY:`** giải thích vì sao điều kiện an toàn được thỏa mãn. Ví dụ trong `dot_i8_f32`: `// SAFETY: Hardware support for AVX2 and FMA was verified at runtime.`
- **Ghi bất biến (invariant)** ở đầu module. `src/gpu_expert.rs` mở đầu bằng danh sách bất biến: context CUDA phải được đặt cho luồng hiện tại, con trỏ trọng số được mượn suốt lời gọi nên pool không thể giải phóng giữa chừng, thứ tự `Drop` đảm bảo giải phóng trước khi context bị hủy.

## 6. FFI: gọi thư viện C

Rust gọi hàm C bằng khối `extern "C"`:

```rust
extern "C" {
    fn getpid() -> i32;
}

fn main() {
    let pid = unsafe { getpid() };
    println!("{pid}");
}
```

Kiểu dữ liệu phải khớp quy ước C: dùng `std::ffi::{c_int, c_char, c_void, CString, CStr}`. Chuỗi Rust không kết thúc bằng byte 0, chuỗi C thì có, nên phải chuyển qua `CString`.

### Nạp thư viện động bằng dlopen

Đồ án không liên kết cứng với CUDA lúc build. Thay vào đó, `src/gpu.rs` dùng `dlopen` để mở `libcuda.so.1` lúc chạy, rồi `dlsym` để lấy địa chỉ từng hàm:

```rust
handle = dlopen(b"libcuda.so.1\0".as_ptr() as *const c_char, RTLD_LAZY | RTLD_LOCAL);
if handle.is_null() { return None; }      // không có driver NVIDIA → chạy CPU
let ptr = dlsym(handle, b"cuInit\0".as_ptr() as *const c_char);
let cu_init: unsafe extern "C" fn(c_int) -> CUresult = std::mem::transmute(ptr);
```

Lợi ích:

- Binary build được và chạy được trên máy **không có** CUDA, chỉ đơn giản là không dùng GPU.
- Không cần bộ công cụ CUDA khi build.

Cùng kỹ thuật được dùng cho cuBLAS và NVRTC (`src/gpu_expert.rs`). Đây cũng là cách PyTorch, game engine và nhiều ứng dụng nạp plugin hoặc driver tùy chọn.

`std::mem::transmute` diễn giải lại bit của một giá trị thành kiểu khác. Ở đây nó biến con trỏ `void*` thành con trỏ hàm có chữ ký cụ thể. Nếu chữ ký sai, chương trình hỏng một cách không đoán trước được, nên đây là chỗ phải kiểm tra kỹ với tài liệu của thư viện C.

## 7. Đo và profile code Rust

- **Đo một đoạn code:** `std::time::Instant::now()` và `.elapsed()`. Dùng `std::hint::black_box(x)` để trình biên dịch không tối ưu bỏ đoạn code đang đo.
- **Benchmark nghiêm túc:** crate `criterion` chạy nhiều lần, tính thống kê, phát hiện thay đổi.
- **Profile:** `perf record -g ./target/release/app` rồi `perf report`, hoặc `cargo flamegraph` để thấy hàm nào chiếm thời gian.
- **Xem mã máy:** `cargo asm` hoặc https://godbolt.org để kiểm tra vòng lặp có được vector hóa không (tìm lệnh `vfmadd...ps` với thanh ghi `ymm`).

Module 11 trình bày phương pháp đo đầy đủ.

---

## Bài tập

### Bài 1 ★ — Vec và capacity

Viết chương trình `push` 1 000 000 số vào một `Vec<u32>` rỗng, in ra mỗi lần `capacity` thay đổi. Có bao nhiêu lần cấp phát lại? Làm lại với `Vec::with_capacity(1_000_000)`.

<details><summary>Lời giải</summary>

```rust
fn main() {
    let mut v = Vec::new();
    let mut last = v.capacity();
    let mut grows = 0;
    for i in 0..1_000_000u32 {
        v.push(i);
        if v.capacity() != last { grows += 1; last = v.capacity(); println!("{last}"); }
    }
    println!("số lần cấp lại: {grows}");
}
```

Capacity tăng theo cấp số nhân (4, 8, 16, ...), khoảng 19–20 lần cấp lại cho 1 triệu phần tử, mỗi lần chép toàn bộ dữ liệu cũ. Với `with_capacity`: 0 lần cấp lại.
</details>

### Bài 2 ★★ — Chuỗi phụ thuộc

Viết hai hàm tích vô hướng cho hai vector `f32` dài 4096: `dot_naive` (một bộ cộng) và `dot_8acc` (8 bộ cộng, `chunks_exact(8)`). Đo mỗi hàm 100 000 lần bằng `Instant` và `black_box`, build `--release`. Nhanh hơn bao nhiêu lần? Kết quả hai hàm có bằng nhau từng bit không, và vì sao?

<details><summary>Lời giải</summary>

```rust
use std::{hint::black_box, time::Instant};

fn dot_naive(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

fn dot_8acc(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0.0f32; 8];
    for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for l in 0..8 { s[l] += x[l] * y[l]; }
    }
    s.iter().sum()
}

fn main() {
    let a: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.002).cos()).collect();
    for (name, f) in [("naive", dot_naive as fn(&[f32], &[f32]) -> f32), ("8acc", dot_8acc)] {
        let t = Instant::now();
        let mut r = 0.0;
        for _ in 0..100_000 { r = f(black_box(&a), black_box(&b)); }
        println!("{name}: {:?} ({r})", t.elapsed());
    }
}
```

Thường thấy `8acc` nhanh hơn 4–8 lần. Hai kết quả lệch nhau ở vài chữ số cuối, vì thứ tự cộng khác nhau và cộng số thực không có tính kết hợp.

Thử thêm: build với `RUSTFLAGS="-C target-cpu=native"` để LLVM được phép dùng AVX2, khoảng cách còn lớn hơn.
</details>

### Bài 3 ★★ — Kernel INT8

Viết `dot_i8_f32(row: &[i8], x: &[f32]) -> f32` theo hai cách: vòng lặp thường và 8 bộ cộng. So sánh với `dot_i8_f32_scalar` trong `src/int8_expert.rs`: hàm đó dùng bao nhiêu bộ cộng, xử lý phần dư không chia hết thế nào?

<details><summary>Lời giải</summary>

`dot_i8_f32_scalar` dùng 4 mảng `[f32; 8]`, tức 32 bộ cộng độc lập, xử lý 32 phần tử mỗi vòng. Phần dư được xử lý hai tầng: các khối 8 phần tử còn lại cộng vào `s_rem`, sau đó dưới 8 phần tử cuối được cộng từng cái vào `sum`. Cấu trúc này khớp với bản AVX2, nên trình biên dịch có thể vector hóa ngay cả bản "scalar".
</details>

### Bài 4 ★★ — Rayon và số luồng

Viết hàm nhân ma trận `4096 × 4096` (f32) với vector bằng Rayon (`par_chunks_exact`). Đo thời gian với `RAYON_NUM_THREADS` = 1, 2, 4, 8 và tối đa số luồng máy bạn có. Vẽ đồ thị tốc độ theo số luồng. Từ bao nhiêu luồng thì không còn nhanh lên? Tính băng thông đạt được (số byte ma trận chia cho thời gian) và so với băng thông RAM của máy.

<details><summary>Lời giải</summary>

Ma trận 4096 × 4096 f32 = 64 MiB. Với laptop dùng DDR4 hai kênh, băng thông thực tế khoảng 20–40 GB/s, nên một lần matvec mất tối thiểu khoảng 2–3 ms dù có bao nhiêu nhân. Thường tốc độ tăng gần tuyến tính đến 2–4 luồng rồi đi ngang. Đó chính là hiện tượng đồ án đo được ở thí nghiệm số luồng (A3): thêm luồng không giúp vì băng thông RAM đã bão hòa.

```rust
use rayon::prelude::*;
fn matvec(m: &[f32], x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    out.par_iter_mut().zip(m.par_chunks_exact(cols))
        .for_each(|(o, row)| *o = row.iter().zip(x).map(|(a, b)| a * b).sum());
}
```
</details>

### Bài 5 ★★ — Arc, Mutex, RwLock

Viết chương trình có một `Arc<RwLock<Vec<u64>>>` thống kê số lần dùng 64 expert. 4 luồng ghi ngẫu nhiên vào bộ đếm 100 000 lần mỗi luồng; 1 luồng cứ 10 ms đọc và in expert được dùng nhiều nhất. Sau đó thay `RwLock<Vec<u64>>` bằng `Vec<AtomicU64>`. So sánh thời gian.

<details><summary>Gợi ý</summary>

Bản atomic thường nhanh hơn nhiều vì không có khóa: `counts[i].fetch_add(1, Ordering::Relaxed)`. `Relaxed` đủ cho bộ đếm thống kê, vì ta chỉ cần mỗi lần tăng không bị mất, không cần thứ tự giữa các biến khác nhau.
</details>

### Bài 6 ★★★ — SIMD intrinsics

Viết `dot_f32_avx2(a: &[f32], b: &[f32]) -> f32` bằng `std::arch::x86_64` với 4 thanh ghi tổng, có `#[target_feature(enable = "avx2,fma")]` và hàm bọc an toàn kiểm tra `is_x86_feature_detected!`. Viết test so với bản thường, sai số tương đối dưới 1e-5 trên 1000 vector ngẫu nhiên có độ dài lẻ (để kiểm phần dư).

<details><summary>Gợi ý</summary>

Bám sát cấu trúc của `dot_i8_f32_avx2` trong `src/int8_expert.rs`, bỏ phần đổi i8 sang f32. Dùng `_mm256_loadu_ps` (không yêu cầu căn lề) và `_mm256_fmadd_ps`. Phần giảm ngang có thể chép nguyên. Lý do test với độ dài lẻ: lỗi hay gặp nhất là quên xử lý phần dư không chia hết 32 hoặc 8.
</details>

### Bài 7 ★★★ — FFI và dlopen

Không khai báo liên kết với `libm`, hãy dùng `dlopen("libm.so.6")` và `dlsym("cos")` để gọi hàm `cos` của thư viện C, rồi so với `f64::cos` của Rust. Xử lý trường hợp không mở được thư viện bằng cách trả `None`.

<details><summary>Lời giải</summary>

```rust
use std::ffi::{c_char, c_int, c_void};

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

fn load_cos() -> Option<unsafe extern "C" fn(f64) -> f64> {
    unsafe {
        let h = dlopen(b"libm.so.6\0".as_ptr() as *const c_char, 1 /* RTLD_LAZY */);
        if h.is_null() { return None; }
        let p = dlsym(h, b"cos\0".as_ptr() as *const c_char);
        if p.is_null() { return None; }
        // SAFETY: cos trong libm có chữ ký double cos(double).
        Some(std::mem::transmute::<*mut c_void, unsafe extern "C" fn(f64) -> f64>(p))
    }
}

fn main() {
    let c = load_cos().expect("không có libm");
    let x = 1.234;
    println!("{} {}", unsafe { c(x) }, x.cos());
}
```

Đây đúng là mẫu `CudaDriverApi::load` trong `src/gpu.rs`, chỉ khác tên thư viện và số hàm.
</details>

### Bài 8 ★★★ — Tái sử dụng bộ đệm

Mô phỏng 10 000 lần "nạp expert" 6 MiB: cách A cấp `vec![0u8; 6 << 20]` mới mỗi lần rồi drop; cách B giữ một pool 8 bộ đệm và tái sử dụng. Đo thời gian và RSS đỉnh (đọc `VmHWM` trong `/proc/self/status`) của từng cách.

<details><summary>Gợi ý</summary>

Với khối lớn (trên ngưỡng `M_MMAP_THRESHOLD` của glibc, mặc định 128 KiB và có thể tự tăng lên tới vài chục MB), malloc dùng `mmap` riêng cho mỗi khối. Hệ điều hành phải cấp trang mới và xóa về 0 mỗi lần, nên cách A chậm hơn rõ. Cách B chỉ chạm vào bộ nhớ đã có. Đây là lý do engine có `recycled_buffers`.
</details>

## Tài liệu đọc thêm

- *The Rustonomicon* (unsafe Rust): https://doc.rust-lang.org/nomicon/
- *Rust Atomics and Locks* (Mara Bos), miễn phí online: https://marabos.nl/atomics/
- Ulrich Drepper, *What Every Programmer Should Know About Memory*.
- Agner Fog, *Optimizing software in C++* (các nguyên lý áp dụng nguyên cho Rust): https://www.agner.org/optimize/
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
