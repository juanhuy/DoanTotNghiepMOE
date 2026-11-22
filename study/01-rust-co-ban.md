# Module 01 — Rust cơ bản

## Mục tiêu

Sau module này bạn đọc được phần lớn code trong `src/`, hiểu vì sao trình biên dịch Rust từ chối một đoạn code, và tự viết được chương trình Rust nhỏ có test.

## 1. Vì sao lại là Rust

Rust là ngôn ngữ biên dịch, nhanh ngang C/C++, nhưng kiểm tra an toàn bộ nhớ ngay lúc biên dịch mà không cần bộ thu gom rác (GC). Ba loại lỗi kinh điển của C/C++ bị chặn từ trình biên dịch:

- **Use-after-free:** dùng vùng nhớ đã giải phóng.
- **Double free:** giải phóng một vùng nhớ hai lần.
- **Data race:** hai luồng cùng ghi vào một chỗ mà không đồng bộ.

Với engine suy luận, điều này quan trọng vì code vừa phải nhanh (vòng nhân ma trận chạy hàng tỷ lần), vừa quản lý nhiều bộ đệm lớn sống lâu (cache expert, KV), vừa chạy nhiều luồng. Viết bằng C++ thì nhanh nhưng dễ hỏng bộ nhớ; viết bằng Python thì an toàn nhưng chậm và khó kiểm soát RAM.

**Dùng khi đi làm:** Rust được dùng ở hạ tầng (Cloudflare, AWS Firecracker), cơ sở dữ liệu (TiKV, Materialize), công cụ AI (tokenizers của Hugging Face, candle, mistral.rs), blockchain, trình duyệt (Firefox), nhân Linux.

## 2. Biến, kiểu và hàm

```rust
fn main() {
    let x = 5;            // bất biến mặc định
    let mut y = 10;       // muốn thay đổi phải ghi mut
    y += x;
    let z: f32 = 1.5;     // khai kiểu tường minh
    let n = 3usize;       // usize: số nguyên không dấu, rộng bằng con trỏ (64 bit trên máy 64 bit)
    println!("{y} {z} {n}");
}

fn add(a: i32, b: i32) -> i32 {
    a + b                 // biểu thức cuối không có dấu ; là giá trị trả về
}
```

Các kiểu số hay gặp trong đồ án:

| Kiểu | Ý nghĩa | Dùng cho |
| --- | --- | --- |
| `u8` / `i8` | số nguyên 8 bit không dấu / có dấu | byte của trọng số INT8 |
| `u16` | 16 bit không dấu | bit pattern của BF16 |
| `f32` | số thực 32 bit | activation, KV, hệ số tỉ lệ |
| `usize` | kích thước, chỉ số mảng | số token, offset |
| `u64` | bộ đếm lớn | số byte đã đọc, số lần cache hit |

Chuyển kiểu bằng `as`: `q as i8 as f32` nghĩa là đọc byte `u8` thành số có dấu `i8`, rồi đổi sang `f32`. Dòng này xuất hiện trong kernel INT8 ở `src/int8_expert.rs` (hàm `dot_i8_f32_scalar`).

## 3. Ownership: luật quan trọng nhất

Mỗi giá trị trong Rust có **đúng một chủ sở hữu**. Khi chủ sở hữu ra khỏi phạm vi (scope), giá trị bị giải phóng.

```rust
fn main() {
    let a = vec![1, 2, 3];   // a sở hữu vùng nhớ trên heap
    let b = a;               // quyền sở hữu CHUYỂN sang b (move)
    // println!("{:?}", a);  // LỖI: a không còn hợp lệ
    println!("{:?}", b);
}                            // b ra khỏi scope → vùng nhớ được giải phóng đúng 1 lần
```

Tại sao lại thiết kế như vậy? Vì nếu cả `a` và `b` cùng trỏ tới một vùng nhớ, khi cả hai ra khỏi scope thì vùng nhớ bị giải phóng hai lần. Rust chọn cách "chỉ một chủ" để luôn biết chính xác ai giải phóng.

Kiểu nhỏ nằm trọn trên stack (`i32`, `f32`, `bool`, `usize`...) được **copy** thay vì move, vì copy rẻ:

```rust
let x = 5;
let y = x;      // copy, x vẫn dùng được
```

Muốn nhân đôi dữ liệu trên heap thì phải gọi `.clone()` tường minh. Nhờ vậy, mọi lần sao chép tốn kém đều hiện rõ trong code.

## 4. Borrowing: mượn thay vì lấy

Thay vì chuyển quyền sở hữu, ta có thể **mượn** bằng tham chiếu:

```rust
fn sum(values: &[f32]) -> f32 {       // &[f32]: mượn chỉ đọc một dãy f32
    values.iter().sum()
}

fn scale(values: &mut [f32], k: f32) { // &mut: mượn để ghi
    for v in values.iter_mut() {
        *v *= k;
    }
}

fn main() {
    let mut data = vec![1.0, 2.0, 3.0];
    let s = sum(&data);       // mượn đọc
    scale(&mut data, 2.0);    // mượn ghi
    println!("{s} {:?}", data);
}
```

**Luật mượn:** tại một thời điểm, chỉ được có

- nhiều tham chiếu đọc `&T`, **hoặc**
- đúng một tham chiếu ghi `&mut T`.

Không bao giờ có cả hai cùng lúc. Luật này loại bỏ data race ngay từ lúc biên dịch: nếu có người đang ghi thì không ai được đọc.

### Ví dụ trong đồ án

`src/int8_expert.rs`, hàm `forward_reuse`:

```rust
pub(crate) fn forward_reuse<'a>(
    &self,                          // mượn đọc expert (trọng số không đổi)
    input: &[f32],                  // mượn đọc vector đầu vào
    scratch: &'a mut ExpertScratch, // mượn ghi vùng nhớ tạm
) -> io::Result<&'a [f32]> {        // trả về tham chiếu vào bên trong scratch
```

Thiết kế này cho phép tính expert hàng nghìn lần mà **không cấp phát bộ nhớ mới**: vùng nhớ tạm `scratch` được tái sử dụng, kết quả là một lát cắt nằm bên trong nó.

## 5. Lifetime

Lifetime là cách Rust theo dõi "tham chiếu này còn hợp lệ đến khi nào". Phần lớn thời gian trình biên dịch tự suy ra. Khi một hàm trả về tham chiếu, đôi khi phải ghi rõ:

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

`'a` nói rằng kết quả sống không lâu hơn cả `x` và `y`. Trong `forward_reuse` ở trên, `'a` nói rằng lát cắt kết quả sống không lâu hơn `scratch`. Nhờ vậy, không thể dùng kết quả sau khi `scratch` đã bị giải phóng hoặc bị ghi đè.

**Cách đọc:** khi thấy `'a`, hãy hỏi "tham chiếu trả về đang trỏ vào dữ liệu của tham số nào?".

## 6. Struct, enum và match

```rust
struct Expert {
    id: usize,
    weights: Vec<u8>,
    scales: Vec<f32>,
}

impl Expert {
    fn bytes(&self) -> usize {
        self.weights.len() + self.scales.len() * 4
    }
}
```

`enum` trong Rust mạnh hơn trong C: mỗi biến thể có thể mang dữ liệu riêng.

```rust
enum DenseMatrix {
    F32(Vec<f32>),
    Bf16 { rows: usize, columns: usize, values: Vec<u16> },
}

fn bytes(m: &DenseMatrix) -> usize {
    match m {
        DenseMatrix::F32(v) => v.len() * 4,
        DenseMatrix::Bf16 { values, .. } => values.len() * 2,
    }
}
```

`match` bắt buộc xử lý **mọi** trường hợp. Thêm một biến thể mới mà quên xử lý ở đâu đó thì code không biên dịch được. Đây là lý do refactor trong Rust an toàn hơn.

### Ví dụ trong đồ án

`src/dense.rs` định nghĩa `DenseStorage` và đọc nó từ biến môi trường:

```rust
pub enum DenseStorage { F32, Bf16 }

impl DenseStorage {
    pub fn from_environment() -> io::Result<Self> {
        match std::env::var("MOE_DENSE_STORAGE").as_deref() {
            Ok("f32") | Err(std::env::VarError::NotPresent) => Ok(Self::F32),
            Ok("bf16") => Ok(Self::Bf16),
            _ => Err(invalid("MOE_DENSE_STORAGE must be f32 or bf16")),
        }
    }
}
```

Một dòng `match` xử lý gọn ba trường hợp: có giá trị hợp lệ, không đặt biến (dùng mặc định), và giá trị sai (báo lỗi ngay lúc khởi động thay vì chạy sai âm thầm). Đây là mẫu tốt để đọc cấu hình trong mọi dự án.

## 7. Option và Result: không có null, không có exception

Rust không có `null`. Giá trị "có thể không có" dùng `Option<T>`:

```rust
enum Option<T> { Some(T), None }

let pos: Option<usize> = cache.iter().position(|e| e.id == 5);
match pos {
    Some(i) => println!("tìm thấy ở {i}"),
    None => println!("không có"),
}
```

Lỗi dùng `Result<T, E>`:

```rust
enum Result<T, E> { Ok(T), Err(E) }
```

Toán tử `?` trả lỗi về cho hàm gọi nếu gặp `Err`:

```rust
fn read_header(path: &str) -> std::io::Result<u64> {
    let mut file = std::fs::File::open(path)?;   // lỗi mở file → trả về ngay
    let mut prefix = [0u8; 8];
    use std::io::Read;
    file.read_exact(&mut prefix)?;               // lỗi đọc → trả về ngay
    Ok(u64::from_le_bytes(prefix))
}
```

Trong đồ án gần như mọi hàm trả `io::Result<T>`, và có một hàm trợ giúp tạo lỗi, `invalid` trong `src/backend.rs`:

```rust
pub(crate) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
```

Các phương thức hay dùng với Option/Result: `ok_or_else` (đổi `None` thành lỗi), `map_err` (đổi kiểu lỗi), `unwrap_or` (giá trị mặc định), `is_some_and`.

**Quy tắc:** chỉ dùng `.unwrap()` khi chắc chắn không thể lỗi, hoặc trong test. Trong code production, `unwrap()` gặp lỗi sẽ làm sập chương trình.

## 8. Kiểm tra tràn số

Nhân hai `usize` lớn có thể tràn. Ở bản debug Rust báo lỗi khi tràn; ở bản release thì kết quả bị quấn vòng (wrap), sai một cách âm thầm. Code đọc tệp từ bên ngoài phải dùng phép tính có kiểm tra. Trong `src/safetensors.rs`:

```rust
let elements = descriptor.shape.iter()
    .try_fold(1usize, |n, d| n.checked_mul(*d))      // None nếu tràn
    .ok_or_else(|| invalid("tensor shape overflow"))?;
```

Vì sao cẩn thận như vậy? Vì header của tệp checkpoint có thể bị hỏng hoặc bị cố tình làm sai. Một shape khổng lồ nhân tràn ra số nhỏ có thể khiến chương trình cấp phát thiếu rồi đọc tràn bộ đệm. Đây là tư duy **không tin dữ liệu đầu vào**, áp dụng cho mọi code phân tích tệp hoặc gói tin mạng.

## 9. Iterator

Rust khuyến khích dùng iterator thay vòng lặp chỉ số, vì vừa gọn vừa giúp trình biên dịch bỏ kiểm tra biên và tự vector hóa:

```rust
let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
```

Một số adaptor hay gặp trong đồ án:

| Adaptor | Ý nghĩa |
| --- | --- |
| `zip` | ghép hai iterator thành cặp |
| `chunks_exact(n)` | cắt thành các khối đúng n phần tử; phần dư lấy bằng `.remainder()` |
| `enumerate` | thêm chỉ số |
| `position` | tìm chỉ số phần tử đầu tiên thỏa điều kiện |
| `fold` / `try_fold` | gộp dần thành một giá trị, `try_` cho phép dừng khi lỗi |
| `par_iter`, `par_chunks_mut` | phiên bản song song của Rayon (module 02) |

`chunks_exact(8)` là kỹ thuật quan trọng: `src/backend.rs::dot_product` dùng nó để có 8 bộ cộng độc lập, giúp CPU tính 8 phép nhân cùng lúc (giải thích ở module 02).

## 10. Trait

Trait giống interface: một tập phương thức mà kiểu phải cài đặt.

```rust
trait Shape {
    fn area(&self) -> f64;
}

struct Circle { r: f64 }
impl Shape for Circle {
    fn area(&self) -> f64 { std::f64::consts::PI * self.r * self.r }
}

fn print_area(s: &impl Shape) {         // nhận bất kỳ kiểu nào cài Shape
    println!("{}", s.area());
}
```

Trait chuẩn hay gặp:

- `Debug` (in bằng `{:?}`), `Clone`, `Copy`, `PartialEq`: thường sinh tự động bằng `#[derive(...)]`.
- `Serialize`/`Deserialize` của serde: chuyển struct sang JSON và ngược lại. `src/main.rs` dùng chúng cho request/response HTTP.
- `Drop`: chạy code khi giá trị bị hủy. `src/olmoe.rs` cài `Drop for OlmoeModel`; module 10 dùng nó để giải phóng bộ nhớ GPU đúng thứ tự.
- `Send` / `Sync`: đánh dấu kiểu được chuyển hoặc chia sẻ giữa các luồng (module 02).
- `Stream`: chuỗi giá trị bất đồng bộ (module 03). `src/engine.rs` cài `Stream for ChatStream`.

## 11. Module, crate và Cargo

- **Crate** là một đơn vị biên dịch (thư viện hoặc chương trình). Đồ án là crate `moe-tier-engine`, khai báo trong `Cargo.toml`.
- **Module** chia code trong crate. `src/lib.rs` liệt kê các module (`pub mod olmoe;`...). `src/main.rs` là chương trình server.
- **Phạm vi truy cập:** `pub` (công khai), `pub(crate)` (chỉ trong crate), không ghi gì (riêng tư trong module).
- **Dependency** khai báo trong `[dependencies]` của `Cargo.toml`. Đồ án dùng ít dependency có chủ đích: `axum`/`tokio` (web, async), `rayon` (song song), `serde` (JSON), `tokenizers` (bộ mã hóa văn bản), `minijinja` (render khuôn mẫu hội thoại).

Lệnh hay dùng:

```bash
cargo build --release        # build bản tối ưu (luôn dùng bản này khi đo tốc độ)
cargo test                   # chạy test
cargo test ten_test          # chạy test có tên chứa chuỗi
cargo clippy -- -D warnings  # kiểm tra lỗi phong cách và lỗi tiềm ẩn
cargo fmt                    # định dạng code
cargo run --release --example dump_logits -- ARG1 ARG2   # chạy một example
```

Bản debug có thể chậm hơn bản release 10–50 lần với code tính toán. Đo hiệu năng bằng bản debug là sai lầm kinh điển.

## 12. Test

```rust
fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]                 // chỉ biên dịch khi chạy test
mod tests {
    use super::*;

    #[test]
    fn adds_two_numbers() {
        assert_eq!(add(2, 3), 5);
    }
}
```

Trong đồ án có hai loại test:

- **Unit test** nằm ngay trong file (khối `mod tests` ở cuối `src/olmoe.rs`, `src/backend.rs`...).
- **Integration test** nằm trong `tests/`, ví dụ `tests/olmoe_reference.rs` so logits của engine với giá trị tham chiếu do PyTorch sinh ra (module 12).

Tên test trong đồ án mô tả hành vi, ví dụ `gpu_expert_compute_matches_cpu_and_skips_vram_downloads`. Đọc tên test là cách nhanh nhất để biết code được kỳ vọng làm gì.

---

## Bài tập

Tạo thư mục `~/study-ex/` và làm từng bài bằng `cargo new`.

### Bài 1 ★ — Ownership

Đoạn code sau có lỗi biên dịch. Giải thích lỗi và sửa theo **hai** cách khác nhau.

```rust
fn total(v: Vec<f32>) -> f32 { v.iter().sum() }

fn main() {
    let data = vec![1.0, 2.0, 3.0];
    let t = total(data);
    println!("{t} {}", data.len());
}
```

<details><summary>Lời giải</summary>

`total` nhận `Vec<f32>` theo giá trị nên `data` bị move vào hàm. Sau lời gọi, `data` không còn dùng được.

- **Cách 1 (đúng nhất):** cho hàm mượn, `fn total(v: &[f32]) -> f32`, và gọi `total(&data)`. Không sao chép, `data` vẫn thuộc `main`.
- **Cách 2:** gọi `total(data.clone())`. Chạy được nhưng tốn một lần sao chép toàn bộ vector. Với vector trọng số vài MB, đây là lãng phí lớn.

Bài học: hàm chỉ cần đọc dữ liệu thì nhận `&[T]` hoặc `&T`.
</details>

### Bài 2 ★ — Luật mượn

Vì sao đoạn này không biên dịch? Sửa lại.

```rust
fn main() {
    let mut cache = vec![10, 20, 30];
    let first = &cache[0];
    cache.push(40);
    println!("{first}");
}
```

<details><summary>Lời giải</summary>

`first` là tham chiếu đọc vào `cache`, còn `push` cần mượn ghi. Hai thứ này không được tồn tại cùng lúc. Lý do sâu xa: `push` có thể phải cấp phát vùng nhớ mới lớn hơn và chép dữ liệu sang, khi đó `first` sẽ trỏ vào vùng nhớ cũ đã giải phóng. Trong C++ đây là lỗi iterator invalidation; Rust chặn từ lúc biên dịch.

Sửa: copy giá trị ra trước (`let first = cache[0];`, vì `i32` là kiểu Copy), hoặc in trước rồi mới `push`.
</details>

### Bài 3 ★ — Enum và match

Viết `enum Precision { Int8, Int4, Bf16, F32 }` và hàm `bytes_per_weight(p: Precision) -> f32` (INT4 là 0,5 byte). Viết thêm hàm tính dung lượng một expert OLMoE (`d = 2048`, `m = 1024`, gồm 3 ma trận `d×m`) theo từng precision, **chưa tính** hệ số tỉ lệ. Viết test cho trường hợp INT8.

<details><summary>Lời giải</summary>

```rust
#[derive(Clone, Copy)]
enum Precision { Int8, Int4, Bf16, F32 }

fn bytes_per_weight(p: Precision) -> f32 {
    match p {
        Precision::Int8 => 1.0,
        Precision::Int4 => 0.5,
        Precision::Bf16 => 2.0,
        Precision::F32 => 4.0,
    }
}

fn expert_bytes(d: usize, m: usize, p: Precision) -> f64 {
    (3 * d * m) as f64 * bytes_per_weight(p) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn int8_expert_is_6_mib() {
        assert_eq!(expert_bytes(2048, 1024, Precision::Int8), 6_291_456.0);
    }
}
```

6 291 456 byte = 6 MiB. Cộng hệ số tỉ lệ FP32 `(2m + d) × 4 = 16 384` byte thì được 6 307 840 byte, đúng con số trong báo cáo.
</details>

### Bài 4 ★★ — Result và `?`

Viết hàm `parse_cache_mib(s: &str) -> Result<usize, String>` nhận chuỗi như `"64"` và trả về số byte (`64 × 1024 × 1024`). Báo lỗi nếu chuỗi không phải số, bằng 0, hoặc phép nhân bị tràn. Dùng `checked_mul`.

<details><summary>Lời giải</summary>

```rust
fn parse_cache_mib(s: &str) -> Result<usize, String> {
    let mib: usize = s.trim().parse().map_err(|e| format!("không phải số: {e}"))?;
    if mib == 0 {
        return Err("cache phải lớn hơn 0".into());
    }
    mib.checked_mul(1024 * 1024).ok_or_else(|| "tràn số".to_string())
}

#[test]
fn cases() {
    assert_eq!(parse_cache_mib("64"), Ok(67_108_864));
    assert!(parse_cache_mib("abc").is_err());
    assert!(parse_cache_mib("0").is_err());
    assert!(parse_cache_mib(&usize::MAX.to_string()).is_err());
}
```
</details>

### Bài 5 ★★ — Iterator

Không dùng vòng lặp `for` có chỉ số, hãy viết:

1. `argmax(v: &[f32]) -> usize`: chỉ số phần tử lớn nhất. Khi bằng nhau thì lấy chỉ số nhỏ hơn.
2. `top_k(v: &[f32], k: usize) -> Vec<usize>`: k chỉ số có giá trị lớn nhất, giảm dần.

So sánh `argmax` của bạn với `argmax_greedy` trong `src/olmoe.rs`.

<details><summary>Lời giải</summary>

```rust
fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0
}

fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}
```

Dùng `>` (không phải `>=`) trong `argmax` để giữ chỉ số nhỏ khi bằng nhau. `total_cmp` so sánh được cả NaN, còn `partial_cmp().unwrap()` sẽ panic nếu gặp NaN.

`top_k` ở đây sắp xếp toàn bộ, tốn O(n log n). Engine dùng cách nhanh hơn: phân hoạch lấy nhóm k phần tử lớn nhất (`select_nth_unstable_by`), rồi chỉ sắp xếp nhóm đó. Xem test `test_zero_alloc_topk_matches_full_sort` trong `src/olmoe.rs`: nó kiểm cách nhanh cho kết quả giống hệt cách sắp xếp toàn bộ.
</details>

### Bài 6 ★★ — Đọc code thật

Mở `src/safetensors.rs`, hàm `add_shard`. Trả lời:

1. 8 byte đầu tiên của tệp `.safetensors` chứa gì?
2. Hàm kiểm tra những điều kiện nào để đảm bảo các tensor không chồng lấn nhau?
3. Vì sao tensor được lưu kèm `offset: start + begin`, thay vì chỉ `begin`?

<details><summary>Lời giải</summary>

1. Độ dài phần header JSON, dạng số nguyên 64 bit little-endian.
2. Mỗi tensor có `end >= begin`, `end` không vượt độ dài phần dữ liệu, và `end - begin` đúng bằng `số phần tử × kích thước kiểu`. Sau đó các khoảng `[begin, end)` được sắp xếp và kiểm tra nối tiếp nhau không có khe hở hay chồng lấn (`begin != cursor` thì lỗi). Cuối cùng, tổng độ dài phải bằng đúng phần dữ liệu của tệp.
3. `begin` tính từ đầu vùng dữ liệu, còn vùng dữ liệu bắt đầu sau 8 byte độ dài và phần header. Lưu luôn vị trí tuyệt đối trong tệp để khi đọc chỉ cần một lần `pread(offset)`, không phải cộng lại mỗi lần.
</details>

### Bài 7 ★★★ — Thiết kế với ownership

Viết `struct ExpertCache` với ngân sách byte cố định, có hai phương thức:

- `get(&mut self, id: usize) -> Option<&Vec<u8>>`
- `insert(&mut self, id: usize, data: Vec<u8>)`: khi vượt ngân sách thì bỏ expert được chèn sớm nhất (FIFO).

Yêu cầu: không dùng `clone()` cho dữ liệu expert. Viết test chứng minh tổng byte không bao giờ vượt ngân sách.

<details><summary>Lời giải</summary>

```rust
use std::collections::VecDeque;

struct ExpertCache {
    budget: usize,
    used: usize,
    entries: VecDeque<(usize, Vec<u8>)>,
}

impl ExpertCache {
    fn new(budget: usize) -> Self { Self { budget, used: 0, entries: VecDeque::new() } }

    fn get(&mut self, id: usize) -> Option<&Vec<u8>> {
        self.entries.iter().find(|(i, _)| *i == id).map(|(_, d)| d)
    }

    fn insert(&mut self, id: usize, data: Vec<u8>) {
        if data.len() > self.budget { return; }           // không bao giờ vừa
        while self.used + data.len() > self.budget {
            let (_, old) = self.entries.pop_front().unwrap();
            self.used -= old.len();                       // old bị drop ở đây, RAM được trả
        }
        self.used += data.len();
        self.entries.push_back((id, data));               // data được move vào, không copy
    }
}

#[test]
fn never_exceeds_budget() {
    let mut c = ExpertCache::new(100);
    for i in 0..50 {
        c.insert(i, vec![0u8; 7 + i % 30]);
        assert!(c.used <= 100);
        assert_eq!(c.used, c.entries.iter().map(|(_, d)| d.len()).sum::<usize>());
    }
}
```

So sánh với engine: `layer.cache` trong `src/olmoe.rs` cũng là danh sách có ngân sách byte, nhưng chọn expert bị loại theo heat (module 09). Engine còn **tái sử dụng** vùng nhớ của expert bị loại (`recycled_buffers`) thay vì drop, để tránh cấp phát lại. Đó là bước tối ưu tiếp theo của bài này.
</details>

### Bài 8 ★★★ — Đọc safetensors

Viết chương trình Rust nhận đường dẫn tới một tệp `.safetensors` (ví dụ một shard trong `../Models/olmoe_merged`) và in ra 10 tensor đầu tiên cùng dtype, shape và số byte. Chỉ dùng `std` và `serde_json`.

<details><summary>Gợi ý và lời giải</summary>

Đọc 8 byte độ dài header, đọc header, `serde_json::from_slice::<serde_json::Value>`, duyệt các key trừ `__metadata__`. Mỗi entry có `dtype`, `shape`, `data_offsets`.

```rust
use std::{fs::File, io::Read};

fn main() -> std::io::Result<()> {
    let path = std::env::args().nth(1).expect("đường dẫn");
    let mut f = File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let json: serde_json::Value = serde_json::from_slice(&header).unwrap();
    for (name, info) in json.as_object().unwrap().iter().filter(|(k, _)| *k != "__metadata__").take(10) {
        let off = &info["data_offsets"];
        let bytes = off[1].as_u64().unwrap() - off[0].as_u64().unwrap();
        println!("{name}: {} {} {} byte", info["dtype"], info["shape"], bytes);
    }
    Ok(())
}
```

Bạn sẽ thấy tensor expert có dtype `I8` và tên dạng `model.layers.N.mlp.experts.E...`, kèm tensor scales `F32`. Xem chính xác tên tensor trong `src/int8_expert.rs` (hàm `load_reusing`).
</details>

## Tài liệu đọc thêm

- *The Rust Programming Language* (sách chính thức, miễn phí): https://doc.rust-lang.org/book/ — chương 4 (ownership), 6 (enum), 9 (lỗi), 10 (trait, lifetime), 11 (test).
- *Rust by Example*: https://doc.rust-lang.org/rust-by-example/
- Rustlings, bài tập nhỏ có chấm tự động: https://github.com/rust-lang/rustlings
