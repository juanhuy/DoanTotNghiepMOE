# Module 03 — Async, web server và streaming

## Mục tiêu

Hiểu cách một web server phục vụ nhiều kết nối bằng ít luồng, vì sao không được chạy phép tính nặng trong code async, cách đẩy từng token về trình duyệt ngay khi sinh ra (streaming), và cách dừng công việc khi người dùng bấm Stop.

**Dùng khi đi làm:** gần như mọi vị trí backend. Các khái niệm event loop, backpressure, timeout, hủy request, rate limit, giao thức streaming (SSE, WebSocket) áp dụng nguyên cho Node.js, Python asyncio, Go, Java. Đặc biệt, mọi API chatbot (OpenAI, Anthropic...) đều stream bằng SSE đúng như đồ án.

## 1. Vì sao cần async

Một web server chờ I/O rất nhiều: chờ client gửi dữ liệu, chờ cơ sở dữ liệu, chờ mạng. Có hai cách phục vụ nhiều kết nối:

- **Mỗi kết nối một luồng hệ điều hành.** Đơn giản, nhưng mỗi luồng tốn vài MB stack và chi phí chuyển ngữ cảnh. Với 10 000 kết nối thì không chịu nổi.
- **Async:** một số ít luồng chạy rất nhiều **tác vụ** (task) nhẹ. Khi tác vụ phải chờ I/O, nó nhường luồng cho tác vụ khác thay vì chặn luồng.

Trong Rust, hàm `async fn` trả về một **future**: một máy trạng thái chưa chạy. Future chỉ tiến triển khi được **runtime** (ở đây là Tokio) gọi `poll`. Tại mỗi `.await`, nếu dữ liệu chưa sẵn sàng thì future trả về `Pending` và runtime chạy future khác.

```rust
async fn handle() -> String {
    let data = fetch_from_db().await;   // nhường luồng trong lúc chờ
    format!("kết quả: {data}")
}
```

## 2. Luật vàng: không chặn luồng async

Các luồng của runtime Tokio phải luôn quay lại vòng sự kiện nhanh. Nếu một tác vụ tính toán liên tục 10 giây mà không có `.await`, luồng đó bị chặn 10 giây, và mọi tác vụ khác xếp trên luồng đó (kể cả trả lời `/health`) cũng phải chờ.

Suy luận một token có thể mất hàng trăm mili giây đến vài giây, nên **không được** chạy trong code async. Cách đúng là đẩy sang thread pool dành cho công việc chặn:

```rust
let result = tokio::task::spawn_blocking(move || {
    model.lock().unwrap().generate(&prompt, 64)   // chạy trên luồng riêng
}).await?;
```

**Trong đồ án:** `src/engine.rs` gọi `tokio::task::spawn_blocking` cả khi nạp mô hình (đọc hàng GB) lẫn khi suy luận. Phía async chỉ nhận request, kiểm tra và chuyển tiếp kết quả.

## 3. Axum: định tuyến HTTP

Axum là web framework xây trên Tokio. Ứng dụng là một `Router` gắn đường dẫn với hàm xử lý (handler):

```rust
fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/experts", get(get_experts))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::max(128 * 1024))   // chặn body quá lớn
        .with_state(state)
}
```

Đây là trích đoạn thật của `src/main.rs`. Một vài điểm đáng học:

- **Extractor:** handler khai báo tham số kiểu `State<AppState>`, `Json<ChatRequest>`; Axum tự trích từ request và tự trả lỗi 400 nếu JSON sai.
- **Giới hạn kích thước body** là biện pháp phòng thủ cơ bản: client không thể gửi request 1 GB làm cạn RAM.
- **Từ chối trường lạ:** struct request dùng `#[serde(deny_unknown_fields)]`, nên request gõ sai tên trường bị báo lỗi thay vì bị lờ đi âm thầm.
- **Tương thích API chuẩn:** endpoint `/v1/chat/completions` và `/v1/models` theo định dạng của OpenAI, nên các client sẵn có dùng được ngay.

### Ánh xạ lỗi sang mã HTTP

`src/main.rs::api_error` đổi `io::ErrorKind` thành mã trạng thái:

| Lỗi bên trong | Mã HTTP | Ý nghĩa |
| --- | --- | --- |
| `InvalidInput` | 400 Bad Request | Request sai (tin nhắn rỗng, vượt ngữ cảnh) |
| `WouldBlock` | 429 Too Many Requests | Mô hình đang bận với request khác |
| `OutOfMemory` | 503 Service Unavailable | RAM guard từ chối |
| khác | 500 Internal Server Error | Lỗi không mong đợi |

Mã đúng giúp client biết nên làm gì: 429 thì thử lại sau, 400 thì sửa request, 503 thì đợi server.

## 4. Kiểm soát đồng thời bằng Semaphore

Mô hình chỉ phục vụ được một request suy luận tại một thời điểm (dùng chung KV, cache, GPU). Engine dùng một `Semaphore` có 1 giấy phép:

```rust
let permit = self.gate.clone().try_acquire_owned().map_err(|_| {
    io::Error::new(io::ErrorKind::WouldBlock, "model busy; retry after current generation")
})?;
```

`try_acquire` không chờ: nếu đang có người dùng thì báo bận ngay (429). Giấy phép được **move** vào tác vụ suy luận (`let _permit = permit;`) và tự trả lại khi tác vụ kết thúc, kể cả khi lỗi hay panic. Đây là mẫu RAII: tài nguyên gắn với vòng đời của một giá trị, nên không thể quên trả.

Thiết kế này chọn **từ chối ngay** thay vì **xếp hàng chờ**. Với một demo một người dùng, từ chối rõ ràng tốt hơn để request treo vô thời hạn. Hệ thống phục vụ nhiều người sẽ cần hàng đợi có giới hạn và gộp nhiều request thành lô (continuous batching, như vLLM).

## 5. Streaming bằng Server-Sent Events (SSE)

### 5.1 Giao thức

SSE là cách đơn giản nhất để server đẩy dữ liệu liên tục về trình duyệt qua một kết nối HTTP bình thường. Response có `Content-Type: text/event-stream`, và mỗi sự kiện là các dòng `data: ...` kết thúc bằng một dòng trống:

```
data: {"choices":[{"delta":{"role":"assistant"}}]}

data: {"choices":[{"delta":{"content":"Mars"}}]}

data: {"choices":[{"delta":{"content":" is"}}]}

data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{...},"metrics":{...}}

data: [DONE]

```

So với WebSocket, SSE chỉ đi một chiều (server → client) nhưng đơn giản hơn nhiều, đi qua proxy dễ, và trình duyệt tự kết nối lại. Chatbot chỉ cần một chiều nên SSE là lựa chọn chuẩn.

### 5.2 Luồng dữ liệu trong đồ án

```
 luồng blocking (spawn_blocking)              luồng async (Tokio)                 trình duyệt
 ┌──────────────────────────────┐   mpsc    ┌─────────────────────────┐   SSE   ┌───────────┐
 │ generate(): mỗi token sinh ra │ ───────► │ SseEvents::poll_next    │ ──────► │ index.html│
 │ gửi ChatEvent::Delta(text)    │ (8 chỗ)  │ đổi thành JSON chunk    │         │ đọc stream│
 │ cuối cùng gửi Finished(...)   │          │ cuối cùng gửi [DONE]    │         │           │
 └──────────────────────────────┘          └─────────────────────────┘         └───────────┘
```

- **Channel có giới hạn:** `tokio::sync::mpsc::channel(8)` chỉ chứa tối đa 8 sự kiện. Nếu client đọc chậm, phía sinh token phải chờ. Cơ chế tự điều tiết này gọi là **backpressure**. Không có nó, server có thể tích lũy vô hạn dữ liệu chưa gửi trong RAM.
- **Timeout khi gửi:** hàm `blocking_stream_send` trong `src/engine.rs` chờ gửi tối đa một khoảng thời gian. Client treo không đọc thì request kết thúc với lỗi `TimedOut` thay vì giữ mô hình mãi.
- **Chuyển từ sync sang async:** luồng blocking không có `.await`, nên nó dùng `runtime.block_on(...)` để chờ việc gửi vào channel.
- **Stream thủ công:** `SseEvents` trong `src/main.rs` cài trait `Stream` bằng tay. Hàm `poll_next` là một máy trạng thái nhỏ: lần đầu gửi `role: assistant`, sau đó chuyển từng `Delta`, khi gặp `Finished` gửi chunk cuối kèm `usage` và `metrics`, rồi gửi `[DONE]`, rồi kết thúc.

### 5.3 Phía trình duyệt

Trình duyệt đọc bằng `fetch` và `response.body.getReader()`, cắt theo dòng trống, parse JSON từng sự kiện (`web/chat_stream.js`). Câu trả lời chỉ được lưu vào lịch sử hội thoại khi nhận đủ: chunk có `finish_reason`, rồi `[DONE]`, rồi stream đóng hợp lệ. Câu trả lời bị dừng giữa chừng không được lưu như câu trả lời hoàn chỉnh. Đây là chi tiết nhỏ nhưng quan trọng về **tính nhất quán dữ liệu**.

## 6. Hủy request

Người dùng bấm Stop thì trình duyệt đóng kết nối. Chuỗi sự kiện sau đó:

1. Axum drop stream `SseEvents`, kéo theo drop phía nhận của channel.
2. Phía gửi thấy channel đã đóng: `tx.is_closed()` trả về `true`.
3. Hàm sinh token gọi một closure kiểm tra ở các điểm định sẵn (giữa các lớp, giữa các token):

```rust
&|| {
    if tx.is_closed() {
        Err(cancelled())                          // client đã ngắt
    } else if Instant::now() > overall_deadline {
        Err(io::Error::new(io::ErrorKind::TimedOut, "..."))
    } else {
        Ok(())
    }
}
```

4. Lỗi lan ngược lên bằng `?`, tác vụ kết thúc, giấy phép Semaphore được trả, và mô hình nhận được request tiếp theo.

Đây là **hủy hợp tác** (cooperative cancellation): không có cách an toàn để "giết" một luồng đang tính giữa chừng, nên code tự kiểm tra cờ hủy ở các điểm an toàn. Mẫu này giống `CancellationToken` trong C#, `context.Context` trong Go, `AbortController` trong JavaScript.

Có thêm một hạn mức tổng (`DEFAULT_MAX_STREAM_DURATION`) để một request không bao giờ chiếm mô hình quá lâu.

Test liên quan trong `src/olmoe.rs`: `disconnect_after_content_stops_decode_and_preserves_model`, `cancellation_during_prefill_leaves_model_reusable`. Tên test nói rõ yêu cầu: hủy xong thì mô hình phải còn dùng được.

## 7. Báo sẵn sàng trước khi stream

Vấn đề: nếu request có lỗi (vượt cửa sổ ngữ cảnh), nên trả HTTP 400 rõ ràng. Nhưng một khi đã gửi header `200 OK` của SSE thì không đổi mã trạng thái được nữa, lỗi chỉ còn gửi được dưới dạng một sự kiện trong stream.

Engine giải quyết bằng một kênh `oneshot` báo sẵn sàng. Tác vụ suy luận kiểm tra đầu vào (dựng prompt, mã hóa token, kiểm tra độ dài) rồi mới báo "ready". Handler HTTP chờ tín hiệu này trước khi trả header SSE. Nhờ vậy, lỗi đầu vào vẫn là mã HTTP đúng; chỉ lỗi xảy ra trong lúc sinh mới đi qua stream.

---

## Bài tập

### Bài 1 ★ — Chặn runtime

Tạo server Axum có hai route: `/fast` trả `"ok"` ngay, và `/slow` chạy vòng lặp tính toán 5 giây (không có `.await`). Chạy Tokio với 1 luồng (`#[tokio::main(flavor = "current_thread")]`). Gọi `/slow` rồi ngay lập tức gọi `/fast` bằng một terminal khác. Điều gì xảy ra? Sửa bằng `spawn_blocking`.

<details><summary>Lời giải</summary>

`/fast` phải chờ đến khi `/slow` xong mới trả lời, vì luồng duy nhất của runtime bị chặn. Sửa:

```rust
async fn slow() -> String {
    tokio::task::spawn_blocking(|| {
        let mut x = 0u64;
        for i in 0..3_000_000_000u64 { x = x.wrapping_add(i); }
        x.to_string()
    }).await.unwrap()
}
```

Sau khi sửa, `/fast` trả lời ngay trong lúc `/slow` đang tính.
</details>

### Bài 2 ★ — SSE bằng curl

Chạy engine bằng `tools/demo.sh olmoe-cpu` rồi gửi:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Where is Mars?"}],"max_tokens":16,"stream":true}'
```

Quan sát định dạng từng sự kiện. Chunk nào chứa `metrics`? Tìm trường `time_to_first_token_ms`, `expert_io_ms`. Cờ `-N` có tác dụng gì?

<details><summary>Lời giải</summary>

Chunk cuối trước `[DONE]` chứa `finish_reason`, `usage` và `metrics`. Cờ `-N` tắt bộ đệm đầu ra của curl để thấy từng sự kiện ngay khi đến; không có cờ này curl có thể gom lại rồi in một lần.
</details>

### Bài 3 ★★ — Semaphore

Gửi hai request stream cùng lúc tới engine (hai terminal, hoặc hai lệnh `curl ... &`). Request thứ hai nhận mã HTTP nào và thông báo gì? Tìm dòng code sinh ra thông báo đó.

<details><summary>Lời giải</summary>

Mã 429 kèm thông báo `model busy; retry after current generation`. Sinh ra ở `try_acquire_owned()` trong `src/engine.rs`, rồi được `api_error` trong `src/main.rs` ánh xạ `WouldBlock` sang `TOO_MANY_REQUESTS`.
</details>

### Bài 4 ★★ — Backpressure

Viết chương trình Tokio: một tác vụ `spawn_blocking` sinh 1000 số, gửi qua `mpsc::channel(4)` bằng `blocking_send`; phía nhận async đọc mỗi số rồi `sleep(10ms)`. In thời điểm gửi và nhận. Quan sát phía gửi bị chặn thế nào khi channel đầy. Đổi sang `unbounded_channel` và so sánh bộ nhớ khi phía nhận rất chậm.

<details><summary>Gợi ý</summary>

Với channel có giới hạn, phía gửi không bao giờ đi trước phía nhận quá 4 phần tử: bộ nhớ ổn định. Với channel không giới hạn, phía gửi chạy xong ngay và cả 1000 phần tử nằm chờ trong RAM. Trong hệ thống thật, phía nhận chậm có thể là một client mạng yếu; không có backpressure, một client chậm có thể làm server hết RAM.
</details>

### Bài 5 ★★ — Hủy hợp tác

Viết hàm `long_job(check: &dyn Fn() -> Result<(), String>) -> Result<u64, String>` lặp 100 bước, mỗi bước ngủ 50 ms và gọi `check()?`. Dùng nó trong `spawn_blocking`, với `check` trả lỗi khi `AtomicBool` "cancelled" được bật. Từ tác vụ async, sau 1 giây bật cờ. Chứng minh công việc dừng trong vòng 50 ms sau khi bật cờ.

<details><summary>Lời giải</summary>

```rust
use std::sync::{atomic::{AtomicBool, Ordering}, Arc};
use std::time::{Duration, Instant};

fn long_job(check: &dyn Fn() -> Result<(), String>) -> Result<u64, String> {
    for i in 0..100 {
        std::thread::sleep(Duration::from_millis(50));
        check()?;
        let _ = i;
    }
    Ok(100)
}

#[tokio::main]
async fn main() {
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = cancel.clone();
    let t0 = Instant::now();
    let job = tokio::task::spawn_blocking(move || {
        long_job(&|| if c2.load(Ordering::Relaxed) { Err("cancelled".into()) } else { Ok(()) })
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    cancel.store(true, Ordering::Relaxed);
    let r = job.await.unwrap();
    println!("{r:?} sau {:?}", t0.elapsed());   // khoảng 1,0–1,05 s
}
```

Độ trễ hủy tối đa bằng khoảng cách giữa hai lần gọi `check`. Trong engine, điểm kiểm tra nằm giữa các lớp và giữa các token, nên bấm Stop sẽ dừng trong vòng vài chục đến vài trăm mili giây.
</details>

### Bài 6 ★★★ — Mini server chat có stream

Viết một server Axum có `POST /chat` nhận `{"prompt": "...", "max_tokens": N}` và stream SSE từng "token". Token giả lập bằng cách tách prompt thành từ, mỗi từ cách nhau 100 ms. Yêu cầu:

- chỉ một request chạy tại một thời điểm (Semaphore, request thứ hai nhận 429);
- channel có giới hạn;
- client ngắt kết nối thì luồng sinh dừng lại (in log để chứng minh);
- sự kiện cuối chứa thống kê (số token, tổng thời gian), sau đó là `[DONE]`.

<details><summary>Gợi ý</summary>

Bám theo cấu trúc `chat_stream_with_session_and_sampling` trong `src/engine.rs` và `SseEvents` trong `src/main.rs`. Có thể dùng `axum::response::sse::Sse` với một stream tạo bằng `tokio_stream::wrappers::ReceiverStream` để không phải tự cài `poll_next`. Để phát hiện client ngắt, kiểm `tx.is_closed()` trong luồng sinh.
</details>

## Tài liệu đọc thêm

- *Tokio tutorial*: https://tokio.rs/tokio/tutorial
- *Async Rust book*: https://rust-lang.github.io/async-book/
- Alice Ryhl, *Async: What is blocking?*: https://ryhl.io/blog/async-what-is-blocking/
- MDN, *Using server-sent events*: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
