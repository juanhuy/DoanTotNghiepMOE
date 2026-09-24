use axum::{
    extract::DefaultBodyLimit,
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive},
        Html, IntoResponse, Response, Sse,
    },
    routing::{get, post},
    Json, Router,
};
use moe_tier_engine::{model::loader::write_demo, InferenceEngine};
use serde::{Deserialize, Serialize};
use std::{io, sync::Arc, time::Duration};
use tracing::info;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    messages: Vec<moe_tier_engine::engine::ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    model: Option<String>,
    session_id: Option<String>,
}
fn default_max_tokens() -> usize {
    32
}

#[derive(Serialize)]
struct ChatResponse {
    model: &'static str,
    choices: Vec<Choice>,
    usage: Usage,
    metrics: Option<moe_tier_engine::generation::GenerationMetrics>,
}
#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}
#[derive(Serialize)]
struct Choice {
    index: usize,
    message: MessageResponse,
    finish_reason: &'static str,
}
#[derive(Serialize)]
struct MessageResponse {
    role: &'static str,
    content: String,
}

type ApiError = (StatusCode, String);

#[derive(Clone)]
struct AppState {
    engine: Arc<InferenceEngine>,
    stream_send_timeout: Duration,
}

const INDEX_HTML: &str = include_str!("../web/index.html");

fn app(state: AppState) -> Router {
    Router::new()
        .route("/", get(|| async { Html(INDEX_HTML) }))
        .route("/health", get(|| async { "ok" }))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::max(128 * 1024))
        .with_state(state)
}

#[tokio::main]
async fn main() -> io::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args: Vec<_> = std::env::args().skip(1).collect();
    if let [flag, directory] = args.as_slice() {
        if flag == "--init-demo" {
            write_demo(directory).await?;
            info!(
                directory,
                "Created untrained tiny model; output will not be meaningful language"
            );
            return Ok(());
        }
    }
    if !args.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: cargo run -- [--init-demo NEW_DIRECTORY]",
        ));
    }
    let directory =
        std::env::var("MOE_MODEL_DIR").unwrap_or_else(|_| "./models/olmoe-1b-7b-int8".into());
    let cache_bytes = std::env::var("MOE_CACHE_BYTES_PER_LAYER")
        .unwrap_or_else(|_| "134217728".into())
        .parse::<usize>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid cache byte budget"))?;
    let context = std::env::var("MOE_CONTEXT")
        .unwrap_or_else(|_| "512".into())
        .parse::<usize>()
        .map_err(io::Error::other)?;
    let dense = std::env::var("MOE_DENSE_BYTES")
        .unwrap_or_else(|_| "3221225472".into())
        .parse::<usize>()
        .map_err(io::Error::other)?;
    let stream_send_timeout = Duration::from_millis(
        std::env::var("MOE_STREAM_SEND_TIMEOUT_MS")
            .unwrap_or_else(|_| "30000".into())
            .parse::<u64>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid stream timeout"))?,
    );
    if stream_send_timeout.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "stream timeout must be positive",
        ));
    }
    let engine =
        Arc::new(InferenceEngine::load_with_limits(&directory, cache_bytes, context, dense).await?);
    let model_name = engine.model_name();
    let app = app(AppState {
        engine,
        stream_send_timeout,
    });
    let address = std::env::var("MOE_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener = tokio::net::TcpListener::bind(&address).await?;
    info!(address, directory, model_name, "CPU MoE server ready");
    axum::serve(listener, app).await
}

async fn chat(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let engine = &state.engine;
    if request
        .model
        .as_deref()
        .is_some_and(|name| name != engine.model_name())
    {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("loaded model is {}", engine.model_name()),
        ));
    }
    if request.stream {
        let stream = engine
            .chat_stream_with_session(
                request.messages,
                request.max_tokens,
                request.session_id,
                state.stream_send_timeout,
            )
            .await
            .map_err(api_error)?;
        return Ok(Sse::new(SseEvents {
            stream,
            start: true,
            done: false,
            ended: false,
        })
        .keep_alive(KeepAlive::default())
        .into_response());
    }
    let result = engine
        .chat_with_session(request.messages, request.max_tokens, request.session_id)
        .await
        .map_err(api_error)?;
    let completion_tokens = result.token_ids.len();
    Ok(Json(ChatResponse {
        model: engine.model_name(),
        choices: vec![Choice {
            index: 0,
            message: MessageResponse {
                role: "assistant",
                content: result.text,
            },
            finish_reason: result.finish_reason,
        }],
        usage: Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens,
            total_tokens: result.prompt_tokens + completion_tokens,
        },
        metrics: result.metrics,
    })
    .into_response())
}

fn api_error(error: io::Error) -> ApiError {
    let status = match error.kind() {
        io::ErrorKind::InvalidInput => StatusCode::BAD_REQUEST,
        io::ErrorKind::WouldBlock => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, error.to_string())
}

struct SseEvents {
    stream: moe_tier_engine::engine::ChatStream,
    start: bool,
    done: bool,
    ended: bool,
}
impl futures_core::Stream for SseEvents {
    type Item = Result<Event, std::convert::Infallible>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use moe_tier_engine::engine::ChatEvent;
        use serde_json::json;
        use std::task::Poll;
        if self.ended {
            return Poll::Ready(None);
        }
        if self.done {
            self.ended = true;
            return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
        }
        let value = if self.start {
            self.start = false;
            json!({"object":"chat.completion.chunk", "model":"olmoe", "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]})
        } else {
            match std::pin::Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(ChatEvent::Delta(text)))) => {
                    json!({"object":"chat.completion.chunk", "model":"olmoe", "choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]})
                }
                Poll::Ready(Some(Ok(ChatEvent::Finished(result)))) => {
                    self.done = true;
                    json!({"object":"chat.completion.chunk", "model":"olmoe", "choices":[{"index":0,"delta":{},"finish_reason":result.finish_reason}],
                        "usage":{"prompt_tokens":result.prompt_tokens,"completion_tokens":result.token_ids.len(),"total_tokens":result.prompt_tokens+result.token_ids.len()},"metrics":result.metrics})
                }
                Poll::Ready(Some(Err(error))) => {
                    self.done = true;
                    json!({"error":{"message":error.to_string(),"type":"generation_error"}})
                }
                Poll::Ready(None) => {
                    self.done = true;
                    json!({"error":{"message":"generation stream ended unexpectedly","type":"generation_error"}})
                }
            }
        };
        Poll::Ready(Some(Ok(Event::default().data(value.to_string()))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn http_stream_reports_payload_error_after_headers() {
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/olmoe/unnormalized-mha");
        let dir = tempfile::tempdir().unwrap();
        for entry in std::fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), dir.path().join(entry.file_name())).unwrap();
        }
        let engine = Arc::new(
            InferenceEngine::load_with_limits(dir.path(), 10_000, 64, 10_000_000)
                .await
                .unwrap(),
        );
        // Simulate storage failure after dense weights/header indexing completed.
        std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join("model.safetensors"))
            .unwrap()
            .set_len(8)
            .unwrap();
        let app = app(AppState {
            engine,
            stream_send_timeout: Duration::from_secs(1),
        });
        let body = serde_json::json!({"messages":[{"role":"user","content":"token3"}],"stream":true,"max_tokens":3});
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1_000_000).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("generation_error"));
        assert_eq!(text.matches("data: [DONE]").count(), 1);
        assert!(!text.contains("\"usage\""));
    }

    #[tokio::test]
    async fn http_stream_and_validation() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/olmoe/unnormalized-mha");
        let engine = Arc::new(
            InferenceEngine::load_with_limits(directory, 10_000, 64, 10_000_000)
                .await
                .unwrap(),
        );
        let app = app(AppState {
            engine,
            stream_send_timeout: Duration::from_secs(1),
        });
        let page = app
            .clone()
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(page.status(), StatusCode::OK);
        assert!(page.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/html"));
        let page = to_bytes(page.into_body(), 1_000_000).await.unwrap();
        let page = String::from_utf8(page.to_vec()).unwrap();
        assert!(page.contains("AbortController"));
        assert!(page.contains("/v1/chat/completions"));
        assert!(page.contains("localStorage"));
        assert!(page.contains("max-tokens"));
        assert!(page.contains("decode_tokens_per_second"));
        assert!(page.contains("session_id"));
        let oversized = app
            .clone()
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("x".repeat(129 * 1024)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
        for (body, status) in [
            (
                serde_json::json!({"messages":[],"stream":true}),
                StatusCode::BAD_REQUEST,
            ),
            (
                serde_json::json!({"messages":[{"role":"user","content":"token3"}],"stream":true,"max_tokens":100}),
                StatusCode::BAD_REQUEST,
            ),
            (
                serde_json::json!({"messages":[{"role":"user","content":"token3"}],"stream":true,"model":"wrong"}),
                StatusCode::BAD_REQUEST,
            ),
            (
                serde_json::json!({"messages":[{"role":"user","content":"token3"}],"stream":true,"session_id":"bad session!"}),
                StatusCode::BAD_REQUEST,
            ),
            (
                serde_json::json!({"messages":[{"role":"user","content":"token3"}],"stream":true,"max_tokens":3}),
                StatusCode::OK,
            ),
            (
                serde_json::json!({"messages":[{"role":"user","content":"token3"}],"stream":false,"max_tokens":3}),
                StatusCode::OK,
            ),
            (
                serde_json::json!({"messages":[{"role":"user","content":"x".repeat(65 * 1024)}],"stream":true,"max_tokens":3}),
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::post("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            let streaming = status == StatusCode::OK && body["stream"] == true;
            if streaming {
                assert_eq!(response.headers()["content-type"], "text/event-stream");
            }
            let bytes = to_bytes(response.into_body(), 1_000_000).await.unwrap();
            if streaming {
                let text = String::from_utf8(bytes.to_vec()).unwrap();
                assert!(text.contains("assistant"));
                assert!(text.contains("finish_reason"));
                assert_eq!(text.matches("data: [DONE]").count(), 1);
                assert!(!text.contains("generation_error"));
            }
        }
    }
}
