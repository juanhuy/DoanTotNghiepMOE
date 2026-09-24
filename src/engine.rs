//! Application facade; OLMoE work runs off the async executor, one request at a time.
use crate::{generation::Generation, model::DecoderModel, olmoe::OlmoeModel};
use serde::{Deserialize, Serialize};
use std::{
    io,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

const MAX_MESSAGES: usize = 128;
const MAX_MESSAGE_BYTES: usize = 64 * 1024;
const DEFAULT_STREAM_SEND_TIMEOUT: Duration = Duration::from_secs(30);

pub type InferenceResponse = Generation;
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

fn validate_messages(messages: &[ChatMessage]) -> io::Result<()> {
    let valid_roles = messages
        .iter()
        .all(|message| matches!(message.role.as_str(), "system" | "user" | "assistant"));
    let bytes = messages.iter().try_fold(0usize, |total, message| {
        total.checked_add(message.content.len())
    });
    if messages.is_empty()
        || messages.len() > MAX_MESSAGES
        || !valid_roles
        || bytes.is_none_or(|bytes| bytes > MAX_MESSAGE_BYTES)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "messages must be nonempty, use supported roles, and stay within request limits",
        ));
    }
    Ok(())
}

fn validate_session_id(session_id: Option<&str>) -> io::Result<()> {
    if session_id.is_some_and(|id| {
        id.is_empty()
            || id.len() > 128
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    }) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "session_id must contain 1-128 ASCII letters, digits, '.', '-' or '_'",
        ));
    }
    Ok(())
}

enum Model {
    Tiny(Box<DecoderModel>),
    Olmoe(Arc<Mutex<OlmoeModel>>),
}
pub struct InferenceEngine {
    model: Model,
    gate: Arc<tokio::sync::Semaphore>,
}
impl InferenceEngine {
    pub async fn load(
        directory: impl AsRef<Path>,
        cache_bytes_per_layer: usize,
    ) -> io::Result<Self> {
        Self::load_with_limits(
            directory,
            cache_bytes_per_layer,
            512,
            3 * 1024 * 1024 * 1024,
        )
        .await
    }
    pub async fn load_with_limits(
        directory: impl AsRef<Path>,
        cache: usize,
        context: usize,
        dense: usize,
    ) -> io::Result<Self> {
        let directory = directory.as_ref().to_owned();
        let model = if directory.join("model.json").exists() {
            Model::Tiny(Box::new(DecoderModel::load(directory, cache).await?))
        } else {
            let model = tokio::task::spawn_blocking(move || {
                let model = OlmoeModel::load(directory, cache, context, dense)?;
                model.require_chat()?;
                Ok::<_, io::Error>(model)
            })
            .await
            .map_err(io::Error::other)??;
            Model::Olmoe(Arc::new(Mutex::new(model)))
        };
        Ok(Self {
            model,
            gate: Arc::new(tokio::sync::Semaphore::new(1)),
        })
    }
    pub fn model_name(&self) -> &'static str {
        match self.model {
            Model::Tiny(_) => "tiny-moe",
            Model::Olmoe(_) => "olmoe",
        }
    }
    pub async fn generate(&self, prompt: &str) -> io::Result<InferenceResponse> {
        self.generate_with_limit(prompt, 32).await
    }
    pub async fn generate_with_limit(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> io::Result<InferenceResponse> {
        match &self.model {
            Model::Tiny(model) => model.generate(prompt, max_tokens).await,
            Model::Olmoe(model) => {
                let permit = self.gate.clone().try_acquire_owned().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::WouldBlock,
                        "model busy; retry after current generation",
                    )
                })?;
                let model = model.clone();
                let prompt = prompt.to_owned();
                tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    model
                        .lock()
                        .map_err(|_| io::Error::other("model lock poisoned"))?
                        .generate(&prompt, max_tokens)
                })
                .await
                .map_err(io::Error::other)?
            }
        }
    }
    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
    ) -> io::Result<InferenceResponse> {
        self.chat_with_session(messages, max_tokens, None).await
    }

    pub async fn chat_with_session(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        session_id: Option<String>,
    ) -> io::Result<InferenceResponse> {
        validate_messages(&messages)?;
        validate_session_id(session_id.as_deref())?;
        match &self.model {
            Model::Tiny(model) => {
                let mut prompt = String::new();
                for m in messages {
                    prompt.push_str(&format!("{}: {}\n", m.role, m.content));
                }
                prompt.push_str("assistant: ");
                model.generate(&prompt, max_tokens).await
            }
            Model::Olmoe(model) => {
                let permit = self.gate.clone().try_acquire_owned().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::WouldBlock,
                        "model busy; retry after current generation",
                    )
                })?;
                let model = model.clone();
                tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    let mut model = model
                        .lock()
                        .map_err(|_| io::Error::other("model lock poisoned"))?;
                    let prompt = model.render_chat(&messages)?;
                    model.generate_controlled(
                        &prompt,
                        max_tokens,
                        session_id.as_deref(),
                        &|| Ok(()),
                        || Ok(()),
                        None,
                    )
                })
                .await
                .map_err(io::Error::other)?
            }
        }
    }
}

/// Bounded output queue. Dropping the receiver cancels the blocking producer.
pub enum ChatEvent {
    Delta(String),
    Finished(Generation),
}

pub struct ChatStream {
    receiver: tokio::sync::mpsc::Receiver<io::Result<ChatEvent>>,
}
impl ChatStream {
    pub async fn recv(&mut self) -> Option<io::Result<ChatEvent>> {
        self.receiver.recv().await
    }
}
impl futures_core::Stream for ChatStream {
    type Item = io::Result<ChatEvent>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

fn blocking_stream_send(
    runtime: &tokio::runtime::Handle,
    sender: &tokio::sync::mpsc::Sender<io::Result<ChatEvent>>,
    timeout: Duration,
    item: io::Result<ChatEvent>,
) -> io::Result<()> {
    runtime.block_on(async {
        tokio::time::timeout(timeout, sender.send(item))
            .await
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::TimedOut,
                    "stream client did not consume output before timeout",
                )
            })?
            .map_err(|_| io::Error::new(io::ErrorKind::Interrupted, "client disconnected"))
    })
}

impl InferenceEngine {
    /// Returns only after prompt validation. Busy/input errors remain HTTP errors.
    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
    ) -> io::Result<ChatStream> {
        self.chat_stream_with_timeout(messages, max_tokens, DEFAULT_STREAM_SEND_TIMEOUT)
            .await
    }

    pub async fn chat_stream_with_timeout(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        send_timeout: Duration,
    ) -> io::Result<ChatStream> {
        self.chat_stream_with_session(messages, max_tokens, None, send_timeout)
            .await
    }

    pub async fn chat_stream_with_session(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        session_id: Option<String>,
        send_timeout: Duration,
    ) -> io::Result<ChatStream> {
        validate_messages(&messages)?;
        validate_session_id(session_id.as_deref())?;
        if send_timeout.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "stream send timeout must be positive",
            ));
        }
        let Model::Olmoe(model) = &self.model else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "streaming requires OLMoE",
            ));
        };
        let permit = self.gate.clone().try_acquire_owned().map_err(|_| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                "model busy; retry after current generation",
            )
        })?;
        let model = model.clone();
        let (tx, receiver) = tokio::sync::mpsc::channel(8);
        // Keep receiver in this future, so aborting validation also cancels the worker.
        let stream = ChatStream { receiver };
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let panic_tx = tx.clone();
        let runtime = tokio::runtime::Handle::current();
        let worker = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut ready_tx = Some(ready_tx);
            let cancelled = || io::Error::new(io::ErrorKind::Interrupted, "client disconnected");
            let result = (|| {
                let mut model = model
                    .lock()
                    .map_err(|_| io::Error::other("model lock poisoned"))?;
                let prompt = model.render_chat(&messages)?;
                model.generate_controlled(
                    &prompt,
                    max_tokens,
                    session_id.as_deref(),
                    &|| {
                        if tx.is_closed() {
                            Err(cancelled())
                        } else {
                            Ok(())
                        }
                    },
                    || {
                        ready_tx
                            .take()
                            .unwrap()
                            .send(Ok(()))
                            .map_err(|_| cancelled())
                    },
                    Some(&mut |delta| {
                        blocking_stream_send(
                            &runtime,
                            &tx,
                            send_timeout,
                            Ok(ChatEvent::Delta(delta.to_owned())),
                        )
                    }),
                )
            })();
            match result {
                Ok(result) => {
                    let _ = blocking_stream_send(
                        &runtime,
                        &tx,
                        send_timeout,
                        Ok(ChatEvent::Finished(result)),
                    );
                }
                Err(error) => {
                    if let Some(ready_tx) = ready_tx {
                        let _ = ready_tx.send(Err(error));
                    } else {
                        let _ = blocking_stream_send(&runtime, &tx, send_timeout, Err(error));
                    }
                }
            }
        });
        tokio::spawn(async move {
            if let Err(error) = worker.await {
                let _ = panic_tx.send(Err(io::Error::other(error))).await;
            }
        });
        ready_rx
            .await
            .map_err(|_| io::Error::other("generation worker stopped during validation"))??;
        Ok(stream)
    }
}

#[cfg(test)]
mod stream_queue_tests {
    use super::*;

    #[tokio::test]
    async fn full_stream_queue_times_out() {
        let (sender, _receiver) = tokio::sync::mpsc::channel(1);
        sender
            .send(Ok(ChatEvent::Delta("first".into())))
            .await
            .unwrap();
        let runtime = tokio::runtime::Handle::current();
        let started = std::time::Instant::now();
        let error = tokio::task::spawn_blocking(move || {
            blocking_stream_send(
                &runtime,
                &sender,
                Duration::from_millis(20),
                Ok(ChatEvent::Delta("blocked".into())),
            )
            .unwrap_err()
        })
        .await
        .unwrap();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
