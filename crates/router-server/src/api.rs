use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{unfold, Stream, StreamExt};
use router_core::{RouterError, RoutingPolicy, Worker};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{error, info, info_span, warn, Instrument};

use crate::metrics::Metrics;
use crate::state::AppState;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub stream: bool,
    /// Preserved and re-serialized verbatim so unknown OpenAI fields survive
    /// proxying (e.g. `stop`, `top_p`, `user`).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl ChatCompletionRequest {
    fn prompt_text(&self) -> String {
        let mut parts = Vec::with_capacity(self.messages.len());
        for message in &self.messages {
            parts.push(format!("{}: {}", message.role, message.content));
        }
        parts.join("\n")
    }
}

/// Errors surfaced over HTTP. Maps onto OpenAI-style `{"error": {...}}` bodies.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    error_type: String,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            error_type: error_type.into(),
            message: message.into(),
        }
    }

    fn upstream(worker: &Worker, message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            format!("worker {} failed: {}", worker.url, message.into()),
        )
    }

    fn no_healthy_workers() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_healthy_workers",
            "no healthy workers are currently available",
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.status.as_u16(),
            }
        }));
        (self.status, body).into_response()
    }
}

impl From<RouterError> for ApiError {
    fn from(error: RouterError) -> Self {
        match error {
            RouterError::NoHealthyWorkers { .. } => ApiError::no_healthy_workers(),
            RouterError::PolicyNotImplemented(policy) => ApiError::new(
                StatusCode::NOT_IMPLEMENTED,
                "unimplemented",
                format!("routing policy `{policy}` is not implemented yet"),
            ),
            RouterError::Config(message) => {
                ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "config_error", message)
            }
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        ApiError::new(rejection.status(), "invalid_request", rejection.body_text())
    }
}

/// Bookkeeping needed to emit one structured log record per proxied request,
/// including streaming requests whose body finishes after the handler returns.
struct RequestTelemetry {
    request_id: String,
    worker: Worker,
    policy: RoutingPolicy,
    matched_prefix_len: usize,
    metrics: Arc<Metrics>,
    started: Instant,
}

impl RequestTelemetry {
    fn finish(&self, ttft: Duration, total: Duration, upstream_prefix_hit: Option<&str>) {
        info!(
            request_id = %self.request_id,
            worker = %self.worker.url,
            pool = %self.worker.pool,
            policy = %self.policy,
            matched_prefix_len = self.matched_prefix_len,
            upstream_prefix_hit = upstream_prefix_hit.unwrap_or("n/a"),
            ttft_ms = ttft.as_millis() as u64,
            total_latency_ms = total.as_millis() as u64,
            "chat completion served"
        );
        self.metrics.observe_request(self.policy, ttft, total);
    }
}

struct ProxyStreamState {
    inner: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>,
    telemetry: RequestTelemetry,
    upstream_prefix_hit: Option<String>,
    streaming: bool,
    done_seen: bool,
    first_chunk_after: Option<Duration>,
}

/// Stream the upstream body back to the client without buffering the whole
/// response. In streaming mode, guarantees the SSE stream is terminated with
/// `data: [DONE]` even if the upstream omits it. Emits the per-request log
/// record (TTFT, total latency, ...) once the body is fully drained.
fn proxy_stream(
    upstream: reqwest::Response,
    telemetry: RequestTelemetry,
    upstream_prefix_hit: Option<String>,
    streaming: bool,
) -> impl Stream<Item = Result<Bytes, io::Error>> + Send {
    let state = ProxyStreamState {
        inner: Box::pin(upstream.bytes_stream()),
        telemetry,
        upstream_prefix_hit,
        streaming,
        done_seen: false,
        first_chunk_after: None,
    };
    unfold(state, |mut state| async move {
        match state.inner.next().await {
            Some(Ok(chunk)) => {
                if state.first_chunk_after.is_none() {
                    state.first_chunk_after = Some(state.telemetry.started.elapsed());
                }
                if state.streaming && !state.done_seen && chunk_contains_done(&chunk) {
                    state.done_seen = true;
                }
                Some((Ok(chunk), state))
            }
            Some(Err(error)) => {
                warn!(
                    request_id = %state.telemetry.request_id,
                    worker = %state.telemetry.worker.url,
                    %error,
                    "upstream stream error"
                );
                state.telemetry.metrics.inc_upstream_error();
                Some((Err(io::Error::other(error.to_string())), state))
            }
            None => {
                if state.streaming && !state.done_seen {
                    state.done_seen = true;
                    Some((Ok(Bytes::from_static(b"data: [DONE]\n\n")), state))
                } else {
                    let ttft = state
                        .first_chunk_after
                        .unwrap_or_else(|| state.telemetry.started.elapsed());
                    state.telemetry.finish(
                        ttft,
                        state.telemetry.started.elapsed(),
                        state.upstream_prefix_hit.as_deref(),
                    );
                    None
                }
            }
        }
    })
}

fn chunk_contains_done(chunk: &[u8]) -> bool {
    chunk.windows(12).any(|window| window == b"data: [DONE]")
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics))
        .with_state(state)
}

/// Liveness: the process is up.
pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// Readiness: ready once at least one worker has passed a health probe.
pub async fn ready(State(state): State<AppState>) -> Response {
    let healthy = state.registry.healthy_count();
    let body = Json(json!({ "status": "ready", "healthy_workers": healthy }));
    if state.registry.any_healthy() {
        (StatusCode::OK, body).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, body).into_response()
    }
}

pub async fn metrics(State(state): State<AppState>) -> Response {
    let body = state.metrics.render(state.registry.healthy_count());
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

/// Pass through `/v1/models` from the first healthy worker.
pub async fn list_models(State(state): State<AppState>) -> Result<Response, ApiError> {
    let healthy = state.registry.healthy_workers();
    let worker = healthy.first().ok_or_else(ApiError::no_healthy_workers)?;
    let upstream_url = format!("{}/v1/models", worker.url.trim_end_matches('/'));
    let upstream = state
        .client
        .get(upstream_url)
        .send()
        .await
        .map_err(|error| ApiError::upstream(worker, error.to_string()))?;
    if !upstream.status().is_success() {
        state.metrics.inc_upstream_error();
        return Err(ApiError::upstream(
            worker,
            format!("upstream returned {}", upstream.status()),
        ));
    }
    let body = Body::from_stream(
        upstream
            .bytes_stream()
            .map(|item| item.map_err(|error| io::Error::other(error.to_string()))),
    );
    Ok(([(header::CONTENT_TYPE, "application/json")], body).into_response())
}

/// OpenAI-compatible chat completions: select a worker, proxy the request, and
/// stream the response body back in both streaming (SSE) and non-streaming modes.
///
/// Note: axum 0.7 only implements `FromRequest` for `Result<T, T::Rejection>`,
/// so the extractor returns `JsonRejection` and we map it to `ApiError` (which
/// yields a 400) immediately. axum 0.8 generalized this to `Result<T, E>`.
pub async fn chat_completions(
    State(state): State<AppState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let payload = payload.map_err(ApiError::from)?;
    let request_id = uuid::Uuid::new_v4().to_string();
    let span_request_id = request_id.clone();
    async move {
        let Json(request) = payload;

        let decision = match state.router.select_for_chat(&request.prompt_text()) {
            Ok(decision) => decision,
            Err(error) => {
                warn!(request_id = %request_id, %error, "no worker selected");
                if matches!(error, RouterError::NoHealthyWorkers { .. }) {
                    state.metrics.inc_no_healthy_workers();
                }
                return Err(ApiError::from(error));
            }
        };
        let worker = decision.worker;
        let upstream_url = format!("{}/v1/chat/completions", worker.url.trim_end_matches('/'));

        // Clock starts before the upstream call so TTFT (first body byte) and total
        // latency include upstream queueing; otherwise non-streaming responses,
        // which arrive fully formed, would report a zero TTFT.
        let started = Instant::now();
        let upstream = state
            .client
            .post(upstream_url)
            .json(&request)
            .send()
            .await
            .map_err(|error| ApiError::upstream(&worker, error.to_string()))?;

        if !upstream.status().is_success() {
            state.metrics.inc_upstream_error();
            error!(
                request_id = %request_id,
                worker = %worker.url,
                status = upstream.status().as_u16(),
                "upstream rejected request"
            );
            return Err(ApiError::upstream(
                &worker,
                format!("upstream returned {}", upstream.status()),
            ));
        }

        let upstream_prefix_hit = upstream
            .headers()
            .get("x-kv-prefix-hit")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);

        let telemetry = RequestTelemetry {
            request_id: request_id.clone(),
            worker: worker.clone(),
            policy: state.router.policy(),
            matched_prefix_len: decision.matched_prefix_len,
            metrics: Arc::clone(&state.metrics),
            started,
        };

        let stream = proxy_stream(upstream, telemetry, upstream_prefix_hit, request.stream);
        let mut builder = Response::builder()
            .status(StatusCode::OK)
            .header("x-request-id", request_id)
            .header("x-router-worker", worker.url)
            .header("x-router-matched-prefix", decision.matched_prefix_len);
        if request.stream {
            builder = builder
                .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
                .header(header::CACHE_CONTROL, "no-cache");
        } else {
            builder = builder.header(header::CONTENT_TYPE, "application/json");
        }
        builder.body(Body::from_stream(stream)).map_err(|error| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                error.to_string(),
            )
        })
    }
    .instrument(info_span!("chat_completion", request_id = %span_request_id))
    .await
}
