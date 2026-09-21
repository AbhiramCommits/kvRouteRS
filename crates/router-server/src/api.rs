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
use router_core::{
    chain_hash, ChatCompletionRequest, InflightGuard, Pool, RouterError, RoutingPolicy, Worker,
};
use serde_json::{json, Value};
use tracing::{error, info, info_span, warn, Instrument};

use crate::metrics::Metrics;
use crate::state::AppState;

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
            RouterError::NoHealthyWorkers { .. } | RouterError::NoHealthyWorkersInPool { .. } => {
                ApiError::no_healthy_workers()
            }
            RouterError::KvTransfer { reason, .. } => {
                ApiError::new(StatusCode::BAD_GATEWAY, "kv_transfer_error", reason)
            }
            RouterError::PolicyNotImplemented(policy) => ApiError::new(
                StatusCode::NOT_IMPLEMENTED,
                "unimplemented",
                format!("routing policy `{policy}` is not implemented yet"),
            ),
            RouterError::DisaggregatedRequiresTwoPhase => ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "invalid_routing_mode",
                "disaggregated policy requires two-phase routing; this endpoint used single-phase",
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
    matched_prefix_blocks: usize,
    score: f64,
    metrics: Arc<Metrics>,
    started: Instant,
    /// Held until the response body is fully drained (or the request fails),
    /// releasing the worker's in-flight count exactly once via `Drop`. Never
    /// read directly — its lifetime is the point.
    _guard: InflightGuard,
    details: CompletionDetails,
}

#[derive(Debug)]
enum CompletionDetails {
    Single,
    Disaggregated {
        prefill_worker: String,
        prefill_ms: u64,
        kv_transfer_ms: u64,
    },
}

impl RequestTelemetry {
    fn finish(&self, ttft: Duration, total: Duration, upstream_prefix_hit: Option<&str>) {
        match &self.details {
            CompletionDetails::Single => info!(
                request_id = %self.request_id,
                worker = %self.worker.url,
                pool = %self.worker.pool,
                policy = %self.policy,
                matched_prefix_blocks = self.matched_prefix_blocks,
                score = self.score,
                upstream_prefix_hit = upstream_prefix_hit.unwrap_or("n/a"),
                ttft_ms = ttft.as_millis() as u64,
                total_latency_ms = total.as_millis() as u64,
                "chat completion served"
            ),
            CompletionDetails::Disaggregated {
                prefill_worker,
                prefill_ms,
                kv_transfer_ms,
            } => info!(
                request_id = %self.request_id,
                prefill_worker = %prefill_worker,
                decode_worker = %self.worker.url,
                pool = %self.worker.pool,
                policy = %self.policy,
                matched_prefix_blocks = self.matched_prefix_blocks,
                score = self.score,
                prefill_ms,
                kv_transfer_ms,
                upstream_prefix_hit = upstream_prefix_hit.unwrap_or("n/a"),
                ttft_ms = ttft.as_millis() as u64,
                total_latency_ms = total.as_millis() as u64,
                "chat completion served (disaggregated)"
            ),
        }
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
/// record (TTFT, total latency, ...) — and drops the in-flight guard — once
/// the body is fully drained.
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

fn prefix_hit_header(upstream: &reqwest::Response) -> Option<String> {
    upstream
        .headers()
        .get("x-kv-prefix-hit")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

/// Build the client-facing response around an upstream body stream.
fn build_response(
    upstream: reqwest::Response,
    telemetry: RequestTelemetry,
    upstream_prefix_hit: Option<String>,
    streaming: bool,
) -> Result<Response, ApiError> {
    let request_id = telemetry.request_id.clone();
    let worker_url = telemetry.worker.url.clone();
    let matched_blocks = telemetry.matched_prefix_blocks;
    let stream = proxy_stream(upstream, telemetry, upstream_prefix_hit, streaming);
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header("x-request-id", request_id)
        .header("x-router-worker", worker_url)
        .header("x-router-matched-prefix", matched_blocks);
    if streaming {
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
    let body = state.metrics.render(
        state.registry.healthy_count(),
        state.router.inflight().total(),
        state.router.prefix_index().len(),
    );
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

/// OpenAI-compatible chat completions. Dispatches to the single-phase proxy or
/// the two-phase (prefill -> KV transfer -> decode) disaggregated flow.
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
        if state.config.routing_policy == RoutingPolicy::Disaggregated {
            serve_disaggregated(&state, &request, &request_id).await
        } else {
            serve_proxied(&state, &request, &request_id).await
        }
    }
    .instrument(info_span!("chat_completion", request_id = %span_request_id))
    .await
}

/// Route to a single worker and stream the upstream response back unchanged
/// (`round_robin` and `cache_aware` policies).
async fn serve_proxied(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
) -> Result<Response, ApiError> {
    let decision = match state.router.select_for_chat(request) {
        Ok(decision) => decision,
        Err(error) => {
            warn!(request_id, %error, "no worker selected");
            if matches!(error, RouterError::NoHealthyWorkers { .. }) {
                state.metrics.inc_no_healthy_workers();
            }
            return Err(ApiError::from(error));
        }
    };
    let worker = decision.worker;
    let guard = InflightGuard::acquire(&state.router.inflight(), worker.id);
    let upstream_url = format!("{}/v1/chat/completions", worker.url.trim_end_matches('/'));

    // Clock starts before the upstream call so TTFT (first body byte) and total
    // latency include upstream queueing; otherwise non-streaming responses,
    // which arrive fully formed, would report a zero TTFT.
    let started = Instant::now();
    let upstream = state
        .client
        .post(upstream_url)
        .json(request)
        .send()
        .await
        .map_err(|error| ApiError::upstream(&worker, error.to_string()))?;

    if !upstream.status().is_success() {
        state.metrics.inc_upstream_error();
        error!(
            request_id,
            worker = %worker.url,
            status = upstream.status().as_u16(),
            "upstream rejected request"
        );
        return Err(ApiError::upstream(
            &worker,
            format!("upstream returned {}", upstream.status()),
        ));
    }

    let upstream_prefix_hit = prefix_hit_header(&upstream);
    let telemetry = RequestTelemetry {
        request_id: request_id.to_string(),
        worker: worker.clone(),
        policy: state.router.policy(),
        matched_prefix_blocks: decision.matched_prefix_len,
        score: decision.score,
        metrics: Arc::clone(&state.metrics),
        started,
        _guard: guard,
        details: CompletionDetails::Single,
    };
    build_response(upstream, telemetry, upstream_prefix_hit, request.stream)
}

/// Two-phase routing for the `disaggregated` policy: prefill on a `prefill`
/// pool worker, simulated KV transfer, then decode on a `decode` pool worker
/// whose stream is what the client actually sees.
async fn serve_disaggregated(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
) -> Result<Response, ApiError> {
    let prompt = request.canonical_prompt();
    let chain = chain_hash(&prompt, state.config.prompt_block_chars);

    // ---- prefill phase: materialize the KV cache ---------------------------
    let prefill = state
        .router
        .select_in_pool(Pool::Prefill, &chain)
        .map_err(|error| {
            warn!(request_id, pool = "prefill", %error, "no worker selected in pool");
            ApiError::from(error)
        })?;
    let prefill_url = format!(
        "{}/v1/chat/completions",
        prefill.worker.url.trim_end_matches('/')
    );
    let prefill_ms = {
        let _guard = InflightGuard::acquire(&state.router.inflight(), prefill.worker.id);
        let started = Instant::now();
        let prefill_response = state
            .client
            .post(prefill_url)
            .json(request)
            .header("x-router-phase", "prefill")
            .send()
            .await
            .map_err(|error| ApiError::upstream(&prefill.worker, error.to_string()))?;
        if !prefill_response.status().is_success() {
            state.metrics.inc_upstream_error();
            return Err(ApiError::upstream(
                &prefill.worker,
                format!("prefill phase returned {}", prefill_response.status()),
            ));
        }
        let mut body = prefill_response.bytes_stream();
        while let Some(chunk) = body.next().await {
            if let Err(error) = chunk {
                warn!(
                    request_id,
                    worker = %prefill.worker.url,
                    %error,
                    "error draining prefill response"
                );
                break;
            }
        }
        started.elapsed().as_millis() as u64
    }; // prefill in-flight guard drops here

    // ---- KV transfer: move the blocks to the decode worker -----------------
    let decode = state
        .router
        .select_in_pool(Pool::Decode, &chain)
        .map_err(|error| {
            warn!(request_id, pool = "decode", %error, "no worker selected in pool");
            ApiError::from(error)
        })?;
    let transfer_started = Instant::now();
    state
        .kv_transfer
        .transfer(prefill.worker.id, decode.worker.id, chain.len())
        .await
        .map_err(ApiError::from)?;
    let kv_transfer_ms = transfer_started.elapsed().as_millis() as u64;
    state
        .metrics
        .observe_kv_transfer(Duration::from_millis(kv_transfer_ms));

    // ---- decode phase: what the client actually sees -----------------------
    let guard = InflightGuard::acquire(&state.router.inflight(), decode.worker.id);
    let started = Instant::now();
    let decode_url = format!(
        "{}/v1/chat/completions",
        decode.worker.url.trim_end_matches('/')
    );
    let decode_response = state
        .client
        .post(decode_url)
        .json(request)
        .header("x-router-phase", "decode")
        .send()
        .await
        .map_err(|error| ApiError::upstream(&decode.worker, error.to_string()))?;
    if !decode_response.status().is_success() {
        state.metrics.inc_upstream_error();
        error!(
            request_id,
            worker = %decode.worker.url,
            status = decode_response.status().as_u16(),
            "decode phase failed"
        );
        return Err(ApiError::upstream(
            &decode.worker,
            format!("decode phase returned {}", decode_response.status()),
        ));
    }

    let upstream_prefix_hit = prefix_hit_header(&decode_response);
    let telemetry = RequestTelemetry {
        request_id: request_id.to_string(),
        worker: decode.worker.clone(),
        policy: state.router.policy(),
        matched_prefix_blocks: decode.matched_prefix_len,
        score: decode.score,
        metrics: Arc::clone(&state.metrics),
        started,
        _guard: guard,
        details: CompletionDetails::Disaggregated {
            prefill_worker: prefill.worker.url,
            prefill_ms,
            kv_transfer_ms,
        },
    };
    build_response(
        decode_response,
        telemetry,
        upstream_prefix_hit,
        request.stream,
    )
}
