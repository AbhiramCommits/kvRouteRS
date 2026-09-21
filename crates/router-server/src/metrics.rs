//! `kvrouter_*` metrics, recorded through the `metrics` crate facade and
//! rendered by `metrics-exporter-prometheus` at `GET /metrics`.
//!
//! Histograms use the crate's dynamic log-scale buckets, which span (and
//! exceed) the 10ms–10s range the TTFT and duration histograms care about.
//! Counters are cumulative; hit rate is derived as
//! `rate(kvrouter_cache_hit_blocks) / rate(kvrouter_prompt_blocks_total)`.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};

use router_core::RoutingPolicy;

use crate::state::AppState;

/// Per-worker cumulative (hit blocks, prompt blocks) from routing decisions.
///
/// Kept outside the `metrics` facade because the facade is write-only: the
/// engine metrics scraper needs to READ these back to derive the predicted hit
/// rate gauge, and only the derived gauges are exposed in the exposition.
static HIT_STATS: OnceLock<Mutex<BTreeMap<String, (u64, u64)>>> = OnceLock::new();

fn hit_stats() -> &'static Mutex<BTreeMap<String, (u64, u64)>> {
    HIT_STATS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Register HELP text for every metric. Must run once after the recorder is
/// installed.
pub fn register_descriptions() {
    describe_counter!(
        "kvrouter_requests_total",
        "Chat completion requests by worker, policy and status."
    );
    describe_counter!(
        "kvrouter_cache_hit_blocks",
        "Cache-hit blocks credited to routing decisions (matched prefix length)."
    );
    describe_counter!(
        "kvrouter_prompt_blocks_total",
        "Prompt blocks across routed requests."
    );
    describe_histogram!("kvrouter_ttft_seconds", "Time to first token, seconds.");
    describe_histogram!(
        "kvrouter_request_duration_seconds",
        "End-to-end request duration, seconds."
    );
    describe_counter!(
        "kvrouter_tokens_generated_total",
        "Tokens streamed back to clients, counted from streamed chunks."
    );
    describe_gauge!(
        "kvrouter_inflight_requests",
        "Requests currently in flight, per worker."
    );
    describe_gauge!(
        "kvrouter_prefix_index_entries",
        "Router-side prefix index size."
    );
    describe_gauge!(
        "kvrouter_worker_up",
        "1 if the worker is currently healthy."
    );
    describe_histogram!(
        "kvrouter_kv_transfer_seconds",
        "KV transfer duration between prefill and decode workers (disaggregated mode)."
    );
    describe_gauge!(
        "kvrouter_engine_cache_hit_rate",
        "Engine-reported prefix cache hit rate (metrics_scrape mode)."
    );
    describe_gauge!(
        "kvrouter_predicted_cache_hit_rate",
        "Router-predicted prefix cache hit rate from its belief index."
    );
    describe_gauge!(
        "kvrouter_hit_rate_prediction_error",
        "Predicted minus engine-reported cache hit rate."
    );
}

/// Record a completed request's (hit blocks, prompt blocks) against a worker.
/// Feeds the predicted hit rate gauge used by the engine metrics scraper.
pub fn record_hit_stats(worker: &str, matched_blocks: u64, prompt_blocks: u64) {
    let mut stats = hit_stats()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = stats.entry(worker.to_string()).or_insert((0, 0));
    entry.0 += matched_blocks;
    entry.1 += prompt_blocks;
}

/// Cumulative router-predicted hit rate for a worker, if it has served traffic.
pub fn predicted_cache_hit_rate(worker: &str) -> Option<f64> {
    let stats = hit_stats()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (hits, prompts) = *stats.get(worker)?;
    (prompts > 0).then(|| hits as f64 / prompts as f64)
}

pub fn set_engine_cache_hit_rate(worker: &str, rate: f64) {
    gauge!("kvrouter_engine_cache_hit_rate", "worker" => worker.to_owned()).set(rate);
}

pub fn set_predicted_cache_hit_rate(worker: &str, rate: f64) {
    gauge!("kvrouter_predicted_cache_hit_rate", "worker" => worker.to_owned()).set(rate);
}

pub fn set_hit_rate_prediction_error(worker: &str, error: f64) {
    // The single most interesting number this project produces: the gap
    // between the hit rate the router *predicted* from its belief index and
    // the hit rate the engine *actually reported*. Positive error = the
    // router over-credits cache residency (stale beliefs: the engine evicted
    // blocks we still remember, or our 512-char blocks disagree with the
    // engine's 16-token blocks). Negative error = the engine holds more than
    // we recorded (traffic that bypassed the router, or the engine's own
    // admission). Track it, don't hide it — it bounds how much mis-routing
    // the belief-based index actually causes.
    gauge!("kvrouter_hit_rate_prediction_error", "worker" => worker.to_owned()).set(error);
}

pub fn record_request(worker: &str, policy: RoutingPolicy, status: u16) {
    counter!(
        "kvrouter_requests_total",
        "worker" => worker.to_owned(),
        "policy" => policy.to_string(),
        "status" => status.to_string()
    )
    .increment(1);
}

pub fn record_cache_hit_blocks(blocks: u64) {
    counter!("kvrouter_cache_hit_blocks").increment(blocks);
}

pub fn record_prompt_blocks(blocks: u64) {
    counter!("kvrouter_prompt_blocks_total").increment(blocks);
}

pub fn record_ttft(policy: RoutingPolicy, seconds: f64) {
    histogram!("kvrouter_ttft_seconds", "policy" => policy.to_string()).record(seconds);
}

pub fn record_request_duration(policy: RoutingPolicy, seconds: f64) {
    histogram!("kvrouter_request_duration_seconds", "policy" => policy.to_string()).record(seconds);
}

pub fn record_tokens_generated(worker: &str, tokens: u64) {
    counter!("kvrouter_tokens_generated_total", "worker" => worker.to_owned()).increment(tokens);
}

pub fn record_kv_transfer(seconds: f64) {
    histogram!("kvrouter_kv_transfer_seconds").record(seconds);
}

pub fn set_inflight(worker: &str, count: u64) {
    gauge!("kvrouter_inflight_requests", "worker" => worker.to_owned()).set(count as f64);
}

pub fn set_prefix_index_entries(entries: u64) {
    gauge!("kvrouter_prefix_index_entries").set(entries as f64);
}

pub fn set_worker_up(worker: &str, up: bool) {
    gauge!("kvrouter_worker_up", "worker" => worker.to_owned()).set(if up { 1.0 } else { 0.0 });
}

/// Re-derive live gauges from their sources right before a scrape: worker
/// health from the registry, in-flight counts from the tracker, and index size
/// from the prefix index. Gauges are otherwise only written on transitions,
/// which would make them lag reality between scrapes.
pub fn refresh_gauges(state: &AppState) {
    for worker in state.registry.all_workers() {
        let healthy = state.registry.is_healthy(worker.id);
        set_worker_up(&worker.url, healthy);
        set_inflight(&worker.url, state.router.inflight().count(worker.id));
    }
    set_prefix_index_entries(state.router.prefix_index().len() as u64);
}
