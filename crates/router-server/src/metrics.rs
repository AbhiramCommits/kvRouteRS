//! `kvrouter_*` metrics, recorded through the `metrics` crate facade and
//! rendered by `metrics-exporter-prometheus` at `GET /metrics`.
//!
//! Histograms use the crate's dynamic log-scale buckets, which span (and
//! exceed) the 10ms–10s range the TTFT and duration histograms care about.
//! Counters are cumulative; hit rate is derived as
//! `rate(kvrouter_cache_hit_blocks) / rate(kvrouter_prompt_blocks_total)`.

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};

use router_core::RoutingPolicy;

use crate::state::AppState;

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
