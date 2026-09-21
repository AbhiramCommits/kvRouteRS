use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use router_core::RoutingPolicy;

/// Hand-rolled Prometheus text exposition, backed by atomics plus one mutex for
/// the labeled request counter.
///
/// Deliberately avoids the `prometheus` crate: the router exposes a handful of
/// counters and one gauge, and plain atomics are trivially safe to update from
/// the request hot path without a global registry.
#[derive(Debug, Default)]
pub struct Metrics {
    requests_by_policy: Mutex<BTreeMap<String, u64>>,
    upstream_errors: AtomicU64,
    no_healthy_workers: AtomicU64,
    health_check_failures: AtomicU64,
    ttft_micros_sum: AtomicU64,
    ttft_count: AtomicU64,
    total_micros_sum: AtomicU64,
    total_count: AtomicU64,
}

impl Metrics {
    pub fn observe_request(&self, policy: RoutingPolicy, ttft: Duration, total: Duration) {
        if let Ok(mut by_policy) = self.requests_by_policy.lock() {
            *by_policy.entry(policy.to_string()).or_insert(0) += 1;
        }
        self.ttft_micros_sum
            .fetch_add(micros(ttft), Ordering::Relaxed);
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
        self.total_micros_sum
            .fetch_add(micros(total), Ordering::Relaxed);
        self.total_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_upstream_error(&self) {
        self.upstream_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_no_healthy_workers(&self) {
        self.no_healthy_workers.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_health_check_failure(&self) {
        self.health_check_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Render the full exposition format, with the current healthy-worker gauge.
    pub fn render(&self, healthy_workers: usize) -> String {
        let mut out = String::new();
        let by_policy = self
            .requests_by_policy
            .lock()
            .map(|entries| entries.clone())
            .unwrap_or_default();
        counter(
            &mut out,
            "router_requests_total",
            "Chat completion requests routed.",
            &by_policy,
        );
        plain_counter(
            &mut out,
            "router_upstream_errors_total",
            "Requests that failed at the upstream worker.",
            self.upstream_errors.load(Ordering::Relaxed),
        );
        plain_counter(
            &mut out,
            "router_no_healthy_workers_total",
            "Requests rejected because no healthy worker was available.",
            self.no_healthy_workers.load(Ordering::Relaxed),
        );
        plain_counter(
            &mut out,
            "router_health_check_failures_total",
            "Failed worker health probes.",
            self.health_check_failures.load(Ordering::Relaxed),
        );
        seconds_counter(
            &mut out,
            "router_ttft_seconds",
            "Time-to-first-token for routed requests.",
            self.ttft_micros_sum.load(Ordering::Relaxed),
            self.ttft_count.load(Ordering::Relaxed),
        );
        seconds_counter(
            &mut out,
            "router_request_latency_seconds",
            "End-to-end latency for routed requests.",
            self.total_micros_sum.load(Ordering::Relaxed),
            self.total_count.load(Ordering::Relaxed),
        );
        gauge(
            &mut out,
            "router_healthy_workers",
            "Workers currently marked healthy.",
            healthy_workers,
        );
        out
    }
}

fn micros(duration: Duration) -> u64 {
    duration.as_micros() as u64
}

fn counter(out: &mut String, name: &str, help: &str, labeled: &BTreeMap<String, u64>) {
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n"));
    if labeled.is_empty() {
        out.push_str(&format!("{name} 0\n"));
    } else {
        for (label, value) in labeled {
            out.push_str(&format!("{name}{{policy=\"{label}\"}} {value}\n"));
        }
    }
}

fn plain_counter(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str(&format!(
        "# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}\n"
    ));
}

fn seconds_counter(out: &mut String, name: &str, help: &str, micros_sum: u64, count: u64) {
    out.push_str(&format!(
        "# HELP {name}_sum {help} (sum, seconds)\n# TYPE {name}_sum counter\n{name}_sum {}\n",
        micros_sum as f64 / 1_000_000.0
    ));
    out.push_str(&format!(
        "# HELP {name}_count {help} (count)\n# TYPE {name}_count counter\n{name}_count {count}\n"
    ));
}

fn gauge(out: &mut String, name: &str, help: &str, value: usize) {
    out.push_str(&format!(
        "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}\n"
    ));
}
