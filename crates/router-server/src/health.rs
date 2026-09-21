use std::time::Duration;

use tracing::{info, warn};

use crate::state::AppState;

/// Background task that probes each worker's `/health` endpoint every
/// `health_check_interval_secs` and mirrors the result into the shared registry.
/// Requests consult the registry on every selection, so a failed probe
/// immediately removes the worker from rotation.
pub fn spawn_health_poller(state: AppState) {
    tokio::spawn(async move {
        let interval = Duration::from_secs(state.config.health_check_interval_secs.max(1));
        loop {
            for worker in state.registry.all_workers() {
                let probe_url = format!("{}/health", worker.url.trim_end_matches('/'));
                let healthy = match state.client.get(&probe_url).send().await {
                    Ok(response) if response.status().is_success() => true,
                    Ok(response) => {
                        warn!(
                            worker = %worker.url,
                            status = response.status().as_u16(),
                            "health probe failed"
                        );
                        false
                    }
                    Err(error) => {
                        warn!(worker = %worker.url, %error, "health probe error");
                        false
                    }
                };
                if !healthy {
                    state.metrics.inc_health_check_failure();
                }
                if state.registry.set_healthy(worker.id, healthy) {
                    info!(worker = %worker.url, healthy, "worker health state changed");
                }
            }
            tokio::time::sleep(interval).await;
        }
    });
}
