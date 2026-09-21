use std::time::Duration;

use tracing::{info, warn};

use crate::state::AppState;

/// Optional engine-side observability: polls each worker's `/metrics` endpoint
/// and records the engine's *reported* prefix cache hit rate next to the
/// router's *predicted* one (see `kvrouter_hit_rate_prediction_error` in
/// `metrics.rs`). Enabled by `metrics_scrape: true` in the router config.
///
/// The mock backends do not expose engine metrics; this is aimed at real
/// vLLM/SGLang deployments (`deploy/vllm`, `deploy/sglang`).
pub fn spawn_engine_metrics_scraper(state: AppState) {
    if !state.config.metrics_scrape {
        info!("engine metrics scraping disabled (metrics_scrape: false)");
        return;
    }
    let metric_name = state.config.engine_cache_hit_metric.clone();
    let interval = Duration::from_secs(state.config.engine_metrics_interval_secs.max(1));
    tokio::spawn(async move {
        loop {
            for worker in state.registry.all_workers() {
                if !state.registry.is_healthy(worker.id) {
                    continue;
                }
                let scrape_url = format!("{}/metrics", worker.url.trim_end_matches('/'));
                match state.client.get(&scrape_url).send().await {
                    Ok(response) if response.status().is_success() => match response.text().await {
                        Ok(text) => {
                            if let Some(rate) =
                                crate::prometheus_text::find_metric(&text, &metric_name)
                            {
                                crate::metrics::set_engine_cache_hit_rate(&worker.url, rate);
                                if let Some(predicted) =
                                    crate::metrics::predicted_cache_hit_rate(&worker.url)
                                {
                                    crate::metrics::set_predicted_cache_hit_rate(
                                        &worker.url,
                                        predicted,
                                    );
                                    crate::metrics::set_hit_rate_prediction_error(
                                        &worker.url,
                                        predicted - rate,
                                    );
                                }
                            }
                        }
                        Err(error) => warn!(
                            worker = %worker.url,
                            %error,
                            "failed to read engine metrics body"
                        ),
                    },
                    Ok(response) => warn!(
                        worker = %worker.url,
                        status = response.status().as_u16(),
                        "engine metrics scrape failed"
                    ),
                    Err(error) => {
                        warn!(worker = %worker.url, %error, "engine metrics scrape error");
                    }
                }
            }
            tokio::time::sleep(interval).await;
        }
    });
}
