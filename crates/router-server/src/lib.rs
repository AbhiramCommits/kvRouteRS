//! kvRouteRS router server, exposed as a library so integration tests (e.g.
//! the real-engine e2e test) can drive the router in-process.

pub mod api;
pub mod engine_metrics;
pub mod health;
pub mod metrics;
pub mod prometheus_text;
pub mod state;

use std::error::Error;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use router_core::{PrefixIndex, RouterConfig};
use tracing::info;
use tracing_subscriber::EnvFilter;

pub const DEFAULT_CONFIG_PATH: &str = "config/router.yaml";
pub const DEFAULT_BIND_ADDRESS: &str = "0.0.0.0:8080";

/// Run the router server: load configuration, spawn the health poller, prefix
/// index eviction and (optionally) engine metrics scrapers, then serve until
/// shutdown.
pub async fn serve() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config_path = std::env::var("ROUTER_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_CONFIG_PATH));
    let config = RouterConfig::load(&config_path)?;
    info!(
        config = %config_path.display(),
        workers = config.workers.len(),
        policy = %config.routing_policy,
        health_check_interval_secs = config.health_check_interval_secs,
        metrics_scrape = config.metrics_scrape,
        "router configuration loaded"
    );

    let bind_address: SocketAddr = std::env::var("ROUTER_BIND")
        .unwrap_or_else(|_| DEFAULT_BIND_ADDRESS.to_string())
        .parse()?;

    let state = state::AppState::new(config.clone())?;
    health::spawn_health_poller(state.clone());
    engine_metrics::spawn_engine_metrics_scraper(state.clone());
    // Bounded prefix index: TTL expiry + LRU cap, applied on a detached
    // background task so the request path never pays eviction cost.
    PrefixIndex::spawn_eviction_task(
        state.router.prefix_index(),
        Duration::from_secs(config.prefix_index_ttl_secs),
        config.prefix_index_max_entries,
    );

    let app = api::build_router(state);
    let listener = tokio::net::TcpListener::bind(bind_address).await?;
    info!(%bind_address, "router listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    info!("router shut down");
    Ok(())
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::error!(%error, "failed to install ctrl-c handler");
        std::future::pending::<()>().await;
    }
}
