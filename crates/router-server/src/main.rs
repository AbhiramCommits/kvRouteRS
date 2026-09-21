mod api;
mod health;
mod metrics;
mod state;

use std::error::Error;
use std::net::SocketAddr;
use std::path::PathBuf;

use router_core::RouterConfig;
use tracing::info;
use tracing_subscriber::EnvFilter;

const DEFAULT_CONFIG_PATH: &str = "config/router.yaml";
const DEFAULT_BIND_ADDRESS: &str = "0.0.0.0:8080";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
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
        "router configuration loaded"
    );

    let bind_address: SocketAddr = std::env::var("ROUTER_BIND")
        .unwrap_or_else(|_| DEFAULT_BIND_ADDRESS.to_string())
        .parse()?;

    let state = state::AppState::new(config.clone())?;
    health::spawn_health_poller(state.clone());

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
