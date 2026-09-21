use std::sync::Arc;
use std::time::Duration;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use router_core::{KvTransfer, MockKvTransfer, Router, RouterConfig, RouterError, WorkerRegistry};

/// Shared application state handed to every axum handler.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RouterConfig>,
    pub registry: Arc<WorkerRegistry>,
    pub router: Arc<Router>,
    pub client: reqwest::Client,
    /// Prometheus exposition handle; `/metrics` renders from it.
    pub prometheus: PrometheusHandle,
    /// KV-transfer connector for the disaggregated flow. Swappable behind this
    /// trait; the mock sleeps a fixed cost, a real deployment drops in a
    /// NIXL-style transfer.
    pub kv_transfer: Arc<dyn KvTransfer>,
}

impl AppState {
    pub fn new(config: RouterConfig) -> Result<Self, RouterError> {
        // Install the global metrics recorder before anything records. The
        // exporter is used in handle-only mode: axum serves /metrics, not the
        // exporter's own HTTP listener.
        let prometheus = PrometheusBuilder::new()
            .install_recorder()
            .map_err(|error| {
                RouterError::Config(format!("failed to install metrics recorder: {error}"))
            })?;
        crate::metrics::register_descriptions();

        let registry = Arc::new(WorkerRegistry::from_config(&config.workers));
        let router = Arc::new(Router::new(Arc::clone(&registry), &config));
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|error| {
                RouterError::Config(format!("failed to build http client: {error}"))
            })?;
        let kv_transfer: Arc<dyn KvTransfer> = Arc::new(MockKvTransfer::new(
            Duration::from_millis(config.kv_transfer_cost_ms),
        ));
        Ok(Self {
            config: Arc::new(config),
            registry,
            router,
            client,
            prometheus,
            kv_transfer,
        })
    }
}
