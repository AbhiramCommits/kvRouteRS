use std::sync::Arc;
use std::time::Duration;

use router_core::{KvTransfer, MockKvTransfer, Router, RouterConfig, RouterError, WorkerRegistry};

use crate::metrics::Metrics;

/// Shared application state handed to every axum handler.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RouterConfig>,
    pub registry: Arc<WorkerRegistry>,
    pub router: Arc<Router>,
    pub client: reqwest::Client,
    pub metrics: Arc<Metrics>,
    /// KV-transfer connector for the disaggregated flow. Swappable behind this
    /// trait; the mock sleeps a fixed cost, a real deployment drops in a
    /// NIXL-style transfer.
    pub kv_transfer: Arc<dyn KvTransfer>,
}

impl AppState {
    pub fn new(config: RouterConfig) -> Result<Self, RouterError> {
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
            metrics: Arc::new(Metrics::default()),
            kv_transfer,
        })
    }
}
