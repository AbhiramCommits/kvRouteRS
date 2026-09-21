use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use tracing::info;

use crate::error::RouterError;
use crate::worker::WorkerId;

/// Boxed future returned by [`KvTransfer::transfer`].
pub type KvTransferFuture<'a> = Pin<Box<dyn Future<Output = Result<(), RouterError>> + Send + 'a>>;

/// Abstraction over moving KV-cache blocks between workers.
///
/// In the disaggregated flow the prefill worker owns the prompt's KV blocks
/// after prefill; before the decode phase can run on a decode-pool worker, the
/// blocks must move. A production deployment drops in a real connector here —
/// a NIXL-style point-to-point KV transfer, a UCX/shared-memory channel, or the
/// engine's native kv-transfer API — behind `Arc<dyn KvTransfer>`. The router
/// core depends only on this trait, so the connector is swappable without
/// touching routing logic.
pub trait KvTransfer: Send + Sync {
    /// Move `blocks` KV blocks of the current prompt from `from` to `to`.
    fn transfer(&self, from: WorkerId, to: WorkerId, blocks: usize) -> KvTransferFuture<'_>;
}

/// Simulated connector: sleeps `cost` and logs the transfer. Kept in-core so
/// the disaggregated flow is runnable and testable without any real fabric.
#[derive(Debug, Clone)]
pub struct MockKvTransfer {
    cost: Duration,
}

impl MockKvTransfer {
    /// A mock transfer that sleeps `cost` to simulate fabric latency.
    pub fn new(cost: Duration) -> Self {
        Self { cost }
    }
}

impl KvTransfer for MockKvTransfer {
    fn transfer(&self, from: WorkerId, to: WorkerId, blocks: usize) -> KvTransferFuture<'_> {
        let cost = self.cost;
        Box::pin(async move {
            tokio::time::sleep(cost).await;
            info!(
                from,
                to,
                blocks,
                transfer_ms = cost.as_millis() as u64,
                "simulated KV transfer"
            );
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_transfer_succeeds() {
        let transfer = MockKvTransfer::new(Duration::from_millis(1));
        transfer.transfer(0, 1, 4).await.unwrap();
    }
}
