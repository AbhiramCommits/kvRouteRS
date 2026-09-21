//! Kubernetes worker discovery.
//!
//! Instead of a static worker list, the router can watch the EndpointSlices
//! backing a headless service and reconcile the worker registry with the live
//! set of pod addresses:
//!
//! - new endpoints are registered as workers (marked unhealthy until the
//!   health poller confirms them),
//! - removed endpoints are dropped from the registry AND purged from the
//!   prefix index — the dead pod's KV cache is gone, and a replacement pod
//!   (new IP, new identity) must not inherit its cache affinity,
//! - when not running in-cluster, discovery logs a warning and falls back to
//!   the static `workers` list from the config.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use futures_util::{StreamExt, TryStreamExt};
use k8s_openapi::api::discovery::v1::{Endpoint, EndpointSlice};
use kube::runtime::watcher::{self, Event};
use kube::{Api, Client, Config, ResourceExt};
use router_core::{Pool, PrefixIndex, WorkerRegistry};
use tracing::{info, warn};

use crate::state::AppState;

/// Worker registrations owned by discovery, keyed by URL.
#[derive(Debug, Default)]
pub struct DiscoveredWorkers {
    by_url: HashMap<String, u64>,
}

/// Reconcile the registry and prefix index against the desired worker set.
///
/// Returns `(removed, added)`. Pure and unit-tested; the watcher layer above
/// only feeds it sets of endpoint URLs.
pub fn reconcile(
    registry: &WorkerRegistry,
    index: &PrefixIndex,
    discovered: &mut DiscoveredWorkers,
    desired_urls: &HashSet<String>,
) -> (usize, usize) {
    let gone: Vec<String> = discovered
        .by_url
        .keys()
        .filter(|url| !desired_urls.contains(*url))
        .cloned()
        .collect();
    for url in &gone {
        if let Some(id) = discovered.by_url.remove(url) {
            index.remove_worker(id);
            registry.remove_worker(id);
        }
    }
    let mut added = 0;
    for url in desired_urls {
        if !discovered.by_url.contains_key(url) {
            let id = registry.register_worker(url.clone(), Pool::Both);
            discovered.by_url.insert(url.clone(), id);
            added += 1;
        }
    }
    (gone.len(), added)
}

/// Worker URLs advertised by one EndpointSlice (ready addresses only).
fn endpoint_urls(slice: &EndpointSlice) -> HashSet<String> {
    let port = slice
        .ports
        .as_ref()
        .and_then(|ports| ports.first())
        .and_then(|port| port.port)
        .unwrap_or(8001);
    let mut urls = HashSet::new();
    for endpoint in &slice.endpoints {
        if !endpoint_is_ready(endpoint) {
            continue;
        }
        for address in &endpoint.addresses {
            // Headless-service worker endpoints are plain HTTP; IPv4 pod IPs.
            urls.insert(format!("http://{address}:{port}"));
        }
    }
    urls
}

fn endpoint_is_ready(endpoint: &Endpoint) -> bool {
    endpoint
        .conditions
        .as_ref()
        .is_none_or(|conditions| conditions.ready.unwrap_or(false))
}

/// Spawn the discovery loop. No-ops (with a warning) when not in-cluster, so
/// the static worker list stays the fallback.
pub fn spawn_worker_discovery(state: AppState) {
    tokio::spawn(async move {
        run_discovery(state).await;
    });
}

async fn run_discovery(state: AppState) {
    let kube_config = match Config::incluster() {
        Ok(config) => config,
        Err(error) => {
            warn!(%error, "not running in-cluster; keeping the static worker configuration");
            return;
        }
    };
    let client = match Client::try_from(kube_config) {
        Ok(client) => client,
        Err(error) => {
            warn!(%error, "failed to build a Kubernetes client; keeping the static worker configuration");
            return;
        }
    };
    let namespace = state
        .config
        .discovery
        .namespace
        .clone()
        .unwrap_or_else(|| client.default_namespace().to_string());
    let service = state.config.discovery.service.clone();
    info!(%namespace, %service, "worker discovery started");

    let mut discovered = DiscoveredWorkers::default();
    let mut backoff = Duration::from_secs(1);
    loop {
        match watch_once(&client, &namespace, &service, &state, &mut discovered).await {
            Ok(()) => backoff = Duration::from_secs(1),
            Err(error) => {
                warn!(
                    %error,
                    retry_in_secs = backoff.as_secs(),
                    "worker discovery watch failed; retrying"
                );
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(30));
            }
        }
    }
}

async fn watch_once(
    client: &Client,
    namespace: &str,
    service: &str,
    state: &AppState,
    discovered: &mut DiscoveredWorkers,
) -> Result<(), watcher::Error> {
    let api: Api<EndpointSlice> = Api::namespaced(client.clone(), namespace);
    let params =
        watcher::Config::default().labels(&format!("kubernetes.io/service-name={service}"));
    let mut stream = watcher::watcher(api, params).boxed();

    // A service can back multiple slices; track each slice's endpoints and
    // reconcile against the union, so an event for one slice never removes
    // the workers advertised by another.
    let mut slices: HashMap<String, HashSet<String>> = HashMap::new();
    while let Some(event) = stream.try_next().await? {
        match event {
            Event::InitApply(slice) | Event::Apply(slice) => {
                let name = slice.name_any();
                let urls = endpoint_urls(&slice);
                slices.insert(name, urls);
            }
            Event::Delete(slice) => {
                slices.remove(&slice.name_any());
            }
            // Watcher lifecycle markers carry no object; the watcher itself
            // recovers from restarts internally.
            Event::Init | Event::InitDone => {}
        }
        let desired: HashSet<String> = slices.values().flatten().cloned().collect();
        let (removed, added) = reconcile(
            &state.registry,
            &state.router.prefix_index(),
            discovered,
            &desired,
        );
        if removed + added > 0 {
            // Keep the in-flight tracker large enough for newly discovered ids.
            state
                .router
                .inflight()
                .ensure_capacity(discovered.by_url.len() + 16);
            info!(
                added,
                removed,
                workers = desired.len(),
                "worker discovery reconciled"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use router_core::WorkerRegistry;

    fn url(ip: &str) -> String {
        format!("http://{ip}:8001")
    }

    #[test]
    fn reconcile_adds_and_removes_workers() {
        let registry = WorkerRegistry::from_config(&[]);
        let index = PrefixIndex::new();
        let mut discovered = DiscoveredWorkers::default();

        let (removed, added) = reconcile(
            &registry,
            &index,
            &mut discovered,
            &HashSet::from([url("10.0.0.1"), url("10.0.0.2")]),
        );
        assert_eq!((removed, added), (0, 2));
        assert_eq!(registry.all_workers().len(), 2);

        // One pod disappears: removed from registry, prefix index purged.
        let chain = router_core::chain_hash("shared prompt", 512);
        let survivor = discovered.by_url[&url("10.0.0.2")];
        let dead = discovered.by_url[&url("10.0.0.1")];
        index.record(dead, &chain);
        index.record(survivor, &chain);
        assert_eq!(index.longest_match(dead, &chain), chain.len());

        let (removed, added) = reconcile(
            &registry,
            &index,
            &mut discovered,
            &HashSet::from([url("10.0.0.2")]),
        );
        assert_eq!((removed, added), (1, 0));
        assert_eq!(registry.all_workers().len(), 1);
        assert_eq!(registry.get(dead), None);
        assert_eq!(index.longest_match(survivor, &chain), chain.len());

        // Replacement pod gets a fresh id and empty affinity.
        let (_, added) = reconcile(
            &registry,
            &index,
            &mut discovered,
            &HashSet::from([url("10.0.0.2"), url("10.0.0.3")]),
        );
        assert_eq!(added, 1);
        let replacement = discovered.by_url[&url("10.0.0.3")];
        assert_eq!(index.longest_match(replacement, &chain), 0);
    }

    #[test]
    fn endpoint_urls_skips_unready_and_uses_first_port() {
        let slice: EndpointSlice = serde_json::from_value(serde_json::json!({
            "apiVersion": "discovery.k8s.io/v1",
            "kind": "EndpointSlice",
            "metadata": {"name": "workers-1"},
            "ports": [{"name": "http", "port": 8001}],
            "endpoints": [
                {"addresses": ["10.0.0.1"], "conditions": {"ready": true}},
                {"addresses": ["10.0.0.2"], "conditions": {"ready": false}},
                {"addresses": ["10.0.0.3"]}
            ]
        }))
        .unwrap();
        let urls = endpoint_urls(&slice);
        assert!(urls.contains(&url("10.0.0.1")));
        assert!(!urls.contains(&url("10.0.0.2")));
        // Kubernetes semantics: `ready` defaults to true when unset.
        assert!(urls.contains(&url("10.0.0.3")));
    }

    #[test]
    fn static_workers_still_parse_for_fallback() {
        let config = router_core::RouterConfig::from_yaml(
            "routing_policy: round_robin\nworkers:\n  - url: http://x:8001\n    pool: both\n",
        )
        .unwrap();
        assert_eq!(config.workers.len(), 1);
        assert!(!config.discovery.enabled);
    }
}
