//! End-to-end check against a real vLLM server with prefix caching enabled.
//!
//! Skipped unless `KVROUTERS_E2E=1` is set AND the test is selected (it is
//! `#[ignore]`d so CI never runs it by accident):
//!
//! ```bash
//! KVROUTERS_E2E=1 VLLM_URL=http://127.0.0.1:8000 \
//!   cargo test -p router-server --test vllm_e2e -- --ignored
//! ```
//!
//! Assumes the vLLM server from `deploy/vllm/serve.sh` is running, i.e.:
//!
//! ```bash
//! python -m vllm.entrypoints.openai.api_server \
//!   --model Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
//! ```

use std::sync::Arc;
use std::time::Duration;

use router_core::{
    ChatCompletionRequest, ChatMessage, Pool, Router, RouterConfig, RoutingPolicy, WorkerConfig,
    WorkerRegistry,
};
use router_server::prometheus_text::find_metric;

/// Tolerance between the router's predicted (512-character-block) hit rate and
/// vLLM's reported (16-token-block) hit rate. The two granularities never
/// agree exactly: a character block boundary rarely coincides with a token
/// block boundary. With a long shared system prompt both should be *high*
/// after warmup, and this bound documents how close the router's belief model
/// gets to engine truth. If a GPU build ever exceeds it, that is the signal
/// that the belief index (TTL, eviction, block size) needs tuning.
const HIT_RATE_TOLERANCE: f64 = 0.30;

const REQUESTS: usize = 40;

fn request(system: &str, suffix: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: suffix.to_string(),
            },
        ],
        max_tokens: Some(16),
        temperature: None,
        stream: false,
        extra: Default::default(),
    }
}

async fn wait_until_ready(client: &reqwest::Client, base: &str) {
    for _ in 0..120 {
        match client.get(format!("{base}/health")).send().await {
            Ok(response) if response.status().is_success() => return,
            Ok(_) | Err(_) => tokio::time::sleep(Duration::from_secs(1)).await,
        }
    }
    panic!("vLLM at {base} did not become ready within 120s");
}

#[tokio::test]
#[ignore]
async fn predicted_hit_rate_tracks_vllm() {
    if std::env::var("KVROUTERS_E2E").as_deref() != Ok("1") {
        eprintln!("skipping vLLM e2e test (set KVROUTERS_E2E=1 to run it)");
        return;
    }
    let vllm_url =
        std::env::var("VLLM_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());

    let config = RouterConfig {
        workers: vec![WorkerConfig {
            url: vllm_url.clone(),
            pool: Pool::Both,
        }],
        routing_policy: RoutingPolicy::CacheAware,
        health_check_interval_secs: 5,
        prompt_block_chars: 512,
        cache_weight: 1.0,
        load_weight: 0.5,
        prefix_index_ttl_secs: 300,
        prefix_index_max_entries: 100_000,
        kv_transfer_cost_ms: 20,
        metrics_scrape: false,
        engine_metrics_interval_secs: 15,
        engine_cache_hit_metric: "vllm:gpu_prefix_cache_hit_rate".to_string(),
        discovery: router_core::DiscoveryConfig::default(),
    };
    let registry = Arc::new(WorkerRegistry::from_config(&config.workers));
    registry.set_healthy(0, true);
    let router = Router::new(registry, &config);
    let client = reqwest::Client::new();

    wait_until_ready(&client, &vllm_url).await;

    // ~6KB system prompt: 12 router blocks, well over a thousand tokens, so
    // both the router's char-block index and vLLM's token-block cache have a
    // large shared prefix to hit on every request after the first.
    let system = "You are a helpful assistant. Answer questions about the fictional city of \
                  Kvropolis. Kvropolis was founded in 1987 as a routing research hub and its \
                  economy depends on prefix caches. "
        .repeat(30);

    let mut predicted_hits = 0u64;
    let mut predicted_blocks = 0u64;
    for i in 0..REQUESTS {
        let req = request(
            &system,
            &format!("Question {i}: what is the weather in Kvropolis?"),
        );
        let decision = router
            .select_for_chat(&req)
            .expect("routing decision should succeed");
        predicted_hits += decision.matched_prefix_len as u64;
        predicted_blocks += decision.prompt_blocks as u64;

        let response = client
            .post(format!("{vllm_url}/v1/chat/completions"))
            .json(&req)
            .send()
            .await
            .expect("vllm request")
            .error_for_status()
            .expect("vllm should return 200");
        let _ = response.bytes().await.expect("drain response body");
    }

    let predicted = predicted_hits as f64 / predicted_blocks as f64;

    let metrics_text = client
        .get(format!("{vllm_url}/metrics"))
        .send()
        .await
        .expect("metrics fetch")
        .error_for_status()
        .expect("metrics endpoint should return 200")
        .text()
        .await
        .expect("metrics body");
    let reported =
        find_metric(&metrics_text, "vllm:gpu_prefix_cache_hit_rate").unwrap_or_else(|| {
            panic!(
                "vllm:gpu_prefix_cache_hit_rate not found on {vllm_url}/metrics; \
             is vLLM started with --enable-prefix-caching?"
            )
        });

    assert!(
        predicted > 0.6,
        "router should predict hits on a shared 6KB system prompt; got {predicted:.3}"
    );
    assert!(
        reported > 0.4,
        "vLLM should report prefix cache hits; got {reported:.3}"
    );
    assert!(
        (predicted - reported).abs() < HIT_RATE_TOLERANCE,
        "predicted {predicted:.3} vs vLLM-reported {reported:.3} exceeds the documented \
         tolerance {HIT_RATE_TOLERANCE}"
    );
}
