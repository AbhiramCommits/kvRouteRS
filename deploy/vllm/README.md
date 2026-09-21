# vLLM backend for kvRouteRS

Serve a vLLM OpenAI-compatible endpoint with block-level prefix caching:

```bash
deploy/vllm/serve.sh
# or, equivalently (vLLM >= 0.6):
# vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
```

## Tested version

- **vLLM v0.6.4.post1**, flags `--enable-prefix-caching`, `--max-model-len`,
  `--gpu-memory-utilization`. Verified against upstream documentation for that
  release. **Not executed on this machine (no GPU)** — on a GPU host, run:

  ```bash
  KVROUTERS_E2E=1 cargo test -p router-server --test vllm_e2e -- --ignored
  ```

## Cache hit metric

`/metrics` (same port as the API server) exposes:

```
vllm:gpu_prefix_cache_hit_rate{model_name="..."} 0.62
```

Enable the router's scraper to surface the predicted-vs-actual gap:

```yaml
metrics_scrape: true
engine_cache_hit_metric: "vllm:gpu_prefix_cache_hit_rate"
```

which produces `kvrouter_engine_cache_hit_rate`, `kvrouter_predicted_cache_hit_rate`,
and `kvrouter_hit_rate_prediction_error` per worker.
