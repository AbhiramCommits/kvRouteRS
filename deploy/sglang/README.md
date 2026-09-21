# SGLang backend for kvRouteRS

Serve an SGLang OpenAI-compatible endpoint with RadixAttention (the radix-tree
prefix cache, SGLang's default scheduler):

```bash
deploy/sglang/serve.sh
```

## Tested version

- **SGLang v0.4.4.post2**, flags `--enable-metrics`, `--mem-fraction-static`.
  RadixAttention needs no flag (it is the default; `--disable-radix-cache`
  turns it off). Verified against upstream documentation for that release.
  **Not executed on this machine (no GPU)**.

## Cache hit metric

SGLang's `/metrics` (with `--enable-metrics`) exposes radix cache counters;
the exact series names vary by build. Check `/metrics` on your build and point
the router at the hit-rate metric:

```yaml
metrics_scrape: true
engine_cache_hit_metric: "<your build's radix cache hit rate metric>"
```

which produces `kvrouter_engine_cache_hit_rate`, `kvrouter_predicted_cache_hit_rate`,
and `kvrouter_hit_rate_prediction_error` per worker.
