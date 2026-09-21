//! Minimal Prometheus text-exposition parsing for engine metrics scraping.
//!
//! Deliberately not a full parser: we need exactly one gauge per worker (the
//! engine's reported prefix cache hit rate), and hand-rolled line matching
//! keeps the router free of another parsing dependency.

/// Look up a metric by name in Prometheus text exposition and return the mean
/// value across all of its series. Returns `None` when the metric is absent.
pub fn find_metric(text: &str, name: &str) -> Option<f64> {
    let mut values = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let name_end = line.find(['{', ' ', '\t']).unwrap_or(line.len());
        if &line[..name_end] != name {
            continue;
        }
        let rest = &line[name_end..];
        let value_start = rest.rfind([' ', '\t'])? + 1;
        if let Ok(value) = rest[value_start..].parse::<f64>() {
            values.push(value);
        }
    }
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"# HELP vllm:gpu_prefix_cache_hit_rate GPU prefix cache hit rate.
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.625
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 2
"#;

    #[test]
    fn finds_labeled_metric() {
        assert_eq!(
            find_metric(SAMPLE, "vllm:gpu_prefix_cache_hit_rate"),
            Some(0.625)
        );
    }

    #[test]
    fn averages_multiple_series() {
        let text = "m{a=\"1\"} 0.2\nm{a=\"2\"} 0.4\n";
        let value = find_metric(text, "m").unwrap();
        assert!((value - 0.3).abs() < f64::EPSILON, "got {value}");
    }

    #[test]
    fn returns_none_for_absent_metric() {
        assert_eq!(find_metric(SAMPLE, "nope"), None);
    }

    #[test]
    fn skips_comments_and_blank_lines() {
        assert_eq!(find_metric("# m 1\n\n", "m"), None);
    }
}
