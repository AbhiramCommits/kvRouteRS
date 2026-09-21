use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One chat message in the OpenAI request format.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChatMessage {
    /// `system`, `user`, or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
}

/// OpenAI-compatible chat completion request body.
///
/// Lives in the core (rather than the HTTP layer) because the routing decision
/// is computed from this data: the canonical prompt feeds the prefix-hash chain,
/// which drives cache-aware scoring. The struct itself is transport-agnostic —
/// it only depends on serde.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    /// Model name, forwarded verbatim to the worker.
    pub model: String,
    /// Conversation so far; feeds the canonical prompt.
    pub messages: Vec<ChatMessage>,
    /// Generation budget, if the client set one.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature, if the client set one.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Whether the client wants an SSE stream.
    #[serde(default)]
    pub stream: bool,
    /// Preserved and re-serialized verbatim so unknown OpenAI fields survive
    /// proxying (e.g. `stop`, `top_p`, `user`).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl ChatCompletionRequest {
    /// Canonical prompt serialization.
    ///
    /// Every request with the same `messages` must produce byte-for-byte the
    /// same string, or prefix-hash chains — and therefore cache-affinity
    /// scoring — will disagree between requests. The format is deliberately
    /// simple and stable: one `"role: content"` line per message, joined by
    /// newlines. Roles are included so that "user: X" and "assistant: X"
    /// exchanges are not mistaken for the same prefix.
    pub fn canonical_prompt(&self) -> String {
        let mut prompt = String::new();
        for (index, message) in self.messages.iter().enumerate() {
            if index > 0 {
                prompt.push('\n');
            }
            prompt.push_str(&message.role);
            prompt.push_str(": ");
            prompt.push_str(&message.content);
        }
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(messages: Vec<(&str, &str)>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "mock".to_string(),
            messages: messages
                .into_iter()
                .map(|(role, content)| ChatMessage {
                    role: role.to_string(),
                    content: content.to_string(),
                })
                .collect(),
            max_tokens: None,
            temperature: None,
            stream: false,
            extra: Default::default(),
        }
    }

    #[test]
    fn canonical_prompt_is_deterministic() {
        let a = request(vec![("user", "hello")]);
        let b = request(vec![("user", "hello")]);
        assert_eq!(a.canonical_prompt(), b.canonical_prompt());
        assert_eq!(a.canonical_prompt(), "user: hello");
    }

    #[test]
    fn canonical_prompt_joins_messages_in_order_with_roles() {
        let request = request(vec![("user", "first"), ("assistant", "second")]);
        assert_eq!(request.canonical_prompt(), "user: first\nassistant: second");
    }

    #[test]
    fn role_changes_are_visible_in_the_prompt() {
        let as_user = request(vec![("user", "same words")]);
        let as_assistant = request(vec![("assistant", "same words")]);
        assert_ne!(as_user.canonical_prompt(), as_assistant.canonical_prompt());
    }

    #[test]
    fn unknown_fields_survive_the_round_trip() {
        let json = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"top_p":0.9}"#;
        let parsed: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.extra.get("top_p").and_then(|v| v.as_f64()),
            Some(0.9)
        );
        let reserialized = serde_json::to_value(&parsed).unwrap();
        assert_eq!(reserialized["top_p"], serde_json::json!(0.9));
    }
}
