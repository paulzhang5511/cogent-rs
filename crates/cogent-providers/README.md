# cogent-providers

LLM provider implementations for [Cogent](https://github.com/paulzhang5511/cogent-rs), implementing [`cogent_core::provider::LLMProvider`].

- [`openai`]: OpenAI (and OpenAI-compatible endpoints) — both `chat_complete` and streaming `chat_stream`.
- [`retry`]: a transport-layer retry decorator built on `backoff` (429 / 5xx / timeouts with exponential backoff).

## Example

```rust,ignore
use cogent_providers::openai::OpenAIProvider;

let provider = OpenAIProvider::new(api_key, model, base_url);
// Wrap with retry for transient-failure resilience.
let provider = cogent_providers::retry::RetryMiddleware::new(provider);
```

Adding another backend (Claude, Ollama, …) means implementing the same `LLMProvider` trait and registering it in the CLI factory — the engine is untouched.

## License

Apache-2.0. See [the repository](https://github.com/paulzhang5511/cogent-rs).
