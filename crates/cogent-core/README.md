# cogent-core

Core of the [Cogent](https://github.com/paulzhang5511/cogent-rs) Rust Agent framework: trait contracts and the ReAct engine.

This crate defines the shared contracts and runtime and depends on **no concrete implementation** — providers, tools, middleware, and memory live downstream and are injected via traits.

## What's inside

- [`types`]: messages and tool-call data types.
- [`state`]: strongly-typed agent state machine.
- [`event`]: multi-subscriber event bus (`broadcast`).
- [`tool`], [`provider`], [`memory`], [`middleware`]: pluggable traits.
- [`engine`]: the ReAct loop (think → tool call → observe → think), with loop detection and parallel tool execution.
- [`config`], [`error`], [`summary`].

## Example

```rust,ignore
use cogent_core::{engine::AgentEngine, memory::WindowMemory, provider::LLMProvider, tool::Tool};

// Build an engine from your provider + tools + memory via a factory;
// the engine only knows the traits, never a concrete backend.
let engine = AgentEngine::new(provider, tools, memory, middlewares, config);
let answer = engine.run(max_context_tokens).await?;
```

## License

Apache-2.0. See [the repository](https://github.com/paulzhang5511/cogent-rs).
