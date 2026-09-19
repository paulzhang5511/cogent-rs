# cogent-macros

Procedural macros for [Cogent](https://github.com/paulzhang5511/cogent-rs).

The `#[cogent_tool]` attribute turns an `async fn` into a type that implements [`cogent_core::tool::Tool`], deriving the parameter JSON Schema from the function signature and doc comments (via `schemars`).

## Example

```rust,ignore
use cogent_macros::cogent_tool;

/// Run a shell command and return its output.
#[cogent_tool]
async fn bash(command: String) -> anyhow::Result<String> {
    // `command` is deserialized from `args["command"]` by the generated impl.
    todo!()
}
```

This generates a `BashTool` struct whose `name()`, `description()`, `parameters_schema()`, and `execute()` are filled in for you.

## License

Apache-2.0. See [the repository](https://github.com/paulzhang5511/cogent-rs).
