# cogent-cli

Command-line interface and REPL for [Cogent](https://github.com/paulzhang5511/cogent-rs), a Rust Agent framework.

Binary name: `cog`.

## Usage

```bash
# interactive REPL
cog repl

# single-shot ReAct run
cog run "your task description"

# list registered tools (no API key needed)
cog tools
```

Configure via environment variables (`OPENAI_API_KEY` required, plus `COGENT_MODEL`, `COGENT_BASE_URL`, `COGENT_MAX_ITERATIONS`, …). See the [repository README](https://github.com/paulzhang5511/cogent-rs) for the full table.

## Install

```bash
cargo install --path crates/cogent-cli
```

## License

Apache-2.0. See [the repository](https://github.com/paulzhang5511/cogent-rs).
