# cogent-tools

System tools for [Cogent](https://github.com/paulzhang5511/cogent-rs), declared with `#[cogent_tool]`.

## Tools

- `bash`: run a shell command (timeout, output cap, structured JSON result).
- `file_read` / `file_write` / `edit`: file IO with path-traversal protection.
- `http_get`: HTTP GET with SSRF protection and response size cap.
- `skill`: load an embedded engineering skill by name.

## Reliability layer

Because model output is untrusted, this crate also ships a defensive layer:

- `sanitize` — clean model output (leading blank lines, U+FFFD, glued comments).
- `encoding` — detect UTF-8 / UTF-8-BOM / GBK.
- `error_recovery` — closest-match suggestions and structured success/failure lists.
- `diff_preview` — show a diff and require confirmation before edits.
- `validate` — post-write validation (JSON / `javac` / encoding consistency) with rollback.

## License

Apache-2.0. See [the repository](https://github.com/paulzhang5511/cogent-rs).
