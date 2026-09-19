# cogent-skills

25 engineering skills embedded at compile time for [Cogent](https://github.com/paulzhang5511/cogent-rs).

Each skill is a `SKILL.md` (plus optional auxiliary files) baked into the binary's `.rodata` via `include_str!` — no runtime file IO, no network. The CLI binary carries the whole skill set in one file.

## Public API

- [`get_skill`]: fetch a skill by name (full content + auxiliary files).
- [`list_skills`]: list all skill metadata.
- [`get_routing_table`]: the `using-agent-skills` meta-skill routing table injected into the system prompt.

## Origin

These skills are adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills) (MIT License, Copyright (c) Addy Osmani and contributors). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. Bundled third-party skill text remains under MIT — see `THIRD_PARTY_NOTICES.md`.
