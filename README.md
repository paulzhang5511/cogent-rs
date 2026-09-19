# Cogent (`cogent-rs`)

[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

Rust 编写的 Agent 开发框架与 CLI 工具。核心是一个 **ReAct 引擎**（思考 → 工具调用 → 观察 → 再思考），通过统一的 trait 契约接入 LLM、工具、记忆与中间件，并以事件总线驱动终端渲染与链路追踪。内置 25 个编译期嵌入的工程技能（skills），按需加载以指导当前任务的工程流程。

二进制命令名为 `cog`。

## 特性

- **ReAct 引擎**：强类型状态机驱动，防死循环检测，多工具并行执行（`join_all`）。
- **可插拔架构**：引擎仅依赖 trait（`LLMProvider` / `Tool` / `Memory` / `AgentMiddleware`），不绑定任何具体实现；新增 Provider 无需改动引擎核心。
- **内置工程技能**：25 个 SKILL.md 编译期嵌入，覆盖规格、TDD、调试、代码评审、安全、发布等完整开发生命周期，零运行时 IO。
- **流式输出**：最终答案经 `chat_stream` 逐 chunk 渲染（打字机效果）。
- **可观测性**：`tracing` 结构化日志 + 可选 OpenTelemetry OTLP 导出。
- **强健性**：传输层指数退避重试（429 / 5xx / 超时）；bash 工具带超时、输出上限与命令注入防护。
- **零网络测试**：全部单元测试无需网络、无需 API Token。

## 架构

纯 virtual workspace，crate 按依赖单向分层：

```
cogent-cli          命令行入口（cog）+ REPL + 事件渲染器 + 引擎装配工厂
   ├── cogent-core        引擎核心：ReAct 循环 / 状态机 / 事件总线 / 记忆 / 中间件 / 配置
   ├── cogent-providers   LLM 接入：OpenAIProvider（chat_complete + chat_stream）+ 重试中间件
   ├── cogent-tools       内置工具：bash / file_read / file_write / edit / http_get / skill
   ├── cogent-skills      25 个工程技能：编译期嵌入 + 路由表 + 按需渲染
   └── cogent-macros      过程宏：#[cogent_tool] 工具声明
```

依赖方向单向收敛：`cli → {core, providers, tools, skills, macros}`，`providers/tools/skills → core`，`core` 不依赖任何兄弟 crate（仅定义 trait）。

具体的 LLM Provider 适配器通过 cargo feature 在 CLI 层按需编译；默认构建仅包含开源的 OpenAI Provider，不含任何专有协议代码。

## 前置要求

- Rust（edition 2024，建议 stable 最新）
- 一个 OpenAI API Key（`cog tools` 除外，其余命令需要）

## 构建

```bash
# 全工作区构建（默认 feature，仅 OpenAI Provider）
cargo build --workspace

# 发布构建
cargo build --workspace --release
```

## 配置

通过环境变量加载（`cogent-core::Config::from_env`）：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI API Key（**必填**，`cog tools` 除外） | — |
| `COGENT_PROVIDER` | Provider 名称（默认构建仅支持 `openai`） | `openai` |
| `COGENT_MODEL` | 模型名称 | `gpt-4o-mini` |
| `COGENT_BASE_URL` | 自定义 endpoint（兼容 OpenAI 的服务 / 本地网关） | — |
| `COGENT_MAX_ITERATIONS` | 最大 ReAct 迭代次数（防死循环硬上限） | `20` |
| `COGENT_MAX_CONTEXT_TOKENS` | 上下文窗口 token 预算 | `8000` |
| `COGENT_OTEL_ENDPOINT` | OTel OTLP endpoint（设置后启用 OTLP 导出） | — |
| `RUST_LOG` | 日志级别过滤（`tracing` env-filter） | `info` |

> 安全边界：API Key 仅保存在内存，绝不写入日志或事件。

## 使用

### 交互式会话（REPL）

```bash
cargo run -p cogent-cli -- repl
```

进入交互式循环，支持多轮对话。内置命令：

| 命令 | 说明 |
| --- | --- |
| `/exit` | 退出会话 |
| `/clear` | 清空会话历史（保留系统种子消息） |
| `/tools` | 列出已注册工具 |
| `/help` | 显示帮助 |

### 单次任务

```bash
cargo run -p cogent-cli -- run "你的任务描述"
```

执行一轮 ReAct 循环后退出，最终答案流式渲染。

### 列出工具

```bash
cargo run -p cogent-cli -- tools
```

列出全部已注册工具（name + description + 参数 JSON Schema）。**无需 API Key**，独立于引擎装配。

### 安装为全局命令（可选）

```bash
cargo install --path crates/cogent-cli
cog repl
cog run "..."
cog tools
```

## 内置工具

| 工具 | 说明 |
| --- | --- |
| `bash` | 执行 shell 命令，返回 exit code / stdout / stderr（结构化 JSON，带超时与输出上限） |
| `file_read` | 读取文件内容 |
| `file_write` | 写入文件 |
| `edit` | 精确字符串替换编辑文件 |
| `http_get` | 发起 HTTP GET 请求（带 SSRF 防护与响应上限） |
| `skill` | 按名称加载完整工程技能正文（主 SKILL.md + 辅助文件） |

## 内置工程技能

`cogent-skills` 在编译期通过 `include_str!` 嵌入 25 个工程技能，覆盖完整开发生命周期：规格驱动（`spec-driven-development`）、测试驱动（`test-driven-development`）、调试与错误恢复、代码评审、API 设计、安全加固、性能优化、可观测性、CI/CD、发布上线等。

- system prompt 注入一张精简「技能路由表」（决策树 + 各技能一句话精髓），模型据此选技能；
- 选定后通过 `skill(name)` 工具（或 bash 拦截的 `cog-skill <name>` 命令）按需加载该技能完整正文，避免一次性灌满上下文；
- 技能内容随二进制分发，运行时零网络、零文件 IO。

完整技能清单可用 `cog tools` 查看，或浏览 `crates/cogent-skills/src/skills/`。

## 开发

```bash
# 运行全部单元测试（无网络、无 Token）
cargo test --workspace

# 仅核心引擎测试
cargo test -p cogent-core

# 格式化
cargo fmt --all

# Lint（零警告门禁）
cargo clippy --workspace --all-targets -- -D warnings

# 覆盖率
cargo tarpaulin --packages cogent-core
```

## 可观测性

- **本地日志**：默认 `tracing` fmt 输出到 stderr，级别由 `RUST_LOG` 控制。
- **OTel 导出**：设置 `COGENT_OTEL_ENDPOINT` 后，`tracing` span 经 `tracing-opentelemetry` 桥接为 OTel trace，通过 OTLP 协议导出到任意 OTel 后端（Jaeger / Tempo / 商业平台）。

```bash
RUST_LOG=debug COGENT_OTEL_ENDPOINT=http://localhost:4317 cog run "..."
```

## 文档

- `docs/prd.md` — 产品需求与系统设计（SRS & SDD）
- `docs/SPEC.md` — 实施规格书（source of truth）

## 贡献

欢迎 Issue 与 PR！提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)；所有参与者需遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。安全漏洞请按 [SECURITY.md](SECURITY.md) 私下报告，勿走公开 Issue。

## 许可证

基于 [Apache License, Version 2.0](LICENSE) 发布。
