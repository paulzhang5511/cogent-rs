//! 引擎装配工厂。
//!
//! 本模块实现 [`build_engine`]——按 [`Config`] 装配完整的 [`AgentEngine`]：
//! - 初始化 `tracing_subscriber`（可选 OTel OTLP 导出，PRD §1.2 落点）。
//! - 按 `config.provider` 构造 Provider（当前仅 `openai`）。
//! - 注册 `cogent-tools` 全部工具。
//! - 构造 `WindowMemory`（注入 system 种子消息）。
//! - 装配 `EventBus` 与中间件链（Tracing / SafetyGuard / TokenLimiter）。
//!
//! # 设计原则
//! - factory 是 CLI 层唯一的"具体实现注入点"：引擎只依赖 trait，
//!   所有具体 provider/工具/中间件在此装配后以 `Arc<dyn Trait>` 注入。
//! - OTel exporter 生命周期由 factory 返回的 [`OtelGuard`] 管理，
//!   程序退出前 `shut_down`（RAII）。
//! - API Key 绝不写入日志。

use std::sync::Arc;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::runtime::Tokio;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use cogent_core::config::Config;
use cogent_core::engine::AgentEngine;
use cogent_core::event::EventBus;
use cogent_core::memory::WindowMemory;
use cogent_core::middleware::AgentMiddleware;
use cogent_core::provider::LLMProvider;
use cogent_core::tool::Tool;
use cogent_core::types::{Message, Role};
use cogent_providers::openai::OpenAIProvider;
use cogent_providers::retry::RetryMiddleware;
use cogent_tools::bash::BashTool;
use cogent_tools::file::{EditTool, FileReadTool, FileWriteTool};
use cogent_tools::http::HttpGetTool;
use cogent_tools::skill::SkillTool;

use crate::middleware::{SafetyGuardMiddleware, TokenLimiterMiddleware, TracingMiddleware};

/// 默认系统种子消息：定义 Agent 的人设与行为约束。
const DEFAULT_SYSTEM_PROMPT: &str = "You are Cogent, a helpful AI agent. \
You can use tools to accomplish tasks. Think step by step, \
call tools when needed, and provide a clear final answer.";

/// `cog-skill` 命令使用提示（所有 provider 注入，强制执行）。
///
/// 部分内置工具集（只读命令类）无法直接调用 `skill` 工具，但能通过
/// shell 命令执行。`cogent-tools` 的 bash 工具拦截
/// `cog-skill <name>` 命令并返回该技能的完整 SKILL.md 正文（含辅助文件），
/// 使各 provider 都能按需加载完整技能内容。
///
/// 此提示是**强制**的：模型必须先按 `using-agent-skills` 决策树选定技能，
/// 再用 `cog-skill <name>` 加载完整正文，然后严格按其步骤执行（不跳验证）。
const COG_SKILL_HINT: &str = "\
MANDATORY SKILL WORKFLOW — you MUST follow this for any non-trivial task:
1. From the 'Using Agent Skills' decision tree above, identify which skill(s) \
apply to the current task.
2. Load the FULL skill content by running the command: cog-skill <name> \
(e.g. `cog-skill test-driven-development`). This returns the complete \
step-by-step workflow. Do NOT guess the steps — load the skill first.
3. Follow the loaded skill's steps IN ORDER. Do not skip verification steps. \
A task is not complete until its verification passes with evidence.
Multiple skills may apply in sequence (e.g. spec-driven-development → \
planning-and-task-breakdown → incremental-implementation → \
test-driven-development → code-review-and-quality). Load each as you reach it. \
For a trivial one-line fix you may skip the skill, but state why.";

/// OTel exporter 生命周期守卫（RAII）。
///
/// 持有 `opentelemetry_otlp` 的 `SdkTracerProvider`，
/// 在 `Drop` 时调用 `shut_down` 确保所有 span 在程序退出前导出。
/// 未启用 OTel 时为 `None`（无操作）。
#[allow(dead_code)]
pub struct OtelGuard {
    /// OTel tracer provider（启用 OTel 时持有）。
    provider: Option<opentelemetry_sdk::trace::TracerProvider>,
}

impl OtelGuard {
    /// 创建空的守卫（未启用 OTel）。
    fn disabled() -> Self {
        Self { provider: None }
    }
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Some(p) = self.provider.take() {
            // 程序退出前 flush 所有未导出的 span
            let _ = p.force_flush();
            tracing::info!("OTel tracer provider shut down");
        }
    }
}

/// 初始化 `tracing_subscriber`，返回 OTel 守卫（若启用 OTel）。
///
/// # 参数
/// - `otel_endpoint`：OTel OTLP endpoint（`None` 则仅本地 fmt 输出）。
///
/// # 返回
/// - 成功：[`OtelGuard`]（持有 OTel provider 生命周期）。
/// - 失败：OTel exporter 构造错误。
///
/// # 说明
/// - 未设置 `otel_endpoint`：仅装 `fmt` layer（本地日志）。
/// - 设置了 `otel_endpoint`：装 `fmt` + `tracing_opentelemetry` layer，
///   将 span 桥接为 OTel trace 并通过 OTLP 导出。
fn init_tracing(otel_endpoint: Option<&str>) -> anyhow::Result<OtelGuard> {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    match otel_endpoint {
        None => {
            // 仅本地 fmt 输出。
            // 用 try_init：全局 subscriber 已存在时（如并行测试）静默跳过，
            // 避免 SetGlobalDefaultError panic。
            let _ = tracing_subscriber::fmt()
                .with_env_filter(env_filter)
                .with_target(false)
                .with_writer(std::io::stderr)
                .try_init();
            Ok(OtelGuard::disabled())
        }
        Some(endpoint) => {
            // 构造 OTLP exporter
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint)
                .build()
                .map_err(|e| anyhow::anyhow!("failed to build OTLP exporter: {e}"))?;

            // 构造 tracer provider（用 Tokio runtime 的 batch exporter）
            let provider = opentelemetry_sdk::trace::TracerProvider::builder()
                .with_batch_exporter(exporter, Tokio)
                .build();

            let tracer = provider.tracer("cogent");

            // 装 fmt + OTel layer（try_init：全局 subscriber 已存在时静默跳过）
            let _ = tracing_subscriber::registry()
                .with(env_filter)
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_target(false)
                        .with_writer(std::io::stderr),
                )
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .try_init();

            tracing::info!(endpoint = %endpoint, "OTel tracing enabled");

            Ok(OtelGuard {
                provider: Some(provider),
            })
        }
    }
}

/// 注册全部系统工具（bash / file_read / file_write / edit / http_get / skill）。
///
/// 独立于引擎装配，供 `cog tools` 子命令在不构造 Provider（无需 API key）
/// 的情况下列出已注册工具。引擎装配（[`build_engine`]）内部也复用本函数。
///
/// # 返回
/// - 工具列表（`Vec<Arc<dyn Tool>>`），顺序固定：bash、file_read、file_write、edit、http_get、skill。
pub fn build_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(BashTool),
        Arc::new(FileReadTool),
        Arc::new(FileWriteTool),
        Arc::new(EditTool),
        Arc::new(HttpGetTool),
        Arc::new(SkillTool),
    ]
}

/// 解析最终 system prompt：追加技能路由表 + 强制技能工作流。
///
/// 所有 provider 均追加：
/// 1. `using-agent-skills` 路由表（真实 meta-skill，~2.6K tokens）：决策树 +
///    强制操作行为 + 生命周期序列 + 25 技能一句话精髓。
/// 2. [`COG_SKILL_HINT`]（强制）：要求模型先按决策树选定技能，再用
///    `cog-skill <name>` 命令加载完整技能正文，然后严格按步骤执行。
///    `cog-skill` 由 bash 工具拦截。
///
/// # 参数
/// - `system_prompt`：调用方自定义 prompt（`None` 用默认人设）。
///
/// # 返回
/// 最终 system prompt 字符串。
fn resolve_system_prompt(system_prompt: Option<String>) -> String {
    let base = system_prompt.unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());
    let mut prompt = base;
    // 真实 using-agent-skills 路由表（所有 provider）
    prompt.push_str("\n\n");
    prompt.push_str(cogent_skills::get_routing_table());
    // 强制 cog-skill 工作流（所有 provider，经 bash 拦截加载完整技能）
    prompt.push_str("\n\n");
    prompt.push_str(COG_SKILL_HINT);
    prompt
}

/// 按配置装配完整的 [`AgentEngine`]。
///
/// # 参数
/// - `config`：框架全局配置。
/// - `system_prompt`：系统种子消息（`None` 则用默认人设）。
///
/// # 返回
/// - 成功：装配好的 [`AgentEngine`]、共享的 [`EventBus`]（供渲染器订阅）
///   与 [`OtelGuard`]（须持有至程序退出）。
/// - 失败：配置错误、Provider 构造错误等。
///
/// # 装配顺序
/// 1. 初始化 tracing（含可选 OTel）。
/// 2. 按 `config.provider` 构造 Provider（当前仅 OpenAI），套 `RetryMiddleware`。
/// 3. 注册全部工具（bash / file_read / file_write / edit / http_get / skill）。
/// 4. 构造 `WindowMemory`，注入 system 种子。
/// 5. 装配 `EventBus`。
/// 6. 装配中间件链（Tracing → SafetyGuard → TokenLimiter）。
/// 7. 构造 `AgentEngine`。
pub async fn build_engine(
    config: &Config,
    system_prompt: Option<String>,
) -> anyhow::Result<(AgentEngine, Arc<EventBus>, OtelGuard)> {
    // 1. 初始化 tracing（含可选 OTel）
    let otel_guard = init_tracing(config.otel_endpoint.as_deref())?;

    // 2. 构造 Provider（按 config.provider 分支）+ Retry 装饰器
    let provider: Arc<dyn LLMProvider> = match config.provider.as_str() {
        "openai" => {
            let api_key = config
                .api_key
                .clone()
                .ok_or_else(|| anyhow::anyhow!("API key is required for OpenAI provider"))?;

            let openai = Arc::new(
                OpenAIProvider::new(api_key, config.model.clone(), config.base_url.clone())
                    .map_err(|e| anyhow::anyhow!("failed to create OpenAI provider: {e}"))?,
            );

            // 套 RetryMiddleware（传输层重试，默认 3 次）
            Arc::new(RetryMiddleware::with_default_retries(openai))
        }
        other => anyhow::bail!("unknown provider: {other}"),
    };

    // 3. 注册全部工具（复用 build_tools，与 `cog tools` 子命令一致）
    let tools: Vec<Arc<dyn Tool>> = build_tools();

    // 4. 构造 WindowMemory，注入 system 种子
    let memory: Arc<dyn cogent_core::memory::Memory> = Arc::new(
        WindowMemory::with_default_estimator(config.max_context_tokens),
    );

    let prompt = resolve_system_prompt(system_prompt);
    memory.set_system(Message::new(Role::System, prompt)).await;

    // 5. 装配 EventBus
    let event_bus = Arc::new(EventBus::new(config.event_bus_capacity));

    // 6. 装配中间件链（按执行顺序：Tracing → SafetyGuard → TokenLimiter）
    let middlewares: Vec<Arc<dyn AgentMiddleware>> = vec![
        Arc::new(TracingMiddleware),
        // 默认安全策略：不禁用整工具，但拦截 bash 破坏性命令模式
        // （rm -rf /、mkfs、dd、fork 炸弹等，见 SafetyGuardMiddleware::DEFAULT_DESTRUCTIVE_PATTERNS）
        Arc::new(SafetyGuardMiddleware::with_default_policy()),
        Arc::new(TokenLimiterMiddleware::new(
            config.max_context_tokens,
            Arc::new(cogent_core::memory::CharBasedEstimator),
        )),
    ];

    // 7. 构造 AgentEngine（传入 event_bus 的克隆，原 Arc 返回给调用方供渲染器订阅）
    let engine = AgentEngine::new(
        provider,
        tools,
        memory,
        middlewares,
        event_bus.clone(),
        config.max_iterations,
    );

    tracing::info!(
        model = %config.model,
        max_iterations = config.max_iterations,
        max_context_tokens = config.max_context_tokens,
        "agent engine built"
    );

    Ok((engine, event_bus, otel_guard))
}

#[cfg(test)]
mod tests {
    //! factory 单元测试。
    //!
    //! 验证工具注册、system 种子注入、中间件链装配。
    //! 不发起真实网络请求（Provider 构造仅验证不 panic）。

    use super::*;

    /// 构造测试用 Config（不读真实环境）。
    fn test_config() -> Config {
        Config {
            provider: "openai".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            base_url: None,
            max_iterations: 5,
            max_context_tokens: 1000,
            event_bus_capacity: 64,
            otel_endpoint: None,
            auto_approve: true,
            java_check: false,
        }
    }

    /// 验证 build_engine 成功装配引擎。
    #[tokio::test]
    async fn test_build_engine() {
        let config = test_config();
        let (engine, _bus, _guard) = build_engine(&config, None).await.unwrap();
        // 引擎初始状态为 Idle
        assert_eq!(engine.state(), cogent_core::state::AgentState::Idle);
    }

    /// 验证 build_engine 注入自定义 system prompt。
    #[tokio::test]
    async fn test_build_engine_custom_prompt() {
        let config = test_config();
        let (engine, _bus, _guard) = build_engine(&config, Some("custom prompt".into()))
            .await
            .unwrap();
        assert_eq!(engine.state(), cogent_core::state::AgentState::Idle);
    }

    /// 验证 build_engine 在缺少 API key 时返回错误。
    #[tokio::test]
    async fn test_build_engine_missing_api_key() {
        let mut config = test_config();
        config.api_key = None;
        assert!(build_engine(&config, None).await.is_err());
    }

    /// 验证 system prompt 含路由表 + cog-skill 强制工作流。
    #[test]
    fn test_resolve_system_prompt() {
        let prompt = resolve_system_prompt(None);
        assert!(prompt.starts_with(DEFAULT_SYSTEM_PROMPT));
        // 含真实 using-agent-skills 路由表
        assert!(prompt.contains("Using Agent Skills"));
        assert!(prompt.contains("Quick Reference"));
        assert!(prompt.contains("test-driven-development"));
        // 含强制 cog-skill 工作流（经 bash 拦截同样可用）
        assert!(prompt.contains("MANDATORY SKILL WORKFLOW"));
        assert!(prompt.contains("cog-skill"));
    }

    /// 验证自定义 prompt 被保留并追加路由表 + cog-skill 工作流。
    #[test]
    fn test_resolve_system_prompt_custom_kept() {
        let prompt = resolve_system_prompt(Some("custom base".into()));
        assert!(prompt.starts_with("custom base"));
        assert!(prompt.contains("Using Agent Skills"));
        assert!(prompt.contains("MANDATORY SKILL WORKFLOW"));
    }
}
