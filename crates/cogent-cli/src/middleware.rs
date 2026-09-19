//! CLI 层中间件实现。
//!
//! 本模块实现 SPEC §7.5 要求的三个引擎中间件：
//! - [`TracingMiddleware`]：观测层，在 LLM/工具调用前后打 `tracing` span（英文日志）。
//! - [`SafetyGuardMiddleware`]：安全层，`before_tool` 对高危工具/参数做策略拦截。
//! - [`TokenLimiterMiddleware`]：资源层，`before_llm` 校验上下文 token 不超上限。
//!
//! 三者均实现 [`cogent_core::middleware::AgentMiddleware`]，由 factory 装配进引擎。
//! 与 `cogent-providers` 的 `RetryMiddleware`（传输层，实现 `LLMProvider`）互补。

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use cogent_core::error::{CogentError, CogentResult};
use cogent_core::middleware::AgentMiddleware;
use cogent_core::provider::LLMResponse;
use cogent_core::types::Message;

/// 观测层中间件：在 LLM 调用与工具执行前后打 `tracing` span。
///
/// 归属观测层，不承载业务/安全逻辑；与 OTel exporter 解耦
/// （中间件只发 span，exporter 由 factory 装配）。
#[derive(Debug, Default)]
pub struct TracingMiddleware;

#[async_trait]
impl AgentMiddleware for TracingMiddleware {
    async fn before_llm(&self, messages: &[Message]) -> CogentResult<()> {
        tracing::info!(message_count = messages.len(), "middleware: before_llm");
        Ok(())
    }

    async fn after_llm(&self, response: &LLMResponse) -> CogentResult<()> {
        tracing::info!(
            prompt_tokens = response.prompt_tokens,
            completion_tokens = response.completion_tokens,
            has_tool_calls = response.has_tool_calls(),
            "middleware: after_llm"
        );
        Ok(())
    }

    async fn before_tool(&self, tool_name: &str, args: &Value) -> CogentResult<()> {
        tracing::info!(
            tool = %tool_name,
            args = %args,
            "middleware: before_tool"
        );
        Ok(())
    }
}

/// 安全层中间件：对高危工具/参数做策略拦截。
///
/// 归属安全层（Q5 决策落点）。维护一个可配置的工具黑名单，
/// `before_tool` 时若工具名在黑名单中则返回 [`CogentError::MiddlewareBlocked`]，
/// 引擎据此中止本轮工具执行。
///
/// # 设计
/// - 黑名单在构造时注入（`Arc<Vec<String>>` 共享，避免每次调用克隆）。
/// - 除工具名黑名单外，对 `bash` 工具的 `command` 参数做破坏性模式
///   子串匹配（[`DEFAULT_DESTRUCTIVE_PATTERNS`]），拦截 `rm -rf /`、
///   `mkfs`、`dd`、fork 炸弹等高危命令（R1 安全加固）。
/// - 默认非空策略由 factory 注入（见 [`default_blocked_tools`]）。
///
/// # 重要边界：这是"提示层"，不是沙箱
/// 破坏性命令拦截是**大小写不敏感的子串匹配**，并非 shell 词法/语法解析，
/// 因此可被轻易绕过（如 `rm -r -f /`、`rm -rf  /` 双空格、变量/命令展开、
/// `bash -c '...'` 二次包裹等）。它的目的是**拦截模型"不小心"发出的高危命令**，
/// 降低误触风险；**不构成安全边界**。需要真正的执行隔离时，应依靠 OS 层手段
/// （容器、无网络/只读用户、seccomp 等），而不是依赖本中间件。
pub struct SafetyGuardMiddleware {
    /// 被禁止调用的工具名列表。
    blocked_tools: Arc<Vec<String>>,
    /// `bash` 工具参数中禁止出现的破坏性命令模式（子串匹配，大小写不敏感）。
    destructive_patterns: Arc<Vec<String>>,
}

/// `bash` 工具默认拦截的破坏性命令模式（子串匹配，小写比较）。
///
/// 覆盖常见高危命令：递归强制删除根、文件系统格式化、磁盘裸写、
/// fork 炸弹、权限篡改、关机重启等。匹配为子串包含（非正则），
/// 误报倾向保守（宁可拦截也不放过）。
pub const DEFAULT_DESTRUCTIVE_PATTERNS: &[&str] = &[
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    "dd of=/dev/",
    ":(){:|:&};:",
    "chmod 777 /",
    "chmod -r 777 /",
    "shutdown",
    "reboot",
    "halt",
    "init 0",
    "init 6",
    "> /dev/sda",
    "> /dev/nvme",
];

impl SafetyGuardMiddleware {
    /// 创建安全守卫中间件（全默认策略）。
    ///
    /// 工具名黑名单用 [`default_blocked_tools`]，破坏性命令模式用
    /// [`DEFAULT_DESTRUCTIVE_PATTERNS`]。
    pub fn with_default_policy() -> Self {
        Self::new(default_blocked_tools())
    }

    /// 创建安全守卫中间件。
    ///
    /// # 参数
    /// - `blocked_tools`：被禁止调用的工具名列表。
    pub fn new(blocked_tools: Vec<String>) -> Self {
        let patterns: Vec<String> = DEFAULT_DESTRUCTIVE_PATTERNS
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        tracing::info!(
            blocked_count = blocked_tools.len(),
            pattern_count = patterns.len(),
            "SafetyGuardMiddleware initialized"
        );
        Self {
            blocked_tools: Arc::new(blocked_tools),
            destructive_patterns: Arc::new(patterns),
        }
    }
}

/// 默认工具名黑名单（空——v1 不禁用任何整工具，破坏性命令由参数模式拦截）。
///
/// 返回 `Vec<String>` 而非 `&'static` 是为方便 factory 按配置追加。
pub fn default_blocked_tools() -> Vec<String> {
    // v1 不禁用任何整工具：bash 是 Agent 核心能力，
    // 破坏性命令在参数层拦截（见 DEFAULT_DESTRUCTIVE_PATTERNS）。
    Vec::new()
}

#[async_trait]
impl AgentMiddleware for SafetyGuardMiddleware {
    async fn before_llm(&self, _messages: &[Message]) -> CogentResult<()> {
        Ok(())
    }

    async fn after_llm(&self, _response: &LLMResponse) -> CogentResult<()> {
        Ok(())
    }

    async fn before_tool(&self, tool_name: &str, args: &Value) -> CogentResult<()> {
        // 1. 检查工具是否在黑名单中
        if self.blocked_tools.iter().any(|t| t == tool_name) {
            tracing::warn!(
                tool = %tool_name,
                "middleware: tool blocked by safety guard"
            );
            return Err(CogentError::MiddlewareBlocked {
                middleware: "SafetyGuard".into(),
                reason: format!("tool '{tool_name}' is blocked by safety policy"),
            });
        }

        // 2. 对 bash 工具检查 command 参数中的破坏性模式（子串匹配，大小写不敏感）
        if tool_name == "bash"
            && let Some(command) = args.get("command").and_then(|v| v.as_str())
        {
            let lower = command.to_lowercase();
            for pat in self.destructive_patterns.iter() {
                if lower.contains(pat.as_str()) {
                    tracing::warn!(
                        tool = %tool_name,
                        pattern = %pat,
                        "middleware: destructive command pattern blocked"
                    );
                    return Err(CogentError::MiddlewareBlocked {
                        middleware: "SafetyGuard".into(),
                        reason: format!(
                            "destructive command pattern '{pat}' is blocked by safety policy"
                        ),
                    });
                }
            }
        }

        Ok(())
    }
}

/// 资源层中间件：校验上下文 token 不超上限。
///
/// 归属资源层。`before_llm` 时用 [`TokenEstimator`] 估算当前上下文 token 数，
/// 若超过上限则返回 [`CogentError::MiddlewareBlocked`]，触发引擎中止
/// （记忆窗口裁剪由 `WindowMemory` 在 `get_context` 时完成，此处为兜底校验）。
pub struct TokenLimiterMiddleware {
    /// 上下文 token 上限。
    max_tokens: usize,
    /// Token 估算器。
    estimator: Arc<dyn cogent_core::memory::TokenEstimator>,
}

impl TokenLimiterMiddleware {
    /// 创建 token 限制中间件。
    ///
    /// # 参数
    /// - `max_tokens`：上下文 token 上限。
    /// - `estimator`：Token 估算器。
    pub fn new(max_tokens: usize, estimator: Arc<dyn cogent_core::memory::TokenEstimator>) -> Self {
        tracing::info!(max_tokens, "TokenLimiterMiddleware initialized");
        Self {
            max_tokens,
            estimator,
        }
    }
}

#[async_trait]
impl AgentMiddleware for TokenLimiterMiddleware {
    async fn before_llm(&self, messages: &[Message]) -> CogentResult<()> {
        // 估算当前上下文总 token 数。
        // 口径与 WindowMemory 完全一致（复用 cogent_core::memory::estimate_message_tokens）：
        // content token + tool_calls 的 JSON 序列化 token，避免两处独立实现漂移。
        let total: usize = messages
            .iter()
            .map(|m| cogent_core::memory::estimate_message_tokens(&*self.estimator, m))
            .sum();

        if total > self.max_tokens {
            tracing::warn!(
                estimated_tokens = total,
                max_tokens = self.max_tokens,
                "middleware: context exceeds token limit"
            );
            return Err(CogentError::MiddlewareBlocked {
                middleware: "TokenLimiter".into(),
                reason: format!(
                    "context tokens ({total}) exceed limit ({})",
                    self.max_tokens
                ),
            });
        }
        Ok(())
    }

    async fn after_llm(&self, _response: &LLMResponse) -> CogentResult<()> {
        Ok(())
    }

    async fn before_tool(&self, _tool_name: &str, _args: &Value) -> CogentResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! 中间件单元测试。
    //!
    //! 验证各中间件的拦截/放行逻辑。

    use super::*;
    use cogent_core::memory::CharBasedEstimator;
    use cogent_core::types::Role;

    /// 验证 TracingMiddleware 放行所有调用。
    #[tokio::test]
    async fn test_tracing_passes() {
        let mw = TracingMiddleware;
        let msgs = vec![Message::new(Role::User, "hi".into())];
        let resp = LLMResponse::text("hi".into(), 5, 3);
        assert!(mw.before_llm(&msgs).await.is_ok());
        assert!(mw.after_llm(&resp).await.is_ok());
        assert!(mw.before_tool("bash", &Value::Null).await.is_ok());
    }

    /// 验证 SafetyGuardMiddleware 拦截黑名单工具。
    #[tokio::test]
    async fn test_safety_blocks() {
        let mw = SafetyGuardMiddleware::new(vec!["bash".into()]);
        assert!(mw.before_tool("bash", &Value::Null).await.is_err());
        // 非黑名单工具放行
        assert!(mw.before_tool("file_read", &Value::Null).await.is_ok());
    }

    /// 验证 SafetyGuardMiddleware 空黑名单放行所有工具。
    #[tokio::test]
    async fn test_safety_empty_blocks_all_pass() {
        let mw = SafetyGuardMiddleware::new(vec![]);
        assert!(mw.before_tool("bash", &Value::Null).await.is_ok());
    }

    /// 验证 SafetyGuardMiddleware 拦截 bash 破坏性命令模式（R1 加固）。
    #[tokio::test]
    async fn test_safety_blocks_destructive_patterns() {
        let mw = SafetyGuardMiddleware::with_default_policy();
        // rm -rf /
        let args = serde_json::json!({"command": "rm -rf /"});
        assert!(mw.before_tool("bash", &args).await.is_err());
        // mkfs
        let args = serde_json::json!({"command": "mkfs.ext4 /dev/sda1"});
        assert!(mw.before_tool("bash", &args).await.is_err());
        // 大小写不敏感
        let args = serde_json::json!({"command": "RM -RF /"});
        assert!(mw.before_tool("bash", &args).await.is_err());
    }

    /// 验证 SafetyGuardMiddleware 放行安全 bash 命令。
    #[tokio::test]
    async fn test_safety_allows_safe_bash() {
        let mw = SafetyGuardMiddleware::with_default_policy();
        let args = serde_json::json!({"command": "ls -la"});
        assert!(mw.before_tool("bash", &args).await.is_ok());
        let args = serde_json::json!({"command": "echo hello"});
        assert!(mw.before_tool("bash", &args).await.is_ok());
    }

    /// 验证 TokenLimiterMiddleware 拦截超限上下文。
    #[tokio::test]
    async fn test_token_limiter_blocks() {
        let mw = TokenLimiterMiddleware::new(10, Arc::new(CharBasedEstimator));
        // 构造一个内容很长的消息（> 40 字符 → > 10 token）
        let long = "x".repeat(100);
        let msgs = vec![Message::new(Role::User, long)];
        assert!(mw.before_llm(&msgs).await.is_err());
    }

    /// 验证 TokenLimiterMiddleware 放行未超限上下文。
    #[tokio::test]
    async fn test_token_limiter_passes() {
        let mw = TokenLimiterMiddleware::new(1000, Arc::new(CharBasedEstimator));
        let msgs = vec![Message::new(Role::User, "hi".into())];
        assert!(mw.before_llm(&msgs).await.is_ok());
    }
}
