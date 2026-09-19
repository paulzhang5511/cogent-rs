//! 中间件 Trait 定义。
//!
//! 本模块定义 [`AgentMiddleware`] trait——引擎执行管道中的拦截器契约。
//! 中间件在 LLM 调用前后、工具执行前插入横切逻辑（追踪、安全、资源限制）。
//!
//! 中间件管道（PRD §2.2）：
//! ```text
//! [Tracing] ──► [Safety Guard] ──► [Token Limiter]
//! ```
//!
//! 设计原则：
/// - 中间件是 `Send + Sync` 的，可安全地在引擎中共享。
/// - 中间件通过 `before_*`/`after_*` 钩子拦截，不改变主流程数据。
/// - 中间件可返回错误（拦截），引擎据此终止当前操作。
/// - 中间件不持有状态（无状态），便于组合和测试。
use async_trait::async_trait;
use serde_json::Value;

use crate::error::CogentResult;
use crate::provider::LLMResponse;
use crate::types::Message;

/// 引擎执行管道中的中间件契约。
///
/// 中间件在引擎的关键节点插入横切逻辑：
/// - `before_llm`：LLM 调用前（可拦截、可修改上下文）。
/// - `after_llm`：LLM 调用后（可记录、可后处理）。
/// - `before_tool`：工具执行前（可拦截、可校验参数）。
///
/// # 实现要求
/// - 所有方法为异步（`async_trait`）。
/// - 返回 `CogentResult<()>`：`Ok(())` 表示放行，`Err` 表示拦截。
/// - 中间件应快速返回（不做 IO 密集型操作），避免阻塞引擎。
///
/// # 示例
/// ```
/// use async_trait::async_trait;
/// use cogent_core::middleware::AgentMiddleware;
/// use cogent_core::error::CogentResult;
/// use cogent_core::types::Message;
/// use serde_json::Value;
///
/// struct LoggingMiddleware;
///
/// #[async_trait]
/// impl AgentMiddleware for LoggingMiddleware {
///     async fn before_llm(&self, _messages: &[Message]) -> CogentResult<()> {
///         tracing::debug!("before_llm called");
///         Ok(())
///     }
///     async fn after_llm(&self, _response: &cogent_core::provider::LLMResponse) -> CogentResult<()> {
///         tracing::debug!("after_llm called");
///         Ok(())
///     }
///     async fn before_tool(&self, _tool_name: &str, _args: &Value) -> CogentResult<()> {
///         tracing::debug!(tool = _tool_name, "before_tool called");
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait AgentMiddleware: Send + Sync {
    /// LLM 调用前的拦截钩子。
    ///
    /// 在引擎调用 `LLMProvider::chat_complete` 之前执行。
    /// 可用于：
    /// - 记录 LLM 调用（TracingMiddleware）。
    /// - 校验上下文 token 不超上限（TokenLimiterMiddleware）。
    ///
    /// # 参数
    /// - `messages`：即将发送给 LLM 的消息列表。
    ///
    /// # 返回
    /// - `Ok(())`：放行，继续调用 LLM。
    /// - `Err(CogentError)`：拦截，引擎终止当前操作。
    async fn before_llm(&self, messages: &[Message]) -> CogentResult<()>;

    /// LLM 调用后的拦截钩子。
    ///
    /// 在引擎收到 `LLMResponse` 之后执行。
    /// 可用于：
    /// - 记录 LLM 响应摘要（TracingMiddleware）。
    ///
    /// # 参数
    /// - `response`：LLM 返回的结构化响应（含 token 用量、工具调用等）。
    ///
    /// # 返回
    /// - `Ok(())`：放行。
    /// - `Err(CogentError)`：拦截（罕见，通常 after 钩子不拦截）。
    async fn after_llm(&self, response: &LLMResponse) -> CogentResult<()>;

    /// 工具执行前的拦截钩子。
    ///
    /// 在引擎执行 `Tool::execute` 之前执行。
    /// 可用于：
    /// - 记录工具调用（TracingMiddleware）。
    /// - 校验工具参数安全（SafetyGuardMiddleware）。
    ///
    /// # 参数
    /// - `tool_name`：工具名称。
    /// - `args`：工具参数（JSON 对象）。
    ///
    /// # 返回
    /// - `Ok(())`：放行，继续执行工具。
    /// - `Err(CogentError)`：拦截，引擎跳过该工具执行。
    async fn before_tool(&self, tool_name: &str, args: &Value) -> CogentResult<()>;
}

#[cfg(test)]
mod tests {
    //! AgentMiddleware trait 单元测试。
    //!
    /// 验证 trait 可被实现、可组合调用。
    use super::*;

    /// 测试用放行中间件（所有钩子返回 Ok）。
    struct PassThroughMiddleware;

    #[async_trait]
    impl AgentMiddleware for PassThroughMiddleware {
        async fn before_llm(&self, _messages: &[Message]) -> CogentResult<()> {
            Ok(())
        }
        async fn after_llm(&self, _response: &LLMResponse) -> CogentResult<()> {
            Ok(())
        }
        async fn before_tool(&self, _tool_name: &str, _args: &Value) -> CogentResult<()> {
            Ok(())
        }
    }

    /// 测试用拦截中间件（before_tool 返回 Err）。
    struct BlockingMiddleware;

    #[async_trait]
    impl AgentMiddleware for BlockingMiddleware {
        async fn before_llm(&self, _messages: &[Message]) -> CogentResult<()> {
            Ok(())
        }
        async fn after_llm(&self, _response: &LLMResponse) -> CogentResult<()> {
            Ok(())
        }
        async fn before_tool(&self, tool_name: &str, _args: &Value) -> CogentResult<()> {
            Err(crate::error::CogentError::MiddlewareBlocked {
                middleware: "BlockingMiddleware".into(),
                reason: format!("tool '{tool_name}' is blocked"),
            })
        }
    }

    /// 验证放行中间件所有钩子返回 Ok。
    #[tokio::test]
    async fn test_pass_through_middleware() {
        let mw = PassThroughMiddleware;
        let msgs = vec![Message::new(crate::types::Role::User, "hi".into())];
        let resp = LLMResponse::text("hi".into(), 5, 3);
        assert!(mw.before_llm(&msgs).await.is_ok());
        assert!(mw.after_llm(&resp).await.is_ok());
        assert!(mw.before_tool("bash", &serde_json::json!({})).await.is_ok());
    }

    /// 验证拦截中间件 before_tool 返回 Err。
    #[tokio::test]
    async fn test_blocking_middleware() {
        let mw = BlockingMiddleware;
        let result = mw
            .before_tool("bash", &serde_json::json!({"command": "rm -rf /"}))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            crate::error::CogentError::MiddlewareBlocked { .. }
        ));
    }

    /// 验证中间件是 Send + Sync。
    #[test]
    fn test_middleware_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PassThroughMiddleware>();
        assert_send_sync::<BlockingMiddleware>();
    }
}
