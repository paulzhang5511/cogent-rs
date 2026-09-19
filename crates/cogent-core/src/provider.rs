//! LLM Provider Trait 定义。
//!
//! 本模块定义 [`LLMProvider`] trait——LLM API 接入的统一契约，
//! 以及 [`LLMResponse`]（LLM 调用的结构化响应）。
//!
//! 引擎通过 `Arc<dyn LLMProvider>` 调用 LLM，无需感知具体 API（OpenAI/Claude/Ollama）。
//! v1 仅实现 OpenAI（见 `cogent-providers` crate），后续按同一 trait 扩展。
//!
//! 设计原则：
//! - Provider 是 `Send + Sync` 的，可安全地在多个任务间共享。
//! - `chat_complete` 用于引擎内部 ReAct 循环（需结构化 `tool_calls`）。
//! - `chat_stream` 供调用方做真流式渲染；引擎本身不调用它（最终答案打字机效果
//!   由 `AgentEngine::render_answer_chunked` 对已算好的答案分块实现，避免二次 LLM 请求）。
//! - 请求/响应序列化结构体在 provider 实现中私有，仅暴露 `LLMResponse`。

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use crate::types::Message;

/// LLM 调用的结构化响应。
///
/// 封装 LLM 返回的核心信息：
/// - `content`：文本回复（可能为空，当 LLM 仅调用工具时）。
/// - `tool_calls`：工具调用请求列表（可能为空，当 LLM 仅回复文本时）。
/// - `prompt_tokens`：输入 token 数（用于用量统计）。
/// - `completion_tokens`：输出 token 数（用于用量统计）。
///
/// `content` 和 `tool_calls` 至少有一个非空（LLM 要么回复文本，要么调用工具）。
#[derive(Debug, Clone)]
pub struct LLMResponse {
    /// 文本回复内容（可能为空字符串）。
    pub content: String,
    /// 工具调用请求列表（可能为空）。
    pub tool_calls: Vec<crate::types::ToolCall>,
    /// 输入（prompt）token 数。
    pub prompt_tokens: u32,
    /// 输出（completion）token 数。
    pub completion_tokens: u32,
}

impl LLMResponse {
    /// 判断响应是否包含工具调用。
    ///
    /// 用于引擎判断 LLM 是否请求调用工具（ReAct 循环的分支条件）。
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// 创建纯文本响应（无工具调用）。
    pub fn text(content: String, prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            content,
            tool_calls: Vec::new(),
            prompt_tokens,
            completion_tokens,
        }
    }

    /// 创建带工具调用的响应。
    pub fn with_tool_calls(
        content: String,
        tool_calls: Vec<crate::types::ToolCall>,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> Self {
        Self {
            content,
            tool_calls,
            prompt_tokens,
            completion_tokens,
        }
    }
}

/// LLM API 接入的统一契约。
///
/// 实现此 trait 的类型即可被引擎用作 LLM 后端。
/// v1 实现：`OpenAIProvider`（见 `cogent-providers` crate）。
///
/// # 方法说明
/// - `chat_complete`：非流式调用，返回完整 `LLMResponse`（含 `tool_calls`）。
///   用于引擎内部 ReAct 循环（需要结构化工具调用）。
/// - `chat_stream`：流式调用，返回 `Stream<Item = Result<String>>`（逐 token chunk）。
///   供调用方直接做真流式渲染。**注意**：引擎 ReAct 循环只用 `chat_complete`，
///   最终答案的打字机效果由 [`crate::engine::AgentEngine::render_answer_chunked`]
///   对已算好的答案分块实现（不发起第二次 LLM 请求），故引擎本身不调用 `chat_stream`。
///
/// # 实现要求
/// - 实现应处理网络错误、超时、API 错误，返回有意义的 `anyhow::Error`。
/// - 实现应设置合理的超时（建议 60s 非流式、120s 流式）。
/// - 实现不应在内部重试（重试由 `RetryMiddleware` 装饰器处理）。
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// 非流式 LLM 调用（返回完整响应）。
    ///
    /// # 参数
    /// - `messages`：对话历史（含系统消息、用户消息、助手消息、工具结果）。
    /// - `tools`：可用工具列表（LLM 可选择调用）。
    ///
    /// # 返回
    /// - 成功：`LLMResponse`（含文本回复和/或工具调用请求）。
    /// - 失败：`anyhow::Error`（网络错误、API 错误、超时等）。
    async fn chat_complete(
        &self,
        messages: &[Message],
        tools: &[std::sync::Arc<dyn crate::tool::Tool>],
    ) -> anyhow::Result<LLMResponse>;

    /// 流式 LLM 调用（逐 token chunk 返回）。
    ///
    /// # 参数
    /// - `messages`：对话历史。
    /// - `tools`：可用工具列表（流式调用通常不传工具，仅用于最终答案渲染）。
    ///
    /// # 返回
    /// - `Stream<Item = Result<String>>`：逐 token chunk 的异步流。
    ///   每个 `Ok(String)` 是一个 token 片段，`Err` 表示流中断。
    ///
    /// # 注意
    /// 流式调用不返回 `tool_calls`（仅文本），因此仅用于最终答案渲染，
    /// 不用于 ReAct 循环（ReAct 循环用 `chat_complete`）。
    /// 引擎本身不调用此方法——最终答案的打字机效果由
    /// [`crate::engine::AgentEngine::render_answer_chunked`] 对已算好的答案分块实现，
    /// 以避免发起第二次 LLM 请求。此方法供需要真流式的调用方直接使用。
    fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[std::sync::Arc<dyn crate::tool::Tool>],
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>>;
}

#[cfg(test)]
mod tests {
    //! LLMProvider trait 单元测试。
    //!
    //! 验证 LLMResponse 构造方法、`has_tool_calls` 判断。

    use super::*;
    use crate::types::ToolCall;

    /// 验证 LLMResponse::text 创建纯文本响应。
    #[test]
    fn test_llm_response_text() {
        let resp = LLMResponse::text("Hello!".into(), 10, 5);
        assert_eq!(resp.content, "Hello!");
        assert!(!resp.has_tool_calls());
        assert_eq!(resp.prompt_tokens, 10);
        assert_eq!(resp.completion_tokens, 5);
    }

    /// 验证 LLMResponse::with_tool_calls 创建带工具调用的响应。
    #[test]
    fn test_llm_response_with_tool_calls() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
            original_name: None,
        };
        let resp = LLMResponse::with_tool_calls(String::new(), vec![call], 20, 10);
        assert!(resp.has_tool_calls());
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "bash");
    }

    /// 验证空 tool_calls 的 has_tool_calls 返回 false。
    #[test]
    fn test_has_tool_calls_empty() {
        let resp = LLMResponse::text("Hi".into(), 5, 3);
        assert!(!resp.has_tool_calls());
    }
}
