//! OpenAI LLM Provider 实现。
//!
//! 本模块实现 [`cogent_core::provider::LLMProvider`] trait 的 OpenAI 接入：
//! - [`OpenAIProvider`]：封装 `reqwest` 客户端，调用 OpenAI Chat Completions API。
//! - `chat_complete`：非流式调用，返回完整 [`LLMResponse`]（含 `tool_calls`）。
//! - `chat_stream`：流式调用（SSE），逐 token chunk 返回文本片段。
//!
//! # 设计原则
//! - 请求/响应序列化结构体为模块私有，仅暴露 `LLMResponse`。
//! - 不在内部重试（重试由 `RetryMiddleware` 装饰器处理）。
//! - 所有 `tracing` 日志用英文，携带结构化字段（`model`、`message_count`、`tool_count` 等）。
//! - API Key 绝不写入日志。

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use cogent_core::provider::{LLMProvider, LLMResponse};
use cogent_core::tool::Tool;
use cogent_core::types::{Message, ToolCall};

/// OpenAI API 默认 base URL。
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// 非流式请求超时（秒）。
const COMPLETE_TIMEOUT_SECS: u64 = 60;

/// 流式请求超时（秒）。
const STREAM_TIMEOUT_SECS: u64 = 120;

// ─── 请求/响应序列化结构体（模块私有） ───────────────────────────────────────

/// OpenAI Chat Completions 请求体。
#[derive(Debug, Serialize)]
struct ChatRequest {
    /// 模型名称。
    model: String,
    /// 对话消息列表。
    messages: Vec<ChatMessage>,
    /// 可用工具列表（可选）。
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ChatTool>>,
    /// 是否启用流式输出。
    stream: bool,
}

/// OpenAI 消息格式（与 `cogent_core::types::Message` 对齐，但字段名匹配 API）。
#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    /// 消息角色（`"system"` / `"user"` / `"assistant"` / `"tool"`）。
    role: String,
    /// 消息内容。
    content: String,
    /// 工具调用列表（仅 assistant 消息）。
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCall>>,
    /// 关联的工具调用 ID（仅 tool 消息）。
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// OpenAI 工具调用格式。
#[derive(Debug, Serialize, Deserialize)]
struct ChatToolCall {
    /// 工具调用唯一标识。
    id: String,
    /// 调用类型（固定为 `"function"`）。
    #[serde(rename = "type")]
    call_type: String,
    /// 函数调用详情。
    function: ChatFunctionCall,
}

/// OpenAI 函数调用详情。
#[derive(Debug, Serialize, Deserialize)]
struct ChatFunctionCall {
    /// 函数名称。
    name: String,
    /// 函数参数（JSON 字符串）。
    arguments: String,
}

/// OpenAI 工具定义格式。
#[derive(Debug, Serialize)]
struct ChatTool {
    /// 工具类型（固定为 `"function"`）。
    #[serde(rename = "type")]
    tool_type: String,
    /// 函数定义。
    function: ChatFunctionDef,
}

/// OpenAI 函数定义。
#[derive(Debug, Serialize)]
struct ChatFunctionDef {
    /// 函数名称。
    name: String,
    /// 函数描述。
    description: String,
    /// 参数 JSON Schema。
    parameters: serde_json::Value,
}

/// OpenAI Chat Completions 非流式响应体。
#[derive(Debug, Deserialize)]
struct ChatResponse {
    /// 响应 ID（API 契约字段，仅用于日志追踪）。
    #[allow(dead_code)]
    id: String,
    /// 响应数据列表。
    choices: Vec<ChatChoice>,
    /// 用量统计。
    #[serde(default)]
    usage: Option<ChatUsage>,
}

/// OpenAI 响应 choice。
#[derive(Debug, Deserialize)]
struct ChatChoice {
    /// 消息内容。
    message: ChatMessage,
    /// 结束原因（`"stop"` / `"tool_calls"` / `"length"` 等）。
    #[allow(dead_code)]
    #[serde(default)]
    finish_reason: Option<String>,
}

/// OpenAI 用量统计。
#[derive(Debug, Deserialize)]
struct ChatUsage {
    /// 输入 token 数。
    prompt_tokens: u32,
    /// 输出 token 数。
    completion_tokens: u32,
}

/// OpenAI SSE 流式响应 chunk。
#[derive(Debug, Deserialize)]
struct StreamChunk {
    /// 响应 ID（API 契约字段，仅用于日志追踪）。
    #[allow(dead_code)]
    #[serde(default)]
    id: Option<String>,
    /// 响应数据列表。
    choices: Vec<StreamChoice>,
}

/// OpenAI SSE 流式 choice。
#[derive(Debug, Deserialize)]
struct StreamChoice {
    /// 增量消息。
    delta: StreamDelta,
    /// 结束原因（API 契约字段，流式结束标记）。
    #[allow(dead_code)]
    #[serde(default)]
    finish_reason: Option<String>,
}

/// OpenAI SSE 流式增量消息。
#[derive(Debug, Deserialize)]
struct StreamDelta {
    /// 增量文本内容。
    #[serde(default)]
    content: Option<String>,
}

// ─── OpenAIProvider 实现 ─────────────────────────────────────────────────────

/// OpenAI LLM Provider。
///
/// 封装 `reqwest` 客户端与 OpenAI API 配置，实现 [`LLMProvider`] trait。
/// 通过 [`OpenAIProvider::new`] 构造，`api_key` 和 `model` 必填，
/// `base_url` 可选（默认 `https://api.openai.com/v1`）。
///
/// # 线程安全
/// `OpenAIProvider` 是 `Send + Sync` 的，可安全地在多个异步任务间共享
/// （通过 `Arc<OpenAIProvider>` 传递给引擎）。
pub struct OpenAIProvider {
    /// HTTP 客户端（内部连接池复用）。
    client: reqwest::Client,
    /// API Key（绝不写入日志）。
    api_key: String,
    /// 模型名称。
    model: String,
    /// API base URL。
    base_url: String,
}

impl OpenAIProvider {
    /// 创建新的 OpenAIProvider。
    ///
    /// # 参数
    /// - `api_key`：OpenAI API Key（必填，非空）。
    /// - `model`：模型名称（如 `"gpt-4o-mini"`）。
    /// - `base_url`：自定义 API endpoint（`None` 则使用默认 `https://api.openai.com/v1`）。
    ///
    /// # 返回
    /// - 成功：构造的 `OpenAIProvider`。
    /// - 失败：`anyhow::Error`（`api_key` 为空或 `base_url` 格式无效）。
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> anyhow::Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("OpenAI API key must not be empty");
        }

        let base_url = base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
            .trim_end_matches('/')
            .to_string();

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(STREAM_TIMEOUT_SECS))
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build HTTP client: {e}"))?;

        tracing::info!(
            model = %model,
            base_url = %base_url,
            "OpenAIProvider initialized"
        );

        Ok(Self {
            client,
            api_key,
            model,
            base_url,
        })
    }

    /// 将 `cogent_core::types::Message` 转换为 OpenAI API 的 `ChatMessage` 格式。
    fn to_chat_message(msg: &Message) -> ChatMessage {
        let role = match msg.role {
            cogent_core::types::Role::System => "system",
            cogent_core::types::Role::User => "user",
            cogent_core::types::Role::Assistant => "assistant",
            cogent_core::types::Role::Tool => "tool",
        };

        let tool_calls = msg.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|tc| ChatToolCall {
                    id: tc.id.clone(),
                    call_type: "function".to_string(),
                    function: ChatFunctionCall {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    },
                })
                .collect()
        });

        ChatMessage {
            role: role.to_string(),
            content: msg.content.clone(),
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
        }
    }

    /// 将工具列表转换为 OpenAI API 的 `ChatTool` 格式。
    fn to_chat_tools(tools: &[Arc<dyn Tool>]) -> Option<Vec<ChatTool>> {
        if tools.is_empty() {
            return None;
        }
        Some(
            tools
                .iter()
                .map(|t| ChatTool {
                    tool_type: "function".to_string(),
                    function: ChatFunctionDef {
                        name: t.name().to_string(),
                        description: t.description().to_string(),
                        parameters: t.parameters_schema(),
                    },
                })
                .collect(),
        )
    }

    /// 构建请求 URL。
    fn request_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// 构建请求头。
    fn request_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", self.api_key)
                .parse()
                .expect("valid header value"),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        headers
    }

    /// 解析非流式响应为 `LLMResponse`。
    fn parse_response(resp: ChatResponse) -> anyhow::Result<LLMResponse> {
        let choice = resp
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("OpenAI response contains no choices"))?;

        let content = choice.message.content.clone();

        let tool_calls: Vec<ToolCall> = choice
            .message
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls
                    .iter()
                    .map(|tc| {
                        let arguments: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments)
                                .unwrap_or(serde_json::json!({}));
                        ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments,
                            original_name: None,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let (prompt_tokens, completion_tokens) = resp
            .usage
            .map(|u| (u.prompt_tokens, u.completion_tokens))
            .unwrap_or((0, 0));

        Ok(LLMResponse {
            content,
            tool_calls,
            prompt_tokens,
            completion_tokens,
        })
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    async fn chat_complete(
        &self,
        messages: &[Message],
        tools: &[Arc<dyn Tool>],
    ) -> anyhow::Result<LLMResponse> {
        let url = self.request_url();
        let chat_messages: Vec<ChatMessage> = messages.iter().map(Self::to_chat_message).collect();
        let chat_tools = Self::to_chat_tools(tools);

        let request = ChatRequest {
            model: self.model.clone(),
            messages: chat_messages,
            tools: chat_tools,
            stream: false,
        };

        tracing::info!(
            model = %self.model,
            message_count = messages.len(),
            tool_count = tools.len(),
            "sending chat_complete request"
        );

        let response = self
            .client
            .post(&url)
            .headers(self.request_headers())
            .timeout(Duration::from_secs(COMPLETE_TIMEOUT_SECS))
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("HTTP request failed: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            tracing::error!(
                status_code = %status,
                body = %body,
                "OpenAI API returned error"
            );
            anyhow::bail!("OpenAI API error (status {status}): {body}");
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("failed to parse OpenAI response: {e}"))?;

        let llm_response = Self::parse_response(chat_response)?;

        tracing::info!(
            model = %self.model,
            prompt_tokens = llm_response.prompt_tokens,
            completion_tokens = llm_response.completion_tokens,
            has_tool_calls = llm_response.has_tool_calls(),
            "chat_complete response received"
        );

        Ok(llm_response)
    }

    fn chat_stream(
        &self,
        messages: &[Message],
        _tools: &[Arc<dyn Tool>],
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>> {
        let url = self.request_url();
        let chat_messages: Vec<ChatMessage> = messages.iter().map(Self::to_chat_message).collect();

        let request = ChatRequest {
            model: self.model.clone(),
            messages: chat_messages,
            tools: None,
            stream: true,
        };

        let client = self.client.clone();
        let headers = self.request_headers();
        let model = self.model.clone();

        tracing::info!(
            model = %model,
            message_count = messages.len(),
            "sending chat_stream request"
        );

        let stream = async_stream::stream! {
            let response = match client
                .post(&url)
                .headers(headers)
                .timeout(Duration::from_secs(STREAM_TIMEOUT_SECS))
                .json(&request)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    tracing::error!(error = %e, "chat_stream HTTP request failed");
                    yield Err(anyhow::anyhow!("HTTP request failed: {e}"));
                    return;
                }
            };

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                tracing::error!(
                    status_code = %status,
                    body = %body,
                    "OpenAI stream API returned error"
                );
                yield Err(anyhow::anyhow!("OpenAI API error (status {status}): {body}"));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            use futures::StreamExt;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        // 按行解析 SSE 事件
                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].trim().to_string();
                            buffer.drain(..=newline_pos);

                            // SSE 格式：`data: {...}` 或 `data: [DONE]`
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" {
                                    tracing::info!(model = %model, "chat_stream completed");
                                    return;
                                }
                                match serde_json::from_str::<StreamChunk>(data) {
                                    Ok(chunk) => {
                                        if let Some(delta) = chunk.choices.first().and_then(|c| c.delta.content.as_deref())
                                            && !delta.is_empty()
                                        {
                                            yield Ok(delta.to_string());
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            error = %e,
                                            data = %data,
                                            "failed to parse SSE chunk"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "chat_stream read error");
                        yield Err(anyhow::anyhow!("stream read error: {e}"));
                        return;
                    }
                }
            }

            tracing::info!(model = %model, "chat_stream ended (no [DONE] marker)");
        };

        Box::pin(stream)
    }
}

#[cfg(test)]
mod tests {
    //! OpenAIProvider 单元测试。
    //!
    //! 验证请求/响应序列化结构、消息转换、工具转换、响应解析。
    //! 不发起真实网络请求（网络测试由集成测试覆盖）。

    use super::*;

    /// 验证 `to_chat_message` 正确转换简单用户消息。
    #[test]
    fn test_to_chat_message_user() {
        let msg = Message::new(cogent_core::types::Role::User, "Hello".into());
        let chat_msg = OpenAIProvider::to_chat_message(&msg);
        assert_eq!(chat_msg.role, "user");
        assert_eq!(chat_msg.content, "Hello");
        assert!(chat_msg.tool_calls.is_none());
        assert!(chat_msg.tool_call_id.is_none());
    }

    /// 验证 `to_chat_message` 正确转换携带工具调用的 assistant 消息。
    #[test]
    fn test_to_chat_message_assistant_with_tool_calls() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
            original_name: None,
        };
        let msg = Message::with_tool_calls("Let me check.".into(), vec![call]);
        let chat_msg = OpenAIProvider::to_chat_message(&msg);
        assert_eq!(chat_msg.role, "assistant");
        let calls = chat_msg.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments, "{\"command\":\"ls\"}");
    }

    /// 验证 `to_chat_message` 正确转换工具结果消息。
    #[test]
    fn test_to_chat_message_tool_result() {
        let msg = Message::tool_result("call_1".into(), "file1\nfile2".into());
        let chat_msg = OpenAIProvider::to_chat_message(&msg);
        assert_eq!(chat_msg.role, "tool");
        assert_eq!(chat_msg.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(chat_msg.content, "file1\nfile2");
    }

    /// 验证 `to_chat_tools` 空列表返回 None。
    #[test]
    fn test_to_chat_tools_empty() {
        let tools: Vec<Arc<dyn Tool>> = Vec::new();
        assert!(OpenAIProvider::to_chat_tools(&tools).is_none());
    }

    /// 验证 `parse_response` 正确解析纯文本响应。
    #[test]
    fn test_parse_response_text() {
        let resp = ChatResponse {
            id: "chatcmpl-123".into(),
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".into(),
                    content: "Hello!".into(),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            }),
        };
        let llm_resp = OpenAIProvider::parse_response(resp).unwrap();
        assert_eq!(llm_resp.content, "Hello!");
        assert!(!llm_resp.has_tool_calls());
        assert_eq!(llm_resp.prompt_tokens, 10);
        assert_eq!(llm_resp.completion_tokens, 5);
    }

    /// 验证 `parse_response` 正确解析带工具调用的响应。
    #[test]
    fn test_parse_response_with_tool_calls() {
        let resp = ChatResponse {
            id: "chatcmpl-456".into(),
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".into(),
                    content: String::new(),
                    tool_calls: Some(vec![ChatToolCall {
                        id: "call_abc".into(),
                        call_type: "function".into(),
                        function: ChatFunctionCall {
                            name: "bash".into(),
                            arguments: "{\"command\":\"ls\"}".into(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: 20,
                completion_tokens: 10,
            }),
        };
        let llm_resp = OpenAIProvider::parse_response(resp).unwrap();
        assert!(llm_resp.has_tool_calls());
        assert_eq!(llm_resp.tool_calls[0].id, "call_abc");
        assert_eq!(llm_resp.tool_calls[0].name, "bash");
        assert_eq!(llm_resp.tool_calls[0].arguments["command"], "ls");
    }

    /// 验证 `parse_response` 在无 choices 时返回错误。
    #[test]
    fn test_parse_response_no_choices() {
        let resp = ChatResponse {
            id: "chatcmpl-789".into(),
            choices: Vec::new(),
            usage: None,
        };
        assert!(OpenAIProvider::parse_response(resp).is_err());
    }

    /// 验证 `OpenAIProvider::new` 在空 API key 时返回错误。
    #[test]
    fn test_new_empty_api_key() {
        assert!(OpenAIProvider::new(String::new(), "gpt-4o-mini".into(), None).is_err());
    }

    /// 验证 `OpenAIProvider::new` 成功构造。
    #[test]
    fn test_new_success() {
        let provider = OpenAIProvider::new("sk-test".into(), "gpt-4o-mini".into(), None).unwrap();
        assert_eq!(provider.model, "gpt-4o-mini");
        assert_eq!(provider.base_url, DEFAULT_BASE_URL);
    }

    /// 验证 `OpenAIProvider::new` 自定义 base_url 去除尾部斜杠。
    #[test]
    fn test_new_custom_base_url_trims_trailing_slash() {
        let provider = OpenAIProvider::new(
            "sk-test".into(),
            "gpt-4o-mini".into(),
            Some("http://localhost:8080/v1/".into()),
        )
        .unwrap();
        assert_eq!(provider.base_url, "http://localhost:8080/v1");
    }

    /// 验证 `request_url` 正确拼接。
    #[test]
    fn test_request_url() {
        let provider = OpenAIProvider::new("sk-test".into(), "gpt-4o-mini".into(), None).unwrap();
        assert_eq!(
            provider.request_url(),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    /// 验证 `ChatRequest` 序列化时 `stream: false` 且无 tools 时跳过 tools 字段。
    #[test]
    fn test_chat_request_serialization() {
        let req = ChatRequest {
            model: "gpt-4o-mini".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            stream: false,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-4o-mini");
        assert_eq!(json["stream"], false);
        assert!(json.get("tools").is_none());
    }

    /// 验证 `StreamChunk` 反序列化。
    #[test]
    fn test_stream_chunk_deserialization() {
        let data =
            r#"{"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: StreamChunk = serde_json::from_str(data).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
    }

    /// 验证 `StreamChunk` 反序列化空 delta。
    #[test]
    fn test_stream_chunk_empty_delta() {
        let data = r#"{"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: StreamChunk = serde_json::from_str(data).unwrap();
        assert!(chunk.choices[0].delta.content.is_none());
    }
}
