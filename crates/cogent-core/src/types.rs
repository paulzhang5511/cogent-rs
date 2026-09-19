//! 消息与工具调用的基础数据类型。
//!
//! 本模块定义 Agent 与 LLM 交互的核心数据结构：
//! - [`Role`]：消息角色（System/User/Assistant/Tool）。
//! - [`ToolCall`]：LLM 返回的工具调用请求。
//! - [`Message`]：一条对话消息，可携带工具调用或工具结果。
//!
//! 这些类型是引擎、Provider、Memory、Tool 之间的通用数据契约，
//! 所有跨模块的消息传递均使用本模块定义的类型。
//!
//! 设计原则：
//! - 所有类型实现 `Serialize`/`Deserialize`，便于 JSON 序列化与 Provider API 对接。
//! - 所有类型实现 `JsonSchema`（通过 `schemars`），供 `#[cogent_tool]` 宏生成参数 Schema。
//! - 可选字段使用 `Option<T>` + `skip_serializing_if`，避免序列化空值。

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// 消息角色，标识消息的发送方。
///
/// 对应 LLM API 中的角色概念：
/// - `System`：系统指令（人设、行为约束），通常作为对话首条消息。
/// - `User`：用户输入。
/// - `Assistant`：LLM 的回复（可能包含工具调用）。
/// - `Tool`：工具执行结果（回传给 LLM 作为观察）。
///
/// 序列化时使用小写字符串（如 `"system"`、`"user"`），与 OpenAI API 格式一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// 系统消息：设定 Agent 的人设、行为约束、全局指令。
    System,
    /// 用户消息：用户的输入或提问。
    User,
    /// 助手消息：LLM 生成的回复，可能包含 `tool_calls`。
    Assistant,
    /// 工具消息：工具执行结果，通过 `tool_call_id` 关联到对应的工具调用。
    Tool,
}

/// LLM 返回的工具调用请求。
///
/// 当 LLM 决定调用某个工具时，会在 Assistant 消息中携带一个或多个 `ToolCall`。
/// 引擎解析后执行对应工具，并将结果以 `Role::Tool` 消息回传。
///
/// 字段说明：
/// - `id`：工具调用唯一标识，由 LLM 生成，用于关联工具结果。
/// - `name`：工具名称，对应 [`crate::tool::Tool::name`]。
/// - `arguments`：工具参数（JSON 对象），由 LLM 根据工具 Schema 生成。
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolCall {
    /// 工具调用唯一标识（由 LLM 生成，如 `"call_abc123"`）。
    pub id: String,
    /// 工具名称（如 `"bash"`、`"file_read"`）。
    pub name: String,
    /// 工具参数（JSON 对象），结构由工具的 `parameters_schema()` 定义。
    pub arguments: serde_json::Value,
    /// 模型原始工具名（映射前，如 `"RunCommand"`）。
    ///
    /// 部分 Provider 将模型工具名映射为 cogent 工具名执行，
    /// 但回传上下文时需按模型原始名渲染（模型只认自己的工具名）。
    /// 无映射时为 `None`（渲染时回退到 [`ToolCall::name`]）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_name: Option<String>,
}

impl ToolCall {
    /// 渲染用工具名：优先 [`ToolCall::original_name`]，否则 [`ToolCall::name`]。
    pub fn display_name(&self) -> &str {
        self.original_name.as_deref().unwrap_or(&self.name)
    }
}

/// 一条对话消息。
///
/// 消息是 Agent 与 LLM 交互的基本单元。根据 `role` 不同，
/// 可选字段 `tool_calls` 和 `tool_call_id` 的语义不同：
/// - `Role::Assistant` + `tool_calls`：LLM 请求调用工具。
/// - `Role::Tool` + `tool_call_id`：工具执行结果，关联到对应的工具调用。
/// - 其他角色：`tool_calls` 和 `tool_call_id` 均为 `None`。
///
/// `content` 字段在所有角色下均存在（`Role::Tool` 时为工具输出文本，
/// `Role::Assistant` 且无工具调用时为最终回复文本）。
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Message {
    /// 消息角色。
    pub role: Role,
    /// 消息文本内容。
    pub content: String,
    /// 工具调用列表（仅 `Role::Assistant` 且 LLM 决定调用工具时存在）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// 关联的工具调用 ID（仅 `Role::Tool` 时存在，指向对应的 `ToolCall::id`）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// 创建一条简单的文本消息（无工具调用、无工具关联）。
    ///
    /// # 参数
    /// - `role`：消息角色。
    /// - `content`：消息文本内容。
    ///
    /// # 示例
    /// ```
    /// use cogent_core::types::{Message, Role};
    /// let msg = Message::new(Role::User, "Hello".to_string());
    /// assert_eq!(msg.role, Role::User);
    /// assert!(msg.tool_calls.is_none());
    /// ```
    pub fn new(role: Role, content: String) -> Self {
        Self {
            role,
            content,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// 创建一条携带工具调用的 Assistant 消息。
    ///
    /// # 参数
    /// - `content`：LLM 的文本回复（可能为空字符串，当 LLM 仅调用工具时）。
    /// - `tool_calls`：工具调用列表。
    pub fn with_tool_calls(content: String, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// 创建一条工具结果消息（`Role::Tool`）。
    ///
    /// # 参数
    /// - `tool_call_id`：关联的工具调用 ID。
    /// - `content`：工具执行结果文本。
    pub fn tool_result(tool_call_id: String, content: String) -> Self {
        Self {
            role: Role::Tool,
            content,
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
        }
    }

    /// 判断此消息是否包含工具调用。
    ///
    /// 用于引擎判断 LLM 是否请求调用工具（ReAct 循环的分支条件）。
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    }
}

#[cfg(test)]
mod tests {
    //! 消息类型单元测试。
    //!
    //! 验证 `Message` 构造方法、`Role` 序列化格式、`ToolCall` 结构。

    use super::*;

    /// 验证 `Message::new` 创建简单消息。
    #[test]
    fn test_message_new() {
        let msg = Message::new(Role::User, "Hello".into());
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello");
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());
        assert!(!msg.has_tool_calls());
    }

    /// 验证 `Message::with_tool_calls` 创建携带工具调用的消息。
    #[test]
    fn test_message_with_tool_calls() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
            original_name: None,
        };
        let msg = Message::with_tool_calls(String::new(), vec![call]);
        assert_eq!(msg.role, Role::Assistant);
        assert!(msg.has_tool_calls());
        assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 1);
    }

    /// 验证 `Message::tool_result` 创建工具结果消息。
    #[test]
    fn test_message_tool_result() {
        let msg = Message::tool_result("call_1".into(), "file1\nfile2".into());
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_1"));
        assert!(!msg.has_tool_calls());
    }

    /// 验证 `Role` 序列化为小写字符串（与 OpenAI API 格式一致）。
    #[test]
    fn test_role_serialization() {
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
    }

    /// 验证 `Role` 从小写字符串反序列化。
    #[test]
    fn test_role_deserialization() {
        let role: Role = serde_json::from_str("\"system\"").unwrap();
        assert_eq!(role, Role::System);
        let role: Role = serde_json::from_str("\"tool\"").unwrap();
        assert_eq!(role, Role::Tool);
    }

    /// 验证 `Message` 序列化时跳过 `None` 字段。
    #[test]
    fn test_message_serialization_skips_none() {
        let msg = Message::new(Role::User, "test".into());
        let json = serde_json::to_value(&msg).unwrap();
        // 简单消息不应包含 tool_calls 和 tool_call_id 字段
        assert!(json.get("tool_calls").is_none());
        assert!(json.get("tool_call_id").is_none());
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "test");
    }

    /// 验证 `Message` 序列化时包含工具调用字段。
    #[test]
    fn test_message_serialization_with_tool_calls() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
            original_name: None,
        };
        let msg = Message::with_tool_calls("thinking...".into(), vec![call]);
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json.get("tool_calls").is_some());
        assert_eq!(json["tool_calls"][0]["name"], "bash");
    }

    /// 验证 `Message` 完整往返序列化（serialize → deserialize 一致性）。
    #[test]
    fn test_message_roundtrip() {
        let call = ToolCall {
            id: "call_abc".into(),
            name: "file_read".into(),
            arguments: serde_json::json!({"path": "/tmp/test.txt"}),
            original_name: None,
        };
        let original = Message::with_tool_calls("Let me read the file.".into(), vec![call]);
        let json = serde_json::to_string(&original).unwrap();
        let restored: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.role, original.role);
        assert_eq!(restored.content, original.content);
        assert_eq!(restored.tool_calls.as_ref().unwrap()[0].id, "call_abc");
    }
}
