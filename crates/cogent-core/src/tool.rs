//! 工具 Trait 定义。
//!
//! 本模块定义 [`Tool`] trait——Agent 可调用工具的统一契约。
//! 所有工具（Bash、File IO、HTTP 等）均实现此 trait，
//! 引擎通过 `Arc<dyn Tool>` 持有工具，无需感知具体实现。
//!
//! 设计原则：
//! - 工具是 `Send + Sync` 的，可安全地在多个异步任务间共享（并行执行）。
//! - `execute` 是异步的，支持 IO 密集型工具（HTTP、文件读写）。
//! - `parameters_schema()` 返回 JSON Schema，供 LLM 理解工具参数结构。
//! - 工具名称和描述是 LLM 选择工具的依据，必须准确、无歧义。

use async_trait::async_trait;
use serde_json::Value;

/// Agent 可调用工具的统一契约。
///
/// 实现此 trait 的类型即可被引擎注册为工具，LLM 可通过 `tool_calls`
/// 请求调用。
///
/// # 实现要求
/// - `name()`：返回工具唯一名称（小写 snake_case，如 `"bash"`、`"file_read"`）。
/// - `description()`：返回工具用途描述（供 LLM 理解何时调用此工具）。
/// - `parameters_schema()`：返回参数 JSON Schema（供 LLM 生成正确参数）。
/// - `execute()`：异步执行工具，返回结果文本。
///
/// # 示例
/// ```
/// use async_trait::async_trait;
/// use cogent_core::tool::Tool;
///
/// struct EchoTool;
///
/// #[async_trait]
/// impl Tool for EchoTool {
///     fn name(&self) -> &str { "echo" }
///     fn description(&self) -> &str { "Echoes the input back." }
///     fn parameters_schema(&self) -> serde_json::Value {
///         serde_json::json!({
///             "type": "object",
///             "properties": {
///                 "text": { "type": "string" }
///             },
///             "required": ["text"]
///         })
///     }
///     async fn execute(&self, args: serde_json::Value) -> anyhow::Result<String> {
///         Ok(args["text"].as_str().unwrap_or("").to_string())
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// 工具唯一名称。
    ///
    /// 名称是 LLM 在 `tool_calls` 中引用工具的标识，必须与
    /// `parameters_schema()` 中的工具名一致。
    ///
    /// 命名约定：小写 snake_case（如 `"bash"`、`"file_read"`、`"http_get"`）。
    fn name(&self) -> &str;

    /// 工具用途描述。
    ///
    /// 描述是 LLM 决定"何时调用此工具"的关键依据，应包含：
    /// - 工具做什么（功能）。
    /// - 适用场景（何时用）。
    /// - 参数含义（各参数代表什么）。
    ///
    /// 描述用英文（与 LLM 交互语言一致）。
    fn description(&self) -> &str;

    /// 工具参数的 JSON Schema。
    ///
    /// 返回符合 JSON Schema 规范的 `Value`，描述工具接受的参数结构。
    /// LLM 根据此 Schema 生成 `arguments` 字段。
    ///
    /// 典型结构：
    /// ```json
    /// {
    ///   "type": "object",
    ///   "properties": {
    ///     "param_name": { "type": "string", "description": "..." }
    ///   },
    ///   "required": ["param_name"]
    /// }
    /// ```
    fn parameters_schema(&self) -> Value;

    /// 异步执行工具。
    ///
    /// # 参数
    /// - `args`：工具参数（JSON 对象），由 LLM 根据 `parameters_schema()` 生成。
    ///
    /// # 返回
    /// - 成功：工具执行结果文本（回传给 LLM 作为 `Role::Tool` 消息内容）。
    /// - 失败：`anyhow::Error`，描述执行错误（引擎会将其作为工具失败处理）。
    ///
    /// # 实现注意
    /// - 实现应处理参数校验（缺失字段、类型错误），返回有意义的错误信息。
    /// - 实现应设置合理的超时，避免无限阻塞。
    /// - 结果文本应简洁、信息密度高（LLM 会将其纳入上下文）。
    async fn execute(&self, args: Value) -> anyhow::Result<String>;
}

#[cfg(test)]
mod tests {
    //! Tool trait 单元测试。
    //!
    /// 验证 trait 可被实现、`Arc<dyn Tool>` 可共享调用。
    use super::*;
    use std::sync::Arc;

    /// 测试用 Echo 工具：回显输入文本。
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes the input text back."
        }
        fn parameters_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                },
                "required": ["text"]
            })
        }
        async fn execute(&self, args: Value) -> anyhow::Result<String> {
            Ok(args["text"].as_str().unwrap_or("").to_string())
        }
    }

    /// 验证 Tool trait 基本方法。
    #[test]
    fn test_tool_name_and_description() {
        let tool = EchoTool;
        assert_eq!(tool.name(), "echo");
        assert!(!tool.description().is_empty());
    }

    /// 验证 parameters_schema 返回合法 JSON Schema。
    #[test]
    fn test_parameters_schema() {
        let tool = EchoTool;
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"].get("text").is_some());
    }

    /// 验证 execute 异步执行并返回结果。
    #[tokio::test]
    async fn test_execute() {
        let tool = EchoTool;
        let result = tool
            .execute(serde_json::json!({"text": "hello"}))
            .await
            .unwrap();
        assert_eq!(result, "hello");
    }

    /// 验证 `Arc<dyn Tool>` 可跨任务共享（Send + Sync）。
    #[tokio::test]
    async fn test_arc_dyn_tool_shared() {
        let tool: Arc<dyn Tool> = Arc::new(EchoTool);
        // 在多个任务中共享调用
        let t1 = tool.clone();
        let t2 = tool.clone();
        let h1 = tokio::spawn(async move { t1.execute(serde_json::json!({"text": "a"})).await });
        let h2 = tokio::spawn(async move { t2.execute(serde_json::json!({"text": "b"})).await });
        assert_eq!(h1.await.unwrap().unwrap(), "a");
        assert_eq!(h2.await.unwrap().unwrap(), "b");
    }
}
