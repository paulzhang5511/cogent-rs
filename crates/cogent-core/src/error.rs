//! Cogent 错误类型定义。
//!
//! 本模块定义框架核心层的所有错误类型，使用 `thiserror` 派生 `std::error::Error`。
//! 各 crate 在库边界使用 `CogentError`，跨 crate 边界可转换为 `anyhow::Error`。
//!
//! 设计原则：
//! - 错误类型是强类型的（enum 变体），而非裸 `String`，便于调用方按变体分支处理。
//! - 每个变体携带足够的上下文信息，使日志与错误报告无需额外查询即可定位问题。
//! - 错误消息用英文（与日志规范一致），便于跨团队检索。

use thiserror::Error;

/// Cogent 框架核心错误类型。
///
/// 覆盖引擎运行、工具执行、配置加载、记忆管理、Provider 调用等所有核心路径的错误。
/// 调用方可通过 `match` 按变体分支处理，或转换为 `anyhow::Error` 向上传播。
#[derive(Debug, Error)]
pub enum CogentError {
    /// LLM Provider 调用失败（网络错误、API 返回错误、超时等）。
    ///
    /// 持有底层 [`anyhow::Error`]，通过 `#[source]` 保留完整错误链
    /// （如 reqwest 的 HTTP 错误 → 底层 IO 错误），调用方可用
    /// `std::error::Error::source()` 逐层追溯根因。
    #[error("provider error: {0}")]
    Provider(#[source] anyhow::Error),

    /// 工具执行失败（命令执行错误、文件 IO 错误、HTTP 请求失败等）。
    ///
    /// `tool_name` 为工具名称，`detail` 为具体错误描述。
    #[error("tool '{tool_name}' failed: {detail}")]
    Tool {
        /// 工具名称（如 "bash"、"file_read"）。
        tool_name: String,
        /// 具体错误描述。
        detail: String,
    },

    /// 配置加载或校验失败（缺少环境变量、值格式错误等）。
    ///
    /// `0` 为具体错误描述。
    #[error("config error: {0}")]
    Config(String),

    /// 检测到死循环：同一工具以相同参数被重复调用。
    ///
    /// 引擎在检测到连续两次相同 `name + arguments` 的工具调用时触发此错误，
    /// 防止无限消耗 Token。
    ///
    /// `tool_name` 为被重复调用的工具名，`call_count` 为累计调用次数。
    #[error("loop detected: tool '{tool_name}' called {call_count} times with identical arguments")]
    LoopDetected {
        /// 被重复调用的工具名称。
        tool_name: String,
        /// 该工具以相同参数被调用的累计次数。
        call_count: usize,
    },

    /// 记忆操作失败（窗口裁剪异常、token 估算溢出等）。
    ///
    /// `0` 为具体错误描述。
    #[error("memory error: {0}")]
    Memory(String),

    /// 达到最大迭代次数限制，引擎强制终止。
    ///
    /// `0` 为配置的最大迭代次数。
    #[error("max iterations ({0}) exceeded, engine terminated")]
    MaxIterationsExceeded(usize),

    /// LLM 返回空响应（无内容且无工具调用）。
    ///
    /// 模型在上下文过长或陷入困境时可能返回空 `content` 且无 `tool_calls`，
    /// 此时引擎无法继续也无法给出答案。显式报错而非静默返回空串，
    /// 让调用方（CLI）向用户呈现明确错误。
    ///
    /// `0` 为发生空响应的迭代次数。
    #[error("empty LLM response at iteration {0} (no content, no tool calls)")]
    EmptyResponse(usize),

    /// 中间件拦截（安全策略、token 限制等）。
    ///
    /// `middleware` 为中间件名称，`reason` 为拦截原因。
    #[error("middleware '{middleware}' blocked: {reason}")]
    MiddlewareBlocked {
        /// 拦截的中间件名称。
        middleware: String,
        /// 拦截原因。
        reason: String,
    },
}

/// 核心层 Result 类型别名，简化函数签名。
pub type CogentResult<T> = Result<T, CogentError>;

#[cfg(test)]
mod tests {
    //! `CogentError` 单元测试。
    //!
    //! 验证各错误变体的 Display 输出格式与 `std::error::Error` trait 实现。

    use super::*;

    /// 验证 Provider 错误的 Display 输出。
    #[test]
    fn test_provider_error_display() {
        let err = CogentError::Provider(anyhow::anyhow!("HTTP 500 Internal Server Error"));
        assert_eq!(
            err.to_string(),
            "provider error: HTTP 500 Internal Server Error"
        );
    }

    /// 验证 Provider 错误保留底层错误链（`source()` 可追溯到根因）。
    #[test]
    fn test_provider_error_source_chain() {
        let root = std::io::Error::new(std::io::ErrorKind::TimedOut, "connection timed out");
        let err = CogentError::Provider(anyhow::anyhow!("request failed: {root}"));
        // source() 应返回底层 anyhow 包装的 io 错误
        let source = std::error::Error::source(&err).expect("应保留 source 链");
        assert!(source.to_string().contains("connection timed out"));
    }

    /// 验证 Tool 错误的 Display 输出（含工具名与详情）。
    #[test]
    fn test_tool_error_display() {
        let err = CogentError::Tool {
            tool_name: "bash".into(),
            detail: "command not found".into(),
        };
        assert_eq!(err.to_string(), "tool 'bash' failed: command not found");
    }

    /// 验证 LoopDetected 错误的 Display 输出。
    #[test]
    fn test_loop_detected_display() {
        let err = CogentError::LoopDetected {
            tool_name: "bash".into(),
            call_count: 2,
        };
        assert_eq!(
            err.to_string(),
            "loop detected: tool 'bash' called 2 times with identical arguments"
        );
    }

    /// 验证 MaxIterationsExceeded 错误的 Display 输出。
    #[test]
    fn test_max_iterations_display() {
        let err = CogentError::MaxIterationsExceeded(10);
        assert_eq!(
            err.to_string(),
            "max iterations (10) exceeded, engine terminated"
        );
    }

    /// 验证 MiddlewareBlocked 错误的 Display 输出。
    #[test]
    fn test_middleware_blocked_display() {
        let err = CogentError::MiddlewareBlocked {
            middleware: "SafetyGuard".into(),
            reason: "blocked command: rm -rf /".into(),
        };
        assert_eq!(
            err.to_string(),
            "middleware 'SafetyGuard' blocked: blocked command: rm -rf /"
        );
    }

    /// 验证 CogentError 实现了 std::error::Error（可通过 anyhow 转换）。
    #[test]
    fn test_error_trait_impl() {
        let err: Box<dyn std::error::Error> =
            Box::new(CogentError::Config("missing OPENAI_API_KEY".into()));
        assert!(err.to_string().contains("OPENAI_API_KEY"));
    }

    /// 验证 CogentError 可转换为 anyhow::Error。
    #[test]
    fn test_anyhow_conversion() {
        let err = CogentError::Provider(anyhow::anyhow!("timeout"));
        let anyhow_err: anyhow::Error = err.into();
        assert!(anyhow_err.to_string().contains("timeout"));
    }
}
