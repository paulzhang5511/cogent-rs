//! Cogent 核心 crate。
//!
//! 本 crate 是框架的地基，定义所有跨 crate 共享的 Trait 契约与核心运行时：
//! - [`types`]：消息与工具调用的基础数据类型。
//! - [`state`]：Agent 强类型状态机。
//! - [`event`]：事件总线（多订阅者 broadcast）。
//! - [`tool`]：工具 Trait。
//! - [`memory`]：记忆 Trait 与滑动窗口实现。
//! - [`provider`]：LLM Provider Trait。
//! - [`middleware`]：中间件 Trait。
//! - [`engine`]：ReAct 引擎。
//! - [`config`]：配置。
//! - [`error`]：错误类型。
//! - [`summary`]：迭代进度总结（max iterations 终止时输出）。
//!
//! 架构原则：本 crate 只依赖 trait，不依赖任何具体 provider/工具/中间件实现。
//! 具体实现全部在下游 crate（providers/tools/cli）通过 factory 注入。

pub mod config;
pub mod engine;
pub mod error;
pub mod event;
pub mod memory;
pub mod middleware;
pub mod provider;
pub mod state;
pub mod summary;
pub mod tool;
pub mod types;

// 公共 re-export：简化下游 crate 的导入路径
pub use config::Config;
pub use engine::AgentEngine;
pub use error::{CogentError, CogentResult};
pub use event::{AgentEvent, EventBus};
pub use memory::{CharBasedEstimator, Memory, TokenEstimator, WindowMemory};
pub use middleware::AgentMiddleware;
pub use provider::{LLMProvider, LLMResponse};
pub use state::AgentState;
pub use tool::Tool;
pub use types::{Message, Role, ToolCall};
