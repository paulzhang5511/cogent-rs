//! 事件总线：多订阅者事件广播。
//!
//! 本模块定义 Agent 运行过程中产生的事件类型 [`AgentEvent`] 和
//! 基于 `tokio::sync::broadcast` 的多订阅者事件总线 [`EventBus`]。
//!
//! 事件总线是引擎与展示/观测层之间的解耦桥梁：
//! - 引擎在关键节点（状态迁移、工具执行、token 输出）发布事件。
//! - 任意数量的订阅者（CLI 渲染器、追踪器、日志器）独立消费事件。
//! - 发送失败（无订阅者）被静默忽略，不影响引擎主流程。
//!
//! 设计原则：
//! - 事件是 `Clone` 的（broadcast channel 要求），每个订阅者收到独立副本。
//! - 事件携带足够的上下文信息（工具名、耗时、token 数），订阅者无需额外查询。
//! - 事件类型实现 `Serialize`/`Deserialize`，便于持久化或远程传输。

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::state::AgentState;

/// Agent 运行过程中产生的事件。
///
/// 事件通过 [`EventBus`] 以多播（broadcast）方式分发给所有订阅者，
/// 实现"引擎逻辑"与"展示/观测"的解耦。
///
/// 事件类型：
/// - `StateChanged`：状态机迁移（如 Thinking → ToolCalling）。
/// - `TokenChunk`：流式输出中的一个 token 片段。
/// - `ToolStarted`：某个工具开始执行。
/// - `ToolFinished`：某个工具执行结束，附带耗时与成功标志。
/// - `TokenUsage`：一次 LLM 调用的 token 用量统计。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent {
    /// Agent 状态机发生迁移（如 Thinking → ToolCalling）。
    ///
    /// 订阅者可据此更新 UI 状态指示器或记录状态迁移日志。
    StateChanged(AgentState),

    /// 流式输出中的一个 token 片段。
    ///
    /// CLI 渲染器逐 chunk 拼接并 flush 到终端，实现打字机效果。
    TokenChunk(String),

    /// 某个工具开始执行。
    ///
    /// `id` 为工具调用 ID（关联 `ToolCall::id`），`name` 为工具名称，
    /// `args` 为工具参数（JSON 对象）。
    ToolStarted {
        /// 工具调用 ID。
        id: String,
        /// 工具名称。
        name: String,
        /// 工具参数（JSON 对象）。
        args: serde_json::Value,
    },

    /// 某个工具执行结束。
    ///
    /// `id` 为工具调用 ID，`duration_ms` 为执行耗时（毫秒），
    /// `success` 为是否执行成功。
    ToolFinished {
        /// 工具调用 ID。
        id: String,
        /// 执行耗时（毫秒）。
        duration_ms: u64,
        /// 是否执行成功。
        success: bool,
    },

    /// 一次 LLM 调用的 token 用量统计。
    ///
    /// `prompt_tokens` 为输入 token 数，`completion_tokens` 为输出 token 数。
    /// 订阅者可据此累计总用量或显示成本估算。
    TokenUsage {
        /// 输入（prompt）token 数。
        prompt_tokens: u32,
        /// 输出（completion）token 数。
        completion_tokens: u32,
    },
}

/// 基于 `tokio::sync::broadcast` 的多订阅者事件总线。
///
/// 引擎在关键节点调用 [`EventBus::publish`] 发布事件；
/// 任意数量的订阅者（CLI 渲染器、日志器）可独立调用 [`EventBus::subscribe`]
/// 获取事件流。
///
/// 线程安全：`EventBus` 内部使用 `broadcast::Sender`，是 `Send + Sync` 的，
/// 可安全地在多个任务间共享（通常通过 `Arc<EventBus>`）。
///
/// # 示例
/// ```
/// use cogent_core::event::{AgentEvent, EventBus};
/// use cogent_core::state::AgentState;
///
/// let bus = EventBus::new(64);
/// let mut rx = bus.subscribe();
///
/// // 发布事件
/// bus.publish(AgentEvent::StateChanged(AgentState::Thinking));
///
/// // 接收事件（用同步的 try_recv，doctest 无异步运行时）
/// let event = rx.try_recv().unwrap();
/// assert!(matches!(event, AgentEvent::StateChanged(AgentState::Thinking)));
/// ```
pub struct EventBus {
    /// broadcast channel 的发送端。
    ///
    /// 持有 Sender 即可发布事件；Receiver 通过 `subscribe()` 创建。
    sender: broadcast::Sender<AgentEvent>,
}

impl EventBus {
    /// 创建容量为 `capacity` 的事件总线。
    ///
    /// `capacity` 为 broadcast channel 的缓冲区大小。当缓冲区满时，
    /// 最慢的订阅者会收到 `RecvError::Lagged`（跳过的事件数），
    /// 不会阻塞发布者。
    ///
    /// # 参数
    /// - `capacity`：缓冲区容量（建议 256，见 `Config::event_bus_capacity`）。
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        tracing::debug!(capacity, "event bus initialized");
        Self { sender }
    }

    /// 发布一个事件给所有当前订阅者。
    ///
    /// 无订阅者时静默丢弃（不返回错误），避免影响引擎主流程。
    /// 这是有意的设计：事件是"尽力而为"的通知，不是关键路径。
    ///
    /// # 参数
    /// - `event`：要发布的事件。
    pub fn publish(&self, event: AgentEvent) {
        // 发送失败（无订阅者）被静默忽略
        let _ = self.sender.send(event);
    }

    /// 订阅事件流，返回一个独立的接收端。
    ///
    /// 每个 `subscribe()` 调用返回独立的 `Receiver`，
    /// 各订阅者独立消费事件，互不影响。
    ///
    /// # 返回
    /// - `broadcast::Receiver<AgentEvent>`：事件接收端。
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.sender.subscribe()
    }
}

impl std::fmt::Debug for EventBus {
    /// 自定义 Debug 实现（broadcast::Sender 不实现 Debug）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBus")
            .field("sender", &format_args!("<broadcast::Sender>"))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    //! 事件总线单元测试。
    //!
    //! 验证事件发布/接收、多订阅者独立性、无订阅者时不 panic。

    use super::*;

    /// 验证基本的事件发布与接收。
    #[tokio::test]
    async fn test_publish_and_receive() {
        let bus = EventBus::new(64);
        let mut rx = bus.subscribe();

        bus.publish(AgentEvent::StateChanged(AgentState::Thinking));

        let event = rx.recv().await.unwrap();
        assert!(matches!(
            event,
            AgentEvent::StateChanged(AgentState::Thinking)
        ));
    }

    /// 验证多订阅者各自独立接收事件。
    #[tokio::test]
    async fn test_multiple_subscribers() {
        let bus = EventBus::new(64);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        bus.publish(AgentEvent::TokenChunk("hello".into()));

        // 两个订阅者都应收到事件
        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert!(matches!(e1, AgentEvent::TokenChunk(_)));
        assert!(matches!(e2, AgentEvent::TokenChunk(_)));
    }

    /// 验证无订阅者时 publish 不 panic（静默丢弃）。
    #[test]
    fn test_publish_no_subscriber() {
        let bus = EventBus::new(64);
        // 不创建任何订阅者，直接发布
        bus.publish(AgentEvent::StateChanged(AgentState::Idle));
        // 不应 panic
    }

    /// 验证 ToolStarted 事件携带完整字段。
    #[tokio::test]
    async fn test_tool_started_event() {
        let bus = EventBus::new(64);
        let mut rx = bus.subscribe();

        bus.publish(AgentEvent::ToolStarted {
            id: "call_1".into(),
            name: "bash".into(),
            args: serde_json::json!({"command": "ls"}),
        });

        let event = rx.recv().await.unwrap();
        match event {
            AgentEvent::ToolStarted { id, name, args } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "bash");
                assert_eq!(args["command"], "ls");
            }
            _ => panic!("expected ToolStarted"),
        }
    }

    /// 验证 ToolFinished 事件携带耗时与成功标志。
    #[tokio::test]
    async fn test_tool_finished_event() {
        let bus = EventBus::new(64);
        let mut rx = bus.subscribe();

        bus.publish(AgentEvent::ToolFinished {
            id: "call_1".into(),
            duration_ms: 150,
            success: true,
        });

        let event = rx.recv().await.unwrap();
        match event {
            AgentEvent::ToolFinished {
                duration_ms,
                success,
                ..
            } => {
                assert_eq!(duration_ms, 150);
                assert!(success);
            }
            _ => panic!("expected ToolFinished"),
        }
    }

    /// 验证 TokenUsage 事件携带 token 统计。
    #[tokio::test]
    async fn test_token_usage_event() {
        let bus = EventBus::new(64);
        let mut rx = bus.subscribe();

        bus.publish(AgentEvent::TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
        });

        let event = rx.recv().await.unwrap();
        match event {
            AgentEvent::TokenUsage {
                prompt_tokens,
                completion_tokens,
            } => {
                assert_eq!(prompt_tokens, 100);
                assert_eq!(completion_tokens, 50);
            }
            _ => panic!("expected TokenUsage"),
        }
    }

    /// 验证事件可序列化/反序列化（往返一致性）。
    #[test]
    fn test_event_serialization_roundtrip() {
        let event = AgentEvent::ToolStarted {
            id: "call_1".into(),
            name: "bash".into(),
            args: serde_json::json!({"command": "ls"}),
        };
        let json = serde_json::to_string(&event).unwrap();
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        // 验证反序列化后的事件类型正确
        assert!(matches!(restored, AgentEvent::ToolStarted { .. }));
    }

    /// 验证 EventBus 是 Send + Sync（可跨任务共享）。
    #[test]
    fn test_event_bus_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EventBus>();
    }

    /// 验证 EventBus 自定义 Debug 实现（不 panic 且包含结构名）。
    #[test]
    fn test_event_bus_debug_impl() {
        let bus = EventBus::new(16);
        let dbg = format!("{bus:?}");
        assert!(dbg.contains("EventBus"));
        assert!(dbg.contains("broadcast::Sender"));
    }
}
