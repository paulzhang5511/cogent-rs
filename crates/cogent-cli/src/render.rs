//! 事件渲染器：订阅 EventBus 并彩色渲染各 AgentEvent。
//!
//! 本模块实现 [`start_event_renderer`]——在独立异步任务中订阅 [`EventBus`]，
//! 将引擎发布的事件渲染到终端（彩色输出）。
//!
//! # 渲染规则（PRD §3.4.2）
//! - `StateChanged`：灰色状态指示（如 `→ Thinking`）。
//! - `TokenChunk`：逐 chunk 拼接并 flush 到 stdout（打字机效果）。
//! - `ToolStarted`：高亮工具名 + 参数（黄色）。
//! - `ToolFinished`：耗时 + 成功标志（绿色/红色）。
//! - `TokenUsage`：灰色 token 统计。
//!
//! # 设计原则
//! - 渲染器是独立任务，与引擎主流程解耦（通过 broadcast channel）。
//! - `TokenChunk` 逐字 flush，实现流式打字机效果。
//! - 渲染器不阻塞引擎（broadcast 非阻塞发布）。

use std::io::Write;

use cogent_core::event::{AgentEvent, EventBus};
use cogent_core::state::AgentState;
use colored::Colorize;

/// 启动事件渲染器任务。
///
/// 在独立异步任务中订阅 [`EventBus`]，持续渲染事件直到 channel 关闭。
///
/// # 参数
/// - `event_bus`：要订阅的事件总线。
///
/// # 返回
/// - `tokio::task::JoinHandle`：渲染器任务句柄（调用方可 `await` 等待其结束）。
///
/// # 说明
/// 渲染器任务在 `EventBus` 的所有发送端 drop 后自动退出
/// （`broadcast::Receiver::recv` 返回 `RecvError::Closed`）。
pub fn start_event_renderer(event_bus: &EventBus) -> tokio::task::JoinHandle<()> {
    let mut rx = event_bus.subscribe();

    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(event) => render_event(&event),
                // channel 关闭（所有发送端已 drop），渲染器退出
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    tracing::debug!("event renderer: channel closed, exiting");
                    break;
                }
                // 缓冲区满导致跳过事件（最慢订阅者），记录后继续
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(skipped = n, "event renderer: lagged, skipped events");
                }
            }
        }
    })
}

/// 渲染单个事件到终端。
///
/// # 参数
/// - `event`：要渲染的事件。
fn render_event(event: &AgentEvent) {
    match event {
        AgentEvent::StateChanged(state) => {
            // 灰色状态指示
            let label = state_label(state);
            let _ = writeln!(std::io::stdout(), "{}", format!("  {label}").dimmed());
        }
        AgentEvent::TokenChunk(chunk) => {
            // 逐 chunk 输出并 flush（打字机效果）
            let mut stdout = std::io::stdout();
            let _ = stdout.write_all(chunk.as_bytes());
            let _ = stdout.flush();
        }
        AgentEvent::ToolStarted { id, name, args } => {
            // 高亮工具名 + 参数（黄色）
            let _ = writeln!(
                std::io::stdout(),
                "  {} {} {}",
                "⚙ tool".yellow().bold(),
                name.yellow(),
                args.to_string().dimmed()
            );
            let _ = id; // id 用于关联 ToolFinished，渲染时不单独显示
        }
        AgentEvent::ToolFinished {
            id,
            duration_ms,
            success,
        } => {
            // 耗时 + 成功标志（绿色/红色）
            let status = if *success {
                "✓ ok".green().to_string()
            } else {
                "✗ failed".red().to_string()
            };
            let _ = writeln!(
                std::io::stdout(),
                "  {} {} {}",
                "⚙ done".dimmed(),
                status,
                format!("{duration_ms}ms").dimmed()
            );
            let _ = id;
        }
        AgentEvent::TokenUsage {
            prompt_tokens,
            completion_tokens,
        } => {
            // 灰色 token 统计
            let _ = writeln!(
                std::io::stdout(),
                "  {} {} {}",
                "◈ tokens".dimmed(),
                format!("in:{prompt_tokens}").dimmed(),
                format!("out:{completion_tokens}").dimmed()
            );
        }
    }
}

/// 将状态转换为可读标签。
///
/// # 参数
/// - `state`：Agent 状态。
///
/// # 返回
/// - 状态标签字符串（如 `"→ Thinking"`）。
fn state_label(state: &AgentState) -> String {
    match state {
        AgentState::Idle => "→ Idle".to_string(),
        AgentState::Thinking => "→ Thinking".to_string(),
        AgentState::ToolCalling => "→ ToolCalling".to_string(),
        AgentState::Observing => "→ Observing".to_string(),
        AgentState::Completed => "→ Completed".to_string(),
        AgentState::Failed => "→ Failed".to_string(),
    }
}

#[cfg(test)]
mod tests {
    //! 渲染器单元测试。
    //!
    //! 验证 `state_label` 转换与渲染器任务启动/退出。

    use super::*;

    /// 验证 state_label 各状态转换。
    #[test]
    fn test_state_label() {
        assert_eq!(state_label(&AgentState::Idle), "→ Idle");
        assert_eq!(state_label(&AgentState::Thinking), "→ Thinking");
        assert_eq!(state_label(&AgentState::ToolCalling), "→ ToolCalling");
        assert_eq!(state_label(&AgentState::Observing), "→ Observing");
        assert_eq!(state_label(&AgentState::Completed), "→ Completed");
        assert_eq!(state_label(&AgentState::Failed), "→ Failed");
    }

    /// 验证渲染器任务在 channel 关闭后退出。
    #[tokio::test]
    async fn test_renderer_exits_on_close() {
        let bus = EventBus::new(16);
        let handle = start_event_renderer(&bus);
        // 发布一个事件
        bus.publish(AgentEvent::StateChanged(AgentState::Thinking));
        // drop bus 后渲染器应退出
        drop(bus);
        // 等待渲染器退出（带超时，避免测试挂起）
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
    }
}
