//! Agent 强类型状态机。
//!
//! 本模块定义 Agent 引擎运行过程中的所有状态及合法迁移路径。
//! 状态机是引擎的核心骨架：每次 LLM 调用、工具执行、错误处理都对应一次状态迁移。
//!
//! 状态迁移图：
//! ```text
//!   Idle ──► Thinking ──► ToolCalling ──► Observing ──► Thinking (循环)
//!                │                                    │
//!                ▼                                    ▼
//!           Completed ◄────────────────────────── Observing
//!                ▲
//!                │
//!              Failed (任意状态出错时)
//! ```
//!
//! 设计原则：
//! - 状态是强类型的（enum），非法迁移在编译期/运行期均可检测。
//! - 每次迁移通过 [`AgentState::transition`] 方法执行，非法迁移返回错误。
//! - 状态变更通过事件总线广播（见 [`crate::event::AgentEvent::StateChanged`]），
//!   实现引擎逻辑与展示/观测的解耦。

use crate::error::{CogentError, CogentResult};
use serde::{Deserialize, Serialize};

/// Agent 引擎运行状态。
///
/// 每个状态代表引擎在 ReAct 循环中的一个阶段：
/// - `Idle`：初始状态，等待用户输入。
/// - `Thinking`：正在调用 LLM 生成回复（可能包含工具调用）。
/// - `ToolCalling`：正在执行 LLM 请求的工具（可能多个并行）。
/// - `Observing`：工具执行完毕，将结果回传给 LLM 作为观察。
/// - `Completed`：LLM 生成了最终回复（无工具调用），任务完成。
/// - `Failed`：引擎因错误（死循环、最大迭代、Provider 错误等）终止。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentState {
    /// 初始状态：引擎已创建，等待用户输入。
    Idle,
    /// 思考状态：正在调用 LLM，等待 LLM 返回（文本回复或工具调用请求）。
    Thinking,
    /// 工具调用状态：LLM 请求了工具，引擎正在执行（可能多个工具并行）。
    ToolCalling,
    /// 观察状态：工具执行完毕，正在将结果组装为 `Role::Tool` 消息回传 LLM。
    Observing,
    /// 完成状态：LLM 生成了最终回复（无工具调用），本轮任务结束。
    Completed,
    /// 失败状态：引擎因错误终止（死循环、最大迭代超限、Provider 错误等）。
    Failed,
}

impl AgentState {
    /// 执行状态迁移，返回迁移后的新状态。
    ///
    /// 仅允许合法迁移，非法迁移返回 [`CogentError::Config`] 错误
    /// （使用 Config 变体是因为状态迁移错误属于配置/流程错误，非运行时 IO 错误）。
    ///
    /// # 合法迁移
    /// - `Idle → Thinking`：用户输入后开始思考。
    /// - `Thinking → ToolCalling`：LLM 返回了工具调用请求。
    /// - `Thinking → Completed`：LLM 返回了最终文本回复（无工具调用）。
    /// - `ToolCalling → Observing`：工具执行完毕。
    /// - `Observing → Thinking`：观察结果回传 LLM，进入下一轮思考。
    /// - 任意状态 → `Failed`：发生不可恢复错误。
    ///
    /// # 参数
    /// - `target`：目标状态。
    ///
    /// # 返回
    /// - 成功：迁移后的新状态（即 `target`）。
    /// - 失败：`CogentError`，描述非法迁移。
    ///
    /// # 示例
    /// ```
    /// use cogent_core::state::AgentState;
    /// let next = AgentState::Idle.transition(AgentState::Thinking).unwrap();
    /// assert_eq!(next, AgentState::Thinking);
    /// // 非法迁移：Idle 不能直接到 ToolCalling
    /// assert!(AgentState::Idle.transition(AgentState::ToolCalling).is_err());
    /// ```
    pub fn transition(self, target: AgentState) -> CogentResult<AgentState> {
        // 任意状态都可以迁移到 Failed（错误终止）
        if target == AgentState::Failed {
            return Ok(AgentState::Failed);
        }

        // 合法迁移表：(当前状态, 目标状态) → 是否合法
        let valid = matches!(
            (self, target),
            (AgentState::Idle, AgentState::Thinking)
                | (AgentState::Thinking, AgentState::ToolCalling)
                | (AgentState::Thinking, AgentState::Completed)
                | (AgentState::ToolCalling, AgentState::Observing)
                | (AgentState::Observing, AgentState::Thinking)
        );

        if valid {
            Ok(target)
        } else {
            Err(CogentError::Config(format!(
                "illegal state transition: {:?} -> {:?}",
                self, target
            )))
        }
    }

    /// 判断当前状态是否为终态（Completed 或 Failed）。
    ///
    /// 终态表示本轮 ReAct 循环已结束，引擎不再接受新的状态迁移
    /// （除非外部重置为 Idle）。
    pub fn is_terminal(self) -> bool {
        matches!(self, AgentState::Completed | AgentState::Failed)
    }

    /// 判断当前状态是否允许继续 ReAct 循环。
    ///
    /// `Thinking`、`ToolCalling`、`Observing` 均为循环中的活跃状态。
    pub fn is_active(self) -> bool {
        matches!(
            self,
            AgentState::Thinking | AgentState::ToolCalling | AgentState::Observing
        )
    }

    /// 轮次边界复位：仅当处于终态（`Completed`/`Failed`）时复位为 `Idle`。
    ///
    /// 与 [`transition`](Self::transition) 不同，本方法是**显式的轮次边界操作**，
    /// 允许终态 → `Idle`（`transition` 禁止该迁移，因为循环内不应回到 Idle）。
    /// 非终态调用为无操作（幂等），返回当前状态，避免在循环中途误复位。
    ///
    /// # 返回
    /// 复位后的状态（终态时为 `Idle`，非终态时为原状态）。
    pub fn reset(self) -> AgentState {
        if self.is_terminal() {
            AgentState::Idle
        } else {
            self
        }
    }
}

impl std::fmt::Display for AgentState {
    /// 状态的人类可读表示（用于日志和 CLI 渲染）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AgentState::Idle => "idle",
            AgentState::Thinking => "thinking",
            AgentState::ToolCalling => "tool_calling",
            AgentState::Observing => "observing",
            AgentState::Completed => "completed",
            AgentState::Failed => "failed",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    //! 状态机单元测试。
    //!
    //! 覆盖所有合法迁移、非法迁移、终态判断、活跃状态判断。

    use super::*;

    /// 验证所有合法状态迁移路径。
    #[test]
    fn test_legal_transitions() {
        // Idle → Thinking
        assert_eq!(
            AgentState::Idle.transition(AgentState::Thinking).unwrap(),
            AgentState::Thinking
        );
        // Thinking → ToolCalling
        assert_eq!(
            AgentState::Thinking
                .transition(AgentState::ToolCalling)
                .unwrap(),
            AgentState::ToolCalling
        );
        // Thinking → Completed
        assert_eq!(
            AgentState::Thinking
                .transition(AgentState::Completed)
                .unwrap(),
            AgentState::Completed
        );
        // ToolCalling → Observing
        assert_eq!(
            AgentState::ToolCalling
                .transition(AgentState::Observing)
                .unwrap(),
            AgentState::Observing
        );
        // Observing → Thinking（循环）
        assert_eq!(
            AgentState::Observing
                .transition(AgentState::Thinking)
                .unwrap(),
            AgentState::Thinking
        );
    }

    /// 验证任意状态可迁移到 Failed。
    #[test]
    fn test_any_state_to_failed() {
        for state in [
            AgentState::Idle,
            AgentState::Thinking,
            AgentState::ToolCalling,
            AgentState::Observing,
            AgentState::Completed,
        ] {
            assert_eq!(
                state.transition(AgentState::Failed).unwrap(),
                AgentState::Failed
            );
        }
    }

    /// 验证非法状态迁移返回错误。
    #[test]
    fn test_illegal_transitions() {
        // Idle 不能直接到 ToolCalling
        assert!(
            AgentState::Idle
                .transition(AgentState::ToolCalling)
                .is_err()
        );
        // Idle 不能直接到 Observing
        assert!(AgentState::Idle.transition(AgentState::Observing).is_err());
        // Thinking 不能直接到 Observing（必须先经过 ToolCalling）
        assert!(
            AgentState::Thinking
                .transition(AgentState::Observing)
                .is_err()
        );
        // ToolCalling 不能直接到 Thinking（必须先经过 Observing）
        assert!(
            AgentState::ToolCalling
                .transition(AgentState::Thinking)
                .is_err()
        );
        // Completed 是终态，不能迁移到 Thinking
        assert!(
            AgentState::Completed
                .transition(AgentState::Thinking)
                .is_err()
        );
        // Failed 是终态，不能迁移到 Thinking
        assert!(AgentState::Failed.transition(AgentState::Thinking).is_err());
    }

    /// 验证终态判断。
    #[test]
    fn test_is_terminal() {
        assert!(AgentState::Completed.is_terminal());
        assert!(AgentState::Failed.is_terminal());
        assert!(!AgentState::Idle.is_terminal());
        assert!(!AgentState::Thinking.is_terminal());
        assert!(!AgentState::ToolCalling.is_terminal());
        assert!(!AgentState::Observing.is_terminal());
    }

    /// 验证活跃状态判断。
    #[test]
    fn test_is_active() {
        assert!(AgentState::Thinking.is_active());
        assert!(AgentState::ToolCalling.is_active());
        assert!(AgentState::Observing.is_active());
        assert!(!AgentState::Idle.is_active());
        assert!(!AgentState::Completed.is_active());
        assert!(!AgentState::Failed.is_active());
    }

    /// 验证 Display 实现。
    #[test]
    fn test_display() {
        assert_eq!(AgentState::Idle.to_string(), "idle");
        assert_eq!(AgentState::Thinking.to_string(), "thinking");
        assert_eq!(AgentState::ToolCalling.to_string(), "tool_calling");
        assert_eq!(AgentState::Observing.to_string(), "observing");
        assert_eq!(AgentState::Completed.to_string(), "completed");
        assert_eq!(AgentState::Failed.to_string(), "failed");
    }

    /// 验证完整 ReAct 循环的状态迁移序列。
    #[test]
    fn test_full_react_cycle() {
        // 模拟完整 ReAct 循环：Idle → Thinking → ToolCalling → Observing → Thinking → Completed
        let mut state = AgentState::Idle;
        state = state.transition(AgentState::Thinking).unwrap();
        state = state.transition(AgentState::ToolCalling).unwrap();
        state = state.transition(AgentState::Observing).unwrap();
        state = state.transition(AgentState::Thinking).unwrap();
        state = state.transition(AgentState::Completed).unwrap();
        assert!(state.is_terminal());
    }

    /// 验证轮次边界复位：终态 → Idle，非终态幂等不变。
    #[test]
    fn test_reset() {
        // 终态复位为 Idle
        assert_eq!(AgentState::Completed.reset(), AgentState::Idle);
        assert_eq!(AgentState::Failed.reset(), AgentState::Idle);
        // 非终态幂等：保持原状态
        assert_eq!(AgentState::Idle.reset(), AgentState::Idle);
        assert_eq!(AgentState::Thinking.reset(), AgentState::Thinking);
        assert_eq!(AgentState::ToolCalling.reset(), AgentState::ToolCalling);
        assert_eq!(AgentState::Observing.reset(), AgentState::Observing);
    }
}
