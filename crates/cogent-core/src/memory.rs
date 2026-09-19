//! 记忆 Trait 与滑动窗口实现。
//!
//! 本模块定义 [`Memory`] trait（Agent 记忆的统一契约）和
//! [`WindowMemory`]（基于 token 预算的滑动窗口实现）。
//!
//! 记忆是 Agent 的"短期工作记忆"：保存当前会话的对话历史，
//! 在每轮 LLM 调用前提供上下文窗口。
//!
//! 设计原则：
//! - 记忆是 `Send + Sync` 的，可安全地在引擎与 CLI 间共享。
//! - `WindowMemory` 始终保留系统种子消息（`system`），仅裁剪会话历史。
//! - Token 估算通过 [`TokenEstimator`] trait 抽象，v1 用零依赖的字符近似。
//! - 窗口裁剪从最旧消息开始丢弃，保留最近的对话上下文。

use async_trait::async_trait;
use std::sync::RwLock;

use crate::error::CogentResult;
use crate::types::{Message, Role};

/// 单条超大消息被纳入窗口时保留的最小片段（token）。
///
/// 低于此值则不值得保留（截断标记本身都放不下），直接丢弃该消息。
const MIN_MESSAGE_SNIPPET_TOKENS: usize = 32;

/// Token 估算器 trait。
///
/// 将文本转换为近似 token 数。v1 使用零依赖的字符近似（`chars/4`），
/// 后续可替换为 `tiktoken` 精确计数（实现同一 trait，引擎无需改动）。
///
/// # 实现要求
/// - `estimate` 应为纯函数（无副作用、无 IO），可高频调用。
/// - 估算值应接近真实 token 数（允许 ±20% 误差）。
pub trait TokenEstimator: Send + Sync {
    /// 估算文本的 token 数。
    ///
    /// # 参数
    /// - `text`：待估算的文本。
    ///
    /// # 返回
    /// - 近似 token 数（`usize`）。
    fn estimate(&self, text: &str) -> usize;

    /// 将 token 预算反向换算为字符数（用于把超长 content 截断到预算内）。
    ///
    /// 默认实现为 `tokens * 4`，与 [`CharBasedEstimator`] 的 `chars/4` 口径对称。
    /// 若替换为 tiktoken 等非均匀估算器，**必须覆盖此方法**，否则截断长度会按
    /// 错误的"4 字符/token"系数计算。
    fn token_to_chars(&self, tokens: usize) -> usize {
        tokens.saturating_mul(4)
    }
}

/// 基于字符数的零依赖 Token 估算器（v1 默认实现）。
///
/// 使用 `text.chars().count() / 4` 近似估算 token 数。
/// 对英文文本，1 token ≈ 4 字符，此近似误差在 ±20% 以内。
/// 对中文文本，1 token ≈ 1-2 字符，此近似会高估（偏保守，安全方向）。
///
/// 零依赖、零拷贝（对 `&str` 操作）、O(n) 时间复杂度。
#[derive(Debug, Clone, Copy, Default)]
pub struct CharBasedEstimator;

impl TokenEstimator for CharBasedEstimator {
    /// 用 `chars().count() / 4` 近似估算 token 数。
    fn estimate(&self, text: &str) -> usize {
        text.chars().count() / 4
    }
}

/// Agent 记忆的统一契约。
///
/// 记忆保存对话历史，在每轮 LLM 调用前提供上下文窗口。
/// 引擎通过 `Arc<dyn Memory>` 持有记忆，无需感知具体实现。
///
/// # 实现要求
/// - 所有方法应为异步（`async_trait`），支持 IO 密集型记忆（如向量数据库）。
/// - `get_context` 返回的消息列表应已按 token 预算裁剪，可直接传给 LLM。
/// - 记忆是 `Send + Sync` 的，可安全地在多个任务间共享。
#[async_trait]
pub trait Memory: Send + Sync {
    /// 设置/替换系统种子消息。
    ///
    /// 系统消息定义 Agent 的人设、行为约束、全局指令。
    /// 在 `get_context` 中始终前置（不受窗口裁剪影响）。
    ///
    /// # 参数
    /// - `message`：系统消息（`Role::System`）。
    async fn set_system(&self, message: Message);

    /// 追加一条会话消息到记忆。
    ///
    /// # 参数
    /// - `message`：要追加的消息（`Role::User`/`Role::Assistant`/`Role::Tool`）。
    async fn append(&self, message: Message);

    /// 获取当前上下文窗口（已按 token 预算裁剪）。
    ///
    /// 返回的消息列表包含：
    /// 1. 系统种子消息（始终前置，若已设置）。
    /// 2. 会话历史（从最近到最旧，超出 token 预算的最旧消息被丢弃）。
    ///
    /// # 参数
    /// - `max_tokens`：上下文窗口的 token 预算上限。
    ///
    /// # 返回
    /// - 裁剪后的消息列表（可直接传给 LLM 的 `messages` 字段）。
    async fn get_context(&self, max_tokens: usize) -> CogentResult<Vec<Message>>;

    /// 清空会话历史（保留系统种子消息）。
    ///
    /// 用于 `/clear` 命令或会话重置。系统种子消息不受影响。
    async fn clear(&self);
}

/// 估算单条消息中 `tool_calls` 的 token 开销（仅工具调用结构，不含 content）。
///
/// 抽为公共函数，供 [`estimate_message_tokens`] 与 `WindowMemory::truncate_to_budget`
/// 共用同一口径（后者需单独扣除 tool_calls 开销后再截断 content）。
fn estimate_tool_call_tokens(estimator: &dyn TokenEstimator, msg: &Message) -> usize {
    msg.tool_calls
        .as_ref()
        .map(|calls| {
            let json = serde_json::to_string(calls).unwrap_or_default();
            estimator.estimate(&json)
        })
        .unwrap_or(0)
}

/// 估算单条消息的 token 数（content + tool_calls 的 JSON 序列化开销）。
///
/// 抽为模块级公共函数，供 `WindowMemory` 窗口裁剪与 `TokenLimiterMiddleware`
/// 兜底校验共用同一口径，避免两处独立实现随演进漂移（如换 tiktoken）。
///
/// # 参数
/// - `estimator`：token 估算器。
/// - `msg`：待估算的消息。
pub fn estimate_message_tokens(estimator: &dyn TokenEstimator, msg: &Message) -> usize {
    let content_tokens = estimator.estimate(&msg.content);
    content_tokens + estimate_tool_call_tokens(estimator, msg)
}

/// 基于 token 预算的滑动窗口记忆实现。
///
/// 内部使用 `RwLock<Vec<Message>>` 保存会话历史，
/// `Option<Message>` 保存系统种子消息。
///
/// 窗口裁剪策略：
/// 1. 始终保留系统种子消息（计入总预算）。
/// 2. 始终保留原始任务（第一条 `User` 消息），为其预留至多 1/3 预算，
///    超长则截断保留头部——防止单条超大工具结果把任务挤出窗口导致模型失忆。
/// 3. 其余历史从尾部（最近）向前累加 token，超出预算时丢弃更旧消息。
/// 4. 单条消息超出剩余预算时截断为带标记片段后纳入，而非整体丢弃。
/// 5. 返回 `[system?, task?, ...recent_messages]`。
///
/// 线程安全：`RwLock` 允许多读单写，适合"多读（get_context）少写（append）"的场景。
pub struct WindowMemory {
    /// 会话历史（按时间顺序，最旧在前）。
    history: RwLock<Vec<Message>>,
    /// 系统种子消息（始终前置，不受窗口裁剪影响）。
    system: RwLock<Option<Message>>,
    /// 上下文窗口的 token 预算上限。
    max_tokens: usize,
    /// Token 估算器（v1 用 `CharBasedEstimator`）。
    estimator: Box<dyn TokenEstimator>,
}

impl WindowMemory {
    /// 创建滑动窗口记忆。
    ///
    /// # 参数
    /// - `max_tokens`：上下文窗口的 token 预算上限。
    /// - `estimator`：Token 估算器（通常用 `Box::new(CharBasedEstimator)`）。
    pub fn new(max_tokens: usize, estimator: Box<dyn TokenEstimator>) -> Self {
        tracing::debug!(max_tokens, "window memory initialized");
        Self {
            history: RwLock::new(Vec::new()),
            system: RwLock::new(None),
            max_tokens,
            estimator,
        }
    }

    /// 创建使用默认 `CharBasedEstimator` 的滑动窗口记忆。
    ///
    /// # 参数
    /// - `max_tokens`：上下文窗口的 token 预算上限。
    pub fn with_default_estimator(max_tokens: usize) -> Self {
        Self::new(max_tokens, Box::new(CharBasedEstimator))
    }

    /// 估算单条消息的 token 数。
    ///
    /// 消息的 token 数 = content 的 token 数 + tool_calls 的 JSON 序列化 token 数。
    /// 委托给模块级 [`estimate_message_tokens`]，与 `TokenLimiterMiddleware` 同口径。
    fn estimate_message_tokens(&self, msg: &Message) -> usize {
        estimate_message_tokens(&*self.estimator, msg)
    }

    /// 将单条消息的 `content` 截断到给定 token 预算内（保留 tool_calls / id）。
    ///
    /// 用于单条消息超出窗口剩余预算的场景（如大文件读取结果）：与其整体丢弃
    /// 导致窗口塌缩，不如保留一个带截断标记的片段。
    ///
    /// token 估算口径为 `chars()/4`（见 [`CharBasedEstimator`]），故反向换算
    /// 字符上限走 [`TokenEstimator::token_to_chars`]（默认 `*4`，换估算器时自动跟随），
    /// 并预留截断标记的余量。
    ///
    /// # 参数
    /// - `msg`：原始消息。
    /// - `budget_tokens`：该消息允许占用的 token 上限。
    ///
    /// # 返回
    /// - `Some`：截断后的消息（未超限时原样克隆返回）。
    /// - `None`：预算连 tool_calls 结构都放不下（截断后 content 将为空，
    ///   向模型发送"空观察"无意义），调用方应丢弃该消息。
    fn truncate_to_budget(&self, msg: &Message, budget_tokens: usize) -> Option<Message> {
        // tool_calls 结构占用的 token 不参与 content 截断，先扣除其开销。
        let tool_call_tokens = estimate_tool_call_tokens(&*self.estimator, msg);
        let content_budget = budget_tokens.saturating_sub(tool_call_tokens);
        if content_budget == 0 {
            return None;
        }

        let content_tokens = self.estimator.estimate(&msg.content);
        if content_tokens <= content_budget {
            return Some(msg.clone());
        }

        // 截断标记（简洁，约 4 token）；仅在保留片段之外仍有余量时附加。
        let marker = "\n…[truncated]";
        let marker_tokens = self.estimator.estimate(marker).max(1);
        let keep_tokens = content_budget.saturating_sub(marker_tokens);
        let total_chars = msg.content.chars().count();

        // 预算连标记都放不下时退化为"尽力取头部片段"，不附加标记，避免只剩标记。
        let (take_chars, with_marker) = if keep_tokens > 0 {
            (self.estimator.token_to_chars(keep_tokens), true)
        } else {
            (self.estimator.token_to_chars(content_budget), false)
        };
        if take_chars == 0 {
            // 预算连一个字符都放不下，截断结果将是空 content，无意义。
            return None;
        }

        let taken: String = msg.content.chars().take(take_chars).collect();
        if taken.chars().count() >= total_chars {
            return Some(msg.clone());
        }

        let head = taken.trim_end();
        let mut out = msg.clone();
        out.content = if with_marker {
            format!("{head}{marker}")
        } else {
            head.to_string()
        };
        Some(out)
    }
}

#[async_trait]
impl Memory for WindowMemory {
    /// 设置/替换系统种子消息。
    async fn set_system(&self, message: Message) {
        let mut system = self.system.write().unwrap_or_else(|e| e.into_inner());
        *system = Some(message);
        tracing::debug!("system message set");
    }

    /// 追加一条会话消息到记忆。
    async fn append(&self, message: Message) {
        // 在 push（move）之前先取出 role，避免 move-after-borrow
        let role = message.role;
        let mut history = self.history.write().unwrap_or_else(|e| e.into_inner());
        history.push(message);
        tracing::trace!(
            role = ?role,
            history_len = history.len(),
            "message appended to memory"
        );
    }

    /// 获取当前上下文窗口（已按 token 预算裁剪）。
    ///
    /// 裁剪策略：
    /// 1. 系统种子消息始终前置（不计入预算）。
    /// 2. 原始任务（第一条 `User` 消息）始终保留——为其预留至多 1/3 预算，
    ///    防止单条超大工具结果把任务挤出窗口导致模型"失忆"重复调用。
    ///    设计取舍：钉选的是**第一条** User 消息（会话的全局目标），而非最近一条；
    ///    后续轮次的用户问题按普通历史参与裁剪。
    /// 3. 其余历史从尾部（最近）向前累加 token，超出预算时丢弃更旧消息。
    /// 4. 单条消息超出剩余预算时，**截断后纳入**（而非整体丢弃并中断遍历）：
    ///    大文件读取 / 大段命令输出至少保留一个带截断标记的片段，保证最新观察
    ///    不会丢失（旧实现遇超限即 `break`，会使窗口塌缩为仅系统消息）。
    ///    若预算连该消息的 tool_calls 结构都放不下，则丢弃该消息（避免空观察）。
    async fn get_context(&self, max_tokens: usize) -> CogentResult<Vec<Message>> {
        let history = self.history.read().unwrap_or_else(|e| e.into_inner());
        let system = self.system.read().unwrap_or_else(|e| e.into_inner());

        let mut result: Vec<Message> = Vec::new();

        // 1. 系统种子消息始终前置（计入总预算，保证 TokenLimiter 校验
        //    system + history 之和不超过 max_tokens）。
        let mut budget = max_tokens;
        if let Some(ref sys) = *system {
            budget = budget.saturating_sub(self.estimate_message_tokens(sys));
            result.push(sys.clone());
        }

        // 2. 定位原始任务（第一条 User 消息），为其预留预算并单独保留。
        let task_idx = history.iter().position(|m| m.role == Role::User);
        let mut task_msg: Option<Message> = None;
        if let Some(idx) = task_idx {
            let task = &history[idx];
            let task_tokens = self.estimate_message_tokens(task).max(1);
            // 小任务按实际大小预留；大任务至多占用 1/3 预算（截断保留头部）。
            let reserve = task_tokens.min((budget / 3).max(1));
            budget = budget.saturating_sub(reserve);
            task_msg = self.truncate_to_budget(task, reserve);
        }

        // 3. 从会话历史尾部（最近）向前累加 token。
        let mut remaining = budget;
        let mut selected: Vec<Message> = Vec::new();
        for (i, msg) in history.iter().enumerate().rev() {
            if Some(i) == task_idx {
                // 原始任务已单独保留，跳过避免重复。
                continue;
            }
            let msg_tokens = self.estimate_message_tokens(msg);
            if msg_tokens <= remaining {
                selected.push(msg.clone());
                remaining -= msg_tokens;
            } else if remaining >= MIN_MESSAGE_SNIPPET_TOKENS {
                // 单条超大消息：截断到剩余预算后纳入，随后无空间容纳更旧消息。
                // 预算连 tool_calls 结构都放不下时丢弃该消息（避免空观察）。
                if let Some(truncated) = self.truncate_to_budget(msg, remaining) {
                    selected.push(truncated);
                }
                remaining = 0;
                break;
            } else {
                // 剩余预算连最小片段都放不下，更旧消息同样放不下。
                break;
            }
        }

        // 4. 恢复时间顺序：任务（最旧的用户消息）→ 近期窗口。
        selected.reverse();
        if let Some(task) = task_msg {
            result.push(task);
        }
        result.extend(selected);

        tracing::debug!(
            total_messages = history.len(),
            selected_messages = result.len(),
            remaining_budget = remaining,
            "context window assembled"
        );

        Ok(result)
    }

    /// 清空会话历史（保留系统种子消息）。
    async fn clear(&self) {
        let mut history = self.history.write().unwrap_or_else(|e| e.into_inner());
        history.clear();
        tracing::debug!("session history cleared (system message preserved)");
    }
}

impl std::fmt::Debug for WindowMemory {
    /// 自定义 Debug 实现（RwLock 内部 Vec 可能很大，不直接打印）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let history_len = self.history.read().unwrap_or_else(|e| e.into_inner()).len();
        let has_system = self
            .system
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        f.debug_struct("WindowMemory")
            .field("history_len", &history_len)
            .field("has_system", &has_system)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    //! 记忆模块单元测试。
    //!
    //! 覆盖：TokenEstimator、WindowMemory 的 append/get_context/clear/set_system、
    /// 窗口裁剪、系统消息保留。
    use super::*;
    use crate::types::Role;

    /// 验证 CharBasedEstimator 的估算。
    #[test]
    fn test_char_based_estimator() {
        let est = CharBasedEstimator;
        // 4 个字符 ≈ 1 token
        assert_eq!(est.estimate("abcd"), 1);
        // 8 个字符 ≈ 2 tokens
        assert_eq!(est.estimate("abcdefgh"), 2);
        // 空字符串 = 0 tokens
        assert_eq!(est.estimate(""), 0);
        // 3 个字符 = 0 tokens（整数除法）
        assert_eq!(est.estimate("abc"), 0);
    }

    /// 验证 WindowMemory 基本 append + get_context。
    #[tokio::test]
    async fn test_append_and_get_context() {
        let mem = WindowMemory::with_default_estimator(1000);
        mem.append(Message::new(Role::User, "Hello".into())).await;
        mem.append(Message::new(Role::Assistant, "Hi there!".into()))
            .await;

        let ctx = mem.get_context(1000).await.unwrap();
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].role, Role::User);
        assert_eq!(ctx[1].role, Role::Assistant);
    }

    /// 验证系统消息始终前置。
    #[tokio::test]
    async fn test_system_message_always_first() {
        let mem = WindowMemory::with_default_estimator(1000);
        mem.set_system(Message::new(
            Role::System,
            "You are a helpful assistant.".into(),
        ))
        .await;
        mem.append(Message::new(Role::User, "Hello".into())).await;

        let ctx = mem.get_context(1000).await.unwrap();
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].role, Role::System);
        assert_eq!(ctx[1].role, Role::User);
    }

    /// 验证窗口裁剪：预算不足时丢弃更旧的非任务消息，但原始任务始终保留。
    #[tokio::test]
    async fn test_window_trimming() {
        // 小预算：只能容纳少量消息
        let mem = WindowMemory::with_default_estimator(10);
        // 每条消息 20 个字符 ≈ 5 tokens
        mem.append(Message::new(Role::User, "aaaaaaaaaaaaaaaaaaaa".into()))
            .await; // msg1：原始任务（最旧）
        mem.append(Message::new(Role::Assistant, "bbbbbbbbbbbbbbbbbbbb".into()))
            .await; // msg2
        mem.append(Message::new(Role::User, "cccccccccccccccccccc".into()))
            .await; // msg3（最新）

        let ctx = mem.get_context(10).await.unwrap();
        // 新策略：原始任务（首条 User）单独预留预算并截断保留；剩余预算容纳最近消息。
        // 任务预留 3 token（截断为头部片段），剩 7 token 容纳 msg3（5 token），msg2 丢弃。
        assert_eq!(ctx.len(), 2);
        // 第一条是被截断保留的原始任务（头部仍为 'a'，且短于原文）
        assert_eq!(ctx[0].role, Role::User);
        assert!(ctx[0].content.starts_with('a'));
        assert!(ctx[0].content.len() < 20);
        // 第二条是最近的消息 msg3
        assert_eq!(ctx[1].content, "cccccccccccccccccccc");
    }

    /// 回归测试（log2.txt）：单条超大工具结果不得使上下文塌缩为仅系统消息。
    ///
    /// 实测模型读取 34KB/39KB 源文件后，单条工具结果（≈9.7k token）超过整个
    /// 8000 token 预算。旧实现遇超限即 `break`，窗口只剩 system，模型丢失任务
    /// 与历史观察，反复发起相同调用直至耗尽迭代次数。
    #[tokio::test]
    async fn test_oversized_tool_result_keeps_task_and_snippet() {
        let mem = WindowMemory::with_default_estimator(8000);
        mem.set_system(Message::new(Role::System, "sys".into()))
            .await;
        // 原始任务
        mem.append(Message::new(Role::User, "分析需求是否都已实现".into()))
            .await;
        // 超大工具结果：39000 字符 ≈ 9750 token，单条即超整个预算
        let huge = "x".repeat(39000);
        mem.append(Message::tool_result("call_1".into(), huge.clone()))
            .await;

        let ctx = mem.get_context(8000).await.unwrap();
        // 旧实现仅返回 system（1 条）；新实现必须同时保留任务与超大结果片段。
        assert!(
            ctx.len() >= 3,
            "task + oversized result must be retained, got {} msgs",
            ctx.len()
        );
        assert_eq!(ctx[0].role, Role::System);
        assert!(
            ctx.iter()
                .any(|m| m.role == Role::User && m.content.contains("分析需求")),
            "original task must survive truncation"
        );
        // 超大结果被截断纳入（带标记），而非整体丢失
        let tool = ctx
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("oversized tool result retained as snippet");
        assert!(tool.content.contains("[truncated]"));
        assert!(tool.content.len() < huge.len());
    }

    /// 回归测试：预算连 tool_calls 结构都放不下时，truncate_to_budget 返回 None
    /// （而非产出空 content 的"空观察"消息）。
    #[test]
    fn test_truncate_to_budget_none_when_tool_calls_exceed_budget() {
        let mem = WindowMemory::with_default_estimator(1000);
        // 构造一条 tool_calls JSON 很大的消息（≈200 token 的调用参数）
        let call = crate::types::ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "x".repeat(800)}),
            original_name: None,
        };
        let msg = Message::with_tool_calls("short".into(), vec![call]);
        let tool_call_tokens = mem.estimate_message_tokens(&msg);
        assert!(
            tool_call_tokens > 10,
            "precondition: tool_calls must be sizable"
        );
        // 预算小于 tool_calls 开销：截断后 content 将为空，应返回 None
        assert!(mem.truncate_to_budget(&msg, 5).is_none());
        // 预算充足时正常截断/保留
        assert!(
            mem.truncate_to_budget(&msg, tool_call_tokens + 10)
                .is_some()
        );
    }

    /// 验证窗口裁剪时系统消息始终保留。
    #[tokio::test]
    async fn test_window_trimming_preserves_system() {
        let mem = WindowMemory::with_default_estimator(10);
        mem.set_system(Message::new(Role::System, "System prompt here.".into()))
            .await;
        mem.append(Message::new(Role::User, "aaaaaaaaaaaaaaaaaaaa".into()))
            .await;
        mem.append(Message::new(Role::Assistant, "bbbbbbbbbbbbbbbbbbbb".into()))
            .await;

        let ctx = mem.get_context(10).await.unwrap();
        // 系统消息始终在首位
        assert_eq!(ctx[0].role, Role::System);
        // 会话消息被裁剪（只保留最近的）
        assert!(ctx.len() <= 3); // system + 最多 2 条会话
    }

    /// 验证 clear 清空会话历史但保留系统消息。
    #[tokio::test]
    async fn test_clear_preserves_system() {
        let mem = WindowMemory::with_default_estimator(1000);
        mem.set_system(Message::new(Role::System, "System.".into()))
            .await;
        mem.append(Message::new(Role::User, "Hello".into())).await;
        mem.append(Message::new(Role::Assistant, "Hi!".into()))
            .await;

        mem.clear().await;

        let ctx = mem.get_context(1000).await.unwrap();
        // 只剩系统消息
        assert_eq!(ctx.len(), 1);
        assert_eq!(ctx[0].role, Role::System);
    }

    /// 验证空记忆的 get_context 返回空（或仅系统消息）。
    #[tokio::test]
    async fn test_empty_memory() {
        let mem = WindowMemory::with_default_estimator(1000);
        let ctx = mem.get_context(1000).await.unwrap();
        assert!(ctx.is_empty());
    }

    /// 验证 WindowMemory 是 Send + Sync。
    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WindowMemory>();
    }

    /// 验证带工具调用的消息 token 估算包含 tool_calls JSON。
    #[tokio::test]
    async fn test_tool_call_token_estimation() {
        let mem = WindowMemory::with_default_estimator(1000);
        let call = crate::types::ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
            original_name: None,
        };
        let msg = Message::with_tool_calls("Let me check.".into(), vec![call]);
        mem.append(msg).await;

        let ctx = mem.get_context(1000).await.unwrap();
        assert_eq!(ctx.len(), 1);
        assert!(ctx[0].has_tool_calls());
    }

    /// 验证 WindowMemory 自定义 Debug 实现（不 panic 且包含关键字段）。
    #[tokio::test]
    async fn test_window_memory_debug_impl() {
        let mem = WindowMemory::with_default_estimator(512);
        mem.set_system(Message::new(Role::System, "S.".into()))
            .await;
        mem.append(Message::new(Role::User, "hi".into())).await;

        let dbg = format!("{mem:?}");
        assert!(dbg.contains("WindowMemory"));
        assert!(dbg.contains("history_len"));
        assert!(dbg.contains("has_system"));
        assert!(dbg.contains("max_tokens"));
    }
}
