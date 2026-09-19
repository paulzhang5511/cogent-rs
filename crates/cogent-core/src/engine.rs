//! ReAct 引擎：Agent 的核心执行循环。
//!
//! 本模块定义 [`AgentEngine`]——驱动 Agent 完成"思考→工具调用→观察→再思考"
//! 循环的核心引擎。引擎是框架的心脏，协调 LLM、工具、记忆、中间件、事件总线。
//!
//! ReAct 循环：
//! ```text
//! 用户输入
//!   │
//!   ▼
//! ┌─────────────────────────────────────────────┐
//! │ 1. 状态 → Thinking                          │
//! │ 2. 中间件 before_llm                        │
//! │ 3. 调用 LLM (chat_complete)                 │
//! │ 4. 中间件 after_llm                         │
//! │ 5. 判断响应：                                │
//! │    ├─ 有 tool_calls → 状态 → ToolCalling    │
//! │    │   ├─ 防死循环检测                       │
//! │    │   ├─ 中间件 before_tool（每个工具）      │
//! │    │   ├─ 并行执行工具 (join_all)            │
//! │    │   ├─ 状态 → Observing                  │
//! │    │   └─ 工具结果回传 memory → 回到 1       │
//! │    └─ 无 tool_calls → 状态 → Completed      │
//! └─────────────────────────────────────────────┘
//!   │
//!   ▼
//! 最终回复
//! ```
//!
//! 设计原则：
//! - 引擎只依赖 trait（`LLMProvider`/`Tool`/`Memory`/`AgentMiddleware`），不依赖具体实现。
//! - 防死循环：连续两次相同 `name + arguments` 的工具调用被拦截。
//! - 多工具并行：单次 LLM 返回的多个工具调用用 `join_all` 并发执行。
//! - 每次 LLM 调用 / 工具执行开 `tracing` span，实现链路追踪。
//! - 状态迁移通过事件总线广播，实现引擎与展示解耦。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use futures::future::join_all;
use tracing::{Instrument, info};

use crate::error::{CogentError, CogentResult};
use crate::event::{AgentEvent, EventBus};
use crate::memory::Memory;
use crate::middleware::AgentMiddleware;
use crate::provider::LLMProvider;
use crate::state::AgentState;
use crate::tool::Tool;
use crate::types::{Message, Role};

/// 单轮 ReAct 迭代的执行结果。
///
/// 由 [`AgentEngine::step`] 返回，表达本轮迭代的两种可能结局：
/// - `Continue`：本轮执行了工具调用，引擎应继续下一轮迭代。
/// - `Final(String)`：LLM 生成了最终回复（无工具调用），引擎应结束循环并返回该文本。
///
/// 设计为独立枚举而非 `Option<String>`，是为了让"继续"与"结束"两种语义
/// 在类型层面显式区分，避免 `None`/`Some` 的歧义。
#[derive(Debug)]
enum StepOutcome {
    /// 本轮执行了工具调用，应继续下一轮迭代。
    Continue,
    /// LLM 生成了最终回复，循环应结束。携带最终回复文本。
    Final(String),
}

/// ReAct 引擎：驱动 Agent 完成思考-工具调用-观察循环。
///
/// 引擎持有 LLM Provider、工具列表、记忆、中间件链、事件总线，
/// 在 [`AgentEngine::run`] 中执行完整的 ReAct 循环。
///
/// # 线程安全
/// 引擎是 `Send` 的（内部均为 `Arc` 或 `Rc` 包装的 `Send` 类型），
/// 可在异步任务中使用。引擎本身不是 `Sync`（`run` 会修改内部状态），
/// 通常每个会话创建一个引擎实例。
pub struct AgentEngine {
    /// LLM Provider（非流式调用用于 ReAct 循环）。
    provider: Arc<dyn LLMProvider>,
    /// 可用工具列表（按名称索引，便于快速查找）。
    tools: Vec<Arc<dyn Tool>>,
    /// 工具名称 → 工具实例的索引（防死循环检测与工具查找用）。
    tool_index: HashMap<String, Arc<dyn Tool>>,
    /// Agent 记忆（保存对话历史）。
    memory: Arc<dyn Memory>,
    /// 中间件链（按顺序执行）。
    middlewares: Vec<Arc<dyn AgentMiddleware>>,
    /// 事件总线（广播状态迁移、工具执行等事件）。
    event_bus: Arc<EventBus>,
    /// 最大 ReAct 迭代次数（防死循环硬上限）。
    max_iterations: usize,
    /// 工具执行历史（max iterations 终止时汇总生成进度总结）。
    tool_history: Vec<crate::summary::ToolRecord>,
    /// 当前状态。
    state: AgentState,
}

impl AgentEngine {
    /// 创建 ReAct 引擎。
    ///
    /// # 参数
    /// - `provider`：LLM Provider。
    /// - `tools`：可用工具列表。
    /// - `memory`：Agent 记忆。
    /// - `middlewares`：中间件链（按执行顺序）。
    /// - `event_bus`：事件总线。
    /// - `max_iterations`：最大迭代次数。
    ///
    /// # 示例
    /// ```
    /// use std::sync::Arc;
    /// use cogent_core::engine::AgentEngine;
    /// use cogent_core::event::EventBus;
    ///
    /// // 实际使用时需传入具体的 provider/tools/memory 实现
    /// // let engine = AgentEngine::new(provider, tools, memory, mws, bus, 10);
    /// ```
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        tools: Vec<Arc<dyn Tool>>,
        memory: Arc<dyn Memory>,
        middlewares: Vec<Arc<dyn AgentMiddleware>>,
        event_bus: Arc<EventBus>,
        max_iterations: usize,
    ) -> Self {
        // 构建工具名称索引（防死循环检测与工具查找用）
        let tool_index = tools
            .iter()
            .map(|t| (t.name().to_string(), t.clone()))
            .collect();

        tracing::info!(
            tool_count = tools.len(),
            middleware_count = middlewares.len(),
            max_iterations,
            "agent engine initialized"
        );

        Self {
            provider,
            tools,
            tool_index,
            memory,
            middlewares,
            event_bus,
            max_iterations,
            tool_history: Vec::new(),
            state: AgentState::Idle,
        }
    }

    /// 获取当前引擎状态。
    pub fn state(&self) -> AgentState {
        self.state
    }

    /// 获取工具执行历史（max iterations 终止时汇总生成进度总结）。
    pub fn tool_history(&self) -> &[crate::summary::ToolRecord] {
        &self.tool_history
    }

    /// 追加一条消息到记忆（供 CLI 层在 `run` 前后注入用户/助手消息）。
    ///
    /// # 参数
    /// - `message`：要追加的消息。
    pub async fn memory_append(&self, message: Message) {
        self.memory.append(message).await;
    }

    /// 清空会话历史（保留 system 种子消息）。
    ///
    /// 供 CLI 层 `/clear` 命令调用。
    pub async fn memory_clear(&self) {
        self.memory.clear().await;
    }

    /// 获取已注册工具列表（供 CLI 层 `/tools` 命令展示）。
    ///
    /// # 返回
    /// - 工具列表的不可变引用。
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    /// 将引擎从终态（`Completed`/`Failed`）重置为 `Idle`，供多轮会话复用。
    ///
    /// 状态机本身不允许 `Completed → Idle` 的常规迁移（终态不可再迁移），
    /// 但多轮 REPL 需要在每轮结束后复位引擎以接受下一轮输入。
    /// 本方法是 CLI 层的"轮次边界"操作，绕过常规迁移表直接复位。
    ///
    /// # 行为
    /// - 仅当当前状态为终态（`Completed`/`Failed`）时复位为 `Idle`。
    /// - 非终态（`Idle`/`Thinking`/`ToolCalling`/`Observing`）调用本方法为无操作
    ///   （幂等），避免在循环中途误复位。
    /// - 复位后发布 `StateChanged(Idle)` 事件，供渲染器/追踪器感知轮次边界。
    ///
    /// # 副作用
    /// - 修改 `self.state`。
    /// - 发布 `AgentEvent::StateChanged(AgentState::Idle)`。
    pub fn reset(&mut self) {
        // 仅终态可复位；非终态幂等无操作（避免循环中途误复位）。
        // 复位逻辑由状态机的 AgentState::reset 承载，引擎不直接改 state 字段，
        // 保持"状态迁移只经状态机"的不变量。
        let next = self.state.reset();
        if next == self.state {
            // 非终态：无变化，不发布事件
            return;
        }
        self.state = next;
        self.event_bus
            .publish(AgentEvent::StateChanged(AgentState::Idle));
        tracing::debug!("engine reset to Idle for next turn");
    }

    /// 将最终答案分块发布为 `AgentEvent::TokenChunk` 事件，供 CLI 渲染器
    /// 实现打字机效果。
    ///
    /// # 设计说明（避免双 LLM 调用）
    /// 引擎内部 ReAct 循环用非流式 `chat_complete`（需结构化 `tool_calls`），
    /// 已得到完整最终答案文本。本方法**不再**调用 `chat_stream` 发起第二次 LLM 请求，
    /// 而是将已有答案按字符分块（模拟流式 chunk 的粒度）逐段发布 `TokenChunk` 事件。
    /// 这样：
    /// - 消除了每轮两次 LLM 调用的成本与延迟翻倍；
    /// - 保证流式渲染的文本与 `run()` 返回的答案完全一致（无两次生成不一致风险）；
    /// - 仍保留打字机视觉效果（渲染器逐 chunk flush）。
    ///
    /// # 参数
    /// - `answer`：`run()` 返回的最终答案文本。
    ///
    /// # 行为
    /// - 按固定字符窗口（约 16 字符）切片，逐块发布 `TokenChunk`。
    ///   不逐字符发布：长答案逐字符会产生 O(n) 次 `String` 分配与 broadcast 克隆，
    ///   并易触发 channel `Lagged`（SPEC §热路径无冗余克隆）。
    /// - 不修改引擎状态（状态已由 `run` 迁移到 `Completed`）。
    /// - 不访问 provider / memory，无 IO。
    pub fn render_answer_chunked(&self, answer: &str) {
        // 按固定字符窗口切片发布（模拟流式 chunk 粒度，渲染器逐块 flush）。
        const CHUNK_CHARS: usize = 16;
        let mut buf = String::with_capacity(CHUNK_CHARS);
        for ch in answer.chars() {
            buf.push(ch);
            if buf.chars().count() >= CHUNK_CHARS {
                self.event_bus
                    .publish(AgentEvent::TokenChunk(std::mem::take(&mut buf)));
            }
        }
        if !buf.is_empty() {
            self.event_bus.publish(AgentEvent::TokenChunk(buf));
        }
        tracing::info!(answer_len = answer.len(), "final answer chunked for render");
    }

    /// 执行一轮完整的 ReAct 循环。
    ///
    /// 从当前状态开始，循环执行"思考→工具调用→观察"，
    /// 直到 LLM 生成最终回复（无工具调用）或达到最大迭代次数。
    ///
    /// # 参数
    /// - `max_context_tokens`：上下文窗口 token 预算（传给 `memory.get_context`）。
    ///
    /// # 返回
    /// - 成功：LLM 的最终回复文本。
    /// - 失败：`CogentError`（死循环、最大迭代超限、Provider 错误等）。
    ///
    /// # 流程
    /// 1. 状态 `Idle → Thinking`。
    /// 2. 循环（最多 `max_iterations` 次）：
    ///    a. 中间件 `before_llm`。
    ///    b. 从 memory 获取上下文窗口。
    ///    c. 调用 LLM `chat_complete`。
    ///    d. 中间件 `after_llm`。
    ///    e. 若响应含 `tool_calls`：
    ///       - 状态 `Thinking → ToolCalling`。
    ///       - 防死循环检测。
    ///       - 中间件 `before_tool`（每个工具）。
    ///       - 并行执行工具。
    ///       - 状态 `ToolCalling → Observing`。
    ///       - 工具结果回传 memory。
    ///       - 状态 `Observing → Thinking`（下一轮）。
    ///
    ///    f. 若响应无 `tool_calls`：
    ///       - 状态 `Thinking → Completed`。
    ///       - 返回最终回复。
    /// 3. 达到最大迭代次数：状态 → `Failed`，返回错误。
    pub async fn run(&mut self, max_context_tokens: usize) -> CogentResult<String> {
        // 状态迁移：Idle → Thinking
        self.state = self.state.transition(AgentState::Thinking)?;
        self.event_bus
            .publish(AgentEvent::StateChanged(AgentState::Thinking));
        info!("react loop started");

        // 防死循环：记录上一轮的工具调用签名（name + arguments）
        // 使用 Vec 保存本轮所有工具调用签名，下一轮对比
        let mut prev_tool_signatures: Vec<String> = Vec::new();

        for iteration in 1..=self.max_iterations {
            // 每轮迭代开一个 span，关联该轮的所有 LLM 调用与工具执行
            let iter_span = tracing::info_span!("react_iteration", iteration);
            let outcome = self
                .step(max_context_tokens, iteration, &mut prev_tool_signatures)
                .instrument(iter_span)
                .await?;

            match outcome {
                StepOutcome::Continue => {}
                StepOutcome::Final(answer) => {
                    // 最终回复写入 memory（以 Role::Assistant），保证多轮上下文完整。
                    // 此前仅工具结果（Role::Tool）被写入，最终回复未入 memory——
                    // 修复后 CLI 不再需要二次追加，且 render_answer_chunked 不再发起第二次 LLM 调用。
                    self.memory
                        .append(Message::new(Role::Assistant, answer.clone()))
                        .await;
                    return Ok(answer);
                }
            }
        }

        // 达到最大迭代次数：转 Failed 并返回错误。
        // 进度总结的"生成文本 → 落盘 output/summary.md → 打印 stdout"属展示/IO
        // 关注点，不在引擎内执行（core 与展示解耦，SPEC §总原则）：由 CLI 层捕获
        // `MaxIterationsExceeded` 后读取 `tool_history()` 自行处理。
        tracing::warn!(
            iterations = self.max_iterations,
            "max iterations reached, engine terminating"
        );
        self.state = self.state.transition(AgentState::Failed)?;
        self.event_bus
            .publish(AgentEvent::StateChanged(AgentState::Failed));
        Err(CogentError::MaxIterationsExceeded(self.max_iterations))
    }

    /// 执行单轮 ReAct 迭代（思考→工具调用→观察）。
    ///
    /// 从 `run` 的循环体提取为独立方法，以便：
    /// 1. 使用 `?` 操作符（方法返回 `CogentResult`，而 `async` 块内不能用 `?`）。
    /// 2. 通过 [`StepOutcome`] 表达"继续循环"或"返回最终答案"，替代 `async` 块内无法使用的 `continue`/`return`。
    ///
    /// # 参数
    /// - `max_context_tokens`：上下文窗口 token 预算。
    /// - `iteration`：当前迭代次数（用于日志与 span）。
    /// - `prev_tool_signatures`：上一轮工具调用签名（防死循环检测，可变引用）。
    ///
    /// # 返回
    /// - `StepOutcome::Continue`：本轮执行了工具调用，应继续下一轮迭代。
    /// - `StepOutcome::Final(String)`：LLM 生成了最终回复，循环应结束。
    async fn step(
        &mut self,
        max_context_tokens: usize,
        iteration: usize,
        prev_tool_signatures: &mut Vec<String>,
    ) -> CogentResult<StepOutcome> {
        // 1. 从 memory 获取上下文窗口（每轮仅取一次，供中间件校验与 LLM 调用复用）
        let context = self.memory.get_context(max_context_tokens).await?;

        // 2. 中间件 before_llm
        self.run_before_llm_middleware(&context).await?;

        // 3. 调用 LLM（开 span 追踪）
        let llm_span = tracing::info_span!("llm_call", messages = context.len());
        let response = {
            let _enter = llm_span.enter();
            info!(messages = context.len(), "llm request started");
            let start = Instant::now();
            let resp = self
                .provider
                .chat_complete(&context, &self.tools)
                .await
                .map_err(CogentError::Provider)?;
            let duration_ms = start.elapsed().as_millis() as u64;
            info!(duration_ms, "llm request completed");
            resp
        };

        // 发布 token 用量事件
        self.event_bus.publish(AgentEvent::TokenUsage {
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
        });

        // 4. 中间件 after_llm
        for mw in &self.middlewares {
            mw.after_llm(&response).await?;
        }

        // 5. 判断响应
        if response.has_tool_calls() {
            // 有工具调用 → ToolCalling
            self.state = self.state.transition(AgentState::ToolCalling)?;
            self.event_bus
                .publish(AgentEvent::StateChanged(AgentState::ToolCalling));

            // 防死循环检测：对比上一轮的工具调用签名
            let current_signatures: Vec<String> = response
                .tool_calls
                .iter()
                .map(|tc| format!("{}:{}", tc.name, tc.arguments))
                .collect();

            // 检测是否有与上一轮完全相同的工具调用。
            // 用集合语义（任一当前签名出现在上一轮即判定）而非按位置 zip：
            // LLM 重排工具调用顺序或改变数量时（prev=[A,B], curr=[B,A]）zip 会漏检。
            let loop_detected = current_signatures
                .iter()
                .any(|curr| prev_tool_signatures.contains(curr));

            if loop_detected {
                // 检测到死循环：注入告警消息，跳过工具执行
                tracing::warn!(
                    iteration,
                    "loop detected: identical tool calls in consecutive iterations"
                );
                // 用 User 角色注入提示（而非伪造 role=tool 消息）：
                // OpenAI API 要求 role=tool 必须携带与 assistant tool_calls 匹配的
                // tool_call_id，空 id 会触发 400。普通 User 文本不涉及 tool 关联，
                // 模型同样能据此收到纠正信号。
                let warning = Message::new(
                    Role::User,
                    "WARNING: You are repeating the same tool call with identical arguments. \
                     Please try a different approach or provide your final answer."
                        .to_string(),
                );
                self.memory.append(warning).await;

                // 状态 Observing → Thinking（与正常路径一致地广播每次 StateChanged，
                // 否则事件总线上的渲染器/观测器会停留在 ToolCalling，状态指示脱节）
                self.state = self.state.transition(AgentState::Observing)?;
                self.event_bus
                    .publish(AgentEvent::StateChanged(AgentState::Observing));
                self.state = self.state.transition(AgentState::Thinking)?;
                self.event_bus
                    .publish(AgentEvent::StateChanged(AgentState::Thinking));
                *prev_tool_signatures = current_signatures;
                return Ok(StepOutcome::Continue);
            }

            // 执行工具调用
            let records = self
                .execute_tool_calls(&response.tool_calls, iteration)
                .await?;
            self.tool_history.extend(records);

            // 状态 → Observing → Thinking
            self.state = self.state.transition(AgentState::Observing)?;
            self.event_bus
                .publish(AgentEvent::StateChanged(AgentState::Observing));
            self.state = self.state.transition(AgentState::Thinking)?;
            self.event_bus
                .publish(AgentEvent::StateChanged(AgentState::Thinking));

            *prev_tool_signatures = current_signatures;
            Ok(StepOutcome::Continue)
        } else {
            // 无工具调用 → 最终回复。
            // 空响应检测：模型返回空 content 且无 tool_calls 时（上下文过长或
            // 陷入困境），显式报错而非静默返回空串（实测 log：answer_len=0，
            // CLI 在 "cog:" 后输出空白，用户无从得知失败原因）。
            if response.content.trim().is_empty() {
                tracing::error!(iteration, "empty LLM response (no content, no tool calls)");
                self.state = self.state.transition(AgentState::Failed)?;
                self.event_bus
                    .publish(AgentEvent::StateChanged(AgentState::Failed));
                return Err(CogentError::EmptyResponse(iteration));
            }
            self.state = self.state.transition(AgentState::Completed)?;
            self.event_bus
                .publish(AgentEvent::StateChanged(AgentState::Completed));
            info!(iteration, "react loop completed");
            Ok(StepOutcome::Final(response.content))
        }
    }

    /// 执行 LLM 返回的所有工具调用（并行）。
    ///
    /// # 参数
    /// - `tool_calls`：工具调用列表。
    /// - `iteration`：当前迭代次数（用于日志）。
    ///
    /// # 流程
    /// 1. 对每个工具调用：中间件 `before_tool`（拦截则跳过）。
    /// 2. 用 `join_all` 并行执行所有未被拦截的工具。
    /// 3. 将每个工具结果以 `Role::Tool` 消息回传 memory。
    async fn execute_tool_calls(
        &self,
        tool_calls: &[crate::types::ToolCall],
        iteration: usize,
    ) -> CogentResult<Vec<crate::summary::ToolRecord>> {
        info!(
            tool_count = tool_calls.len(),
            iteration, "executing tool calls"
        );

        // 为每个工具调用创建执行任务
        let mut tasks = Vec::new();

        for tc in tool_calls {
            // 查找工具
            let tool = match self.tool_index.get(&tc.name) {
                Some(t) => t.clone(),
                None => {
                    // 未知工具：返回结构化错误（含可用工具列表）给 LLM，
                    // 让模型知道该工具不可用并改用可用工具，避免盲目重试死循环。
                    tracing::warn!(tool_name = %tc.name, "unknown tool requested");
                    let available: Vec<&str> = self.tool_index.keys().map(|s| s.as_str()).collect();
                    let err_msg = Message::tool_result(
                        tc.id.clone(),
                        format!(
                            "Error: unknown tool '{}'. Available tools: [{}]. \
                             Use one of the available tools instead.",
                            tc.name,
                            available.join(", ")
                        ),
                    );
                    self.memory.append(err_msg).await;
                    continue;
                }
            };

            // 中间件 before_tool（拦截则跳过）
            let mut blocked = false;
            for mw in &self.middlewares {
                if let Err(e) = mw.before_tool(&tc.name, &tc.arguments).await {
                    tracing::warn!(
                        tool_name = %tc.name,
                        reason = %e,
                        "tool call blocked by middleware"
                    );
                    let err_msg = Message::tool_result(
                        tc.id.clone(),
                        format!("Error: tool call blocked: {e}"),
                    );
                    self.memory.append(err_msg).await;
                    blocked = true;
                    break;
                }
            }
            if blocked {
                continue;
            }

            // 发布 ToolStarted 事件
            self.event_bus.publish(AgentEvent::ToolStarted {
                id: tc.id.clone(),
                name: tc.name.clone(),
                args: tc.arguments.clone(),
            });

            // 创建并行执行任务
            let tool_call_id = tc.id.clone();
            let tool_name = tc.name.clone();
            let args = tc.arguments.clone();
            tasks.push(async move {
                let start = Instant::now();
                let result = tool.execute(args).await;
                let duration_ms = start.elapsed().as_millis() as u64;
                (tool_call_id, tool_name, duration_ms, result)
            });
        }

        // 并行执行所有工具
        let results = join_all(tasks).await;

        // 处理每个工具结果
        let mut records = Vec::with_capacity(results.len());
        for (tool_call_id, tool_name, duration_ms, result) in results {
            let success = result.is_ok();
            // 发布 ToolFinished 事件
            self.event_bus.publish(AgentEvent::ToolFinished {
                id: tool_call_id.clone(),
                duration_ms,
                success,
            });

            // 将结果回传 memory
            let (content, error) = match result {
                Ok(output) => {
                    tracing::info!(
                        tool_name = %tool_name,
                        duration_ms,
                        "tool executed successfully"
                    );
                    (output, None)
                }
                Err(e) => {
                    tracing::error!(
                        tool_name = %tool_name,
                        duration_ms,
                        error = %e,
                        "tool execution failed"
                    );
                    (
                        format!("Error: tool '{tool_name}' failed: {e}"),
                        Some(e.to_string()),
                    )
                }
            };

            let tool_msg = Message::tool_result(tool_call_id, content);
            self.memory.append(tool_msg).await;

            // 记录工具执行历史（max iterations 终止时汇总生成进度总结）
            records.push(crate::summary::ToolRecord {
                tool_name,
                success,
                error,
            });
        }

        Ok(records)
    }

    /// 执行所有中间件的 `before_llm` 钩子。
    ///
    /// 复用 `step()` 已取到的 `context`，避免每轮对全量历史做两遍 token 估算。
    async fn run_before_llm_middleware(&self, context: &[Message]) -> CogentResult<()> {
        for mw in &self.middlewares {
            mw.before_llm(context).await?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for AgentEngine {
    /// 自定义 Debug 实现（内部 Arc<dyn> 不实现 Debug）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentEngine")
            .field("state", &self.state)
            .field("tool_count", &self.tools.len())
            .field("middleware_count", &self.middlewares.len())
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    //! AgentEngine 单元测试。
    //!
    /// 使用 Mock LLMProvider 和 MockTool 验证：
    /// - ReAct 循环完整流程。
    /// - 防死循环检测。
    /// - 多工具并行执行。
    /// - 中间件拦截。
    /// - 最大迭代次数限制。
    use super::*;
    use crate::memory::WindowMemory;
    use crate::provider::LLMResponse;
    use crate::types::ToolCall;
    use async_trait::async_trait;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// Mock LLM Provider：按顺序返回预置响应队列。
    ///
    /// 用于测试 ReAct 循环，无需真实 LLM API。
    struct MockLLMProvider {
        /// 预置响应队列（按顺序弹出）。
        responses: Mutex<VecDeque<LLMResponse>>,
    }

    impl MockLLMProvider {
        /// 创建 Mock Provider，预置响应列表。
        fn new(responses: Vec<LLMResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockLLMProvider {
        async fn chat_complete(
            &self,
            _messages: &[Message],
            _tools: &[Arc<dyn Tool>],
        ) -> anyhow::Result<LLMResponse> {
            let mut queue = self.responses.lock().unwrap();
            queue
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("MockLLMProvider: no more responses"))
        }

        fn chat_stream(
            &self,
            _messages: &[Message],
            _tools: &[Arc<dyn Tool>],
        ) -> std::pin::Pin<Box<dyn futures::Stream<Item = anyhow::Result<String>> + Send>> {
            Box::pin(futures::stream::empty())
        }
    }

    /// Mock 工具：返回预置结果，可配置延迟（用于并行测试）。
    struct MockTool {
        name: String,
        result: String,
        delay_ms: u64,
        /// 记录执行次数（验证防死循环）。
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockTool {
        fn new(name: &str, result: &str) -> Self {
            Self {
                name: name.into(),
                result: result.into(),
                delay_ms: 0,
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn with_delay(name: &str, result: &str, delay_ms: u64) -> Self {
            Self {
                name: name.into(),
                result: result.into(),
                delay_ms,
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.call_count.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "Mock tool for testing"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<String> {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            Ok(self.result.clone())
        }
    }

    /// 创建测试用引擎（Mock Provider + Mock Tools）。
    fn make_engine(
        responses: Vec<LLMResponse>,
        tools: Vec<Arc<dyn Tool>>,
        max_iterations: usize,
    ) -> (AgentEngine, Arc<EventBus>) {
        let provider = Arc::new(MockLLMProvider::new(responses));
        let memory = Arc::new(WindowMemory::with_default_estimator(8000));
        let bus = Arc::new(EventBus::new(64));
        let engine = AgentEngine::new(
            provider,
            tools,
            memory,
            Vec::new(),
            bus.clone(),
            max_iterations,
        );
        (engine, bus)
    }

    /// 验证完整 ReAct 循环：思考→工具调用→观察→最终回复，并广播对应 AgentEvent。
    #[tokio::test]
    async fn test_full_react_cycle() {
        // 第 1 轮：LLM 请求调用工具
        let tool_call = ToolCall {
            id: "call_1".into(),
            name: "mock".into(),
            arguments: serde_json::json!({"x": 1}),
            original_name: None,
        };
        let resp1 = LLMResponse::with_tool_calls("Let me check.".into(), vec![tool_call], 10, 5);
        // 第 2 轮：LLM 生成最终回复
        let resp2 = LLMResponse::text("The answer is 42.".into(), 20, 10);

        let tool = Arc::new(MockTool::new("mock", "result_42"));
        let (mut engine, bus) = make_engine(vec![resp1, resp2], vec![tool.clone()], 10);

        // 订阅事件总线，验证 ReAct 循环广播了正确的 AgentEvent 序列
        let mut rx = bus.subscribe();

        let result = engine.run(8000).await.unwrap();
        assert_eq!(result, "The answer is 42.");
        assert_eq!(engine.state(), AgentState::Completed);
        // 工具应被执行 1 次
        assert_eq!(tool.call_count(), 1);

        // 收集所有事件（带超时，避免挂起）
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }

        // 验证关键事件存在（顺序由状态机保证，此处验证集合完整性）
        let state_changes: Vec<AgentState> = events
            .iter()
            .filter_map(|e| {
                if let AgentEvent::StateChanged(s) = e {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            state_changes.contains(&AgentState::Thinking),
            "should have Thinking state"
        );
        assert!(
            state_changes.contains(&AgentState::ToolCalling),
            "should have ToolCalling state"
        );
        assert!(
            state_changes.contains(&AgentState::Observing),
            "should have Observing state"
        );
        assert!(
            state_changes.contains(&AgentState::Completed),
            "should have Completed state"
        );

        // 验证工具事件
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolStarted { name, .. } if name == "mock")),
            "should have ToolStarted event"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolFinished { success, .. } if *success)),
            "should have ToolFinished(success) event"
        );

        // 验证 TokenUsage 事件
        assert!(
            events.iter().any(
                |e| matches!(e, AgentEvent::TokenUsage { prompt_tokens, .. } if *prompt_tokens > 0)
            ),
            "should have TokenUsage event"
        );
    }

    /// 验证防死循环：连续两次相同工具调用被拦截。
    #[tokio::test]
    async fn test_loop_detection() {
        // 第 1 轮：LLM 请求调用工具（签名 "mock:{\"x\":1}"）
        let tc1 = ToolCall {
            id: "call_1".into(),
            name: "mock".into(),
            arguments: serde_json::json!({"x": 1}),
            original_name: None,
        };
        let resp1 = LLMResponse::with_tool_calls(String::new(), vec![tc1], 10, 5);
        // 第 2 轮：LLM 再次请求相同工具调用（死循环）
        let tc2 = ToolCall {
            id: "call_2".into(),
            name: "mock".into(),
            arguments: serde_json::json!({"x": 1}),
            original_name: None,
        };
        let resp2 = LLMResponse::with_tool_calls(String::new(), vec![tc2], 10, 5);
        // 第 3 轮：LLM 生成最终回复
        let resp3 = LLMResponse::text("Done.".into(), 20, 5);

        let tool = Arc::new(MockTool::new("mock", "result"));
        let (mut engine, _bus) = make_engine(vec![resp1, resp2, resp3], vec![tool.clone()], 10);

        let result = engine.run(8000).await.unwrap();
        assert_eq!(result, "Done.");
        // 工具只应被执行 1 次（第 2 轮被死循环检测拦截）
        assert_eq!(tool.call_count(), 1);
    }

    /// 验证死循环检测对工具调用**重排**也生效（集合语义，非按位置 zip）。
    /// 旧实现用 zip 按位置配对，prev=[A,B] 与 curr=[B,A] 会漏检。
    #[tokio::test]
    async fn test_loop_detection_reordered() {
        // 第 1 轮：两个工具调用 [A, B]
        let a1 = ToolCall {
            id: "a1".into(),
            name: "mock".into(),
            arguments: serde_json::json!({"x": 1}),
            original_name: None,
        };
        let b1 = ToolCall {
            id: "b1".into(),
            name: "mock".into(),
            arguments: serde_json::json!({"y": 2}),
            original_name: None,
        };
        let resp1 = LLMResponse::with_tool_calls(String::new(), vec![a1, b1], 10, 5);
        // 第 2 轮：相同两个调用但顺序颠倒 [B, A] —— 仍是死循环
        let a2 = ToolCall {
            id: "a2".into(),
            name: "mock".into(),
            arguments: serde_json::json!({"x": 1}),
            original_name: None,
        };
        let b2 = ToolCall {
            id: "b2".into(),
            name: "mock".into(),
            arguments: serde_json::json!({"y": 2}),
            original_name: None,
        };
        let resp2 = LLMResponse::with_tool_calls(String::new(), vec![b2, a2], 10, 5);
        // 第 3 轮：最终回复
        let resp3 = LLMResponse::text("Done.".into(), 20, 5);

        let tool = Arc::new(MockTool::new("mock", "result"));
        let (mut engine, _bus) = make_engine(vec![resp1, resp2, resp3], vec![tool.clone()], 10);

        let result = engine.run(8000).await.unwrap();
        assert_eq!(result, "Done.");
        // 第 2 轮被拦截，工具只执行第 1 轮的 2 次
        assert_eq!(tool.call_count(), 2);
    }

    /// 验证多工具并行执行：总耗时 ≈ 单工具最大耗时（非累加）。
    #[tokio::test]
    async fn test_parallel_tool_execution() {
        // LLM 一次返回 3 个工具调用，每个延迟 100ms
        let tc1 = ToolCall {
            id: "c1".into(),
            name: "t1".into(),
            arguments: serde_json::json!({}),
            original_name: None,
        };
        let tc2 = ToolCall {
            id: "c2".into(),
            name: "t2".into(),
            arguments: serde_json::json!({}),
            original_name: None,
        };
        let tc3 = ToolCall {
            id: "c3".into(),
            name: "t3".into(),
            arguments: serde_json::json!({}),
            original_name: None,
        };
        let resp1 = LLMResponse::with_tool_calls(String::new(), vec![tc1, tc2, tc3], 10, 5);
        let resp2 = LLMResponse::text("All done.".into(), 20, 5);

        let t1 = Arc::new(MockTool::with_delay("t1", "r1", 100));
        let t2 = Arc::new(MockTool::with_delay("t2", "r2", 100));
        let t3 = Arc::new(MockTool::with_delay("t3", "r3", 100));
        let (mut engine, _bus) = make_engine(
            vec![resp1, resp2],
            vec![t1.clone(), t2.clone(), t3.clone()],
            10,
        );

        let start = Instant::now();
        let result = engine.run(8000).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result, "All done.");
        // 并行执行：总耗时应接近 100ms（单工具），而非 300ms（串行）
        // 允许 250ms 阈值（CI 环境可能有抖动）
        assert!(
            elapsed.as_millis() < 250,
            "parallel execution took {}ms (expected < 250ms)",
            elapsed.as_millis()
        );
        // 所有工具都应被执行
        assert_eq!(t1.call_count(), 1);
        assert_eq!(t2.call_count(), 1);
        assert_eq!(t3.call_count(), 1);
    }

    /// 验证最大迭代次数限制：达到上限后返回错误。
    #[tokio::test]
    async fn test_max_iterations_exceeded() {
        // LLM 始终返回工具调用（永不生成最终回复）
        let responses: Vec<LLMResponse> = (0..20)
            .map(|i| {
                let tc = ToolCall {
                    id: format!("call_{i}"),
                    name: "mock".into(),
                    // 每次用不同参数避免触发死循环检测
                    arguments: serde_json::json!({"i": i}),
                    original_name: None,
                };
                LLMResponse::with_tool_calls(String::new(), vec![tc], 10, 5)
            })
            .collect();

        let tool = Arc::new(MockTool::new("mock", "result"));
        let (mut engine, _bus) = make_engine(responses, vec![tool], 3); // max_iterations = 3

        let result = engine.run(8000).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, CogentError::MaxIterationsExceeded(3)));
        assert_eq!(engine.state(), AgentState::Failed);
        // 工具执行历史应记录 3 次（每轮 1 次工具调用）
        assert_eq!(engine.tool_history().len(), 3);
        assert!(engine.tool_history().iter().all(|r| r.success));
        assert!(engine.tool_history().iter().all(|r| r.tool_name == "mock"));
    }

    /// 验证中间件拦截：before_tool 返回 Err 时工具不执行。
    #[tokio::test]
    async fn test_middleware_blocks_tool() {
        // 拦截所有名为 "blocked_tool" 的调用
        struct BlockerMiddleware;
        #[async_trait]
        impl AgentMiddleware for BlockerMiddleware {
            async fn before_llm(&self, _m: &[Message]) -> CogentResult<()> {
                Ok(())
            }
            async fn after_llm(&self, _response: &LLMResponse) -> CogentResult<()> {
                Ok(())
            }
            async fn before_tool(&self, name: &str, _args: &serde_json::Value) -> CogentResult<()> {
                if name == "blocked_tool" {
                    Err(CogentError::MiddlewareBlocked {
                        middleware: "Blocker".into(),
                        reason: "blocked by policy".into(),
                    })
                } else {
                    Ok(())
                }
            }
        }

        let tc = ToolCall {
            id: "c1".into(),
            name: "blocked_tool".into(),
            arguments: serde_json::json!({}),
            original_name: None,
        };
        let resp1 = LLMResponse::with_tool_calls(String::new(), vec![tc], 10, 5);
        let resp2 = LLMResponse::text("OK.".into(), 20, 5);

        let tool = Arc::new(MockTool::new("blocked_tool", "should_not_run"));
        let provider = Arc::new(MockLLMProvider::new(vec![resp1, resp2]));
        let memory = Arc::new(WindowMemory::with_default_estimator(8000));
        let bus = Arc::new(EventBus::new(64));
        let mws: Vec<Arc<dyn AgentMiddleware>> = vec![Arc::new(BlockerMiddleware)];
        let mut engine = AgentEngine::new(provider, vec![tool.clone()], memory, mws, bus, 10);

        let result = engine.run(8000).await.unwrap();
        assert_eq!(result, "OK.");
        // 工具应被拦截，不执行
        assert_eq!(tool.call_count(), 0);
    }

    /// 验证未知工具：LLM 请求不存在的工具时返回错误消息。
    #[tokio::test]
    async fn test_unknown_tool() {
        let tc = ToolCall {
            id: "c1".into(),
            name: "nonexistent".into(),
            arguments: serde_json::json!({}),
            original_name: None,
        };
        let resp1 = LLMResponse::with_tool_calls(String::new(), vec![tc], 10, 5);
        let resp2 = LLMResponse::text("Handled.".into(), 20, 5);

        // 不注册任何工具
        let (mut engine, _bus) = make_engine(vec![resp1, resp2], Vec::new(), 10);
        let result = engine.run(8000).await.unwrap();
        assert_eq!(result, "Handled.");
    }

    /// 验证事件总线收到状态迁移事件。
    #[tokio::test]
    async fn test_state_change_events() {
        let resp = LLMResponse::text("Direct answer.".into(), 10, 5);
        let (mut engine, bus) = make_engine(vec![resp], Vec::new(), 10);

        // 订阅事件
        let mut rx = bus.subscribe();
        let _ = engine.run(8000).await.unwrap();

        // 应收到 StateChanged 事件（Thinking → Completed）
        let mut got_thinking = false;
        let mut got_completed = false;
        for _ in 0..10 {
            match rx.try_recv() {
                Ok(AgentEvent::StateChanged(AgentState::Thinking)) => got_thinking = true,
                Ok(AgentEvent::StateChanged(AgentState::Completed)) => got_completed = true,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert!(got_thinking, "should receive Thinking state event");
        assert!(got_completed, "should receive Completed state event");
    }

    /// 验证 TokenUsage 事件发布。
    #[tokio::test]
    async fn test_token_usage_event() {
        let resp = LLMResponse::text("Answer.".into(), 100, 50);
        let (mut engine, bus) = make_engine(vec![resp], Vec::new(), 10);

        let mut rx = bus.subscribe();
        let _ = engine.run(8000).await.unwrap();

        let mut got_usage = false;
        for _ in 0..10 {
            match rx.try_recv() {
                Ok(AgentEvent::TokenUsage {
                    prompt_tokens,
                    completion_tokens,
                }) => {
                    assert_eq!(prompt_tokens, 100);
                    assert_eq!(completion_tokens, 50);
                    got_usage = true;
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert!(got_usage, "should receive TokenUsage event");
    }

    /// 失败工具：`execute` 始终返回错误（覆盖工具执行失败分支）。
    struct FailingTool;

    #[async_trait]
    impl Tool for FailingTool {
        fn name(&self) -> &str {
            "failing"
        }
        fn description(&self) -> &str {
            "Always fails"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<String> {
            Err(anyhow::anyhow!("boom"))
        }
    }

    /// 验证工具执行失败：错误被格式化为 `Role::Tool` 消息回传，循环继续。
    #[tokio::test]
    async fn test_tool_execution_failure() {
        let tc = ToolCall {
            id: "c1".into(),
            name: "failing".into(),
            arguments: serde_json::json!({}),
            original_name: None,
        };
        let resp1 = LLMResponse::with_tool_calls(String::new(), vec![tc], 10, 5);
        let resp2 = LLMResponse::text("Recovered.".into(), 20, 5);

        let (mut engine, _bus) = make_engine(vec![resp1, resp2], vec![Arc::new(FailingTool)], 10);
        let result = engine.run(8000).await.unwrap();
        assert_eq!(result, "Recovered.");
        // 工具结果消息应包含错误信息
        let ctx = engine.memory.get_context(8000).await.unwrap();
        let tool_msgs: Vec<_> = ctx
            .iter()
            .filter(|m| m.role == crate::types::Role::Tool)
            .collect();
        assert_eq!(tool_msgs.len(), 1);
        assert!(tool_msgs[0].content.contains("failed"));
    }

    /// 验证引擎访问器：`memory_append` / `memory_clear` / `tools`。
    #[tokio::test]
    async fn test_engine_accessors() {
        let resp = LLMResponse::text("ok".into(), 10, 5);
        let tool = Arc::new(MockTool::new("mock", "r"));
        let (engine, _bus) = make_engine(vec![resp], vec![tool.clone()], 10);

        // tools 访问器
        assert_eq!(engine.tools().len(), 1);
        assert_eq!(engine.tools()[0].name(), "mock");

        // memory_append / memory_clear
        engine
            .memory_append(Message::new(crate::types::Role::User, "hi".into()))
            .await;
        let ctx = engine.memory.get_context(8000).await.unwrap();
        assert_eq!(ctx.len(), 1);
        engine.memory_clear().await;
        let ctx = engine.memory.get_context(8000).await.unwrap();
        assert!(ctx.is_empty());
    }

    /// 验证 AgentEngine 自定义 Debug 实现（不 panic 且包含关键字段）。
    #[test]
    fn test_engine_debug_impl() {
        let resp = LLMResponse::text("ok".into(), 10, 5);
        let (engine, _bus) = make_engine(vec![resp], Vec::new(), 7);
        let dbg = format!("{engine:?}");
        assert!(dbg.contains("AgentEngine"));
        assert!(dbg.contains("Idle"));
        assert!(dbg.contains("max_iterations"));
    }

    /// 验证 reset：终态（Completed）复位为 Idle，并发布 StateChanged 事件。
    #[tokio::test]
    async fn test_reset_from_completed() {
        let resp = LLMResponse::text("done".into(), 10, 5);
        let (mut engine, bus) = make_engine(vec![resp], Vec::new(), 10);
        let _ = engine.run(8000).await.unwrap();
        assert_eq!(engine.state(), AgentState::Completed);

        let mut rx = bus.subscribe();
        engine.reset();
        assert_eq!(engine.state(), AgentState::Idle);

        // 应收到 StateChanged(Idle) 事件
        let mut got_idle = false;
        for _ in 0..10 {
            match rx.try_recv() {
                Ok(AgentEvent::StateChanged(AgentState::Idle)) => got_idle = true,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert!(got_idle, "should receive StateChanged(Idle) after reset");
    }

    /// 验证 reset 幂等：非终态（Idle）调用 reset 为无操作。
    #[test]
    fn test_reset_idempotent_on_idle() {
        let resp = LLMResponse::text("ok".into(), 10, 5);
        let (mut engine, _bus) = make_engine(vec![resp], Vec::new(), 10);
        assert_eq!(engine.state(), AgentState::Idle);
        engine.reset();
        // 仍为 Idle（无操作）
        assert_eq!(engine.state(), AgentState::Idle);
    }

    /// 验证多轮会话：run → reset → run 可连续执行（状态机复位）。
    #[tokio::test]
    async fn test_multi_turn_via_reset() {
        // 两轮各一个最终回复
        let resp1 = LLMResponse::text("first".into(), 10, 5);
        let resp2 = LLMResponse::text("second".into(), 10, 5);
        let (mut engine, _bus) = make_engine(vec![resp1, resp2], Vec::new(), 10);

        let r1 = engine.run(8000).await.unwrap();
        assert_eq!(r1, "first");
        assert_eq!(engine.state(), AgentState::Completed);

        // 复位后第二轮可继续
        engine.reset();
        let r2 = engine.run(8000).await.unwrap();
        assert_eq!(r2, "second");
        assert_eq!(engine.state(), AgentState::Completed);
    }

    /// 验证 render_answer_chunked：按字符分块发布 TokenChunk 事件。
    #[test]
    fn test_render_answer_chunked() {
        let resp = LLMResponse::text("Hi!".into(), 10, 5);
        let (engine, bus) = make_engine(vec![resp], Vec::new(), 10);

        let mut rx = bus.subscribe();
        engine.render_answer_chunked("Hi!");

        // "Hi!" 仅 3 字符（< 16 字符窗口），整块作为单个 TokenChunk 发出
        let mut received = Vec::new();
        for _ in 0..10 {
            match rx.try_recv() {
                Ok(AgentEvent::TokenChunk(c)) => received.push(c),
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert_eq!(received, vec!["Hi!"]);
    }

    /// 验证 C1 修复：run() 的最终答案被写入 memory（多轮上下文完整），
    /// 且 render_answer_chunked 不发起第二次 LLM 调用（答案与 run() 一致）。
    #[tokio::test]
    async fn test_run_writes_final_answer_to_memory() {
        let resp = LLMResponse::text("The answer is 42.".into(), 20, 10);
        let (mut engine, bus) = make_engine(vec![resp], Vec::new(), 10);

        let mut rx = bus.subscribe();

        let answer = engine.run(8000).await.unwrap();
        assert_eq!(answer, "The answer is 42.");

        // 最终答案应已写入 memory（Role::Assistant）
        let ctx = engine.memory.get_context(8000).await.unwrap();
        let assistant_msgs: Vec<_> = ctx.iter().filter(|m| m.role == Role::Assistant).collect();
        assert_eq!(assistant_msgs.len(), 1);
        assert_eq!(assistant_msgs[0].content, "The answer is 42.");

        // render_answer_chunked 发布的 TokenChunk 拼接应与 run() 返回值完全一致
        engine.render_answer_chunked(&answer);
        let mut received = String::new();
        for _ in 0..100 {
            match rx.try_recv() {
                Ok(AgentEvent::TokenChunk(c)) => received.push_str(&c),
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert_eq!(received, answer, "chunked render must match run() answer");
    }
}
