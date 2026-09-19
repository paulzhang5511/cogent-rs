//! 传输层重试装饰器。
//!
//! 本模块实现 [`RetryMiddleware`]——基于装饰器模式包装任意 [`LLMProvider`]，
//! 在遇到**可重试错误**（429 / 5xx / 网络超时 / 连接失败）时按指数退避自动重试。
//!
//! # 与 `AgentMiddleware` 的区别
//! - `RetryMiddleware` 实现 [`LLMProvider`]（传输层重试），由 CLI factory 套在真实
//!   provider 外层，对引擎透明。
//! - `AgentMiddleware`（Tracing/SafetyGuard/TokenLimiter）实现引擎中间件 trait
//!   （业务层拦截），由引擎 ReAct 循环驱动。
//!
//! 两者互补：Provider 级 Retry 处理传输层瞬时故障，引擎级 SelfCorrecting 处理业务层。
//!
//! # 设计原则
//! - 不可重试错误（4xx 业务错误、参数错误等）直接抛出，不重试。
//! - 所有 `tracing` 日志用英文，携带结构化字段（`attempt`、`max_retries`、`delay_ms`）。
//! - 重试次数与退避参数可配置，默认 3 次。

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use backoff::backoff::Backoff;
use backoff::{ExponentialBackoff, ExponentialBackoffBuilder};
use futures::{Stream, StreamExt};

use cogent_core::provider::{LLMProvider, LLMResponse};
use cogent_core::tool::Tool;
use cogent_core::types::Message;

/// 默认最大重试次数。
const DEFAULT_MAX_RETRIES: u32 = 3;

/// 初始退避间隔（毫秒）。
const INITIAL_INTERVAL_MS: u64 = 500;

/// 最大单次退避间隔（秒）。
const MAX_INTERVAL_SECS: u64 = 10;

/// 退避总时长上限（秒），防止无限等待。
const MAX_TOTAL_TIME_SECS: u64 = 60;

/// 从错误消息中提取 HTTP 状态码。
///
/// 查找 "status" 关键字后紧跟的 3 位数字（如 "status 429"、"(status 503)"）。
/// 返回 `Some(code)` 或 `None`（未找到）。
///
/// # 参数
/// - `msg`：已转小写的错误消息。
fn extract_status_code(msg: &str) -> Option<u16> {
    let idx = msg.find("status")?;
    let after = &msg[idx + "status".len()..];
    // 跳过 "status" 后的空白字符
    let digits: String = after
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.len() == 3 {
        digits.parse().ok()
    } else {
        None
    }
}

/// 判断错误是否为可重试的瞬时传输层错误。
///
/// 可重试错误包括：
/// - HTTP 429（限流）。
/// - HTTP 5xx（服务端错误）。
/// - 网络层错误（连接失败、超时、DNS 解析失败等）。
///
/// 不可重试错误（如 400 参数错误、401 认证失败）直接返回，避免无谓重试。
///
/// # 参数
/// - `err`：待判断的错误。
///
/// # 返回
/// - `true`：可重试（瞬时故障）。
/// - `false`：不可重试（业务/参数错误）。
fn is_retryable(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    let lower = msg.to_lowercase();

    // 网络层错误：连接失败、超时、DNS 等
    if lower.contains("http request failed")
        || lower.contains("timeout")
        || lower.contains("connection")
        || lower.contains("dns")
    {
        return true;
    }

    // API 状态码错误：从错误消息中提取 "status <code>" 中的状态码
    // 手动解析（避免引入正则依赖），兼容 "status 429"、"(status 503)" 等格式
    if let Some(code) = extract_status_code(&lower) {
        // 429 限流或 5xx 服务端错误可重试
        return code == 429 || (500..600).contains(&code);
    }

    false
}

/// 构建指数退避策略。
///
/// 初始间隔 500ms，每次翻倍，单次上限 10s，总时长上限 60s。
fn build_backoff() -> ExponentialBackoff {
    ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_millis(INITIAL_INTERVAL_MS))
        .with_max_interval(Duration::from_secs(MAX_INTERVAL_SECS))
        .with_max_elapsed_time(Some(Duration::from_secs(MAX_TOTAL_TIME_SECS)))
        .with_multiplier(2.0)
        .build()
}

/// 传输层重试装饰器。
///
/// 包装任意 [`LLMProvider`]，在可重试错误发生时按指数退避自动重试。
/// 通过 [`RetryMiddleware::new`] 构造，`max_retries` 为最大重试次数
/// （不含首次调用）。
///
/// # 用法
/// ```rust,ignore
/// let openai = Arc::new(OpenAIProvider::new(key, model, None)?);
/// let provider: Arc<dyn LLMProvider> = Arc::new(RetryMiddleware::new(openai, 3));
/// ```
pub struct RetryMiddleware {
    /// 被包装的内部 provider。
    inner: Arc<dyn LLMProvider>,
    /// 最大重试次数（不含首次调用）。
    max_retries: u32,
}

impl RetryMiddleware {
    /// 创建重试装饰器。
    ///
    /// # 参数
    /// - `inner`：被包装的 provider。
    /// - `max_retries`：最大重试次数（不含首次调用）。
    pub fn new(inner: Arc<dyn LLMProvider>, max_retries: u32) -> Self {
        tracing::info!(max_retries, "RetryMiddleware initialized");
        Self { inner, max_retries }
    }

    /// 创建使用默认重试次数（3 次）的装饰器。
    pub fn with_default_retries(inner: Arc<dyn LLMProvider>) -> Self {
        Self::new(inner, DEFAULT_MAX_RETRIES)
    }
}

#[async_trait]
impl LLMProvider for RetryMiddleware {
    async fn chat_complete(
        &self,
        messages: &[Message],
        tools: &[Arc<dyn Tool>],
    ) -> anyhow::Result<LLMResponse> {
        let mut backoff = build_backoff();
        let mut attempt: u32 = 0;

        loop {
            match self.inner.chat_complete(messages, tools).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    // 不可重试错误或重试次数耗尽：直接返回错误
                    if !is_retryable(&e) || attempt >= self.max_retries {
                        tracing::error!(
                            attempt,
                            max_retries = self.max_retries,
                            retryable = is_retryable(&e),
                            error = %e,
                            "chat_complete failed, no more retries"
                        );
                        return Err(e);
                    }

                    attempt += 1;
                    let delay = backoff
                        .next_backoff()
                        .unwrap_or(Duration::from_secs(MAX_INTERVAL_SECS));
                    tracing::warn!(
                        attempt,
                        max_retries = self.max_retries,
                        delay_ms = delay.as_millis() as u64,
                        error = %e,
                        "chat_complete transient error, retrying"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[Arc<dyn Tool>],
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>> {
        // 流式调用是惰性的：HTTP 请求在首次 poll 时才发起。
        // 因此重试逻辑必须在 stream 内部实现——poll 到首个错误时判断是否重试，
        // 重试则丢弃当前 stream 并重新建立。
        //
        // 注意：messages/tools 是借用，而返回的 stream 是 'static + Send，
        // 因此需先克隆为 owned 数据移入 stream。
        let messages_owned: Vec<Message> = messages.to_vec();
        let tools_owned: Vec<Arc<dyn Tool>> = tools.to_vec();
        let inner = self.inner.clone();
        let max_retries = self.max_retries;

        Box::pin(async_stream::stream! {
            let mut backoff = build_backoff();
            let mut attempt: u32 = 0;

            loop {
                let mut stream = inner.chat_stream(&messages_owned, &tools_owned);

                // poll 首个元素：成功则转发全部，失败则判断是否重试
                match stream.next().await {
                    Some(Ok(first)) => {
                        yield Ok(first);
                        // 转发剩余元素
                        while let Some(item) = stream.next().await {
                            yield item;
                        }
                        return;
                    }
                    Some(Err(e)) => {
                        if is_retryable(&e) && attempt < max_retries {
                            attempt += 1;
                            let delay = backoff
                                .next_backoff()
                                .unwrap_or(Duration::from_secs(MAX_INTERVAL_SECS));
                            tracing::warn!(
                                attempt,
                                max_retries,
                                delay_ms = delay.as_millis() as u64,
                                error = %e,
                                "chat_stream transient error, retrying"
                            );
                            tokio::time::sleep(delay).await;
                            continue;
                        }
                        tracing::error!(
                            attempt,
                            max_retries,
                            retryable = is_retryable(&e),
                            error = %e,
                            "chat_stream failed, no more retries"
                        );
                        yield Err(e);
                        return;
                    }
                    None => {
                        // 空流（无数据），直接结束
                        return;
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    //! RetryMiddleware 单元测试。
    //!
    //! 验证可重试错误判断、重试成功/失败路径、重试次数上限。
    //! 使用 Mock provider 模拟瞬时故障，不发起真实网络请求。

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 模拟瞬时故障的 provider：前 `fail_times` 次调用返回可重试错误，
    /// 之后返回成功。用于验证重试逻辑。
    struct FlakyProvider {
        /// 已调用次数（原子计数，跨重试共享）。
        calls: Arc<AtomicUsize>,
        /// 需要失败的次数。
        fail_times: usize,
        /// 错误类型：`true` 为可重试（429），`false` 为不可重试（400）。
        retryable: bool,
    }

    impl FlakyProvider {
        fn new(calls: Arc<AtomicUsize>, fail_times: usize, retryable: bool) -> Self {
            Self {
                calls,
                fail_times,
                retryable,
            }
        }
    }

    #[async_trait]
    impl LLMProvider for FlakyProvider {
        async fn chat_complete(
            &self,
            _messages: &[Message],
            _tools: &[Arc<dyn Tool>],
        ) -> anyhow::Result<LLMResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_times {
                if self.retryable {
                    Err(anyhow::anyhow!(
                        "OpenAI API error (status 429): rate limited"
                    ))
                } else {
                    Err(anyhow::anyhow!(
                        "OpenAI API error (status 400): bad request"
                    ))
                }
            } else {
                Ok(LLMResponse::text("ok".into(), 1, 1))
            }
        }

        fn chat_stream(
            &self,
            _messages: &[Message],
            _tools: &[Arc<dyn Tool>],
        ) -> Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>> {
            Box::pin(futures::stream::empty())
        }
    }

    /// 验证 `is_retryable` 对 429 返回 true。
    #[test]
    fn test_is_retryable_429() {
        let err = anyhow::anyhow!("OpenAI API error (status 429): rate limited");
        assert!(is_retryable(&err));
    }

    /// 验证 `is_retryable` 对 5xx 返回 true。
    #[test]
    fn test_is_retryable_5xx() {
        for code in [500, 502, 503, 504] {
            let err = anyhow::anyhow!("OpenAI API error (status {code}): server error");
            assert!(is_retryable(&err), "expected {code} to be retryable");
        }
    }

    /// 验证 `is_retryable` 对 400 返回 false。
    #[test]
    fn test_is_retryable_400() {
        let err = anyhow::anyhow!("OpenAI API error (status 400): bad request");
        assert!(!is_retryable(&err));
    }

    /// 验证 `is_retryable` 对 401 返回 false。
    #[test]
    fn test_is_retryable_401() {
        let err = anyhow::anyhow!("OpenAI API error (status 401): unauthorized");
        assert!(!is_retryable(&err));
    }

    /// 验证 `is_retryable` 对网络超时返回 true。
    #[test]
    fn test_is_retryable_timeout() {
        let err = anyhow::anyhow!("HTTP request failed: operation timed out");
        assert!(is_retryable(&err));
    }

    /// 验证 `is_retryable` 对连接失败返回 true。
    #[test]
    fn test_is_retryable_connection() {
        let err = anyhow::anyhow!("HTTP request failed: connection refused");
        assert!(is_retryable(&err));
    }

    /// 验证可重试错误在重试后成功。
    #[tokio::test]
    async fn test_chat_complete_retries_then_succeeds() {
        let calls = Arc::new(AtomicUsize::new(0));
        // 前 2 次失败（429），第 3 次成功
        let flaky = Arc::new(FlakyProvider::new(calls.clone(), 2, true));
        let mw = RetryMiddleware::new(flaky, 3);

        let resp = mw
            .chat_complete(&[], &[])
            .await
            .expect("should succeed after retries");
        assert_eq!(resp.content, "ok");
        // 共调用 3 次（2 次失败 + 1 次成功）
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// 验证不可重试错误立即返回（不重试）。
    #[tokio::test]
    async fn test_chat_complete_non_retryable_fails_immediately() {
        let calls = Arc::new(AtomicUsize::new(0));
        // 总是返回 400（不可重试）
        let flaky = Arc::new(FlakyProvider::new(calls.clone(), 100, false));
        let mw = RetryMiddleware::new(flaky, 3);

        let result = mw.chat_complete(&[], &[]).await;
        assert!(result.is_err());
        // 仅调用 1 次（不重试）
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// 验证重试次数耗尽后返回错误。
    #[tokio::test]
    async fn test_chat_complete_exhausts_retries() {
        let calls = Arc::new(AtomicUsize::new(0));
        // 总是返回 429（可重试），但 max_retries=2
        let flaky = Arc::new(FlakyProvider::new(calls.clone(), 100, true));
        let mw = RetryMiddleware::new(flaky, 2);

        let result = mw.chat_complete(&[], &[]).await;
        assert!(result.is_err());
        // 共调用 3 次（1 次初始 + 2 次重试）
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// 验证 `with_default_retries` 使用默认 3 次。
    #[tokio::test]
    async fn test_with_default_retries() {
        let calls = Arc::new(AtomicUsize::new(0));
        // 总是返回 429（可重试）
        let flaky = Arc::new(FlakyProvider::new(calls.clone(), 100, true));
        let mw = RetryMiddleware::with_default_retries(flaky);

        let result = mw.chat_complete(&[], &[]).await;
        assert!(result.is_err());
        // 共调用 4 次（1 次初始 + 3 次默认重试）
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }

    /// 验证 `build_backoff` 产生合法的退避间隔，且无 jitter 时指数递增。
    #[test]
    fn test_build_backoff_increases() {
        // 生产退避带随机 jitter（backoff crate 默认 randomization_factor=0.5），
        // 单次采样可能 d2 略小于 d1，故只断言范围合法（0 < d <= max_interval）。
        let mut backoff = build_backoff();
        let d1 = backoff.next_backoff().unwrap();
        let d2 = backoff.next_backoff().unwrap();
        let max = Duration::from_secs(MAX_INTERVAL_SECS);
        assert!(d1 > Duration::ZERO && d1 <= max);
        assert!(d2 > Duration::ZERO && d2 <= max);

        // 无 jitter（randomization_factor=0）时验证指数递增语义（multiplier=2.0）
        let mut plain = ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_millis(INITIAL_INTERVAL_MS))
            .with_max_interval(max)
            .with_max_elapsed_time(Some(Duration::from_secs(MAX_TOTAL_TIME_SECS)))
            .with_multiplier(2.0)
            .with_randomization_factor(0.0)
            .build();
        let p1 = plain.next_backoff().unwrap();
        let p2 = plain.next_backoff().unwrap();
        assert!(p2 >= p1, "指数退避第二次间隔应大于等于第一次");
    }
}
