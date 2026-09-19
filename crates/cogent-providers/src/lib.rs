//! Cogent LLM Provider crate。
//!
//! 实现 [`cogent_core::provider::LLMProvider`] 的具体接入：
//! - [`openai`]：OpenAI（兼容 OpenAI 协议的服务）。
//! - [`retry`]：基于 `backoff` 的传输层重试装饰器。
//!
//! 后续接入 Claude/Ollama 等：新增模块实现同一 trait，并在 CLI factory 注册。

pub mod openai;
pub mod retry;
