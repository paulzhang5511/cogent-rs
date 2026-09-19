//! 框架全局配置。
//!
//! 本模块定义 [`Config`]——框架的全局配置结构，v1 仅从环境变量加载。
//! 配置在 CLI 启动时通过 [`Config::from_env`] 加载，传递给 factory 装配引擎。
//!
//! 环境变量：
//! - `COGENT_PROVIDER`：Provider 名称（默认 `openai`）。
//! - `COGENT_MODEL`：模型名称（默认 `gpt-4o-mini`）。
//! - `OPENAI_API_KEY`：OpenAI API Key（`provider == "openai"` 时必填）。
//! - `COGENT_BASE_URL`：自定义 API endpoint（可选）。
//! - `COGENT_MAX_ITERATIONS`：最大 ReAct 迭代次数（默认 20）。
//! - `COGENT_MAX_CONTEXT_TOKENS`：上下文窗口 token 预算（默认 8000）。
//! - `COGENT_OTEL_ENDPOINT`：OpenTelemetry OTLP endpoint（可选，设置后启用 OTLP 导出）。
//!
//! 设计原则：
//! - 配置加载失败（缺少关键项）返回明确错误，fail-fast。
//! - API Key 绝不写入日志或事件（安全边界）。
//! - `Default` 实现提供测试用默认值（不读真实环境）。

use crate::error::{CogentError, CogentResult};

/// 框架全局配置。
///
/// 所有引擎、Provider、Memory 的参数均从此结构读取。
/// 在 CLI 启动时通过 [`Config::from_env`] 加载，传递给 factory。
///
/// # 字段说明
/// - `provider`：LLM Provider 类型。
/// - `model`：模型名称（如 `"gpt-4o-mini"`）。
/// - `api_key`：API Key（OpenAI 必填）。
/// - `base_url`：自定义 API endpoint（可选）。
/// - `max_iterations`：最大 ReAct 迭代次数（防死循环的硬上限）。
/// - `max_context_tokens`：上下文窗口 token 预算。
/// - `event_bus_capacity`：事件总线缓冲区容量。
/// - `otel_endpoint`：OpenTelemetry OTLP endpoint（None 则仅本地 fmt）。
#[derive(Debug, Clone)]
pub struct Config {
    /// LLM Provider 名称（如 `"openai"`）。
    pub provider: String,
    /// 模型名称。
    pub model: String,
    /// API Key（OpenAI 必填，其他 Provider 可能不需要）。
    pub api_key: Option<String>,
    /// 自定义 API endpoint（兼容 OpenAI 兼容服务，如本地网关）。
    pub base_url: Option<String>,
    /// 最大 ReAct 迭代次数（防死循环硬上限，默认 20）。
    pub max_iterations: usize,
    /// 上下文窗口 token 预算（默认 8000）。
    pub max_context_tokens: usize,
    /// 事件总线缓冲区容量（默认 256）。
    pub event_bus_capacity: usize,
    /// OpenTelemetry OTLP endpoint（None 则仅本地 fmt 输出）。
    pub otel_endpoint: Option<String>,
    /// 非交互模式下 edit 是否自动批准（默认 `true`，diff 记录到日志）。
    ///
    /// `false` 时 edit 拒绝落盘，diff 回传给模型让其重新生成。
    pub auto_approve: bool,
    /// 落盘后是否对 `.java` 文件做编译检查（默认 `false`，opt-in）。
    ///
    /// `true` 时调用 `javac` 编译，失败则回滚；`javac` 不存在时跳过并警告。
    pub java_check: bool,
}

impl Config {
    /// 从环境变量加载配置。
    ///
    /// # 环境变量
    /// - `COGENT_PROVIDER`：Provider 名称（默认 `openai`）。
    /// - `COGENT_MODEL`：模型名称（默认 `gpt-4o-mini`）。
    /// - `OPENAI_API_KEY`：OpenAI API Key（`provider == "openai"` 时 **必填**）。
    /// - `COGENT_BASE_URL`：自定义 API endpoint（可选）。
    /// - `COGENT_MAX_ITERATIONS`：最大迭代次数（默认 20）。
    /// - `COGENT_MAX_CONTEXT_TOKENS`：上下文 token 预算（默认 8000）。
    /// - `COGENT_OTEL_ENDPOINT`：OTel OTLP endpoint（可选）。
    ///
    /// # 返回
    /// - 成功：加载的 `Config`。
    /// - 失败：`CogentError::Config`（缺少必填项或值格式错误）。
    ///
    /// # 安全
    /// API Key 仅保存在内存中，绝不写入日志或事件。
    pub fn from_env() -> CogentResult<Self> {
        // Provider 名称（默认 openai）
        let provider = std::env::var("COGENT_PROVIDER").unwrap_or_else(|_| "openai".to_string());

        // 模型名称（默认 gpt-4o-mini）
        let model = std::env::var("COGENT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        // API Key（仅 openai 需要）
        let api_key = if provider.eq_ignore_ascii_case("openai") {
            let key = std::env::var("OPENAI_API_KEY")
                .map_err(|_| CogentError::Config("missing required env: OPENAI_API_KEY".into()))?;
            if key.is_empty() {
                return Err(CogentError::Config(
                    "OPENAI_API_KEY is set but empty".into(),
                ));
            }
            Some(key)
        } else {
            None
        };

        // 自定义 base URL（可选）
        let base_url = std::env::var("COGENT_BASE_URL").ok();

        // 最大迭代次数（默认 20）
        let max_iterations = std::env::var("COGENT_MAX_ITERATIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);

        // 上下文 token 预算（默认 8000）
        let max_context_tokens = std::env::var("COGENT_MAX_CONTEXT_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8000);

        // OTel endpoint（可选）
        let otel_endpoint = std::env::var("COGENT_OTEL_ENDPOINT").ok();

        // 非交互模式 edit 自动批准（默认 true）
        let auto_approve = std::env::var("COGENT_AUTO_APPROVE")
            .ok()
            .map(|s| s != "false" && s != "0")
            .unwrap_or(true);

        // 落盘后 Java 编译检查（默认 false，opt-in）
        let java_check = std::env::var("COGENT_JAVA_CHECK")
            .ok()
            .map(|s| s == "true" || s == "1")
            .unwrap_or(false);

        tracing::info!(
            provider = %provider,
            model = %model,
            max_iterations,
            max_context_tokens,
            otel_enabled = otel_endpoint.is_some(),
            "config loaded from environment"
        );

        Ok(Self {
            provider,
            model,
            api_key,
            base_url,
            max_iterations,
            max_context_tokens,
            event_bus_capacity: 256,
            otel_endpoint,
            auto_approve,
            java_check,
        })
    }
}

impl Default for Config {
    /// 测试用默认配置（不读真实环境）。
    ///
    /// 单元测试和 Mock 场景直接使用 `Config::default()` 构造，
    /// 避免依赖真实环境变量。
    fn default() -> Self {
        Self {
            provider: "openai".into(),
            model: "gpt-4o-mini".into(),
            api_key: None,
            base_url: None,
            max_iterations: 20,
            max_context_tokens: 8000,
            event_bus_capacity: 256,
            otel_endpoint: None,
            auto_approve: true,
            java_check: false,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Config 单元测试。
    //!
    /// 验证 Config::default、from_env 错误处理。
    use super::*;
    use std::sync::Mutex;

    /// 串行化所有依赖环境变量的测试，避免并行竞争。
    ///
    /// edition 2024 中 `set_var`/`remove_var` 是 unsafe 且影响进程全局状态，
    /// 多个测试并行修改同一环境变量会互相干扰（如一个测试移除 `OPENAI_API_KEY`
    /// 时，另一个测试正依赖它）。用 Mutex 强制这些测试串行执行。
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// 验证 Config::default 提供合理默认值。
    #[test]
    fn test_config_default() {
        let cfg = Config::default();
        assert_eq!(cfg.provider, "openai");
        assert_eq!(cfg.model, "gpt-4o-mini");
        assert!(cfg.api_key.is_none());
        assert_eq!(cfg.max_iterations, 20);
        assert_eq!(cfg.max_context_tokens, 8000);
        assert_eq!(cfg.event_bus_capacity, 256);
        assert!(cfg.otel_endpoint.is_none());
    }

    /// 验证 from_env 在缺少 OPENAI_API_KEY 时返回错误。
    ///
    /// 注意：edition 2024 中 `set_var`/`remove_var` 是 unsafe（可能影响其他线程）。
    /// 测试中用 unsafe 块包裹，且测试间环境变量隔离。
    /// 显式固定 `COGENT_PROVIDER=openai`，避免外部环境干扰断言。
    #[test]
    fn test_from_env_missing_api_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        // 保存原始值，固定 provider 为 openai 并移除 API key
        let original = [
            std::env::var("OPENAI_API_KEY").ok(),
            std::env::var("COGENT_PROVIDER").ok(),
        ];
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::set_var("COGENT_PROVIDER", "openai");
        }

        let result = Config::from_env();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, CogentError::Config(_)));
        assert!(err.to_string().contains("OPENAI_API_KEY"));

        // 恢复原始环境
        let names = ["OPENAI_API_KEY", "COGENT_PROVIDER"];
        for (name, orig) in names.iter().zip(original) {
            if let Some(v) = orig {
                unsafe { std::env::set_var(name, v) };
            } else {
                unsafe { std::env::remove_var(name) };
            }
        }
    }

    /// 验证 from_env 在设置所有环境变量时成功加载。
    #[test]
    fn test_from_env_success() {
        let _guard = ENV_LOCK.lock().unwrap();
        // 保存原始值，设置测试环境变量（显式固定 provider，避免外部环境干扰）
        let original = [
            std::env::var("COGENT_PROVIDER").ok(),
            std::env::var("COGENT_MODEL").ok(),
            std::env::var("OPENAI_API_KEY").ok(),
            std::env::var("COGENT_MAX_ITERATIONS").ok(),
            std::env::var("COGENT_MAX_CONTEXT_TOKENS").ok(),
        ];
        unsafe {
            std::env::set_var("COGENT_PROVIDER", "openai");
            std::env::set_var("COGENT_MODEL", "test-model");
            std::env::set_var("OPENAI_API_KEY", "test-key-12345");
            std::env::set_var("COGENT_MAX_ITERATIONS", "5");
            std::env::set_var("COGENT_MAX_CONTEXT_TOKENS", "4000");
        }

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.provider, "openai");
        assert_eq!(cfg.model, "test-model");
        assert_eq!(cfg.api_key.as_deref(), Some("test-key-12345"));
        assert_eq!(cfg.max_iterations, 5);
        assert_eq!(cfg.max_context_tokens, 4000);

        // 恢复原始环境
        let names = [
            "COGENT_PROVIDER",
            "COGENT_MODEL",
            "OPENAI_API_KEY",
            "COGENT_MAX_ITERATIONS",
            "COGENT_MAX_CONTEXT_TOKENS",
        ];
        for (name, orig) in names.iter().zip(original) {
            if let Some(v) = orig {
                unsafe { std::env::set_var(name, v) };
            } else {
                unsafe { std::env::remove_var(name) };
            }
        }
    }

    /// 验证 from_env 解析无效的 max_iterations 时回退默认值。
    #[test]
    fn test_from_env_invalid_max_iterations_falls_back() {
        let _guard = ENV_LOCK.lock().unwrap();
        let original = [
            std::env::var("COGENT_PROVIDER").ok(),
            std::env::var("OPENAI_API_KEY").ok(),
            std::env::var("COGENT_MAX_ITERATIONS").ok(),
        ];
        unsafe {
            std::env::set_var("COGENT_PROVIDER", "openai");
            std::env::set_var("OPENAI_API_KEY", "test-key");
            std::env::set_var("COGENT_MAX_ITERATIONS", "not-a-number");
        }

        let cfg = Config::from_env().unwrap();
        // 无效值回退默认 20
        assert_eq!(cfg.max_iterations, 20);

        // 恢复原始环境
        let names = ["COGENT_PROVIDER", "OPENAI_API_KEY", "COGENT_MAX_ITERATIONS"];
        for (name, orig) in names.iter().zip(original) {
            if let Some(v) = orig {
                unsafe { std::env::set_var(name, v) };
            } else {
                unsafe { std::env::remove_var(name) };
            }
        }
    }

    /// 验证 from_env 在 OPENAI_API_KEY 为空字符串时返回错误。
    #[test]
    fn test_from_env_empty_api_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        let original = [
            std::env::var("COGENT_PROVIDER").ok(),
            std::env::var("OPENAI_API_KEY").ok(),
        ];
        unsafe {
            std::env::set_var("COGENT_PROVIDER", "openai");
            std::env::set_var("OPENAI_API_KEY", "");
        }

        let result = Config::from_env();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, CogentError::Config(_)));
        assert!(err.to_string().contains("empty"));

        // 恢复原始环境
        let names = ["COGENT_PROVIDER", "OPENAI_API_KEY"];
        for (name, orig) in names.iter().zip(original) {
            if let Some(v) = orig {
                unsafe { std::env::set_var(name, v) };
            } else {
                unsafe { std::env::remove_var(name) };
            }
        }
    }
}
