//! 落盘后校验模块。
//!
//! 在 edit/file_write 工具落盘后校验文件合法性，非法则回滚（实测：
//! 修改 JSON 后不验证合法性，可能产出非法 JSON；修改 Java 后不编译，
//! 可能产出编译错误）。
//!
//! # 校验项（按文件扩展名）
//! - `.json`：`serde_json` 解析，非法 → 回滚 + [`ValidateError::InvalidJson`]。
//! - `.java` 且 `java_check=true`：调用 `javac` 编译，失败 → 回滚 +
//!   [`ValidateError::CompileFailed`]。`javac` 不存在时跳过并警告（不报错）。
//! - 编码一致性：重新检测文件编码，与写入时编码不一致 → 回滚 +
//!   [`ValidateError::EncodingMismatch`]。
//!
//! # 回滚机制
//! 校验失败时用 `original_content`（落盘前内容）按 `expected_encoding` 写回，
//! 不留半改状态。
//!
//! # 设计说明
//! 本模块为纯函数（接收 `java_check` 参数，不直接读 config），由调用方
//! （`file.rs`，Task 7）从 `Config` 传入。`javac` 编译单文件可能因缺少依赖
//! （import 的其他类）失败，故 `java_check` 默认 `false`（opt-in）。

use std::path::Path;

use thiserror::Error;

use crate::encoding::{Encoding, EncodingError, detect_encoding, write_with_encoding};

/// 落盘后校验失败。
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidateError {
    /// JSON 文件解析失败（非法 JSON）。
    ///
    /// `line`/`col` 为解析错误位置（1-based）。
    #[error("invalid JSON at line {line}, column {col}")]
    InvalidJson {
        /// 解析错误行号（1-based）。
        line: usize,
        /// 解析错误列号（1-based）。
        col: usize,
    },

    /// Java 编译失败（`javac` 返回非零退出码）。
    ///
    /// `stderr` 为 `javac` 的标准错误输出（含编译错误详情）。
    #[error("java compilation failed: {stderr}")]
    CompileFailed {
        /// `javac` 标准错误输出。
        stderr: String,
    },

    /// 编码不一致（落盘后文件编码与写入时不同）。
    ///
    /// `expected` 为写入时编码，`actual` 为落盘后检测到的编码。
    #[error("encoding mismatch: expected {expected:?}, got {actual:?}")]
    EncodingMismatch {
        /// 写入时编码。
        expected: Encoding,
        /// 落盘后检测到的编码。
        actual: Encoding,
    },

    /// 文件读取或回滚写入失败（IO 错误）。
    ///
    /// `0` 为底层错误描述。
    #[error("io error during validation: {0}")]
    Io(String),
}

/// 落盘后校验文件合法性，失败时回滚。
///
/// # 参数
/// - `path`：文件路径。
/// - `original_content`：落盘前内容（回滚用）。
/// - `expected_encoding`：写入时编码（编码一致性校验 + 回滚编码）。
/// - `java_check`：是否对 `.java` 文件做编译检查（`true` 时调用 `javac`）。
///
/// # 返回
/// - 成功：`()`（校验通过，文件保留）。
/// - 失败：[`ValidateError`]（已回滚到 `original_content`）。
pub fn validate_after_write(
    path: &Path,
    original_content: &str,
    expected_encoding: Encoding,
    java_check: bool,
) -> Result<(), ValidateError> {
    // 1. 编码一致性校验
    let actual = detect_encoding(path).map_err(|e| ValidateError::Io(e.to_string()))?;
    if actual != expected_encoding {
        rollback(path, original_content, expected_encoding)?;
        return Err(ValidateError::EncodingMismatch {
            expected: expected_encoding,
            actual,
        });
    }

    // 2. 按扩展名校验
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();

    match ext {
        "json" => {
            let content =
                std::fs::read_to_string(path).map_err(|e| ValidateError::Io(e.to_string()))?;
            if let Err(e) = serde_json::from_str::<serde_json::Value>(&content) {
                let line = e.line();
                let col = e.column();
                rollback(path, original_content, expected_encoding)?;
                return Err(ValidateError::InvalidJson { line, col });
            }
        }
        "java" if java_check => {
            if let Some(stderr) = compile_java(path) {
                rollback(path, original_content, expected_encoding)?;
                return Err(ValidateError::CompileFailed { stderr });
            }
        }
        _ => {}
    }

    Ok(())
}

/// 回滚文件到 `original_content`（按 `expected_encoding` 写回）。
fn rollback(
    path: &Path,
    original_content: &str,
    expected_encoding: Encoding,
) -> Result<(), ValidateError> {
    write_with_encoding(path, original_content, expected_encoding)
        .map_err(|e: EncodingError| ValidateError::Io(e.to_string()))
}

/// 调用 `javac` 编译单个 `.java` 文件。
///
/// # 返回
/// - `None`：编译成功，或 `javac` 不存在（跳过并警告）。
/// - `Some(stderr)`：编译失败，`stderr` 为 `javac` 标准错误输出。
fn compile_java(path: &Path) -> Option<String> {
    // 用临时目录作为 .class 输出，避免污染源码目录
    let out_dir = std::env::temp_dir().join(format!(
        "cogent_javac_{}",
        path.file_name()?.to_string_lossy()
    ));
    let _ = std::fs::create_dir_all(&out_dir);

    let output = std::process::Command::new("javac")
        .arg("-d")
        .arg(&out_dir)
        .arg(path)
        .output();

    // 清理临时输出目录（best-effort）
    let _ = std::fs::remove_dir_all(&out_dir);

    match output {
        Ok(out) if out.status.success() => None,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            Some(stderr)
        }
        // javac 不存在或启动失败：跳过并警告（不报错）
        Err(e) => {
            tracing::warn!(
                error = %e,
                "javac not available, skipping java compilation check"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    //! `post-write-validate` 单元测试。
    //!
    //! 覆盖 JSON 合法/非法、编码一致/不一致、java_check 开关等成功标准。

    use super::*;

    /// 验证合法 JSON 文件校验通过。
    #[test]
    fn test_validate_json_valid() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.json");
        std::fs::write(&file, r#"{"key": "value"}"#).unwrap();
        let result = validate_after_write(&file, r#"{"key": "value"}"#, Encoding::Utf8, false);
        assert!(result.is_ok());
    }

    /// 验证非法 JSON 文件回滚 + 返回 `InvalidJson`。
    #[test]
    fn test_validate_json_invalid_rolls_back() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.json");
        // 落盘后内容是非法 JSON
        std::fs::write(&file, "{invalid json").unwrap();
        // original_content 是合法 JSON（回滚目标）
        let original = r#"{"key": "value"}"#;
        let result = validate_after_write(&file, original, Encoding::Utf8, false);
        assert!(matches!(result, Err(ValidateError::InvalidJson { .. })));
        // 验证已回滚到 original_content
        assert_eq!(std::fs::read_to_string(&file).unwrap(), original);
    }

    /// 验证编码一致时校验通过。
    #[test]
    fn test_validate_encoding_match() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello").unwrap();
        let result = validate_after_write(&file, "hello", Encoding::Utf8, false);
        assert!(result.is_ok());
    }

    /// 验证编码不一致时回滚 + 返回 `EncodingMismatch`。
    #[test]
    fn test_validate_encoding_mismatch_rolls_back() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        // 落盘后文件是 GBK（与 expected Utf8 不一致）
        let (cow, _, _) = encoding_rs::GBK.encode("中文");
        std::fs::write(&file, cow.as_ref()).unwrap();
        let original = "中文";
        let result = validate_after_write(&file, original, Encoding::Utf8, false);
        assert!(matches!(
            result,
            Err(ValidateError::EncodingMismatch { .. })
        ));
        // 验证已回滚（按 Utf8 写回）
        assert_eq!(std::fs::read_to_string(&file).unwrap(), original);
    }

    /// 验证 `java_check=false` 时跳过 Java 校验（即使 .java 文件）。
    #[test]
    fn test_validate_java_check_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.java");
        // 内容不是合法 Java，但 java_check=false 时不编译
        std::fs::write(&file, "not valid java").unwrap();
        let result = validate_after_write(&file, "not valid java", Encoding::Utf8, false);
        assert!(result.is_ok());
    }

    /// 验证 `java_check=true` 且 `javac` 不存在时跳过（不报错）。
    #[test]
    fn test_validate_java_check_no_javac_skips() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.java");
        std::fs::write(&file, "class A {}").unwrap();
        // 大多数 CI/开发环境无 javac，compile_java 返回 None（跳过）
        // 若有 javac 且 "class A {}" 编译成功，也返回 None
        let result = validate_after_write(&file, "class A {}", Encoding::Utf8, true);
        // 无论 javac 是否存在，"class A {}" 应通过（成功或跳过）
        assert!(result.is_ok());
    }

    /// 验证非 JSON/Java 文件跳过内容校验。
    #[test]
    fn test_validate_other_extension_skips() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "any content").unwrap();
        let result = validate_after_write(&file, "any content", Encoding::Utf8, false);
        assert!(result.is_ok());
    }
}
