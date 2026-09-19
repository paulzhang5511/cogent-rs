//! 文件 IO 工具：读取与写入文件。
//!
//! 通过 `#[cogent_tool]` 宏声明，提供 `file_read` 与 `file_write` 两个工具。
//!
//! # 安全约束（Path Traversal 防护）
//! - 所有路径先 `canonicalize`（解析 `..`、符号链接等），再校验位于
//!   允许根目录（进程当前工作目录）内，否则拒绝访问。
//! - 拒绝越界路径（如 `../../etc/passwd`）。
//!
//! # 测试策略
//! 核心逻辑（`validate_path`/`read_file`/`write_file`）接收 `root` 参数，
//! 便于用 `tempfile` 隔离测试；`#[cogent_tool]` 函数为薄封装，以 CWD 为 root。

use std::path::{Path, PathBuf};

use cogent_macros::cogent_tool;

use crate::diff_preview::{confirm_non_interactive, generate_diff};
use crate::encoding::{Encoding, read_with_encoding, write_with_encoding};
use crate::error_recovery::format_not_found_error;
use crate::sanitize::{sanitize_edit_content, sanitize_write_content};
use crate::validate::validate_after_write;

/// 可靠性配置（从环境变量读取，由 `#[cogent_tool]` 薄封装传入核心函数）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReliabilityConfig {
    /// 非交互模式 edit 是否自动批准（默认 `true`）。
    auto_approve: bool,
    /// 落盘后是否对 `.java` 文件做编译检查（默认 `false`）。
    java_check: bool,
}

impl ReliabilityConfig {
    /// 从环境变量读取（`COGENT_AUTO_APPROVE` / `COGENT_JAVA_CHECK`）。
    fn from_env() -> Self {
        Self {
            auto_approve: std::env::var("COGENT_AUTO_APPROVE")
                .ok()
                .map(|s| s != "false" && s != "0")
                .unwrap_or(true),
            java_check: std::env::var("COGENT_JAVA_CHECK")
                .ok()
                .map(|s| s == "true" || s == "1")
                .unwrap_or(false),
        }
    }
}

impl Default for ReliabilityConfig {
    /// 测试用默认配置（自动批准、不做 Java 编译检查）。
    fn default() -> Self {
        Self {
            auto_approve: true,
            java_check: false,
        }
    }
}

/// 校验路径是否位于允许根目录内（Path Traversal 防护）。
///
/// # 参数
/// - `path`：待校验的路径（原始输入，可能含 `..`）。
/// - `root`：允许访问的根目录。
///
/// # 返回
/// - 成功：`canonicalize` 后的绝对路径（已确认在 `root` 内）。
/// - 失败：路径越界或父目录不存在。
///
/// # 注意
/// 若目标文件不存在（如 `file_write` 的新文件），则 `canonicalize` 其父目录
/// 后拼接文件名，避免对不存在路径的 `canonicalize` 失败。
///
/// # TOCTOU 说明
/// 本校验在 `canonicalize` 时解析符号链接与 `..`，但 `canonicalize` 与
/// 后续 `read`/`write` 之间存在时间窗口（TOCTOU）：攻击者可在此窗口内
/// 将路径组件替换为符号链接指向根目录外的目标。v1 单用户本地场景下风险可接受；
/// 多用户/不可信环境应改用 `O_NOFOLLOW` 逐段打开或 `openat2`（Linux 5.6+）
/// 的 `RESOLVE_BENEATH` 标志消除竞态。
fn validate_path(path: &str, root: &Path) -> anyhow::Result<PathBuf> {
    let raw = Path::new(path);

    // 若路径存在则直接 canonicalize；否则 canonicalize 父目录后拼接文件名
    let canonical = if raw.exists() {
        raw.canonicalize()
            .map_err(|e| anyhow::anyhow!("invalid path '{path}': {e}"))?
    } else {
        // 父目录须存在（否则无法确定真实路径）
        let parent = raw
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .ok_or_else(|| anyhow::anyhow!("invalid path '{path}': no parent directory"))?;
        let canonical_parent = parent
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("parent directory of '{path}' does not exist: {e}"))?;
        let file_name = raw
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("invalid path '{path}': no file name"))?;
        canonical_parent.join(file_name)
    };

    // 校验位于允许根目录内
    let canonical_root = root
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("failed to canonicalize root: {e}"))?;

    if !canonical.starts_with(&canonical_root) {
        anyhow::bail!(
            "path '{path}' is outside the allowed root directory '{}'",
            canonical_root.display()
        );
    }

    Ok(canonical)
}

/// 读取文件内容（核心逻辑，接收 root 参数便于测试）。
///
/// # 参数
/// - `path`：文件路径（相对或绝对，须位于 `root` 内）。
/// - `root`：允许访问的根目录。
///
/// # 返回
/// - 成功：文件内容文本。
/// - 失败：路径越界、文件不存在或读取错误。
async fn read_file(path: &str, root: &Path) -> anyhow::Result<String> {
    let validated = validate_path(path, root)?;

    tracing::info!(path = %validated.display(), "reading file");

    let content = tokio::fs::read_to_string(&validated)
        .await
        .map_err(|e| anyhow::anyhow!("failed to read file '{}': {e}", validated.display()))?;

    tracing::info!(
        path = %validated.display(),
        content_len = content.len(),
        "file read completed"
    );

    Ok(content)
}

/// 写入内容到文件（核心逻辑，接收 root 参数便于测试）。
///
/// 可靠性增强：
/// - **编码保持**：已存在文件按原编码写入（避免 UTF-8/GBK 转换损坏）。
/// - **diff 预览**：已存在文件落盘前生成 unified diff，非交互按 `auto_approve`
///   确认（`false` 时拒绝落盘，diff 回传给模型）。
/// - **落盘后校验**：JSON 合法性 / Java 编译（`java_check`）/ 编码一致性，
///   失败回滚到原内容。
///
/// # 参数
/// - `path`：文件路径（相对或绝对，须位于 `root` 内）。
/// - `content`：要写入的文件内容。
/// - `root`：允许访问的根目录。
/// - `cfg`：可靠性配置（`auto_approve` / `java_check`）。
///
/// # 返回
/// - 成功：确认写入的字节数文本。
/// - 失败：路径越界、diff 被拒、校验失败（已回滚）或写入错误。
async fn write_file(
    path: &str,
    content: &str,
    root: &Path,
    cfg: ReliabilityConfig,
) -> anyhow::Result<String> {
    let validated = validate_path(path, root)?;

    // 输出净化：模型常改用 file_write 整文件重写，重写内容同样带前导空行 /
    // U+FFFB / 注释粘连，落盘前净化（shadow content 为净化后内容）。
    let content = sanitize_write_content(content)
        .map_err(|e| anyhow::anyhow!("write content rejected by sanitizer: {e}"))?;

    // 已存在文件：读取原内容（编码保持 + diff 预览 + 回滚基准）
    let (original, encoding) = if validated.exists() {
        let (text, enc) = read_with_encoding(&validated)
            .map_err(|e| anyhow::anyhow!("failed to read file '{}': {e}", validated.display()))?;
        (Some(text), enc)
    } else {
        (None, Encoding::Utf8)
    };

    // diff 预览（仅已存在文件；新文件无原内容可比对）
    if let Some(orig) = &original {
        let diff = generate_diff(&validated.display().to_string(), orig, &content);
        let confirm =
            confirm_non_interactive(&validated.display().to_string(), &diff, cfg.auto_approve);
        if !confirm.approved {
            anyhow::bail!(
                "edit rejected (auto_approve=false). Proposed diff:\n{}",
                confirm.diff
            );
        }
    }

    tracing::info!(
        path = %validated.display(),
        content_len = content.len(),
        encoding = ?encoding,
        "writing file"
    );

    // 按原编码写入（新文件用 UTF-8）
    write_with_encoding(&validated, &content, encoding)
        .map_err(|e| anyhow::anyhow!("failed to write file '{}': {e}", validated.display()))?;

    // 落盘后校验（失败回滚到原内容）
    if let Some(orig) = &original {
        validate_after_write(&validated, orig, encoding, cfg.java_check).map_err(|e| {
            anyhow::anyhow!(
                "post-write validation failed for '{}': {e}",
                validated.display()
            )
        })?;
    }

    tracing::info!(path = %validated.display(), "file write completed");

    Ok(format!(
        "wrote {} bytes to {}",
        content.len(),
        validated.display()
    ))
}

/// 读取文件内容并返回。
///
/// # 参数
/// - `path`：文件路径（相对或绝对，须位于当前工作目录内）。
///
/// # 返回
/// - 成功：文件内容文本。
/// - 失败：路径越界、文件不存在或读取错误。
#[cogent_tool]
async fn file_read(path: String) -> anyhow::Result<String> {
    let root = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current working directory: {e}"))?;
    read_file(&path, &root).await
}

/// 写入内容到文件。
///
/// # 参数
/// - `path`：文件路径（相对或绝对，须位于当前工作目录内）。
/// - `content`：要写入的文件内容。
///
/// # 返回
/// - 成功：确认写入的字节数。
/// - 失败：路径越界或写入错误。
#[cogent_tool]
async fn file_write(path: String, content: String) -> anyhow::Result<String> {
    let root = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current working directory: {e}"))?;
    write_file(&path, &content, &root, ReliabilityConfig::from_env()).await
}

/// 精确替换文件中的文本片段（核心逻辑，接收 root 参数便于测试）。
///
/// 与 `write_file`（整文件覆盖）不同，本函数只做**精确字符串替换**，
/// 避免模型为改一行而重写整个文件（实测 log：模型用 `python -c` 经 bash
/// 改 Java，引号转义在 PowerShell 下反复失败，耗尽迭代）。
///
/// 可靠性增强：
/// - **输出净化**：`new_string` 去前导空行 / 检测 U+FFFB / 检测注释粘连，
///   非法则拒绝落盘（`old_string` 原样保留用于匹配）。
/// - **错误恢复**：`old_string` 未找到时返回最接近匹配（±3 行上下文），
///   帮助模型定位差异。
/// - **编码保持**：按原编码读写（避免 UTF-8/GBK 转换损坏）。
/// - **diff 预览**：落盘前生成 unified diff，非交互按 `auto_approve` 确认。
/// - **落盘后校验**：JSON / Java 编译（`java_check`）/ 编码一致性，失败回滚。
///
/// # 参数
/// - `path`：文件路径（相对或绝对，须位于 `root` 内）。
/// - `old_string`：要被替换的原文（须与文件内容逐字节一致，含缩进）。
/// - `new_string`：替换后的新文本。
/// - `replace_all`：`true` 替换全部出现；`false` 仅替换唯一一处。
/// - `root`：允许访问的根目录。
/// - `cfg`：可靠性配置（`auto_approve` / `java_check`）。
///
/// # 返回
/// - 成功：替换次数确认文本。
/// - 失败：路径越界、文件不存在、`old_string` 未找到（0 次）、
///   或 `replace_all=false` 时 `old_string` 出现多次（歧义，须提供更多上下文）。
async fn edit_file(
    path: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
    root: &Path,
    cfg: ReliabilityConfig,
) -> anyhow::Result<String> {
    if old_string.is_empty() {
        anyhow::bail!("old_string must not be empty");
    }

    // 输出净化：仅净化 new_string（old_string 原样保留用于匹配）
    let (_old, sanitized_new) = sanitize_edit_content(old_string, new_string)
        .map_err(|e| anyhow::anyhow!("edit content rejected by sanitizer: {e}"))?;
    if old_string == sanitized_new {
        anyhow::bail!("old_string and new_string must differ");
    }

    let validated = validate_path(path, root)?;

    // 按原编码读取（编码保持）
    let (content, encoding) = read_with_encoding(&validated)
        .map_err(|e| anyhow::anyhow!("failed to read file '{}': {e}", validated.display()))?;

    let occurrences = content.matches(old_string).count();
    if occurrences == 0 {
        // 错误恢复：返回最接近匹配（±3 行上下文）帮助模型定位差异
        anyhow::bail!(
            "{}",
            format_not_found_error(&validated.display().to_string(), &content, old_string)
        );
    }
    if !replace_all && occurrences > 1 {
        anyhow::bail!(
            "old_string is ambiguous: found {occurrences} occurrences in '{}'. \
             Provide more surrounding context to make it unique, or set replace_all=true",
            validated.display()
        );
    }

    let updated = if replace_all {
        content.replace(old_string, &sanitized_new)
    } else {
        content.replacen(old_string, &sanitized_new, 1)
    };

    // diff 预览（落盘前确认）
    let diff = generate_diff(&validated.display().to_string(), &content, &updated);
    let confirm =
        confirm_non_interactive(&validated.display().to_string(), &diff, cfg.auto_approve);
    if !confirm.approved {
        anyhow::bail!(
            "edit rejected (auto_approve=false). Proposed diff:\n{}",
            confirm.diff
        );
    }

    // 按原编码写入
    write_with_encoding(&validated, &updated, encoding)
        .map_err(|e| anyhow::anyhow!("failed to write file '{}': {e}", validated.display()))?;

    // 落盘后校验（失败回滚到原内容）
    validate_after_write(&validated, &content, encoding, cfg.java_check).map_err(|e| {
        anyhow::anyhow!(
            "post-write validation failed for '{}': {e}",
            validated.display()
        )
    })?;

    tracing::info!(
        path = %validated.display(),
        occurrences,
        replace_all,
        "file edit completed"
    );

    let count = if replace_all { occurrences } else { 1 };
    Ok(format!(
        "replaced {count} occurrence(s) in {}",
        validated.display()
    ))
}

/// 精确替换文件中的文本片段。
///
/// 用 `old_string` → `new_string` 做精确替换，避免整文件重写。
/// `old_string` 须与文件内容逐字节一致（含缩进）；默认要求唯一匹配，
/// 出现多次时须提供更多上下文或设 `replace_all=true`。
///
/// # 参数
/// - `path`：文件路径（相对或绝对，须位于当前工作目录内）。
/// - `old_string`：要被替换的原文。
/// - `new_string`：替换后的新文本。
/// - `replace_all`：是否替换全部出现（默认 `false`，仅替换唯一一处）。
///
/// # 返回
/// - 成功：替换次数确认。
/// - 失败：路径越界、文件不存在、`old_string` 未找到或出现多次。
#[cogent_tool]
async fn edit(
    path: String,
    old_string: String,
    new_string: String,
    replace_all: Option<bool>,
) -> anyhow::Result<String> {
    let root = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current working directory: {e}"))?;
    edit_file(
        &path,
        &old_string,
        &new_string,
        replace_all.unwrap_or(false),
        &root,
        ReliabilityConfig::from_env(),
    )
    .await
}

#[cfg(test)]
mod tests {
    //! 文件 IO 工具单元测试。
    //!
    //! 验证核心逻辑（`validate_path`/`read_file`/`write_file`）与工具元数据。
    //! 使用 `tempfile` 隔离测试目录，避免污染仓库。

    use super::*;
    use cogent_core::tool::Tool;

    /// 验证 `validate_path` 接受根目录内的路径。
    #[test]
    fn test_validate_path_inside_root() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "x").unwrap();
        let result = validate_path(file.to_str().unwrap(), dir.path()).unwrap();
        assert!(result.starts_with(dir.path().canonicalize().unwrap()));
    }

    /// 验证 `validate_path` 拒绝越界路径（Path Traversal 防护）。
    #[test]
    fn test_validate_path_traversal_rejected() {
        let dir = tempfile::tempdir().unwrap();
        // 构造指向根目录外的路径
        let evil = dir
            .path()
            .join("..")
            .join("..")
            .join("..")
            .join("etc")
            .join("passwd");
        let result = validate_path(evil.to_str().unwrap(), dir.path());
        assert!(result.is_err());
    }

    /// 验证 `validate_path` 对不存在但父目录存在的新文件返回拼接路径。
    #[test]
    fn test_validate_path_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let new_file = dir.path().join("new.txt");
        let result = validate_path(new_file.to_str().unwrap(), dir.path()).unwrap();
        // 结果应以 new.txt 结尾，且位于 canonicalize 后的根目录内
        assert!(result.ends_with("new.txt"));
        let canonical_root = dir.path().canonicalize().unwrap();
        assert!(result.starts_with(&canonical_root));
    }

    /// 验证 `read_file` 读取文件内容。
    #[tokio::test]
    async fn test_read_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        std::fs::write(&file, "hello world").unwrap();
        let content = read_file(file.to_str().unwrap(), dir.path()).await.unwrap();
        assert_eq!(content, "hello world");
    }

    /// 验证 `write_file` 写入文件。
    #[tokio::test]
    async fn test_write_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("out.txt");
        let result = write_file(
            file.to_str().unwrap(),
            "written content",
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await
        .unwrap();
        assert!(result.contains("wrote"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "written content");
    }

    /// 验证 `read_file` 越界路径被拒绝。
    #[tokio::test]
    async fn test_read_file_traversal_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let evil = dir
            .path()
            .join("..")
            .join("..")
            .join("..")
            .join("etc")
            .join("passwd");
        let result = read_file(evil.to_str().unwrap(), dir.path()).await;
        assert!(result.is_err());
    }

    /// 验证 `write_file` 越界路径被拒绝。
    #[tokio::test]
    async fn test_write_file_traversal_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let evil = dir
            .path()
            .join("..")
            .join("..")
            .join("..")
            .join("etc")
            .join("passwd");
        let result = write_file(
            evil.to_str().unwrap(),
            "x",
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await;
        assert!(result.is_err());
    }

    /// 验证 `edit_file` 唯一匹配时精确替换。
    #[tokio::test]
    async fn test_edit_file_single_replace() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello world\nfoo bar\n").unwrap();
        let result = edit_file(
            file.to_str().unwrap(),
            "foo bar",
            "baz qux",
            false,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await
        .unwrap();
        assert!(result.contains("replaced 1"));
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "hello world\nbaz qux\n"
        );
    }

    /// 验证 `edit_file` 在 `old_string` 未找到时报错。
    #[tokio::test]
    async fn test_edit_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello world").unwrap();
        let result = edit_file(
            file.to_str().unwrap(),
            "absent",
            "x",
            false,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    /// 验证 `edit_file` 在 `replace_all=false` 且 `old_string` 出现多次时报错（歧义）。
    #[tokio::test]
    async fn test_edit_file_ambiguous() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "dup dup dup").unwrap();
        let result = edit_file(
            file.to_str().unwrap(),
            "dup",
            "x",
            false,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ambiguous"));
    }

    /// 验证 `edit_file` 在 `replace_all=true` 时替换全部出现。
    #[tokio::test]
    async fn test_edit_file_replace_all() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "dup dup dup").unwrap();
        let result = edit_file(
            file.to_str().unwrap(),
            "dup",
            "x",
            true,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await
        .unwrap();
        assert!(result.contains("replaced 3"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "x x x");
    }

    /// 验证 `edit_file` 拒绝空 `old_string`。
    #[tokio::test]
    async fn test_edit_file_empty_old_string() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello").unwrap();
        let result = edit_file(
            file.to_str().unwrap(),
            "",
            "x",
            false,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await;
        assert!(result.is_err());
    }

    /// 验证 `edit_file` 拒绝 `old_string == new_string`（无意义替换）。
    #[tokio::test]
    async fn test_edit_file_same_string() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello").unwrap();
        let result = edit_file(
            file.to_str().unwrap(),
            "hello",
            "hello",
            false,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await;
        assert!(result.is_err());
    }

    /// 验证 `edit_file` 越界路径被拒绝。
    #[tokio::test]
    async fn test_edit_file_traversal_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let evil = dir
            .path()
            .join("..")
            .join("..")
            .join("..")
            .join("etc")
            .join("passwd");
        let result = edit_file(
            evil.to_str().unwrap(),
            "a",
            "b",
            false,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await;
        assert!(result.is_err());
    }

    /// 验证 edit 工具名称。
    #[test]
    fn test_edit_name() {
        let tool = EditTool;
        assert_eq!(tool.name(), "edit");
    }

    /// 验证 edit 参数 Schema 包含 path / old_string / new_string / replace_all。
    #[test]
    fn test_edit_schema() {
        let tool = EditTool;
        let schema = tool.parameters_schema();
        assert!(schema["properties"].get("path").is_some());
        assert!(schema["properties"].get("old_string").is_some());
        assert!(schema["properties"].get("new_string").is_some());
        assert!(schema["properties"].get("replace_all").is_some());
    }

    /// 验证 file_read 工具名称。
    #[test]
    fn test_file_read_name() {
        let tool = FileReadTool;
        assert_eq!(tool.name(), "file_read");
    }

    /// 验证 file_write 工具名称。
    #[test]
    fn test_file_write_name() {
        let tool = FileWriteTool;
        assert_eq!(tool.name(), "file_write");
    }

    /// 验证 file_read 参数 Schema。
    #[test]
    fn test_file_read_schema() {
        let tool = FileReadTool;
        let schema = tool.parameters_schema();
        assert!(schema["properties"].get("path").is_some());
        assert_eq!(schema["required"][0], "path");
    }

    /// 验证 file_write 参数 Schema 包含 path 和 content。
    #[test]
    fn test_file_write_schema() {
        let tool = FileWriteTool;
        let schema = tool.parameters_schema();
        assert!(schema["properties"].get("path").is_some());
        assert!(schema["properties"].get("content").is_some());
        assert_eq!(schema["required"].as_array().unwrap().len(), 2);
    }

    // ===== 可靠性增强集成测试 =====

    /// 验证 `edit_file` 净化 `new_string` 前导空行（sanitize 集成）。
    #[tokio::test]
    async fn test_edit_file_sanitizes_leading_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello world\n").unwrap();
        // new_string 带前导空行，应被净化后落盘
        let result = edit_file(
            file.to_str().unwrap(),
            "hello world",
            "\n\nnew content",
            false,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await
        .unwrap();
        assert!(result.contains("replaced 1"));
        // 前导空行被去除
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "new content\n");
    }

    /// 验证 `edit_file` 拒绝含 U+FFFB 替换符的 `new_string`（sanitize 集成）。
    #[tokio::test]
    async fn test_edit_file_rejects_replacement_char() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello").unwrap();
        let result = edit_file(
            file.to_str().unwrap(),
            "hello",
            "wor\u{FFFB}ld",
            false,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sanitizer"));
        // 文件未被修改
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello");
    }

    /// 验证 `edit_file` 在 `old_string` 未找到时返回最接近匹配（error_recovery 集成）。
    #[tokio::test]
    async fn test_edit_file_not_found_returns_closest_match() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "line1\n    target code\nline3\n").unwrap();
        // old_string 缩进与文件不一致（多 4 空格），未精确匹配
        let result = edit_file(
            file.to_str().unwrap(),
            "        target code",
            "x",
            false,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"));
        // 增强错误信息含最接近匹配上下文
        assert!(err.contains("Closest match"));
        assert!(err.contains("target code"));
    }

    /// 验证 `edit_file` 保持 GBK 编码（encoding 集成）。
    #[tokio::test]
    async fn test_edit_file_preserves_gbk_encoding() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        // 写入 GBK 编码的中文内容
        let (cow, _, _) = encoding_rs::GBK.encode("中文内容");
        std::fs::write(&file, cow.as_ref()).unwrap();
        // 编辑（替换中文片段）
        let result = edit_file(
            file.to_str().unwrap(),
            "中文",
            "修改",
            false,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await
        .unwrap();
        assert!(result.contains("replaced 1"));
        // 验证文件仍为 GBK 编码（非 UTF-8）
        let bytes = std::fs::read(&file).unwrap();
        let enc = crate::encoding::detect_encoding_from_bytes(&bytes).unwrap();
        assert_eq!(enc, crate::encoding::Encoding::Gbk);
        // 验证内容正确
        let (text, _) = crate::encoding::read_with_encoding(&file).unwrap();
        assert_eq!(text, "修改内容");
    }

    /// 验证 `edit_file` 在 `auto_approve=false` 时拒绝落盘（diff_preview 集成）。
    #[tokio::test]
    async fn test_edit_file_rejected_when_auto_approve_false() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello").unwrap();
        let cfg = ReliabilityConfig {
            auto_approve: false,
            java_check: false,
        };
        let result = edit_file(
            file.to_str().unwrap(),
            "hello",
            "world",
            false,
            dir.path(),
            cfg,
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rejected"));
        // 文件未被修改
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello");
    }

    /// 验证 `write_file` 对非法 JSON 回滚（validate 集成）。
    #[tokio::test]
    async fn test_write_file_invalid_json_rolls_back() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.json");
        let original = r#"{"key": "value"}"#;
        std::fs::write(&file, original).unwrap();
        // 写入非法 JSON
        let result = write_file(
            file.to_str().unwrap(),
            "{invalid json",
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("validation failed")
        );
        // 文件已回滚到原内容
        assert_eq!(std::fs::read_to_string(&file).unwrap(), original);
    }

    /// 验证 `write_file` 对合法 JSON 通过（validate 集成）。
    #[tokio::test]
    async fn test_write_file_valid_json_passes() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.json");
        std::fs::write(&file, r#"{"key": "old"}"#).unwrap();
        let result = write_file(
            file.to_str().unwrap(),
            r#"{"key": "new"}"#,
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await
        .unwrap();
        assert!(result.contains("wrote"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), r#"{"key": "new"}"#);
    }

    /// 验证 `write_file` 净化整文件内容的前导空行（sanitize 集成）。
    ///
    /// 模型常改用 file_write 整文件重写，重写内容带前导空行，落盘前应去除。
    #[tokio::test]
    async fn test_write_file_sanitizes_leading_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.java");
        std::fs::write(&file, "package x;\n").unwrap();
        // 整文件重写内容带前导空行
        let result = write_file(
            file.to_str().unwrap(),
            "\n\npackage x;\nclass A {}\n",
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await
        .unwrap();
        assert!(result.contains("wrote"));
        // 前导空行被去除
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "package x;\nclass A {}\n"
        );
    }

    /// 验证 `write_file` 拒绝含 U+FFFB 的整文件内容（sanitize 集成）。
    #[tokio::test]
    async fn test_write_file_rejects_replacement_char() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello").unwrap();
        let result = write_file(
            file.to_str().unwrap(),
            "wor\u{FFFB}ld",
            dir.path(),
            ReliabilityConfig::default(),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sanitizer"));
        // 文件未被修改
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello");
    }
}
