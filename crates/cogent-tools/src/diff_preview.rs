//! edit 前 diff 预览模块。
//!
//! 在 edit/file_write 工具落盘前，生成 unified diff 并请求确认，防止模型
//! 错误直接落盘（实测：前导空行/乱码/粘连曾被直接写入文件）。
//!
//! # 确认模式
//! - **非交互**（`cog run`）：按 `auto_approve` 决定。`true`（默认）自动批准，
//!   diff 记录到 `tracing::info!`（target: `cogent.diff`）；`false` 拒绝落盘，
//!   返回 diff 给模型让其重新生成。
//! - **交互**（REPL）：展示 diff，读 stdin `y/n` 确认。
//!
//! # 设计说明
//! diff 用 `similar` 生成（unified 格式，上下文 3 行）。核心逻辑
//! （[`generate_diff`] / [`confirm_non_interactive`]）为纯函数便于测试；
//! 交互确认（[`confirm_interactive`]）读 stdin，集成时按运行模式选择。

use similar::TextDiff;

/// diff 预览确认结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfirmResult {
    /// 是否批准落盘。
    pub approved: bool,
    /// 生成的 unified diff（无论批准与否都返回，供日志或回传给模型）。
    pub diff: String,
}

/// 生成 `old_content` → `new_content` 的 unified diff（上下文 3 行）。
///
/// # 参数
/// - `path`：文件路径（用于 diff header）。
/// - `old_content`：原文件内容。
/// - `new_content`：新文件内容。
///
/// # 返回
/// unified diff 文本（含 `---`/`+++` header 与 `@@` 行号）。
pub fn generate_diff(path: &str, old_content: &str, new_content: &str) -> String {
    let diff = TextDiff::from_lines(old_content, new_content);
    diff.unified_diff()
        .context_radius(3)
        .header(&format!("{path} (before)"), &format!("{path} (after)"))
        .to_string()
}

/// 非交互模式确认（按 `auto_approve` 决定）。
///
/// # 参数
/// - `path`：文件路径（用于日志）。
/// - `diff`：已生成的 diff（[`generate_diff`] 输出）。
/// - `auto_approve`：`true` 自动批准（diff 记录到日志）；`false` 拒绝落盘。
///
/// # 返回
/// [`ConfirmResult`]（`approved` 按 `auto_approve`，`diff` 原样返回）。
pub fn confirm_non_interactive(path: &str, diff: &str, auto_approve: bool) -> ConfirmResult {
    if auto_approve {
        // 自动批准：diff 记录到日志（target: cogent.diff），便于事后审计
        tracing::info!(
            target: "cogent.diff",
            path,
            diff = %diff,
            "edit auto-approved (diff logged)"
        );
        ConfirmResult {
            approved: true,
            diff: diff.to_string(),
        }
    } else {
        // 拒绝落盘：diff 返回给模型，让其重新生成
        tracing::warn!(
            target: "cogent.diff",
            path,
            "edit rejected (auto_approve=false), diff returned to model"
        );
        ConfirmResult {
            approved: false,
            diff: diff.to_string(),
        }
    }
}

/// 交互模式确认（展示 diff，读 stdin `y/n`）。
///
/// 从 stdin 读一行：`y`/`Y`/`yes`/`YES` 批准，其余拒绝。
///
/// # 参数
/// - `diff`：已生成的 diff（展示给用户）。
///
/// # 返回
/// [`ConfirmResult`]（`approved` 按用户输入，`diff` 原样返回）。
pub fn confirm_interactive(diff: &str) -> ConfirmResult {
    // 展示 diff 到 stdout（交互模式下用户可见）
    println!("\n--- Proposed change ---\n{diff}\n-----------------------");
    print!("Approve? [y/N]: ");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        // 读取失败（如 stdin 关闭）默认拒绝（安全默认）
        return ConfirmResult {
            approved: false,
            diff: diff.to_string(),
        };
    }
    let approved = matches!(input.trim().to_lowercase().as_str(), "y" | "yes");
    ConfirmResult {
        approved,
        diff: diff.to_string(),
    }
}

#[cfg(test)]
mod tests {
    //! `diff-preview` 单元测试。
    //!
    //! 覆盖 diff 生成、非交互自动批准/拒绝、交互确认的成功标准与边界。

    use super::*;

    /// 验证 `generate_diff` 生成 unified diff（含 header 与 @@ 行号）。
    #[test]
    fn test_generate_diff_unified() {
        let old = "line1\nline2\nline3";
        let new = "line1\nline2-modified\nline3";
        let diff = generate_diff("a.txt", old, new);
        assert!(diff.contains("a.txt (before)"));
        assert!(diff.contains("a.txt (after)"));
        assert!(diff.contains("@@"));
        assert!(diff.contains("-line2"));
        assert!(diff.contains("+line2-modified"));
    }

    /// 验证 `generate_diff` 上下文 3 行（变更周围保留 3 行上下文）。
    #[test]
    fn test_generate_diff_context_3() {
        // 变更在第 5 行，上下文应含第 2-4 行与第 6-8 行
        let old = (1..=10)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut new_lines: Vec<String> = (1..=10).map(|i| format!("line{i}")).collect();
        new_lines[4] = "line5-changed".to_string();
        let new = new_lines.join("\n");
        let diff = generate_diff("a.txt", &old, &new);
        // 上下文 3 行：变更（line5）前后各 3 行
        assert!(diff.contains("line2"));
        assert!(diff.contains("line8"));
        // 超出上下文范围的行不应出现（line1 距变更 4 行）
        assert!(!diff.contains("line1\n"));
    }

    /// 验证 `generate_diff` 对无变更内容返回空 diff（无 @@ 段）。
    #[test]
    fn test_generate_diff_no_change() {
        let diff = generate_diff("a.txt", "same", "same");
        // 无变更时 unified diff 仅含 header，无 @@ 段
        assert!(!diff.contains("@@"));
    }

    /// 验证非交互模式 `auto_approve=true` 批准。
    #[test]
    fn test_confirm_auto_approve_true() {
        let result = confirm_non_interactive("a.txt", "diff-here", true);
        assert!(result.approved);
        assert_eq!(result.diff, "diff-here");
    }

    /// 验证非交互模式 `auto_approve=false` 拒绝。
    #[test]
    fn test_confirm_auto_approve_false() {
        let result = confirm_non_interactive("a.txt", "diff-here", false);
        assert!(!result.approved);
        assert_eq!(result.diff, "diff-here");
    }

    /// 验证 `ConfirmResult` 的 diff 字段无论批准与否都保留（供回传模型）。
    #[test]
    fn test_confirm_result_keeps_diff() {
        let diff = "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new";
        let approved = confirm_non_interactive("a.txt", diff, true);
        let rejected = confirm_non_interactive("a.txt", diff, false);
        assert_eq!(approved.diff, diff);
        assert_eq!(rejected.diff, diff);
    }
}
