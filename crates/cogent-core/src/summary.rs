//! 迭代进度总结模块。
//!
//! max iterations 到达时，输出已完成/未完成/建议三段总结，避免"做到一半
//! 停了"无感知（实测：跑满迭代上限直接终止，用户不知哪些文件
//! 已改、哪些失败）。
//!
//! # 设计说明
//! 本模块为纯函数（[`generate_summary`]），不依赖引擎内部状态、不做文件 IO。
//! 引擎在 max iterations 终止路径只返回错误；**生成文本、落盘、打印 stdout
//! 属展示/IO 关注点，由 CLI 层完成**（core 与展示解耦）。
//!
//! 本模块放 `cogent-core`（引擎层），因 max iterations 终止逻辑在引擎。

/// 单次工具执行记录（引擎追踪的迭代历史元素）。
///
/// 引擎在每轮工具执行后追加记录，max iterations 到达时汇总生成总结。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRecord {
    /// 工具名称（如 `edit`、`file_write`、`file_read`）。
    pub tool_name: String,
    /// 是否执行成功。
    pub success: bool,
    /// 失败原因（成功时为 `None`）。
    pub error: Option<String>,
}

/// 生成 max iterations 进度总结文本。
///
/// 三段结构：
/// - **Completed**：成功执行过的工具（按名称去重，保留首次出现顺序）。
/// - **Failed**：失败的工具调用（含原因，按 `(tool_name, error)` 去重）。
/// - **Suggestion**：基于失败项生成下一步建议。
///
/// # 参数
/// - `max_iterations`：配置的最大迭代次数。
/// - `history`：工具执行历史（按执行顺序）。
///
/// # 返回
/// 总结文本（英文，与日志规范一致）。
pub fn generate_summary(max_iterations: usize, history: &[ToolRecord]) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Max iterations ({max_iterations}) reached."));

    // Completed：成功工具按名称去重（保留首次顺序）
    let mut completed: Vec<&str> = Vec::new();
    for r in history.iter().filter(|r| r.success) {
        let name = r.tool_name.as_str();
        if !completed.contains(&name) {
            completed.push(name);
        }
    }
    lines.push("Completed:".to_string());
    if completed.is_empty() {
        lines.push("  (none)".to_string());
    } else {
        for name in &completed {
            lines.push(format!("- {name}"));
        }
    }

    // Failed：失败调用按 (tool_name, error) 去重
    lines.push("Failed:".to_string());
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut failed_lines: Vec<String> = Vec::new();
    for r in history.iter().filter(|r| !r.success) {
        let err = r
            .error
            .clone()
            .unwrap_or_else(|| "unknown error".to_string());
        let key = (r.tool_name.clone(), err.clone());
        if !seen.contains(&key) {
            seen.push(key);
            failed_lines.push(format!("- {}: {}", r.tool_name, err));
        }
    }
    if failed_lines.is_empty() {
        lines.push("  (none)".to_string());
    } else {
        lines.extend(failed_lines);
    }

    // Suggestion：基于失败项生成建议
    lines.push("Suggestion:".to_string());
    let has_failures = history.iter().any(|r| !r.success);
    if !has_failures {
        lines.push(
            "  All tool calls succeeded. Increase max_iterations or refine the prompt to \
             converge faster."
                .to_string(),
        );
    } else {
        // 取失败工具名（去重）生成针对性建议
        let mut failed_tools: Vec<&str> = Vec::new();
        for r in history.iter().filter(|r| !r.success) {
            let name = r.tool_name.as_str();
            if !failed_tools.contains(&name) {
                failed_tools.push(name);
            }
        }
        lines.push(format!(
            "  Re-run with corrected arguments for: {}.",
            failed_tools.join(", ")
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    //! `iteration-summary` 单元测试。
    //!
    //! 覆盖已完成/未完成/建议三段、去重、空历史、文件写入等成功标准。

    use super::*;

    fn rec(name: &str, success: bool, error: Option<&str>) -> ToolRecord {
        ToolRecord {
            tool_name: name.to_string(),
            success,
            error: error.map(|s| s.to_string()),
        }
    }

    /// 验证总结含三段（Completed/Failed/Suggestion）。
    #[test]
    fn test_summary_three_sections() {
        let history = vec![
            rec("edit", true, None),
            rec("edit", false, Some("old_string not found")),
        ];
        let s = generate_summary(40, &history);
        assert!(s.contains("Max iterations (40) reached."));
        assert!(s.contains("Completed:"));
        assert!(s.contains("Failed:"));
        assert!(s.contains("Suggestion:"));
    }

    /// 验证 Completed 段含成功工具（去重）。
    #[test]
    fn test_summary_completed_dedup() {
        let history = vec![
            rec("edit", true, None),
            rec("edit", true, None), // 重复，应去重
            rec("file_read", true, None),
        ];
        let s = generate_summary(10, &history);
        // edit 只出现一次（Completed 段）
        let completed_section = s.split("Failed:").next().unwrap();
        assert_eq!(completed_section.matches("- edit").count(), 1);
        assert!(completed_section.contains("- file_read"));
    }

    /// 验证 Failed 段含失败工具 + 原因（去重）。
    #[test]
    fn test_summary_failed_with_reason() {
        let history = vec![
            rec("edit", false, Some("old_string not found")),
            rec("edit", false, Some("old_string not found")), // 重复，应去重
            rec("file_write", false, Some("content is null")),
        ];
        let s = generate_summary(10, &history);
        let failed_section = s
            .split("Failed:")
            .nth(1)
            .unwrap()
            .split("Suggestion:")
            .next()
            .unwrap();
        assert_eq!(failed_section.matches("old_string not found").count(), 1);
        assert!(failed_section.contains("file_write"));
        assert!(failed_section.contains("content is null"));
    }

    /// 验证 Suggestion 段基于失败项生成针对性建议。
    #[test]
    fn test_summary_suggestion_from_failures() {
        let history = vec![rec("edit", false, Some("old_string not found"))];
        let s = generate_summary(10, &history);
        let suggestion = s.split("Suggestion:").nth(1).unwrap();
        assert!(suggestion.contains("edit"));
        assert!(suggestion.contains("corrected arguments"));
    }

    /// 验证全成功时 Suggestion 建议增加迭代或优化 prompt。
    #[test]
    fn test_summary_suggestion_all_success() {
        let history = vec![rec("edit", true, None)];
        let s = generate_summary(10, &history);
        let suggestion = s.split("Suggestion:").nth(1).unwrap();
        assert!(suggestion.contains("Increase max_iterations"));
    }

    /// 验证空历史时 Completed/Failed 均为 (none)。
    #[test]
    fn test_summary_empty_history() {
        let s = generate_summary(5, &[]);
        assert!(s.contains("Completed:\n  (none)"));
        assert!(s.contains("Failed:\n  (none)"));
    }
}
