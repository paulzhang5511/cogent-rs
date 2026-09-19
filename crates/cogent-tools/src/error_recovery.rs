//! 错误反馈增强模块。
//!
//! 让模型能从错误反馈中自纠（实测：`old_string not found` 时
//! 仅报"不匹配"，模型难以定位差异，反复失败耗尽迭代）：
//! - [`find_closest_match`]：`old_string not found` 时，用 Levenshtein 距离
//!   在文件内容中找最接近的片段，返回 ±3 行上下文，让模型看到"文件实际长什么样"。
//! - [`format_tool_outcomes`]：同轮多工具调用部分失败时，返回结构化成败清单，
//!   让模型知道哪些成功、哪些失败及原因。
//!
//! # 设计说明
//! 本模块放 `cogent-tools`（工具执行层），
//! 因为集成点（`edit_file` 的 `old_string not found` 错误路径）在 `cogent-tools`，
//! 且 crate 依赖方向不允许 `cogent-tools` 反向依赖 provider crate。

use strsim::levenshtein;

/// 最接近匹配结果。
///
/// `file_content` 为最接近片段 ±3 行的上下文（供模型对照），`line` 为片段
/// 起始行号（1-based），`distance` 为 Levenshtein 距离（越小越接近）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosestMatch {
    /// 最接近片段的起始行号（1-based）。
    pub line: usize,
    /// 最接近片段 ±3 行的上下文文本。
    pub context: String,
    /// 与 `old_string` 的 Levenshtein 距离。
    pub distance: usize,
}

/// 在文件内容中查找与 `old_string` 最接近的片段。
///
/// 用滑动窗口（窗口行数 = `old_string` 行数）遍历文件，对每个窗口计算
/// Levenshtein 距离，返回距离最小的窗口。搜索窗口限制 ±50 行（性能保护），
/// 文件过大时仅搜索前 500 行。
///
/// # 参数
/// - `file_content`：文件当前完整内容。
/// - `old_string`：模型提供的旧内容（未匹配到）。
///
/// # 返回
/// - 成功：最接近的 [`ClosestMatch`]。
/// - 失败：`None`（文件为空或 `old_string` 为空）。
pub fn find_closest_match(file_content: &str, old_string: &str) -> Option<ClosestMatch> {
    if file_content.is_empty() || old_string.is_empty() {
        return None;
    }

    let old_lines: Vec<&str> = old_string.lines().collect();
    let window = old_lines.len().max(1);
    let file_lines: Vec<&str> = file_content.lines().collect();

    // 性能保护：文件过大时仅搜索前 500 行
    let search_limit = file_lines.len().min(500);
    if search_limit < window {
        // 文件行数不足一个窗口，整体作为候选
        let distance = levenshtein(old_string, file_content);
        return Some(ClosestMatch {
            line: 1,
            context: file_content.to_string(),
            distance,
        });
    }

    let mut best: Option<(usize, usize)> = None; // (distance, start_line_0based)
    for start in 0..=(search_limit - window) {
        let window_text: String = file_lines[start..start + window].join("\n");
        let distance = levenshtein(old_string, &window_text);
        // 保留距离最小的窗口（严格小于才更新，保证取首个最接近）
        match best {
            Some((best_dist, _)) if distance >= best_dist => {}
            _ => best = Some((distance, start)),
        }
    }

    let (distance, start) = best?;
    // ±3 行上下文
    let ctx_start = start.saturating_sub(3);
    let ctx_end = (start + window + 3).min(file_lines.len());
    let context = file_lines[ctx_start..ctx_end].join("\n");

    Some(ClosestMatch {
        line: start + 1,
        context,
        distance,
    })
}

/// 生成 `old_string not found` 的增强错误信息（含最接近匹配）。
///
/// # 参数
/// - `path`：文件路径（用于错误信息）。
/// - `file_content`：文件当前完整内容。
/// - `old_string`：模型提供的旧内容（未匹配到）。
///
/// # 返回
/// 增强错误信息字符串（含最接近匹配 ±3 行上下文 + 差异说明）。
pub fn format_not_found_error(path: &str, file_content: &str, old_string: &str) -> String {
    match find_closest_match(file_content, old_string) {
        Some(m) => format!(
            "old_string not found in '{path}': the text to replace does not match the file content \
             (check whitespace/indentation and that the file was read recently).\n\
             Closest match at line {} (Levenshtein distance {}):\n{}",
            m.line, m.distance, m.context
        ),
        None => format!(
            "old_string not found in '{path}': the text to replace does not match the file content \
             (check whitespace/indentation and that the file was read recently)."
        ),
    }
}

/// 单个工具调用的执行结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutcome {
    /// 工具名称（如 `edit`、`file_write`）。
    pub tool_name: String,
    /// 是否成功。
    pub success: bool,
    /// 失败原因（成功时为 `None`）。
    pub error: Option<String>,
}

/// 生成同轮多工具调用的结构化成败清单。
///
/// 让模型知道哪些工具成功、哪些失败及原因，便于针对性重试。
///
/// # 参数
/// - `outcomes`：本轮所有工具调用的结果（按执行顺序）。
///
/// # 返回
/// 结构化清单文本（每行一个工具：`- tool_name: SUCCESS` 或 `- tool_name: FAILED - reason`）。
pub fn format_tool_outcomes(outcomes: &[ToolOutcome]) -> String {
    if outcomes.is_empty() {
        return "No tool calls in this iteration.".to_string();
    }
    let mut lines = Vec::with_capacity(outcomes.len() + 1);
    lines.push("Tool results:".to_string());
    for o in outcomes {
        match &o.error {
            Some(err) => lines.push(format!("- {}: FAILED - {}", o.tool_name, err)),
            None => lines.push(format!("- {}: SUCCESS", o.tool_name)),
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    //! `error-recovery` 单元测试。
    //!
    //! 覆盖最接近匹配、增强错误信息、结构化成败清单的成功标准与边界。

    use super::*;

    /// 验证 `find_closest_match` 找到最接近片段（缩进差异）。
    #[test]
    fn test_find_closest_match_indent_diff() {
        let file = "line1\n    old code\nline3\nline4";
        // old_string 缩进比文件多 4 空格
        let old = "        old code";
        let m = find_closest_match(file, old).unwrap();
        assert_eq!(m.line, 2);
        assert!(m.context.contains("old code"));
        assert!(m.distance > 0); // 有差异
    }

    /// 验证 `find_closest_match` 对完全匹配返回距离 0。
    #[test]
    fn test_find_closest_match_exact() {
        let file = "line1\n    old code\nline3";
        let m = find_closest_match(file, "    old code").unwrap();
        assert_eq!(m.line, 2);
        assert_eq!(m.distance, 0);
    }

    /// 验证 `find_closest_match` 对空文件返回 `None`。
    #[test]
    fn test_find_closest_match_empty_file() {
        assert!(find_closest_match("", "old").is_none());
    }

    /// 验证 `find_closest_match` 对空 `old_string` 返回 `None`。
    #[test]
    fn test_find_closest_match_empty_old() {
        assert!(find_closest_match("content", "").is_none());
    }

    /// 验证 `find_closest_match` 对文件行数不足一个窗口时整体作为候选。
    #[test]
    fn test_find_closest_match_short_file() {
        let file = "only one line";
        let m = find_closest_match(file, "only one line\nextra").unwrap();
        assert_eq!(m.line, 1);
        assert!(m.context.contains("only one line"));
    }

    /// 验证 `format_not_found_error` 含最接近匹配上下文。
    #[test]
    fn test_format_not_found_error_with_match() {
        let file = "line1\n    old code\nline3";
        let err = format_not_found_error("a.java", file, "        old code");
        assert!(err.contains("old_string not found in 'a.java'"));
        assert!(err.contains("Closest match at line 2"));
        assert!(err.contains("old code"));
    }

    /// 验证 `format_not_found_error` 对空文件不含最接近匹配段。
    #[test]
    fn test_format_not_found_error_no_match() {
        let err = format_not_found_error("a.java", "", "old");
        assert!(err.contains("old_string not found in 'a.java'"));
        assert!(!err.contains("Closest match"));
    }

    /// 验证 `format_tool_outcomes` 生成结构化成败清单（1 成功 2 失败）。
    #[test]
    fn test_format_tool_outcomes_mixed() {
        let outcomes = vec![
            ToolOutcome {
                tool_name: "edit".into(),
                success: true,
                error: None,
            },
            ToolOutcome {
                tool_name: "edit".into(),
                success: false,
                error: Some("old_string not found".into()),
            },
            ToolOutcome {
                tool_name: "file_write".into(),
                success: false,
                error: Some("content is null".into()),
            },
        ];
        let out = format_tool_outcomes(&outcomes);
        assert!(out.contains("Tool results:"));
        assert!(out.contains("- edit: SUCCESS"));
        assert!(out.contains("- edit: FAILED - old_string not found"));
        assert!(out.contains("- file_write: FAILED - content is null"));
    }

    /// 验证 `format_tool_outcomes` 对空列表返回提示。
    #[test]
    fn test_format_tool_outcomes_empty() {
        let out = format_tool_outcomes(&[]);
        assert!(out.contains("No tool calls"));
    }

    /// 验证 `format_tool_outcomes` 对全成功列表。
    #[test]
    fn test_format_tool_outcomes_all_success() {
        let outcomes = vec![
            ToolOutcome {
                tool_name: "edit".into(),
                success: true,
                error: None,
            },
            ToolOutcome {
                tool_name: "file_read".into(),
                success: true,
                error: None,
            },
        ];
        let out = format_tool_outcomes(&outcomes);
        assert_eq!(out.matches("SUCCESS").count(), 2);
        assert!(!out.contains("FAILED"));
    }
}
