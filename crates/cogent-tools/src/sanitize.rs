//! 模型输出净化模块。
//!
//! 在模型输出的 edit 内容落盘前，净化模型输出中实测暴露的已知质量问题：
//! - 前导空行去除（模型 edit 内容常自带 `\n` 开头，导致文件顶部多出空行）。
//! - U+FFFB 替换符检测（模型中文输出损坏时产生 `EF BF BD`，无法推断原字符，
//!   拒绝落盘让模型重新生成）。
//! - 注释粘连检测（模型注释与下一行代码缺换行，合并成一行破坏代码结构）。
//!
//! # 设计原则
//! - 仅净化 `new_string`（替换内容），不修改 `old_string`（用于匹配文件内容，
//!   修改会导致匹配失败）。
//! - 检测到的质量问题**拒绝落盘**（返回 `SanitizeError`），不自动修复
//!   （U+FFFB 无法推断原字符；注释粘连自动补换行可能改变语义）。

use thiserror::Error;

/// 净化 edit 内容时检测到的质量问题。
///
/// 各变体携带足够上下文，使日志与错误报告无需额外查询即可定位问题。
/// 错误消息用英文（与日志规范一致），便于跨团队检索。
#[derive(Debug, Error, PartialEq, Eq, Clone, Copy)]
pub enum SanitizeError {
    /// `new_string` 全为空行（无有效内容）。
    ///
    /// 模型可能返回纯换行的替换内容，落盘会清空目标片段，须拒绝。
    #[error("new_string is empty (only blank lines)")]
    EmptyContent,

    /// `new_string` 含 U+FFFB 替换符（模型输出损坏，无法推断原字符）。
    ///
    /// `position` 为 U+FFFB 在 `new_string` 中的字符偏移（0-based）。
    #[error("new_string contains U+FFFB replacement character at char offset {position}")]
    ReplacementChar {
        /// U+FFFB 的字符偏移（0-based）。
        position: usize,
    },

    /// 注释与代码粘连（注释行后紧跟代码行，无空行分隔）。
    ///
    /// `line` 为注释行的行号（1-based）。
    #[error("comment glued to code at line {line}")]
    CommentGlued {
        /// 注释行的行号（1-based）。
        line: usize,
    },
}

/// 净化 edit 内容，返回净化后的 `(old_string, new_string)`。
///
/// 仅净化 `new_string`（去前导空行 + 检测 U+FFFB + 检测注释粘连），
/// `old_string` 原样返回（用于匹配文件内容，不可修改）。
///
/// # 参数
/// - `old_string`：模型提供的旧内容（不修改，用于匹配文件内容）。
/// - `new_string`：模型提供的新内容（净化目标）。
///
/// # 返回
/// - 成功：净化后的 `(old_string, new_string)`。
/// - 失败：`SanitizeError`（空内容 / 替换符 / 注释粘连）。
pub fn sanitize_edit_content(
    old_string: &str,
    new_string: &str,
) -> Result<(String, String), SanitizeError> {
    // 仅净化 new_string；old_string 原样返回（用于匹配文件内容）
    let sanitized = strip_leading_blank_lines(new_string)?;
    check_replacement_char(&sanitized)?;
    check_comment_glued(&sanitized)?;
    Ok((old_string.to_string(), sanitized))
}

/// 净化整文件写入内容（`file_write` 用）。
///
/// 模型常无视"用 edit 精确修改"指令而改用 `file_write` 整文件重写，
/// 重写内容同样带前导空行 / U+FFFB / 注释粘连。本函数对整文件内容应用
/// 与 [`sanitize_edit_content`] 相同的三项净化，返回净化后的完整内容。
///
/// # 参数
/// - `content`：模型提供的整文件内容（净化目标）。
///
/// # 返回
/// - 成功：净化后的完整文件内容。
/// - 失败：`SanitizeError`（空内容 / 替换符 / 注释粘连）。
pub fn sanitize_write_content(content: &str) -> Result<String, SanitizeError> {
    let sanitized = strip_leading_blank_lines(content)?;
    check_replacement_char(&sanitized)?;
    check_comment_glued(&sanitized)?;
    Ok(sanitized)
}

/// 去除 `new_string` 开头连续空行（`\n`/`\r`）。
///
/// 仅去除**开头**空行，保留内容内部空行。若去除后为空（原串全为空行或空串），
/// 返回 `SanitizeError::EmptyContent`。
fn strip_leading_blank_lines(s: &str) -> Result<String, SanitizeError> {
    let trimmed = s.trim_start_matches(['\n', '\r']);
    if trimmed.is_empty() {
        return Err(SanitizeError::EmptyContent);
    }
    Ok(trimmed.to_string())
}

/// 检测 `new_string` 是否含 U+FFFB 替换符（`EF BF BD`）。
///
/// U+FFFB 是模型中文输出损坏的标志（无法推断原字符），检测到即拒绝落盘。
fn check_replacement_char(s: &str) -> Result<(), SanitizeError> {
    // 用字符偏移（非字节偏移），便于日志定位
    if let Some(pos) = s.chars().position(|c| c == '\u{FFFB}') {
        return Err(SanitizeError::ReplacementChar { position: pos });
    }
    Ok(())
}

/// 检测 `new_string` 是否存在注释与代码粘连。
///
/// 启发式规则：某行 trim 后以 `//` 或 `#` 开头（行注释），且该行末尾无
/// 块注释结束符 `*/`，且下一行非空且非注释 → 判定为粘连。
///
/// 边界：仅检测行注释，不检测块注释（`/* */`）内部粘连；注释行是最后一行
/// 时不判定（无下一行）。
fn check_comment_glued(s: &str) -> Result<(), SanitizeError> {
    let lines: Vec<&str> = s.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        // 行注释：trim 后以 // 或 # 开头
        if !(trimmed.starts_with("//") || trimmed.starts_with('#')) {
            continue;
        }
        // 注释行末尾有块注释结束符 */ 时跳过（如 /* ... */ 单行块注释）
        if line.trim_end().ends_with("*/") {
            continue;
        }
        // 下一行存在、非空、非注释 → 粘连
        if let Some(next) = lines.get(i + 1) {
            let next_trimmed = next.trim();
            if !next_trimmed.is_empty()
                && !next_trimmed.starts_with("//")
                && !next_trimmed.starts_with('#')
            {
                return Err(SanitizeError::CommentGlued { line: i + 1 });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! `output-sanitize` 单元测试。
    //!
    //! 覆盖前导空行去除、U+FFFB 检测、注释粘连检测的成功标准与边界。

    use super::*;

    /// 验证前导空行去除：`"\n\n    code"` → `"    code"`。
    #[test]
    fn test_strip_leading_blank_lines() {
        let (old, new) = sanitize_edit_content("old", "\n\n    code").unwrap();
        assert_eq!(old, "old");
        assert_eq!(new, "    code");
    }

    /// 验证前导空行去除保留内容内部空行。
    #[test]
    fn test_strip_leading_preserves_internal_blanks() {
        let (old, new) = sanitize_edit_content("old", "\n\nline1\n\nline2").unwrap();
        assert_eq!(old, "old");
        assert_eq!(new, "line1\n\nline2");
    }

    /// 验证 `new_string` 全为空行时返回 `EmptyContent`。
    #[test]
    fn test_empty_content_all_blank_lines() {
        let err = sanitize_edit_content("old", "\n\n\n").unwrap_err();
        assert_eq!(err, SanitizeError::EmptyContent);
    }

    /// 验证 `new_string` 为空串时返回 `EmptyContent`。
    #[test]
    fn test_empty_content_empty_string() {
        let err = sanitize_edit_content("old", "").unwrap_err();
        assert_eq!(err, SanitizeError::EmptyContent);
    }

    /// 验证 `new_string` 含 U+FFFB 时返回 `ReplacementChar`（含字符偏移）。
    #[test]
    fn test_replacement_char_detected() {
        // "ab\u{FFFB}c"：U+FFFB 在字符偏移 2
        let err = sanitize_edit_content("old", "ab\u{FFFB}c").unwrap_err();
        assert_eq!(err, SanitizeError::ReplacementChar { position: 2 });
    }

    /// 验证 `new_string` 无 U+FFFB 时通过替换符检测。
    #[test]
    fn test_no_replacement_char_passes() {
        let (_, new) = sanitize_edit_content("old", "正常中文内容").unwrap();
        assert_eq!(new, "正常中文内容");
    }

    /// 验证注释与代码粘连（无空行分隔）返回 `CommentGlued`。
    #[test]
    fn test_comment_glued_detected() {
        let err = sanitize_edit_content("old", "// 注释\n    code").unwrap_err();
        assert_eq!(err, SanitizeError::CommentGlued { line: 1 });
    }

    /// 验证注释与代码有空行分隔时通过。
    #[test]
    fn test_comment_with_blank_line_passes() {
        let (_, new) = sanitize_edit_content("old", "// 注释\n\n    code").unwrap();
        assert_eq!(new, "// 注释\n\n    code");
    }

    /// 验证 `#` 开头的注释粘连也被检测（Python/Shell 风格）。
    #[test]
    fn test_hash_comment_glued_detected() {
        let err = sanitize_edit_content("old", "# 注释\ncode").unwrap_err();
        assert_eq!(err, SanitizeError::CommentGlued { line: 1 });
    }

    /// 验证注释行末尾有 `*/`（单行块注释）时不判定粘连。
    #[test]
    fn test_block_comment_end_not_glued() {
        let (_, new) = sanitize_edit_content("old", "/* 注释 */\ncode").unwrap();
        assert_eq!(new, "/* 注释 */\ncode");
    }

    /// 验证注释行是最后一行时不判定粘连（无下一行）。
    #[test]
    fn test_comment_last_line_not_glued() {
        let (_, new) = sanitize_edit_content("old", "code\n// 注释").unwrap();
        assert_eq!(new, "code\n// 注释");
    }

    /// 验证 `old_string` 不被修改（仅净化 `new_string`）。
    #[test]
    fn test_old_string_unchanged() {
        let (old, _) = sanitize_edit_content("  old with spaces  ", "\n\nnew").unwrap();
        assert_eq!(old, "  old with spaces  ");
    }

    /// 验证多行注释粘连检测定位到正确行号（1-based）。
    #[test]
    fn test_comment_glued_line_number() {
        // 第 1 行 code，第 2 行注释，第 3 行 code → 粘连在第 2 行
        let err = sanitize_edit_content("old", "code\n// 注释\nmore").unwrap_err();
        assert_eq!(err, SanitizeError::CommentGlued { line: 2 });
    }

    /// 验证 `sanitize_write_content` 去除整文件内容前导空行。
    #[test]
    fn test_write_content_strip_leading() {
        let out = sanitize_write_content("\n\npackage x;\nclass A {}\n").unwrap();
        assert_eq!(out, "package x;\nclass A {}\n");
    }

    /// 验证 `sanitize_write_content` 保留内容内部空行。
    #[test]
    fn test_write_content_preserves_internal_blanks() {
        let out = sanitize_write_content("\n\nline1\n\nline2").unwrap();
        assert_eq!(out, "line1\n\nline2");
    }

    /// 验证 `sanitize_write_content` 拒绝含 U+FFFB 的内容。
    #[test]
    fn test_write_content_rejects_replacement_char() {
        let err = sanitize_write_content("wor\u{FFFB}ld").unwrap_err();
        assert!(matches!(err, SanitizeError::ReplacementChar { .. }));
    }

    /// 验证 `sanitize_write_content` 拒绝注释粘连。
    #[test]
    fn test_write_content_rejects_comment_glued() {
        let err = sanitize_write_content("code\n// 注释\nmore").unwrap_err();
        assert_eq!(err, SanitizeError::CommentGlued { line: 2 });
    }

    /// 验证 `sanitize_write_content` 拒绝全空内容。
    #[test]
    fn test_write_content_rejects_empty() {
        let err = sanitize_write_content("\n\n").unwrap_err();
        assert_eq!(err, SanitizeError::EmptyContent);
    }
}
