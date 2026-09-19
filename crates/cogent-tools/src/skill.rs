//! 技能加载工具：按名称返回完整工程技能内容。
//!
//! 通过 `#[cogent_tool]` 宏声明，LLM 调用 `skill(name)` 获取指定技能的
//! 完整 SKILL.md 内容（含辅助文件），用于指导当前任务的工程流程。
//!
//! # 设计说明
//! - 技能内容编译期嵌入 `cogent-skills` crate（`include_str!` 进 `.rodata`），
//!   运行时零 IO。
//! - 一次返回主内容 + 全部辅助文件（避免 LLM 多次调用）。
//! - 未知技能名报错时附带可用技能列表（LLM 可据此纠正）。
//!
//! # 使用场景
//! LLM 收到任务后，根据 system prompt 中的路由表（`using-agent-skills`）
//! 判断该用哪个技能，调用 `skill(name)` 加载完整流程指导。

use cogent_macros::cogent_tool;

/// 加载指定工程技能的完整内容。
///
/// # 参数
/// - `name`：技能名称（如 `"test-driven-development"`、`"spec-driven-development"`）。
///
/// # 返回
/// - 成功：技能完整内容（SKILL.md 全文 + 辅助文件，以 `---` 分隔）。
/// - 失败：未知技能名（错误信息含全部可用技能列表）。
#[cogent_tool]
async fn skill(name: String) -> anyhow::Result<String> {
    if let Some(entry) = cogent_skills::get_skill(&name) {
        tracing::info!(
            skill = %entry.meta.name,
            phase = %entry.meta.phase,
            content_len = entry.content.len(),
            auxiliary_count = entry.auxiliary.len(),
            "loading skill"
        );
    }
    cogent_skills::render_skill(&name).map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    //! `skill` 工具单元测试。
    //!
    //! 覆盖已知技能返回内容、辅助文件拼接、未知技能报错等成功标准。
    //! 通过 `SkillTool` 结构体的 `Tool::execute` 调用（`#[cogent_tool]` 宏
    //! 消费原函数，仅生成结构体）。

    use super::SkillTool;
    use cogent_core::tool::Tool;

    /// 验证 `skill` 工具返回已知技能的完整内容。
    #[tokio::test]
    async fn test_skill_returns_content() {
        let tool = SkillTool;
        let result = tool
            .execute(serde_json::json!({"name": "test-driven-development"}))
            .await
            .unwrap();
        assert!(result.contains("Test-Driven Development"));
        assert!(result.len() > 1000);
    }

    /// 验证 `skill` 工具返回含辅助文件的技能时拼接完整。
    #[tokio::test]
    async fn test_skill_with_auxiliary() {
        let tool = SkillTool;
        let result = tool
            .execute(serde_json::json!({"name": "idea-refine"}))
            .await
            .unwrap();
        // 主内容
        assert!(result.contains("Idea Refine"));
        // 辅助文件以 --- 分隔
        assert!(result.contains("\n\n---\n\n"));
        // 3 个辅助文件 → 至少 3 个分隔符
        let sep_count = result.matches("\n\n---\n\n").count();
        assert!(sep_count >= 3, "expected >= 3 separators, got {sep_count}");
    }

    /// 验证 `skill` 工具对未知技能报错且含可用列表。
    #[tokio::test]
    async fn test_skill_unknown_name() {
        let tool = SkillTool;
        let result = tool
            .execute(serde_json::json!({"name": "nonexistent-skill"}))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown skill"));
        assert!(err.contains("test-driven-development"));
        assert!(err.contains("spec-driven-development"));
    }

    /// 验证 `skill` 工具对空名称报错。
    #[tokio::test]
    async fn test_skill_empty_name() {
        let tool = SkillTool;
        let result = tool.execute(serde_json::json!({"name": ""})).await;
        assert!(result.is_err());
    }

    /// 验证 `skill` 工具返回的无辅助文件技能不以分隔符结尾。
    #[tokio::test]
    async fn test_skill_no_auxiliary() {
        let tool = SkillTool;
        let result = tool
            .execute(serde_json::json!({"name": "spec-driven-development"}))
            .await
            .unwrap();
        // spec-driven-development 无辅助文件，不应以 --- 分隔符结尾
        assert!(!result.ends_with("\n\n---\n\n"));
    }
}
