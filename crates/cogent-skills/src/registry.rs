//! 技能静态注册表。
//!
//! 25 个工程技能（来自 addyosmani/agent-skills）通过 `include_str!` 编译期嵌入
//! 二进制 `.rodata` 段，运行时零 IO。`cog.exe` 单文件即可携带全部技能。
//!
//! # 设计
//! - [`SKILLS`]：静态注册表（`LazyLock` 初始化，编译期固定）。
//! - [`get_skill`]：按名称查询（返回完整内容 + 辅助文件）。
//! - [`list_skills`]：列出全部技能元数据（供 `skill` 工具报错时展示可用列表）。
//! - [`get_routing_table`]：返回 `using-agent-skills` 路由表（system prompt 注入用）。
//!
//! # 辅助文件
//! 部分技能有辅助文件（如 `idea-refine` 的 frameworks/examples/refinement-criteria），
//! 随主技能一起返回（一次调用拿全，避免 LLM 多次调用）。

use std::sync::LazyLock;

/// 技能元数据（名称 / 阶段 / 一句话描述）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkillMeta {
    /// 技能名称（与 `agent-skills/skills/<name>/` 目录名一致）。
    pub name: &'static str,
    /// 生命周期阶段（Define / Plan / Build / Verify / Review / Ship / Meta）。
    pub phase: &'static str,
    /// 一句话描述（供 LLM 判断该用哪个技能）。
    pub description: &'static str,
}

/// 技能条目（元数据 + 编译期嵌入的内容 + 辅助文件）。
#[derive(Debug)]
pub struct SkillEntry {
    /// 技能元数据。
    pub meta: SkillMeta,
    /// 主技能内容（SKILL.md 全文，`include_str!` 嵌入）。
    pub content: &'static str,
    /// 辅助文件内容（`include_str!` 嵌入，随主技能一起返回）。
    pub auxiliary: &'static [&'static str],
}

/// 静态注册表（编译期固定，25 个技能）。
///
/// 顺序按生命周期阶段组织（Define → Plan → Build → Verify → Review → Ship → Meta），
/// 与 `using-agent-skills` 的 Quick Reference 表一致。
pub static SKILLS: LazyLock<Vec<SkillEntry>> = LazyLock::new(|| {
    vec![
        // ===== Define =====
        SkillEntry {
            meta: SkillMeta {
                name: "interview-me",
                phase: "Define",
                description: "Surface what the user actually wants before any plan, spec, or code exists",
            },
            content: include_str!("skills/interview-me.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "idea-refine",
                phase: "Define",
                description: "Refine ideas through structured divergent and convergent thinking",
            },
            content: include_str!("skills/idea-refine.md"),
            auxiliary: &[
                include_str!("skills/auxiliary/idea-refine-frameworks.md"),
                include_str!("skills/auxiliary/idea-refine-examples.md"),
                include_str!("skills/auxiliary/idea-refine-refinement-criteria.md"),
            ],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "spec-driven-development",
                phase: "Define",
                description: "Requirements and acceptance criteria before code",
            },
            content: include_str!("skills/spec-driven-development.md"),
            auxiliary: &[],
        },
        // ===== Plan =====
        SkillEntry {
            meta: SkillMeta {
                name: "planning-and-task-breakdown",
                phase: "Plan",
                description: "Decompose into small, verifiable tasks",
            },
            content: include_str!("skills/planning-and-task-breakdown.md"),
            auxiliary: &[],
        },
        // ===== Build =====
        SkillEntry {
            meta: SkillMeta {
                name: "incremental-implementation",
                phase: "Build",
                description: "Thin vertical slices, test each before expanding",
            },
            content: include_str!("skills/incremental-implementation.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "source-driven-development",
                phase: "Build",
                description: "Verify against official docs before implementing",
            },
            content: include_str!("skills/source-driven-development.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "doubt-driven-development",
                phase: "Build",
                description: "Adversarial fresh-context review of every non-trivial decision",
            },
            content: include_str!("skills/doubt-driven-development.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "context-engineering",
                phase: "Build",
                description: "Right context at the right time",
            },
            content: include_str!("skills/context-engineering.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "frontend-ui-engineering",
                phase: "Build",
                description: "Production-quality UI with accessibility",
            },
            content: include_str!("skills/frontend-ui-engineering.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "api-and-interface-design",
                phase: "Build",
                description: "Stable interfaces with clear contracts",
            },
            content: include_str!("skills/api-and-interface-design.md"),
            auxiliary: &[],
        },
        // ===== Verify =====
        SkillEntry {
            meta: SkillMeta {
                name: "test-driven-development",
                phase: "Verify",
                description: "Failing test first, then make it pass",
            },
            content: include_str!("skills/test-driven-development.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "browser-testing-with-devtools",
                phase: "Verify",
                description: "Chrome DevTools MCP for runtime verification",
            },
            content: include_str!("skills/browser-testing-with-devtools.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "debugging-and-error-recovery",
                phase: "Verify",
                description: "Reproduce, localize, fix, guard",
            },
            content: include_str!("skills/debugging-and-error-recovery.md"),
            auxiliary: &[],
        },
        // ===== Review =====
        SkillEntry {
            meta: SkillMeta {
                name: "code-review-and-quality",
                phase: "Review",
                description: "Five-axis review with quality gates",
            },
            content: include_str!("skills/code-review-and-quality.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "code-simplification",
                phase: "Review",
                description: "Preserve behavior while reducing unnecessary complexity",
            },
            content: include_str!("skills/code-simplification.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "security-and-hardening",
                phase: "Review",
                description: "OWASP prevention, input validation, least privilege",
            },
            content: include_str!("skills/security-and-hardening.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "performance-optimization",
                phase: "Review",
                description: "Measure first, optimize only what matters",
            },
            content: include_str!("skills/performance-optimization.md"),
            auxiliary: &[],
        },
        // ===== Ship =====
        SkillEntry {
            meta: SkillMeta {
                name: "git-workflow-and-versioning",
                phase: "Ship",
                description: "Atomic commits, clean history",
            },
            content: include_str!("skills/git-workflow-and-versioning.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "ci-cd-and-automation",
                phase: "Ship",
                description: "Automated quality gates on every change",
            },
            content: include_str!("skills/ci-cd-and-automation.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "deprecation-and-migration",
                phase: "Ship",
                description: "Remove old systems and migrate users safely",
            },
            content: include_str!("skills/deprecation-and-migration.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "documentation-and-adrs",
                phase: "Ship",
                description: "Document the why, not just the what",
            },
            content: include_str!("skills/documentation-and-adrs.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "observability-and-instrumentation",
                phase: "Ship",
                description: "Structured logs, RED metrics, traces, symptom-based alerts",
            },
            content: include_str!("skills/observability-and-instrumentation.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "shipping-and-launch",
                phase: "Ship",
                description: "Pre-launch checklist, monitoring, rollback plan",
            },
            content: include_str!("skills/shipping-and-launch.md"),
            auxiliary: &[],
        },
        // ===== Meta =====
        SkillEntry {
            meta: SkillMeta {
                name: "using-agent-skills",
                phase: "Meta",
                description: "Discovers and invokes agent skills (meta-skill governing all others)",
            },
            content: include_str!("skills/using-agent-skills.md"),
            auxiliary: &[],
        },
        SkillEntry {
            meta: SkillMeta {
                name: "constraint-driven-development",
                phase: "Meta",
                description: "Establish a project quality bar as a written contract",
            },
            content: include_str!("skills/constraint-driven-development.md"),
            auxiliary: &[include_str!("skills/auxiliary/constraint-floor-guard.md")],
        },
    ]
});

/// 按名称查询技能（返回完整内容 + 辅助文件）。
///
/// # 参数
/// - `name`：技能名称（与 [`SkillMeta::name`] 一致）。
///
/// # 返回
/// - 命中：`Some(&SkillEntry)`（含 `content` 与 `auxiliary`）。
/// - 未命中：`None`。
pub fn get_skill(name: &str) -> Option<&'static SkillEntry> {
    SKILLS.iter().find(|s| s.meta.name == name)
}

/// 列出全部技能元数据（供 `skill` 工具与 `cog-skill` 拦截报错时展示可用列表）。
///
/// # 返回
/// 全部 25 个 [`SkillMeta`]（按生命周期阶段顺序）。
pub fn list_skills() -> Vec<SkillMeta> {
    SKILLS.iter().map(|s| s.meta).collect()
}

/// 列出全部技能名称（供 `skill` 工具与 `cog-skill` 拦截报错时展示可用列表，逗号分隔）。
///
/// # 返回
/// 全部 25 个技能名称的 `Vec`（按生命周期阶段顺序）。
pub fn list_names() -> Vec<&'static str> {
    SKILLS.iter().map(|s| s.meta.name).collect()
}

/// 获取路由表（system prompt 注入用，~2.6K tokens，含决策树 + 强制操作行为 + 生命周期 + Quick Reference）。
///
/// # 返回
/// `using-agent-skills` 技能全文（真实 meta-skill）。
pub fn get_routing_table() -> &'static str {
    include_str!("skills/using-agent-skills.md")
}

/// 渲染技能完整内容（主内容 + 辅助文件，以 `\n\n---\n\n` 分隔）。
///
/// 供 `cogent-tools` 的 `skill` 工具与 `cog-skill` bash 拦截共用，
/// 消除两处重复的拼接逻辑。
///
/// # 参数
/// - `name`：技能名称。
///
/// # 返回
/// - 命中：`Ok(渲染后的完整内容)`。
/// - 未命中：`Err(错误信息，含全部可用技能列表)`。
pub fn render_skill(name: &str) -> Result<String, String> {
    let entry = get_skill(name).ok_or_else(|| {
        format!(
            "unknown skill '{}'. Available skills: {}",
            name,
            list_names().join(", ")
        )
    })?;
    let mut out = String::with_capacity(
        entry.content.len() + entry.auxiliary.iter().map(|a| a.len()).sum::<usize>(),
    );
    out.push_str(entry.content);
    for aux in entry.auxiliary {
        out.push_str("\n\n---\n\n");
        out.push_str(aux);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! `cogent-skills` registry 单元测试。
    //!
    //! 覆盖 get_skill 命中/未命中、list_skills 数量、get_routing_table 非空、
    //! 辅助文件完整性等成功标准。

    use super::*;

    /// 验证 `get_skill` 命中已知技能（内容非空）。
    #[test]
    fn test_get_skill_hit() {
        let entry = get_skill("test-driven-development").expect("TDD skill should exist");
        assert_eq!(entry.meta.name, "test-driven-development");
        assert_eq!(entry.meta.phase, "Verify");
        assert!(!entry.content.is_empty());
        assert!(entry.content.contains("Test-Driven Development"));
    }

    /// 验证 `get_skill` 未命中返回 `None`。
    #[test]
    fn test_get_skill_miss() {
        assert!(get_skill("nonexistent-skill").is_none());
        assert!(get_skill("").is_none());
    }

    /// 验证 `list_skills` 返回 25 个技能。
    #[test]
    fn test_list_skills_count() {
        assert_eq!(list_skills().len(), 25);
    }

    /// 验证 `list_skills` 中每个技能名称唯一且非空。
    #[test]
    fn test_list_skills_unique_names() {
        let skills = list_skills();
        let mut names: Vec<&str> = skills.iter().map(|s| s.name).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "skill names must be unique");
        assert!(names.iter().all(|n| !n.is_empty()));
    }

    /// 验证 `get_routing_table` 返回非空且含阶段表。
    #[test]
    fn test_get_routing_table() {
        let table = get_routing_table();
        assert!(!table.is_empty());
        assert!(table.contains("Quick Reference"));
        assert!(table.contains("test-driven-development"));
        // 真实 meta-skill：含决策树 + 强制操作行为 + 生命周期序列
        assert!(table.contains("Skill Discovery"));
        assert!(table.contains("Core Operating Behaviors"));
        assert!(table.contains("Lifecycle Sequence"));
    }

    /// 验证 `render_skill` 命中：返回主内容（无辅助文件时不含分隔符）。
    #[test]
    fn test_render_skill_hit_no_aux() {
        let out = render_skill("test-driven-development").unwrap();
        assert!(out.contains("Test-Driven Development"));
        assert!(!out.contains("\n\n---\n\n"));
    }

    /// 验证 `render_skill` 命中含辅助文件：以 `---` 分隔拼接。
    #[test]
    fn test_render_skill_hit_with_aux() {
        let out = render_skill("idea-refine").unwrap();
        assert!(out.contains("Idea Refine"));
        // 3 个辅助文件 → 至少 3 个分隔符
        assert!(out.matches("\n\n---\n\n").count() >= 3);
    }

    /// 验证 `render_skill` 未命中：返回 Err 且含可用列表。
    #[test]
    fn test_render_skill_miss() {
        let err = render_skill("nonexistent").unwrap_err();
        assert!(err.contains("unknown skill"));
        assert!(err.contains("test-driven-development"));
    }

    /// 验证 `idea-refine` 含 3 个辅助文件。
    #[test]
    fn test_idea_refine_auxiliary() {
        let entry = get_skill("idea-refine").expect("idea-refine should exist");
        assert_eq!(entry.auxiliary.len(), 3);
        assert!(entry.auxiliary.iter().all(|a| !a.is_empty()));
    }

    /// 验证 `constraint-driven-development` 含 1 个辅助文件。
    #[test]
    fn test_constraint_auxiliary() {
        let entry = get_skill("constraint-driven-development").expect("should exist");
        assert_eq!(entry.auxiliary.len(), 1);
        assert!(!entry.auxiliary[0].is_empty());
    }

    /// 验证无辅助文件的技能 `auxiliary` 为空。
    #[test]
    fn test_no_auxiliary() {
        let entry = get_skill("spec-driven-development").expect("should exist");
        assert!(entry.auxiliary.is_empty());
    }

    /// 验证所有技能内容非空（编译期嵌入完整性）。
    #[test]
    fn test_all_skills_content_nonempty() {
        for entry in SKILLS.iter() {
            assert!(
                !entry.content.is_empty(),
                "skill '{}' content is empty",
                entry.meta.name
            );
            assert!(
                !entry.meta.description.is_empty(),
                "skill '{}' description is empty",
                entry.meta.name
            );
        }
    }

    /// 验证阶段值合法（7 个已知阶段之一）。
    #[test]
    fn test_all_phases_valid() {
        let valid = [
            "Define", "Plan", "Build", "Verify", "Review", "Ship", "Meta",
        ];
        for entry in SKILLS.iter() {
            assert!(
                valid.contains(&entry.meta.phase),
                "skill '{}' has invalid phase '{}'",
                entry.meta.name,
                entry.meta.phase
            );
        }
    }
}
