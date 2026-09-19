//! `cogent-skills` 集成测试。
//!
//! 验证全部 25 个技能可查询、内容非空、阶段合法、辅助文件完整。

use cogent_skills::{SKILLS, get_routing_table, get_skill, list_names, list_skills};

/// 验证注册表含 25 个技能。
#[test]
fn test_registry_has_25_skills() {
    assert_eq!(SKILLS.len(), 25);
}

/// 验证全部技能名称可查询（无遗漏）。
#[test]
fn test_all_names_queryable() {
    let names = list_names();
    assert_eq!(names.len(), 25);
    for name in &names {
        assert!(get_skill(name).is_some(), "skill '{name}' not queryable");
    }
}

/// 验证全部技能内容非空且含标题（# 开头）。
#[test]
fn test_all_content_has_title() {
    for entry in SKILLS.iter() {
        assert!(
            entry.content.starts_with("---") || entry.content.starts_with('#'),
            "skill '{}' content should start with frontmatter or title",
            entry.meta.name
        );
    }
}

/// 验证路由表含全部 25 个技能名称（LLM 可据此路由）。
#[test]
fn test_routing_table_mentions_all_skills() {
    let table = get_routing_table();
    for name in list_names() {
        assert!(table.contains(name), "routing table missing skill '{name}'");
    }
}

/// 验证阶段分布正确（Define 3 / Plan 1 / Build 6 / Verify 3 / Review 4 / Ship 6 / Meta 2）。
#[test]
fn test_phase_distribution() {
    let skills = list_skills();
    let count = |phase: &str| skills.iter().filter(|s| s.phase == phase).count();
    assert_eq!(count("Define"), 3);
    assert_eq!(count("Plan"), 1);
    assert_eq!(count("Build"), 6);
    assert_eq!(count("Verify"), 3);
    assert_eq!(count("Review"), 4);
    assert_eq!(count("Ship"), 6);
    assert_eq!(count("Meta"), 2);
}

/// 验证辅助文件总数正确（idea-refine 3 + constraint 1 = 4）。
#[test]
fn test_auxiliary_total() {
    let total: usize = SKILLS.iter().map(|s| s.auxiliary.len()).sum();
    assert_eq!(total, 4);
}
