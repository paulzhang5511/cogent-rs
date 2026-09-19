//! Cogent 内置工程技能 crate。
//!
//! 将 25 个工程技能（来自 addyosmani/agent-skills）通过 `include_str!` 编译期嵌入
//! 二进制 `.rodata` 段，运行时零 IO。`cog.exe` 单文件即可携带全部技能。
//!
//! # 公开 API
//! - [`get_skill`]：按名称查询技能（返回完整内容 + 辅助文件）。
//! - [`list_skills`]：列出全部技能元数据。
//! - [`get_routing_table`]：返回 `using-agent-skills` 路由表（真实 meta-skill，system prompt 注入用）。
//!
//! # 类型
//! - [`SkillMeta`]：技能元数据（名称 / 阶段 / 描述）。
//! - [`SkillEntry`]：技能条目（元数据 + 内容 + 辅助文件）。
//! - [`SKILLS`]：静态注册表（25 个技能）。
//!
//! # 设计说明
//! 本 crate 为纯数据 + 查询，无外部依赖。`using-agent-skills` 路由表由
//! `cogent-cli` 的 factory 注入 system prompt（决策树 + 强制操作行为 + 生命周期）；
//! 完整技能正文由 `cogent-tools` 的 bash 工具拦截 `cog-skill <name>` 命令按需加载
//! （两个 provider 均可用）。引擎核心循环零改动。

pub mod registry;

pub use registry::{
    SKILLS, SkillEntry, SkillMeta, get_routing_table, get_skill, list_names, list_skills,
    render_skill,
};
