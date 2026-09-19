//! Cogent 系统工具 crate。
//!
//! 通过 `#[cogent_tool]` 宏声明的系统工具：
//! - [`bash`]：执行 shell 命令。
//! - [`file`]：文件读写。
//! - [`http`]：HTTP GET。
//! - [`skill`]：加载内置工程技能（25 个 SKILL.md 编译期嵌入）。
//!
//! 可靠性增强模块（模型输出不可信前提下的防护层）：
//! - [`sanitize`]：模型输出净化（前导空行 / U+FFFB / 注释粘连）。
//! - [`encoding`]：文件编码检测（UTF-8 / UTF-8-BOM / GBK）。
//! - [`error_recovery`]：错误反馈增强（最接近匹配 / 结构化成败清单）。
//! - [`diff_preview`]：edit 前 diff 预览 + 确认。
//! - [`validate`]：落盘后校验（JSON / Java 编译 / 编码一致性）+ 回滚。

pub mod bash;
pub mod diff_preview;
pub mod encoding;
pub mod error_recovery;
pub mod file;
pub mod http;
pub mod sanitize;
pub mod skill;
pub mod validate;
