//! 交互式 REPL 会话。
//!
//! 本模块实现 [`run_repl`]——基于 `rustyline` 的交互式会话循环：
//! 读取用户输入 → 追加 `Role::User` 到 memory → `engine.run()` 驱动 ReAct
//! → 渲染最终答案。支持 `/exit`、`/clear`、`/tools` 内置命令。
//!
//! # 设计原则
//! - REPL 是同步的（`rustyline` 阻塞读取），引擎 `run` 是异步的，
//!   通过 `tokio::task::block_in_place` 或直接在 async 上下文中 `await` 桥接。
//! - 每轮会话：用户输入 → 引擎 ReAct 循环（内部 `chat_complete`）→ 最终答案。
//! - 内置命令以 `/` 开头，不经过引擎。
//! - 历史文件保存在 `~/.cogent_history`（可选，失败静默忽略）。

use cogent_core::engine::AgentEngine;
use cogent_core::types::{Message, Role};
use colored::Colorize;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

/// 历史文件路径（`~/.cogent_history`）。
fn history_path() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| std::path::PathBuf::from(home).join(".cogent_history"))
}

/// 打印已注册工具列表（`/tools` 命令）。
///
/// # 参数
/// - `engine`：Agent 引擎（读取其工具列表）。
fn print_tools(engine: &AgentEngine) {
    let tools = engine.tools();
    println!("{}", format!("{} registered tools:", tools.len()).bold());
    for tool in tools {
        println!(
            "  {} — {}",
            tool.name().cyan(),
            tool.description().lines().next().unwrap_or("")
        );
    }
}

/// 运行交互式 REPL 会话。
///
/// # 参数
/// - `engine`：装配好的 Agent 引擎（`&mut`，每轮 `run` 会修改内部状态）。
/// - `max_context_tokens`：上下文窗口 token 预算（传给 `engine.run`）。
///
/// # 返回
/// - 成功：`Ok(())`（用户输入 `/exit` 或 EOF 时正常退出）。
/// - 失败：引擎运行错误（Provider 错误、死循环等）。
///
/// # 内置命令
/// - `/exit`：退出 REPL。
/// - `/clear`：清空会话历史（保留 system 种子）。
/// - `/tools`：列出已注册工具。
pub async fn run_repl(engine: &mut AgentEngine, max_context_tokens: usize) -> anyhow::Result<()> {
    let mut rl = DefaultEditor::new()?;

    // 加载历史（失败静默忽略）
    if let Some(path) = history_path() {
        let _ = rl.load_history(&path);
    }

    println!(
        "{}",
        "Cogent REPL — type /help for commands, /exit to quit.".dimmed()
    );

    loop {
        // 读取用户输入（阻塞）
        let line = match rl.readline("you> ") {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                // Ctrl+C：打印提示，继续循环
                println!("{}", "(ctrl-c, /exit to quit)".dimmed());
                continue;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl+D / EOF：退出
                break;
            }
            Err(e) => {
                return Err(anyhow::anyhow!("readline error: {e}"));
            }
        };

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        // 处理内置命令
        if let Some(cmd) = input.strip_prefix('/') {
            match cmd {
                "exit" | "quit" => break,
                "clear" => {
                    engine.memory_clear().await;
                    println!("{}", "session cleared".dimmed());
                    continue;
                }
                "tools" => {
                    print_tools(engine);
                    continue;
                }
                "help" => {
                    println!("{}", "/exit — quit\n/clear — clear session\n/tools — list tools\n/help — this help".dimmed());
                    continue;
                }
                _ => {
                    println!("{}", format!("unknown command: /{cmd}").yellow());
                    continue;
                }
            }
        }

        // 追加用户消息到 memory
        engine
            .memory_append(Message::new(Role::User, input.to_string()))
            .await;

        // 运行 ReAct 循环（内部 chat_complete 驱动思考-工具-观察，
        // 返回最终答案并以 Role::Assistant 写入 memory）
        match engine.run(max_context_tokens).await {
            Ok(answer) => {
                // 分块渲染最终答案（打字机效果，无第二次 LLM 调用）
                println!("\n{}", "cog:".bold().cyan());
                engine.render_answer_chunked(&answer);
                // 流式输出后补一个换行
                println!();
            }
            Err(e) => {
                // max iterations：进度总结的生成/落盘/打印在展示层处理
                if let cogent_core::error::CogentError::MaxIterationsExceeded(iter) = &e {
                    crate::emit_iteration_summary(engine, *iter);
                }
                println!("{}", format!("error: {e}").red());
            }
        }

        // 轮次边界：将引擎从终态（Completed/Failed）复位为 Idle，
        // 使下一轮输入可重新进入 ReAct 循环（多轮会话支持）。
        engine.reset();

        // 保存历史（失败静默忽略）
        let _ = rl.add_history_entry(input);
    }

    // 退出前保存历史
    if let Some(path) = history_path() {
        let _ = rl.save_history(&path);
    }

    println!("{}", "goodbye".dimmed());
    Ok(())
}
