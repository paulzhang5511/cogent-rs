//! Cogent CLI 入口（命令名 `cog`）。
//!
//! 子命令：
//! - `repl`：交互式会话（rustyline 循环）。
//! - `run <prompt>`：单次任务（执行后退出）。
//! - `tools`：列出已注册工具（name + description + schema）。
//!
//! # 配置
//! 通过环境变量加载（见 [`cogent_core::config::Config::from_env`]）：
//! - `OPENAI_API_KEY`（必填）、`COGENT_MODEL`、`COGENT_BASE_URL`、
//!   `COGENT_MAX_ITERATIONS`、`COGENT_MAX_CONTEXT_TOKENS`、`COGENT_OTEL_ENDPOINT`。
//!
//! # 追踪
//! 若设置 `COGENT_OTEL_ENDPOINT`，则启用 OTel OTLP 导出（PRD §1.2）；
//! 否则仅本地 fmt 日志。OTel exporter 生命周期由 [`factory::OtelGuard`] 管理，
//! 程序退出前 `shut_down`。

mod factory;
mod middleware;
mod render;
mod repl;

use clap::{Parser, Subcommand};
use cogent_core::config::Config;
use cogent_core::error::CogentError;
use cogent_core::types::{Message, Role};
use colored::Colorize;

use crate::factory::build_engine;

/// Cogent CLI Agent Interface。
#[derive(Parser)]
#[command(name = "cog", about = "Cogent CLI Agent Interface", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// CLI 子命令。
#[derive(Subcommand)]
enum Commands {
    /// 交互式会话（REPL）。
    Repl,
    /// 单次任务：执行 prompt 后退出。
    Run {
        /// 要执行的任务描述。
        prompt: String,
    },
    /// 列出已注册工具（name + description + schema）。
    Tools,
}

/// 程序入口。
///
/// 解析 CLI 参数 → 按子命令分发：
/// - `tools`：仅注册并列出工具，**不加载配置、不装配引擎**（无需 API key，SPEC §7.5）。
/// - `repl` / `run`：加载配置 → 装配引擎 → 启动渲染器 → 执行。
///
/// OTel 守卫（`OtelGuard`）持有至函数返回，确保 span 在退出前导出。
async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // `cog tools` 独立路径：不依赖 API key / 引擎，直接列出已注册工具
    if let Commands::Tools = cli.command {
        let tools = factory::build_tools();
        print_tools(&tools);
        return Ok(());
    }

    // 其余子命令（repl / run）需要完整引擎
    // 加载配置（环境变量）
    let config = Config::from_env().map_err(|e| anyhow::anyhow!("config error: {e}"))?;

    // 装配引擎（含 tracing 初始化、共享 EventBus 与 OTel 守卫）
    let (mut engine, event_bus, otel_guard) = build_engine(&config, None).await?;

    // 启动事件渲染器（订阅引擎共享的 EventBus，彩色渲染）
    let _renderer = render::start_event_renderer(&event_bus);

    match cli.command {
        Commands::Repl => {
            repl::run_repl(&mut engine, config.max_context_tokens).await?;
        }
        Commands::Run { prompt } => {
            run_once(&mut engine, &prompt, config.max_context_tokens).await?;
        }
        // Tools 已在上方提前返回，此处逻辑不可达。
        // 不用 unreachable!（panic 风险），改为安全兜底：正常退出，
        // otel_guard 仍被 drop，flush span。
        Commands::Tools => {}
    }

    // otel_guard 在此 drop，flush 所有未导出的 span
    drop(otel_guard);
    Ok(())
}

/// 执行单次任务（`cog run <prompt>`）。
///
/// # 参数
/// - `engine`：Agent 引擎。
/// - `prompt`：任务描述。
/// - `max_context_tokens`：上下文窗口 token 预算。
///
/// # 流程
/// 1. 追加用户消息到 memory。
/// 2. `engine.run` 驱动 ReAct 循环（内部 `chat_complete`），返回最终答案
///    并将其以 `Role::Assistant` 写入 memory（多轮上下文完整）。
/// 3. `engine.render_answer_chunked` 将答案分块发布 `TokenChunk` 事件
///    （渲染器打字机效果），**不发起第二次 LLM 调用**。
async fn run_once(
    engine: &mut cogent_core::engine::AgentEngine,
    prompt: &str,
    max_context_tokens: usize,
) -> anyhow::Result<()> {
    // 追加用户消息
    engine
        .memory_append(Message::new(Role::User, prompt.to_string()))
        .await;

    // 运行 ReAct 循环（内部 chat_complete，得到最终答案并写入 memory）
    match engine.run(max_context_tokens).await {
        Ok(answer) => {
            // 分块渲染最终答案（打字机效果，无第二次 LLM 调用）
            println!("\n{}", "cog:".bold().cyan());
            engine.render_answer_chunked(&answer);
            // 流式输出后补一个换行（TokenChunk 分块输出无结尾换行）
            println!();
            Ok(())
        }
        Err(e) => {
            // max iterations：core 只返回错误，进度总结的生成/落盘/打印在展示层处理
            if let CogentError::MaxIterationsExceeded(iter) = e {
                emit_iteration_summary(engine, iter);
            }
            Err(anyhow::anyhow!("{e:#}"))
        }
    }
}

/// 捕获到 `MaxIterationsExceeded` 时：生成进度总结、写 `output/summary.md`、打印。
///
/// 从 core 引擎层移上来（core 只产出纯文本 `ToolRecord` 历史，
/// 文件落盘与标准输出属展示/IO 关注点，不进引擎）。
fn emit_iteration_summary(engine: &cogent_core::engine::AgentEngine, iterations: usize) {
    let summary = cogent_core::summary::generate_summary(iterations, engine.tool_history());
    let path = std::path::PathBuf::from("output").join("summary.md");

    // 写盘失败不致命：仍把总结打印到 stdout。
    let written = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &summary)
    })();

    println!("\n{summary}");
    match written {
        Ok(()) => println!("(iteration summary written to {})", path.display()),
        Err(e) => eprintln!("(failed to write summary file: {e})"),
    }
}

/// 打印已注册工具列表（`cog tools` 子命令）。
///
/// # 参数
/// - `tools`：已注册工具列表（`&[Arc<dyn Tool>]`）。
fn print_tools(tools: &[std::sync::Arc<dyn cogent_core::tool::Tool>]) {
    println!("{}", format!("{} registered tools:", tools.len()).bold());
    for tool in tools {
        println!();
        println!("  {}", tool.name().cyan().bold());
        // 描述（可能多行，缩进显示）
        for line in tool.description().lines() {
            println!("    {line}");
        }
        // 参数 Schema（JSON 格式化）
        let schema = tool.parameters_schema();
        let schema_str = serde_json::to_string_pretty(&schema).unwrap_or_default();
        for line in schema_str.lines() {
            println!("    {line}");
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("{}", format!("error: {e:#}").red());
        std::process::exit(1);
    }
}
