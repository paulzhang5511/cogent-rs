//! Bash 工具：执行 shell 命令。
//!
//! 通过 `#[cogent_tool]` 宏声明，执行一条 shell 命令并以结构化 JSON
//! （`{"exit_code":N,"stdout":"...","stderr":"..."}`）回传结果。
//!
//! # 安全约束
//! - 工作目录限定为进程当前工作目录（`std::env::current_dir()`）。
//! - 超时默认 120 秒（可用 `COGENT_BASH_TIMEOUT_SECS` 覆盖），超时后强制终止子进程。
//! - 命令白名单/黑名单策略拦截交给 `SafetyGuardMiddleware`（Q5 决策），
//!   工具内不硬编码白名单。
//!
//! # `cog-skill <name>` 桥接（有意保留）
//! 本工具会在 spawn shell **之前**拦截形如 `cog-skill <name>` 的命令（含链式前缀，
//! 如 `cd /x && cog-skill foo`），直接返回技能 SKILL.md 正文，而不执行 shell。
//!
//! 这与结构化的 `skill(name)` 工具**互补而非重复**：
//! - `skill(name)` 工具：标准结构化入口，工具可用时的首选；
//! - `cog-skill` 桥接：兜底通道——当模型只能依赖 shell 命令（如只读命令集
//!   不开放结构化工具）时，仍能通过一条命令按需加载技能正文。
//!
//! 之所以写在 bash 里而不单独抽工具，是为了兼容"模型只知道跑命令"的退化场景；
//! 代价是工具层依赖了 `cogent-skills` crate（已知耦合，见架构审查报告 m1）。

use std::time::Duration;

use cogent_macros::cogent_tool;

/// Bash 命令默认超时（秒）。
///
/// 递归文件操作（如 `Get-ChildItem -Recurse | Select-String` 在大仓库上）
/// 可能超过 30s，故默认放宽到 120s。可通过环境变量 `COGENT_BASH_TIMEOUT_SECS`
/// 覆盖（见 [`bash_timeout_secs`]）。
const BASH_TIMEOUT_SECS: u64 = 120;

/// 环境变量名：覆盖 bash 命令超时（秒）。
const BASH_TIMEOUT_ENV: &str = "COGENT_BASH_TIMEOUT_SECS";

/// stdout/stderr 各自的最大保留字节数（超出部分截断）。
///
/// 防止 `cat 大文件`、`ls -R` 等命令产生超大输出撑爆内存（OOM）。
/// 截断后在输出末尾追加标记，让 LLM 知道内容被截断。
const BASH_OUTPUT_MAX_BYTES: usize = 1024 * 1024; // 1 MiB

/// 截断过大的输出，超出 [`BASH_OUTPUT_MAX_BYTES`] 的部分丢弃并追加标记。
///
/// 直接在字节边界截断后用 `from_utf8_lossy` 转换：若截断点落在多字节字符中间，
/// 该残缺字符会被替换为 U+FFFD，不会 panic 或产生非法 UTF-8。
fn truncate_output(bytes: Vec<u8>) -> String {
    if bytes.len() <= BASH_OUTPUT_MAX_BYTES {
        return String::from_utf8_lossy(&bytes).into_owned();
    }
    let kept = String::from_utf8_lossy(&bytes[..BASH_OUTPUT_MAX_BYTES]).into_owned();
    let truncated_bytes = bytes.len() - BASH_OUTPUT_MAX_BYTES;
    format!(
        "{kept}\n...[truncated {truncated_bytes} bytes, output exceeded {BASH_OUTPUT_MAX_BYTES} bytes]"
    )
}

/// 解析 bash 命令超时（秒）。
///
/// 优先读环境变量 `COGENT_BASH_TIMEOUT_SECS`（正整数），缺失或非法时回退
/// 默认 [`BASH_TIMEOUT_SECS`]。
///
/// # 返回
/// 超时秒数（≥ 1）。
fn bash_timeout_secs() -> u64 {
    std::env::var(BASH_TIMEOUT_ENV)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(BASH_TIMEOUT_SECS)
}

/// 拦截 `cog-skill <name>` 命令，返回技能完整内容（不 spawn shell）。
///
/// 部分内置工具集（只读命令类）无法直接调用 `skill` 工具，但能通过
/// shell 命令执行。此拦截让模型用 `cog-skill <name>` 按需
/// 加载任意工程技能的完整 SKILL.md 正文（含辅助文件），实现"技能真实用到"。
///
/// 支持链式命令（如 `cd /x && cog-skill foo`）：按 shell 分隔符切分子命令，
/// 取首个以 `cog-skill` 开头的子命令。
///
/// # 参数
/// - `command`：原始命令字符串。
///
/// # 返回
/// - 命中 `cog-skill`：`Some(结构化 JSON)`（与 bash 输出格式一致）。
/// - 非 `cog-skill` 命令：`None`（走正常 shell 执行路径）。
fn try_handle_cog_skill(command: &str) -> Option<String> {
    let name = extract_cog_skill_name(command)?;

    let (exit_code, stdout, stderr) = match cogent_skills::render_skill(&name) {
        Ok(content) => {
            if let Some(e) = cogent_skills::get_skill(&name) {
                tracing::info!(
                    skill = %e.meta.name,
                    phase = %e.meta.phase,
                    content_len = e.content.len(),
                    auxiliary_count = e.auxiliary.len(),
                    "cog-skill: loading skill"
                );
            }
            (0, content, String::new())
        }
        Err(e) => (1, String::new(), e),
    };

    Some(
        serde_json::json!({
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        })
        .to_string(),
    )
}

/// 从命令中提取 `cog-skill` 的技能名。
///
/// 按 shell 分隔符（`&&`/`||`/`;`/`|`/换行）切分子命令，取首个以
/// `cog-skill` 开头的子命令的下一个 token 作为技能名。处理
/// `cd /x && cog-skill foo` 这类前缀命令。
///
/// # 返回
/// - 命中：`Some(技能名)`（无参数时为空串）。
/// - 未命中：`None`。
fn extract_cog_skill_name(command: &str) -> Option<String> {
    for segment in split_shell_segments(command) {
        let mut tokens = segment.split_whitespace();
        if tokens.next() == Some("cog-skill") {
            return Some(tokens.next().unwrap_or("").to_string());
        }
    }
    None
}

/// 按 shell 分隔符（`&&`/`||`/`;`/`|`/换行）切分命令为子命令段。
///
/// 仅用于 `cog-skill` 拦截的命令识别，非完整 shell 解析（不处理引号内
/// 分隔符等边界，对拦截场景足够）。
fn split_shell_segments(command: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0;
    let bytes = command.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'&' if i + 1 < bytes.len() && bytes[i + 1] == b'&' => {
                segments.push(&command[start..i]);
                i += 2;
                start = i;
            }
            b'|' if i + 1 < bytes.len() && bytes[i + 1] == b'|' => {
                segments.push(&command[start..i]);
                i += 2;
                start = i;
            }
            b'|' | b';' | b'\n' => {
                segments.push(&command[start..i]);
                i += 1;
                start = i;
            }
            _ => i += 1,
        }
    }
    segments.push(&command[start..]);
    segments
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

/// 执行一条 shell 命令并返回其输出。
///
/// # 参数
/// - `command`：要执行的 shell 命令字符串。
///
/// # 返回
/// - 成功：结构化 JSON 字符串 `{"exit_code":N,"stdout":"...","stderr":"..."}`。
/// - 失败：命令执行错误（超时、启动失败等）。
#[cogent_tool]
async fn bash(command: String) -> anyhow::Result<String> {
    // 工作目录限定为进程当前工作目录
    let working_dir = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current working directory: {e}"))?;

    // 超时（秒）：默认 120s，可用 COGENT_BASH_TIMEOUT_SECS 覆盖
    let timeout_secs = bash_timeout_secs();

    tracing::info!(
        command = %command,
        working_dir = %working_dir.display(),
        timeout_secs,
        "executing bash command"
    );

    // 拦截 cog-skill 命令：返回技能完整内容，不 spawn shell
    if let Some(result) = try_handle_cog_skill(&command) {
        return Ok(result);
    }

    // 启动子进程（跨平台：Windows 用 powershell -NoProfile -Command，Unix 用 sh -c）
    // Windows 不用 cmd /c：Rust 的 CreateProcess 引号转义（反斜杠转义）与 cmd.exe
    // 不兼容，嵌套引号（如 powershell -Command "..."）会被破坏。PowerShell 原生
    // 处理引号，且能运行 cmd 命令，兼容性更好。
    #[cfg(windows)]
    let mut cmd = tokio::process::Command::new("powershell");
    #[cfg(windows)]
    cmd.args(["-NoProfile", "-Command"]);

    #[cfg(not(windows))]
    let mut cmd = tokio::process::Command::new("sh");
    #[cfg(not(windows))]
    cmd.arg("-c");

    cmd.arg(&command).current_dir(&working_dir);
    // tokio::process::Command 默认不接管 stdout/stderr（与 std 不同），
    // 必须显式 piped 才能捕获输出。
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn bash command: {e}"))?;

    // 取出 stdout/stderr 句柄，与 wait 并发读取：若只 wait 不读管道，子进程写满
    // 管道缓冲区后会阻塞，wait 永不返回（死锁）。
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let result = tokio::time::timeout(Duration::from_secs(timeout_secs), async {
        let (status, stdout, stderr) = tokio::join!(
            child.wait(),
            async {
                let mut buf = Vec::new();
                if let Some(mut s) = stdout {
                    tokio::io::AsyncReadExt::read_to_end(&mut s, &mut buf).await?;
                }
                Ok::<_, std::io::Error>(buf)
            },
            async {
                let mut buf = Vec::new();
                if let Some(mut s) = stderr {
                    tokio::io::AsyncReadExt::read_to_end(&mut s, &mut buf).await?;
                }
                Ok::<_, std::io::Error>(buf)
            },
        );
        (status, stdout, stderr)
    })
    .await;

    let (status, stdout, stderr) = match result {
        Ok(r) => r,
        Err(_) => {
            // 超时：显式 kill 子进程（drop 不会终止它，仅 drop 会留孤儿进程），
            // 再 wait 回收，避免僵尸进程。
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(anyhow::anyhow!(
                "bash command timed out after {timeout_secs}s: {command}"
            ));
        }
    };

    let status = status.map_err(|e| anyhow::anyhow!("failed to wait for bash command: {e}"))?;
    let stdout = stdout.map_err(|e| anyhow::anyhow!("failed to read bash stdout: {e}"))?;
    let stderr = stderr.map_err(|e| anyhow::anyhow!("failed to read bash stderr: {e}"))?;

    let exit_code = status.code().unwrap_or(-1);
    let stdout = truncate_output(stdout);
    let stderr = truncate_output(stderr);

    tracing::info!(
        exit_code,
        stdout_len = stdout.len(),
        stderr_len = stderr.len(),
        "bash command completed"
    );

    // 结构化 JSON 输出：stdout/stderr 作为独立字段回传，无歧义。
    // 旧有损文本格式（`exit code: N\nstdout:\n<out>\nstderr:\n<err>`，段标记仅在
    // 非空时出现）在"stdout 含字面量 `stderr:\n` 且无真 stderr"时不可区分；
    // JSON 彻底解决该问题。
    let output = serde_json::json!({
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
    })
    .to_string();

    // 非零退出码视为失败：返回 Err（携带完整 JSON 输出），让引擎标记工具失败、
    // 模型看到错误并纠正。若以 Ok 返回，引擎 `result.is_ok()` 判成功，模型误以为
    // 命令成功（实测 log：exit_code=1 仍显示 "tool executed successfully"，
    // 导致文件被反复写坏）。
    if exit_code != 0 {
        return Err(anyhow::anyhow!(
            "command exited with code {exit_code}: {output}"
        ));
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    //! Bash 工具单元测试。
    //!
    //! 验证命令执行、退出码捕获、超时处理。

    use super::*;
    use cogent_core::tool::Tool;

    /// 验证 `cog-skill <name>` 命中已知技能：返回 exit_code=0 + 技能正文。
    #[test]
    fn test_cog_skill_hit() {
        let result = try_handle_cog_skill("cog-skill test-driven-development").unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["exit_code"], 0);
        assert!(
            v["stdout"]
                .as_str()
                .unwrap()
                .contains("Test-Driven Development")
        );
        assert!(v["stderr"].as_str().unwrap().is_empty());
    }

    /// 验证 `cog-skill` 含辅助文件的技能：正文含 `---` 分隔符。
    #[test]
    fn test_cog_skill_with_auxiliary() {
        let result = try_handle_cog_skill("cog-skill idea-refine").unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        let stdout = v["stdout"].as_str().unwrap();
        assert!(stdout.contains("Idea Refine"));
        assert!(stdout.contains("\n\n---\n\n"));
    }

    /// 验证 `cog-skill` 未知技能：exit_code=1 + stderr 含可用列表。
    #[test]
    fn test_cog_skill_unknown() {
        let result = try_handle_cog_skill("cog-skill nonexistent-skill").unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["exit_code"], 1);
        assert!(v["stderr"].as_str().unwrap().contains("unknown skill"));
        assert!(
            v["stderr"]
                .as_str()
                .unwrap()
                .contains("test-driven-development")
        );
    }

    /// 验证 `cog-skill` 无参数：exit_code=1 + 提示。
    #[test]
    fn test_cog_skill_no_name() {
        let result = try_handle_cog_skill("cog-skill").unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["exit_code"], 1);
        assert!(v["stderr"].as_str().unwrap().contains("unknown skill"));
    }

    /// 验证非 `cog-skill` 命令返回 None（走正常 shell 路径）。
    #[test]
    fn test_cog_skill_non_matching() {
        assert!(try_handle_cog_skill("echo hello").is_none());
        assert!(try_handle_cog_skill("cog-skillx foo").is_none());
        assert!(try_handle_cog_skill("").is_none());
        assert!(try_handle_cog_skill("  ").is_none());
    }

    /// 验证 `cog-skill` 命令带前导空白也能命中。
    #[test]
    fn test_cog_skill_leading_whitespace() {
        let result = try_handle_cog_skill("   cog-skill spec-driven-development").unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["exit_code"], 0);
        assert!(
            v["stdout"]
                .as_str()
                .unwrap()
                .contains("Spec-Driven Development")
        );
    }

    /// 验证链式命令 `cd /x && cog-skill foo` 也能命中（#1 修复）。
    #[test]
    fn test_cog_skill_chained_command() {
        let result = try_handle_cog_skill("cd /x && cog-skill test-driven-development").unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["exit_code"], 0);
        assert!(
            v["stdout"]
                .as_str()
                .unwrap()
                .contains("Test-Driven Development")
        );
    }

    /// 验证 `cog-skill` 在链式命令中间段也能命中。
    #[test]
    fn test_cog_skill_mid_chain() {
        let result = try_handle_cog_skill("echo hi; cog-skill idea-refine").unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["exit_code"], 0);
        assert!(v["stdout"].as_str().unwrap().contains("Idea Refine"));
    }

    /// 验证 `split_shell_segments` 按 `&&`/`||`/`;`/`|`/换行切分。
    #[test]
    fn test_split_shell_segments() {
        assert_eq!(
            split_shell_segments("a && b || c; d | e"),
            vec!["a", "b", "c", "d", "e"]
        );
        assert_eq!(split_shell_segments("a\nb"), vec!["a", "b"]);
        assert_eq!(split_shell_segments("  "), Vec::<&str>::new());
        assert_eq!(split_shell_segments("single"), vec!["single"]);
    }

    /// 验证 `extract_cog_skill_name` 提取技能名（含链式）。
    #[test]
    fn test_extract_cog_skill_name() {
        assert_eq!(
            extract_cog_skill_name("cog-skill foo").as_deref(),
            Some("foo")
        );
        assert_eq!(
            extract_cog_skill_name("cd /x && cog-skill bar").as_deref(),
            Some("bar")
        );
        assert_eq!(extract_cog_skill_name("cog-skill").as_deref(), Some(""));
        assert_eq!(extract_cog_skill_name("echo hi"), None);
    }

    /// 验证 truncate_output：未超限时原样返回。
    #[test]
    fn test_truncate_output_under_limit() {
        let out = truncate_output(b"hello world".to_vec());
        assert_eq!(out, "hello world");
    }

    /// 验证 truncate_output：超限时截断并追加标记，且保留前缀。
    #[test]
    fn test_truncate_output_over_limit() {
        let big = vec![b'a'; BASH_OUTPUT_MAX_BYTES + 100];
        let out = truncate_output(big);
        assert!(out.starts_with(&"a".repeat(100)), "应保留前缀");
        assert!(out.contains("truncated 100 bytes"), "应含截断标记: {out}");
        // 截断后总长应远小于原始大小（前缀 + 标记）
        assert!(out.len() < BASH_OUTPUT_MAX_BYTES + 200);
    }

    /// 验证 truncate_output：截断点落在多字节字符中间时不 panic（替换为 U+FFFD）。
    #[test]
    fn test_truncate_output_multibyte_boundary() {
        // 用多字节字符填满，确保截断点大概率落在字符中间
        let big = "中".repeat(BASH_OUTPUT_MAX_BYTES / 3 + 10);
        let bytes = big.into_bytes();
        let out = truncate_output(bytes);
        // 不 panic 即通过；结果应为合法 String
        assert!(!out.is_empty());
    }

    /// 验证简单命令执行成功，输出为结构化 JSON。
    #[tokio::test]
    async fn test_bash_echo() {
        let tool = BashTool;
        let result = tool
            .execute(serde_json::json!({"command": "echo hello"}))
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).expect("bash 输出应为 JSON");
        assert_eq!(v["exit_code"], 0);
        assert!(v["stdout"].as_str().unwrap().contains("hello"));
    }

    /// 验证非零退出码返回 Err（携带含 exit_code 的 JSON 输出）。
    #[tokio::test]
    async fn test_bash_nonzero_exit() {
        let tool = BashTool;
        let result = tool
            .execute(serde_json::json!({"command": "exit 42"}))
            .await;
        let err = result.expect_err("非零退出码应返回 Err");
        let msg = err.to_string();
        assert!(msg.contains("exited with code 42"), "unexpected: {msg}");
        assert!(msg.contains("\"exit_code\":42"), "unexpected: {msg}");
    }

    /// 验证 stderr 捕获（JSON 格式）。
    #[tokio::test]
    async fn test_bash_stderr() {
        // Windows 用 PowerShell（[Console]::Error.WriteLine 写 stderr），
        // Unix 用 sh（echo error >&2）。
        #[cfg(windows)]
        let command = "[Console]::Error.WriteLine('error')";
        #[cfg(not(windows))]
        let command = "echo error >&2";
        let tool = BashTool;
        let result = tool
            .execute(serde_json::json!({"command": command}))
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).expect("bash 输出应为 JSON");
        assert!(v["stderr"].as_str().unwrap().contains("error"));
    }

    /// 验证 stdout 含字面量 "stderr:\n" 且无真 stderr 时，JSON 格式能正确区分
    /// （旧有损文本格式在此场景不可解，JSON 结构化回传彻底解决）。
    #[tokio::test]
    async fn test_bash_stdout_contains_stderr_marker() {
        #[cfg(windows)]
        let command = "[Console]::Out.WriteLine('line1'); [Console]::Out.WriteLine('stderr:'); [Console]::Out.WriteLine('forged')";
        #[cfg(not(windows))]
        let command = "printf 'line1\\nstderr:\\nforged\\n'";
        let tool = BashTool;
        let result = tool
            .execute(serde_json::json!({"command": command}))
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).expect("bash 输出应为 JSON");
        let stdout = v["stdout"].as_str().unwrap();
        // Windows PowerShell 输出 CRLF，Unix 输出 LF；统一换行后断言。
        let normalized = stdout.replace("\r\n", "\n");
        // stdout 段保留完整内容（含伪造的 "stderr:\nforged"）
        assert!(normalized.contains("stderr:\nforged"));
        // 无真 stderr，stderr 字段应为空
        assert_eq!(v["stderr"].as_str().unwrap(), "");
    }

    /// 验证工具名称。
    #[test]
    fn test_bash_name() {
        let tool = BashTool;
        assert_eq!(tool.name(), "bash");
    }

    /// 验证工具描述非空。
    #[test]
    fn test_bash_description() {
        let tool = BashTool;
        assert!(!tool.description().is_empty());
    }

    /// 验证参数 Schema 包含 command 字段。
    #[test]
    fn test_bash_schema() {
        let tool = BashTool;
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"].get("command").is_some());
        assert_eq!(schema["required"][0], "command");
    }

    /// 串行化依赖环境变量的测试，避免并行竞争（edition 2024 中 set_var 影响进程全局）。
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// 验证 bash_timeout_secs：环境变量缺失时回退默认 120s。
    #[test]
    fn test_bash_timeout_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        let original = std::env::var(BASH_TIMEOUT_ENV).ok();
        unsafe { std::env::remove_var(BASH_TIMEOUT_ENV) };
        assert_eq!(bash_timeout_secs(), BASH_TIMEOUT_SECS);
        assert_eq!(BASH_TIMEOUT_SECS, 120);
        if let Some(v) = original {
            unsafe { std::env::set_var(BASH_TIMEOUT_ENV, v) };
        }
    }

    /// 验证 bash_timeout_secs：环境变量为正整数时生效。
    #[test]
    fn test_bash_timeout_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        let original = std::env::var(BASH_TIMEOUT_ENV).ok();
        unsafe { std::env::set_var(BASH_TIMEOUT_ENV, "45") };
        assert_eq!(bash_timeout_secs(), 45);
        if let Some(v) = original {
            unsafe { std::env::set_var(BASH_TIMEOUT_ENV, v) };
        } else {
            unsafe { std::env::remove_var(BASH_TIMEOUT_ENV) };
        }
    }

    /// 验证 bash_timeout_secs：非法值（非数字 / 0）回退默认。
    #[test]
    fn test_bash_timeout_env_invalid_falls_back() {
        let _guard = ENV_LOCK.lock().unwrap();
        let original = std::env::var(BASH_TIMEOUT_ENV).ok();
        for bad in ["not-a-number", "0", "-5"] {
            unsafe { std::env::set_var(BASH_TIMEOUT_ENV, bad) };
            assert_eq!(bash_timeout_secs(), BASH_TIMEOUT_SECS, "bad value {bad}");
        }
        if let Some(v) = original {
            unsafe { std::env::set_var(BASH_TIMEOUT_ENV, v) };
        } else {
            unsafe { std::env::remove_var(BASH_TIMEOUT_ENV) };
        }
    }

    /// 验证超时真正终止子进程：长命令应在超时后被 kill 并返回超时错误，
    /// 而非挂起或留下孤儿进程。
    ///
    /// 锁跨 await 持有是有意为之（环境变量串行化隔离），单线程 runtime 下不会死锁。
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn test_bash_timeout_kills_child() {
        let _guard = ENV_LOCK.lock().unwrap();
        let original = std::env::var(BASH_TIMEOUT_ENV).ok();
        unsafe { std::env::set_var(BASH_TIMEOUT_ENV, "1") };

        let tool = BashTool;
        // 跨平台长命令：Windows 用 Start-Sleep，Unix 用 sleep。
        #[cfg(windows)]
        let cmd = "Start-Sleep -Seconds 30";
        #[cfg(not(windows))]
        let cmd = "sleep 30";

        let start = std::time::Instant::now();
        let result = tool.execute(serde_json::json!({"command": cmd})).await;
        let elapsed = start.elapsed();

        if let Some(v) = original {
            unsafe { std::env::set_var(BASH_TIMEOUT_ENV, v) };
        } else {
            unsafe { std::env::remove_var(BASH_TIMEOUT_ENV) };
        }

        let err = result.expect_err("超时命令应返回错误");
        assert!(err.to_string().contains("timed out"), "应为超时错误: {err}");
        // 应在 ~1s 超时后立即返回，而非等满 30s（证明子进程被 kill）。
        assert!(elapsed.as_secs() < 10, "超时后应快速返回，实际 {elapsed:?}");
    }
}
