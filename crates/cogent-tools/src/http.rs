//! HTTP 工具：GET 请求。
//!
//! 通过 `#[cogent_tool]` 宏声明，发送 HTTP GET 请求并返回响应 body。
//!
//! # 安全约束
//! - 仅允许 `http` / `https` 协议（拒绝 `file://`、`ftp://` 等）。
//! - SSRF 防护：目标不得为私有/保留地址（回环、内网、链路本地、
//!   云元数据 `169.254.169.254`、多播等）。URL 来自 LLM 输出，属不可信输入。
//! - 禁用自动重定向：重定向目标可能指向内网（SSRF 绕过），
//!   3xx 响应原样返回，由调用方（LLM）决定是否换 URL 重试。
//! - 超时 15 秒。

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::time::Duration;

use cogent_macros::cogent_tool;

/// HTTP GET 默认超时（秒）。
const HTTP_TIMEOUT_SECS: u64 = 15;

/// 校验 URL 协议（http/https）并解析。
///
/// # 参数
/// - `url`：待校验的 URL。
///
/// # 返回
/// - 成功：解析后的 [`reqwest::Url`]。
/// - 失败：URL 非法或协议非 http/https。
fn validate_scheme(url: &str) -> anyhow::Result<reqwest::Url> {
    let parsed =
        reqwest::Url::parse(url).map_err(|e| anyhow::anyhow!("invalid URL '{url}': {e}"))?;
    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        anyhow::bail!("only http/https URLs are allowed, got: {url}");
    }
    Ok(parsed)
}

/// SSRF 防护：校验 URL 目标不是私有/保留地址。
///
/// IP 字面量直接校验；域名先 DNS 解析，再校验每个解析出的 IP。
///
/// # 已知局限
/// 本检查是"先解析、后请求"，两次 DNS 解析之间存在时间窗（DNS rebinding）。
/// 完全稳健的实现需在 reqwest 自定义 connector 中于建连时校验；
/// v1 采用此尽力而为的缓解措施。
async fn assert_public_target(parsed: &reqwest::Url) -> anyhow::Result<()> {
    // 用 url::Host 枚举类型安全地识别 IP 字面量（IPv4/IPv6）与域名。
    // 注意：不能用 host_str() 直接 parse——它对 IPv6 返回带方括号形式
    // （如 "[::1]"），parse::<IpAddr>() 会失败并误走 DNS 分支。
    match parsed.host() {
        // IP 字面量：直接校验，无需 DNS。
        Some(url::Host::Ipv4(ip)) => {
            if is_disallowed_ip(IpAddr::V4(ip)) {
                anyhow::bail!("URL target '{ip}' is a private/reserved address (SSRF blocked)");
            }
            Ok(())
        }
        Some(url::Host::Ipv6(ip)) => {
            if is_disallowed_ip(IpAddr::V6(ip)) {
                anyhow::bail!("URL target '{ip}' is a private/reserved address (SSRF blocked)");
            }
            Ok(())
        }
        // 域名：数字编码检查 + DNS 解析后逐个校验。
        Some(url::Host::Domain(domain)) => {
            let host: &str = domain;

            // 全数字 host（如 "2130706433" = 127.0.0.1）或 0x 前缀（如 "0x7f000001"）
            // 是 IP 的非标准数字编码，部分解析器会将其解释为内网地址；
            // 合法公网域名不会呈此形态，直接拒绝。
            if host.bytes().all(|b| b.is_ascii_digit())
                || host.to_ascii_lowercase().starts_with("0x")
            {
                anyhow::bail!(
                    "URL target '{host}' is a non-standard numeric IP encoding (SSRF blocked)"
                );
            }

            let lookup = (host, 80);
            let addrs = tokio::net::lookup_host(&lookup)
                .await
                .map_err(|e| anyhow::anyhow!("failed to resolve host '{host}': {e}"))?;
            let mut checked = false;
            for addr in addrs {
                checked = true;
                if is_disallowed_ip(addr.ip()) {
                    anyhow::bail!(
                        "URL target '{host}' resolves to private/reserved address {} (SSRF blocked)",
                        addr.ip()
                    );
                }
            }
            if !checked {
                anyhow::bail!("host '{host}' resolved to no addresses");
            }
            Ok(())
        }
        None => anyhow::bail!("URL has no host: {parsed}"),
    }
}

/// 判断 IP 是否为私有/保留地址（SSRF 拦截目标）。
fn is_disallowed_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_disallowed_ipv4(v4),
        IpAddr::V6(v6) => {
            // IPv4 映射地址（::ffff:a.b.c.d）按内嵌 IPv4 校验，
            // 防止以 IPv6 形态绕过 IPv4 内网段拦截。
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_disallowed_ipv4(v4);
            }
            is_disallowed_ipv6(v6)
        }
    }
}

/// IPv4 私有/保留段校验（RFC 1918 / RFC 5735 / RFC 6598 等）。
fn is_disallowed_ipv4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    // 0.0.0.0/8（本网络）
    if o[0] == 0 {
        return true;
    }
    // 10.0.0.0/8（内网）
    if o[0] == 10 {
        return true;
    }
    // 100.64.0.0/10（CGNAT，RFC 6598）
    if o[0] == 100 && (o[1] & 0xC0) == 64 {
        return true;
    }
    // 127.0.0.0/8（回环）
    if o[0] == 127 {
        return true;
    }
    // 169.254.0.0/16（链路本地，含云元数据 169.254.169.254）
    if o[0] == 169 && o[1] == 254 {
        return true;
    }
    // 172.16.0.0/12（内网）
    if o[0] == 172 && (16..=31).contains(&o[1]) {
        return true;
    }
    // 192.0.0.0/24（IETF 协议分配）、192.0.2.0/24（TEST-NET-1）、192.88.99.0/24（6to4 中继）
    if o[0] == 192 && ((o[1] == 0 && (o[2] == 0 || o[2] == 2)) || (o[1] == 88 && o[2] == 99)) {
        return true;
    }
    // 192.168.0.0/16（内网）
    if o[0] == 192 && o[1] == 168 {
        return true;
    }
    // 198.18.0.0/15（基准测试，RFC 2544）
    if o[0] == 198 && (o[1] == 18 || o[1] == 19) {
        return true;
    }
    // 198.51.100.0/24（TEST-NET-2）、203.0.113.0/24（TEST-NET-3）
    if o[0] == 198 && o[1] == 51 && o[2] == 100 {
        return true;
    }
    if o[0] == 203 && o[1] == 0 && o[2] == 113 {
        return true;
    }
    // 224.0.0.0/4（多播）、240.0.0.0/4（保留，含 255.255.255.255 广播）
    if o[0] >= 224 {
        return true;
    }
    false
}

/// IPv6 私有/保留段校验。
fn is_disallowed_ipv6(ip: Ipv6Addr) -> bool {
    let seg0 = ip.segments()[0];
    // ::/128（未指定）
    if ip.is_unspecified() {
        return true;
    }
    // ::1/128（回环）
    if ip.is_loopback() {
        return true;
    }
    // fc00::/7（唯一本地地址，ULA）
    if (seg0 & 0xFE00) == 0xFC00 {
        return true;
    }
    // fe80::/10（链路本地）
    if (seg0 & 0xFFC0) == 0xFE80 {
        return true;
    }
    // ff00::/8（多播）
    if (seg0 & 0xFF00) == 0xFF00 {
        return true;
    }
    // 64:ff9b::/96（NAT64，RFC 6052）：前 3 个 16 位段固定为 0x0064, 0xff9b, 0x0000
    let seg = ip.segments();
    if seg[0] == 0x0064 && seg[1] == 0xff9b && seg[2] == 0x0000 {
        return true;
    }
    false
}

/// 共享 HTTP 客户端（进程级单例，复用连接池）。
///
/// 禁用自动重定向：重定向目标可能指向内网（SSRF 绕过），
/// 3xx 响应原样返回，由调用方（LLM）决定是否换 URL 重试。
fn http_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("failed to build HTTP client")
    })
}

/// 发送 HTTP GET 请求并返回响应 body。
///
/// # 参数
/// - `url`：要请求的 URL（须为 http/https，且目标为公网地址）。
///
/// # 返回
/// - 成功：响应 body 文本（3xx 重定向响应原样返回，不自动跟随）。
/// - 失败：URL 协议非法、目标为私有/保留地址（SSRF）、请求超时或网络错误。
#[cogent_tool]
async fn http_get(url: String) -> anyhow::Result<String> {
    let parsed = validate_scheme(&url)?;
    assert_public_target(&parsed).await?;

    tracing::info!(url = %url, timeout_secs = HTTP_TIMEOUT_SECS, "sending HTTP GET request");

    let response = http_client()
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("HTTP GET request to '{url}' failed: {e}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| anyhow::anyhow!("failed to read response body from '{url}': {e}"))?;

    tracing::info!(
        url = %url,
        status_code = %status,
        body_len = body.len(),
        "HTTP GET request completed"
    );

    Ok(format!("status: {status}\n\n{body}"))
}

#[cfg(test)]
mod tests {
    //! HTTP 工具单元测试。
    //!
    //! 验证 URL 协议校验、SSRF 拦截与工具元数据。
    //! SSRF 测试仅用 IP 字面量（无需真实 DNS/网络）。

    use super::*;
    use cogent_core::tool::Tool;

    /// 验证 http 协议通过校验。
    #[test]
    fn test_validate_scheme_http() {
        assert!(validate_scheme("http://example.com").is_ok());
    }

    /// 验证 https 协议通过校验。
    #[test]
    fn test_validate_scheme_https() {
        assert!(validate_scheme("https://example.com").is_ok());
    }

    /// 验证 file 协议被拒绝。
    #[test]
    fn test_validate_scheme_file_rejected() {
        assert!(validate_scheme("file:///etc/passwd").is_err());
    }

    /// 验证 ftp 协议被拒绝。
    #[test]
    fn test_validate_scheme_ftp_rejected() {
        assert!(validate_scheme("ftp://example.com").is_err());
    }

    /// 验证 http_get 拒绝非 http/https URL（不发起真实请求）。
    #[tokio::test]
    async fn test_http_get_rejects_bad_protocol() {
        let tool = HttpGetTool;
        let result = tool
            .execute(serde_json::json!({"url": "file:///etc/passwd"}))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("only http/https"));
    }

    /// 验证 http_get 拒绝回环地址（SSRF，不发起真实请求）。
    #[tokio::test]
    async fn test_http_get_rejects_loopback() {
        let tool = HttpGetTool;
        let result = tool
            .execute(serde_json::json!({"url": "http://127.0.0.1/admin"}))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("SSRF blocked"));
    }

    /// 验证 is_disallowed_ipv4 覆盖各私有/保留段。
    #[test]
    fn test_is_disallowed_ipv4() {
        let blocked = [
            "0.0.0.0",
            "10.0.0.1",
            "10.255.255.255",
            "100.64.0.1",
            "100.127.255.255",
            "127.0.0.1",
            "169.254.169.254",
            "172.16.0.1",
            "172.31.255.255",
            "192.0.0.1",
            "192.0.2.44",
            "192.88.99.1",
            "192.168.1.1",
            "198.18.0.1",
            "198.19.255.255",
            "198.51.100.7",
            "203.0.113.9",
            "224.0.0.1",
            "239.255.255.255",
            "240.0.0.1",
            "255.255.255.255",
        ];
        for s in blocked {
            let ip: Ipv4Addr = s.parse().unwrap();
            assert!(is_disallowed_ipv4(ip), "{s} 应被拦截");
        }
        // 边界：紧邻私有段的公网地址应放行
        let allowed = [
            "1.1.1.1",
            "8.8.8.8",
            "9.255.255.255",
            "11.0.0.1",
            "100.128.0.1",
            "126.255.255.255",
            "128.0.0.1",
            "169.253.255.255",
            "169.255.0.1",
            "172.15.255.255",
            "172.32.0.1",
            "192.0.1.1",
            "192.1.0.1",
            "192.2.0.1",
            "192.3.0.1",
            "192.167.255.255",
            "192.169.0.1",
            "223.255.255.255",
        ];
        for s in allowed {
            let ip: Ipv4Addr = s.parse().unwrap();
            assert!(!is_disallowed_ipv4(ip), "{s} 应放行");
        }
    }

    /// 验证 is_disallowed_ipv6 覆盖各私有/保留段。
    #[test]
    fn test_is_disallowed_ipv6() {
        let blocked = [
            "::",
            "::1",
            "fe80::1",
            "fc00::1",
            "fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "ff02::1",
            "64:ff9b::1",
        ];
        for s in blocked {
            let ip: Ipv6Addr = s.parse().unwrap();
            assert!(is_disallowed_ipv6(ip), "{s} 应被拦截");
        }
        // 公网 IPv6 应放行
        let allowed = ["2001:4860:4860::8888", "2606:4700:4700::1111"];
        for s in allowed {
            let ip: Ipv6Addr = s.parse().unwrap();
            assert!(!is_disallowed_ipv6(ip), "{s} 应放行");
        }
    }

    /// 验证 IPv4 映射地址（::ffff:a.b.c.d）按内嵌 IPv4 拦截。
    #[test]
    fn test_ipv4_mapped_bypass_blocked() {
        // ::ffff:127.0.0.1 若按纯 IPv6 校验会放行，必须按内嵌 IPv4 拦截
        let ip: IpAddr = "::ffff:127.0.0.1".parse().unwrap();
        assert!(is_disallowed_ip(ip));
        let ip: IpAddr = "::ffff:169.254.169.254".parse().unwrap();
        assert!(is_disallowed_ip(ip));
        // 映射公网地址应放行
        let ip: IpAddr = "::ffff:8.8.8.8".parse().unwrap();
        assert!(!is_disallowed_ip(ip));
    }

    /// 验证 assert_public_target 拦截各类内网 IP 字面量（无需网络）。
    #[tokio::test]
    async fn test_assert_public_target_blocks_private_literals() {
        let blocked = [
            "http://127.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/",
            "http://172.16.0.1/",
            "http://192.168.1.1/",
            "http://100.64.0.1/",
            "http://0.0.0.0/",
            "http://224.0.0.1/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[fe80::1]/",
            "http://[fc00::1]/",
            // 非标准数字编码
            "http://2130706433/",
            "http://0x7f000001/",
        ];
        for url in blocked {
            let parsed = validate_scheme(url).unwrap();
            let err = assert_public_target(&parsed).await.err().unwrap();
            assert!(
                err.to_string().contains("SSRF blocked"),
                "{url} 应被拦截: {err}"
            );
        }
    }

    /// 验证 assert_public_target 放行公网 IP 字面量（无需网络）。
    #[tokio::test]
    async fn test_assert_public_target_allows_public_literals() {
        for url in [
            "http://8.8.8.8/",
            "https://1.1.1.1/",
            "http://[2001:4860:4860::8888]/",
        ] {
            let parsed = validate_scheme(url).unwrap();
            assert!(assert_public_target(&parsed).await.is_ok(), "{url} 应放行");
        }
    }

    /// 验证 http_get 工具名称。
    #[test]
    fn test_http_get_name() {
        let tool = HttpGetTool;
        assert_eq!(tool.name(), "http_get");
    }

    /// 验证 http_get 参数 Schema。
    #[test]
    fn test_http_get_schema() {
        let tool = HttpGetTool;
        let schema = tool.parameters_schema();
        assert!(schema["properties"].get("url").is_some());
        assert_eq!(schema["required"][0], "url");
    }

    /// 验证 assert_public_target 对 192.0.x/192.2.x 边界的精确拦截（回归：曾过度拦截整个 /16）。
    #[tokio::test]
    async fn test_assert_public_target_192_boundaries() {
        // 应拦截：仅 192.0.0.0/24、192.0.2.0/24、192.88.99.0/24
        for url in [
            "http://192.0.0.1/",
            "http://192.0.2.1/",
            "http://192.88.99.1/",
        ] {
            let parsed = validate_scheme(url).unwrap();
            assert!(
                assert_public_target(&parsed).await.is_err(),
                "{url} 应被拦截"
            );
        }
        // 应放行：紧邻保留段的公网地址（旧实现误拦整个 192.0.0.0/16 与 192.2.0.0/16）
        for url in [
            "http://192.0.1.1/",
            "http://192.1.0.1/",
            "http://192.2.0.1/",
            "http://192.3.0.1/",
        ] {
            let parsed = validate_scheme(url).unwrap();
            assert!(assert_public_target(&parsed).await.is_ok(), "{url} 应放行");
        }
    }

    /// 验证共享客户端不跟随重定向（回归：SSRF 可经 302 绕过初始目标校验）。
    ///
    /// 本地起一个最小 HTTP 服务返回 302 → 内网地址，断言客户端原样返回 302
    /// 而非跟随到 Location。
    #[tokio::test]
    async fn test_http_client_does_not_follow_redirects() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf).await;
            let response = "HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let response = http_client()
            .get(format!("http://{addr}/"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::FOUND);
        // 若客户端跟随了重定向，会尝试连接 169.254.169.254 并失败/挂起；
        // 此处能立即拿到 302 即证明未跟随。
        server.await.unwrap();
    }
}
