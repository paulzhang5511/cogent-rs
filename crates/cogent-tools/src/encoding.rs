//! 文件编码检测模块。
//!
//! 检测目标文件编码（UTF-8 / UTF-8-BOM / GBK），写入时保持原编码，
//! 避免中文损坏（实测：GBK 文件被按 UTF-8 写入导致中文乱码）。
//!
//! # 检测策略（按字节有效性，优先级从高到低）
//! 1. **BOM 检测**：文件头 3 字节为 `EF BB BF` → `Utf8Bom`（BOM 是权威标志）。
//! 2. **UTF-8 验证**：`String::from_utf8` 严格解码成功 → `Utf8`。
//! 3. **GBK 回退**：UTF-8 解码失败（GBK 中文字节通常非合法 UTF-8）→
//!    尝试 GBK 解码，无错误 → `Gbk`。
//! 4. **未知**：均失败 → `EncodingError::Unknown`。
//!
//! # 设计说明
//! 检测基于**字节有效性**而非 U+FFFB 启发式：GBK 中文字节序列通常不构成
//! 合法 UTF-8，`String::from_utf8` 会失败，从而可靠地落入 GBK 回退分支。
//! U+FFFB 替换符是模型输出损坏的标志，由 [`crate::sanitize`] 模块处理，
//! 不在此处作为编码判定依据（避免把含 U+FFFB 的合法 UTF-8 文件误判为 GBK）。

use std::path::Path;

use thiserror::Error;

/// 文件编码类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// UTF-8（无 BOM）。
    Utf8,
    /// UTF-8 带 BOM（`EF BB BF`）。
    Utf8Bom,
    /// GBK（中文 Windows 常见编码）。
    Gbk,
}

/// 编码检测或读写失败。
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EncodingError {
    /// 文件读取失败（IO 错误）。
    ///
    /// `0` 为底层 IO 错误描述。
    #[error("failed to read file: {0}")]
    Io(String),

    /// 文件写入失败（IO 错误）。
    ///
    /// `0` 为底层 IO 错误描述。
    #[error("failed to write file: {0}")]
    Write(String),

    /// 无法识别文件编码（非 UTF-8 且非 GBK）。
    #[error("unable to detect file encoding (not valid UTF-8 or GBK)")]
    Unknown,
}

/// 检测文件编码。
///
/// # 参数
/// - `path`：文件路径。
///
/// # 返回
/// - 成功：检测到的 [`Encoding`]。
/// - 失败：[`EncodingError`]（IO 错误或无法识别编码）。
pub fn detect_encoding(path: &Path) -> Result<Encoding, EncodingError> {
    let bytes = std::fs::read(path).map_err(|e| EncodingError::Io(e.to_string()))?;
    detect_encoding_from_bytes(&bytes)
}

/// 从字节内容检测编码（核心逻辑，接收字节便于测试）。
///
/// # 参数
/// - `bytes`：文件原始字节。
///
/// # 返回
/// - 成功：检测到的 [`Encoding`]。
/// - 失败：[`EncodingError::Unknown`]（非 UTF-8 且非 GBK）。
pub fn detect_encoding_from_bytes(bytes: &[u8]) -> Result<Encoding, EncodingError> {
    // 1. BOM 检测（权威标志，优先于内容验证）
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Ok(Encoding::Utf8Bom);
    }
    // 2. UTF-8 严格验证
    if std::str::from_utf8(bytes).is_ok() {
        return Ok(Encoding::Utf8);
    }
    // 3. GBK 回退（GBK 中文字节通常非合法 UTF-8，落入此分支）
    // decode 返回 (Cow<str>, &Encoding, lossy: bool)，lossy=false 表示无错误
    let (_cow, _enc, lossy) = encoding_rs::GBK.decode(bytes);
    if !lossy {
        return Ok(Encoding::Gbk);
    }
    // 4. 未知
    Err(EncodingError::Unknown)
}

/// 按检测到的编码读取文件内容。
///
/// # 参数
/// - `path`：文件路径。
///
/// # 返回
/// - 成功：`(文件内容文本, 检测到的编码)`。
/// - 失败：[`EncodingError`]。
pub fn read_with_encoding(path: &Path) -> Result<(String, Encoding), EncodingError> {
    let bytes = std::fs::read(path).map_err(|e| EncodingError::Io(e.to_string()))?;
    let encoding = detect_encoding_from_bytes(&bytes)?;
    let text = decode_bytes(&bytes, encoding);
    Ok((text, encoding))
}

/// 按指定编码写入文件内容（保持原编码）。
///
/// # 参数
/// - `path`：文件路径。
/// - `content`：要写入的文本内容。
/// - `encoding`：目标编码（应与原文件一致，避免编码转换）。
///
/// # 返回
/// - 成功：`()`。
/// - 失败：[`EncodingError::Write`]（IO 错误）。
pub fn write_with_encoding(
    path: &Path,
    content: &str,
    encoding: Encoding,
) -> Result<(), EncodingError> {
    let bytes = encode_content(content, encoding);
    std::fs::write(path, &bytes).map_err(|e| EncodingError::Write(e.to_string()))?;
    Ok(())
}

/// 按指定编码解码字节为文本。
fn decode_bytes(bytes: &[u8], encoding: Encoding) -> String {
    match encoding {
        Encoding::Utf8 => String::from_utf8_lossy(bytes).into_owned(),
        Encoding::Utf8Bom => {
            // 跳过 BOM（3 字节）后按 UTF-8 解码
            let without_bom = &bytes[3.min(bytes.len())..];
            String::from_utf8_lossy(without_bom).into_owned()
        }
        Encoding::Gbk => encoding_rs::GBK.decode(bytes).0.into_owned(),
    }
}

/// 按指定编码将文本编码为字节。
fn encode_content(content: &str, encoding: Encoding) -> Vec<u8> {
    match encoding {
        Encoding::Utf8 => content.as_bytes().to_vec(),
        Encoding::Utf8Bom => {
            let mut bytes = vec![0xEF, 0xBB, 0xBF];
            bytes.extend_from_slice(content.as_bytes());
            bytes
        }
        Encoding::Gbk => {
            let (cow, _, _) = encoding_rs::GBK.encode(content);
            cow.as_ref().to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    //! `encoding-detect` 单元测试。
    //!
    //! 覆盖 UTF-8 / UTF-8-BOM / GBK 检测、写入保持编码、未知编码等成功标准。

    use super::*;

    /// 验证 UTF-8 无 BOM 文件检测为 `Utf8`。
    #[test]
    fn test_detect_utf8() {
        let bytes = "hello 中文".as_bytes();
        assert_eq!(detect_encoding_from_bytes(bytes).unwrap(), Encoding::Utf8);
    }

    /// 验证 UTF-8 BOM 文件检测为 `Utf8Bom`。
    #[test]
    fn test_detect_utf8_bom() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice("hello".as_bytes());
        assert_eq!(
            detect_encoding_from_bytes(&bytes).unwrap(),
            Encoding::Utf8Bom
        );
    }

    /// 验证 GBK 文件（含中文）检测为 `Gbk`。
    #[test]
    fn test_detect_gbk() {
        // "万象棋" 的 GBK 字节（非合法 UTF-8，应落入 GBK 回退）
        let (cow, _, _) = encoding_rs::GBK.encode("万象棋");
        let gbk_bytes = cow.as_ref().to_vec();
        // 前置断言：这些字节确实非合法 UTF-8（否则测试无效）
        assert!(std::str::from_utf8(&gbk_bytes).is_err());
        assert_eq!(
            detect_encoding_from_bytes(&gbk_bytes).unwrap(),
            Encoding::Gbk
        );
    }

    /// 验证无法识别的字节序列返回 `Unknown`。
    #[test]
    fn test_detect_unknown() {
        // 0xFF 0xFE 0xFF 既非 UTF-8 BOM（EF BB BF），也非合法 UTF-8，
        // 且 GBK 解码 0xFF 0xFE 会产生错误
        let bytes = [0xFFu8, 0xFE, 0xFF, 0x00, 0x81];
        assert_eq!(
            detect_encoding_from_bytes(&bytes),
            Err(EncodingError::Unknown)
        );
    }

    /// 验证 `read_with_encoding` 读取 UTF-8 文件。
    #[test]
    fn test_read_utf8() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("utf8.txt");
        std::fs::write(&file, "hello 中文").unwrap();
        let (text, encoding) = read_with_encoding(&file).unwrap();
        assert_eq!(text, "hello 中文");
        assert_eq!(encoding, Encoding::Utf8);
    }

    /// 验证 `read_with_encoding` 读取 GBK 文件（中文正确解码）。
    #[test]
    fn test_read_gbk() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("gbk.txt");
        let (cow, _, _) = encoding_rs::GBK.encode("万象棋夺冠");
        std::fs::write(&file, cow.as_ref()).unwrap();
        let (text, encoding) = read_with_encoding(&file).unwrap();
        assert_eq!(text, "万象棋夺冠");
        assert_eq!(encoding, Encoding::Gbk);
    }

    /// 验证 `write_with_encoding` 写入 GBK 文件后仍为 GBK（编码保持）。
    #[test]
    fn test_write_gbk_preserves_encoding() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("gbk_out.txt");
        write_with_encoding(&file, "万象棋", Encoding::Gbk).unwrap();
        // 重新检测应为 GBK
        assert_eq!(detect_encoding(&file).unwrap(), Encoding::Gbk);
        // 内容正确
        let (text, _) = read_with_encoding(&file).unwrap();
        assert_eq!(text, "万象棋");
    }

    /// 验证 `write_with_encoding` 写入 UTF-8 BOM 文件时保留 BOM。
    #[test]
    fn test_write_utf8_bom_preserves_bom() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("bom_out.txt");
        write_with_encoding(&file, "hello", Encoding::Utf8Bom).unwrap();
        let bytes = std::fs::read(&file).unwrap();
        assert!(bytes.starts_with(&[0xEF, 0xBB, 0xBF]));
        // 重新检测应为 Utf8Bom
        assert_eq!(detect_encoding(&file).unwrap(), Encoding::Utf8Bom);
        // 内容正确（BOM 被跳过）
        let (text, _) = read_with_encoding(&file).unwrap();
        assert_eq!(text, "hello");
    }

    /// 验证 `write_with_encoding` 写入 UTF-8 文件（无 BOM）。
    #[test]
    fn test_write_utf8_no_bom() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("utf8_out.txt");
        write_with_encoding(&file, "hello 中文", Encoding::Utf8).unwrap();
        let bytes = std::fs::read(&file).unwrap();
        assert!(!bytes.starts_with(&[0xEF, 0xBB, 0xBF]));
        assert_eq!(detect_encoding(&file).unwrap(), Encoding::Utf8);
    }

    /// 验证空文件检测为 `Utf8`（空字节是合法 UTF-8）。
    #[test]
    fn test_detect_empty_file() {
        assert_eq!(detect_encoding_from_bytes(&[]).unwrap(), Encoding::Utf8);
    }
}
