//! Cogent 过程宏 crate。
//!
//! 提供 `#[cogent_tool]` 过程宏：标注在 `async fn` 上，自动从函数签名
//! 与 doc 注释生成实现 [`cogent_core::tool::Tool`] 的结构体，并用
//! `schemars` 派生参数 JSON Schema。
//!
//! # 用法
//!
//! ```rust,ignore
//! use cogent_macros::cogent_tool;
//!
//! /// 执行一条 shell 命令并返回其输出
//! #[cogent_tool]
//! async fn bash(command: String) -> anyhow::Result<String> {
//!     // command 由 execute 从 args 反序列化得到
//!     // ...
//! }
//! // 生成：
//! //   pub struct BashTool;
//! //   impl Tool for BashTool {
//! //       fn name(&self) -> &str { "bash" }
//! //       async fn execute(&self, args: Value) -> anyhow::Result<String> {
//! //           let command: String = /* 从 args["command"] 反序列化 */;
//! //           /* 原函数体 */
//! //       }
//! //   }
//! ```
//!
//! # 属性参数
//!
//! - `name = "..."`：覆盖工具名称（默认为函数名）。
//!
//! # 生成规则
//!
//! - 函数参数为工具入参：字段名即参数名，Schema 由 `schemars::schema_for!` 派生，
//!   所有入参均标记为 `required`。
//! - 生成的结构体为**单元结构体**（无字段）；`execute` 在调用时从 `args`
//!   反序列化各入参为局部变量，再执行原函数体。
//! - 函数体直接嵌入生成的 `execute` 方法，返回类型须为 `anyhow::Result<String>`
//!   （或可经 `?` 转换为 `anyhow::Error` 的 `Result`）。
//! - 原 fn 被宏消费（不再生成裸 fn），仅生成 `XxxTool` 结构体，避免命名冲突。
//!
//! # 配置说明
//!
//! 工具的运行配置（如工作目录、超时）不在宏生成范围内，由工具函数内部
//! 使用默认值或常量处理（见 `cogent-tools` 各工具实现）。

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{FnArg, Ident, ItemFn, Lit, Meta, Pat, Token, Type, parse_macro_input};

/// 将 snake_case 标识符转换为 PascalCase。
///
/// 例如：`file_read` → `FileRead`，`bash` → `Bash`，`http_get` → `HttpGet`。
/// 用于从函数名生成结构体名前缀。
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// 从函数属性中提取 doc 注释，拼接为工具描述字符串。
///
/// Rust 的 `///` doc 注释在 AST 中表示为 `#[doc = "..."]` 属性（`Meta::NameValue`）。
/// 本函数遍历所有 `doc` 属性，从 `Meta::NameValue` 的 `value` 中提取字符串字面量，
/// 去除前导空格后以换行符拼接为完整的描述文本。若无 doc 注释则返回空字符串。
fn extract_description(attrs: &[syn::Attribute]) -> String {
    attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(|attr| match &attr.meta {
            Meta::NameValue(nv) => match &nv.value {
                syn::Expr::Lit(expr_lit) => match &expr_lit.lit {
                    Lit::Str(s) => Some(s.value().trim_start().to_string()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// 辅助类型：解析逗号分隔的 `Meta` 列表。
///
/// `Punctuated` 本身不实现 `Parse` trait，
/// 因此需要一个包装类型来桥接 `syn::parse2` 与 `ParseStream::parse_terminated`。
struct MetaList(Punctuated<Meta, Token![,]>);

impl syn::parse::Parse for MetaList {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(MetaList(input.parse_terminated(Meta::parse, Token![,])?))
    }
}

/// `#[cogent_tool]` 过程宏入口。
///
/// 标注在 `async fn` 上，消费原函数并生成一个实现 [`cogent_core::tool::Tool`]
/// 的**单元结构体**。结构体名称为函数名的 PascalCase 形式加 `Tool` 后缀
/// （如 `bash` → `BashTool`，`file_read` → `FileReadTool`）。
///
/// 生成的结构体实现以下方法：
/// - `name()`：返回工具名称（可由 `name = "..."` 属性覆盖，默认为函数名）。
/// - `description()`：返回函数的 doc 注释内容。
/// - `parameters_schema()`：返回参数的 JSON Schema（对象类型，所有字段 required）。
/// - `execute()`：从 `args` 反序列化各入参为局部变量后执行原函数体。
///
/// # 约束
///
/// - 函数必须是 `async fn`。
/// - 函数参数须为简单标识符（不支持解构模式、不支持 `&self`）。
/// - 函数返回类型须为 `anyhow::Result<String>`（或可经 `?` 转换的 `Result`）。
/// - 参数类型须实现 `schemars::JsonSchema`。
#[proc_macro_attribute]
pub fn cogent_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(item as ItemFn);

    // 解析属性参数（如 name = "..."），无参数时为空列表
    let attr_tokens: TokenStream2 = attr.into();
    let metas: Punctuated<Meta, Token![,]> = syn::parse2::<MetaList>(attr_tokens)
        .map(|m| m.0)
        .unwrap_or_default();

    // 提取 name 覆盖值
    let mut tool_name_override: Option<String> = None;
    for meta in metas.iter() {
        if let Meta::NameValue(nv) = meta
            && nv.path.is_ident("name")
            && let syn::Expr::Lit(expr_lit) = &nv.value
            && let Lit::Str(s) = &expr_lit.lit
        {
            tool_name_override = Some(s.value());
        }
    }

    // 函数名与最终工具名
    let fn_name = &item_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let tool_name_str = tool_name_override.unwrap_or_else(|| fn_name_str.clone());

    // 生成结构体名：PascalCase(函数名) + "Tool"
    let struct_name = Ident::new(
        &format!("{}Tool", to_pascal_case(&fn_name_str)),
        fn_name.span(),
    );

    // 提取 doc 注释作为工具描述
    let description = extract_description(&item_fn.attrs);

    // 提取参数列表（名称 + 类型），过滤掉 &self 与非简单标识符参数
    let params: Vec<(Ident, Type)> = item_fn
        .sig
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(pat_type) => {
                if let Pat::Ident(pat_ident) = *pat_type.pat.clone() {
                    // 跳过 &self（不支持，工具入参均从 args 反序列化）
                    if pat_ident.ident == "self" {
                        None
                    } else {
                        Some((pat_ident.ident, *pat_type.ty.clone()))
                    }
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();

    // 生成参数反序列化语句：从 JSON 对象中提取各字段并反序列化为对应类型
    let param_declarations: Vec<TokenStream2> = params
        .iter()
        .map(|(name, ty)| {
            let name_str = name.to_string();
            quote! {
                let #name: #ty = serde_json::from_value(
                    args.get(#name_str)
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                )
                .map_err(|e| anyhow::anyhow!(
                    "tool '{}': failed to deserialize parameter '{}': {}",
                    #tool_name_str, #name_str, e
                ))?;
            }
        })
        .collect();

    // 生成 parameters_schema 中 properties 的插入语句
    let schema_props: Vec<TokenStream2> = params
        .iter()
        .map(|(name, ty)| {
            let name_str = name.to_string();
            quote! {
                props.insert(
                    #name_str.to_string(),
                    serde_json::to_value(schemars::schema_for!(#ty)).unwrap(),
                );
            }
        })
        .collect();

    // 生成 required 数组的元素
    let required_fields: Vec<TokenStream2> = params
        .iter()
        .map(|(name, _)| {
            let name_str = name.to_string();
            quote! { #name_str.to_string() }
        })
        .collect();

    // 原函数体（直接嵌入 execute 方法）
    let block = &item_fn.block;

    // 生成完整的单元结构体定义与 Tool trait 实现
    let expanded = quote! {
        /// 由 `#[cogent_tool]` 宏自动生成的工具结构体。
        ///
        /// 对应原函数 `#fn_name_str`，实现 [`cogent_core::tool::Tool`] trait。
        /// 入参在 `execute` 调用时从 `args` 反序列化，故本结构体无字段。
        pub struct #struct_name;

        #[async_trait::async_trait]
        impl cogent_core::tool::Tool for #struct_name {
            fn name(&self) -> &str {
                #tool_name_str
            }

            fn description(&self) -> &str {
                #description
            }

            fn parameters_schema(&self) -> serde_json::Value {
                let mut props = serde_json::Map::new();
                #(#schema_props)*
                serde_json::json!({
                    "type": "object",
                    "properties": props,
                    "required": [#(#required_fields),*],
                })
            }

            async fn execute(
                &self,
                args: serde_json::Value,
            ) -> anyhow::Result<String> {
                #(#param_declarations)*
                #block
            }
        }
    };

    TokenStream::from(expanded)
}
