use crate::gen::dataset::Example;
use anyhow::{bail, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Provider enum ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Provider {
    /// Anthropic native API — x-api-key header
    /// env: ANTHROPIC_API_KEY
    Anthropic { api_key: String },

    /// Claude Code OAuth token — reuse existing claude CLI auth.
    /// Reads from macOS Keychain: "Claude Code-credentials".
    /// Uses Authorization: Bearer <access_token>.
    /// Falls back to ANTHROPIC_API_KEY if keychain unavailable.
    ClaudeOAuth { access_token: String },

    /// Groq free tier — OpenAI-compat at api.groq.com
    /// env: GROQ_API_KEY
    Groq { api_key: String },

    /// Any OpenAI-compatible endpoint (openai, openrouter, together, etc.)
    /// env: OPENAI_API_KEY (or SMOLTRAIN_API_KEY for custom)
    OpenAICompat { base_url: String, api_key: Option<String> },

    /// Local server — no auth (ollama, omlx, lm-studio, etc.)
    Local { base_url: String },
}

impl Provider {
    /// Resolve provider from config string + environment.
    /// Config format: "anthropic" | "claude-oauth" | "groq" | "openai" | "ollama" | "omlx" | "local:<url>"
    pub fn resolve(name: &str) -> Result<Self> {
        match name {
            "anthropic" => {
                let key = std::env::var("ANTHROPIC_API_KEY")
                    .context("ANTHROPIC_API_KEY not set (provider=anthropic)")?;
                Ok(Provider::Anthropic { api_key: key })
            }

            "claude-oauth" | "claude" => {
                match read_claude_oauth_token() {
                    Ok(token) => Ok(Provider::ClaudeOAuth { access_token: token }),
                    Err(e) => {
                        eprintln!("claude-oauth keychain read failed ({e}), checking ANTHROPIC_TOKEN...");
                        // Hermes stores its OAuth token as ANTHROPIC_TOKEN
                        if let Ok(token) = std::env::var("ANTHROPIC_TOKEN") {
                            if !token.is_empty() {
                                return Ok(Provider::ClaudeOAuth { access_token: token });
                            }
                        }
                        eprintln!("ANTHROPIC_TOKEN not set, falling back to ANTHROPIC_API_KEY");
                        let key = std::env::var("ANTHROPIC_API_KEY")
                            .context("No auth available: set ANTHROPIC_TOKEN or ANTHROPIC_API_KEY")?;
                        Ok(Provider::Anthropic { api_key: key })
                    }
                }
            }

            "groq" => {
                let key = std::env::var("GROQ_API_KEY")
                    .context("GROQ_API_KEY not set (provider=groq)")?;
                Ok(Provider::Groq { api_key: key })
            }

            "openai" => {
                let key = std::env::var("OPENAI_API_KEY").ok();
                Ok(Provider::OpenAICompat {
                    base_url: "https://api.openai.com/v1".into(),
                    api_key: key,
                })
            }

            "ollama" => Ok(Provider::Local {
                base_url: std::env::var("OLLAMA_HOST")
                    .unwrap_or_else(|_| "http://localhost:11434/v1".into()),
            }),

            "omlx" => Ok(Provider::Local {
                base_url: "http://localhost:8000/v1".into(),
            }),

            other if other.starts_with("local:") => {
                let url = other.trim_start_matches("local:").to_string();
                Ok(Provider::Local { base_url: url })
            }

            other => {
                // Treat as a base URL with optional SMOLTRAIN_API_KEY
                if other.starts_with("http") {
                    let key = std::env::var("SMOLTRAIN_API_KEY").ok()
                        .or_else(|| std::env::var("OPENAI_API_KEY").ok());
                    Ok(Provider::OpenAICompat {
                        base_url: other.to_string(),
                        api_key: key,
                    })
                } else {
                    bail!("Unknown provider: '{other}'\nValid: anthropic, claude-oauth, groq, openai, ollama, omlx, local:<url>")
                }
            }
        }
    }

    fn is_anthropic_format(&self) -> bool {
        matches!(self, Provider::Anthropic { .. } | Provider::ClaudeOAuth { .. })
    }
}

// ── OAuth token reader (macOS Keychain) ────────────────────────────────────────

#[derive(Deserialize)]
struct ClaudeCredentials {
    #[serde(rename = "claudeAiOauth")]
    oauth: ClaudeOAuth,
}

#[derive(Deserialize)]
struct ClaudeOAuth {
    #[serde(rename = "accessToken")]
    access_token: String,
    #[serde(rename = "expiresAt")]
    expires_at: u64, // epoch milliseconds
}

fn read_claude_oauth_token() -> Result<String> {
    // macOS only — uses `security` CLI to read from Keychain
    #[cfg(not(target_os = "macos"))]
    bail!("claude-oauth keychain only supported on macOS");

    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("security")
            .args(["find-generic-password", "-s", "Claude Code-credentials", "-w"])
            .output()
            .context("security CLI not available")?;

        if !output.status.success() {
            bail!("keychain entry 'Claude Code-credentials' not found — run `claude` once to authenticate");
        }

        let raw = String::from_utf8(output.stdout)
            .context("keychain output not UTF-8")?;
        let creds: ClaudeCredentials = serde_json::from_str(raw.trim())
            .context("parse Claude Code credentials JSON")?;

        // Check expiry
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        if creds.oauth.expires_at < now_ms {
            bail!("Claude OAuth token expired — run `claude` to refresh");
        }

        Ok(creds.oauth.access_token)
    }
}

// ── SupervisorClient ───────────────────────────────────────────────────────────

pub struct SupervisorClient {
    client: Client,
    provider: Provider,
    model: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}
#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}
#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}
#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}
#[derive(Deserialize)]
struct OpenAIMessage {
    content: String,
}

impl SupervisorClient {
    pub fn new(provider_name: &str, model: &str) -> Result<Self> {
        let provider = Provider::resolve(provider_name)?;
        Ok(Self {
            client: Client::new(),
            provider,
            model: model.to_string(),
        })
    }

    pub async fn generate(
        &self,
        class: &str,
        count: usize,
        goal: &str,
        all_classes: &[String],
        class_descriptions: Option<&std::collections::HashMap<String, String>>,
    ) -> Result<Vec<Example>> {
        let classes_str = all_classes.join(", ");
        let prompt = if let Some(descs) = class_descriptions {
            let class_description = descs.get(class).map(|s| s.as_str()).unwrap_or("");
            let contrast_classes: String = all_classes
                .iter()
                .filter(|c| c.as_str() != class)
                .filter_map(|c| descs.get(c).map(|d| format!("\"{}\" ({})", c, d)))
                .collect::<Vec<_>>()
                .join(" | ");
            format!(
                "Generate {count} diverse, realistic example texts for the classification label \"{class}\".\n\
                Task: {goal}\n\
                Class definition: {class_description}\n\
                All labels: {classes_str}\n\n\
                Contrastive rules:\n\
                - Each example must CLEARLY belong to \"{class}\" and NOT to any other label\n\
                - Especially avoid ambiguity with: {contrast_classes}\n\
                - Focus on the BOUNDARY cases — examples that could be confused but are definitively \"{class}\"\n\
                - Varied phrasing and length (short and long examples)\n\
                - Each example on its own line, plain text only, no numbering, no quotes\n\n\
                Output exactly {count} lines:"
            )
        } else {
            format!(
                "Generate {count} diverse, realistic example texts for the classification label \"{class}\".\n\
                Task: {goal}\n\
                All possible labels: {classes_str}\n\n\
                Rules:\n\
                - Each example on its own line, plain text only\n\
                - No numbering, no quotes, no prefixes\n\
                - Varied phrasing and length (short and long examples)\n\
                - Each example clearly belongs to \"{class}\" not other labels\n\n\
                Output exactly {count} lines:"
            )
        };

        let text = if self.provider.is_anthropic_format() {
            self.call_anthropic(&prompt).await?
        } else {
            self.call_openai(&prompt).await?
        };

        let examples = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .take(count)
            .map(|l| Example {
                input: l.to_string(),
                label: class.to_string(),
            })
            .collect();

        Ok(examples)
    }

    async fn call_anthropic(&self, prompt: &str) -> Result<String> {
        let body = json!({
            "model": self.model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}]
        });

        let mut req = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        req = match &self.provider {
            Provider::Anthropic { api_key } =>
                req.header("x-api-key", api_key),
            Provider::ClaudeOAuth { access_token } =>
                req.header("Authorization", format!("Bearer {access_token}")),
            _ => unreachable!(),
        };

        let resp = req.json(&body).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await?;
            bail!("Anthropic API {status}: {body}");
        }
        let parsed: AnthropicResponse = resp.json().await?;
        Ok(parsed.content.into_iter().map(|c| c.text).collect::<Vec<_>>().join(""))
    }

    async fn call_openai(&self, prompt: &str) -> Result<String> {
        let body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
        });

        let url = match &self.provider {
            Provider::Groq { .. } => "https://api.groq.com/openai/v1/chat/completions",
            Provider::OpenAICompat { base_url, .. } => base_url,
            Provider::Local { base_url } => base_url,
            _ => unreachable!(),
        };

        // Append /chat/completions if url looks like a base url (no path)
        let url = if url.ends_with("/v1") || url.ends_with("/v1/") {
            format!("{}/chat/completions", url.trim_end_matches('/'))
        } else {
            url.to_string()
        };

        let mut req = self.client.post(&url);

        req = match &self.provider {
            Provider::Groq { api_key } =>
                req.header("Authorization", format!("Bearer {api_key}")),
            Provider::OpenAICompat { api_key: Some(k), .. } =>
                req.header("Authorization", format!("Bearer {k}")),
            Provider::Local { .. } | Provider::OpenAICompat { api_key: None, .. } => req,
            _ => unreachable!(),
        };

        let resp = req.json(&body).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await?;
            bail!("API {status}: {body}");
        }
        let parsed: OpenAIResponse = resp.json().await?;
        Ok(parsed.choices.into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default())
    }
}
