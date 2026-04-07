use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use super::dataset::Example;

pub struct SupervisorClient {
    provider: String,
    model: String,
    base_url: String,
    api_key: Option<String>,
    http: reqwest::Client,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

// Anthropic-native response shape
#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

impl SupervisorClient {
    pub fn new(
        provider: &str,
        model: &str,
        base_url: Option<&str>,
        api_key: Option<&str>,
    ) -> Result<Self> {
        let base_url = match base_url {
            Some(u) => u.to_string(),
            None => match provider {
                "anthropic" => "https://api.anthropic.com".to_string(),
                _ => "https://api.openai.com".to_string(),
            },
        };
        Ok(Self {
            provider: provider.to_string(),
            model: model.to_string(),
            base_url,
            api_key: api_key.map(|s| s.to_string()),
            http: reqwest::Client::new(),
        })
    }

    pub async fn generate_examples(
        &self,
        goal: &str,
        all_classes: &[String],
        target_class: &str,
        n: usize,
    ) -> Result<Vec<Example>> {
        let classes_list = all_classes.join(", ");
        let prompt = format!(
            r#"You are generating training data for a text classifier.

Task: {goal}
Classes: {classes_list}
Target class: {target_class}

Generate exactly {n} diverse, realistic examples that should be classified as "{target_class}".
Examples should vary in style, length, and phrasing. Be realistic — these are real user messages.

Output format: one example per line, no numbering, no quotes, just the raw text.
Output only the examples, nothing else."#
        );

        let text = self.complete(&prompt).await?;

        let examples: Vec<Example> = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| Example {
                input: l.to_string(),
                label: target_class.to_string(),
            })
            .take(n)
            .collect();

        Ok(examples)
    }

    async fn complete(&self, prompt: &str) -> Result<String> {
        match self.provider.as_str() {
            "anthropic" => self.complete_anthropic(prompt).await,
            _ => self.complete_openai(prompt).await,
        }
    }

    async fn complete_anthropic(&self, prompt: &str) -> Result<String> {
        let key = self.api_key.as_deref()
            .context("ANTHROPIC_API_KEY not set")?;

        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        });

        let resp = self.http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("anthropic request")?;

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("Anthropic API error {status}: {text}");
        }

        let parsed: AnthropicResponse = serde_json::from_str(&text)
            .context("parse anthropic response")?;
        Ok(parsed.content.into_iter().map(|c| c.text).collect::<Vec<_>>().join(""))
    }

    async fn complete_openai(&self, prompt: &str) -> Result<String> {
        let req = ChatRequest {
            model: self.model.clone(),
            messages: vec![Message { role: "user".into(), content: prompt.into() }],
            max_tokens: 4096,
        };

        let mut builder = self.http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("content-type", "application/json");

        if let Some(key) = &self.api_key {
            builder = builder.header("authorization", format!("Bearer {key}"));
        }

        let resp = builder.json(&req).send().await.context("openai request")?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("API error {status}: {text}");
        }

        let parsed: ChatResponse = serde_json::from_str(&text)
            .context("parse chat response")?;
        Ok(parsed.choices.into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default())
    }
}
