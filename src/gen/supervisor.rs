use crate::config::TaskConfig;
use crate::gen::dataset::Example;
use anyhow::{bail, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

pub struct SupervisorClient {
    client: Client,
    url: String,
    model: String,
    api_key: String,
    is_anthropic: bool,
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
    pub fn new(cfg: &TaskConfig) -> Result<Self> {
        let api_key = std::env::var(&cfg.gen.api_key_env)
            .unwrap_or_default();
        let is_anthropic = cfg.gen.supervisor_url.contains("anthropic.com");
        Ok(Self {
            client: Client::new(),
            url: cfg.gen.supervisor_url.clone(),
            model: cfg.gen.supervisor_model.clone(),
            api_key,
            is_anthropic,
        })
    }

    pub async fn generate(
        &self,
        class: &str,
        count: usize,
        goal: &str,
        all_classes: &[String],
    ) -> Result<Vec<Example>> {
        let classes_str = all_classes.join(", ");
        let prompt = format!(
            "Generate {count} diverse, realistic example texts for the classification label \"{class}\".\n\
            Task: {goal}\n\
            All possible labels: {classes_str}\n\n\
            Rules:\n\
            - Each example on its own line, plain text only\n\
            - No numbering, no quotes, no prefixes\n\
            - Varied phrasing and length\n\
            - Clearly belongs to \"{class}\" not other labels\n\n\
            Output {count} lines:"
        );

        let text = if self.is_anthropic {
            self.call_anthropic(&prompt).await?
        } else {
            self.call_openai(&prompt).await?
        };

        let examples: Vec<Example> = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
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

        let resp = self.client
            .post(&self.url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await?;
            bail!("Anthropic API error {status}: {body}");
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

        let resp = self.client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await?;
            bail!("OpenAI API error {status}: {body}");
        }

        let parsed: OpenAIResponse = resp.json().await?;
        Ok(parsed.choices.into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default())
    }
}
