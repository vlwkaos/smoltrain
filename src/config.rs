use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskConfig {
    pub task: TaskMeta,
    pub base_model: BaseModel,
    pub gen: GenConfig,
    pub train: TrainConfig,
    pub export: ExportConfig,
    pub serve: ServeConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskMeta {
    pub name: String,
    pub goal: String,
    pub classes: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BaseModel {
    pub repo: String,
    pub path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GenConfig {
    pub n_per_class: usize,
    pub supervisor: SupervisorConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SupervisorConfig {
    pub provider: String,
    pub model: String,
    pub base_url: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub lora_rank: usize,
    pub lora_alpha: usize,
    pub batch_size: usize,
    pub max_seq_len: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExportConfig {
    pub format: String,
    pub quantization: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ServeConfig {
    pub socket: String,
}

/// ~/.local/share/smoltrain/<task>/
pub fn task_dir(name: &str) -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("~/.local/share"))
        .join("smoltrain")
        .join(name)
}

pub fn config_path(name: &str) -> PathBuf {
    task_dir(name).join("smoltrain.toml")
}

pub fn dataset_path(name: &str) -> PathBuf {
    task_dir(name).join("train.jsonl")
}

pub fn model_dir(name: &str) -> PathBuf {
    task_dir(name).join("model")
}

pub fn gguf_path(name: &str) -> PathBuf {
    task_dir(name).join(format!("{name}-classifier.gguf"))
}

pub fn load(name: &str) -> Result<TaskConfig> {
    let path = config_path(name);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("task '{}' not found at {}", name, path.display()))?;
    toml::from_str(&raw).context("parse smoltrain.toml")
}

pub fn new_task(name: &str, goal: &str, classes: &[String], base: &str) -> Result<()> {
    let dir = task_dir(name);
    std::fs::create_dir_all(&dir)?;

    let cfg = format!(
        r#"[task]
name = "{name}"
goal = "{goal}"
classes = [{classes}]

[base_model]
repo = "{base}"

[gen]
n_per_class = 150

[gen.supervisor]
provider = "anthropic"
model = "claude-sonnet-4-6"

[train]
epochs = 3
learning_rate = 2e-4
lora_rank = 8
lora_alpha = 16
batch_size = 4
max_seq_len = 256

[export]
format = "gguf"
quantization = "Q8_0"

[serve]
socket = "/tmp/smoltrain-{name}.sock"
"#,
        classes = classes
            .iter()
            .map(|c| format!("\"{}\"", c))
            .collect::<Vec<_>>()
            .join(", ")
    );

    std::fs::write(config_path(name), cfg)?;
    println!("Created task '{}' at {}", name, dir.display());
    println!("Next: smoltrain gen {} --n {} --supervisor claude", name, classes.len() * 150);
    Ok(())
}
