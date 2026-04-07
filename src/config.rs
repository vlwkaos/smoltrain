use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub task: TaskMeta,
    pub base_model: BaseModel,
    pub gen: GenConfig,
    pub train: TrainConfig,
    pub serve: ServeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMeta {
    pub name: String,
    pub goal: String,
    pub classes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModel {
    pub repo: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenConfig {
    /// Provider name: "anthropic" | "claude-oauth" | "groq" | "openai" | "ollama" | "omlx" | "local:<url>"
    pub supervisor: String,
    pub supervisor_model: String,
    pub n_per_class: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub max_seq_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeConfig {
    pub socket: String,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            task: TaskMeta {
                name: String::new(),
                goal: String::new(),
                classes: vec![],
            },
            base_model: BaseModel {
                repo: "distilbert-base-uncased".to_string(),
            },
            gen: GenConfig {
                supervisor: "claude-oauth".to_string(),
                supervisor_model: "claude-haiku-4-5".to_string(),
                n_per_class: 150,
            },
            train: TrainConfig {
                epochs: 3,
                lr: 2e-5,
                batch_size: 16,
                max_seq_len: 128,
            },
            serve: ServeConfig {
                socket: String::new(),
            },
        }
    }
}

pub fn data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("~/.local/share"))
        .join("smoltrain")
}

pub fn task_dir(name: &str) -> PathBuf {
    data_dir().join(name)
}

pub fn config_path(name: &str) -> PathBuf {
    task_dir(name).join("smoltrain.toml")
}

pub fn dataset_path(name: &str) -> PathBuf {
    task_dir(name).join("dataset.jsonl")
}

pub fn model_dir(name: &str) -> PathBuf {
    task_dir(name).join("model")
}

pub fn onnx_dir(name: &str) -> PathBuf {
    task_dir(name).join("onnx")
}

pub fn onnx_path(name: &str) -> PathBuf {
    onnx_dir(name).join("model_int8.onnx")
}

pub fn tokenizer_path(name: &str) -> PathBuf {
    onnx_dir(name).join("tokenizer.json")
}

pub fn load(name: &str) -> Result<TaskConfig> {
    let path = config_path(name);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("Cannot read config at {}", path.display()))?;
    let cfg: TaskConfig = toml::from_str(&text)
        .with_context(|| format!("Invalid TOML in {}", path.display()))?;
    Ok(cfg)
}

pub fn new_task(name: &str, classes: Vec<String>, goal: String) -> Result<()> {
    let dir = task_dir(name);
    std::fs::create_dir_all(&dir)?;

    let mut cfg = TaskConfig::default();
    cfg.task.name = name.to_string();
    cfg.task.classes = classes;
    cfg.task.goal = goal;
    cfg.serve.socket = dir.join("serve.sock").to_string_lossy().to_string();

    let text = toml::to_string_pretty(&cfg)?;
    std::fs::write(config_path(name), text)?;
    Ok(())
}
