use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Example {
    pub input: String,
    pub label: String,
}

pub fn load(path: &Path) -> Result<Vec<Example>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(path)?;
    let examples = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Example>(l))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(examples)
}

pub fn save(path: &Path, examples: &[Example]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = examples
        .iter()
        .map(|e| serde_json::to_string(e))
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    std::fs::write(path, content + "\n")?;
    Ok(())
}
