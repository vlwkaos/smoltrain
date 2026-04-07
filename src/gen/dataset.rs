use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub input: String,
    pub label: String,
}

pub fn load(path: &Path) -> Result<Vec<Example>> {
    if !path.exists() {
        return Ok(vec![]);
    }
    let text = std::fs::read_to_string(path)?;
    let mut examples = vec![];
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        examples.push(serde_json::from_str(line)?);
    }
    Ok(examples)
}

pub fn save(path: &Path, examples: &[Example]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = String::new();
    for ex in examples {
        out.push_str(&serde_json::to_string(ex)?);
        out.push('\n');
    }
    std::fs::write(path, out)?;
    Ok(())
}
