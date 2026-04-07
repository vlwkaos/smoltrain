use anyhow::{Context, Result};
use std::process::Command;
use crate::config;

pub fn run(task: &str, epochs: usize) -> Result<()> {
    let cfg = config::load(task)?;
    let dataset = config::dataset_path(task);
    let model_dir = config::model_dir(task);

    if !dataset.exists() {
        anyhow::bail!("No dataset found. Run: smoltrain gen {task}");
    }

    std::fs::create_dir_all(&model_dir)?;

    let script = find_train_script()?;

    println!("Fine-tuning '{}' for {} epochs...", cfg.base_model.repo, epochs);
    println!("Dataset: {}", dataset.display());
    println!("Output: {}", model_dir.display());

    let status = Command::new("python3")
        .arg(&script)
        .arg("--model").arg(&cfg.base_model.repo)
        .arg("--dataset").arg(&dataset)
        .arg("--output").arg(&model_dir)
        .arg("--epochs").arg(epochs.to_string())
        .arg("--lr").arg(cfg.train.learning_rate.to_string())
        .arg("--lora-rank").arg(cfg.train.lora_rank.to_string())
        .arg("--lora-alpha").arg(cfg.train.lora_alpha.to_string())
        .arg("--batch-size").arg(cfg.train.batch_size.to_string())
        .arg("--max-seq-len").arg(cfg.train.max_seq_len.to_string())
        .arg("--classes").arg(cfg.task.classes.join(","))
        .arg("--goal").arg(&cfg.task.goal)
        .status()
        .context("failed to run train.py — is Python 3 with mlx-lm installed?")?;

    if !status.success() {
        anyhow::bail!("Training failed with exit code: {:?}", status.code());
    }

    println!("Training complete. Model at {}", model_dir.display());
    println!("Next: smoltrain eval {task}  OR  smoltrain export {task}");
    Ok(())
}

fn find_train_script() -> Result<std::path::PathBuf> {
    // Look for scripts/train.py relative to the binary
    let exe = std::env::current_exe()?;
    let candidates = [
        exe.parent().and_then(|p| p.parent()).map(|p| p.join("scripts/train.py")),
        Some(std::path::PathBuf::from("scripts/train.py")),
    ];
    for c in candidates.into_iter().flatten() {
        if c.exists() { return Ok(c); }
    }
    anyhow::bail!("scripts/train.py not found")
}
