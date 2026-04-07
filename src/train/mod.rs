use crate::config::TaskConfig;
use anyhow::{bail, Context, Result};
use std::process::Command;
use tracing::info;

pub fn run(cfg: &TaskConfig) -> Result<()> {
    let name = &cfg.task.name;
    let dataset_path = crate::config::dataset_path(name);
    let model_dir = crate::config::model_dir(name);

    anyhow::ensure!(
        dataset_path.exists(),
        "dataset not found at {}. Run `smoltrain gen {}` first.",
        dataset_path.display(),
        name
    );

    std::fs::create_dir_all(&model_dir)?;

    let classes = cfg.task.classes.join(",");
    let script = find_script("scripts/train.py")?;

    info!("launching training script: {}", script.display());

    let status = Command::new("python3")
        .arg(&script)
        .arg("--model").arg(&cfg.base_model.repo)
        .arg("--dataset").arg(&dataset_path)
        .arg("--output").arg(&model_dir)
        .arg("--epochs").arg(cfg.train.epochs.to_string())
        .arg("--classes").arg(&classes)
        .arg("--goal").arg(&cfg.task.goal)
        .arg("--lr").arg(cfg.train.lr.to_string())
        .arg("--batch-size").arg(cfg.train.batch_size.to_string())
        .arg("--max-seq-len").arg(cfg.train.max_seq_len.to_string())
        .status()
        .context("failed to launch python3")?;

    if !status.success() {
        bail!("training script failed with exit code {:?}", status.code());
    }

    println!("training complete -> {}", model_dir.display());
    Ok(())
}

fn find_script(relative: &str) -> Result<std::path::PathBuf> {
    let p = std::path::Path::new(relative);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(relative);
            if candidate.exists() {
                return Ok(candidate);
            }
            let candidate = dir.join("..").join("..").join(relative);
            if candidate.exists() {
                return Ok(candidate.canonicalize()?);
            }
        }
    }
    bail!("cannot find {relative}");
}
