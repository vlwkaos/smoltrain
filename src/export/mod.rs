use crate::config::TaskConfig;
use anyhow::{bail, Context, Result};
use std::process::Command;
use tracing::info;

pub fn run(cfg: &TaskConfig) -> Result<()> {
    let name = &cfg.task.name;
    let model_dir = crate::config::model_dir(name);
    let onnx_dir = crate::config::onnx_dir(name);

    anyhow::ensure!(
        model_dir.exists(),
        "model not found at {}. Run `smoltrain train {}` first.",
        model_dir.display(),
        name
    );

    std::fs::create_dir_all(&onnx_dir)?;

    let script = find_script("scripts/export.py")?;
    info!("launching export script: {}", script.display());

    let status = Command::new("python3")
        .arg(&script)
        .arg("--model-dir").arg(&model_dir)
        .arg("--output").arg(&onnx_dir)
        .arg("--seq-len").arg(cfg.train.max_seq_len.to_string())
        .status()
        .context("failed to launch python3")?;

    if !status.success() {
        bail!("export script failed with exit code {:?}", status.code());
    }

    let onnx_path = crate::config::onnx_path(name);
    println!("exported to {}", onnx_path.display());
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
