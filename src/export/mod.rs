use anyhow::{Context, Result};
use std::process::Command;
use crate::config;

pub fn run(task: &str, format: &str) -> Result<()> {
    let cfg = config::load(task)?;
    let model_dir = config::model_dir(task);
    let out = config::gguf_path(task);

    if !model_dir.exists() {
        anyhow::bail!("No trained model. Run: smoltrain train {task}");
    }

    match format {
        "gguf" => export_gguf(task, &cfg, &model_dir, &out),
        "mlx" => {
            println!("MLX format: model already at {}", model_dir.display());
            Ok(())
        }
        _ => anyhow::bail!("Unknown format: {format}. Use 'gguf' or 'mlx'"),
    }
}

fn export_gguf(
    task: &str,
    cfg: &crate::config::TaskConfig,
    model_dir: &std::path::Path,
    out: &std::path::Path,
) -> Result<()> {
    println!("Exporting to GGUF ({})...", cfg.export.quantization);

    // Try mlx-lm convert first (for MLX-trained models)
    let status = Command::new("python3")
        .arg("-m").arg("mlx_lm.convert")
        .arg("--hf-path").arg(model_dir)
        .arg("--mlx-path").arg(model_dir.join("mlx"))
        .arg("--quantize")
        .status();

    // Then use llama.cpp convert_hf_to_gguf.py + quantize
    let llama_convert = find_llama_convert()?;

    let status = Command::new("python3")
        .arg(&llama_convert)
        .arg(model_dir)
        .arg("--outtype").arg("f16")
        .arg("--outfile").arg(out.with_extension("f16.gguf"))
        .status()
        .context("llama.cpp convert_hf_to_gguf.py not found — install llama.cpp")?;

    if !status.success() {
        anyhow::bail!("GGUF conversion failed");
    }

    // Quantize
    let quant = &cfg.export.quantization;
    let q_out = out;
    let status = Command::new("llama-quantize")
        .arg(out.with_extension("f16.gguf"))
        .arg(q_out)
        .arg(quant)
        .status();

    match status {
        Ok(s) if s.success() => {
            let _ = std::fs::remove_file(out.with_extension("f16.gguf"));
            println!("Exported: {}", q_out.display());
            println!("Next: smoltrain serve {task}");
        }
        _ => {
            // llama-quantize not found — keep f16
            std::fs::rename(out.with_extension("f16.gguf"), q_out)?;
            println!("Exported (f16, quantize manually): {}", q_out.display());
        }
    }

    Ok(())
}

fn find_llama_convert() -> Result<std::path::PathBuf> {
    // Common install locations for llama.cpp convert script
    let candidates = [
        "/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py",
        "/usr/local/share/llama.cpp/convert_hf_to_gguf.py",
        "convert_hf_to_gguf.py",
    ];
    for p in candidates {
        if std::path::Path::new(p).exists() {
            return Ok(p.into());
        }
    }
    anyhow::bail!("convert_hf_to_gguf.py not found. Install: brew install llama.cpp")
}
