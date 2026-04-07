mod supervisor;
mod dataset;

pub use dataset::Example;

use anyhow::Result;
use crate::config;

pub async fn run(task: &str, n: usize, supervisor: &str) -> Result<()> {
    let cfg = config::load(task)?;
    let classes = &cfg.task.classes;
    let n_per_class = n / classes.len();

    println!("Generating {} examples per class ({} total) for task '{}'",
        n_per_class, n_per_class * classes.len(), task);

    let sup_cfg = &cfg.gen.supervisor;
    let provider = if supervisor == "claude" { "anthropic" } else { supervisor };

    let api_key = match provider {
        "anthropic" => std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .or_else(|| std::env::var("CLAUDE_API_KEY").ok()),
        _ => std::env::var("OPENAI_API_KEY").ok(),
    };

    let client = supervisor::SupervisorClient::new(
        provider,
        &sup_cfg.model,
        sup_cfg.base_url.as_deref(),
        api_key.as_deref(),
    )?;

    let dataset_path = config::dataset_path(task);
    let mut existing = dataset::load(&dataset_path).unwrap_or_default();
    let existing_count = existing.len();

    for class in classes {
        let class_count = existing.iter().filter(|e| &e.label == class).count();
        let need = n_per_class.saturating_sub(class_count);
        if need == 0 {
            println!("  {class}: already have {class_count}, skipping");
            continue;
        }
        println!("  Generating {need} examples for class '{class}'...");

        let examples = client.generate_examples(
            &cfg.task.goal,
            classes,
            class,
            need,
        ).await?;

        println!("    Got {} examples", examples.len());
        existing.extend(examples);
    }

    dataset::save(&dataset_path, &existing)?;
    println!("Dataset saved: {} total examples at {}", existing.len(), dataset_path.display());
    Ok(())
}
