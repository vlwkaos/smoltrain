pub mod dataset;
pub mod supervisor;

use crate::config::TaskConfig;
use anyhow::Result;
use dataset::{load, save};
use supervisor::SupervisorClient;
use tracing::info;

pub async fn run(cfg: &TaskConfig, count_per_class: usize) -> Result<()> {
    let dataset_path = crate::config::dataset_path(&cfg.task.name);
    let mut examples = load(&dataset_path).unwrap_or_default();

    let client = SupervisorClient::new(cfg)?;

    for class in &cfg.task.classes {
        let existing = examples.iter().filter(|e| &e.label == class).count();
        let needed = count_per_class.saturating_sub(existing);
        if needed == 0 {
            info!("class '{}' already has {} examples, skipping", class, existing);
            continue;
        }
        info!("generating {} examples for class '{}'", needed, class);
        let new_examples = client.generate(class, needed, &cfg.task.goal, &cfg.task.classes).await?;
        examples.extend(new_examples);
        save(&dataset_path, &examples)?;
        info!("saved {} total examples", examples.len());
    }

    println!("dataset: {} examples across {} classes", examples.len(), cfg.task.classes.len());
    Ok(())
}
