use anyhow::Result;
use crate::config;
use crate::gen::Example;
use crate::serve::Classifier;

pub fn run(task: &str) -> Result<()> {
    let cfg = config::load(task)?;
    let gguf = config::gguf_path(task);
    let dataset = config::dataset_path(task);

    if !gguf.exists() {
        anyhow::bail!("No exported model. Run: smoltrain export {task}");
    }

    let examples = crate::gen::dataset::load(&dataset)?;
    if examples.is_empty() {
        anyhow::bail!("No dataset. Run: smoltrain gen {task}");
    }

    let classifier = Classifier::load(&gguf, &cfg.task.classes)?;
    let classes = &cfg.task.classes;

    // Confusion matrix: rows=actual, cols=predicted
    let n = classes.len();
    let idx = |c: &str| classes.iter().position(|x| x == c).unwrap_or(0);

    let mut matrix = vec![vec![0usize; n]; n];
    let mut correct = 0;

    for ex in &examples {
        let pred = classifier.classify(&ex.input)?;
        let actual_i = idx(&ex.label);
        let pred_i = idx(&pred);
        matrix[actual_i][pred_i] += 1;
        if pred == ex.label { correct += 1; }
    }

    let acc = correct as f64 / examples.len() as f64 * 100.0;
    println!("Accuracy: {correct}/{} = {acc:.1}%", examples.len());
    println!();
    println!("Confusion matrix (rows=actual, cols=predicted):");

    let col_w = classes.iter().map(|c| c.len()).max().unwrap_or(8).max(6);
    print!("{:col_w$}  ", "");
    for c in classes { print!("{:col_w$}  ", c); }
    println!();

    for (i, row) in matrix.iter().enumerate() {
        print!("{:col_w$}  ", classes[i]);
        for val in row { print!("{:col_w$}  ", val); }
        println!();
    }

    println!();
    for (i, class) in classes.iter().enumerate() {
        let tp = matrix[i][i];
        let fp: usize = (0..n).map(|r| if r != i { matrix[r][i] } else { 0 }).sum();
        let fn_: usize = (0..n).map(|c| if c != i { matrix[i][c] } else { 0 }).sum();
        let prec = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let rec = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
        println!("  {class:col_w$}  prec={prec:.2}  rec={rec:.2}  f1={f1:.2}");
    }

    Ok(())
}

// Re-export for use in eval/mod.rs
pub(crate) mod dataset {
    pub use crate::gen::dataset::*;
}
