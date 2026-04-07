use crate::classify::OnnxClassifier;
use crate::config::TaskConfig;
use crate::gen::dataset::load;
use anyhow::Result;

pub fn run(cfg: &TaskConfig) -> Result<()> {
    let name = &cfg.task.name;
    let dataset_path = crate::config::dataset_path(name);

    let examples = load(&dataset_path)?;
    anyhow::ensure!(!examples.is_empty(), "dataset is empty");

    let mut classifier = OnnxClassifier::load(cfg)?;
    let classes = &cfg.task.classes;

    let n = classes.len();
    // confusion[actual][predicted]
    let mut confusion = vec![vec![0usize; n]; n];
    let mut correct = 0usize;

    for ex in &examples {
        let predicted = classifier.classify(&ex.input)?;
        let actual_idx = classes.iter().position(|c| c == &ex.label).unwrap_or(n);
        let pred_idx = classes.iter().position(|c| c == &predicted).unwrap_or(n);
        if actual_idx < n && pred_idx < n {
            confusion[actual_idx][pred_idx] += 1;
        }
        if predicted == ex.label {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / examples.len() as f64 * 100.0;
    println!("accuracy: {:.1}% ({}/{})", accuracy, correct, examples.len());
    println!();

    // per-class metrics
    println!("{:<20} {:>8} {:>8} {:>8}", "class", "prec", "recall", "f1");
    println!("{}", "-".repeat(50));
    for (i, class) in classes.iter().enumerate() {
        let tp = confusion[i][i];
        let fp: usize = (0..n).filter(|&r| r != i).map(|r| confusion[r][i]).sum();
        let fn_: usize = (0..n).filter(|&c| c != i).map(|c| confusion[i][c]).sum();
        let prec = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let rec = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
        println!("{:<20} {:>8.3} {:>8.3} {:>8.3}", class, prec, rec, f1);
    }

    println!();
    println!("confusion matrix (rows=actual, cols=predicted):");
    print!("{:<20}", "");
    for c in classes {
        print!("{:>10}", c);
    }
    println!();
    for (i, class) in classes.iter().enumerate() {
        print!("{:<20}", class);
        for j in 0..n {
            print!("{:>10}", confusion[i][j]);
        }
        println!();
    }

    Ok(())
}
