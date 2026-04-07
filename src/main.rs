use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "smoltrain", about = "Train tiny ONNX text classifiers")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a new classification task
    New {
        /// Task name
        name: String,
        /// Comma-separated class labels
        #[arg(long)]
        classes: String,
        /// What the classifier should do (goal description)
        #[arg(long)]
        goal: String,
    },
    /// Generate training examples via supervisor LLM
    Gen {
        /// Task name
        name: String,
        /// Examples per class to generate
        #[arg(long, default_value = "50")]
        count: usize,
    },
    /// Fine-tune DistilBERT on the dataset
    Train {
        /// Task name
        name: String,
    },
    /// Evaluate the exported ONNX model on the dataset
    Eval {
        /// Task name
        name: String,
    },
    /// Export fine-tuned model to ONNX INT8
    Export {
        /// Task name
        name: String,
    },
    /// Run the classifier daemon on a Unix socket
    Serve {
        /// Task name
        name: String,
    },
    /// Classify a single input and print the class
    Classify {
        /// Task name
        name: String,
        /// Text to classify
        input: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::New { name, classes, goal } => {
            let class_list: Vec<String> = classes.split(',').map(|s| s.trim().to_string()).collect();
            smoltrain::config::new_task(&name, class_list, goal)?;
            println!("Created task '{name}'");
        }
        Command::Gen { name, count } => {
            let cfg = smoltrain::config::load(&name)?;
            smoltrain::gen::run(&cfg, count).await?;
        }
        Command::Train { name } => {
            let cfg = smoltrain::config::load(&name)?;
            smoltrain::train::run(&cfg)?;
        }
        Command::Eval { name } => {
            let cfg = smoltrain::config::load(&name)?;
            smoltrain::eval::run(&cfg)?;
        }
        Command::Export { name } => {
            let cfg = smoltrain::config::load(&name)?;
            smoltrain::export::run(&cfg)?;
        }
        Command::Serve { name } => {
            let cfg = smoltrain::config::load(&name)?;
            smoltrain::serve::run(&cfg)?;
        }
        Command::Classify { name, input } => {
            let cfg = smoltrain::config::load(&name)?;
            let mut classifier = smoltrain::classify::OnnxClassifier::load(&cfg)?;
            let label = classifier.classify(&input)?;
            println!("{label}");
        }
    }

    Ok(())
}
