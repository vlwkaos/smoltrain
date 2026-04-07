use clap::{Parser, Subcommand};

mod config;
mod gen;
mod train;
mod eval;
mod export;
mod serve;

#[derive(Parser)]
#[command(name = "smoltrain", version, about = "Train tiny task-specific classifier models")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new smoltrain task
    New {
        name: String,
        #[arg(long)] goal: String,
        #[arg(long, value_delimiter = ',')] classes: Vec<String>,
        #[arg(long, default_value = "Qwen/Qwen3-0.6B")] base: String,
    },
    /// Generate training data via supervisor model
    Gen {
        task: String,
        #[arg(long, default_value_t = 500)] n: usize,
        #[arg(long, default_value = "claude")] supervisor: String,
    },
    /// Fine-tune the base model
    Train {
        task: String,
        #[arg(long, default_value_t = 3)] epochs: usize,
    },
    /// Evaluate accuracy + confusion matrix
    Eval { task: String },
    /// Export model to GGUF or MLX
    Export {
        task: String,
        #[arg(long, default_value = "gguf")] format: String,
    },
    /// Run hot classifier daemon
    Serve {
        task: String,
        #[arg(long)] port: Option<u16>,
        #[arg(long)] socket: Option<String>,
    },
    /// Single classify call (for testing)
    Classify {
        task: String,
        input: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::New { name, goal, classes, base } => {
            config::new_task(&name, &goal, &classes, &base)?;
        }
        Commands::Gen { task, n, supervisor } => {
            gen::run(&task, n, &supervisor).await?;
        }
        Commands::Train { task, epochs } => {
            train::run(&task, epochs)?;
        }
        Commands::Eval { task } => {
            eval::run(&task)?;
        }
        Commands::Export { task, format } => {
            export::run(&task, &format)?;
        }
        Commands::Serve { task, port, socket } => {
            serve::run(&task, port, socket).await?;
        }
        Commands::Classify { task, input } => {
            let label = serve::classify_once(&task, &input)?;
            println!("{label}");
        }
    }

    Ok(())
}
