mod classifier;
pub use classifier::Classifier;

use anyhow::Result;
use std::io::{BufRead, BufReader, Write};
use crate::config;

/// Run as a hot daemon — load model once, serve over unix socket or TCP.
pub async fn run(task: &str, port: Option<u16>, socket: Option<String>) -> Result<()> {
    let cfg = config::load(task)?;
    let gguf = config::gguf_path(task);

    if !gguf.exists() {
        anyhow::bail!("Model not found: {}. Run: smoltrain export {task}", gguf.display());
    }

    let classifier = Classifier::load(&gguf, &cfg.task.classes)?;
    println!("Model loaded: {} classes: [{}]",
        task, cfg.task.classes.join(", "));

    let socket_path = socket
        .unwrap_or_else(|| cfg.serve.socket.clone());

    if port.is_some() {
        run_tcp(classifier, port.unwrap()).await
    } else {
        run_unix(classifier, &socket_path)
    }
}

fn run_unix(classifier: Classifier, path: &str) -> Result<()> {
    use std::os::unix::net::UnixListener;

    let _ = std::fs::remove_file(path);
    let listener = UnixListener::bind(path)?;
    println!("Listening on unix:{path}");

    for stream in listener.incoming() {
        let stream = stream?;
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut writer = stream;
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            let input = line.trim();
            if input.is_empty() { line.clear(); continue; }
            let label = classifier.classify(input).unwrap_or_else(|_| "unknown".to_string());
            writeln!(writer, "{label}")?;
            line.clear();
        }
    }
    Ok(())
}

async fn run_tcp(classifier: Classifier, port: u16) -> Result<()> {
    use tokio::net::TcpListener;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader as TokioBufReader};
    use std::sync::Arc;

    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    println!("Listening on 127.0.0.1:{port}");

    let classifier = Arc::new(classifier);

    loop {
        let (stream, _) = listener.accept().await?;
        let c = classifier.clone();
        tokio::spawn(async move {
            let (reader, mut writer) = stream.into_split();
            let mut lines = TokioBufReader::new(reader).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let input = line.trim().to_string();
                if input.is_empty() { continue; }
                let label = c.classify(&input).unwrap_or_else(|_| "unknown".to_string());
                let _ = writer.write_all(format!("{label}\n").as_bytes()).await;
            }
        });
    }
}

/// One-shot classify without daemon — useful for testing or fallback.
pub fn classify_once(task: &str, input: &str) -> Result<String> {
    let cfg = config::load(task)?;
    let gguf = config::gguf_path(task);
    let classifier = Classifier::load(&gguf, &cfg.task.classes)?;
    classifier.classify(input)
}
