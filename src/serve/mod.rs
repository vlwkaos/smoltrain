use crate::classify::OnnxClassifier;
use crate::config::TaskConfig;
use anyhow::Result;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use tracing::{error, info};

pub fn run(cfg: &TaskConfig) -> Result<()> {
    let socket_path = &cfg.serve.socket;
    if std::path::Path::new(socket_path).exists() {
        std::fs::remove_file(socket_path)?;
    }

    let mut classifier = OnnxClassifier::load(cfg)?;
    info!("classifier loaded, listening on {}", socket_path);

    let listener = UnixListener::bind(socket_path)?;
    println!("serving on {socket_path}");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let mut reader = BufReader::new(stream.try_clone()?);
                let mut writer = stream;
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line) {
                        Ok(0) => break,
                        Ok(_) => {
                            let input = line.trim_end_matches('\n').trim_end_matches('\r');
                            match classifier.classify(input) {
                                Ok(label) => {
                                    let _ = writeln!(writer, "{label}");
                                }
                                Err(e) => {
                                    error!("classify error: {e}");
                                    let _ = writeln!(writer, "ERROR: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            error!("read error: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => error!("accept error: {e}"),
        }
    }

    Ok(())
}
