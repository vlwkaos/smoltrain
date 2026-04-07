use llama_cpp_2::llama_backend::LlamaBackend;
use std::sync::OnceLock;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

pub fn init() -> anyhow::Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let b = LlamaBackend::init()
        .map_err(|e| anyhow::anyhow!("llama backend init: {e}"))?;
    Ok(BACKEND.get_or_init(|| b))
}
