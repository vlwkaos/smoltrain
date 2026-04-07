/// Logit-based classifier using llama.cpp.
///
/// Same mechanism as ir's Reranker (reranker.rs) — but generalised to N classes
/// instead of Yes/No. We run one forward pass and read argmax over the label token IDs.
///
/// Latency: ~10-50ms with model pre-loaded (no cold-load penalty).
/// Cold-load: ~200-400ms (mitigated by daemon pattern in serve/mod.rs).

use anyhow::{Context, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
};
use std::num::NonZeroU32;
use std::path::Path;

const CONTEXT_SIZE: u32 = 512;

pub struct Classifier {
    model: LlamaModel,
    class_token_ids: Vec<(String, i32)>,
}

// SAFETY: LlamaModel is not Send by default in some llama-cpp-2 versions.
// The daemon uses a single-threaded unix socket loop, so this is safe.
// For the TCP path we wrap in Arc and only call from one thread per request.
unsafe impl Send for Classifier {}
unsafe impl Sync for Classifier {}

impl Classifier {
    /// Load model and resolve the first token ID for each class label.
    pub fn load(model_path: &Path, classes: &[String]) -> Result<Self> {
        let backend = crate::llm_backend::init()?;
        let model = LlamaModel::load_from_file(
            backend,
            model_path,
            &LlamaModelParams::default().with_n_gpu_layers(gpu_layers()),
        )
        .with_context(|| format!("load classifier model: {}", model_path.display()))?;

        // Resolve first token ID for each class label.
        // e.g. "code" → token id for "code" (or first subword token if BPE splits it)
        let class_token_ids = classes
            .iter()
            .map(|class| {
                let tokens = model
                    .str_to_token(class, AddBos::Never)
                    .with_context(|| format!("tokenize class '{class}'"))?;
                let id = tokens.first().map(|t| t.0).unwrap_or(0);
                Ok((class.clone(), id))
            })
            .collect::<Result<Vec<_>>>()?;

        println!("Class token IDs: {:?}", class_token_ids);

        Ok(Self { model, class_token_ids })
    }

    /// Classify input text. Returns the class label with highest logit.
    pub fn classify(&self, input: &str) -> Result<String> {
        let prompt = build_prompt(input, &self.class_token_ids.iter().map(|(c, _)| c.as_str()).collect::<Vec<_>>());

        let backend = crate::llm_backend::init()?;
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_SIZE))
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);

        let mut ctx = self.model
            .new_context(backend, ctx_params)
            .context("classifier context")?;

        let tokens = self.model
            .str_to_token(&prompt, AddBos::Always)
            .context("tokenize")?;

        if tokens.is_empty() {
            return Ok(self.class_token_ids[0].0.clone());
        }

        let n = tokens.len().min(CONTEXT_SIZE as usize - 1);
        let mut batch = LlamaBatch::new(n, 1);
        for (i, &tok) in tokens[..n].iter().enumerate() {
            let logits = i == n - 1;
            batch.add(tok, i as i32, &[0], logits)
                .context("batch add")?;
        }

        ctx.decode(&mut batch).context("decode")?;

        let logits = ctx.get_logits_ith((n - 1) as i32);

        // Argmax over class token IDs
        let (best_class, _) = self.class_token_ids
            .iter()
            .map(|(class, token_id)| {
                let logit = logits.get(*token_id as usize).copied().unwrap_or(f32::NEG_INFINITY);
                (class.as_str(), logit)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Ok(best_class.to_string())
    }
}

fn build_prompt(input: &str, classes: &[&str]) -> String {
    let class_list = classes.join("|");
    format!(
        "<|im_start|>system\nClassify the following request into one of: {class_list}\nRespond with only the class name.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn gpu_layers() -> u32 {
    std::env::var("SMOLTRAIN_GPU_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if cfg!(target_os = "macos") { 99 } else { 0 })
}
