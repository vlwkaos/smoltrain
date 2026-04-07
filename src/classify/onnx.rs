use crate::config::TaskConfig;
use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

pub struct OnnxClassifier {
    session: Session,
    tokenizer: Tokenizer,
    classes: Vec<String>,
}

impl OnnxClassifier {
    pub fn load(cfg: &TaskConfig) -> Result<Self> {
        let name = &cfg.task.name;
        let onnx_path = crate::config::onnx_path(name);
        let tokenizer_path = crate::config::tokenizer_path(name);

        anyhow::ensure!(
            onnx_path.exists(),
            "ONNX model not found at {}. Run `smoltrain export {}` first.",
            onnx_path.display(),
            name
        );
        anyhow::ensure!(
            tokenizer_path.exists(),
            "tokenizer.json not found at {}",
            tokenizer_path.display()
        );

        let session = Session::builder()
            .context("failed to create ONNX session builder")?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("failed to load ONNX model from {}", onnx_path.display()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        Ok(Self {
            session,
            tokenizer,
            classes: cfg.task.classes.clone(),
        })
    }

    pub fn classify(&mut self, input: &str) -> Result<String> {
        let encoding = self.tokenizer
            .encode(input, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let seq_len = encoding.get_ids().len();

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

        let input_ids = Array2::from_shape_vec((1, seq_len), ids)
            .context("failed to build input_ids array")?;
        let attention_mask = Array2::from_shape_vec((1, seq_len), mask)
            .context("failed to build attention_mask array")?;
        let token_type_ids = Array2::from_shape_vec((1, seq_len), type_ids)
            .context("failed to build token_type_ids array")?;

        let t_input_ids = Tensor::from_array(input_ids)
            .context("failed to create input_ids tensor")?;
        let t_attention_mask = Tensor::from_array(attention_mask)
            .context("failed to create attention_mask tensor")?;
        let t_token_type_ids = Tensor::from_array(token_type_ids)
            .context("failed to create token_type_ids tensor")?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => t_input_ids,
            "attention_mask" => t_attention_mask,
            "token_type_ids" => t_token_type_ids,
        ])?;

        let (_shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .context("failed to extract logits tensor")?;

        let class_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.classes
            .get(class_idx)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("class index {class_idx} out of range"))
    }
}
