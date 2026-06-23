//! Embedding generation: dense vector representations of text chunks.
//!
//! The [`Embedder`] trait defines the interface for generating embeddings.
//! The concrete [`OrtEmbedder`] implementation (ONNX Runtime) is behind the
//! `embeddings` cargo feature.
//!
//! # Pipeline integration
//!
//! Embeddings are generated AFTER chunking, as the final step before Arrow
//! assembly. Use [`run_pipeline_with_embeddings`][crate::pipeline::run_pipeline_with_embeddings]
//! to generate embeddings inline.
//!
//! # `OrtEmbedder` (feature: `embeddings`)
//!
//! Loads an ONNX-format sentence-transformer model (e.g. MiniLM-L6-v2) and
//! runs inference via the `ort` crate:
//!
//! 1. Tokenize text using the model's `tokenizer.json` (WordPiece/BPE).
//! 2. Run ONNX inference to produce token-level embeddings.
//! 3. Mean-pool across the sequence dimension (attention-mask weighted).
//! 4. L2-normalize the pooled vector.
//!
//! Native-only — the wasm target delegates embeddings to JavaScript
//! (`@xenova/transformers` or `onnxruntime-web`).

use crate::error::Result;

/// Generates dense vector embeddings for text chunks.
///
/// Implementations must be `Send + Sync` for use in multi-threaded contexts.
pub trait Embedder: Send + Sync {
    /// Returns the dimensionality of the embedding vectors.
    fn dim(&self) -> usize;

    /// Generates embeddings for a batch of texts. Returns one `Vec<f32>`
    /// per input text, each of length [`dim`](Self::dim).
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::BitVanesError`] if inference fails.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

// ---------------------------------------------------------------------------
// Math helpers (available without the `embeddings` feature for testing)
// ---------------------------------------------------------------------------

/// Mean-pools a sequence of token embeddings, weighted by the attention mask.
///
/// `hidden_state` is a flattened `(seq_len, hidden_dim)` slice.
/// `attention_mask` has length `seq_len`.
#[cfg(any(feature = "embeddings", test))]
pub(crate) fn mean_pool(
    hidden_state: &[f32],
    attention_mask: &[i64],
    hidden_dim: usize,
) -> Vec<f32> {
    let seq_len = attention_mask.len();
    debug_assert_eq!(
        hidden_state.len(),
        seq_len * hidden_dim,
        "hidden_state must be seq_len * hidden_dim"
    );

    let mut pooled = vec![0.0f32; hidden_dim];
    let mut total_weight = 0.0f32;

    for (i, &mask_val) in attention_mask.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let weight = mask_val as f32;
        total_weight += weight;
        let offset = i * hidden_dim;
        for j in 0..hidden_dim {
            pooled[j] += hidden_state[offset + j] * weight;
        }
    }

    if total_weight > 0.0 {
        for v in &mut pooled {
            *v /= total_weight;
        }
    }
    pooled
}

/// L2-normalizes a vector in place. If the norm is zero, the vector is
/// left unchanged (all-zeros is a valid zero vector).
#[cfg(any(feature = "embeddings", test))]
pub(crate) fn l2_normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// OrtEmbedder (behind the `embeddings` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "embeddings")]
mod ort_embedder {
    use std::path::Path;

    use ndarray::Array2;
    use ort::session::builder::GraphOptimizationLevel;
    use ort::{inputs, session::Session, value::Tensor};
    use tokenizers::Tokenizer;

    use crate::embed::{Embedder, l2_normalize, mean_pool};
    use crate::error::{BitVanesError, Result};

    /// ONNX Runtime-backed embedder for sentence-transformer models.
    ///
    /// Loads a quantized or full-precision ONNX model and generates
    /// embeddings via tokenization -> inference -> mean pooling -> L2
    /// normalization.
    ///
    /// # Construction
    ///
    /// Requires two files:
    /// - `model_path`: the `.onnx` model file (e.g. `model_quantized.onnx`)
    /// - `tokenizer_path`: the `tokenizer.json` file from Hugging Face
    ///
    /// Both are typically downloaded from the Hugging Face Hub and cached
    /// locally. The CLI handles download + caching; the constructor just
    /// loads from local paths.
    pub struct OrtEmbedder {
        session: std::sync::Mutex<Session>,
        tokenizer: Tokenizer,
        dim: usize,
        max_seq_len: usize,
    }

    impl OrtEmbedder {
        /// Creates a new embedder from local model and tokenizer files.
        ///
        /// # Arguments
        ///
        /// - `model_path`: Path to the ONNX model file.
        /// - `tokenizer_path`: Path to the `tokenizer.json` file.
        /// - `dim`: The embedding dimension (e.g. 384 for MiniLM-L6-v2).
        /// - `max_seq_len`: Maximum token sequence length (e.g. 256).
        ///
        /// # Errors
        ///
        /// Returns [`BitVanesError::InvalidInput`] if the model or tokenizer
        /// cannot be loaded.
        pub fn new(
            model_path: &Path,
            tokenizer_path: &Path,
            dim: usize,
            max_seq_len: usize,
        ) -> Result<Self> {
            let session = Session::builder()
                .map_err(|e| BitVanesError::InvalidInput(format!("ort session builder: {e}")))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| BitVanesError::InvalidInput(format!("ort optimization: {e}")))?
                .with_intra_threads(1)
                .map_err(|e| BitVanesError::InvalidInput(format!("ort threads: {e}")))?
                .commit_from_file(model_path)
                .map_err(|e| {
                    BitVanesError::InvalidInput(format!(
                        "ort model load from {}: {e}",
                        model_path.display()
                    ))
                })?;

            let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
                BitVanesError::InvalidInput(format!(
                    "tokenizer load from {}: {e}",
                    tokenizer_path.display()
                ))
            })?;

            Ok(Self {
                session: std::sync::Mutex::new(session),
                tokenizer,
                dim,
                max_seq_len,
            })
        }
    }

    impl Embedder for OrtEmbedder {
        fn dim(&self) -> usize {
            self.dim
        }

        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut session = self
                .session
                .lock()
                .map_err(|e| BitVanesError::InvalidInput(format!("session lock: {e}")))?;

            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                let embedding = embed_one(
                    &mut session,
                    &self.tokenizer,
                    text,
                    self.dim,
                    self.max_seq_len,
                )?;
                results.push(embedding);
            }
            Ok(results)
        }
    }

    fn embed_one(
        session: &mut Session,
        tokenizer: &Tokenizer,
        text: &str,
        dim: usize,
        max_seq_len: usize,
    ) -> Result<Vec<f32>> {
        // 1. Tokenize
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| BitVanesError::InvalidInput(format!("tokenization: {e}")))?;

        let ids = encoding.get_ids();
        let attention = encoding.get_attention_mask();

        // 2. Pad or truncate to max_seq_len
        let (input_ids, attn_mask) = pad_or_truncate(ids, attention, max_seq_len);

        // 3. Build input tensors (shape: [1, seq_len])
        let input_ids_arr = Array2::from_shape_vec((1, max_seq_len), input_ids)
            .map_err(|e| BitVanesError::InvalidInput(format!("ndarray shape: {e}")))?;
        let attn_arr = Array2::from_shape_vec((1, max_seq_len), attn_mask.clone())
            .map_err(|e| BitVanesError::InvalidInput(format!("ndarray shape: {e}")))?;

        // 4. Run inference
        let ids_tensor = Tensor::from_array(input_ids_arr)
            .map_err(|e| BitVanesError::InvalidInput(format!("input_ids tensor: {e}")))?;
        let attn_tensor = Tensor::from_array(attn_arr)
            .map_err(|e| BitVanesError::InvalidInput(format!("attention_mask tensor: {e}")))?;
        let outputs = session
            .run(inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => attn_tensor,
            ])
            .map_err(|e| BitVanesError::InvalidInput(format!("ort inference: {e}")))?;

        // 5. Extract last_hidden_state (shape: [1, seq_len, hidden_dim])
        let hidden = outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .map_err(|e| BitVanesError::InvalidInput(format!("tensor extract: {e}")))?;

        let view = hidden.view();
        let hidden_dim = view.shape()[2];
        let flat: Vec<f32> = view.iter().copied().collect();

        // 6. Mean pool (attention-mask weighted)
        let pooled = mean_pool(&flat, &attn_mask, hidden_dim.min(dim));

        // 7. L2 normalize
        let mut result = pooled;
        l2_normalize(&mut result);

        Ok(result)
    }

    /// Pads or truncates token IDs and attention mask to `max_len`.
    fn pad_or_truncate(ids: &[u32], attention: &[u32], max_len: usize) -> (Vec<i64>, Vec<i64>) {
        let mut padded_ids = vec![0i64; max_len];
        let mut padded_attn = vec![0i64; max_len];
        let len = ids.len().min(max_len);
        for i in 0..len {
            padded_ids[i] = i64::from(ids[i]);
            padded_attn[i] = i64::from(attention[i]);
        }
        (padded_ids, padded_attn)
    }
}

#[cfg(feature = "embeddings")]
pub use ort_embedder::OrtEmbedder;

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial embedder for testing the pipeline without a real model.
    struct ZeroEmbedder {
        dim: usize,
    }

    impl Embedder for ZeroEmbedder {
        fn dim(&self) -> usize {
            self.dim
        }
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; self.dim]).collect())
        }
    }

    #[test]
    fn zero_embedder_produces_correct_dimensions() {
        let e = ZeroEmbedder { dim: 384 };
        let result = e.embed(&["hello", "world"]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 384);
    }

    #[test]
    fn mean_pool_uniform_attention_averages_tokens() {
        // 3 tokens, 2 dims, all attention=1
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2],[3,4],[5,6]]
        let mask = vec![1i64, 1, 1];
        let pooled = mean_pool(&hidden, &mask, 2);
        assert!(((pooled[0] - 3.0).abs() < 1e-6), "dim0: {}", pooled[0]); // (1+3+5)/3=3
        assert!(((pooled[1] - 4.0).abs() < 1e-6), "dim1: {}", pooled[1]); // (2+4+6)/3=4
    }

    #[test]
    fn mean_pool_respects_attention_mask() {
        // 3 tokens, 2 dims, only first two attended
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![1i64, 1, 0];
        let pooled = mean_pool(&hidden, &mask, 2);
        assert!(((pooled[0] - 2.0).abs() < 1e-6), "dim0: {}", pooled[0]); // (1+3)/2=2
        assert!(((pooled[1] - 3.0).abs() < 1e-6), "dim1: {}", pooled[1]); // (2+4)/2=3
    }

    #[test]
    fn l2_normalize_produces_unit_vector() {
        let mut v = vec![3.0, 4.0]; // norm = 5
        l2_normalize(&mut v);
        assert!(((v[0] - 0.6).abs() < 1e-6), "v0: {}", v[0]); // 3/5
        assert!(((v[1] - 0.8).abs() < 1e-6), "v1: {}", v[1]); // 4/5
    }

    #[test]
    fn l2_normalize_handles_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }
}
