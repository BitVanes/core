//! Full pipeline orchestration: parse -> scrub -> chunk -> `RecordBatch`.
//!
//! This is the entry point called by both the wasm wrapper and the CLI.
//! It ties the four stages together into a single call that produces an
//! Arrow [`RecordBatch`] ready for FFI export or IPC streaming.

use arrow::array::RecordBatch;

use crate::arrow_io::batch::{chunks_to_batch, chunks_to_batch_with_embeddings};
use crate::chunk::{chunk_document, chunk_document_semantic};
use crate::embed::Embedder;
use crate::error::Result;
use crate::parse::parse_bytes;
use crate::schema::{ChunkStrategy, PipelineConfig};
use crate::scrub::scrub_document;

/// Runs the full ETL pipeline on `bytes` and returns an Arrow
/// [`RecordBatch`] containing the chunked output.
///
/// Stages:
/// 1. Parse raw bytes into a [`Document`] via the configured format.
/// 2. Scrub PII via the configured [`ScrubProfile`].
/// 3. Chunk the scrubbed text via BPE token boundaries.
/// 4. Assemble chunks into an Arrow `RecordBatch`.
///
/// # Errors
///
/// Propagates [`crate::error::BitVanesError`] from any stage.
pub fn run_pipeline(bytes: &[u8], cfg: &PipelineConfig) -> Result<RecordBatch> {
    let doc = parse_bytes(bytes, cfg)?;
    let (scrubbed_doc, _offset_map) = scrub_document(doc, &cfg.scrub)?;
    let chunks = chunk_document(&scrubbed_doc, &cfg.chunk, cfg.source_label.as_deref())?;
    let batch = chunks_to_batch(&chunks)?;
    Ok(batch)
}

/// Like [`run_pipeline`] but generates embeddings for each chunk and fills
/// the `embedding` column with real `Float32` vectors.
///
/// The embedder is provided by the caller (typically an [`OrtEmbedder`]
/// loaded from a local model file, or a test stub).
///
/// # Errors
///
/// Propagates [`crate::error::BitVanesError`] from any pipeline stage or
/// the embedder.
///
/// [`OrtEmbedder`]: crate::embed::OrtEmbedder
pub fn run_pipeline_with_embeddings(
    bytes: &[u8],
    cfg: &PipelineConfig,
    embedder: &dyn Embedder,
) -> Result<RecordBatch> {
    let doc = parse_bytes(bytes, cfg)?;
    let (scrubbed_doc, _offset_map) = scrub_document(doc, &cfg.scrub)?;
    let chunks = chunk_document(&scrubbed_doc, &cfg.chunk, cfg.source_label.as_deref())?;

    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let embeddings = embedder.embed(&texts)?;
    let dim = embedder.dim();

    let batch = chunks_to_batch_with_embeddings(&chunks, &embeddings, dim)?;
    Ok(batch)
}

/// Like [`run_pipeline`] but honours [`ChunkStrategy::Semantic`]: when the
/// config asks for semantic chunking, `embedder` guides where cuts happen.
/// For [`ChunkStrategy::Structural`] the embedder is unused and this is
/// equivalent to [`run_pipeline`].
///
/// # Errors
///
/// Propagates [`crate::error::BitVanesError`] from any stage or the embedder.
pub fn run_pipeline_with_strategy(
    bytes: &[u8],
    cfg: &PipelineConfig,
    embedder: &dyn Embedder,
) -> Result<RecordBatch> {
    let doc = parse_bytes(bytes, cfg)?;
    let (scrubbed_doc, _offset_map) = scrub_document(doc, &cfg.scrub)?;
    let chunks = match cfg.chunk.strategy {
        ChunkStrategy::Structural => {
            chunk_document(&scrubbed_doc, &cfg.chunk, cfg.source_label.as_deref())?
        }
        ChunkStrategy::Semantic { .. } => chunk_document_semantic(
            &scrubbed_doc,
            &cfg.chunk,
            embedder,
            cfg.source_label.as_deref(),
        )?,
    };
    let batch = chunks_to_batch(&chunks)?;
    Ok(batch)
}

/// Runs [`run_pipeline`] over many inputs, in parallel when the `parallel`
/// feature is enabled and sequentially otherwise. Returns one result per
/// input so a single failing document does not abort the batch.
///
/// # Errors
///
/// Each element of the returned `Vec` propagates the per-input error
/// independently; an `Ok` element is a fully assembled `RecordBatch`.
pub fn run_pipeline_batch(inputs: &[&[u8]], cfg: &PipelineConfig) -> Vec<Result<RecordBatch>> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        inputs.par_iter().map(|b| run_pipeline(b, cfg)).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        inputs.iter().map(|b| run_pipeline(b, cfg)).collect()
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::schema::{ChunkConfig, DocumentFormat};

    #[test]
    fn batch_processes_multiple_inputs() {
        let cfg = PipelineConfig {
            format: DocumentFormat::Text,
            chunk: ChunkConfig {
                max_tokens: 512,
                ..ChunkConfig::default()
            },
            ..PipelineConfig::default()
        };
        let inputs: Vec<&[u8]> = vec![b"first document", b"second document", b"third"];
        let results = run_pipeline_batch(&inputs, &cfg);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok(), "batch element failed: {:?}", r.as_ref().err());
            assert!(r.as_ref().unwrap().num_rows() > 0);
        }
    }

    #[test]
    fn batch_isolates_per_input_failures() {
        let cfg = PipelineConfig {
            format: DocumentFormat::Pdf, // unresolvable without cli-pdf in tests
            ..PipelineConfig::default()
        };
        let inputs: Vec<&[u8]> = vec![b"not pdf", b"also not pdf"];
        let results = run_pipeline_batch(&inputs, &cfg);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(std::result::Result::is_err));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ChunkConfig, DocumentFormat, PipelineConfig, ScrubProfile};

    #[test]
    fn pipeline_produces_nonempty_batch_from_markdown() {
        let cfg = PipelineConfig {
            format: DocumentFormat::Markdown,
            scrub: ScrubProfile::default(),
            chunk: ChunkConfig {
                max_tokens: 512,
                ..ChunkConfig::default()
            },
            source_label: Some("test.md".to_string()),
            embeddings: None,
        };
        let input = b"# Title\n\nHello world. This is a test.";
        let batch = run_pipeline(input, &cfg).unwrap();
        assert!(batch.num_rows() > 0);
        assert_eq!(batch.num_columns(), 9);
    }

    #[test]
    fn pipeline_with_pii_scrubbing_redacts_email() {
        use arrow::array::{Array, StringArray};

        let cfg = PipelineConfig {
            format: DocumentFormat::Markdown,
            scrub: ScrubProfile {
                patterns: vec![crate::schema::BuiltInPattern::Email],
                custom: vec![],
            },
            chunk: ChunkConfig::default(),
            source_label: None,
            embeddings: None,
        };
        let input = b"Contact alice@example.com for info.";
        let batch = run_pipeline(input, &cfg).unwrap();
        assert_eq!(batch.num_rows(), 1);

        let text_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(
            text_col.value(0).contains("[EMAIL]"),
            "email should be redacted"
        );
        assert!(
            !text_col.value(0).contains("alice@example.com"),
            "raw email should not survive"
        );
    }

    #[test]
    fn pipeline_empty_input_produces_empty_batch() {
        let cfg = PipelineConfig::default();
        let batch = run_pipeline(b"", &cfg).unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn pipeline_preserves_heading_path() {
        use arrow::array::{Array, ListArray};

        let cfg = PipelineConfig {
            format: DocumentFormat::Markdown,
            ..PipelineConfig::default()
        };
        let input = b"# Architecture\n\nThe system has layers.\n\n## Storage\n\nWe use Arrow.";
        let batch = run_pipeline(input, &cfg).unwrap();
        assert!(batch.num_rows() >= 1);

        let heading_col = batch
            .column(4)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        // The chunk should have a non-null heading_path (under "Architecture"
        // or "Architecture > Storage").
        let has_heading = (0..heading_col.len()).any(|i| !heading_col.is_null(i));
        assert!(has_heading, "at least one chunk should have heading_path");
    }
}
